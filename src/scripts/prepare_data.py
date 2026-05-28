"""
prepare_data.py – Carga y preprocesamiento del dataset para el motor semántico.

Responsabilidad única:
    Leer el CSV, validar columnas, normalizar texto y construir
    la matriz de embeddings ponderados por frecuencia de tag.

Salida (PreparedData):
    - df         → DataFrame indexado listo para lookup por posición
    - embeddings → np.ndarray (n_games × embedding_dim), norma unitaria
    - model      → SentenceTransformer ya cargado (reutilizado en search)

Estrategia de vectorización (Forma A — weighted embeddings):
    Para cada juego, el vector final es el promedio ponderado de los
    embeddings de sus tags individuales, usando frecuencia relativa
    como peso:

        game_vector = Σ (count(tag) / total_tags) * embed(tag)

    Esto da más peso a los tags que dominan el perfil del juego y
    preserva la semántica real entre conceptos similares (RPG ≈ JRPG).

    Los vectores se normalizan a norma unitaria → coseno = producto punto.

Este módulo no conoce ni el pipeline ni la lógica de búsqueda.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from engine.semantic_engine import SemanticEngineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contenedor de salida (value object inmutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedData:
    """
    Resultado del proceso de preparación, listo para inyectar en SemanticEngine.

    Atributos:
        df          DataFrame normalizado con columnas originales.
        embeddings  Matriz de vectores (n_games × embedding_dim), norma unitaria.
        model       SentenceTransformer ya cargado.
    """
    df: pd.DataFrame
    embeddings: np.ndarray
    model: object          # SentenceTransformer — tipado como object para no
                           # forzar la importación en quien solo usa PreparedData


# ---------------------------------------------------------------------------
# Preparador
# ---------------------------------------------------------------------------

class PrepareData:
    """
    Orquesta la carga del CSV y la construcción del índice de embeddings.

    Uso:
        preparer = PrepareData(config)
        prepared = preparer.prepare()
        # → PreparedData(df, embeddings, model)

    Caché:
        Si config.embeddings_cache_path está definido y el archivo existe,
        se cargan los embeddings desde disco y se omite el paso costoso.
        Al generar embeddings nuevos, se guardan automáticamente en ese path.
        Si el CSV cambia, borrar el .npy manualmente para forzar reconstrucción.
    """

    def __init__(self, config: SemanticEngineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def prepare(self) -> PreparedData:
        """Ejecuta el pipeline de preparación y retorna un PreparedData."""
        from sentence_transformers import SentenceTransformer

        # ── Pasos heredados: carga, validación, normalización ─────────────
        df = self._load(self.config.csv_path)
        df = self._validate(df)
        df = self._normalize(df)

        # ── Cargar modelo ─────────────────────────────────────────────────
        logger.info(
            "PrepareData: cargando modelo '%s'...", self.config.embedding_model
        )
        model = SentenceTransformer(self.config.embedding_model)

        # ── Caché de embeddings ───────────────────────────────────────────
        cache_path = (
            Path(self.config.embeddings_cache_path)
            if self.config.embeddings_cache_path
            else None
        )
        if cache_path and cache_path.exists():
            logger.info(
                "PrepareData: cargando embeddings desde caché '%s'.", cache_path
            )
            embeddings = np.load(cache_path)
            logger.info(
                "PrepareData completado (caché): %d juegos, dim=%d.",
                len(df), embeddings.shape[1],
            )
            return PreparedData(df=df, embeddings=embeddings, model=model)

        # ── Construir embeddings ponderados ───────────────────────────────
        embeddings = self._build_weighted_embeddings(df, model)

        # ── Guardar caché ─────────────────────────────────────────────────
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(
                "PrepareData: embeddings guardados en '%s'.", cache_path
            )

        logger.info(
            "PrepareData completado: %d juegos, dim=%d.",
            len(df), embeddings.shape[1],
        )
        return PreparedData(df=df, embeddings=embeddings, model=model)

    # ------------------------------------------------------------------
    # Pasos heredados (sin cambios respecto a la versión TF-IDF)
    # ------------------------------------------------------------------

    def _load(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, sep=self.config.separator, header=0)
        df.columns = df.columns.str.strip()
        logger.debug("CSV cargado: %d filas desde '%s'.", len(df), csv_path)
        return df.reset_index(drop=True)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {
            self.config.id_column,
            self.config.tag_column,
            self.config.genre_column,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en el CSV: {missing}")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.config.tag_column] = (
            df[self.config.tag_column].fillna("").str.lower().str.strip()
        )
        df[self.config.genre_column] = (
            df[self.config.genre_column].fillna("").str.lower().str.strip()
        )
        return df

    # ------------------------------------------------------------------
    # Construcción de embeddings ponderados (reemplaza _build_text_field
    # y _build_index de la versión TF-IDF)
    # ------------------------------------------------------------------

    def _build_weighted_embeddings(
        self,
        df: pd.DataFrame,
        model,
    ) -> np.ndarray:
        """
        Construye un vector por juego como promedio ponderado de embeddings
        de tags individuales, con la frecuencia relativa como peso.

        Pasos:
            1. Parsear los tags de cada juego desde la columna tag_column.
               El CSV usa coma como separador interno de tags dentro del campo.
            2. Recopilar todos los tags únicos del corpus.
            3. Embedear todos los tags únicos en un único batch (eficiente).
            4. Para cada juego: weighted_sum = Σ (count/total) * embed(tag).
            5. Normalizar cada vector a norma unitaria.

        Los géneros (genre_column) se incluyen como tags adicionales con
        peso 1 para mantener la señal que aportaba la versión TF-IDF.
        """
        cfg = self.config

        # ── 1. Parsear tags + géneros por juego ───────────────────────────
        tag_lists: list[list[str]] = []
        all_tags: set[str] = set()

        for _, row in df.iterrows():
            tags   = self._split_field(row[cfg.tag_column])
            genres = self._split_field(row[cfg.genre_column])
            # Géneros se añaden una vez (peso natural = 1 ocurrencia)
            combined = tags + genres
            tag_lists.append(combined)
            all_tags.update(combined)

        # ── 2. Embeddear todos los tags únicos en un batch ────────────────
        unique_tags = sorted(all_tags)
        logger.info(
            "PrepareData: embedeando %d tags únicos con '%s'...",
            len(unique_tags), cfg.embedding_model,
        )
        tag_embeddings: np.ndarray = model.encode(
            unique_tags,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,    # norma unitaria por tag individual
        )                                 # shape: (n_unique_tags, dim)

        tag_index: dict[str, int] = {tag: i for i, tag in enumerate(unique_tags)}
        dim = tag_embeddings.shape[1]

        # ── 3. Construir vector ponderado por juego ───────────────────────
        game_vectors = np.zeros((len(df), dim), dtype=np.float32)

        for row_idx, tags in enumerate(tag_lists):
            if not tags:
                # Juego sin tags: vector nulo → score 0 en cualquier búsqueda
                continue

            counts = Counter(tags)
            total  = sum(counts.values())

            weighted_sum = np.zeros(dim, dtype=np.float32)
            for tag, count in counts.items():
                if tag not in tag_index:
                    continue                             # tag fuera del vocabulario
                weight = count / total                  # frecuencia relativa [0, 1]
                weighted_sum += weight * tag_embeddings[tag_index[tag]]

            # Normalizar a norma unitaria → coseno = producto punto en search()
            norm = np.linalg.norm(weighted_sum)
            if norm > 0:
                game_vectors[row_idx] = weighted_sum / norm

        return game_vectors                             # (n_games, dim)

    @staticmethod
    def _split_field(raw: str) -> list[str]:
        """
        Parsea un campo de tags/géneros del CSV.
        El separador interno del campo es coma (distinto del separador de columnas).
        """
        if not raw or not isinstance(raw, str):
            return []
        return [t.strip() for t in raw.split(",") if t.strip()]