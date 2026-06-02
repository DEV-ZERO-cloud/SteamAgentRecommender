"""
prepare_data.py – Carga y preprocesamiento del dataset para el motor semántico.

Responsabilidad única:
    Leer el CSV, validar columnas, normalizar texto y construir
    el vectorizador TF-IDF + matriz de términos.

Salida (PreparedData):
    - df         → DataFrame indexado listo para lookup por posición
    - vectorizer → TfidfVectorizer ya ajustado
    - matrix     → Matriz dispersa (n_games × n_terms)

Este módulo no conoce ni el pipeline ni la lógica de búsqueda.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

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
        df          DataFrame normalizado con columna '_text' y columnas originales.
        vectorizer  TfidfVectorizer ya ajustado sobre el corpus.
        matrix      Matriz TF-IDF dispersa (n_games × n_terms).
    """
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: spmatrix


# ---------------------------------------------------------------------------
# Preparador
# ---------------------------------------------------------------------------

class PrepareData:
    """
    Orquesta la carga del CSV y la construcción del índice TF-IDF.

    Uso:
        preparer = PrepareData(config)
        prepared = preparer.prepare()
        # → PreparedData(df, vectorizer, matrix)
    """

    def __init__(self, config: SemanticEngineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def prepare(self) -> PreparedData:
        """Ejecuta el pipeline de preparación y retorna un PreparedData."""
        df = self._load(self.config.csv_path)
        df = self._validate(df)
        df = self._normalize(df)
        df = self._build_text_field(df)
        vectorizer, matrix = self._build_index(df)

        logger.info(
            "PrepareData completado: %d juegos, %d términos TF-IDF.",
            len(df),
            matrix.shape[1],
        )
        return PreparedData(df=df, vectorizer=vectorizer, matrix=matrix)

    # ------------------------------------------------------------------
    # Pasos internos
    # ------------------------------------------------------------------

    def _load(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, sep=self.config.separator, header=0)
        print(type(df.columns))
        print(repr(df.columns.tolist()[:3]))
        df.columns = df.columns.str.strip()
        logger.debug("CSV cargado: %d filas desde '%s'.", len(df), csv_path)
        return df.reset_index(drop=True)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {self.config.id_column, self.config.tag_column, self.config.genre_column}
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

    def _build_text_field(self, df: pd.DataFrame) -> pd.DataFrame:
        # Tags se repiten para darles mayor peso que los géneros
        df["_text"] = (
            df[self.config.tag_column] + " "
            + df[self.config.tag_column] + " "
            + df[self.config.genre_column]
        )
        return df

    def _build_index(self, df: pd.DataFrame) -> tuple[TfidfVectorizer, spmatrix]:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # unigramas + bigramas ("story-rich", "open world")
            min_df=1,
            sublinear_tf=True,   # escala log en frecuencia de términos
        )
        matrix = vectorizer.fit_transform(df["_text"])
        return vectorizer, matrix