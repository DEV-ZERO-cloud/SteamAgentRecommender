"""
semantic_engine.py – Motor de búsqueda semántica con weighted embeddings.

Responsabilidad única (Paso 2 del pipeline):
    Recibe una lista de tags del usuario y devuelve una lista de tuplas
    (GameID, score) ordenadas por relevancia semántica.

Contrato de salida (idéntico a la versión TF-IDF):
    List[tuple[int, float]]  →  [(app_id, similarity_score), ...]

Estrategia interna:
    La query del usuario se embedea con el mismo esquema de ponderación
    por frecuencia que los vectores de los juegos (construidos en PrepareData).
    La similitud se calcula como coseno — equivalente a producto punto dado
    que todos los vectores están normalizados a norma unitaria.

No carga datos ni construye índices: recibe un PreparedData ya construido
mediante inyección de dependencia (igual que la versión anterior).
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración compartida (importada también por PrepareData)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticEngineConfig:
    """Parámetros de construcción y búsqueda del motor semántico."""
    csv_path: str | Path
    separator: str = "|"
    id_column: str = "appid"
    tag_column: str = "tags"
    genre_column: str = "genres"
    default_top_k: int = 20
    default_min_score: float = 0.05
    # Modelo de sentence-transformers.
    # "all-MiniLM-L6-v2"  → ~90 MB, 384 dims, rápido, buena calidad.
    # "all-mpnet-base-v2" → ~420 MB, 768 dims, mayor calidad, más lento.
    embedding_model: str = "all-MiniLM-L6-v2"
    # Ruta para caché de embeddings en disco (None = sin caché).
    # Si el CSV cambia, borrar el .npy para forzar reconstrucción.
    embeddings_cache_path: str | Path | None = None


# ---------------------------------------------------------------------------
# Motor semántico
# ---------------------------------------------------------------------------

class SemanticEngine:
    """
    Motor de búsqueda semántica con weighted embeddings (Forma A).

    Recibe un PreparedData ya construido (inyección de dependencia).
    El contrato de salida es idéntico a la versión TF-IDF anterior,
    por lo que el pipeline no requiere ningún cambio.

    Uso típico:
        config   = SemanticEngineConfig(csv_path="src/data/steam_rpg.csv")
        prepared = PrepareData(config).prepare()
        engine   = SemanticEngine(config, prepared)

        candidates = engine.search(["open-world", "fantasy"], top_k=20)
        # → [(730, 0.91), (570, 0.87), ...]
    """

    def __init__(self, config: SemanticEngineConfig, prepared) -> None:
        """
        Args:
            config:   Configuración con columnas y defaults de búsqueda.
            prepared: PreparedData (df, embeddings, model) de PrepareData.
        """
        self.config  = config
        self._df     = prepared.df
        self._model  = prepared.model        # SentenceTransformer
        self._matrix = prepared.embeddings  # (n_games, dim), norma unitaria

        logger.info(
            "SemanticEngine listo: %d juegos indexados (dim=%d).",
            len(self._df),
            self._matrix.shape[1],
        )

    # ------------------------------------------------------------------
    # API pública — misma firma que la versión TF-IDF
    # ------------------------------------------------------------------

    def search(
        self,
        tags: List[str],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> List[tuple[int, float]]:
        """
        Busca los juegos más similares semánticamente a la lista de tags.

        Args:
            tags:      Lista de tags de preferencia del usuario.
            top_k:     Máximo de candidatos a retornar.
                       Por defecto usa config.default_top_k.
            min_score: Umbral mínimo de similitud coseno [0.0 - 1.0].
                       Por defecto usa config.default_min_score.

        Returns:
            List[tuple[int, float]] — (GameID, score) ordenado mayor a menor,
            filtrado por top_k y min_score.
        """
        if not tags:
            logger.warning("search() llamado con lista de tags vacía.")
            return []

        _top_k     = top_k     if top_k     is not None else self.config.default_top_k
        _min_score = min_score if min_score is not None else self.config.default_min_score

        query_vec   = self._embed_query(tags)                        # (1, dim)
        similarities = cosine_similarity(query_vec, self._matrix).flatten()

        ranked_indices = similarities.argsort()[::-1][:_top_k]

        results: List[tuple[int, float]] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < _min_score:
                continue
            game_id = int(self._df.iloc[idx][self.config.id_column])
            results.append((game_id, round(score, 6)))

        logger.debug(
            "search(tags=%s, top_k=%d, min_score=%.3f) → %d candidatos.",
            tags, _top_k, _min_score, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_query(self, tags: List[str]) -> np.ndarray:
        """
        Embedea la lista de tags del usuario como promedio ponderado
        por frecuencia, con el mismo esquema que los vectores de juegos.

        Ejemplo: tags=["rpg", "rpg", "open-world"]
            counts  = {"rpg": 2, "open-world": 1}  → total = 3
            weights = {"rpg": 0.667, "open-world": 0.333}
            vector  = 0.667*emb("rpg") + 0.333*emb("open-world")  → normalizado
        """
        normalized = [t.strip().lower() for t in tags if t.strip()]
        if not normalized:
            return np.zeros((1, self._matrix.shape[1]), dtype=np.float32)

        counts = Counter(normalized)
        total  = sum(counts.values())

        unique_tags = list(counts.keys())
        tag_embs    = self._model.encode(
            unique_tags,
            normalize_embeddings=True,
            show_progress_bar=False,
        )                                                   # (n_unique, dim)

        weights   = np.array([counts[t] / total for t in unique_tags], dtype=np.float32)
        query_vec = (tag_embs * weights[:, None]).sum(axis=0, keepdims=True)

        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        return query_vec                                    # (1, dim)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_semantic_engine(
    csv_path: str | Path | None = None,
    **kwargs,
) -> SemanticEngine:
    """
    Factory que construye PrepareData + SemanticEngine.

    Args:
        csv_path: Ruta al CSV. Si es None, usa la env var CSV_PATH.
        **kwargs: Parámetros opcionales para SemanticEngineConfig.

    Returns:
        SemanticEngine listo para llamar a .search().
    """
    from scripts.prepare_data import PrepareData  # evita circular import

    resolved_path = csv_path or os.getenv("CSV_PATH")
    if not resolved_path:
        raise EnvironmentError(
            "Se requiere csv_path o la variable de entorno CSV_PATH."
        )

    config   = SemanticEngineConfig(csv_path=resolved_path, **kwargs)
    prepared = PrepareData(config).prepare()
    return SemanticEngine(config, prepared)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    engine = build_semantic_engine(
        embedding_model="all-MiniLM-L6-v2",
        embeddings_cache_path="src/data/embeddings_cache.npy",
    )

    candidates = engine.search(
        tags=["open-world", "story-rich", "fantasy"],
        top_k=10,
        min_score=0.05,
    )

    print("\nCandidatos semánticos:\n")
    for game_id, score in candidates:
        print(f"  AppID={game_id}  score={score:.4f}")