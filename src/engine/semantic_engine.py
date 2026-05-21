"""
semantic_engine.py – Motor de búsqueda semántica para el pipeline de recomendación.

Responsabilidad única (Paso 2 del pipeline):
    Recibe una lista de tags del usuario y devuelve una lista de tuplas
    (GameID, score) ordenadas por relevancia semántica.

Contrato de salida:
    List[tuple[int, float]]  →  [(app_id, similarity_score), ...]

No carga datos ni construye índices: recibe un PreparedData ya construido
mediante inyección de dependencia.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

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

# ---------------------------------------------------------------------------
# Motor semántico
# ---------------------------------------------------------------------------

class SemanticEngine:
    """
    Motor TF-IDF sobre tags + géneros de juegos Steam.

    Recibe un PreparedData ya construido (inyección de dependencia).
    No sabe cómo se cargó ni cómo se indexó el corpus.

    Uso típico desde el pipeline:
        config   = SemanticEngineConfig(csv_path="src\data\steam_rpg.csv")
        prepared = PrepareData(config).prepare()
        engine   = SemanticEngine(config, prepared)

        candidates = engine.search(["open-world", "fantasy"], top_k=20)
        # → [(730, 0.91), (570, 0.87), ...]
    """

    def __init__(self, config: SemanticEngineConfig, prepared) -> None:
        """
        Args:
            config:   Configuración con nombres de columnas y defaults de búsqueda.
            prepared: PreparedData (df, vectorizer, matrix) inyectado desde afuera.
        """
        self.config = config
        self._df = prepared.df
        self._vectorizer = prepared.vectorizer
        self._matrix = prepared.matrix

        logger.info(
            "SemanticEngine listo: %d juegos indexados.",
            len(self._df),
        )

    # ------------------------------------------------------------------
    # API pública
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
            tags:       Lista de tags de preferencia del usuario.
            top_k:      Máximo de candidatos a retornar.
                        Por defecto usa config.default_top_k.
            min_score:  Umbral mínimo de similitud coseno [0.0 - 1.0].
                        Por defecto usa config.default_min_score.

        Returns:
            List[tuple[int, float]] — (GameID, score) ordenado de mayor a menor,
            filtrado por top_k y min_score.
        """
        if not tags:
            logger.warning("search() llamado con lista de tags vacía.")
            return []

        _top_k = top_k if top_k is not None else self.config.default_top_k
        _min_score = min_score if min_score is not None else self.config.default_min_score

        query_vec = self._vectorizer.transform([self._build_query(tags)])
        similarities = cosine_similarity(query_vec, self._matrix).flatten()

        ranked_indices = similarities.argsort()[::-1][:_top_k]

        results: List[tuple[int, float]] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < _min_score:
                continue  # salta este, pero sigue evaluando los demás
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

    def _build_query(self, tags: List[str]) -> str:
        """Normaliza y une los tags en un string de consulta."""
        return " ".join(tag.strip().lower() for tag in tags if tag.strip())


# ---------------------------------------------------------------------------
# Factory — orquesta PrepareData + SemanticEngine
# ---------------------------------------------------------------------------

def build_semantic_engine(
    csv_path: str | Path | None = None,
    **kwargs,
) -> "SemanticEngine":
    """
    Factory que construye el pipeline completo de preparación + motor.

    Instancia PrepareData, ejecuta prepare() e inyecta el resultado
    en SemanticEngine. Es el único lugar donde ambas clases se conocen.

    Args:
        csv_path: Ruta al CSV. Si es None, usa la env var CSV_PATH.
        **kwargs: Parámetros opcionales para SemanticEngineConfig.

    Returns:
        SemanticEngine listo para llamar a .search().
    """
    from scripts.prepare_data import PrepareData  # importación local: evita circular import

    resolved_path = csv_path or os.getenv("CSV_PATH")
    if not resolved_path:
        raise EnvironmentError(
            "Se requiere csv_path o la variable de entorno CSV_PATH."
        )

    config = SemanticEngineConfig(csv_path=resolved_path, **kwargs)
    prepared = PrepareData(config).prepare()
    return SemanticEngine(config, prepared)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    engine = build_semantic_engine("steam_rpg.csv")

    candidates = engine.search(
        tags=["open-world", "story-rich", "fantasy"],
        top_k=10,
        min_score=0.05,
    )

    print("\nCandidatos semánticos:\n")
    for game_id, score in candidates:
        print(f"  AppID={game_id}  score={score:.4f}")