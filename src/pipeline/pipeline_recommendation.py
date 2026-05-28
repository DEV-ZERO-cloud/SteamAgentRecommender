"""
pipeline_recommendation.py – Pipeline de recomendación unificado.

CAMBIOS respecto a versión anterior:
    - SemanticEngine reemplazado por KnowledgeEngine.semantic_search()
      (FAISS sobre texto completo: name + description + genres + tags).
      Elimina la redundancia de tener dos índices separados.
    - top_k del semantic search subido a top_k * 20 para mejorar recall.
    - query se convierte a string de texto para FAISS (en lugar de lista de tags
      pasada a TF-IDF), aprovechando la búsqueda semántica real del modelo.

Flujo:
    Paso 1: query de la request
    Paso 2: KnowledgeEngine.semantic_search() → {app_id: similarity}  [FAISS]
    Paso 3: KnowledgeEngine.get_games()       → [Game, ...]
    Paso 4: ParametersEngine.score()          → [GameScore, ...]
    Paso 5: PrologEngine.filter()             → [EnrichedScore, ...]
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from engine.knowledge_engine import KnowledgeEngine
from engine.parameters_engine import ParametersEngine, ScoreFilters
from engine.prolog_engine import PrologEngine, EnrichedScore

load_dotenv()


class PipelineRecommendation:
    def __init__(self):
        csv_path        = os.getenv("CSV_PATH")
        parameters_path = os.getenv("PARAMETERS_PATH")
        index_dir       = os.getenv("EMBEDDINGS_DIR", "src/embeddings")

        # Paso 2+3: base de conocimiento con FAISS incorporado
        self.knowledge_engine = KnowledgeEngine(
            csv_path=csv_path,
            index_dir=index_dir,
        )
        self.knowledge_engine.build()

        # Paso 4: scoring por parameters
        self.parameters_engine = ParametersEngine(parameters_path=parameters_path)

        # Paso 5: motor lógico simbólico
        self.prolog_engine = PrologEngine()

    def recommend(
        self,
        query: list[str] | str,
        top_k: int = 5,
        filters: ScoreFilters | None = None,
        disliked_tags: list[str] | str | None = None,
        max_price: float = 0.0,
    ) -> list[EnrichedScore]:
        """
        Ejecuta el pipeline completo y retorna recomendaciones enriquecidas.

        Args:
            query:         Lista de tags o string de búsqueda libre.
            top_k:         Número máximo de resultados finales.
            filters:       Filtros opcionales de precio, fecha, rating, etc.
            disliked_tags: Tags que el usuario rechaza.
            max_price:     Precio máximo. 0 = sin límite.

        Returns:
            Lista de EnrichedScore ordenada por relevancia.
        """
        # Paso 1 — normalizar query
        if isinstance(query, list):
            query_tags  = [t.strip() for t in query if t.strip()]
            query_text  = ", ".join(query_tags)   # texto para FAISS
        else:
            query_text  = query.strip()
            query_tags  = [t.strip() for t in query.split(",") if t.strip()]

        # Paso 2 — búsqueda semántica con FAISS
        # top_k * 20 da margen suficiente para que Prolog filtre con calidad
        # sin agotar el catálogo relevante (era top_k * 5, recall demasiado bajo)
        faiss_k = min(top_k * 20, len(self.knowledge_engine.games))
        semantic_scores = self.knowledge_engine.semantic_search(
            query=query_text,
            k=faiss_k,
        )

        # Paso 3 — IDs → objetos Game
        games = self.knowledge_engine.get_games(list(semantic_scores.keys()))

        # Paso 4 — scoring por parameters
        scored = self.parameters_engine.score(games, semantic_scores, filters)

        # Paso 5 — motor lógico
        games_by_id = {g.app_id: g for g in games}

        # Configurar preferencias del motor lógico para esta request
        if disliked_tags:
            if isinstance(disliked_tags, list):
                self.prolog_engine.disliked_tags = {t.strip() for t in disliked_tags}
            else:
                self.prolog_engine.disliked_tags = {
                    t.strip() for t in disliked_tags.split(",")
                }
        else:
            self.prolog_engine.disliked_tags = set()

        self.prolog_engine.max_price = max_price

        enriched = self.prolog_engine.filter(scored, games_by_id, query_tags)

        return enriched[:top_k]