"""
pipeline.py – Pipeline de recomendación con Paso 5 activo.

    Paso 1: Obtención de la query de la request
    Paso 2: semantic_engine   → búsqueda semántica por tags → {app_id: similarity}
    Paso 3: knowledge_engine  → normaliza IDs a objetos Game → [Game, ...]
    Paso 4: parameters_engine → scoring por filtros → [GameScore, ...]
    Paso 5: prolog_engine     → reglas lógicas + enriquecimiento → [EnrichedScore, ...]
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from engine.knowledge_engine import KnowledgeEngine
from engine.semantic_engine import SemanticEngine, SemanticEngineConfig
from engine.parameters_engine import ParametersEngine, ScoreFilters
from engine.prolog_engine import PrologEngine, EnrichedScore
from scripts.prepare_data import PrepareData

load_dotenv()


class PipelineRecommendation:
    def __init__(self):
        csv_path        = os.getenv("CSV_PATH")
        parameters_path = os.getenv("PARAMETERS_PATH")

        # Paso 2: motor semántico TF-IDF
        config = SemanticEngineConfig(
            csv_path=csv_path,
            separator="|",
            id_column="appid",
            tag_column="tags",
            genre_column="genres",
        )
        prepared = PrepareData(config).prepare()
        self.semantic_engine = SemanticEngine(config, prepared)

        # Paso 3: base de conocimiento CSV → objetos Game
        self.knowledge_engine = KnowledgeEngine(csv_path=csv_path)
        self.knowledge_engine.build()

        # Paso 4: motor de scoring por parameters
        self.parameters_engine = ParametersEngine(parameters_path=parameters_path)

        # Paso 5: motor lógico simbólico
        self.prolog_engine = PrologEngine()

    def recommend(
        self,
        query: list[str],
        top_k: int = 5,
        filters: ScoreFilters | None = None,
        disliked_tags: str | None = None,
        max_price: float = 0.0,
    ) -> list[EnrichedScore]:
        """
        Ejecuta el pipeline completo y retorna recomendaciones enriquecidas.

        Args:
            query:         Texto de búsqueda o lista de tags como string.
            top_k:         Número máximo de resultados finales.
            filters:       Filtros opcionales de precio, fecha, rating, etc.
            disliked_tags: Tags que el usuario rechaza (penalización Capa 7).
            max_price:     Precio máximo aceptado. 0 = sin límite.

        Returns:
            Lista de EnrichedScore ordenada por relevancia.
        """
        # Paso 1: la query viene de la request (ya disponible)

        # Paso 2: búsqueda semántica → {app_id: similarity}
        # Se recuperan más candidatos de los necesarios para que Prolog tenga
        # material suficiente (se filtrará en Paso 5).
        query_tags = query if isinstance(query, list) else [t.strip() for t in query.split(",")]

        semantic_results = self.semantic_engine.search(tags=query_tags, top_k=top_k * 5)
        semantic_scores  = dict(semantic_results)

        # Paso 3: IDs → objetos Game
        games = self.knowledge_engine.get_games(list(semantic_scores.keys()))

        # Paso 4: scoring por parameters → [GameScore, ...]
        scored = self.parameters_engine.score(games, semantic_scores, filters)

        # Paso 5: motor lógico → filtrar + enriquecer → [EnrichedScore, ...]
        games_by_id = {g.app_id: g for g in games}

        # Configurar preferencias del motor lógico para esta request
        if disliked_tags:
            if isinstance(disliked_tags, list):
                self.prolog_engine.disliked_tags = {t.strip() for t in disliked_tags}
            else:
                self.prolog_engine.disliked_tags = {t.strip() for t in disliked_tags.split(",")}
        else:
            self.prolog_engine.disliked_tags = set()
        self.prolog_engine.max_price = max_price

        enriched = self.prolog_engine.filter(scored, games_by_id, query_tags)

        return enriched[:top_k]


