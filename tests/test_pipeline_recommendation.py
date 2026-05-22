# test_pipeline.py
"""
Test de integración del pipeline completo.
"""
from __future__ import annotations

import os
import time
import pytest
from dotenv import load_dotenv

load_dotenv()

from engine.parameters_engine import ScoreFilters
from engine.prolog_engine import EnrichedScore
from pipeline.pipeline_recommendation import PipelineRecommendation


# ── Fixture: una sola instancia del pipeline para todos los tests ─────────────

@pytest.fixture(scope="module")
def pipeline():
    """Inicializa el pipeline una sola vez (carga CSV + índice FAISS)."""
    print("\n[fixture] Inicializando PipelineRecommendation…")
    return PipelineRecommendation()


# ── Helper ────────────────────────────────────────────────────────────────────

def assert_enriched(results: list[EnrichedScore], top_k: int):
    """Validaciones comunes a todos los resultados."""
    assert isinstance(results, list), "El resultado debe ser una lista"
    assert len(results) <= top_k,     "No puede superar top_k"
    for r in results:
        assert isinstance(r, EnrichedScore)
        assert r.game_score.app_id > 0
        assert r.game_score.parameter_score >= 0
        assert r.game_score.semantic_score  >= 0
        assert r.price_tier  is not None
        assert r.rating_tier is not None
        assert 0.0 <= r.tag_overlap <= 1.0
        assert isinstance(r.explanations, list)
        assert r.is_rpg, f"Juego {r.game_score.name} no es RPG pero pasó el filtro"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pipeline_basic(pipeline):
    """Flujo mínimo: query simple, sin filtros."""
    results = pipeline.recommend(
        query="rpg, fantasy, open world",
        top_k=5,
    )
    assert_enriched(results, top_k=5)
    assert len(results) > 0, "Debe retornar al menos 1 resultado"


def test_pipeline_returns_only_rpgs(pipeline):
    """Todos los resultados deben ser RPGs según logic_engine."""
    results = pipeline.recommend(
        query="rpg, adventure, story-rich",
        top_k=10,
    )
    for r in results:
        assert r.is_rpg, f"{r.game_score.name} no es RPG"


def test_pipeline_disliked_tags_excluded(pipeline):
    """Juegos con tags rechazados (penalización >= 0.45) no deben aparecer."""
    from engine import logic_engine

    disliked = "horror, gore, violence"
    disliked_set = {t.strip().lower() for t in disliked.split(",")}

    results = pipeline.recommend(
        query         = "rpg, fantasy, turn-based",
        disliked_tags = disliked,
        top_k         = 10,
    )

    for r in results:
        game = pipeline.knowledge_engine.id_to_game.get(r.game_score.app_id)
        assert game is not None
        game_tags = {t.lower().strip() for t in game.tags}
        penalty = logic_engine.dislike_penalty(game_tags, disliked_set)
        assert penalty < 0.45, (
            f"{r.game_score.name} tiene penalty={penalty:.2f} pero apareció en resultados"
        )


def test_pipeline_price_filter(pipeline):
    """Con filtro de precio activo, todos los resultados deben estar en rango."""
    filters = ScoreFilters(
        isPrice  = True,
        MinPrice = 0.0,
        MaxPrice = 20.0,
    )
    results = pipeline.recommend(
        query     = "rpg, indie",
        top_k     = 10,
        filters   = filters,
        max_price = 20.0,
    )
    assert_enriched(results, top_k=10)
    for r in results:
        assert r.price_tier in ("free", "budget", "mid_range"), (
            f"{r.game_score.name} tiene price_tier={r.price_tier} pero max_price=20"
        )


def test_pipeline_date_filter(pipeline):
    """Con filtro de fecha, el parameter_score de juegos fuera de rango debe ser menor."""
    filters = ScoreFilters(
        isDate  = True,
        MinYear = 2018,
        MaxYear = 2024,
    )
    results = pipeline.recommend(
        query   = "rpg, fantasy",
        top_k   = 10,
        filters = filters,
    )
    assert_enriched(results, top_k=10)


def test_pipeline_positive_rate_filter(pipeline):
    """Juegos con recommendations_ratio bajo no deben aparecer con filtro activo."""
    filters = ScoreFilters(
        isPositiveRate  = True,
        MinPositiveRate = 0.80,
    )
    results = pipeline.recommend(
        query   = "rpg, action",
        top_k   = 10,
        filters = filters,
    )
    assert_enriched(results, top_k=10)


def test_pipeline_all_filters_combined(pipeline):
    """Test con todos los filtros activos simultáneamente."""
    filters = ScoreFilters(
        isPrice            = True,
        MinPrice           = 0.0,
        MaxPrice           = 40.0,
        isPositiveRate     = True,
        MinPositiveRate    = 0.75,
        isDate             = True,
        MinYear            = 2015,
        MaxYear            = 2024,
        isRecommendations  = True,
        MinRecommendations = 500.0,
    )
    results = pipeline.recommend(
        query         = ["rpg, fantasy, magic, exploration"],
        disliked_tags = ["horror, gore"],
        top_k         = 5,
        filters       = filters,
        max_price     = 40.0,
    )
    assert_enriched(results, top_k=5)


def test_pipeline_order_by_parameter_score(pipeline):
    """Los resultados deben estar ordenados por parameter_score DESC."""
    results = pipeline.recommend(
        query  = "rpg, turn-based, strategy",
        top_k  = 10,
    )
    scores = [r.game_score.parameter_score for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Orden incorrecto: {scores}"
    )


def test_pipeline_enrichment_explanations(pipeline):
    """Cada resultado debe tener al menos una explicación."""
    results = pipeline.recommend(
        query  = "rpg, fantasy, open world",
        top_k  = 5,
    )
    for r in results:
        assert len(r.explanations) >= 1, (
            f"{r.game_score.name} no tiene explicaciones"
        )
        valid = {"reason_tag", "reason_similar", "reason_subgenre"}
        for exp in r.explanations:
            assert exp in valid, f"Explicación desconocida: {exp}"


def test_pipeline_top_k_respected(pipeline):
    """top_k debe limitar exactamente la cantidad de resultados."""
    for k in [1, 3, 5, 10]:
        results = pipeline.recommend(query="rpg", top_k=k)
        assert len(results) <= k, f"top_k={k} pero se retornaron {len(results)}"


def test_pipeline_no_duplicate_app_ids(pipeline):
    """No debe haber app_ids duplicados en los resultados."""
    results = pipeline.recommend(
        query  = "rpg, adventure",
        top_k  = 10,
    )
    ids = [r.game_score.app_id for r in results]
    assert len(ids) == len(set(ids)), f"IDs duplicados: {ids}"


def test_pipeline_performance(pipeline):
    """El pipeline completo debe responder en menos de 5 segundos."""
    start = time.time()
    pipeline.recommend(query="rpg, fantasy, open world", top_k=5)
    elapsed = time.time() - start
    assert elapsed < 5.0, f"Pipeline tardó {elapsed:.2f}s (límite: 5s)"


def test_pipeline_empty_disliked(pipeline):
    """disliked_tags vacío no debe crashear."""
    results = pipeline.recommend(
        query         = "rpg, fantasy",
        disliked_tags = "",
        top_k         = 5,
    )
    assert isinstance(results, list)


def test_pipeline_query_single_tag(pipeline):
    """Query con un solo tag debe retornar resultados válidos."""
    results = pipeline.recommend(query="rpg", top_k=5)
    assert_enriched(results, top_k=5)

