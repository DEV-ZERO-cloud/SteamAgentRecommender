"""
tests/test_knowledge_engine.py
Tests del KnowledgeEngine: carga de datos, objetos Game y búsqueda semántica.
"""
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from engine.knowledge_engine import KnowledgeEngine

load_dotenv()
CSV_PATH = os.getenv("CSV_PATH")

@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Construye el KnowledgeEngine una sola vez para todos los tests."""
    index_dir = tmp_path_factory.mktemp("embeddings")
    ke = KnowledgeEngine(csv_path=CSV_PATH, index_dir=index_dir)
    ke.build()
    return ke


# ── Carga de datos ────────────────────────────────────────────────────────────

def test_games_loaded(engine):
    assert len(engine.games) > 0, "Debe cargar al menos un juego"

def test_id_to_game_index(engine):
    assert len(engine.id_to_game) == len(engine.games)

def test_game_fields(engine):
    game = engine.games[0]
    assert isinstance(game.app_id, int)
    assert isinstance(game.name, str) and game.name
    assert isinstance(game.tags, list)
    assert isinstance(game.price, float)
    assert 0.0 <= game.recommendations_ratio <= 1.0
    assert 0.0 <= game.rating <= 10.0

def test_release_year_parsed(engine):
    games_with_year = [g for g in engine.games if g.release_year is not None]
    assert len(games_with_year) > 0
    for g in games_with_year:
        assert 1980 <= g.release_year <= 2030

# ── Lookup por ID ─────────────────────────────────────────────────────────────

def test_get_game_existing(engine):
    first = engine.games[0]
    result = engine.get_game(first.app_id)
    assert result is not None
    assert result.app_id == first.app_id

def test_get_game_missing(engine):
    assert engine.get_game(999999999) is None

def test_get_games_list(engine):
    ids = [g.app_id for g in engine.games[:3]]
    result = engine.get_games(ids)
    assert len(result) == 3
    assert [g.app_id for g in result] == ids

def test_get_games_ignores_missing(engine):
    ids = [engine.games[0].app_id, 999999999]
    result = engine.get_games(ids)
    assert len(result) == 1

# ── Búsqueda semántica ────────────────────────────────────────────────────────

def test_semantic_search_returns_results(engine):
    results = engine.semantic_search("dark fantasy RPG", k=10)
    assert isinstance(results, dict)
    assert len(results) > 0

def test_semantic_search_scores_between_0_and_1(engine):
    results = engine.semantic_search("open world adventure", k=10)
    for app_id, score in results.items():
        assert 0.0 <= score <= 1.0, f"Score fuera de rango: {score}"

def test_semantic_search_respects_k(engine):
    k = 5
    results = engine.semantic_search("RPG", k=k)
    assert len(results) <= k

def test_semantic_search_returns_valid_ids(engine):
    results = engine.semantic_search("souls-like dungeon", k=10)
    for app_id in results:
        assert engine.get_game(app_id) is not None

# ── Índice FAISS ──────────────────────────────────────────────────────────────

def test_index_reloads_from_disk(tmp_path):
    """Segunda llamada a build() debe cargar desde disco, no reconstruir."""
    ke = KnowledgeEngine(csv_path=CSV_PATH, index_dir=tmp_path / "idx")
    ke.build()               # construye y guarda
    ke2 = KnowledgeEngine(csv_path=CSV_PATH, index_dir=tmp_path / "idx")
    ke2.build()              # debe cargar desde disco
    assert len(ke2.games) == len(ke.games)

def test_force_rebuild(tmp_path):
    ke = KnowledgeEngine(csv_path=CSV_PATH, index_dir=tmp_path / "idx")
    ke.build()
    ke.build(force=True)     # no debe lanzar excepción
    assert len(ke.games) > 0