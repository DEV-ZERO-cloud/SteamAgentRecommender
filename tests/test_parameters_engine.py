"""
tests/test_parameters_engine.py
Tests del ParametersEngine con datos sintéticos y reales.

Juego real: Project Zomboid (app_id=108600)
    price                = 19.99
    recommendations_ratio       = 138070 / 146874 ≈ 0.9401
    release_year         = 2013
"""
import pytest
import os
from engine.parameters_engine import ParametersEngine, ScoreFilters
from models.game import Game
from dotenv import load_dotenv

load_dotenv()
PARAMETERS_PATH = os.getenv(
    "PARAMETERS_PATH",
    "src/knowledge/parameters.json"
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return ParametersEngine(PARAMETERS_PATH)


def make_game(**kwargs) -> Game:
    """Crea un Game con valores base sobreescribibles."""
    defaults = dict(
        app_id=1,
        name="Test Game",
        price=10.0,
        recommendations_ratio=0.85,
        release_year=2020,
        tags=["rpg"],
        genres=["RPG"],
        categories=[],
        positive_reviews=850,
        negative_reviews=150,
        total_reviews=1000,
        rating=8.0,
    )
    defaults.update(kwargs)
    return Game(**defaults)


PROJECT_ZOMBOID = Game(
    app_id                   = 108600,
    name                     = "Project Zomboid",
    price                    = 19.99,
    positive_reviews         = 138070,
    negative_reviews         = 8804,
    total_reviews            = 146874,
    recommendations_ratio           = round(138070 / 146874, 4),  # 0.9401
    release_year             = 2013,
    tags                     = ["Survival", "Zombies", "Open World", "Multiplayer", "Sandbox",
                                "Post-apocalyptic", "Co-op", "Crafting", "Building", "Indie",
                                "Simulation", "RPG", "Survival Horror", "Realistic", "Isometric"],
    genres                   = ["Indie", "RPG", "Simulation", "Early Access"],
    categories               = ["Single-player", "Multi-player", "Co-op"],
    rating                   = 8.0,
)

SEMANTIC_SCORES_SYNTHETIC = {1: 0.9, 2: 0.7, 3: 0.5}
SEMANTIC_SCORES_REAL      = {108600: 0.87}


# ── Sin filtros → score = 100 ─────────────────────────────────────────────────

def test_sin_filtros_score_es_100(engine):
    scored = engine.score([make_game()], SEMANTIC_SCORES_SYNTHETIC, ScoreFilters())
    assert scored[0].parameter_score == 100

def test_sin_filtros_score_es_100_real(engine):
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL, ScoreFilters())
    assert scored[0].parameter_score == 100

def test_sin_filtros_todos_flags_son_1(engine):
    scored = engine.score([make_game()], SEMANTIC_SCORES_SYNTHETIC, ScoreFilters())
    s = scored[0]
    assert s.price_flag == 1
    assert s.positive_rate_flag == 1
    assert s.date_flag == 1
    assert s.recommendations_flag == 1

def test_sin_filtros_todos_flags_son_1_real(engine):
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL, ScoreFilters())
    s = scored[0]
    assert s.price_flag == 1
    assert s.positive_rate_flag == 1
    assert s.date_flag == 1
    assert s.recommendations_flag == 1


# ── Filtro de precio ──────────────────────────────────────────────────────────

def test_precio_dentro_del_rango(engine):
    scored = engine.score([make_game(price=15.0)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isPrice=True, MinPrice=10.0, MaxPrice=20.0))
    assert scored[0].price_flag == 1

def test_precio_fuera_del_rango(engine):
    scored = engine.score([make_game(price=25.0)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isPrice=True, MinPrice=10.0, MaxPrice=20.0))
    assert scored[0].price_flag == 0

def test_precio_inactivo_siempre_es_1(engine):
    scored = engine.score([make_game(price=999.0)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isPrice=False))
    assert scored[0].price_flag == 1

def test_precio_real_cumple_rango_10_a_20(engine):
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isPrice=True, MinPrice=10.0, MaxPrice=20.0))
    assert scored[0].price_flag == 1

def test_precio_real_no_cumple_rango_0_a_10(engine):
    # 19.99 > 10 → no cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isPrice=True, MinPrice=0.0, MaxPrice=10.0))
    assert scored[0].price_flag == 0

def test_precio_real_exacto_en_limite_superior(engine):
    # 19.99 <= 19.99 → cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isPrice=True, MinPrice=0.0, MaxPrice=19.99))
    assert scored[0].price_flag == 1


# ── Filtro de positive rate ───────────────────────────────────────────────────

def test_positive_rate_cumple(engine):
    scored = engine.score([make_game(recommendations_ratio=0.9)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isPositiveRate=True, MinPositiveRate=0.8))
    assert scored[0].positive_rate_flag == 1

def test_positive_rate_no_cumple(engine):
    scored = engine.score([make_game(recommendations_ratio=0.7)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isPositiveRate=True, MinPositiveRate=0.8))
    assert scored[0].positive_rate_flag == 0

def test_positive_rate_inactivo_siempre_es_1(engine):
    scored = engine.score([make_game(recommendations_ratio=0.0)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isPositiveRate=False))
    assert scored[0].positive_rate_flag == 1

def test_positive_rate_real_cumple_umbral_0_9(engine):
    # 0.9401 >= 0.9 → cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isPositiveRate=True, MinPositiveRate=0.9))
    assert scored[0].positive_rate_flag == 1

def test_positive_rate_real_no_cumple_umbral_0_95(engine):
    # 0.9401 < 0.95 → no cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isPositiveRate=True, MinPositiveRate=0.95))
    assert scored[0].positive_rate_flag == 0


# ── Filtro de fecha ───────────────────────────────────────────────────────────

def test_fecha_dentro_del_rango(engine):
    scored = engine.score([make_game(release_year=2020)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isDate=True, MinYear=2015, MaxYear=2025))
    assert scored[0].date_flag == 1

def test_fecha_fuera_del_rango(engine):
    scored = engine.score([make_game(release_year=2010)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isDate=True, MinYear=2015, MaxYear=2025))
    assert scored[0].date_flag == 0

def test_fecha_none_es_0(engine):
    scored = engine.score([make_game(release_year=None)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isDate=True, MinYear=2015, MaxYear=2025))
    assert scored[0].date_flag == 0

def test_fecha_inactiva_siempre_es_1(engine):
    scored = engine.score([make_game(release_year=1990)], SEMANTIC_SCORES_SYNTHETIC,
                          ScoreFilters(isDate=False))
    assert scored[0].date_flag == 1

def test_fecha_real_cumple_rango_2010_2020(engine):
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isDate=True, MinYear=2010, MaxYear=2020))
    assert scored[0].date_flag == 1

def test_fecha_real_no_cumple_rango_2015_2025(engine):
    # 2013 < 2015 → no cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isDate=True, MinYear=2015, MaxYear=2025))
    assert scored[0].date_flag == 0

def test_fecha_real_exacta_en_limite_inferior(engine):
    # MinYear=2013 → 2013 >= 2013 → cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isDate=True, MinYear=2013, MaxYear=2025))
    assert scored[0].date_flag == 1

def test_recommendations_real_no_cumple_minimo_1(engine):
    # 0.0 < 1 → no cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isRecommendations=True, MinRecommendations=1.0))
    assert scored[0].recommendations_flag == 0

def test_recommendations_real_cumple_minimo_0(engine):
    # 0.0 >= 0.0 → cumple
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL,
                          ScoreFilters(isRecommendations=True, MinRecommendations=0.0))
    assert scored[0].recommendations_flag == 1


# ── Scores parciales ──────────────────────────────────────────────────────────

def test_score_real_precio_y_positive_rate_cumplen(engine):
    # price=1(40) + positive_rate=1(30) + date=0 + recommendations=0 = 70
    filters = ScoreFilters(
        isPrice=True,           MinPrice=10.0, MaxPrice=20.0,
        isPositiveRate=True,    MinPositiveRate=0.8,
        isDate=True,            MinYear=2015,  MaxYear=2025,  # 2013 no cumple
        isRecommendations=True, MinRecommendations=1.0,       # 0.0 no cumple
    )
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL, filters)
    assert scored[0].parameter_score == 70

def test_score_real_solo_positive_rate_cumple(engine):
    # price=0 + positive_rate=1(30) + date=0 + recommendations=0 = 30
    filters = ScoreFilters(
        isPrice=True,           MinPrice=0.0,  MaxPrice=10.0,  # 19.99 no cumple
        isPositiveRate=True,    MinPositiveRate=0.8,
        isDate=True,            MinYear=2015,  MaxYear=2025,   # 2013 no cumple
        isRecommendations=True, MinRecommendations=1.0,        # 0.0 no cumple
    )
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL, filters)
    assert scored[0].parameter_score == 30

def test_score_real_ninguno_cumple(engine):
    filters = ScoreFilters(
        isPrice=True,           MinPrice=0.0,  MaxPrice=10.0,
        isPositiveRate=True,    MinPositiveRate=0.95,
        isDate=True,            MinYear=2015,  MaxYear=2025,
        isRecommendations=True, MinRecommendations=1.0,
    )
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL, filters)
    assert scored[0].parameter_score == 0


# ── Ordenamiento ──────────────────────────────────────────────────────────────


# ── GameScore fields ──────────────────────────────────────────────────────────

def test_game_score_semantic_score(engine):
    scored = engine.score([make_game()], {1: 0.75}, ScoreFilters())
    assert scored[0].semantic_score == 0.75

def test_game_score_semantic_score_real(engine):
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL, ScoreFilters())
    assert scored[0].semantic_score == 0.87

def test_game_score_app_id_y_name(engine):
    scored = engine.score([make_game(name="Dark Souls")], {1: 0.9}, ScoreFilters())
    assert scored[0].app_id == 1
    assert scored[0].name == "Dark Souls"

def test_game_score_app_id_y_name_real(engine):
    scored = engine.score([PROJECT_ZOMBOID], SEMANTIC_SCORES_REAL, ScoreFilters())
    assert scored[0].app_id == 108600
    assert scored[0].name == "Project Zomboid"

def test_lista_vacia_retorna_vacia(engine):
    assert engine.score([], {}, ScoreFilters()) == []