"""
test_semantic_engine.py – Tests unitarios del motor de búsqueda semántica.

Ejecutar:
    pytest test_semantic_engine.py -v
"""

from __future__ import annotations

import pytest
import pandas as pd
import os
from dotenv import load_dotenv
from unittest.mock import patch

from engine.semantic_engine import SemanticEngine, SemanticEngineConfig
from scripts.prepare_data import PrepareData, PreparedData

load_dotenv()
CSV_PATH = os.getenv("CSV_PATH")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_CSV_DATA = """appid|name|tags|genres
1|The Witcher 3|open-world;story-rich;fantasy;rpg|rpg;adventure
2|Dark Souls|difficult;action;dark-fantasy;rpg|rpg;action
3|Stardew Valley|farming;relaxing;casual;simulation|simulation;indie
4|Baldur's Gate 3|turn-based;story-rich;fantasy;rpg|rpg;strategy
5|Hollow Knight|platformer;difficult;action;indie|action;indie
"""


@pytest.fixture
def tmp_csv(tmp_path) -> str:
    """Crea un CSV temporal con datos mock y retorna su path."""
    csv_file = tmp_path / "steam_rpg.csv"
    csv_file.write_text(MOCK_CSV_DATA, encoding="utf-8")
    return str(csv_file)


@pytest.fixture
def config(tmp_csv) -> SemanticEngineConfig:
    return SemanticEngineConfig(csv_path=tmp_csv)


@pytest.fixture
def prepared(config) -> PreparedData:
    return PrepareData(config).prepare()


@pytest.fixture
def engine(config, prepared) -> SemanticEngine:
    return SemanticEngine(config, prepared)


# ---------------------------------------------------------------------------
# PrepareData
# ---------------------------------------------------------------------------

class TestPrepareData:

    def test_df_tiene_filas_correctas(self, prepared):
        assert len(prepared.df) == 5

    def test_columnas_presentes(self, prepared):
        assert "appid" in prepared.df.columns
        assert "tags" in prepared.df.columns
        assert "genres" in prepared.df.columns
        assert "name" in prepared.df.columns

    def test_tags_normalizados_a_minusculas(self, prepared):
        for valor in prepared.df["tags"]:
            assert valor == valor.lower()

    def test_text_field_combina_tags_y_generos(self, prepared):
        primera_fila = prepared.df.iloc[0]
        assert primera_fila["tags"] in primera_fila["_text"]
        assert primera_fila["genres"] in primera_fila["_text"]

    def test_vectorizer_ajustado(self, prepared):
        # Si el vectorizador está ajustado, tiene vocabulario
        assert len(prepared.vectorizer.vocabulary_) > 0

    def test_matrix_dimensiones(self, prepared):
        n_juegos, n_terminos = prepared.matrix.shape
        assert n_juegos == 5
        assert n_terminos > 0

    def test_columnas_faltantes_lanza_error(self, tmp_path):
        csv_roto = tmp_path / "roto.csv"
        csv_roto.write_text("appid|name\n1|Juego\n", encoding="utf-8")
        config = SemanticEngineConfig(csv_path=str(csv_roto))
        with pytest.raises(ValueError, match="Columnas faltantes"):
            PrepareData(config).prepare()

    def test_tags_vacios_no_rompen_pipeline(self, tmp_path):
        csv = tmp_path / "vacios.csv"
        csv.write_text("appid|name|tags|genres\n1|Juego||rpg\n", encoding="utf-8")
        config = SemanticEngineConfig(csv_path=str(csv))
        prepared = PrepareData(config).prepare()
        assert prepared.df.iloc[0]["tags"] == ""


# ---------------------------------------------------------------------------
# SemanticEngine.search()
# ---------------------------------------------------------------------------

class TestSemanticEngineSearch:

    def test_retorna_lista_de_tuplas(self, engine):
        results = engine.search(["fantasy", "rpg"])
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_game_id_es_entero(self, engine):
        results = engine.search(["fantasy"])
        for game_id, _ in results:
            assert isinstance(game_id, int)

    def test_score_es_float_entre_0_y_1(self, engine):
        results = engine.search(["fantasy"])
        for _, score in results:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_resultados_ordenados_descendente(self, engine):
        results = engine.search(["fantasy", "rpg", "story-rich"])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limita_resultados(self, engine):
        results = engine.search(["rpg"], top_k=2)
        assert len(results) <= 2

    def test_min_score_filtra_irrelevantes(self, engine):
        results = engine.search(["fantasy"], min_score=0.99)
        for _, score in results:
            assert score >= 0.99

    def test_tags_relevantes_rankean_primero(self, engine):
        """Witcher 3 y BG3 deben aparecer antes que Stardew para tags de fantasy RPG."""
        results = engine.search(["fantasy", "story-rich", "rpg"], top_k=5)
        top_ids = [game_id for game_id, _ in results[:2]]
        # AppID 1 = Witcher 3, AppID 4 = Baldur's Gate 3
        assert 1 in top_ids or 4 in top_ids

    def test_tags_vacios_retorna_lista_vacia(self, engine):
        results = engine.search([])
        assert results == []

    def test_tags_con_espacios_se_normalizan(self, engine):
        """Tags con espacios extra no deben romper el motor."""
        results = engine.search(["  fantasy  ", " rpg "])
        assert isinstance(results, list)

    def test_tag_inexistente_retorna_lista_vacia_o_score_bajo(self, engine):
        """Un tag que no existe en el corpus debe dar score 0 en todos."""
        results = engine.search(["xyztaginexistente123"], min_score=0.01)
        assert results == []

    def test_top_k_por_defecto_usa_config(self, engine):
        """Sin top_k explícito debe usar config.default_top_k (20 > 5 juegos)."""
        results = engine.search(["rpg"])
        assert len(results) <= engine.config.default_top_k

    def test_min_score_por_defecto_usa_config(self, engine):
        results = engine.search(["rpg"])
        for _, score in results:
            assert score >= engine.config.default_min_score


# ---------------------------------------------------------------------------
# SemanticEngineConfig
# ---------------------------------------------------------------------------

class TestSemanticEngineConfig:

    def test_valores_por_defecto(self, tmp_csv):
        config = SemanticEngineConfig(csv_path=tmp_csv)
        assert config.separator == "|"
        assert config.tag_column == "tags"
        assert config.genre_column == "genres"
        assert config.id_column == "appid"
        assert config.default_top_k == 20
        assert config.default_min_score == 0.05

    def test_config_es_inmutable(self, tmp_csv):
        config = SemanticEngineConfig(csv_path=tmp_csv)
        with pytest.raises(Exception):
            config.default_top_k = 999