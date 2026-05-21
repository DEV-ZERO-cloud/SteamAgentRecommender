"""
parameters_engine.py – Motor de scoring basado en parameters.json

Fórmula:
    score = (date * w_date) + (price * w_price) + (positive_rate * w_rate) + (recommendations * w_rec)

    donde cada variable es 0 o 1:
        - 1 si el juego cumple el criterio (o si el filtro no está activo)
        - 0 si el juego NO cumple el criterio

Pesos por defecto (parameters.json):
    date=10, price=40, positive_rate=30, recommendations=20  → max score = 100

Flujo: semantic → knowledge → parameters_engine
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from models.game import Game
import os
from dotenv import load_dotenv

load_dotenv()

# ── Modelo de filtros (viene del body de la request) ─────────────────────────

@dataclass
class ScoreFilters:
    """Parámetros opcionales que llegan desde el endpoint."""
    isPrice: bool = False
    MinPrice: float = 0.0
    MaxPrice: float = 999999.0

    isPositiveRate: bool = False
    MinPositiveRate: float = 0.0

    isDate: bool = False
    MinYear: int = 0
    MaxYear: int = 9999

    isRecommendations: bool = False
    MinRecommendations: float = 0.0


# ── Modelo de salida ──────────────────────────────────────────────────────────

@dataclass
class GameScore:
    """Score final de un juego tras aplicar parameters."""
    app_id: int
    name: str
    semantic_score: float   # similitud del motor semántico (0-1)
    parameter_score: float  # score de parameters (0-100)
    # desglose de criterios (útil para debug y explicaciones)
    date_flag: int
    price_flag: int
    positive_rate_flag: int
    recommendations_flag: int


# ── Motor ─────────────────────────────────────────────────────────────────────

class ParametersEngine:
    """
    Aplica las reglas de parameters.json a una lista de objetos Game.

    Uso:
        engine = ParametersEngine("src/data/parameters.json")
        scores = engine.score(games, semantic_scores, filters)
    """

    def __init__(self, parameters_path: str | Path):
        self._weights = self._load_weights(parameters_path)

    # ── API pública ───────────────────────────────────────────────────────────

    def score(
        self,
        games: list[Game],
        semantic_scores: dict[int, float],
        filters: Optional[ScoreFilters] = None,
    ) -> list[GameScore]:
        """
        Calcula el score de cada juego y retorna la lista ordenada de mayor a menor.

        Args:
            games:           Lista de objetos Game (output de KnowledgeEngine).
            semantic_scores: {app_id: similarity} del motor semántico.
            filters:         Filtros opcionales del body de la request.

        Returns:
            Lista de GameScore ordenada por parameter_score desc.
        """
        if filters is None:
            filters = ScoreFilters()

        w_date, w_price, w_rate, w_rec = self._weights
        results: list[GameScore] = []

        for game in games:
            date_flag         = self._eval_date(game, filters)
            price_flag        = self._eval_price(game, filters)
            positive_rate_flag = self._eval_positive_rate(game, filters)
            recommendations_flag = self._eval_recommendations(game, filters)

            parameter_score = (
                date_flag          * w_date  +
                price_flag         * w_price +
                positive_rate_flag * w_rate  +
                recommendations_flag * w_rec
            )

            results.append(GameScore(
                app_id               = game.app_id,
                name                 = game.name,
                semantic_score       = round(semantic_scores.get(game.app_id, 0.0), 6),
                parameter_score      = parameter_score,
                date_flag            = date_flag,
                price_flag           = price_flag,
                positive_rate_flag   = positive_rate_flag,
                recommendations_flag = recommendations_flag,
            ))

        return sorted(results, key=lambda s: s.parameter_score, reverse=True)

    # ── Evaluadores de criterios ──────────────────────────────────────────────

    def _eval_price(self, game: Game, f: ScoreFilters) -> int:
        if not f.isPrice:
            return 1
        return 1 if f.MinPrice <= game.price <= f.MaxPrice else 0

    def _eval_positive_rate(self, game: Game, f: ScoreFilters) -> int:
        if not f.isPositiveRate:
            return 1
        return 1 if game.positive_ratio >= f.MinPositiveRate else 0

    def _eval_date(self, game: Game, f: ScoreFilters) -> int:
        if not f.isDate:
            return 1
        if game.release_year is None:
            return 0
        return 1 if f.MinYear <= game.release_year <= f.MaxYear else 0

    def _eval_recommendations(self, game: Game, f: ScoreFilters) -> int:
        if not f.isRecommendations:
            return 1
        return 1 if game.recommendations_quantity >= f.MinRecommendations else 0

    # ── Carga de pesos ────────────────────────────────────────────────────────

    @staticmethod
    def _load_weights(path: str | Path) -> tuple[int, int, int, int]:
        if path is None:
            raise ValueError("parameters_path no puede ser None")

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        w = data["default_condition"]
        return w[0], w[1], w[2], w[3]