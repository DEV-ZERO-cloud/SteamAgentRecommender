"""
strategy.py – Patrón Strategy para las etapas del pipeline de recomendación.

Cada Strategy encapsula una fase:
  1. FilterStrategy  – filtra candidatos según reglas duras
  2. RankStrategy    – ordena por score híbrido
  3. DiversifyStrategy – diversificación por subgénero (evitar repetición)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from models.game import Game
from models.user import UserPreferences
from models.score import compute_preference_score, compute_rating_bonus, combine_scores


# ─────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────
class RecommendationStrategy(ABC):
    @abstractmethod
    def apply(self, games: list[Game], **kwargs) -> list[Game]:
        ...


# ─────────────────────────────────────────────
# 1. Filtrado por reglas duras
# ─────────────────────────────────────────────
class FilterStrategy(RecommendationStrategy):
    """Elimina juegos que no cumplen restricciones del usuario."""

    def apply(
        self,
        games: list[Game],
        prefs: UserPreferences,
        played_ids: list[int] | None = None,
        **kwargs,
    ) -> list[Game]:
        played_ids = set(played_ids or [])
        result = []

        for g in games:
            # excluir ya jugados
            if g.app_id in played_ids:
                continue
            # free-to-play obligatorio
            if prefs.free_to_play_only and g.price > 0:
                continue
            # precio máximo
            if prefs.max_price is not None and g.price > prefs.max_price:
                continue
            # rating mínimo
            if g.rating < prefs.min_rating:
                continue
            # tags excluidos (si hay >2 matches de tags no deseados → skip)
            if prefs.disliked_tags:
                game_tags_lower = {t.lower() for t in g.tags}
                bad = sum(1 for t in prefs.disliked_tags if t.lower() in game_tags_lower)
                if bad > 2:
                    continue
            result.append(g)

        return result


# ─────────────────────────────────────────────
# 2. Ranking híbrido
# ─────────────────────────────────────────────
class RankStrategy(RecommendationStrategy):
    """Ordena los juegos por score combinado (semántico + preferencias + rating)."""

    def apply(
        self,
        games: list[Game],
        semantic_scores: dict[int, float] | None = None,
        prefs: UserPreferences | None = None,
        **kwargs,
    ) -> list[Game]:
        semantic_scores = semantic_scores or {}
        prefs = prefs or UserPreferences()

        def _score(g: Game) -> float:
            sem  = semantic_scores.get(g.app_id, 0.0)
            pref = compute_preference_score(g, prefs)
            rat  = compute_rating_bonus(g)
            return combine_scores(sem, pref, rat)

        return sorted(games, key=_score, reverse=True)


# ─────────────────────────────────────────────
# 3. Diversificación por subgénero
# ─────────────────────────────────────────────
_SUBGENRE_MAP: dict[str, list[str]] = {
    "action_rpg":    ["action rpg", "hack and slash"],
    "jrpg":          ["jrpg", "anime", "turn-based"],
    "open_world":    ["open world", "exploration", "sandbox"],
    "tactical_rpg":  ["tactical rpg", "turn-based strategy"],
    "roguelike_rpg": ["roguelike", "roguelite", "dungeon crawler"],
    "crpg":          ["party-based rpg", "classic rpg", "isometric"],
    "soulslike":     ["souls-like", "difficult", "dark fantasy"],
}


def _detect_subgenre(game: Game) -> str:
    tags_lower = {t.lower() for t in game.tags}
    best = ("other", 0)
    for sub, keywords in _SUBGENRE_MAP.items():
        hits = sum(1 for kw in keywords if kw in tags_lower)
        if hits > best[1]:
            best = (sub, hits)
    return best[0]


class DiversifyStrategy(RecommendationStrategy):
    """
    Limita la cantidad de juegos por subgénero para evitar listas monótonas.
    Por defecto max 2 juegos por subgénero en los primeros N resultados.
    """

    def __init__(self, max_per_subgenre: int = 2, top_n: int = 10):
        self.max_per_subgenre = max_per_subgenre
        self.top_n = top_n

    def apply(self, games: list[Game], **kwargs) -> list[Game]:
        seen: dict[str, int] = {}
        diverse: list[Game] = []
        overflow: list[Game] = []

        for g in games:
            sub = _detect_subgenre(g)
            count = seen.get(sub, 0)
            if count < self.max_per_subgenre:
                seen[sub] = count + 1
                diverse.append(g)
            else:
                overflow.append(g)

            if len(diverse) >= self.top_n:
                break

        return diverse + overflow