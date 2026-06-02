"""
score.py – Calculadora de scores híbrida.

Combina:
  • semantic_score  : similitud coseno entre query embedding y game embedding (FAISS)
  • preference_score: score derivado de reglas Prolog + filtros de preferencias
  • rating_bonus    : boost proporcional al rating del juego
"""
from __future__ import annotations

import numpy as np
from src.models.game import Game
from src.models.user import UserPreferences

# Pesos del score final (deben sumar 1.0)
W_SEMANTIC    = 0.45
W_PREFERENCE  = 0.35
W_RATING      = 0.20


def compute_preference_score(game: Game, prefs: UserPreferences) -> float:
    """
    Score 0-1 basado en:
      - overlap de tags preferidos
      - penalización por tags no deseados
      - filtro de precio
      - filtro de plataforma
    """
    score = 0.0

    # --- Tag overlap (preferidos) ---
    if prefs.preferred_tags:
        game_tags_lower = {t.lower() for t in game.tags}
        pref_lower      = {t.lower() for t in prefs.preferred_tags}
        overlap = len(game_tags_lower & pref_lower)
        score += (overlap / len(pref_lower)) * 0.6

    # --- Sin preferencias de tags: score neutro ---
    else:
        score += 0.3

    # --- Penalización por tags no deseados ---
    if prefs.disliked_tags:
        game_tags_lower = {t.lower() for t in game.tags}
        bad_lower       = {t.lower() for t in prefs.disliked_tags}
        bad_count = len(game_tags_lower & bad_lower)
        penalty = min(bad_count * 0.15, 0.45)
        score -= penalty

    # --- Filtro de precio ---
    if prefs.free_to_play_only and game.price > 0:
        return 0.0
    if prefs.max_price is not None and game.price > prefs.max_price:
        return 0.0

    # --- Filtro rating mínimo ---
    if game.rating < prefs.min_rating:
        return 0.0

    # --- Plataforma preferida ---
    if prefs.preferred_platforms:
        plat_lower = {p.lower() for p in game.platforms}
        pref_plat  = {p.lower() for p in prefs.preferred_platforms}
        if plat_lower & pref_plat:
            score += 0.1

    return float(np.clip(score, 0.0, 1.0))


def compute_rating_bonus(game: Game) -> float:
    """Normaliza el rating de 0-10 a 0-1 con boost en la zona alta."""
    r = np.clip(game.rating, 0.0, 10.0) / 10.0
    # Curva suave: penaliza fuerte por debajo de 0.6, premia sobre 0.8
    return float(r ** 1.5)


def combine_scores(
    semantic_score: float,
    preference_score: float,
    rating_bonus: float,
) -> float:
    """Score final ponderado."""
    return float(
        W_SEMANTIC   * np.clip(semantic_score, 0.0, 1.0)
        + W_PREFERENCE * np.clip(preference_score, 0.0, 1.0)
        + W_RATING     * np.clip(rating_bonus, 0.0, 1.0)
    )