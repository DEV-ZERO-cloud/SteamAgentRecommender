"""
explanation.py – Genera explicaciones legibles para cada recomendación.

Las explicaciones se construyen de forma determinista (sin LLM) usando
las puntuaciones parciales y los metadatos del juego.
"""
from __future__ import annotations

from models.game import Game
from models.user import UserPreferences


def generate_explanation(
    game: Game,
    prefs: UserPreferences,
    semantic_score: float,
    preference_score: float,
    final_score: float,
) -> str:
    parts: list[str] = []

    # --- Coincidencia de tags ---
    if prefs.preferred_tags:
        game_tags_lower = {t.lower() for t in game.tags}
        matched = [t for t in prefs.preferred_tags if t.lower() in game_tags_lower]
        if matched:
            parts.append(f"Coincide con tus tags favoritos: {', '.join(matched)}.")

    # --- Similitud semántica ---
    if semantic_score >= 0.75:
        parts.append("Muy similar semánticamente a tu búsqueda.")
    elif semantic_score >= 0.50:
        parts.append("Bastante relacionado con lo que buscas.")

    # --- Rating ---
    if game.rating >= 9.0:
        parts.append(f"Calificación excepcional: {game.rating:.1f}/10 ({game.recommendations_ratio*100:.0f}% positivas).")
    elif game.rating >= 8.0:
        parts.append(f"Muy bien valorado: {game.rating:.1f}/10.")
    elif game.rating >= 7.0:
        parts.append(f"Buena recepción de la comunidad: {game.rating:.1f}/10.")

    # --- Precio ---
    if game.price == 0:
        parts.append("Free to Play – sin costo.")
    elif game.price <= 10:
        parts.append(f"Precio accesible: ${game.price:.2f}.")

    # --- Plataformas ---
    if prefs.preferred_platforms and game.platforms:
        plat_lower = {p.lower() for p in game.platforms}
        matched_plat = [p for p in prefs.preferred_platforms if p.lower() in plat_lower]
        if matched_plat:
            parts.append(f"Disponible en {', '.join(matched_plat)}.")

    # --- Score final como resumen ---
    pct = int(final_score * 100)
    parts.append(f"Score de compatibilidad: {pct}%.")

    return " ".join(parts) if parts else "Recomendado por similitud general con tu perfil."