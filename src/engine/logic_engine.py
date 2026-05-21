"""
motor_logico.py – Capa lógica simbólica en Python puro.

Mismas reglas que la versión pyDatalog pero sin dependencias externas.
Implementa las 8 capas del motor híbrido neuro-simbólico como funciones
Python puras, manteniendo el mismo contrato de interfaz.

CAPAS:
    1. played / high_engagement
    2. likes_tag
    3. similar_game       (overlap de tags >= 2)
    4. subgenre / is_rpg  (tags canónicos RPG)
    5. candidate / valid_candidate
    6. discounted_game    (inactiva: sin columna discount en CSV)
    7. price_tier / rating_tier + tag_overlap_score / dislike_penalty
    8. recommend / explanation
"""
from __future__ import annotations
from dotenv import load_dotenv

import os
import pandas as pd

load_dotenv()
# =============================================================================
# § 4 — TAGS CANÓNICOS RPG (lowercase, igual que los tags normalizados del CSV)
# =============================================================================

TAGS_PATH = os.getenv("TAGS_PATH")
_rpg_tags: set[str] | None = None

def _get_rpg_tags() -> set[str]:
    global _rpg_tags
    if _rpg_tags is None:
        if not TAGS_PATH:
            raise EnvironmentError("TAGS_PATH no está definida en el entorno.")
        _rpg_tags = {row.lower() for row in pd.read_csv(TAGS_PATH, sep="|")["name"]}
    return _rpg_tags

# =============================================================================
# § 3 — SIMILARIDAD
# =============================================================================

def similar_game(tags1: set[str], tags2: set[str]) -> bool:
    """Dos juegos son similares si comparten al menos 2 tags distintos."""
    return len(tags1 & tags2) >= 2


def compute_similar_pairs(tags_by_gid: dict[str, set[str]]) -> set[tuple[str, str]]:
    """
    Calcula todos los pares similares sobre el conjunto de candidatos.
    O(n²) pero n <= 50, barato en la práctica.
    Retorna set de (gid1, gid2) donde similar_game es True.
    """
    gids = list(tags_by_gid.keys())
    pairs: set[tuple[str, str]] = set()
    for i, g1 in enumerate(gids):
        for g2 in gids[i + 1:]:
            if similar_game(tags_by_gid[g1], tags_by_gid[g2]):
                pairs.add((g1, g2))
                pairs.add((g2, g1))  # simétrico
    return pairs


# =============================================================================
# § 4 — DOMINIO RPG
# =============================================================================

def is_rpg(game_tags: set[str]) -> bool:
    return bool(game_tags & _get_rpg_tags())

def get_subgenres(game_tags: set[str]) -> set[str]:
    return game_tags & _get_rpg_tags()

# =============================================================================
# § 5 — CANDIDATOS
# =============================================================================

def get_candidates(
    liked_tags: set[str],
    tags_by_gid: dict[str, set[str]],
    similar_pairs: set[tuple[str, str]],
    played_gids: set[str],
) -> set[str]:
    """
    Un juego es candidato si:
      A) tiene al menos un tag que el usuario prefiere, O
      B) es similar a un juego que el usuario jugó.

    valid_candidate = candidato que el usuario NO ha jugado.
    Con usuario sintético played_gids siempre es vacío.
    """
    candidates: set[str] = set()

    # Vía A — por tag preferido
    for gid, tags in tags_by_gid.items():
        if tags & liked_tags:
            candidates.add(gid)

    # Vía B — por similaridad con juego jugado
    for played_gid in played_gids:
        for g1, g2 in similar_pairs:
            if g1 == played_gid:
                candidates.add(g2)

    # valid_candidate: excluir jugados
    return candidates - played_gids


# =============================================================================
# § 7 — CLASIFICADORES Y SCORING
# =============================================================================

def get_price_tier(price: float) -> str:
    """Clasifica el precio en 5 niveles."""
    if price == 0:
        return 'free'
    elif price <= 10:
        return 'budget'
    elif price <= 30:
        return 'mid_range'
    elif price <= 60:
        return 'premium'
    else:
        return 'deluxe'


def get_rating_tier(rating: float) -> str:
    """Clasifica el rating en 5 niveles."""
    if rating >= 9.0:
        return 'masterpiece'
    elif rating >= 8.0:
        return 'excellent'
    elif rating >= 7.0:
        return 'good'
    elif rating >= 5.0:
        return 'mixed'
    else:
        return 'poor'


def tag_overlap_score(game_tags: set[str], preferred_tags: set[str]) -> float:
    """
    Score de solapamiento entre tags del juego y preferidos del usuario.
    Formula: |game_tags ∩ preferred_tags| / |preferred_tags|
    Retorna [0.0, 1.0].
    """
    if not preferred_tags:
        return 0.0
    return len(game_tags & preferred_tags) / len(preferred_tags)


def dislike_penalty(game_tags: set[str], disliked_tags: set[str]) -> float:
    """
    Penalización por tags no deseados.
    Formula: |game_tags ∩ disliked_tags| x 0.15
    Umbral de exclusión: >= 0.45
    """
    return len(game_tags & disliked_tags) * 0.15


def is_recommendable(
    game_tags: set[str],
    disliked_tags: set[str],
    max_price: float,
    price: float,
) -> bool:
    """
    True si el juego pasa todos los filtros de calidad:
      1. Es un RPG.
      2. dislike_penalty < 0.45.
      3. price <= max_price (si max_price > 0).
    """
    # Solo filtra por dislike y precio
    if dislike_penalty(game_tags, disliked_tags) >= 0.45:
        return False
    if max_price > 0 and price > max_price:
        return False
    return True


# =============================================================================
# § 8 — EXPLICABILIDAD
# =============================================================================

def get_explanations(
    gid: str,
    liked_tags: set[str],
    tags_by_gid: dict[str, set[str]],
    similar_pairs: set[tuple[str, str]],
    played_gids: set[str],
) -> list[str]:
    """
    Retorna las razones estructuradas por las que se recomienda gid.

    Razones posibles:
        'reason_tag'      — comparte un tag preferido
        'reason_similar'  — similar a un juego jugado
        'reason_subgenre' — pertenece al subgénero RPG
    """
    reasons: set[str] = set()
    game_tags = tags_by_gid.get(gid, set())

    # reason_tag
    if game_tags & liked_tags:
        reasons.add('reason_tag')

    # reason_similar
    for played_gid in played_gids:
        if (played_gid, gid) in similar_pairs:
            reasons.add('reason_similar')
            break

    # reason_subgenre
    if get_subgenres(game_tags):
        reasons.add('reason_subgenre')

    return list(reasons)
