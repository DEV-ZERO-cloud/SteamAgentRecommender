"""
test_prolog_engine.py – Evaluación del motor lógico-simbólico (Paso 5 / PrologEngine).

Métricas:
    - Accuracy   : predicciones correctas sobre el total de juegos evaluados
    - Precision  : de los recomendados por Prolog, ¿cuántos son correctos?
    - Recall     : de los realmente recomendables, ¿cuántos detectó Prolog?
    - F1-score   : media armónica
    - Support    : juegos positivos / negativos en cada caso

Estrategia:
    El ground truth se computa de forma determinista re-implementando las reglas
    del motor lógico fuera del engine:
      1. is_rpg(game_tags)                           → True/False
      2. dislike_penalty(game_tags, disliked) < 0.45
      3. price <= max_price (si max_price > 0)
      4. Al menos 1 tag compartido con liked_tags (candidate via A)

    Adicionalmente se evalúan sub-componentes del motor lógico como unidades
    individuales (similar_game, price_tier, rating_tier, tag_overlap_score,
    dislike_penalty), verificando que su salida coincide con la implementación.

Casos de prueba:
    Se diseñan escenarios con distintas combinaciones de liked_tags,
    disliked_tags y max_price para medir la sensibilidad del filtro Prolog.
"""

from __future__ import annotations

import sys
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

# =============================================================================
# Configuración
# =============================================================================

CSV_PATH        = os.getenv("CSV_PATH",        "src/data/steam_rpg_games.csv")
PARAMETERS_PATH = os.getenv("PARAMETERS_PATH", "src/data/parameters.json")
TAGS_PATH       = os.getenv("TAGS_PATH",        "src/data/tags.csv")
CSV_SEP         = "|"

# Casos de prueba para PrologEngine
PROLOG_CASES: List[Dict] = [
    {
        "name": "RPG Fantasy sin restricciones",
        "query_tags":    ["rpg", "fantasy", "magic"],
        "disliked_tags": [],
        "max_price":     0.0,   # sin límite de precio
    },
    {
        "name": "Action RPG - rechaza violencia extrema",
        "query_tags":    ["action rpg", "combat", "hack and slash"],
        "disliked_tags": ["gore", "nudity"],
        "max_price":     0.0,
    },
    {
        "name": "RPG económico (max $15)",
        "query_tags":    ["rpg", "indie", "story rich"],
        "disliked_tags": [],
        "max_price":     15.0,
    },
    {
        "name": "Juego de nicho - tags poco comunes",
        "query_tags":    ["rogue-like", "dungeon crawler", "procedural generation"],
        "disliked_tags": ["multiplayer", "online"],
        "max_price":     30.0,
    },
    {
        "name": "RPG clásico - rechaza múltiples tags",
        "query_tags":    ["turn-based", "classic rpg", "jrpg"],
        "disliked_tags": ["violent", "gore", "sexual content", "nudity"],
        "max_price":     0.0,
    },
    {
        "name": "Todos los filtros activos (permisivo)",
        "query_tags":    ["rpg"],
        "disliked_tags": [],
        "max_price":     999.0,
    },
    {
        "name": "Filtro muy restrictivo (precio bajo + dislikes)",
        "query_tags":    ["rpg", "adventure"],
        "disliked_tags": ["early access", "multiplayer", "pvp", "mmo"],
        "max_price":     10.0,
    },
]


# =============================================================================
# Helpers de parsing
# =============================================================================

def _parse_tags(value) -> Set[str]:
    if not isinstance(value, str):
        return set()
    return {t.strip().lower() for t in value.split(",") if t.strip() and t.strip() != "+"}


def _parse_year(value: str) -> Optional[int]:
    s = str(value or "").strip()
    for part in reversed(s.replace("-", " ").split()):
        if len(part) == 4 and part.isdigit():
            return int(part)
    return None


def load_rpg_tags(tags_path: str) -> Set[str]:
    """Carga los tags canónicos RPG desde el CSV de tags."""
    try:
        df = pd.read_csv(tags_path, sep="|")
        return {row.lower() for row in df["name"]}
    except Exception as e:
        logger.warning("No se pudo cargar TAGS_PATH (%s): %s. Usando conjunto vacío.", tags_path, e)
        return set()


# =============================================================================
# Ground truth determinista (réplica de las reglas del motor lógico)
# =============================================================================

def gt_is_rpg(game_tags: Set[str], rpg_tags: Set[str]) -> bool:
    return bool(game_tags & rpg_tags)


def gt_dislike_penalty(game_tags: Set[str], disliked_tags: Set[str]) -> float:
    return len(game_tags & disliked_tags) * 0.15


def gt_is_candidate(game_tags: Set[str], liked_tags: Set[str]) -> bool:
    """Vía A: el juego tiene al menos 1 tag preferido por el usuario."""
    return bool(game_tags & liked_tags)


def gt_is_recommendable(
    game_tags: Set[str],
    disliked_tags: Set[str],
    max_price: float,
    price: float,
    rpg_tags: Set[str],
) -> bool:
    if not gt_is_rpg(game_tags, rpg_tags):
        return False
    if gt_dislike_penalty(game_tags, disliked_tags) >= 0.45:
        return False
    if max_price > 0 and price > max_price:
        return False
    return True


def gt_similar_game(tags1: Set[str], tags2: Set[str]) -> bool:
    return len(tags1 & tags2) >= 2


def gt_price_tier(price: float) -> str:
    if price == 0:          return "free"
    elif price <= 10:       return "budget"
    elif price <= 30:       return "mid_range"
    elif price <= 60:       return "premium"
    else:                   return "deluxe"


def gt_rating_tier(rating: float) -> str:
    if rating >= 9.0:       return "masterpiece"
    elif rating >= 8.0:     return "excellent"
    elif rating >= 7.0:     return "good"
    elif rating >= 5.0:     return "mixed"
    else:                   return "poor"


def gt_tag_overlap_score(game_tags: Set[str], preferred_tags: Set[str]) -> float:
    if not preferred_tags:
        return 0.0
    return len(game_tags & preferred_tags) / len(preferred_tags)


# =============================================================================
# Evaluación de sub-componentes del logic_engine
# =============================================================================

def evaluate_subcomponents(df: pd.DataFrame, logic_engine) -> None:
    """
    Verifica que las funciones puras del logic_engine coincidan
    con el ground truth computado de forma independiente.
    Imprime un reporte de coincidencia (debería ser 100% en todos).
    """
    print("\n── EVALUACIÓN DE SUB-COMPONENTES (logic_engine) " + "─" * 30)

    # Tomar muestra representativa
    sample = df.sample(min(200, len(df)), random_state=42)

    # ── price_tier ────────────────────────────────────────────────────────────
    gt_pt   = [gt_price_tier(float(r.get("price") or 0))      for _, r in sample.iterrows()]
    pred_pt = [logic_engine.get_price_tier(float(r.get("price") or 0)) for _, r in sample.iterrows()]
    acc_pt  = accuracy_score(gt_pt, pred_pt)
    print(f"\n  price_tier accuracy        : {acc_pt:.4f}  (esperado: 1.0000)")

    # ── tag_overlap_score ─────────────────────────────────────────────────────
    preferred = {"rpg", "fantasy", "action"}
    errors_overlap = 0
    for _, row in sample.iterrows():
        tags = _parse_tags(str(row.get("tags", "")))
        gt_v  = round(gt_tag_overlap_score(tags, preferred), 4)
        pred_v = round(logic_engine.tag_overlap_score(tags, preferred), 4)
        if abs(gt_v - pred_v) > 1e-6:
            errors_overlap += 1
    acc_overlap = 1.0 - errors_overlap / len(sample)
    print(f"  tag_overlap_score accuracy : {acc_overlap:.4f}  (esperado: 1.0000)")

    # ── dislike_penalty ───────────────────────────────────────────────────────
    disliked = {"gore", "nudity", "violent"}
    errors_dp = 0
    for _, row in sample.iterrows():
        tags = _parse_tags(str(row.get("tags", "")))
        gt_v   = round(gt_dislike_penalty(tags, disliked), 4)
        pred_v = round(logic_engine.dislike_penalty(tags, disliked), 4)
        if abs(gt_v - pred_v) > 1e-6:
            errors_dp += 1
    acc_dp = 1.0 - errors_dp / len(sample)
    print(f"  dislike_penalty accuracy   : {acc_dp:.4f}  (esperado: 1.0000)")

    # ── similar_game ──────────────────────────────────────────────────────────
    pairs_sample = list(zip(
        sample.iloc[:len(sample)//2].iterrows(),
        sample.iloc[len(sample)//2:].iterrows(),
    ))[:50]
    errors_sim = 0
    for (_, r1), (_, r2) in pairs_sample:
        t1 = _parse_tags(str(r1.get("tags", "")))
        t2 = _parse_tags(str(r2.get("tags", "")))
        gt_v   = gt_similar_game(t1, t2)
        pred_v = logic_engine.similar_game(t1, t2)
        if gt_v != pred_v:
            errors_sim += 1
    acc_sim = 1.0 - errors_sim / max(len(pairs_sample), 1)
    print(f"  similar_game accuracy      : {acc_sim:.4f}  (esperado: 1.0000)")

    # ── rating_tier ───────────────────────────────────────────────────────────
    test_ratings = [0.0, 3.5, 5.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    errors_rt = sum(
        1 for r in test_ratings
        if gt_rating_tier(r) != logic_engine.get_rating_tier(r)
    )
    acc_rt = 1.0 - errors_rt / len(test_ratings)
    print(f"  rating_tier accuracy       : {acc_rt:.4f}  (esperado: 1.0000)")

    subcomp_scores = [acc_pt, acc_overlap, acc_dp, acc_sim, acc_rt]
    print(f"\n  Promedio sub-componentes   : {np.mean(subcomp_scores):.4f}")


# =============================================================================
# Evaluación principal del PrologEngine
# =============================================================================

@dataclass
class PrologCaseResult:
    case_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    support_pos: int
    support_neg: int
    pct_filtered_out: float    # % de candidatos que Prolog descartó
    avg_tag_overlap: float     # overlap promedio de los recomendados


def evaluate_prolog_case(
    case: Dict,
    df: pd.DataFrame,
    engine,               # PrologEngine
    knowledge_engine,     # KnowledgeEngine
    parameters_engine,    # ParametersEngine
    rpg_tags: Set[str],
) -> PrologCaseResult:
    from engine.parameters_engine import ScoreFilters, GameScore

    query_tags    = case["query_tags"]
    disliked_tags = {t.lower().strip() for t in case["disliked_tags"]}
    max_price     = case["max_price"]
    liked_tags    = {t.lower().strip() for t in query_tags}

    # Configurar PrologEngine para este caso
    engine.disliked_tags = disliked_tags
    engine.max_price     = max_price

    # Preparar GameScores sintéticos para todos los juegos del corpus
    games = knowledge_engine.games
    semantic_scores = {g.app_id: 0.5 for g in games}
    scored = parameters_engine.score(games, semantic_scores, ScoreFilters())

    games_by_id = {g.app_id: g for g in games}
    enriched = engine.filter(scored, games_by_id, query_tags)

    recommended_ids = {e.game_score.app_id for e in enriched}
    n_candidates    = len(scored)
    pct_filtered    = 1.0 - len(recommended_ids) / max(n_candidates, 1)

    # Ground truth: computar la lista de juegos realmente recomendables
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        app_id    = int(row["appid"])
        game_tags = _parse_tags(str(row.get("tags", "")))
        price     = float(row.get("price") or 0.0)

        # Ground truth
        is_candidate = gt_is_candidate(game_tags, liked_tags)
        is_rec       = gt_is_recommendable(game_tags, disliked_tags, max_price, price, rpg_tags)
        gt_label     = 1 if (is_candidate and is_rec) else 0

        # Predicción del engine
        pred_label   = 1 if app_id in recommended_ids else 0

        y_true.append(gt_label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    avg_overlap = float(np.mean([e.tag_overlap for e in enriched])) if enriched else 0.0

    return PrologCaseResult(
        case_name=case["name"],
        accuracy=round(float(acc), 4),
        precision=round(float(p), 4),
        recall=round(float(r), 4),
        f1=round(float(f1), 4),
        support_pos=int(np.sum(y_true)),
        support_neg=int(len(y_true) - np.sum(y_true)),
        pct_filtered_out=round(pct_filtered, 4),
        avg_tag_overlap=round(avg_overlap, 4),
    )


def evaluate_prolog_engine() -> Tuple[List[PrologCaseResult], object]:
    try:
        from engine.prolog_engine      import PrologEngine
        from engine.knowledge_engine   import KnowledgeEngine
        from engine.parameters_engine  import ParametersEngine
        from engine                    import logic_engine as le
    except ImportError as e:
        logger.error("Import error: %s", e)
        sys.exit(1)

    logger.info("Cargando CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
    logger.info("%d juegos cargados.", len(df))

    rpg_tags = load_rpg_tags(TAGS_PATH)
    logger.info("%d tags RPG canónicos cargados.", len(rpg_tags))

    logger.info("Construyendo KnowledgeEngine…")
    ke = KnowledgeEngine(csv_path=CSV_PATH)
    ke.build()

    from engine.parameters_engine import ParametersEngine
    pe = ParametersEngine(parameters_path=PARAMETERS_PATH)

    results: List[PrologCaseResult] = []
    for case in PROLOG_CASES:
        logger.info("Evaluando caso Prolog: %s", case["name"])
        prolog = PrologEngine(
            disliked_tags=case["disliked_tags"],
            max_price=case["max_price"],
        )
        result = evaluate_prolog_case(case, df, prolog, ke, pe, rpg_tags)
        results.append(result)

    # Evaluar sub-componentes del logic_engine
    evaluate_subcomponents(df, le)

    return results, le


# =============================================================================
# Reporte en consola
# =============================================================================

def print_report(results: List[PrologCaseResult]) -> None:
    print("\n" + "═" * 80)
    print("  EVALUACIÓN — PrologEngine / motor_logico (Paso 5)")
    print("═" * 80)

    table_data = [
        [
            r.case_name,
            r.accuracy,
            r.precision,
            r.recall,
            r.f1,
            r.support_pos,
            r.support_neg,
            f"{r.pct_filtered_out:.1%}",
            r.avg_tag_overlap,
        ]
        for r in results
    ]

    print(tabulate(
        table_data,
        headers=[
            "Caso",
            "Accuracy", "Precision", "Recall", "F1",
            "Support(+)", "Support(-)",
            "% Filtrados", "Avg Overlap"
        ],
        tablefmt="rounded_outline",
        floatfmt=".4f",
    ))

    # Promedios
    print("\n── PROMEDIOS GLOBALES " + "─" * 57)
    metrics = {
        "Accuracy" : np.mean([r.accuracy  for r in results]),
        "Precision": np.mean([r.precision for r in results]),
        "Recall"   : np.mean([r.recall    for r in results]),
        "F1"       : np.mean([r.f1        for r in results]),
    }
    for name, val in metrics.items():
        print(f"  {name:<12}: {val:.4f}")

    # Classification report consolidado
    print("\n── CLASSIFICATION REPORT CONSOLIDADO (todos los casos)")
    # Reconstruir y_true / y_pred ya no es posible aquí sin re-correr,
    # pero podemos mostrar un resumen macro con los datos ya calculados.
    avg_row = [[
        "MACRO AVG",
        round(metrics["Accuracy"],  4),
        round(metrics["Precision"], 4),
        round(metrics["Recall"],    4),
        round(metrics["F1"],        4),
        sum(r.support_pos for r in results),
        sum(r.support_neg for r in results),
    ]]
    print(tabulate(
        avg_row,
        headers=["", "Accuracy", "Precision", "Recall", "F1", "Total Support(+)", "Total Support(-)"],
        tablefmt="rounded_outline",
        floatfmt=".4f",
    ))

    # Análisis de sensibilidad al filtrado
    print("\n── ANÁLISIS DE SENSIBILIDAD (% candidatos filtrados)")
    sens_table = [
        [r.case_name, f"{r.pct_filtered_out:.1%}", r.support_pos, r.f1]
        for r in sorted(results, key=lambda x: x.pct_filtered_out, reverse=True)
    ]
    print(tabulate(
        sens_table,
        headers=["Caso", "% Filtrados", "Positivos GT", "F1"],
        tablefmt="rounded_outline",
        floatfmt=".4f",
    ))
    print("═" * 80)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    results, _ = evaluate_prolog_engine()
    print_report(results)