"""
test_parameters_engine.py – Evaluación del motor de scoring por parámetros (Paso 4).

Métricas:
    - Accuracy   : % de predicciones correctas (pasa/no-pasa filtro)
    - Precision  : de los clasificados como "pasa", ¿cuántos realmente pasan?
    - Recall     : de los que realmente pasan, ¿cuántos fueron detectados?
    - F1-score   : media armónica de Precision y Recall
    - Support    : número de juegos por clase en cada caso

Estrategia:
    Para cada ScoreFilters de prueba, se evalúa cada flag individual
    (date_flag, price_flag, positive_rate_flag, recommendations_flag)
    como un clasificador binario (0/1).
    El ground truth se computa de forma determinista aplicando
    las mismas reglas del engine directamente sobre los objetos Game.

    También se evalúa el score final agregado: si parameter_score > umbral
    → "recomendado" vs "no recomendado".
"""

from __future__ import annotations

import sys
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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


# =============================================================================
# Configuración
# =============================================================================

CSV_PATH        = os.getenv("CSV_PATH",        "src/data/steam_rpg.csv")
PARAMETERS_PATH = os.getenv("PARAMETERS_PATH", "src/data/parameters.json")
CSV_SEP         = "|"

# Umbral de parameter_score para clasificar como "recomendado" (score máx = 100)
RECOMMEND_THRESHOLD = 60

# Casos de prueba: distintas combinaciones de ScoreFilters
# Cada caso describe el escenario y los filtros a aplicar
FILTER_CASES: List[Dict] = [
    {
        "name": "Sin filtros activos",
        "filters": {
            "isPrice": False,
            "isPositiveRate": False,
            "isDate": False,
            "isRecommendations": False,
        },
        # Con todos los filtros inactivos, TODOS los juegos deben pasar (score = 100)
        "expected_all_pass": True,
    },
    {
        "name": "Solo filtro precio (0–20 USD)",
        "filters": {
            "isPrice": True,
            "MinPrice": 0.0,
            "MaxPrice": 20.0,
            "isPositiveRate": False,
            "isDate": False,
            "isRecommendations": False,
        },
        "expected_all_pass": False,
    },
    {
        "name": "Solo filtro positive rate (≥ 0.75)",
        "filters": {
            "isPrice": False,
            "isPositiveRate": True,
            "MinPositiveRate": 0.75,
            "isDate": False,
            "isRecommendations": False,
        },
        "expected_all_pass": False,
    },
    {
        "name": "Solo filtro fecha (2015–2023)",
        "filters": {
            "isPrice": False,
            "isPositiveRate": False,
            "isDate": True,
            "MinYear": 2015,
            "MaxYear": 2023,
            "isRecommendations": False,
        },
        "expected_all_pass": False,
    },
    {
        "name": "Solo filtro recomendaciones (≥ 500)",
        "filters": {
            "isPrice": False,
            "isPositiveRate": False,
            "isDate": False,
            "isRecommendations": True,
            "MinRecommendations": 500.0,
        },
        "expected_all_pass": False,
    },
    {
        "name": "Filtros combinados: precio + rating",
        "filters": {
            "isPrice": True,
            "MinPrice": 5.0,
            "MaxPrice": 40.0,
            "isPositiveRate": True,
            "MinPositiveRate": 0.70,
            "isDate": False,
            "isRecommendations": False,
        },
        "expected_all_pass": False,
    },
    {
        "name": "Filtros combinados: todos activos (estrictos)",
        "filters": {
            "isPrice": True,
            "MinPrice": 0.0,
            "MaxPrice": 15.0,
            "isPositiveRate": True,
            "MinPositiveRate": 0.80,
            "isDate": True,
            "MinYear": 2018,
            "MaxYear": 2024,
            "isRecommendations": True,
            "MinRecommendations": 1000.0,
        },
        "expected_all_pass": False,
    },
    {
        "name": "Filtros combinados: todos activos (permisivos)",
        "filters": {
            "isPrice": True,
            "MinPrice": 0.0,
            "MaxPrice": 999.0,
            "isPositiveRate": True,
            "MinPositiveRate": 0.0,
            "isDate": True,
            "MinYear": 1990,
            "MaxYear": 2030,
            "isRecommendations": True,
            "MinRecommendations": 0.0,
        },
        # Con límites tan permisivos, casi todos deben pasar
        "expected_all_pass": True,
    },
]


# =============================================================================
# Ground truth determinista
# =============================================================================

def _parse_list(value) -> list:
    if not isinstance(value, str):
        return []
    return [t.strip() for t in value.split(",") if t.strip() and t.strip() != "+"]


def _parse_year(value: str) -> Optional[int]:
    s = str(value or "").strip()
    for part in reversed(s.replace("-", " ").split()):
        if len(part) == 4 and part.isdigit():
            return int(part)
    return None


def compute_ground_truth_flags(row: pd.Series, filters: Dict) -> Dict[str, int]:
    """
    Calcula los flags ground-truth directamente desde la fila del CSV,
    replicando la lógica del ParametersEngine sin instanciarlo.
    """
    price     = float(row.get("price") or 0.0)
    pos       = float(row.get("positive_reviews") or 0)
    neg       = float(row.get("negative_reviews") or 0)
    total     = pos + neg
    pos_ratio = round(pos / total, 4) if total > 0 else 0.0
    year      = _parse_year(str(row.get("release_date") or ""))
    recs      = float(row.get("recommendations") or 0)

    # date_flag
    if not filters.get("isDate", False):
        date_flag = 1
    elif year is None:
        date_flag = 0
    else:
        date_flag = 1 if filters.get("MinYear", 0) <= year <= filters.get("MaxYear", 9999) else 0

    # price_flag
    if not filters.get("isPrice", False):
        price_flag = 1
    else:
        price_flag = 1 if filters.get("MinPrice", 0) <= price <= filters.get("MaxPrice", 999999) else 0

    # positive_rate_flag
    if not filters.get("isPositiveRate", False):
        pos_flag = 1
    else:
        pos_flag = 1 if pos_ratio >= filters.get("MinPositiveRate", 0) else 0

    # recommendations_flag
    if not filters.get("isRecommendations", False):
        rec_flag = 1
    else:
        rec_flag = 1 if recs >= filters.get("MinRecommendations", 0) else 0

    return {
        "date_flag":            date_flag,
        "price_flag":           price_flag,
        "positive_rate_flag":   pos_flag,
        "recommendations_flag": rec_flag,
    }


# =============================================================================
# Evaluación por caso
# =============================================================================

@dataclass
class FlagEvalResult:
    case_name: str
    flag_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    support_pos: int   # juegos con flag=1
    support_neg: int   # juegos con flag=0


@dataclass
class CaseEvalResult:
    case_name: str
    flag_results: List[FlagEvalResult]
    # Métricas del score agregado (parameter_score > RECOMMEND_THRESHOLD)
    score_accuracy: float
    score_precision: float
    score_recall: float
    score_f1: float
    score_support: Tuple[int, int]   # (negativos, positivos)
    # Casos esperados vs reales
    pct_passing: float   # % de juegos con parameter_score == 100


def evaluate_case(
    case: Dict,
    df: pd.DataFrame,
    engine,
    semantic_scores: Dict[int, float],
    games: list,
) -> CaseEvalResult:
    from engine.parameters_engine import ScoreFilters

    f = case["filters"]
    sf = ScoreFilters(**{k: v for k, v in f.items() if hasattr(ScoreFilters, k) or True})
    # Construir ScoreFilters dinámicamente
    sf = ScoreFilters()
    for attr, val in f.items():
        if hasattr(sf, attr):
            setattr(sf, attr, val)

    # Obtener predicciones del engine
    scored = engine.score(games, semantic_scores, sf)
    pred_by_id = {s.app_id: s for s in scored}

    # Ground truth y predicciones por flag
    flag_names = ["date_flag", "price_flag", "positive_rate_flag", "recommendations_flag"]
    gt_flags   = {fn: [] for fn in flag_names}
    pred_flags = {fn: [] for fn in flag_names}
    gt_score_label  = []
    pred_score_label = []

    w = [10, 40, 30, 20]  # pesos por defecto

    for _, row in df.iterrows():
        app_id = int(row["appid"])
        if app_id not in pred_by_id:
            continue

        gs  = pred_by_id[app_id]
        gt  = compute_ground_truth_flags(row, f)

        for fn in flag_names:
            gt_flags[fn].append(gt[fn])
            pred_flags[fn].append(getattr(gs, fn))

        # Score ground truth (misma fórmula)
        gt_score = (
            gt["date_flag"]            * w[0] +
            gt["price_flag"]           * w[1] +
            gt["positive_rate_flag"]   * w[2] +
            gt["recommendations_flag"] * w[3]
        )
        gt_score_label.append(1 if gt_score >= RECOMMEND_THRESHOLD else 0)
        pred_score_label.append(1 if gs.parameter_score >= RECOMMEND_THRESHOLD else 0)

    flag_results: List[FlagEvalResult] = []
    for fn in flag_names:
        yt = gt_flags[fn]
        yp = pred_flags[fn]
        if not yt:
            continue
        acc = accuracy_score(yt, yp)
        p, r, f1, sup = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        flag_results.append(FlagEvalResult(
            case_name=case["name"],
            flag_name=fn,
            accuracy=round(acc, 4),
            precision=round(float(p), 4),
            recall=round(float(r), 4),
            f1=round(float(f1), 4),
            support_pos=int(np.sum(yt)),
            support_neg=int(len(yt) - np.sum(yt)),
        ))

    # Score agregado
    s_acc  = accuracy_score(gt_score_label, pred_score_label) if gt_score_label else 0.0
    s_p, s_r, s_f1, s_sup = precision_recall_fscore_support(
        gt_score_label, pred_score_label,
        average="binary", zero_division=0
    ) if gt_score_label else (0, 0, 0, None)

    pct_pass = np.mean([1 if s.parameter_score == 100 else 0 for s in scored])

    return CaseEvalResult(
        case_name=case["name"],
        flag_results=flag_results,
        score_accuracy=round(float(s_acc), 4),
        score_precision=round(float(s_p), 4),
        score_recall=round(float(s_r), 4),
        score_f1=round(float(s_f1), 4),
        score_support=(
            int(np.sum([1 - x for x in gt_score_label])),
            int(np.sum(gt_score_label)),
        ),
        pct_passing=round(float(pct_pass), 4),
    )


# =============================================================================
# Evaluador principal
# =============================================================================

def evaluate_parameters_engine() -> List[CaseEvalResult]:
    try:
        from engine.parameters_engine import ParametersEngine, ScoreFilters
        from engine.knowledge_engine  import KnowledgeEngine
    except ImportError as e:
        logger.error("Import error: %s", e)
        sys.exit(1)

    logger.info("Cargando CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
    logger.info("%d juegos cargados.", len(df))

    logger.info("Construyendo KnowledgeEngine…")
    ke = KnowledgeEngine(csv_path=CSV_PATH)
    ke.build()
    games = ke.games

    # Semantic scores falsos pero uniformes (para aislar la evaluación del ParametersEngine)
    semantic_scores = {g.app_id: 0.5 for g in games}

    logger.info("Instanciando ParametersEngine…")
    engine = ParametersEngine(parameters_path=PARAMETERS_PATH)

    case_results: List[CaseEvalResult] = []
    for case in FILTER_CASES:
        logger.info("Evaluando caso: %s", case["name"])
        result = evaluate_case(case, df, engine, semantic_scores, games)
        case_results.append(result)

    return case_results


# =============================================================================
# Reporte en consola
# =============================================================================

def print_report(case_results: List[CaseEvalResult]) -> None:
    print("\n" + "═" * 80)
    print("  EVALUACIÓN — ParametersEngine (Paso 4)")
    print("═" * 80)

    for cr in case_results:
        print(f"\n▶ Caso: {cr.case_name}")
        print(f"   % juegos con score=100 : {cr.pct_passing:.1%}")

        # Tabla de flags
        flag_table = [
            [fr.flag_name, fr.accuracy, fr.precision, fr.recall, fr.f1,
             fr.support_pos, fr.support_neg]
            for fr in cr.flag_results
        ]
        print(tabulate(
            flag_table,
            headers=["Flag", "Accuracy", "Precision", "Recall", "F1", "Support(1)", "Support(0)"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        ))

        # Score agregado
        print(f"\n   Score agregado (umbral ≥ {RECOMMEND_THRESHOLD}):")
        agg_table = [[
            cr.score_accuracy,
            cr.score_precision,
            cr.score_recall,
            cr.score_f1,
            cr.score_support[0],
            cr.score_support[1],
        ]]
        print(tabulate(
            agg_table,
            headers=["Accuracy", "Precision", "Recall", "F1", "Support(0)", "Support(1)"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        ))

    # Resumen global
    print("\n── RESUMEN GLOBAL (promedio entre casos) " + "─" * 38)
    global_table = []
    for flag_name in ["date_flag", "price_flag", "positive_rate_flag", "recommendations_flag"]:
        all_flags = [fr for cr in case_results for fr in cr.flag_results if fr.flag_name == flag_name]
        if not all_flags:
            continue
        global_table.append([
            flag_name,
            round(np.mean([f.accuracy  for f in all_flags]), 4),
            round(np.mean([f.precision for f in all_flags]), 4),
            round(np.mean([f.recall    for f in all_flags]), 4),
            round(np.mean([f.f1        for f in all_flags]), 4),
            sum(f.support_pos for f in all_flags),
        ])
    print(tabulate(
        global_table,
        headers=["Flag", "Avg Accuracy", "Avg Precision", "Avg Recall", "Avg F1", "Total Support"],
        tablefmt="rounded_outline",
        floatfmt=".4f",
    ))

    # Classification report global del score agregado
    print(f"\n── CLASSIFICATION REPORT GLOBAL (score agregado, umbral={RECOMMEND_THRESHOLD})")
    print("   (promedio ponderado de todos los casos)")
    accs  = [cr.score_accuracy  for cr in case_results]
    precs = [cr.score_precision for cr in case_results]
    recs  = [cr.score_recall    for cr in case_results]
    f1s   = [cr.score_f1        for cr in case_results]
    print(f"   Accuracy  : {np.mean(accs):.4f}")
    print(f"   Precision : {np.mean(precs):.4f}")
    print(f"   Recall    : {np.mean(recs):.4f}")
    print(f"   F1        : {np.mean(f1s):.4f}")
    print("═" * 80)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    case_results = evaluate_parameters_engine()
    print_report(case_results)