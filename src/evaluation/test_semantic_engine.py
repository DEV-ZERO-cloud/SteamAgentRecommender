"""
test_semantic_engine.py – Evaluación del motor semántico (Paso 2 del pipeline).

Métricas:
    - Precision@K : de los K resultados retornados, ¿cuántos son relevantes?
    - Recall@K    : de todos los relevantes, ¿cuántos fueron recuperados en K?
    - F1@K        : media armónica de Precision@K y Recall@K
    - Accuracy@K  : ¿al menos 1 resultado relevante entre los K primeros?
    - Support     : número de casos de prueba por categoría

Estrategia de ground truth:
    Se define un conjunto de casos (query_tags → app_ids relevantes conocidos).
    La relevancia se determina por solapamiento de tags: un juego es relevante
    si comparte al menos `MIN_TAG_OVERLAP` tags con la query (umbral configurable).
    Esto permite generar ground truth automáticamente desde el CSV sin anotación manual.
"""

from __future__ import annotations

import sys
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

# ── Ajusta el path según tu estructura de proyecto ───────────────────────────
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuración
# =============================================================================

CSV_PATH        = os.getenv("CSV_PATH", "src/data/steam_rpg.csv")
CSV_SEP         = "|"
ID_COL          = "appid"
TAG_COL         = "tags"
GENRE_COL       = "genres"
TOP_K_VALUES    = [5, 10, 20]          # evaluar para distintos K
MIN_TAG_OVERLAP = 2                    # tags en común para considerar relevante
MIN_SCORE       = 0.05

# Casos de prueba: (nombre_del_caso, tags_query, set_de_tags_esperados_en_resultados)
# La columna "expected_tags" son tags que DEBEN aparecer en los juegos recuperados
# para considerarlos relevantes (ground truth débil pero reproducible).
TEST_CASES: List[Dict] = [
    {
        "name": "Fantasy RPG",
        "tags": ["fantasy", "rpg", "magic"],
        "required_tags": {"fantasy", "rpg"},
    },
    {
        "name": "Dark Souls-like",
        "tags": ["souls-like", "dark fantasy", "difficult"],
        "required_tags": {"souls-like", "action rpg"},
    },
    {
        "name": "Open World Adventure",
        "tags": ["open world", "adventure", "exploration"],
        "required_tags": {"open world", "adventure"},
    },
    {
        "name": "Turn-Based Strategy",
        "tags": ["turn-based", "strategy", "tactical"],
        "required_tags": {"turn-based", "strategy"},
    },
    {
        "name": "Indie RPG",
        "tags": ["indie", "rpg", "pixel graphics"],
        "required_tags": {"indie", "rpg"},
    },
    {
        "name": "Multiplayer Action",
        "tags": ["multiplayer", "action", "co-op"],
        "required_tags": {"multiplayer", "action"},
    },
    {
        "name": "Story Rich",
        "tags": ["story rich", "choices matter", "narrative"],
        "required_tags": {"story rich"},
    },
    {
        "name": "Horror RPG",
        "tags": ["horror", "dark", "psychological"],
        "required_tags": {"horror"},
    },
]


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class CaseResult:
    name: str
    k: int
    retrieved_ids: List[int]
    relevant_ids: Set[int]
    precision: float
    recall: float
    f1: float
    accuracy: float          # 1 si hay al menos 1 relevante en top-K
    support: int             # |relevant_ids| en el corpus

@dataclass
class EvalReport:
    results: List[CaseResult] = field(default_factory=list)

    def avg_precision(self, k: int) -> float:
        vals = [r.precision for r in self.results if r.k == k]
        return float(np.mean(vals)) if vals else 0.0

    def avg_recall(self, k: int) -> float:
        vals = [r.recall for r in self.results if r.k == k]
        return float(np.mean(vals)) if vals else 0.0

    def avg_f1(self, k: int) -> float:
        vals = [r.f1 for r in self.results if r.k == k]
        return float(np.mean(vals)) if vals else 0.0

    def avg_accuracy(self, k: int) -> float:
        vals = [r.accuracy for r in self.results if r.k == k]
        return float(np.mean(vals)) if vals else 0.0


# =============================================================================
# Helpers de ground truth
# =============================================================================

def _parse_tags(value) -> Set[str]:
    if not isinstance(value, str):
        return set()
    return {t.strip().lower() for t in value.split(",") if t.strip() and t.strip() != "+"}


def build_ground_truth(df: pd.DataFrame, required_tags: Set[str]) -> Set[int]:
    """
    Un juego es relevante si comparte al menos MIN_TAG_OVERLAP tags
    con required_tags (ground truth automático desde el CSV).
    """
    relevant: Set[int] = set()
    for _, row in df.iterrows():
        game_tags = _parse_tags(str(row.get(TAG_COL, "")))
        game_tags |= _parse_tags(str(row.get(GENRE_COL, "")))
        if len(game_tags & required_tags) >= MIN_TAG_OVERLAP:
            relevant.add(int(row[ID_COL]))
    return relevant


# =============================================================================
# Métricas
# =============================================================================

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    top_k = set(retrieved[:k])
    if not top_k:
        return 0.0
    return len(top_k & relevant) / k


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def f1_at_k(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def accuracy_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


# =============================================================================
# Evaluador principal
# =============================================================================

def evaluate_semantic_engine() -> EvalReport:
    """Carga el SemanticEngine, corre los casos de prueba y compila el reporte."""

    # ── Importación lazy para no romper si las dependencias no están instaladas
    try:
        from engine.semantic_engine import SemanticEngine, SemanticEngineConfig
        from scripts.prepare_data import PrepareData
    except ImportError as e:
        logger.error("No se pudo importar SemanticEngine: %s", e)
        logger.info("Asegúrate de ejecutar desde la raíz del proyecto con el env activo.")
        sys.exit(1)

    logger.info("Cargando CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
    logger.info("CSV cargado: %d juegos.", len(df))

    logger.info("Construyendo SemanticEngine…")
    config   = SemanticEngineConfig(
        csv_path=CSV_PATH,
        separator=CSV_SEP,
        id_column=ID_COL,
        tag_column=TAG_COL,
        genre_column=GENRE_COL,
        default_top_k=max(TOP_K_VALUES),
        default_min_score=MIN_SCORE,
    )
    prepared = PrepareData(config).prepare()
    engine   = SemanticEngine(config, prepared)

    report = EvalReport()

    for case in TEST_CASES:
        name          = case["name"]
        tags          = case["tags"]
        required_tags = {t.lower() for t in case["required_tags"]}

        # Ground truth automático
        relevant_ids = build_ground_truth(df, required_tags)
        if not relevant_ids:
            logger.warning("[%s] Sin juegos relevantes en corpus (ajusta MIN_TAG_OVERLAP).", name)

        # Búsqueda
        results = engine.search(tags=tags, top_k=max(TOP_K_VALUES), min_score=MIN_SCORE)
        retrieved_ids = [game_id for game_id, _ in results]

        for k in TOP_K_VALUES:
            p   = precision_at_k(retrieved_ids, relevant_ids, k)
            r   = recall_at_k(retrieved_ids, relevant_ids, k)
            f1  = f1_at_k(p, r)
            acc = accuracy_at_k(retrieved_ids, relevant_ids, k)

            report.results.append(CaseResult(
                name=name,
                k=k,
                retrieved_ids=retrieved_ids[:k],
                relevant_ids=relevant_ids,
                precision=round(p, 4),
                recall=round(r, 4),
                f1=round(f1, 4),
                accuracy=acc,
                support=len(relevant_ids),
            ))

    return report


# =============================================================================
# Reporte en consola
# =============================================================================

def print_report(report: EvalReport) -> None:
    print("\n" + "═" * 72)
    print("  EVALUACIÓN — SemanticEngine (Paso 2)")
    print("═" * 72)

    for k in TOP_K_VALUES:
        rows = [r for r in report.results if r.k == k]
        table_data = [
            [r.name, r.precision, r.recall, r.f1, int(r.accuracy), r.support]
            for r in rows
        ]
        print(f"\n── @K={k} {'─' * 55}")
        print(tabulate(
            table_data,
            headers=["Caso", "Precision", "Recall", "F1", "Accuracy", "Support"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        ))
        print(f"\n  Promedio @K={k}:")
        print(f"    Precision : {report.avg_precision(k):.4f}")
        print(f"    Recall    : {report.avg_recall(k):.4f}")
        print(f"    F1        : {report.avg_f1(k):.4f}")
        print(f"    Accuracy  : {report.avg_accuracy(k):.4f}")

    # ── Tabla resumen de promedios
    print("\n── RESUMEN DE PROMEDIOS {'─' * 44}")
    summary = [
        [
            f"@K={k}",
            report.avg_precision(k),
            report.avg_recall(k),
            report.avg_f1(k),
            report.avg_accuracy(k),
        ]
        for k in TOP_K_VALUES
    ]
    print(tabulate(
        summary,
        headers=["K", "Avg Precision", "Avg Recall", "Avg F1", "Avg Accuracy"],
        tablefmt="rounded_outline",
        floatfmt=".4f",
    ))

    # ── Classification report binario por caso (relevante vs no relevante)
    print("\n── CLASSIFICATION REPORT (binario: relevante vs no relevante) @K=10")
    y_true_all, y_pred_all = [], []
    for r in report.results:
        if r.k != 10:
            continue
        # Para cada ID recuperado: ¿es relevante?
        for rid in r.retrieved_ids:
            y_pred_all.append(1)
            y_true_all.append(1 if rid in r.relevant_ids else 0)
        # Falsos negativos (relevantes no recuperados)
        missed = r.relevant_ids - set(r.retrieved_ids)
        for _ in missed:
            y_true_all.append(1)
            y_pred_all.append(0)

    if y_true_all:
        print(classification_report(
            y_true_all, y_pred_all,
            target_names=["No relevante", "Relevante"],
            zero_division=0,
        ))

    print("═" * 72)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    report = evaluate_semantic_engine()
    print_report(report)