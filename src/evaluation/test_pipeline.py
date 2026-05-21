"""
test_pipeline_offline.py – Evaluación offline con datos reales de Steam.

Problema raíz corregido en esta versión:
    El catálogo (446 juegos) es todo RPGs, igual que best_rpgs (58 juegos).
    Con tags compartidos masivamente (rpg=86%, singleplayer=76%, adventure=69%),
    un centroide TF-IDF simple no discrimina.

Solución — Ground truth por similitud TF-IDF ponderada con tres mejoras:

    1. Preservación de frases multi-palabra:
       "story rich" → "story_rich" (un solo token, no dos)
       Evita que "story" y "rich" pesen por separado e incorrectamente.

    2. Penalización de tags ultra-genéricos (max_df):
       Tags que aparecen en >MAX_DF_BEST de los juegos best_rpgs
       se excluyen del vocabulario del centroide.
       Esto evita que 'rpg' (86%) domine el vector y diluya tags distintivos.

    3. Umbral dinámico por percentil (no valor fijo):
       SIM_THRESHOLD_PERCENTILE = 40 → los juegos en el top-40%
       de similitud con best_rpgs se consideran relevantes fuertes.
       Esto se auto-calibra al tamaño y distribución de cualquier catálogo.

Graded Relevance 0-3:
    Grado 3 → sim >= p(SIM_STRONG_PCT)  AND  (jugado OR tag preferido)
    Grado 2 → sim >= p(SIM_STRONG_PCT)  OR   jugado por el usuario
    Grado 1 → sim >= p(SIM_WEAK_PCT)    AND  tag preferido del usuario
    Grado 0 → no relevante

Métricas:
    Precision@K · Recall@K · F1@K · Hit-Rate@K   (K ∈ {3,5,10})
    MRR · NDCG@K · Coverage · Personalization · Latencia
"""

from __future__ import annotations

import ast
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuración
# =============================================================================

CSV_PATH        = os.getenv("CSV_PATH",        "src/data/steam_rpg_games.csv")
PARAMETERS_PATH = os.getenv("PARAMETERS_PATH", "src/data/parameters.json")
BEST_GAMES_PATH = os.getenv("BEST_GAMES_PATH", "src/data/best_rpgs_steam.csv")
USERS_CSV_PATH  = os.getenv("USERS_CSV_PATH",  "src/data/steam_users.csv")

SEP = "|"

TOP_K_VALUES   = [3, 5, 10]
PIPELINE_TOP_K = 10
MAX_USERS      = None

# ── Parámetros del ground truth ponderado ────────────────────────────────────
# Percentil sobre las similitudes del catálogo vs centroide best_rpgs:
#   SIM_STRONG_PCT=40 → top 40% del catálogo = relevantes fuertes (grado 2/3)
#   SIM_WEAK_PCT=20   → siguiente 20% = relevantes débiles (grado 1)
# Ajusta estos valores para más/menos discriminación.
SIM_STRONG_PCT = 40     # % de juegos del catálogo que serán grado 2/3
SIM_WEAK_PCT   = 20     # % adicional que será grado 1

# TF-IDF: excluir tags que aparecen en más del MAX_DF_BEST de best_rpgs
# (evita que 'rpg', 'singleplayer' dominen el centroide)
MAX_DF_BEST = 0.80      # ignorar tags presentes en >80% de best_rpgs
MIN_DF_BEST = 2         # ignorar tags que aparecen en <2 juegos de best_rpgs

# Popularidad: positive_ratio mínimo para señal débil
POP_MIN_RATIO = 0.80


# =============================================================================
# Parsing
# =============================================================================

def _parse_list_str(value) -> list:
    if pd.isna(value) or str(value).strip() == "":
        return []
    try:
        r = ast.literal_eval(str(value).strip())
        return r if isinstance(r, list) else []
    except (ValueError, SyntaxError):
        s = str(value).strip().strip("[]")
        return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]


def parse_ids(value) -> Set[int]:
    result = set()
    for x in _parse_list_str(value):
        try:
            result.add(int(x))
        except (ValueError, TypeError):
            pass
    return result


def parse_pref_tags(value) -> Set[str]:
    return {str(t).strip().lower() for t in _parse_list_str(value) if str(t).strip()}


def row_to_phrase_doc(row: pd.Series) -> str:
    """
    Convierte tags y géneros de una fila en un documento de frases.
    Reemplaza espacios dentro de cada tag por '_' para preservar
    frases multi-palabra como tokens únicos.

    "story rich, open world" → "story_rich open_world"
    """
    parts = []
    for field in ["tags", "genres"]:
        val = str(row.get(field, "") or "")
        for tag in val.split(","):
            tag = tag.strip().lower()
            if tag and tag != "+" and tag != "nan":
                parts.append(tag.replace(" ", "_"))
    return " ".join(parts)


# =============================================================================
# Carga de datos
# =============================================================================

def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=SEP)
    logger.info("Catálogo: %d juegos", len(df))
    return df


def load_best_games() -> pd.DataFrame:
    df = pd.read_csv(BEST_GAMES_PATH, sep=SEP)
    if "app_id" in df.columns and "appid" not in df.columns:
        df = df.rename(columns={"app_id": "appid"})
    logger.info("Best-games curados: %d juegos", len(df))
    return df


def load_users() -> pd.DataFrame:
    df = pd.read_csv(USERS_CSV_PATH, sep=SEP)
    if MAX_USERS:
        df = df.head(MAX_USERS)
    logger.info("Usuarios: %d", len(df))
    return df


# =============================================================================
# Ground truth por similitud TF-IDF ponderada con umbral dinámico
# =============================================================================

def build_similarity_scores(
    catalog_df:    pd.DataFrame,
    best_games_df: pd.DataFrame,
) -> Tuple[Dict[int, float], float, float]:
    """
    Calcula la similitud coseno de cada juego del catálogo contra el
    centroide TF-IDF de best_rpgs, usando:
        - Frases preservadas (story_rich, open_world, action_rpg...)
        - sublinear_tf=True: reduce el peso de tags muy frecuentes
        - max_df=MAX_DF_BEST: excluye tags ultra-genéricos del vocabulario

    Retorna:
        sim_scores  → {appid: similarity}
        thresh_strong → umbral del percentil SIM_STRONG_PCT sobre el catálogo
        thresh_weak   → umbral del percentil (100 - SIM_WEAK_PCT) sobre el catálogo
    """
    cat_ids   = catalog_df["appid"].astype(int).tolist()
    cat_docs  = [row_to_phrase_doc(r) for _, r in catalog_df.iterrows()]
    best_docs = [row_to_phrase_doc(r) for _, r in best_games_df.iterrows()]

    # Ajustar vocabulario sobre best_rpgs para que el centroide sea informativo
    vec_best = TfidfVectorizer(
        sublinear_tf=True,
        max_df=MAX_DF_BEST,
        min_df=MIN_DF_BEST,
    )
    best_matrix = vec_best.fit_transform(best_docs)
    centroid    = np.asarray(best_matrix.mean(axis=0))   # (1, vocab)

    # Proyectar catálogo al mismo espacio vectorial
    cat_matrix = vec_best.transform(cat_docs)
    sims       = cosine_similarity(cat_matrix, centroid).flatten()

    sim_scores = {aid: float(s) for aid, s in zip(cat_ids, sims)}

    # Umbrales dinámicos basados en la distribución real del catálogo
    sim_values    = np.array(list(sim_scores.values()))
    thresh_strong = float(np.percentile(sim_values, 100 - SIM_STRONG_PCT))
    thresh_weak   = float(np.percentile(sim_values, 100 - SIM_STRONG_PCT - SIM_WEAK_PCT))

    n_strong = int((sim_values >= thresh_strong).sum())
    n_weak   = int(((sim_values >= thresh_weak) & (sim_values < thresh_strong)).sum())
    n_zero   = int((sim_values < thresh_weak).sum())

    logger.info(
        "Ground truth TF-IDF ponderado:  "
        "grado 2/3 (sim≥%.3f, top %d%%): %d juegos  |  "
        "grado 1 (sim≥%.3f): %d juegos  |  grado 0: %d juegos",
        thresh_strong, SIM_STRONG_PCT, n_strong,
        thresh_weak, n_weak, n_zero,
    )

    # Tags más informativos del centroide (para interpretabilidad)
    fn        = vec_best.get_feature_names_out()
    cf        = centroid.flatten()
    top_terms = [(fn[i].replace("_", " "), round(float(cf[i]), 4))
                 for i in cf.argsort()[::-1][:8]]
    logger.info(
        "Tags más informativos del centroide: %s",
        ", ".join(f"{t}({w})" for t, w in top_terms),
    )

    return sim_scores, thresh_strong, thresh_weak


def build_popularity_scores(catalog_df: pd.DataFrame) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for _, row in catalog_df.iterrows():
        pos   = float(row.get("positive_reviews") or 0)
        neg   = float(row.get("negative_reviews") or 0)
        total = pos + neg
        scores[int(row["appid"])] = pos / total if total > 0 else 0.0
    pop_count = sum(1 for v in scores.values() if v >= POP_MIN_RATIO)
    logger.info("Juegos con positive_ratio >= %.0f%%: %d", POP_MIN_RATIO * 100, pop_count)
    return scores


# =============================================================================
# Graded Relevance 0-3
# =============================================================================

def grade_game(
    app_id:         int,
    sim_scores:     Dict[int, float],
    thresh_strong:  float,
    thresh_weak:    float,
    pop_scores:     Dict[int, float],
    played_ids:     Set[int],
    preferred_tags: Set[str],
    catalog_tags:   Dict[int, Set[str]],
) -> int:
    sim       = sim_scores.get(app_id, 0.0)
    is_strong = sim >= thresh_strong
    is_weak   = thresh_weak <= sim < thresh_strong
    is_played = app_id in played_ids
    tags      = catalog_tags.get(app_id, set())
    tag_match = bool(tags & preferred_tags)
    is_pop    = pop_scores.get(app_id, 0.0) >= POP_MIN_RATIO

    if is_strong and (is_played or tag_match):
        return 3
    if is_strong or is_played:
        return 2
    if (is_weak or is_pop) and tag_match:
        return 1
    return 0


def build_relevance_map(
    all_ids:        List[int],
    sim_scores:     Dict[int, float],
    thresh_strong:  float,
    thresh_weak:    float,
    pop_scores:     Dict[int, float],
    played_ids:     Set[int],
    preferred_tags: Set[str],
    catalog_tags:   Dict[int, Set[str]],
    min_grade:      int = 1,
) -> Tuple[Dict[int, int], Set[int]]:
    grades = {
        aid: grade_game(
            aid, sim_scores, thresh_strong, thresh_weak,
            pop_scores, played_ids, preferred_tags, catalog_tags,
        )
        for aid in all_ids
    }
    relevant = {aid for aid, g in grades.items() if g >= min_grade}
    return grades, relevant


def build_catalog_tags_index(df: pd.DataFrame) -> Dict[int, Set[str]]:
    return {
        int(row["appid"]): {
            t.strip().lower()
            for f in [str(row.get("tags", "")), str(row.get("genres", ""))]
            for t in f.split(",")
            if t.strip() and t.strip() != "+"
        }
        for _, row in df.iterrows()
    }


# =============================================================================
# Métricas
# =============================================================================

def precision_at_k(ret, rel, k):
    return len(set(ret[:k]) & rel) / k if ret[:k] else 0.0

def recall_at_k(ret, rel, k):
    return len(set(ret[:k]) & rel) / len(rel) if rel else 0.0

def f1_at_k(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def hit_rate_at_k(ret, rel, k):
    return float(bool(set(ret[:k]) & rel))

def mrr(ret, rel):
    for i, rid in enumerate(ret, 1):
        if rid in rel:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ret, grades, k):
    dcg   = sum(grades.get(rid, 0) / np.log2(i + 2) for i, rid in enumerate(ret[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg  = sum(g / np.log2(i + 2) for i, g in enumerate(ideal) if g > 0)
    return dcg / idcg if idcg > 0 else 0.0

def personalization_score(lists):
    if len(lists) < 2:
        return 0.0
    overlaps = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            s1, s2 = set(lists[i]), set(lists[j])
            u = s1 | s2
            if u:
                overlaps.append(len(s1 & s2) / len(u))
    return round(1.0 - float(np.mean(overlaps)), 4) if overlaps else 0.0

def coverage_score(retrieved, catalog):
    return round(len(retrieved & catalog) / len(catalog), 4) if catalog else 0.0


# =============================================================================
# Estructuras de resultado
# =============================================================================

@dataclass
class UserResult:
    user_id:        str
    user_name:      str
    query_tags:     List[str]
    retrieved_ids:  List[int]
    relevant_ids:   Set[int]
    grade_map:      Dict[int, int]
    latency_ms:     float
    metrics_by_k:   Dict[int, Dict[str, float]] = field(default_factory=dict)
    mrr_score:      float = 0.0
    ndcg_by_k:      Dict[int, float]            = field(default_factory=dict)
    n_strong_ret:   int   = 0
    n_played_ret:   int   = 0
    avg_grade:      float = 0.0
    avg_sim_ret:    float = 0.0   # similitud promedio de los retornados

@dataclass
class EvalReport:
    user_results:    List[UserResult] = field(default_factory=list)
    personalization: float = 0.0
    coverage:        float = 0.0
    avg_latency_ms:  float = 0.0
    thresh_strong:   float = 0.0
    thresh_weak:     float = 0.0


# =============================================================================
# Evaluación por usuario
# =============================================================================

def evaluate_user(
    user_row:      pd.Series,
    pipeline,
    all_ids:       List[int],
    catalog_tags:  Dict[int, Set[str]],
    sim_scores:    Dict[int, float],
    thresh_strong: float,
    thresh_weak:   float,
    pop_scores:    Dict[int, float],
) -> UserResult:

    user_id        = str(user_row["user_id"])
    user_name      = str(user_row.get("name", user_id))
    played_ids     = parse_ids(user_row.get("played_app_ids", ""))
    preferred_tags = parse_pref_tags(user_row.get("preferred_tags", ""))

    query = ", ".join(preferred_tags) if preferred_tags else "rpg"

    t0 = time.perf_counter()
    try:
        enriched = pipeline.recommend(
            query=query,
            top_k=PIPELINE_TOP_K,
            filters=None,
            disliked_tags=None,
            max_price=0.0,
        )
    except Exception as e:
        logger.error("[%s] pipeline.recommend falló: %s", user_name, e)
        enriched = []
    latency = (time.perf_counter() - t0) * 1000

    retrieved_ids = [e.game_score.app_id for e in enriched]

    grades, relevant_ids = build_relevance_map(
        all_ids, sim_scores, thresh_strong, thresh_weak,
        pop_scores, played_ids, preferred_tags, catalog_tags, min_grade=1,
    )

    metrics_by_k: Dict[int, Dict[str, float]] = {}
    ndcg_by_k:    Dict[int, float]            = {}
    for k in TOP_K_VALUES:
        p   = precision_at_k(retrieved_ids, relevant_ids, k)
        r   = recall_at_k(retrieved_ids, relevant_ids, k)
        f1  = f1_at_k(p, r)
        acc = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        metrics_by_k[k] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(acc, 4),
        }
        ndcg_by_k[k] = round(ndcg_at_k(retrieved_ids, grades, k), 4)

    mrr_score  = round(mrr(retrieved_ids, relevant_ids), 4)
    avg_grade  = round(float(np.mean(
        [grades.get(rid, 0) for rid in retrieved_ids]
    )), 4) if retrieved_ids else 0.0
    avg_sim    = round(float(np.mean(
        [sim_scores.get(rid, 0) for rid in retrieved_ids]
    )), 4) if retrieved_ids else 0.0
    n_strong   = sum(1 for rid in retrieved_ids if sim_scores.get(rid, 0) >= thresh_strong)
    n_played   = len(set(retrieved_ids) & played_ids)

    logger.info(
        "  %-22s | ret=%2d | GT_rel=%3d (%2.0f%%) | "
        "MRR=%.3f | P@5=%.3f | F1@5=%.3f | NDCG@5=%.3f | avg_sim=%.3f | %.0fms",
        user_name[:22], len(retrieved_ids), len(relevant_ids),
        100 * len(relevant_ids) / max(len(all_ids), 1),
        mrr_score,
        metrics_by_k.get(5, {}).get("precision", 0),
        metrics_by_k.get(5, {}).get("f1", 0),
        ndcg_by_k.get(5, 0),
        avg_sim, latency,
    )

    return UserResult(
        user_id=user_id,
        user_name=user_name,
        query_tags=list(preferred_tags),
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        grade_map=grades,
        latency_ms=round(latency, 1),
        metrics_by_k=metrics_by_k,
        mrr_score=mrr_score,
        ndcg_by_k=ndcg_by_k,
        n_strong_ret=n_strong,
        n_played_ret=n_played,
        avg_grade=avg_grade,
        avg_sim_ret=avg_sim,
    )


# =============================================================================
# Evaluador principal
# =============================================================================

def run_evaluation() -> EvalReport:
    try:
        from pipeline.pipeline_recommendation import PipelineRecommendation
    except ImportError as e:
        logger.error("No se pudo importar PipelineRecommendation: %s", e)
        sys.exit(1)

    catalog_df    = load_catalog()
    best_games_df = load_best_games()
    users_df      = load_users()

    all_ids      = catalog_df["appid"].astype(int).tolist()
    catalog_set  = set(all_ids)
    catalog_tags = build_catalog_tags_index(catalog_df)

    sim_scores, thresh_strong, thresh_weak = build_similarity_scores(
        catalog_df, best_games_df
    )
    pop_scores = build_popularity_scores(catalog_df)

    logger.info("Inicializando PipelineRecommendation…")
    pipeline = PipelineRecommendation()

    report = EvalReport(thresh_strong=thresh_strong, thresh_weak=thresh_weak)
    for _, user_row in users_df.iterrows():
        result = evaluate_user(
            user_row, pipeline, all_ids, catalog_tags,
            sim_scores, thresh_strong, thresh_weak, pop_scores,
        )
        report.user_results.append(result)

    retrieved_lists = [r.retrieved_ids for r in report.user_results]
    retrieved_union = {rid for lst in retrieved_lists for rid in lst}
    report.personalization = personalization_score(retrieved_lists)
    report.coverage        = coverage_score(retrieved_union, catalog_set)
    report.avg_latency_ms  = round(float(np.mean(
        [r.latency_ms for r in report.user_results]
    )), 1)
    return report


# =============================================================================
# Reporte
# =============================================================================

def _avg(results, k, metric):
    return float(np.mean([r.metrics_by_k.get(k, {}).get(metric, 0) for r in results]))


def print_report(report: EvalReport) -> None:
    results = report.user_results
    if not results:
        print("Sin resultados.")
        return

    LINE = "═" * 102
    ts, tw = report.thresh_strong, report.thresh_weak

    print(f"\n{LINE}")
    print("  EVALUACIÓN OFFLINE — Pipeline con Ground Truth Real (Steam)")
    print(f"  Ground truth: TF-IDF ponderado (frases + max_df={MAX_DF_BEST}) | "
          f"umbrales dinámicos: fuerte≥{ts:.3f} / débil≥{tw:.3f}")
    print(f"  Parámetros: top {SIM_STRONG_PCT}% catálogo = grado 2/3 | "
          f"siguiente {SIM_WEAK_PCT}% = grado 1")
    print(LINE)

    for k in TOP_K_VALUES:
        print(f"\n── Métricas por usuario @K={k} {'─' * 64}")
        table = [
            [
                r.user_name[:26],
                r.metrics_by_k.get(k, {}).get("precision", 0),
                r.metrics_by_k.get(k, {}).get("recall",    0),
                r.metrics_by_k.get(k, {}).get("f1",        0),
                r.metrics_by_k.get(k, {}).get("accuracy",  0),
                r.ndcg_by_k.get(k, 0),
                len(r.relevant_ids),
                r.avg_grade,
            ]
            for r in results
        ]
        print(tabulate(
            table,
            headers=["Usuario", "Precision", "Recall", "F1", "Hit-Rate",
                     f"NDCG@{k}", "Support", "Avg Grade"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        ))
        print(
            f"\n  Promedios @K={k}:  "
            f"P={_avg(results,k,'precision'):.4f}  "
            f"R={_avg(results,k,'recall'):.4f}  "
            f"F1={_avg(results,k,'f1'):.4f}  "
            f"Hit={_avg(results,k,'accuracy'):.4f}  "
            f"NDCG={np.mean([r.ndcg_by_k.get(k,0) for r in results]):.4f}"
        )

    print(f"\n── MRR y Similitud de Resultados {'─' * 66}")
    table = [
        [
            r.user_name[:26],
            r.mrr_score,
            r.n_strong_ret,
            r.avg_sim_ret,
            r.avg_grade,
            ", ".join(r.query_tags[:4]),
            f"{r.latency_ms:.0f} ms",
        ]
        for r in results
    ]
    print(tabulate(
        table,
        headers=["Usuario", "MRR", f"sim≥{ts:.2f} ret.",
                 "Avg Sim ret.", "Avg Grade", "Tags (query)", "Latencia"],
        tablefmt="rounded_outline",
        floatfmt=".4f",
    ))
    avg_mrr = float(np.mean([r.mrr_score for r in results]))
    print(f"\n  MRR promedio: {avg_mrr:.4f}")

    print(f"\n── Distribución de Grados en Resultados Retornados {'─' * 48}")
    all_grades = [r.grade_map.get(rid, 0) for r in results for rid in r.retrieved_ids]
    total = len(all_grades)
    labels = {
        3: f"sim≥{ts:.2f} + perfil usuario   ",
        2: f"sim≥{ts:.2f} ó jugado           ",
        1: f"sim≥{tw:.2f} + tag match         ",
        0:  "No relevante                    ",
    }
    if total:
        for g in [3, 2, 1, 0]:
            cnt = all_grades.count(g)
            bar = "█" * int(cnt / total * 44)
            print(f"  Grado {g}  [{labels[g]}]:  {cnt:>4} ({cnt/total:>5.1%})  {bar}")

    print(f"\n── Métricas Sistémicas {'─' * 76}")
    sys_table = [[
        f"{report.coverage:.1%}",
        f"{report.personalization:.4f}",
        f"{report.avg_latency_ms:.0f} ms",
        len(results),
    ]]
    print(tabulate(
        sys_table,
        headers=["Coverage", "Personalization", "Avg Latencia", "# Usuarios"],
        tablefmt="rounded_outline",
    ))
    print("  Coverage        : % del catálogo recomendado al menos 1 vez")
    print("  Personalization : 1.0 = cada usuario recibe recomendaciones únicas")

    K_BIN = 5
    print(f"\n── Classification Report Global (binario @K={K_BIN}) {'─' * 50}")
    y_true, y_pred = [], []
    for r in results:
        for rid in r.retrieved_ids[:K_BIN]:
            y_pred.append(1)
            y_true.append(1 if rid in r.relevant_ids else 0)
        for _ in r.relevant_ids - set(r.retrieved_ids[:K_BIN]):
            y_true.append(1)
            y_pred.append(0)

    if y_true and len(set(y_true)) > 1:
        print(classification_report(
            y_true, y_pred,
            target_names=["No relevante", "Relevante"],
            zero_division=0,
        ))
    else:
        print(f"  Sin varianza — ajusta SIM_STRONG_PCT (actual={SIM_STRONG_PCT})")

    best_k  = max(TOP_K_VALUES, key=lambda k: _avg(results, k, "f1"))
    best_f1 = _avg(results, best_k, "f1")
    best_u  = max(results, key=lambda r: r.mrr_score)
    worst_u = min(results, key=lambda r: r.mrr_score)

    print(f"\n── Resumen Ejecutivo {'─' * 78}")
    print(f"  Umbral fuerte (grado 2/3)  : sim ≥ {ts:.3f}  "
          f"(top {SIM_STRONG_PCT}% del catálogo)")
    print(f"  Umbral débil  (grado 1)    : sim ≥ {tw:.3f}  "
          f"(siguiente {SIM_WEAK_PCT}%)")
    print(f"  Mejor K por F1             : @K={best_k}  (F1 promedio = {best_f1:.4f})")
    print(f"  MRR promedio               : {avg_mrr:.4f}")
    print(f"  Mejor usuario (MRR)        : {best_u.user_name}  (MRR={best_u.mrr_score:.4f})")
    print(f"  Peor usuario  (MRR)        : {worst_u.user_name}  (MRR={worst_u.mrr_score:.4f})")
    print(f"  Coverage                   : {report.coverage:.1%}")
    print(f"  Personalization            : {report.personalization:.4f}")
    print(f"  Latencia promedio          : {report.avg_latency_ms:.0f} ms")
    print(LINE)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    report = run_evaluation()
    print_report(report)