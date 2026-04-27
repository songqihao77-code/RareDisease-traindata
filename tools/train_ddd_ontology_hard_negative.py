from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.export_top50_candidates import export_top50_candidates
from tools.train_deeprare_target_light_reranker import (
    load_mondo_resource,
    markdown_table,
    relation_to_gold,
    write_csv,
    write_text,
)


REPORT_DIR = PROJECT_ROOT / "reports" / "ddd_ontology_hard_negative"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ddd_ontology_hard_negative"

MAINLINE_DIR = PROJECT_ROOT / "outputs" / "mainline_full_pipeline"
LIGHT_DIR = PROJECT_ROOT / "outputs" / "deeprare_target_light_reranker"
TRAIN_RAW = LIGHT_DIR / "hgnn_top50_candidates_train_raw.csv"
VAL_RAW = MAINLINE_DIR / "stage4_candidates" / "top50_candidates_validation.csv"
TEST_RAW = MAINLINE_DIR / "stage4_candidates" / "top50_candidates_test.csv"
FINAL_METRICS = MAINLINE_DIR / "mainline_final_metrics.csv"
FINAL_CASE_RANKS = MAINLINE_DIR / "mainline_final_case_ranks.csv"
DDD_WEIGHTS = MAINLINE_DIR / "stage5_ddd_rerank" / "ddd_val_selected_grid_weights.json"

DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
TRAIN_CONFIG = MAINLINE_DIR / "configs" / "stage2_finetune.yaml"
CHECKPOINT = MAINLINE_DIR / "stage2_finetune" / "checkpoints" / "best.pt"
MONDO_JSON = PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json"
DISEASE_INDEX = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx"
HYPEREDGE_CSV = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "rare_disease_hgnn_clean_package_v59" / "v59_hyperedge_weighted_patched.csv"

DDD_TARGET = (0.48, 0.60, 0.63)

FEATURES = [
    "current_score",
    "current_rank_recip",
    "hgnn_score",
    "hgnn_rank_recip",
    "hgnn_margin",
    "exact_overlap",
    "ic_weighted_overlap",
    "semantic_ic_overlap",
    "semantic_coverage_score",
    "case_coverage",
    "disease_coverage",
    "jaccard_overlap",
    "shared_hpo_count",
    "disease_hpo_count",
    "log1p_disease_hpo_count",
    "max_exact_overlap_in_case",
    "max_ic_overlap_in_case",
    "evidence_rank_by_ic_recip",
]

NEGATIVE_MIX_RATIO = {
    "top50_above_gold": 1.0,
    "same_parent_sibling": 1.0,
    "high_hpo_overlap": 1.0,
    "hyperedge_similar": 1.0,
    "similar_case_false": 1.0,
    "random": 1.0,
}


@dataclass(slots=True)
class CandidateModel:
    model_key: str
    c_value: float
    max_negatives_per_case: int
    model: LogisticRegression
    scaler: StandardScaler
    alpha_model: float
    weight_current: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDD ontology-aware hard-negative pairwise reranker.")
    parser.add_argument("--force-rebuild-train-candidates", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def ensure_train_candidates(force: bool) -> None:
    if TRAIN_RAW.is_file() and not force:
        return
    export_top50_candidates(
        data_config_path=DATA_CONFIG,
        train_config_path=TRAIN_CONFIG,
        checkpoint_path=CHECKPOINT,
        output_path=TRAIN_RAW,
        top_k=50,
        case_source="train",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_ddd_weights() -> dict[str, float]:
    payload = load_json(DDD_WEIGHTS)
    return {str(k): float(v) for k, v in payload.get("weights", {}).items()}


def minmax_by_case(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    mins = values.groupby(df["case_id"]).transform("min")
    maxs = values.groupby(df["case_id"]).transform("max")
    denom = (maxs - mins).replace(0, 1.0)
    return (values - mins) / denom


def load_ddd_candidates(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    df = df[df["dataset_name"].eq("DDD")].copy()
    df["split"] = split
    for col in [
        "original_rank",
        "hgnn_score",
        "exact_overlap",
        "ic_weighted_overlap",
        "case_coverage",
        "disease_coverage",
        "disease_hpo_count",
        "shared_hpo_count",
        "jaccard_overlap",
        "semantic_ic_overlap",
        "semantic_coverage_score",
        "hgnn_margin",
        "max_exact_overlap_in_case",
        "max_ic_overlap_in_case",
        "evidence_rank_by_ic",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    weights = load_ddd_weights()
    current_score = (
        weights.get("w_hgnn", 0.0) * minmax_by_case(df, "hgnn_score")
        + weights.get("w_ic", 0.0) * df["ic_weighted_overlap"]
        + weights.get("w_exact", 0.0) * df["exact_overlap"]
        + weights.get("w_semantic", 0.0) * df["semantic_ic_overlap"]
        + weights.get("w_case_cov", 0.0) * df["case_coverage"]
        + weights.get("w_dis_cov", 0.0) * df["disease_coverage"]
        - weights.get("w_size", 0.0) * np.log1p(df["disease_hpo_count"])
    )
    df["current_score"] = current_score
    df["current_rank"] = (
        df.sort_values(["case_id", "current_score", "original_rank"], ascending=[True, False, True], kind="stable")
        .groupby("case_id")
        .cumcount()
        + 1
    )
    df["is_gold"] = df["candidate_id"].astype(str).eq(df["gold_id"].astype(str)).astype(int)
    df["hgnn_rank_recip"] = 1.0 / df["original_rank"].replace(0, np.nan).fillna(999)
    df["current_rank_recip"] = 1.0 / df["current_rank"].replace(0, np.nan).fillna(999)
    df["evidence_rank_by_ic_recip"] = 1.0 / df["evidence_rank_by_ic"].replace(0, np.nan).fillna(999)
    df["log1p_disease_hpo_count"] = np.log1p(df["disease_hpo_count"])
    return df.sort_values(["case_id", "current_rank", "original_rank"], kind="stable").reset_index(drop=True)


def load_disease_pool() -> set[str]:
    if not DISEASE_INDEX.is_file():
        return set()
    df = pd.read_excel(DISEASE_INDEX)
    return set(df["mondo_id"].astype(str))


def load_disease_hpos() -> dict[str, set[str]]:
    if not HYPEREDGE_CSV.is_file():
        return {}
    df = pd.read_csv(HYPEREDGE_CSV, usecols=["mondo_id", "hpo_id"], dtype=str)
    grouped = df.dropna().groupby("mondo_id")["hpo_id"].apply(lambda s: set(s.astype(str)))
    return grouped.to_dict()


def hpo_set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def relation_name(candidate: str, gold: str, mondo: dict[str, Any]) -> str:
    parent_child, sibling, shared = relation_to_gold(candidate, gold, mondo)
    if candidate == gold:
        return "gold"
    if parent_child:
        return "parent_child"
    if sibling:
        return "same_parent_or_sibling"
    if shared > 0:
        return "shared_ancestor"
    return "unrelated"


def add_relation_features(df: pd.DataFrame, mondo: dict[str, Any], disease_hpos: dict[str, set[str]]) -> pd.DataFrame:
    rels = []
    parent_child = []
    sibling = []
    shared = []
    hyper_sim = []
    for cand, gold in zip(df["candidate_id"].astype(str), df["gold_id"].astype(str), strict=False):
        pc, sib, sh = relation_to_gold(cand, gold, mondo)
        parent_child.append(pc)
        sibling.append(sib)
        shared.append(sh)
        rels.append(relation_name(cand, gold, mondo))
        hyper_sim.append(hpo_set_similarity(disease_hpos.get(cand, set()), disease_hpos.get(gold, set())))
    out = df.copy()
    out["mondo_relation_to_gold"] = rels
    out["mondo_parent_child_flag"] = parent_child
    out["mondo_sibling_flag"] = sibling
    out["mondo_shared_ancestor_score"] = shared
    out["disease_hyperedge_similarity_to_gold"] = hyper_sim
    return out


def rank_bucket(rank: int | float) -> str:
    rank = int(rank)
    if rank == 1:
        return "rank=1"
    if rank <= 3:
        return "rank 2-3"
    if rank <= 5:
        return "rank 4-5"
    if rank <= 20:
        return "rank 6-20"
    if rank <= 50:
        return "rank 21-50"
    return "rank>50"


def metric_from_ranks(ranks: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(ranks, errors="coerce").fillna(999999).to_numpy(dtype=int)
    return {
        "num_cases": int(len(arr)),
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "rank_le_50": float(np.mean(arr <= 50)),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
    }


def current_case_ranks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in df.groupby("case_id", sort=False):
        gold = group[group["is_gold"] == 1]
        rank = int(gold["current_rank"].min()) if not gold.empty else 999999
        rows.append({"case_id": case_id, "gold_mondo": str(group["gold_id"].iloc[0]), "current_rank": rank})
    return pd.DataFrame(rows)


def write_error_profile(test_df: pd.DataFrame) -> None:
    final_ranks = pd.read_csv(FINAL_CASE_RANKS, dtype={"case_id": str, "dataset": str})
    ranks = final_ranks[final_ranks["dataset"].eq("DDD")][["case_id", "gold_id", "final_rank"]].rename(
        columns={"gold_id": "gold_mondo", "final_rank": "current_rank"}
    )
    ranks["current_rank"] = pd.to_numeric(ranks["current_rank"], errors="coerce").fillna(999999).astype(int)
    metrics = pd.read_csv(FINAL_METRICS)
    ddd = metrics[metrics["dataset"].eq("DDD")].iloc[0].to_dict()
    bucket_counts = Counter(rank_bucket(rank) for rank in ranks["current_rank"])
    rank_bucket_df = pd.DataFrame(
        [
            {
                "bucket": bucket,
                "count": int(bucket_counts.get(bucket, 0)),
                "rate": float(bucket_counts.get(bucket, 0) / len(ranks)),
            }
            for bucket in ["rank=1", "rank 2-3", "rank 4-5", "rank 6-20", "rank 21-50", "rank>50"]
        ]
    )
    write_csv(rank_bucket_df, REPORT_DIR / "ddd_rank_bucket.csv")
    n = int(ddd["cases"])
    gaps = {
        "Top1 target gap": int(math.ceil(max(DDD_TARGET[0] - float(ddd["top1"]), 0.0) * n)),
        "Top3 target gap": int(math.ceil(max(DDD_TARGET[1] - float(ddd["top3"]), 0.0) * n)),
        "Top5 target gap": int(math.ceil(max(DDD_TARGET[2] - float(ddd["top5"]), 0.0) * n)),
    }
    gold_top50_rank_gt5 = int(((ranks["current_rank"] > 5) & (ranks["current_rank"] <= 50)).sum())
    gold_not_top50 = int((ranks["current_rank"] > 50).sum())
    diagnosis = "主要是 top50 内排序问题，但仍存在一部分候选召回问题。"
    if float(ddd["rank_le_50"]) < 0.70:
        diagnosis = "候选召回问题较重，hard negative training 不足以单独解决。"
    lines = [
        "# DDD Hard Negative Error Profile",
        "",
        f"- DDD num_cases: {n}",
        f"- current Top1/Top3/Top5/Rank<=50: {float(ddd['top1']):.4f}/{float(ddd['top3']):.4f}/{float(ddd['top5']):.4f}/{float(ddd['rank_le_50']):.4f}",
        f"- rank=1: {bucket_counts.get('rank=1', 0)}",
        f"- rank 2-3: {bucket_counts.get('rank 2-3', 0)}",
        f"- rank 4-5: {bucket_counts.get('rank 4-5', 0)}",
        f"- rank 6-20: {bucket_counts.get('rank 6-20', 0)}",
        f"- rank 21-50: {bucket_counts.get('rank 21-50', 0)}",
        f"- rank>50: {bucket_counts.get('rank>50', 0)}",
        f"- Top1 target gap: {gaps['Top1 target gap']} cases",
        f"- Top3 target gap: {gaps['Top3 target gap']} cases",
        f"- Top5 target gap: {gaps['Top5 target gap']} cases",
        f"- gold in top50 but rank>5: {gold_top50_rank_gt5}",
        f"- gold not in top50: {gold_not_top50}",
        "",
        "## 诊断结论",
        f"- {diagnosis}",
        f"- Rank<=50 = {float(ddd['rank_le_50']):.4f}，说明 hard negative training 合理，尤其适合处理 gold 已在 top50 但排在错误近邻之后的病例。",
        f"- rank>50 有 {gold_not_top50} 例，后续仍需要 candidate expansion；但当前最大可恢复池主要来自 rank 6-50 和 rank 2-5 的局部排序。",
        "",
        "## Rank Bucket",
        markdown_table(rank_bucket_df),
    ]
    write_text(REPORT_DIR / "ddd_error_profile.md", "\n".join(lines))


def choose_negative_rows(group: pd.DataFrame, max_per_type: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    gold_rows = group[group["is_gold"] == 1]
    if gold_rows.empty:
        return []
    gold = gold_rows.sort_values("current_rank", kind="stable").iloc[0]
    gold_rank = int(gold["current_rank"])
    gold_overlap = float(gold["ic_weighted_overlap"])
    rows: list[dict[str, Any]] = []

    def add_rows(mask: pd.Series, neg_type: str, order_cols: list[str], ascending: list[bool]) -> None:
        sub = group[mask & (group["is_gold"] == 0)].copy()
        if sub.empty:
            return
        sub = sub.sort_values(order_cols, ascending=ascending, kind="stable").head(max_per_type)
        for _, neg in sub.iterrows():
            rows.append(format_negative_row(neg, gold, neg_type, gold_rank, gold_overlap))

    add_rows(group["current_rank"] < gold_rank, "top50_above_gold", ["current_rank"], [True])
    add_rows(group["mondo_sibling_flag"].eq(1) | group["mondo_parent_child_flag"].eq(1), "same_parent_sibling", ["current_rank"], [True])
    add_rows(group["ic_weighted_overlap"] >= group["ic_weighted_overlap"].quantile(0.80), "high_hpo_overlap", ["ic_weighted_overlap", "current_rank"], [False, True])
    add_rows(group["disease_hyperedge_similarity_to_gold"] >= group["disease_hyperedge_similarity_to_gold"].quantile(0.80), "hyperedge_similar", ["disease_hyperedge_similarity_to_gold", "current_rank"], [False, True])
    add_rows(group["current_rank"] <= 5, "similar_case_false", ["current_rank"], [True])
    random_pool = group[group["is_gold"] == 0]
    if not random_pool.empty:
        take = random_pool.sample(n=min(max_per_type, len(random_pool)), random_state=int(rng.integers(0, 2**31 - 1)))
        for _, neg in take.iterrows():
            rows.append(format_negative_row(neg, gold, "random", gold_rank, gold_overlap))

    seen = set()
    deduped = []
    for row in rows:
        key = (row["case_id"], row["negative_mondo"], row["negative_type"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def format_negative_row(neg: pd.Series, gold: pd.Series, neg_type: str, gold_rank: int, gold_overlap: float) -> dict[str, Any]:
    return {
        "case_id": str(neg["case_id"]),
        "gold_mondo": str(gold["gold_id"]),
        "negative_mondo": str(neg["candidate_id"]),
        "negative_type": neg_type,
        "current_rank_negative": int(neg["current_rank"]),
        "current_rank_gold": int(gold_rank),
        "negative_above_gold_flag": int(int(neg["current_rank"]) < int(gold_rank)),
        "mondo_relation_to_gold": str(neg["mondo_relation_to_gold"]),
        "hpo_overlap_case_negative": float(neg["ic_weighted_overlap"]),
        "hpo_overlap_case_gold": float(gold_overlap),
        "disease_hyperedge_similarity_to_gold": float(neg["disease_hyperedge_similarity_to_gold"]),
        "similarity_case_score_if_any": float(neg["current_score"]),
        "split": str(neg["split"]),
    }


def build_hard_negatives(df: pd.DataFrame, split: str, max_per_type: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _, group in df.groupby("case_id", sort=False):
        rows.extend(choose_negative_rows(group, max_per_type=max_per_type, rng=rng))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(["case_id", "negative_mondo", "negative_type"], keep="first")
    out["split"] = split
    return out


def write_negative_summary(train_neg: pd.DataFrame, val_neg: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    both = pd.concat([train_neg, val_neg], ignore_index=True)
    type_counts = both.groupby(["split", "negative_type"]).size().reset_index(name="count")
    write_csv(type_counts, REPORT_DIR / "hard_negative_type_counts.csv")
    rows = []
    for split, neg, cand in [("train", train_neg, train_df), ("validation", val_neg, val_df)]:
        current = current_case_ranks(cand)
        rows.append(
            {
                "split": split,
                "num_cases": int(cand["case_id"].nunique()),
                "hard_negative_rows": int(len(neg)),
                "avg_hard_negatives_per_case": float(len(neg) / max(cand["case_id"].nunique(), 1)),
                "gold_in_top50_case_coverage": float((current["current_rank"] <= 50).mean()),
                "rank_2_5_cases": int(current["current_rank"].between(2, 5).sum()),
                "rank_6_20_cases": int(current["current_rank"].between(6, 20).sum()),
                "rank_21_50_cases": int(current["current_rank"].between(21, 50).sum()),
                "rank_gt_50_cases": int((current["current_rank"] > 50).sum()),
            }
        )
    overview = pd.DataFrame(rows)
    lines = [
        "# Hard Negative Summary",
        "",
        "## Overview",
        markdown_table(overview),
        "",
        "## Negative Type Counts",
        markdown_table(type_counts),
    ]
    write_text(REPORT_DIR / "hard_negative_summary.md", "\n".join(lines))


def build_pairwise_examples(cand: pd.DataFrame, neg_table: pd.DataFrame, features: list[str], max_negatives_per_case: int) -> tuple[np.ndarray, np.ndarray]:
    by_case_candidate = {(row.case_id, row.candidate_id): row for row in cand.itertuples(index=False)}
    gold_by_case = {row.case_id: row for row in cand[cand["is_gold"] == 1].itertuples(index=False)}
    x_rows = []
    y_rows = []
    for case_id, group in neg_table.groupby("case_id", sort=False):
        gold = gold_by_case.get(case_id)
        if gold is None:
            continue
        pos = np.asarray([float(getattr(gold, f)) for f in features], dtype=float)
        for _, neg in group.head(max_negatives_per_case).iterrows():
            neg_row = by_case_candidate.get((case_id, str(neg["negative_mondo"])))
            if neg_row is None:
                continue
            neg_values = np.asarray([float(getattr(neg_row, f)) for f in features], dtype=float)
            diff = pos - neg_values
            x_rows.append(diff)
            y_rows.append(1)
            x_rows.append(-diff)
            y_rows.append(0)
    if not x_rows:
        raise ValueError("No pairwise hard-negative examples were built.")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=int)


def train_pairwise_model(train_df: pd.DataFrame, train_neg: pd.DataFrame, c_value: float, max_negatives_per_case: int) -> tuple[LogisticRegression, StandardScaler]:
    x, y = build_pairwise_examples(train_df, train_neg, FEATURES, max_negatives_per_case=max_negatives_per_case)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(x_scaled, y)
    return model, scaler


def score_candidates(df: pd.DataFrame, model: LogisticRegression, scaler: StandardScaler, alpha_model: float, weight_current: float) -> pd.DataFrame:
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw = scaler.transform(x) @ model.coef_.reshape(-1)
    model_score = pd.Series(raw, index=df.index)
    model_min = model_score.groupby(df["case_id"]).transform("min")
    model_max = model_score.groupby(df["case_id"]).transform("max")
    model_norm = (model_score - model_min) / (model_max - model_min).replace(0, 1.0)
    cur_score = pd.to_numeric(df["current_score"], errors="coerce").fillna(0.0)
    cur_min = cur_score.groupby(df["case_id"]).transform("min")
    cur_max = cur_score.groupby(df["case_id"]).transform("max")
    cur_norm = (cur_score - cur_min) / (cur_max - cur_min).replace(0, 1.0)
    out = df.copy()
    out["hn_model_score"] = raw
    out["rerank_score"] = alpha_model * model_norm + weight_current * cur_norm
    return out


def ranks_from_scored(scored: pd.DataFrame, rank_col: str = "rerank_score") -> pd.DataFrame:
    rows = []
    for case_id, group in scored.groupby("case_id", sort=False):
        ordered = group.sort_values([rank_col, "current_rank", "original_rank"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
        hits = np.flatnonzero(ordered["is_gold"].to_numpy(dtype=int) == 1)
        rerank = int(hits[0] + 1) if len(hits) else 999999
        gold = group[group["is_gold"] == 1]
        current_rank = int(gold["current_rank"].min()) if not gold.empty else 999999
        rows.append({"case_id": case_id, "gold_mondo": str(group["gold_id"].iloc[0]), "current_rank": current_rank, "new_rank": rerank})
    return pd.DataFrame(rows)


def validation_grid(train_df: pd.DataFrame, val_df: pd.DataFrame, train_neg: pd.DataFrame) -> tuple[pd.DataFrame, CandidateModel]:
    rows = []
    best: CandidateModel | None = None
    best_obj = -1e18
    current_metrics = metric_from_ranks(current_case_ranks(val_df)["current_rank"])
    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]:
        for max_negs in [8, 16, 24, 32]:
            model, scaler = train_pairwise_model(train_df, train_neg, c_value=c_value, max_negatives_per_case=max_negs)
            for alpha, cur_w in [(0.10, 0.90), (0.20, 0.80), (0.35, 0.65), (0.50, 0.50), (0.75, 0.25), (1.00, 0.00)]:
                scored = score_candidates(val_df, model, scaler, alpha, cur_w)
                ranks = ranks_from_scored(scored)
                metrics = metric_from_ranks(ranks["new_rank"])
                deltas = {k: metrics[k] - current_metrics[k] for k in ["top1", "top3", "top5", "rank_le_50"]}
                objective = 4 * deltas["top1"] + 3 * deltas["top3"] + 3 * deltas["top5"]
                penalties = []
                if metrics["top5"] < current_metrics["top5"]:
                    penalty = 10 * (current_metrics["top5"] - metrics["top5"])
                    objective -= penalty
                    penalties.append(f"DDD_top5_drop:{penalty:.4f}")
                if metrics["rank_le_50"] < current_metrics["rank_le_50"]:
                    penalty = 5 * (current_metrics["rank_le_50"] - metrics["rank_le_50"])
                    objective -= penalty
                    penalties.append(f"DDD_rank50_drop:{penalty:.4f}")
                model_key = f"ddd_pairwise_C{c_value}_neg{max_negs}_a{alpha}_cur{cur_w}"
                row = {
                    "model_key": model_key,
                    "strategy": "DDD-specific frozen-feature pairwise hard-negative reranker",
                    "negative_mix_top50_above_gold": NEGATIVE_MIX_RATIO["top50_above_gold"],
                    "negative_mix_same_parent_sibling": NEGATIVE_MIX_RATIO["same_parent_sibling"],
                    "negative_mix_high_hpo_overlap": NEGATIVE_MIX_RATIO["high_hpo_overlap"],
                    "negative_mix_hyperedge_similar": NEGATIVE_MIX_RATIO["hyperedge_similar"],
                    "negative_mix_similar_case_false": NEGATIVE_MIX_RATIO["similar_case_false"],
                    "negative_mix_random": NEGATIVE_MIX_RATIO["random"],
                    "C": c_value,
                    "max_negatives_per_case": max_negs,
                    "alpha_model": alpha,
                    "weight_current": cur_w,
                    "validation_objective": objective,
                    "validation_selected_reason": "无硬约束惩罚" if not penalties else "; ".join(penalties),
                    "current_top1": current_metrics["top1"],
                    "current_top3": current_metrics["top3"],
                    "current_top5": current_metrics["top5"],
                    "current_rank_le_50": current_metrics["rank_le_50"],
                    "new_top1": metrics["top1"],
                    "new_top3": metrics["top3"],
                    "new_top5": metrics["top5"],
                    "new_rank_le_50": metrics["rank_le_50"],
                    "delta_top1": deltas["top1"],
                    "delta_top3": deltas["top3"],
                    "delta_top5": deltas["top5"],
                    "delta_rank_le_50": deltas["rank_le_50"],
                    "top1_case_gap": int(math.ceil(max(DDD_TARGET[0] - metrics["top1"], 0.0) * metrics["num_cases"])),
                    "top3_case_gap": int(math.ceil(max(DDD_TARGET[1] - metrics["top3"], 0.0) * metrics["num_cases"])),
                    "top5_case_gap": int(math.ceil(max(DDD_TARGET[2] - metrics["top5"], 0.0) * metrics["num_cases"])),
                }
                rows.append(row)
                if objective > best_obj:
                    best_obj = objective
                    best = CandidateModel(model_key, c_value, max_negs, model, scaler, alpha, cur_w)
    if best is None:
        raise RuntimeError("No validation config selected.")
    return pd.DataFrame(rows).sort_values("validation_objective", ascending=False), best


def save_selected_config(best: CandidateModel) -> None:
    payload = {
        "protocol": "DDD-specific train hard negatives; validation selection; fixed DDD test evaluation once",
        "model_key": best.model_key,
        "strategy": "DDD-specific frozen-feature pairwise hard-negative reranker",
        "C": best.c_value,
        "negative_mix_ratio": NEGATIVE_MIX_RATIO,
        "max_negatives_per_case": best.max_negatives_per_case,
        "alpha_model": best.alpha_model,
        "weight_current": best.weight_current,
        "features": FEATURES,
        "affects_only_dataset": "DDD",
        "does_not_modify_encoder": True,
    }
    (OUTPUT_DIR / "selected_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "selected_model.pkl").open("wb") as f:
        pickle.dump({"config": payload, "model": best.model, "scaler": best.scaler}, f)


def write_validation_selected(best: CandidateModel, grid: pd.DataFrame) -> None:
    top = grid.iloc[0]
    lines = [
        "# Validation Selected Config",
        "",
        f"- selected_config: `{best.model_key}`",
        "- model scope: DDD-specific only; other datasets keep current mainline outputs.",
        "- model type: frozen-feature pairwise hard-negative reranker; HGNN encoder/checkpoint is not modified.",
        f"- C: {best.c_value}",
        f"- max_negatives_per_case: {best.max_negatives_per_case}",
        f"- alpha_model: {best.alpha_model}",
        f"- weight_current: {best.weight_current}",
        f"- validation_objective: {float(top['validation_objective']):.6f}",
        f"- validation_selected_reason: {top['validation_selected_reason']}",
        "",
        "## Top Validation Rows",
        markdown_table(grid.head(10)),
    ]
    write_text(REPORT_DIR / "validation_selected_config.md", "\n".join(lines))


def fixed_test_eval(test_df: pd.DataFrame, best: CandidateModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = score_candidates(test_df, best.model, best.scaler, best.alpha_model, best.weight_current)
    ranks = ranks_from_scored(scored)
    final_ranks = pd.read_csv(FINAL_CASE_RANKS, dtype={"case_id": str, "dataset": str})
    final_ddd = final_ranks[final_ranks["dataset"].eq("DDD")][["case_id", "final_rank"]].copy()
    final_ddd["final_rank"] = pd.to_numeric(final_ddd["final_rank"], errors="coerce").fillna(999999).astype(int)
    ranks = ranks.drop(columns=["current_rank"]).merge(final_ddd, on="case_id", how="left").rename(columns={"final_rank": "current_rank"})
    ranks["current_rank"] = pd.to_numeric(ranks["current_rank"], errors="coerce").fillna(999999).astype(int)
    metrics_current = metric_from_ranks(ranks["current_rank"])
    metrics_new = metric_from_ranks(ranks["new_rank"])
    n = metrics_current["num_cases"]
    row = {
        "dataset": "DDD",
        "model_scope": "DDD-specific",
        "num_cases": n,
        "current_top1": metrics_current["top1"],
        "new_top1": metrics_new["top1"],
        "target_top1": DDD_TARGET[0],
        "delta_top1": metrics_new["top1"] - metrics_current["top1"],
        "current_top3": metrics_current["top3"],
        "new_top3": metrics_new["top3"],
        "target_top3": DDD_TARGET[1],
        "delta_top3": metrics_new["top3"] - metrics_current["top3"],
        "current_top5": metrics_current["top5"],
        "new_top5": metrics_new["top5"],
        "target_top5": DDD_TARGET[2],
        "delta_top5": metrics_new["top5"] - metrics_current["top5"],
        "current_rank_le_50": metrics_current["rank_le_50"],
        "new_rank_le_50": metrics_new["rank_le_50"],
        "delta_rank_le_50": metrics_new["rank_le_50"] - metrics_current["rank_le_50"],
    }
    for metric, target in [("top1", DDD_TARGET[0]), ("top3", DDD_TARGET[1]), ("top5", DDD_TARGET[2])]:
        row[f"case_gap_before_{metric}"] = int(math.ceil(max(target - row[f"current_{metric}"], 0.0) * n))
        row[f"case_gap_after_{metric}"] = int(math.ceil(max(target - row[f"new_{metric}"], 0.0) * n))
    row["deeprare_ddd_target_reached"] = bool(row["new_top1"] >= DDD_TARGET[0] and row["new_top3"] >= DDD_TARGET[1] and row["new_top5"] >= DDD_TARGET[2])
    return pd.DataFrame([row]), ranks


def write_fixed_test_md(results: pd.DataFrame) -> None:
    row = results.iloc[0]
    lines = [
        "# DDD Ontology Hard Negative Fixed Test Results",
        "",
        "该模型是 DDD-specific enhancement，只对 DDD 应用；MIMIC / MME / RAMEDIS / MyGene2 保持 current mainline，不混写为 ALL general model。",
        "",
        "## DDD Current vs Hard-Negative Model vs DeepRare Target",
        markdown_table(results),
        "",
        "## 结论",
        f"- Top1: {row['current_top1']:.4f} -> {row['new_top1']:.4f}, target 0.48, delta {row['delta_top1']:.4f}.",
        f"- Top3: {row['current_top3']:.4f} -> {row['new_top3']:.4f}, target 0.60, delta {row['delta_top3']:.4f}.",
        f"- Top5: {row['current_top5']:.4f} -> {row['new_top5']:.4f}, target 0.63, delta {row['delta_top5']:.4f}.",
        f"- Rank<=50: {row['current_rank_le_50']:.4f} -> {row['new_rank_le_50']:.4f}.",
        f"- 是否达到 DeepRare DDD target: {'是' if row['deeprare_ddd_target_reached'] else '否'}。",
    ]
    write_text(REPORT_DIR / "fixed_test_results.md", "\n".join(lines))


def case_delta(ranks: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    out = ranks.copy()
    for k in [1, 3, 5]:
        out[f"current_top{k}"] = out["current_rank"] <= k
        out[f"new_top{k}"] = out["new_rank"] <= k
        out[f"top{k}_delta"] = out[f"new_top{k}"].astype(int) - out[f"current_top{k}"].astype(int)
    out["rank_delta"] = out["current_rank"] - out["new_rank"]
    out["rank_change"] = np.where(out["rank_delta"] > 0, "improved", np.where(out["rank_delta"] < 0, "worsened", "unchanged"))
    type_by_case = neg.groupby("case_id")["negative_type"].apply(lambda s: "|".join(sorted(set(s)))).to_dict()
    out["train_like_hard_negative_types_present_in_test_top50"] = out["case_id"].map(type_by_case).fillna("")
    return out


def write_case_delta_summary(delta: pd.DataFrame) -> None:
    row = {
        "dataset": "DDD",
        "num_cases": int(len(delta)),
        "top1_gained_cases": int((delta["top1_delta"] == 1).sum()),
        "top1_lost_cases": int((delta["top1_delta"] == -1).sum()),
        "top3_gained_cases": int((delta["top3_delta"] == 1).sum()),
        "top3_lost_cases": int((delta["top3_delta"] == -1).sum()),
        "top5_gained_cases": int((delta["top5_delta"] == 1).sum()),
        "top5_lost_cases": int((delta["top5_delta"] == -1).sum()),
        "rank_improved": int((delta["rank_change"] == "improved").sum()),
        "rank_worsened": int((delta["rank_change"] == "worsened").sum()),
        "rank_unchanged": int((delta["rank_change"] == "unchanged").sum()),
        "final_rank_2_5_to_rank1": int((delta["current_rank"].between(2, 5) & (delta["new_rank"] <= 1)).sum()),
        "final_rank_4_5_to_top3": int((delta["current_rank"].between(4, 5) & (delta["new_rank"] <= 3)).sum()),
        "final_rank_6_50_to_top5": int((delta["current_rank"].between(6, 50) & (delta["new_rank"] <= 5)).sum()),
        "rank_gt50_recalled_to_top50": int(((delta["current_rank"] > 50) & (delta["new_rank"] <= 50)).sum()),
    }
    summary = pd.DataFrame([row])
    type_gain = (
        delta[delta["rank_change"].eq("improved")]["train_like_hard_negative_types_present_in_test_top50"]
        .str.get_dummies(sep="|")
        .sum()
        .reset_index()
    )
    if not type_gain.empty:
        type_gain.columns = ["negative_type", "improved_case_count_with_type_present"]
    write_csv(summary, REPORT_DIR / "case_level_delta_summary.csv")
    write_csv(type_gain, REPORT_DIR / "hard_negative_type_contribution.csv")
    lines = [
        "# DDD Case-Level Delta Summary",
        "",
        markdown_table(summary),
        "",
        "## Hard Negative 类型对改进的贡献",
        "这里统计的是 improved cases 的 test top50 中出现过哪些 hard-negative-like 类型；不是因果归因。",
        markdown_table(type_gain),
    ]
    write_text(REPORT_DIR / "case_level_delta_summary.md", "\n".join(lines))


def bootstrap_ci(delta: pd.DataFrame, iters: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    current = delta["current_rank"].to_numpy(dtype=int)
    new = delta["new_rank"].to_numpy(dtype=int)
    n = len(delta)
    rows = []
    for k in [1, 3, 5]:
        values = []
        deltas = []
        for _ in range(iters):
            idx = rng.integers(0, n, size=n)
            cur_v = float(np.mean(current[idx] <= k))
            new_v = float(np.mean(new[idx] <= k))
            values.append(new_v)
            deltas.append(new_v - cur_v)
        for name, arr in [(f"top{k}", np.asarray(values)), (f"delta_top{k}", np.asarray(deltas))]:
            rows.append(
                {
                    "dataset": "DDD",
                    "metric": name,
                    "mean": float(arr.mean()),
                    "ci95_low": float(np.quantile(arr, 0.025)),
                    "ci95_high": float(np.quantile(arr, 0.975)),
                    "num_cases": n,
                    "stable_above_current": bool(name.startswith("delta") and np.quantile(arr, 0.025) > 0),
                }
            )
    return pd.DataFrame(rows)


def write_bootstrap_md(ci: pd.DataFrame, results: pd.DataFrame) -> None:
    row = results.iloc[0]
    lines = [
        "# DDD Bootstrap 95% CI",
        "",
        markdown_table(ci),
        "",
        "## 稳定性判断",
        f"- fixed test Top1/Top3/Top5: {row['new_top1']:.4f}/{row['new_top3']:.4f}/{row['new_top5']:.4f}.",
        f"- DeepRare target: 0.48/0.60/0.63.",
        f"- delta 是否稳定超过 current: {bool(ci[ci['metric'].str.startswith('delta')]['stable_above_current'].any())}.",
    ]
    write_text(REPORT_DIR / "bootstrap_ci.md", "\n".join(lines))


def write_paper_and_next(results: pd.DataFrame) -> None:
    row = results.iloc[0]
    reached = bool(row["deeprare_ddd_target_reached"])
    lines = [
        "# Recommended Paper Table",
        "",
        f"1. hard negative model 是否能作为 DDD 新主线：{'可以，作为 DDD-specific enhancement' if reached and row['delta_top1'] >= 0 and row['delta_top3'] >= 0 and row['delta_top5'] >= 0 else '暂不建议作为 DDD 新主线'}。",
        f"2. 是否达到 DeepRare DDD target：{'是' if reached else '否'}。",
        "3. 是否牺牲 MIMIC / MME / RAMEDIS / MyGene2：否；本模型是 DDD-specific，只对 DDD 应用，其他数据集保持 current mainline。",
        "4. 如果是 DDD-specific model，是否只能作为 dataset-specific enhancement：是，不能混写为 ALL general model。",
        "5. light reranker 是否仍作为负结果附表：是。",
        "6. 图对比学习是否仍后置：是。",
        "7. 下一步是否需要 MIMIC top1-oriented listwise reranker：是，MIMIC 仍需要单独路线。",
    ]
    write_text(REPORT_DIR / "recommended_paper_table.md", "\n".join(lines))
    if not reached:
        next_lines = [
            "# Next After Failed DDD Target",
            "",
            "- candidate expansion：需要。当前 rank>50 仍有大量 case，单靠 top50 内排序无法补齐所有 Top5 gap。",
            "- listwise reranker：需要。当前 pairwise hard-negative head 仍可能牺牲部分已靠前病例，应做 listwise/top-k constrained objective。",
            "- encoder-level hard negative fine-tuning：需要作为下一阶段，但必须保持 encoder 架构不变，只新增可开关 sampler/loss，并输出到独立 checkpoint。",
            "- label/mapping/outlier audit：需要，尤其检查 DDD sibling/parent-child 近邻是否存在标注粒度问题。",
            "- 图对比学习：仍后置，等 candidate expansion 和 encoder-level HN fine-tuning 后再评估。",
        ]
        write_text(REPORT_DIR / "next_after_failed_ddd_target.md", "\n".join(next_lines))


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_train_candidates(args.force_rebuild_train_candidates)

    mondo = load_mondo_resource(MONDO_JSON)
    disease_hpos = load_disease_hpos()
    _ = load_disease_pool()

    train_df = add_relation_features(load_ddd_candidates(TRAIN_RAW, "train"), mondo, disease_hpos)
    val_df = add_relation_features(load_ddd_candidates(VAL_RAW, "validation"), mondo, disease_hpos)
    test_df = add_relation_features(load_ddd_candidates(TEST_RAW, "test"), mondo, disease_hpos)

    write_error_profile(test_df)

    train_neg = build_hard_negatives(train_df, "train", max_per_type=5, seed=args.random_seed)
    val_neg = build_hard_negatives(val_df, "validation", max_per_type=5, seed=args.random_seed + 1)
    write_csv(train_neg, OUTPUT_DIR / "ddd_train_hard_negatives.csv")
    write_csv(val_neg, OUTPUT_DIR / "ddd_validation_hard_negatives.csv")
    write_negative_summary(train_neg, val_neg, train_df, val_df)

    grid, best = validation_grid(train_df, val_df, train_neg)
    write_csv(grid, REPORT_DIR / "validation_grid.csv")
    save_selected_config(best)
    write_validation_selected(best, grid)

    results, test_ranks = fixed_test_eval(test_df, best)
    write_csv(results, REPORT_DIR / "fixed_test_results.csv")
    write_csv(test_ranks, OUTPUT_DIR / "fixed_test_case_ranks.csv")
    write_fixed_test_md(results)

    test_neg_like = build_hard_negatives(test_df, "test_analysis_only", max_per_type=5, seed=args.random_seed + 2)
    delta = case_delta(test_ranks, test_neg_like)
    write_csv(delta, REPORT_DIR / "case_level_delta.csv")
    write_case_delta_summary(delta)

    ci = bootstrap_ci(delta, args.bootstrap_iters, args.random_seed)
    write_csv(ci, REPORT_DIR / "bootstrap_ci.csv")
    write_bootstrap_md(ci, results)
    write_paper_and_next(results)

    manifest = {
        "strategy": "DDD-specific frozen-feature pairwise hard-negative reranker",
        "selected_config": str((OUTPUT_DIR / "selected_config.json").resolve()),
        "fixed_test_results": str((REPORT_DIR / "fixed_test_results.csv").resolve()),
        "does_not_modify_encoder": True,
        "does_not_overwrite_mainline": True,
        "test_side_tuning": False,
        "bootstrap_iters": args.bootstrap_iters,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
