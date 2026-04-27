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
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.export_top50_candidates import export_top50_candidates


REPORT_DIR = PROJECT_ROOT / "reports" / "deeprare_target_light_reranker"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "deeprare_target_light_reranker"

MAINLINE_DIR = PROJECT_ROOT / "outputs" / "mainline_full_pipeline"
TRAIN_RAW = OUTPUT_DIR / "hgnn_top50_candidates_train_raw.csv"
VAL_RAW = MAINLINE_DIR / "stage4_candidates" / "top50_candidates_validation.csv"
TEST_RAW = MAINLINE_DIR / "stage4_candidates" / "top50_candidates_test.csv"
VAL_EXPANSION = PROJECT_ROOT / "outputs" / "mimic_residual_after_similar_case" / "residual_expanded_candidates_validation.csv"
TEST_EXPANSION = PROJECT_ROOT / "outputs" / "mimic_residual_after_similar_case" / "residual_expanded_candidates_test_analysis_only.csv"
TEST_SIMILAR_RANKED = MAINLINE_DIR / "stage6_mimic_similar_case" / "similar_case_fixed_test_ranked_candidates.csv"
DDD_WEIGHTS = MAINLINE_DIR / "stage5_ddd_rerank" / "ddd_val_selected_grid_weights.json"
FINAL_METRICS = MAINLINE_DIR / "mainline_final_metrics_with_sources.csv"
FINAL_CASE_RANKS = MAINLINE_DIR / "mainline_final_case_ranks.csv"
DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
TRAIN_CONFIG = MAINLINE_DIR / "configs" / "stage2_finetune.yaml"
CHECKPOINT = MAINLINE_DIR / "stage2_finetune" / "checkpoints" / "best.pt"
MONDO_JSON = PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json"

PROTECTED_DATASETS = {"MME", "RAMEDIS", "MyGene2"}
TARGETED_DATASETS = {"DDD", "HMS", "LIRICAL", "mimic_rag_0425", "mimic_test_recleaned_mondo_hpo_rows"}
DATASET_ORDER = ["DDD", "HMS", "LIRICAL", "MME", "MyGene2", "RAMEDIS", "mimic_rag_0425", "mimic_test_recleaned_mondo_hpo_rows"]

DEEPRARE_TARGETS = {
    "MME": (0.78, 0.85, 0.90),
    "HMS": (0.57, 0.65, 0.71),
    "LIRICAL": (0.56, 0.65, 0.68),
    "RAMEDIS": (0.73, 0.83, 0.85),
    "mimic_test_recleaned_mondo_hpo_rows": (0.29, 0.37, 0.39),
    "MyGene2": (0.76, 0.80, 0.81),
    "DDD": (0.48, 0.60, 0.63),
}

BASE_NUMERIC_FEATURES = [
    "hgnn_score",
    "hgnn_rank",
    "hgnn_rank_recip",
    "hgnn_margin",
    "final_current_score",
    "final_current_rank",
    "final_current_rank_recip",
    "similar_case_score",
    "similar_case_rank_recip",
    "exact_hpo_overlap_count",
    "exact_hpo_overlap_ratio",
    "ic_weighted_overlap",
    "semantic_overlap",
    "case_coverage",
    "disease_coverage",
    "jaccard_overlap",
    "max_exact_overlap_in_case",
    "max_ic_overlap_in_case",
    "evidence_rank_by_ic_recip",
    "case_hpo_count",
    "disease_hpo_count",
    "log1p_case_hpo_count",
    "log1p_disease_hpo_count",
    "candidate_source_hgnn",
    "candidate_source_similar_case",
    "candidate_source_hpo_expansion",
    "candidate_source_mondo_expansion",
    "candidate_source_count",
    "obsolete_label_flag",
    "multilabel_case_flag",
]


@dataclass(slots=True)
class TrainedCandidateModel:
    model_key: str
    model_type: str
    hyperparams: dict[str, Any]
    model: Any
    scaler: StandardScaler | None
    features: list[str]
    score_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate DeepRare target-aware light reranker.")
    parser.add_argument("--force-rebuild-candidates", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无记录_"
    use_df = df.head(max_rows).copy() if max_rows else df.copy()
    rendered = use_df.copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].map(fmt)
    columns = list(rendered.columns)
    rows = rendered.values.tolist()
    widths = [max(len(str(col)), *(len(str(row[i])) for row in rows)) for i, col in enumerate(columns)]
    lines = [
        "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |")
    return "\n".join(lines)


def mondo_id_from_iri(value: Any) -> str | None:
    text = str(value or "")
    if "MONDO_" not in text:
        return None
    return "MONDO:" + text.rsplit("MONDO_", 1)[1].replace("_", ":")[:7]


def load_mondo_resource(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"parents": {}, "children": {}, "ancestors": {}, "deprecated": set(), "replacements": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = data.get("graphs", [{}])[0]
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    deprecated: set[str] = set()
    replacements: dict[str, str] = {}

    for node in graph.get("nodes", []):
        mondo = mondo_id_from_iri(node.get("id"))
        if not mondo:
            continue
        meta = node.get("meta", {}) or {}
        label = str(node.get("lbl", ""))
        if bool(meta.get("deprecated", False)) or label.lower().startswith("obsolete"):
            deprecated.add(mondo)
        for item in meta.get("basicPropertyValues", []) or []:
            pred = str(item.get("pred", ""))
            replacement = mondo_id_from_iri(item.get("val"))
            if pred.endswith("IAO_0100001") and replacement:
                replacements[mondo] = replacement

    for edge in graph.get("edges", []):
        if edge.get("pred") != "is_a":
            continue
        sub = mondo_id_from_iri(edge.get("sub"))
        obj = mondo_id_from_iri(edge.get("obj"))
        if sub and obj:
            parents[sub].add(obj)
            children[obj].add(sub)

    ancestor_cache: dict[str, set[str]] = {}

    def ancestors(mondo: str) -> set[str]:
        if mondo in ancestor_cache:
            return ancestor_cache[mondo]
        seen: set[str] = set()
        stack = list(parents.get(mondo, set()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(parents.get(current, set()) - seen)
        ancestor_cache[mondo] = seen
        return seen

    return {
        "parents": parents,
        "children": children,
        "ancestors_fn": ancestors,
        "deprecated": deprecated,
        "replacements": replacements,
    }


def relation_to_gold(candidate: str, gold: str, mondo: dict[str, Any]) -> tuple[int, int, float]:
    if not candidate or not gold:
        return 0, 0, 0.0
    parents = mondo["parents"]
    ancestors_fn = mondo["ancestors_fn"]
    cand_parents = parents.get(candidate, set())
    gold_parents = parents.get(gold, set())
    parent_child = int(candidate in parents.get(gold, set()) or gold in parents.get(candidate, set()))
    sibling = int(bool(cand_parents & gold_parents) and candidate != gold)
    cand_anc = ancestors_fn(candidate)
    gold_anc = ancestors_fn(gold)
    shared = cand_anc & gold_anc
    denom = max(len(cand_anc | gold_anc), 1)
    shared_score = float(len(shared) / denom)
    return parent_child, sibling, shared_score


def ensure_train_raw_candidates(force: bool) -> None:
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


def read_raw_hgnn(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    rename = {
        "dataset_name": "dataset",
        "candidate_id": "candidate_mondo",
        "gold_id": "gold_mondo",
        "original_rank": "hgnn_rank",
        "shared_hpo_count": "exact_hpo_overlap_count",
        "exact_overlap": "exact_hpo_overlap_ratio",
        "semantic_ic_overlap": "semantic_overlap",
    }
    df = df.rename(columns=rename)
    df["split"] = split
    df["candidate_source_hgnn"] = 1
    df["candidate_source_similar_case"] = 0
    df["candidate_source_hpo_expansion"] = 0
    df["candidate_source_mondo_expansion"] = 0
    return df


def append_expansion_rows(base: pd.DataFrame, path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        return base
    exp = pd.read_csv(path, dtype={"case_id": str, "gold_mondo": str, "candidate_id": str})
    if exp.empty:
        return base
    rows = []
    dataset_by_case = base.groupby("case_id")["dataset"].first().to_dict()
    for _, row in exp.iterrows():
        case_id = str(row["case_id"])
        candidate = str(row["candidate_id"])
        expansion_type = str(row.get("expansion_type", ""))
        dataset = dataset_by_case.get(case_id)
        if dataset is None:
            if "mimic" in case_id.lower():
                dataset = "mimic_test_recleaned_mondo_hpo_rows" if split == "test" else "mimic_rag_0425"
            else:
                dataset = Path(case_id.split("::")[1]).stem if "::" in case_id else "unknown"
        rows.append(
            {
                "split": split,
                "case_id": case_id,
                "dataset": dataset,
                "gold_mondo": str(row.get("gold_mondo", "")),
                "candidate_mondo": candidate,
                "hgnn_rank": 999,
                "hgnn_score": 0.0,
                "similar_case_score": float(pd.to_numeric(row.get("score", 0.0), errors="coerce") or 0.0),
                "similar_case_rank": float(pd.to_numeric(row.get("expansion_rank", 999), errors="coerce") or 999),
                "candidate_source_hgnn": 0,
                "candidate_source_similar_case": int("similar_case" in expansion_type),
                "candidate_source_hpo_expansion": int("hpo" in expansion_type.lower()),
                "candidate_source_mondo_expansion": int("mondo" in expansion_type.lower()),
            }
        )
    all_df = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    all_df = (
        all_df.sort_values(
            ["case_id", "candidate_mondo", "candidate_source_hgnn", "candidate_source_similar_case"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .drop_duplicates(["case_id", "candidate_mondo"], keep="first")
        .reset_index(drop=True)
    )
    return all_df


def merge_test_similar_ranked(base: pd.DataFrame) -> pd.DataFrame:
    if not TEST_SIMILAR_RANKED.is_file():
        return base
    sim = pd.read_csv(TEST_SIMILAR_RANKED, dtype={"case_id": str, "gold_id": str, "candidate_id": str})
    sim = sim.rename(
        columns={
            "gold_id": "gold_mondo",
            "candidate_id": "candidate_mondo",
            "score": "similar_ranked_score",
            "rank": "similar_ranked_rank",
            "similar_case_source_score": "similar_case_source_score",
        }
    )
    dataset_by_case = base.groupby("case_id")["dataset"].first().to_dict()
    rows = []
    for _, row in sim.iterrows():
        rows.append(
            {
                "split": "test",
                "case_id": row["case_id"],
                "dataset": dataset_by_case.get(row["case_id"], "mimic_test_recleaned_mondo_hpo_rows"),
                "gold_mondo": row["gold_mondo"],
                "candidate_mondo": row["candidate_mondo"],
                "hgnn_rank": 999,
                "hgnn_score": float(row.get("hgnn_component", 0.0)),
                "similar_case_score": float(row.get("similar_case_source_score", row.get("similar_ranked_score", 0.0))),
                "similar_case_rank": float(row.get("similar_ranked_rank", 999)),
                "similar_ranked_score": float(row.get("similar_ranked_score", 0.0)),
                "similar_ranked_rank": float(row.get("similar_ranked_rank", 999)),
                "final_current_score": float(row.get("similar_ranked_score", 0.0)),
                "final_current_rank": float(row.get("similar_ranked_rank", 999)),
                "candidate_source_hgnn": 0,
                "candidate_source_similar_case": 1,
                "candidate_source_hpo_expansion": 0,
                "candidate_source_mondo_expansion": 0,
            }
        )
    sim_df = pd.DataFrame(rows)
    all_df = pd.concat([base, sim_df], ignore_index=True)
    agg: dict[str, str] = {
        "split": "first",
        "dataset": "first",
        "gold_mondo": "first",
    }
    for col in all_df.columns:
        if col in {"case_id", "candidate_mondo", "split", "dataset", "gold_mondo"}:
            continue
        if col.startswith("candidate_source_"):
            agg[col] = "max"
        elif col.endswith("_rank") or col == "hgnn_rank":
            agg[col] = "min"
        else:
            agg[col] = "max"
    merged = all_df.groupby(["case_id", "candidate_mondo"], as_index=False).agg(agg)
    return merged


def load_ddd_weights() -> dict[str, float]:
    if not DDD_WEIGHTS.is_file():
        return {}
    payload = json.loads(DDD_WEIGHTS.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in payload.get("weights", {}).items()}


def minmax_by_case(series: pd.Series, case_ids: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mins = values.groupby(case_ids).transform("min")
    maxs = values.groupby(case_ids).transform("max")
    denom = (maxs - mins).replace(0, 1.0)
    return (values - mins) / denom


def derive_case_hpo_count(df: pd.DataFrame) -> pd.Series:
    shared = pd.to_numeric(df.get("exact_hpo_overlap_count", 0.0), errors="coerce").fillna(0.0)
    coverage = pd.to_numeric(df.get("case_coverage", 0.0), errors="coerce").fillna(0.0)
    derived = shared / coverage.replace(0, np.nan)
    by_case = derived.groupby(df["case_id"]).transform("median")
    return by_case.fillna(0.0).round().astype(int)


def apply_current_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    weights = load_ddd_weights()
    for col in [
        "hgnn_score",
        "hgnn_rank",
        "exact_hpo_overlap_ratio",
        "ic_weighted_overlap",
        "case_coverage",
        "disease_coverage",
        "disease_hpo_count",
        "semantic_overlap",
        "similar_case_score",
        "similar_case_rank",
        "similar_ranked_score",
        "similar_ranked_rank",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["ddd_grid_score"] = (
        weights.get("w_hgnn", 0.0) * minmax_by_case(out["hgnn_score"], out["case_id"])
        + weights.get("w_ic", 0.0) * out["ic_weighted_overlap"]
        + weights.get("w_exact", 0.0) * out["exact_hpo_overlap_ratio"]
        + weights.get("w_semantic", 0.0) * out["semantic_overlap"]
        + weights.get("w_case_cov", 0.0) * out["case_coverage"]
        + weights.get("w_dis_cov", 0.0) * out["disease_coverage"]
        - weights.get("w_size", 0.0) * np.log1p(out["disease_hpo_count"])
    )

    out["final_current_score"] = out.get("final_current_score", np.nan)
    out["final_current_rank"] = out.get("final_current_rank", np.nan)
    ddd_mask = out["dataset"].eq("DDD")
    mimic_mask = out["dataset"].astype(str).str.contains("mimic", case=False, na=False)
    out.loc[ddd_mask, "final_current_score"] = out.loc[ddd_mask, "ddd_grid_score"]
    out.loc[ddd_mask, "final_current_rank"] = (
        out.loc[ddd_mask]
        .sort_values(["case_id", "ddd_grid_score", "hgnn_rank"], ascending=[True, False, True], kind="stable")
        .groupby("case_id")
        .cumcount()
        + 1
    )
    similar_rank = pd.to_numeric(out.get("similar_ranked_rank", np.nan), errors="coerce")
    similar_score = pd.to_numeric(out.get("similar_ranked_score", np.nan), errors="coerce")
    out.loc[mimic_mask & similar_rank.notna() & (similar_rank > 0), "final_current_rank"] = similar_rank[
        mimic_mask & similar_rank.notna() & (similar_rank > 0)
    ]
    out.loc[mimic_mask & similar_score.notna(), "final_current_score"] = similar_score[mimic_mask & similar_score.notna()]
    missing_score = pd.to_numeric(out["final_current_score"], errors="coerce").isna()
    out.loc[missing_score, "final_current_score"] = out.loc[missing_score, "hgnn_score"]
    missing_rank = pd.to_numeric(out["final_current_rank"], errors="coerce").isna() | (pd.to_numeric(out["final_current_rank"], errors="coerce") <= 0)
    out.loc[missing_rank, "final_current_rank"] = out.loc[missing_rank, "hgnn_rank"]
    return out


def finalize_candidate_features(df: pd.DataFrame, mondo: dict[str, Any]) -> pd.DataFrame:
    out = apply_current_rank_features(df)
    numeric_defaults = {
        "exact_hpo_overlap_count": 0,
        "exact_hpo_overlap_ratio": 0.0,
        "ic_weighted_overlap": 0.0,
        "semantic_overlap": 0.0,
        "case_coverage": 0.0,
        "disease_coverage": 0.0,
        "jaccard_overlap": 0.0,
        "hgnn_margin": 0.0,
        "max_exact_overlap_in_case": 0.0,
        "max_ic_overlap_in_case": 0.0,
        "evidence_rank_by_ic": 999,
        "disease_hpo_count": 0,
    }
    for col, default in numeric_defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    for col in [
        "candidate_source_hgnn",
        "candidate_source_similar_case",
        "candidate_source_hpo_expansion",
        "candidate_source_mondo_expansion",
    ]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out["label_is_strict_gold"] = out["candidate_mondo"].astype(str).eq(out["gold_mondo"].astype(str)).astype(int)
    out["label_is_any_gold"] = out["label_is_strict_gold"]
    out["case_hpo_count"] = derive_case_hpo_count(out)
    out["candidate_source_count"] = out[
        [
            "candidate_source_hgnn",
            "candidate_source_similar_case",
            "candidate_source_hpo_expansion",
            "candidate_source_mondo_expansion",
        ]
    ].sum(axis=1)
    out["hgnn_rank_recip"] = 1.0 / pd.to_numeric(out["hgnn_rank"], errors="coerce").replace(0, np.nan).fillna(999)
    out["final_current_rank_recip"] = 1.0 / pd.to_numeric(out["final_current_rank"], errors="coerce").replace(0, np.nan).fillna(999)
    out["similar_case_rank_recip"] = 1.0 / pd.to_numeric(out["similar_case_rank"], errors="coerce").replace(0, np.nan).fillna(999)
    out["evidence_rank_by_ic_recip"] = 1.0 / pd.to_numeric(out["evidence_rank_by_ic"], errors="coerce").replace(0, np.nan).fillna(999)
    out["log1p_case_hpo_count"] = np.log1p(pd.to_numeric(out["case_hpo_count"], errors="coerce").fillna(0.0))
    out["log1p_disease_hpo_count"] = np.log1p(pd.to_numeric(out["disease_hpo_count"], errors="coerce").fillna(0.0))

    parent_child = []
    sibling = []
    shared_ancestor = []
    deprecated = mondo["deprecated"]
    replacements = mondo["replacements"]
    for candidate, gold in zip(out["candidate_mondo"].astype(str), out["gold_mondo"].astype(str), strict=False):
        pc, sib, shared = relation_to_gold(candidate, gold, mondo)
        parent_child.append(pc)
        sibling.append(sib)
        shared_ancestor.append(shared)
    out["mondo_parent_child_flag"] = parent_child
    out["mondo_sibling_flag"] = sibling
    out["mondo_shared_ancestor_score"] = shared_ancestor
    out["synonym_or_replacement_flag"] = [
        int(replacements.get(str(c)) == str(g) or replacements.get(str(g)) == str(c))
        for c, g in zip(out["candidate_mondo"], out["gold_mondo"], strict=False)
    ]
    out["obsolete_label_flag"] = out["candidate_mondo"].astype(str).isin(deprecated).astype(int)
    out["multilabel_case_flag"] = out["gold_mondo"].astype(str).str.contains(r"[|;]").astype(int)

    for dataset in DATASET_ORDER:
        out[f"dataset__{dataset}"] = out["dataset"].eq(dataset).astype(float)

    columns = [
        "split",
        "dataset",
        "case_id",
        "candidate_mondo",
        "gold_mondo",
        "label_is_strict_gold",
        "label_is_any_gold",
        "hgnn_score",
        "hgnn_rank",
        "similar_case_score",
        "similar_case_rank",
        "final_current_score",
        "final_current_rank",
        "exact_hpo_overlap_count",
        "exact_hpo_overlap_ratio",
        "ic_weighted_overlap",
        "semantic_overlap",
        "mondo_parent_child_flag",
        "mondo_sibling_flag",
        "mondo_shared_ancestor_score",
        "synonym_or_replacement_flag",
        "candidate_source_hgnn",
        "candidate_source_similar_case",
        "candidate_source_hpo_expansion",
        "candidate_source_mondo_expansion",
        "candidate_source_count",
        "case_hpo_count",
        "disease_hpo_count",
        "obsolete_label_flag",
        "multilabel_case_flag",
        "hgnn_rank_recip",
        "hgnn_margin",
        "final_current_rank_recip",
        "similar_case_rank_recip",
        "case_coverage",
        "disease_coverage",
        "jaccard_overlap",
        "max_exact_overlap_in_case",
        "max_ic_overlap_in_case",
        "evidence_rank_by_ic",
        "evidence_rank_by_ic_recip",
        "log1p_case_hpo_count",
        "log1p_disease_hpo_count",
        "ddd_grid_score",
        *[f"dataset__{dataset}" for dataset in DATASET_ORDER],
    ]
    for col in columns:
        if col not in out.columns:
            out[col] = 0
    return out[columns].sort_values(["case_id", "final_current_rank", "hgnn_rank"], kind="stable").reset_index(drop=True)


def build_candidate_table(split: str, raw_path: Path, mondo: dict[str, Any]) -> pd.DataFrame:
    base = read_raw_hgnn(raw_path, split)
    if split == "validation":
        base = append_expansion_rows(base, VAL_EXPANSION, split)
    elif split == "test":
        base = append_expansion_rows(base, TEST_EXPANSION, split)
        base = merge_test_similar_ranked(base)
    return finalize_candidate_features(base, mondo)


def feature_columns(df: pd.DataFrame) -> list[str]:
    dataset_cols = [col for col in df.columns if col.startswith("dataset__")]
    # Gold-relative MONDO flags are intentionally excluded from model scoring.
    return BASE_NUMERIC_FEATURES + dataset_cols


def prepare_xy(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use_df = df.copy()
    for col in features:
        use_df[col] = pd.to_numeric(use_df[col], errors="coerce").fillna(0.0)
    x = use_df[features].to_numpy(dtype=float)
    y = use_df["label_is_strict_gold"].to_numpy(dtype=int)
    weights = np.where(y == 1, 20.0, 1.0)
    return x, y, weights


def build_pairwise_examples(df: pd.DataFrame, features: list[str], max_negatives: int = 20) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for _, group in df.groupby("case_id", sort=False):
        gold = group[group["label_is_strict_gold"] == 1]
        if gold.empty:
            continue
        pos = gold.sort_values("hgnn_rank", kind="stable").iloc[0]
        pos_rank = float(pos["hgnn_rank"])
        neg = group[group["label_is_strict_gold"] == 0].copy()
        if neg.empty:
            continue
        neg["hard_negative_score"] = 0.0
        neg.loc[neg["hgnn_rank"] < pos_rank, "hard_negative_score"] += 4.0
        neg.loc[neg["final_current_rank"] <= 5, "hard_negative_score"] += 3.0
        neg.loc[neg["mondo_sibling_flag"] == 1, "hard_negative_score"] += 2.0
        neg.loc[neg["mondo_parent_child_flag"] == 1, "hard_negative_score"] += 2.0
        neg["hard_negative_score"] += pd.to_numeric(neg["ic_weighted_overlap"], errors="coerce").fillna(0.0)
        neg["hard_negative_score"] += pd.to_numeric(neg["similar_case_score"], errors="coerce").fillna(0.0)
        neg = neg.sort_values(["hard_negative_score", "final_current_rank", "hgnn_rank"], ascending=[False, True, True]).head(max_negatives)
        pos_values = pos[features].to_numpy(dtype=float)
        for _, neg_row in neg.iterrows():
            neg_values = neg_row[features].to_numpy(dtype=float)
            diff = pos_values - neg_values
            x_rows.append(diff)
            y_rows.append(1)
            x_rows.append(-diff)
            y_rows.append(0)
    if not x_rows:
        raise ValueError("No pairwise examples were created.")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=int)


def pointwise_tree_training_sample(train_df: pd.DataFrame, seed: int, max_negatives_per_case: int = 8) -> pd.DataFrame:
    rows = []
    for _, group in train_df.groupby("case_id", sort=False):
        pos = group[group["label_is_strict_gold"] == 1]
        if not pos.empty:
            rows.append(pos)
        neg = group[group["label_is_strict_gold"] == 0].sort_values(
            ["final_current_rank", "hgnn_rank", "ic_weighted_overlap"],
            ascending=[True, True, False],
            kind="stable",
        ).head(max_negatives_per_case)
        if not neg.empty:
            rows.append(neg)
    if not rows:
        return train_df
    sampled = pd.concat(rows, ignore_index=True)
    if len(sampled) > 180_000:
        sampled = sampled.sample(n=180_000, random_state=seed)
    return sampled.reset_index(drop=True)


def train_models(train_df: pd.DataFrame, features: list[str], seed: int) -> list[TrainedCandidateModel]:
    models: list[TrainedCandidateModel] = []
    x, y, weights = prepare_xy(train_df, features)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(x_scaled, y, sample_weight=weights)
        models.append(TrainedCandidateModel(f"pointwise_logistic_C{c}", "linear_pointwise_logistic", {"C": c}, model, scaler, features, "predict_proba"))

    for alpha in [0.1, 1.0, 10.0]:
        model = RidgeClassifier(alpha=alpha, class_weight="balanced", random_state=seed)
        model.fit(x_scaled, y, sample_weight=weights)
        models.append(TrainedCandidateModel(f"pointwise_ridge_alpha{alpha}", "linear_pointwise_ridge", {"alpha": alpha}, model, scaler, features, "decision_function"))

    tree_df = pointwise_tree_training_sample(train_df, seed=seed)
    tree_x, tree_y, tree_weights = prepare_xy(tree_df, features)
    for max_iter in [80, 160]:
        try:
            model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.06, l2_regularization=0.05, random_state=seed)
            model.fit(tree_x, tree_y, sample_weight=tree_weights)
            models.append(TrainedCandidateModel(f"pointwise_hist_gbdt_iter{max_iter}", "gbdt_pointwise_hist", {"max_iter": max_iter}, model, None, features, "predict_proba"))
        except (PermissionError, OSError, RuntimeError) as exc:
            print(f"[WARN] skip HistGradientBoostingClassifier max_iter={max_iter}: {exc}")

    try:
        model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=2, random_state=seed)
        model.fit(tree_x, tree_y, sample_weight=tree_weights)
        models.append(TrainedCandidateModel("pointwise_gbdt_depth2", "gbdt_pointwise", {"n_estimators": 80, "max_depth": 2}, model, None, features, "predict_proba"))
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"[WARN] skip GradientBoostingClassifier: {exc}")

    pair_x, pair_y = build_pairwise_examples(train_df, features)
    pair_scaler = StandardScaler()
    pair_x_scaled = pair_scaler.fit_transform(pair_x)
    for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(pair_x_scaled, pair_y)
        models.append(TrainedCandidateModel(f"pairwise_linear_C{c}", "pairwise_linear_logistic", {"C": c}, model, pair_scaler, features, "pairwise_linear"))
    return models


def raw_model_score(df: pd.DataFrame, candidate: TrainedCandidateModel) -> np.ndarray:
    x = df[candidate.features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if candidate.scaler is not None:
        x = candidate.scaler.transform(x)
    if candidate.score_mode == "predict_proba":
        return candidate.model.predict_proba(x)[:, 1]
    if candidate.score_mode == "decision_function":
        return candidate.model.decision_function(x)
    if candidate.score_mode == "pairwise_linear":
        return x @ candidate.model.coef_.reshape(-1)
    raise ValueError(f"Unsupported score mode: {candidate.score_mode}")


def normalize_scores_by_case(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, index=df.index, dtype=float)
    mins = series.groupby(df["case_id"]).transform("min")
    maxs = series.groupby(df["case_id"]).transform("max")
    denom = (maxs - mins).replace(0, 1.0)
    return ((series - mins) / denom).to_numpy(dtype=float)


def score_with_variant(df: pd.DataFrame, candidate: TrainedCandidateModel | None, alpha: float, current_weight: float) -> pd.DataFrame:
    out = df.copy()
    current_norm = normalize_scores_by_case(out, pd.to_numeric(out["final_current_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    if candidate is None:
        model_norm = np.zeros(len(out), dtype=float)
    else:
        model_norm = normalize_scores_by_case(out, raw_model_score(out, candidate))
    hgnn_norm = normalize_scores_by_case(out, pd.to_numeric(out["hgnn_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    out["reranker_score"] = alpha * model_norm + current_weight * current_norm + max(0.0, 1.0 - alpha - current_weight) * hgnn_norm
    protected = out["dataset"].isin(PROTECTED_DATASETS)
    out.loc[protected, "reranker_score"] = current_norm[protected.to_numpy()]
    return out


def ranks_from_scored(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, case_id), group in scored.groupby(["dataset", "case_id"], sort=False):
        ordered = group.sort_values(["reranker_score", "final_current_rank", "hgnn_rank"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
        hits = ordered.index[ordered["label_is_strict_gold"].to_numpy(dtype=int) == 1].tolist()
        rank = int(hits[0] + 1) if hits else 999999
        gold = str(group["gold_mondo"].iloc[0])
        current_hits = group.loc[group["label_is_strict_gold"] == 1, "final_current_rank"]
        current_rank = int(current_hits.min()) if not current_hits.empty and pd.notna(current_hits.min()) else 999999
        rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "gold_mondo": gold,
                "current_rank": current_rank,
                "reranker_rank": rank,
            }
        )
    return pd.DataFrame(rows)


def summarize_ranks(ranks: pd.DataFrame, rank_col: str = "reranker_rank") -> pd.DataFrame:
    rows = []
    for dataset, group in ranks.groupby("dataset", sort=True):
        arr = pd.to_numeric(group[rank_col], errors="coerce").fillna(999999).to_numpy(dtype=int)
        rows.append(
            {
                "dataset": dataset,
                "num_cases": int(len(arr)),
                "top1": float(np.mean(arr <= 1)),
                "top3": float(np.mean(arr <= 3)),
                "top5": float(np.mean(arr <= 5)),
                "rank_le_50": float(np.mean(arr <= 50)),
                "median_rank": float(np.median(arr)),
                "mean_rank": float(np.mean(arr)),
            }
        )
    if rows:
        all_arr = pd.to_numeric(ranks[rank_col], errors="coerce").fillna(999999).to_numpy(dtype=int)
        rows.append(
            {
                "dataset": "ALL",
                "num_cases": int(len(all_arr)),
                "top1": float(np.mean(all_arr <= 1)),
                "top3": float(np.mean(all_arr <= 3)),
                "top5": float(np.mean(all_arr <= 5)),
                "rank_le_50": float(np.mean(all_arr <= 50)),
                "median_rank": float(np.median(all_arr)),
                "mean_rank": float(np.mean(all_arr)),
            }
        )
    return pd.DataFrame(rows)


def metric_lookup(metrics: pd.DataFrame, dataset: str, metric: str) -> float:
    row = metrics[metrics["dataset"] == dataset]
    if row.empty:
        return 0.0
    return float(row.iloc[0][metric])


def validation_objective(current: pd.DataFrame, reranked: pd.DataFrame) -> tuple[float, str]:
    def delta(dataset: str, metric: str) -> float:
        return metric_lookup(reranked, dataset, metric) - metric_lookup(current, dataset, metric)

    score = 0.0
    score += 3 * delta("mimic_rag_0425", "top1")
    score += 2 * delta("mimic_rag_0425", "top3")
    score += 3 * delta("DDD", "top1")
    score += 2 * delta("DDD", "top3")
    score += 2 * delta("DDD", "top5")
    score += 1 * delta("LIRICAL", "top1")

    penalties: list[str] = []
    mimic_top5 = metric_lookup(reranked, "mimic_rag_0425", "top5")
    if mimic_top5 < 0.39:
        penalty = 10 * (0.39 - mimic_top5)
        score -= penalty
        penalties.append(f"mimic_top5_below_0.39:{penalty:.4f}")
    for dataset in ["MME", "RAMEDIS", "MyGene2"]:
        for metric in ["top1", "top3", "top5"]:
            drop = metric_lookup(current, dataset, metric) - metric_lookup(reranked, dataset, metric)
            if drop > 1e-12:
                penalty = 5 * drop
                score -= penalty
                penalties.append(f"{dataset}_{metric}_drop_vs_current:{penalty:.4f}")
    for metric in ["top1", "top3", "top5"]:
        drop = metric_lookup(current, "ALL", metric) - metric_lookup(reranked, "ALL", metric)
        if drop > 0.01:
            penalty = 2 * drop
            score -= penalty
            penalties.append(f"ALL_{metric}_drop:{penalty:.4f}")
    reason = "无硬约束惩罚" if not penalties else "; ".join(penalties)
    return score, reason


def remaining_case_gap(dataset: str, metrics: pd.Series) -> tuple[int | None, int | None, int | None]:
    target_dataset = "mimic_test_recleaned_mondo_hpo_rows" if dataset == "mimic_rag_0425" else dataset
    if target_dataset not in DEEPRARE_TARGETS:
        return None, None, None
    targets = DEEPRARE_TARGETS[target_dataset]
    n = int(metrics["num_cases"])
    return tuple(int(math.ceil(max(target - float(metrics[f"top{k}"]), 0.0) * n)) for k, target in zip([1, 3, 5], targets, strict=True))


def target_reached_for_validation(dataset: str, metrics: pd.Series) -> bool | None:
    target_dataset = "mimic_test_recleaned_mondo_hpo_rows" if dataset == "mimic_rag_0425" else dataset
    if target_dataset not in DEEPRARE_TARGETS:
        return None
    top1, top3, top5 = DEEPRARE_TARGETS[target_dataset]
    return bool(float(metrics["top1"]) >= top1 and float(metrics["top3"]) >= top3 and float(metrics["top5"]) >= top5)


def evaluate_model_grid(val_df: pd.DataFrame, models: list[TrainedCandidateModel]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    current_scored = val_df.copy()
    current_scored["reranker_score"] = pd.to_numeric(current_scored["final_current_score"], errors="coerce").fillna(0.0)
    current_ranks = ranks_from_scored(current_scored)
    current_metrics = summarize_ranks(current_ranks)

    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    variants: list[tuple[TrainedCandidateModel | None, float, float, str]] = [(None, 0.0, 1.0, "current_rank_baseline")]
    for model in models:
        for alpha, current_weight in [(0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (1.00, 0.00)]:
            variants.append((model, alpha, current_weight, f"{model.model_key}_a{alpha}_cur{current_weight}"))

    for model, alpha, current_weight, variant_key in variants:
        scored = score_with_variant(val_df, model, alpha, current_weight)
        ranks = ranks_from_scored(scored)
        metrics = summarize_ranks(ranks)
        objective, reason = validation_objective(current_metrics, metrics)
        row: dict[str, Any] = {
            "model_key": variant_key,
            "model_type": model.model_type if model else "current_rank_baseline",
            "alpha_model": alpha,
            "weight_current": current_weight,
            "validation_objective": objective,
            "validation_selected_reason": reason,
        }
        for dataset in sorted(metrics["dataset"].unique()):
            mrow = metrics[metrics["dataset"] == dataset].iloc[0]
            for metric in ["num_cases", "top1", "top3", "top5", "rank_le_50"]:
                row[f"{dataset}_{metric}"] = mrow[metric]
            gaps = remaining_case_gap(dataset, mrow)
            if gaps[0] is not None:
                row[f"{dataset}_top1_case_gap"] = gaps[0]
                row[f"{dataset}_top3_case_gap"] = gaps[1]
                row[f"{dataset}_top5_case_gap"] = gaps[2]
            reached = target_reached_for_validation(dataset, mrow)
            if reached is not None:
                row[f"{dataset}_target_reached"] = reached
        gap_cols = [key for key in row if key.endswith("_case_gap")]
        reached_cols = [key for key in row if key.endswith("_target_reached")]
        row["validation_remaining_case_gap_total"] = int(sum(int(row[key]) for key in gap_cols if pd.notna(row[key])))
        row["validation_target_reached_count"] = int(sum(bool(row[key]) for key in reached_cols))
        rows.append(row)
        if selected is None or objective > selected["validation_objective"]:
            selected = {
                "model": model,
                "alpha_model": alpha,
                "weight_current": current_weight,
                "model_key": variant_key,
                "validation_objective": objective,
                "validation_selected_reason": reason,
                "validation_metrics": metrics,
            }

    if selected is None:
        raise RuntimeError("No validation model selected.")
    return pd.DataFrame(rows).sort_values("validation_objective", ascending=False), selected, current_metrics


def save_selected_model(selected: dict[str, Any], features: list[str]) -> None:
    model = selected["model"]
    payload = {
        "protocol": "train split fit; validation split model selection; test fixed evaluation once",
        "model_key": selected["model_key"],
        "model_type": model.model_type if model else "current_rank_baseline",
        "hyperparams": model.hyperparams if model else {},
        "alpha_model": selected["alpha_model"],
        "weight_current": selected["weight_current"],
        "protected_datasets": sorted(PROTECTED_DATASETS),
        "targeted_datasets": sorted(TARGETED_DATASETS),
        "features": features,
        "model": model.model if model else None,
        "scaler": model.scaler if model else None,
        "score_mode": model.score_mode if model else "current",
    }
    with (OUTPUT_DIR / "selected_model.pkl").open("wb") as f:
        pickle.dump(payload, f)
    config = {k: v for k, v in payload.items() if k not in {"model", "scaler"}}
    config["validation_objective"] = selected["validation_objective"]
    config["validation_selected_reason"] = selected["validation_selected_reason"]
    (OUTPUT_DIR / "selected_model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_validation_selected_md(selected: dict[str, Any], current_metrics: pd.DataFrame) -> None:
    lines = [
        "# Validation Selected Model",
        "",
        f"- selected_model: `{selected['model_key']}`",
        f"- objective: {selected['validation_objective']:.6f}",
        f"- selected_reason: {selected['validation_selected_reason']}",
        "- selection protocol: train split fit; validation split selection; test fixed evaluation only.",
        "- gold-relative MONDO relation flags are used for hard negative construction/audit, not as scoring features.",
        "",
        "## Validation Current Metrics",
        markdown_table(current_metrics),
        "",
        "## Validation Selected Metrics",
        markdown_table(selected["validation_metrics"]),
    ]
    write_text(REPORT_DIR / "validation_selected_model.md", "\n".join(lines))


def fixed_test_results(test_ranks: pd.DataFrame) -> pd.DataFrame:
    rerank_metrics = summarize_ranks(test_ranks, "reranker_rank")
    current_metrics = pd.read_csv(FINAL_METRICS, dtype={"dataset": str})
    current_metrics = current_metrics.rename(
        columns={
            "cases": "current_num_cases",
            "top1": "current_top1",
            "top3": "current_top3",
            "top5": "current_top5",
            "rank_le_50": "current_rank_le_50",
        }
    )
    rows: list[dict[str, Any]] = []
    for _, rerow in rerank_metrics.iterrows():
        dataset = str(rerow["dataset"])
        if dataset == "ALL":
            cur = current_metrics[current_metrics["dataset"] == "ALL"]
        else:
            cur = current_metrics[current_metrics["dataset"] == dataset]
        if cur.empty:
            continue
        currow = cur.iloc[0]
        target = DEEPRARE_TARGETS.get(dataset)
        row = {
            "dataset": dataset,
            "num_cases": int(rerow["num_cases"]),
            "current_top1": float(currow["current_top1"]),
            "reranker_top1": float(rerow["top1"]),
            "target_top1": target[0] if target else np.nan,
            "delta_top1": float(rerow["top1"]) - float(currow["current_top1"]),
            "current_top3": float(currow["current_top3"]),
            "reranker_top3": float(rerow["top3"]),
            "target_top3": target[1] if target else np.nan,
            "delta_top3": float(rerow["top3"]) - float(currow["current_top3"]),
            "current_top5": float(currow["current_top5"]),
            "reranker_top5": float(rerow["top5"]),
            "target_top5": target[2] if target else np.nan,
            "delta_top5": float(rerow["top5"]) - float(currow["current_top5"]),
            "current_rank_le_50": float(currow["current_rank_le_50"]),
            "reranker_rank_le_50": float(rerow["rank_le_50"]),
            "delta_rank_le_50": float(rerow["rank_le_50"]) - float(currow["current_rank_le_50"]),
        }
        if target:
            row["case_gap_before_top1"] = int(math.ceil(max(target[0] - row["current_top1"], 0) * row["num_cases"]))
            row["case_gap_after_top1"] = int(math.ceil(max(target[0] - row["reranker_top1"], 0) * row["num_cases"]))
            row["case_gap_before_top3"] = int(math.ceil(max(target[1] - row["current_top3"], 0) * row["num_cases"]))
            row["case_gap_after_top3"] = int(math.ceil(max(target[1] - row["reranker_top3"], 0) * row["num_cases"]))
            row["case_gap_before_top5"] = int(math.ceil(max(target[2] - row["current_top5"], 0) * row["num_cases"]))
            row["case_gap_after_top5"] = int(math.ceil(max(target[2] - row["reranker_top5"], 0) * row["num_cases"]))
            row["deeprare_target_reached"] = bool(
                row["reranker_top1"] >= target[0] and row["reranker_top3"] >= target[1] and row["reranker_top5"] >= target[2]
            )
        else:
            row["deeprare_target_reached"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def write_fixed_test_md(results: pd.DataFrame) -> None:
    focus = results[results["dataset"].isin(["mimic_test_recleaned_mondo_hpo_rows", "DDD", "LIRICAL", "HMS", "MME", "RAMEDIS", "MyGene2"])].copy()
    lines = [
        "# Fixed Test Results",
        "",
        "这是 validation selected light reranker 的 fixed test evaluation。test 未用于模型、权重或规则选择。",
        "",
        "## Current Mainline vs Light Reranker",
        markdown_table(focus),
        "",
        "## 重点数据集分析",
    ]
    by_dataset = {row["dataset"]: row for _, row in results.iterrows()}
    mimic = by_dataset.get("mimic_test_recleaned_mondo_hpo_rows")
    if mimic is not None:
        lines.append(
            f"- MIMIC-IV-Rare: Top1 {mimic['reranker_top1']:.4f} vs target 0.29; "
            f"Top3 {mimic['reranker_top3']:.4f} vs target 0.37; "
            f"Top5 {mimic['reranker_top5']:.4f}，{'保持' if mimic['reranker_top5'] >= 0.39 else '未保持'} >= 0.39。"
        )
    ddd = by_dataset.get("DDD")
    if ddd is not None:
        lines.append(
            f"- DDD: Top1/Top3/Top5 = {ddd['reranker_top1']:.4f}/{ddd['reranker_top3']:.4f}/{ddd['reranker_top5']:.4f}; "
            "target = 0.48/0.60/0.63。"
        )
    lirical = by_dataset.get("LIRICAL")
    if lirical is not None:
        lines.append(
            f"- LIRICAL: after case gap Top1/Top3/Top5 = "
            f"{int(lirical['case_gap_after_top1'])}/{int(lirical['case_gap_after_top3'])}/{int(lirical['case_gap_after_top5'])}。"
        )
    hms = by_dataset.get("HMS")
    if hms is not None:
        lines.append(
            f"- HMS: Top1/Top3/Top5 = {hms['reranker_top1']:.4f}/{hms['reranker_top3']:.4f}/{hms['reranker_top5']:.4f}；样本量小，标注 high variance。"
        )
    protected = [name for name in ["MME", "RAMEDIS", "MyGene2"] if name in by_dataset]
    for name in protected:
        row = by_dataset[name]
        lines.append(
            f"- {name}: reranker Top1/Top3/Top5 = {row['reranker_top1']:.4f}/{row['reranker_top3']:.4f}/{row['reranker_top5']:.4f}，"
            f"{'仍达标' if row['deeprare_target_reached'] else '未达标'}。"
        )
    write_text(REPORT_DIR / "fixed_test_results.md", "\n".join(lines))


def build_case_delta(test_ranks: pd.DataFrame) -> pd.DataFrame:
    final = pd.read_csv(FINAL_CASE_RANKS, dtype={"case_id": str, "dataset": str, "gold_id": str})
    final = final[["case_id", "dataset", "gold_id", "final_rank", "module_applied"]].rename(
        columns={"gold_id": "gold_mondo", "final_rank": "current_mainline_rank"}
    )
    merged = final.merge(test_ranks[["case_id", "reranker_rank"]], on="case_id", how="left")
    merged["reranker_rank"] = pd.to_numeric(merged["reranker_rank"], errors="coerce").fillna(999999).astype(int)
    merged["current_mainline_rank"] = pd.to_numeric(merged["current_mainline_rank"], errors="coerce").fillna(999999).astype(int)
    for k in [1, 3, 5]:
        merged[f"current_top{k}"] = merged["current_mainline_rank"] <= k
        merged[f"reranker_top{k}"] = merged["reranker_rank"] <= k
        merged[f"top{k}_delta"] = merged[f"reranker_top{k}"].astype(int) - merged[f"current_top{k}"].astype(int)
    merged["rank_delta"] = merged["current_mainline_rank"] - merged["reranker_rank"]
    merged["rank_change"] = np.where(merged["rank_delta"] > 0, "improved", np.where(merged["rank_delta"] < 0, "worsened", "unchanged"))
    return merged


def write_case_delta_summary(delta: pd.DataFrame) -> None:
    rows = []
    for dataset, group in delta.groupby("dataset", sort=True):
        row = {"dataset": dataset, "num_cases": int(len(group))}
        for k in [1, 3, 5]:
            row[f"top{k}_gained_cases"] = int((group[f"top{k}_delta"] == 1).sum())
            row[f"top{k}_lost_cases"] = int((group[f"top{k}_delta"] == -1).sum())
        counts = group["rank_change"].value_counts()
        row["rank_improved"] = int(counts.get("improved", 0))
        row["rank_worsened"] = int(counts.get("worsened", 0))
        row["rank_unchanged"] = int(counts.get("unchanged", 0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    write_csv(summary, REPORT_DIR / "case_level_delta_summary.csv")
    mimic = delta[delta["dataset"] == "mimic_test_recleaned_mondo_hpo_rows"]
    ddd = delta[delta["dataset"] == "DDD"]
    protected = delta[delta["dataset"].isin(PROTECTED_DATASETS)]
    lines = [
        "# Case-Level Delta Summary",
        "",
        markdown_table(summary),
        "",
        "## Focus Counts",
        f"- MIMIC final rank 2-5 -> rank1: {int(((mimic['current_mainline_rank'].between(2, 5)) & (mimic['reranker_rank'] <= 1)).sum())}",
        f"- MIMIC final rank 4-5 -> rank<=3: {int(((mimic['current_mainline_rank'].between(4, 5)) & (mimic['reranker_rank'] <= 3)).sum())}",
        f"- DDD final rank 6-50 -> top5: {int(((ddd['current_mainline_rank'].between(6, 50)) & (ddd['reranker_rank'] <= 5)).sum())}",
        f"- DDD final rank 2-5 -> top1: {int(((ddd['current_mainline_rank'].between(2, 5)) & (ddd['reranker_rank'] <= 1)).sum())}",
        f"- 已达标数据集 top5 lost cases: {int((protected['top5_delta'] == -1).sum())}",
    ]
    write_text(REPORT_DIR / "case_level_delta_summary.md", "\n".join(lines))


def bootstrap_ci(delta: pd.DataFrame, iters: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for dataset, group in delta.groupby("dataset", sort=True):
        n = len(group)
        if n == 0:
            continue
        current = group["current_mainline_rank"].to_numpy(dtype=int)
        rerank = group["reranker_rank"].to_numpy(dtype=int)
        samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(iters):
            idx = rng.integers(0, n, size=n)
            for k in [1, 3, 5]:
                cur_val = float(np.mean(current[idx] <= k))
                rer_val = float(np.mean(rerank[idx] <= k))
                samples[f"top{k}"].append(rer_val)
                samples[f"delta_top{k}"].append(rer_val - cur_val)
        for metric, values in samples.items():
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "mean": float(arr.mean()),
                    "ci95_low": float(np.quantile(arr, 0.025)),
                    "ci95_high": float(np.quantile(arr, 0.975)),
                    "num_cases": n,
                    "high_variance": bool(n < 60),
                }
            )
    return pd.DataFrame(rows)


def write_bootstrap_md(ci: pd.DataFrame) -> None:
    lines = [
        "# Bootstrap 95% CI",
        "",
        "CI 基于 selected light reranker 的 fixed test case-level ranks。HMS、LIRICAL、MME、MyGene2 因样本量小需按 high_variance 解释。",
        "",
        markdown_table(ci),
        "",
        "## 稳定性判断",
    ]
    for dataset in ["mimic_test_recleaned_mondo_hpo_rows", "DDD"]:
        sub = ci[(ci["dataset"] == dataset) & (ci["metric"].str.startswith("delta_top"))]
        stable = sub[sub["ci95_low"] > 0]["metric"].tolist()
        lines.append(f"- {dataset}: 95% CI 下界大于 0 的 delta 指标：{', '.join(stable) if stable else 'none'}。")
    write_text(REPORT_DIR / "bootstrap_ci.md", "\n".join(lines))


def write_paper_recommendation(results: pd.DataFrame) -> None:
    reached = results[(results["dataset"] != "ALL") & (results["deeprare_target_reached"] == True)]["dataset"].tolist()
    not_reached = results[(results["dataset"] != "ALL") & (results["deeprare_target_reached"] == False)]["dataset"].tolist()
    protected_drop = results[
        results["dataset"].isin(PROTECTED_DATASETS)
        & ((results["delta_top1"] < -1e-12) | (results["delta_top3"] < -1e-12) | (results["delta_top5"] < -1e-12))
    ]
    mimic = results[results["dataset"] == "mimic_test_recleaned_mondo_hpo_rows"].iloc[0]
    ddd = results[results["dataset"] == "DDD"].iloc[0]
    can_main = bool(
        protected_drop.empty
        and mimic["reranker_top5"] >= 0.39
        and (mimic["delta_top1"] > 0 or mimic["delta_top3"] > 0)
        and ddd["delta_top1"] >= 0
        and ddd["delta_top3"] >= 0
        and ddd["delta_top5"] >= 0
    )
    lines = [
        "# Recommended Paper Table",
        "",
        f"1. current mainline 仍应作为主表 baseline：是。",
        f"2. light reranker 是否可作为主表新方法：{'可以，作为不改 encoder 的 lightweight reranker' if can_main else '暂不建议作为主表新方法；本次更适合作为附表/负结果分析'}。",
        f"3. 是否达到 DeepRare target：达到的数据集为 {', '.join(reached) if reached else 'none'}。",
        f"4. 达到 target 的 dataset：{', '.join(reached) if reached else 'none'}。",
        f"5. 仍未达到 target 的 dataset：{', '.join(not_reached) if not_reached else 'none'}。",
        f"6. 是否牺牲已达标数据集：{'否' if protected_drop.empty else '是，需要回滚或加强 protection'}。",
        "7. gated rerank 建议保留为附表，不与 strict exact 主结果混写。",
        "8. any-label / relaxed MONDO 只能 supplementary，不能替代 strict exact。",
        "9. 图对比学习仍后置。",
        f"10. 如果 DDD 仍未达标：{'是，下一步进入 ontology-aware hard negative training' if 'DDD' in not_reached else 'DDD 已达标，可暂不进入'}。",
    ]
    write_text(REPORT_DIR / "recommended_paper_table.md", "\n".join(lines))


def write_next_after_failed(results: pd.DataFrame) -> None:
    by_dataset = {row["dataset"]: row for _, row in results.iterrows()}
    mimic = by_dataset.get("mimic_test_recleaned_mondo_hpo_rows")
    ddd = by_dataset.get("DDD")
    failed = False
    if mimic is not None and (mimic["reranker_top1"] < 0.29 or mimic["reranker_top3"] < 0.37):
        failed = True
    if ddd is not None and (ddd["reranker_top1"] < 0.48 or ddd["reranker_top3"] < 0.60 or ddd["reranker_top5"] < 0.63):
        failed = True
    if not failed:
        return
    lines = [
        "# Next After Failed Target",
        "",
        "- MIMIC：如果 Top1 < 0.29 或 Top3 < 0.37，需要更强 top1-oriented pairwise/listwise reranker，并保留 Top5 >= 0.39 约束。",
        "- DDD：如果 Top1/Top3/Top5 仍未达到 0.48/0.60/0.63，应进入 ontology-aware hard negative training。",
        "- LIRICAL：优先做 outlier/mapping 修复，不建议盲目训练。",
        "- HMS：只做 high-variance 附表观察。",
        "- 图对比学习仍为 P4，继续后置。",
    ]
    write_text(REPORT_DIR / "next_after_failed_target.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_train_raw_candidates(force=args.force_rebuild_candidates)
    mondo = load_mondo_resource(MONDO_JSON)

    train_df = build_candidate_table("train", TRAIN_RAW, mondo)
    val_df = build_candidate_table("validation", VAL_RAW, mondo)
    test_df = build_candidate_table("test", TEST_RAW, mondo)
    write_csv(train_df, OUTPUT_DIR / "train_candidates.csv")
    write_csv(val_df, OUTPUT_DIR / "validation_candidates.csv")
    write_csv(test_df, OUTPUT_DIR / "test_candidates.csv")

    features = feature_columns(train_df)
    models = train_models(train_df, features, args.random_seed)
    grid, selected, val_current_metrics = evaluate_model_grid(val_df, models)
    write_csv(grid, REPORT_DIR / "validation_model_grid.csv")
    write_validation_selected_md(selected, val_current_metrics)
    save_selected_model(selected, features)

    selected_scored = score_with_variant(test_df, selected["model"], selected["alpha_model"], selected["weight_current"])
    test_ranks = ranks_from_scored(selected_scored)
    write_csv(test_ranks, OUTPUT_DIR / "fixed_test_case_ranks.csv")

    results = fixed_test_results(test_ranks)
    write_csv(results, REPORT_DIR / "fixed_test_results.csv")
    write_fixed_test_md(results)

    delta = build_case_delta(test_ranks)
    write_csv(delta, REPORT_DIR / "case_level_delta.csv")
    write_case_delta_summary(delta)

    ci = bootstrap_ci(delta, args.bootstrap_iters, args.random_seed)
    write_csv(ci, REPORT_DIR / "bootstrap_ci.csv")
    write_bootstrap_md(ci)

    write_paper_recommendation(results)
    write_next_after_failed(results)

    manifest = {
        "train_candidates": str((OUTPUT_DIR / "train_candidates.csv").resolve()),
        "validation_candidates": str((OUTPUT_DIR / "validation_candidates.csv").resolve()),
        "test_candidates": str((OUTPUT_DIR / "test_candidates.csv").resolve()),
        "selected_model": str((OUTPUT_DIR / "selected_model.pkl").resolve()),
        "selected_model_config": str((OUTPUT_DIR / "selected_model_config.json").resolve()),
        "validation_selected_model": selected["model_key"],
        "validation_objective": selected["validation_objective"],
        "fixed_test_results": str((REPORT_DIR / "fixed_test_results.csv").resolve()),
        "bootstrap_iters": args.bootstrap_iters,
        "strict_exact_main_result": True,
        "test_side_tuning": False,
        "protected_datasets": sorted(PROTECTED_DATASETS),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
