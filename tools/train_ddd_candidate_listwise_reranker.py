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

from src.data.dataset import load_case_files
from src.evaluation.evaluator import load_yaml_config
from src.training.trainer import resolve_train_files, split_train_val_by_case
from tools.train_ddd_ontology_hard_negative import (
    DDD_TARGET,
    DDD_WEIGHTS,
    FINAL_CASE_RANKS,
    FINAL_METRICS,
    HYPEREDGE_CSV,
    MONDO_JSON,
    TEST_RAW,
    TRAIN_CONFIG,
    TRAIN_RAW,
    VAL_RAW,
    hpo_set_similarity,
    load_ddd_weights,
    load_disease_hpos,
    load_mondo_resource,
    metric_from_ranks,
    minmax_by_case,
    rank_bucket,
)
from tools.train_deeprare_target_light_reranker import markdown_table, relation_to_gold, write_csv, write_text


REPORT_DIR = PROJECT_ROOT / "reports" / "ddd_candidate_listwise_reranker"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ddd_candidate_listwise_reranker"
FAILED_DIR = PROJECT_ROOT / "reports" / "ddd_ontology_hard_negative"
FAILED_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ddd_ontology_hard_negative"

DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DDD_TRAIN_FILE = PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "train" / "DDD.csv"
DDD_TEST_FILE = PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "test" / "DDD.csv"

BASE_FEATURES = [
    "current_score",
    "current_rank_recip",
    "hgnn_score",
    "hgnn_rank_recip",
    "failed_pairwise_score",
    "exact_hpo_overlap_count",
    "exact_hpo_overlap_ratio",
    "ic_weighted_overlap",
    "semantic_overlap",
    "case_coverage",
    "disease_coverage",
    "jaccard_overlap",
    "candidate_source_hgnn",
    "candidate_source_mondo_expansion",
    "candidate_source_hpo_expansion",
    "candidate_source_similar_case",
    "candidate_source_count",
    "case_hpo_count",
    "disease_hpo_count",
    "log1p_case_hpo_count",
    "log1p_disease_hpo_count",
    "max_exact_overlap_in_case",
    "max_ic_overlap_in_case",
    "evidence_rank_by_ic_recip",
]


@dataclass(slots=True)
class SelectedModel:
    model_key: str
    model_type: str
    model: Any
    scaler: StandardScaler | None
    score_mode: str
    alpha_model: float
    weight_current: float
    hyperparams: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDD candidate expansion + listwise/top-k-aware reranker.")
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-expanded-candidates", type=int, default=220)
    return parser.parse_args()


def read_csv_typed(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})


def load_case_hpos() -> tuple[dict[str, set[str]], dict[str, str]]:
    data_cfg = load_yaml_config(DATA_CONFIG)
    train_cfg = load_yaml_config(TRAIN_CONFIG)
    case_id_col = str(data_cfg.get("case_id_col", "case_id"))
    label_col = str(data_cfg.get("label_col", "mondo_label"))
    hpo_col = str(data_cfg.get("hpo_col", "hpo_id"))
    train_files = [path for path in resolve_train_files(train_cfg["paths"]) if Path(path).name == "DDD.csv"]
    train_all = load_case_files(
        file_paths=[str(path) for path in train_files],
        case_id_col=case_id_col,
        label_col=label_col,
        disease_index_path=train_cfg["paths"]["disease_index_path"],
        split_namespace="train",
    )
    train_df, val_df = split_train_val_by_case(
        train_all,
        val_ratio=float(train_cfg["data"]["val_ratio"]),
        random_seed=int(train_cfg["data"]["random_seed"]),
        case_id_col=case_id_col,
    )
    test_df = load_case_files(
        file_paths=[str(DDD_TEST_FILE)],
        case_id_col=case_id_col,
        label_col=label_col,
        disease_index_path=train_cfg["paths"]["disease_index_path"],
        split_namespace="test",
    )
    case_hpos: dict[str, set[str]] = {}
    case_gold: dict[str, str] = {}
    for df in [train_df, val_df, test_df]:
        for case_id, group in df.groupby(case_id_col, sort=False):
            case_hpos[str(case_id)] = set(group[hpo_col].dropna().astype(str))
            case_gold[str(case_id)] = str(group[label_col].iloc[0])
    return case_hpos, case_gold


def load_base_candidates(path: Path, split: str) -> pd.DataFrame:
    df = read_csv_typed(path)
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
    df["current_score"] = (
        weights.get("w_hgnn", 0.0) * minmax_by_case(df, "hgnn_score")
        + weights.get("w_ic", 0.0) * df["ic_weighted_overlap"]
        + weights.get("w_exact", 0.0) * df["exact_overlap"]
        + weights.get("w_semantic", 0.0) * df["semantic_ic_overlap"]
        + weights.get("w_case_cov", 0.0) * df["case_coverage"]
        + weights.get("w_dis_cov", 0.0) * df["disease_coverage"]
        - weights.get("w_size", 0.0) * np.log1p(df["disease_hpo_count"])
    )
    df["current_rank"] = (
        df.sort_values(["case_id", "current_score", "original_rank"], ascending=[True, False, True], kind="stable")
        .groupby("case_id")
        .cumcount()
        + 1
    )
    df["source_hgnn"] = 1
    return df


def mondo_neighbor_candidates(seed_candidates: list[str], mondo: dict[str, Any], limit: int = 80) -> list[str]:
    parents = mondo["parents"]
    children = mondo["children"]
    ancestors_fn = mondo["ancestors_fn"]
    out: list[str] = []
    seen: set[str] = set()
    for cand in seed_candidates[:20]:
        related = set()
        related.update(parents.get(cand, set()))
        related.update(children.get(cand, set()))
        for parent in parents.get(cand, set()):
            related.update(children.get(parent, set()))
        related.update(list(ancestors_fn(cand))[:10])
        for item in related:
            if item != cand and item not in seen:
                seen.add(item)
                out.append(item)
                if len(out) >= limit:
                    return out
    return out


def build_hpo_inverted(disease_hpos: dict[str, set[str]]) -> dict[str, set[str]]:
    inv: dict[str, set[str]] = defaultdict(set)
    for disease, hpos in disease_hpos.items():
        for hpo in hpos:
            inv[hpo].add(disease)
    return inv


def hpo_retrieval(case_hpos: set[str], disease_hpos: dict[str, set[str]], inverted: dict[str, set[str]], limit: int = 220) -> list[tuple[str, float, int]]:
    counts: Counter[str] = Counter()
    for hpo in case_hpos:
        counts.update(inverted.get(hpo, set()))
    scored = []
    case_n = max(len(case_hpos), 1)
    for disease, shared in counts.items():
        dis_n = max(len(disease_hpos.get(disease, set())), 1)
        score = float(shared / math.sqrt(case_n * dis_n))
        scored.append((disease, score, int(shared)))
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored[:limit]


def similar_case_candidates(case_id: str, case_hpos: set[str], train_case_hpos: dict[str, set[str]], train_case_gold: dict[str, str], limit: int = 40) -> list[tuple[str, float]]:
    rows = []
    for other_id, other_hpos in train_case_hpos.items():
        if other_id == case_id:
            continue
        score = hpo_set_similarity(case_hpos, other_hpos)
        if score > 0:
            rows.append((train_case_gold[other_id], float(score)))
    rows.sort(key=lambda x: (-x[1], x[0]))
    deduped = []
    seen = set()
    for mondo, score in rows:
        if mondo in seen:
            continue
        seen.add(mondo)
        deduped.append((mondo, score))
        if len(deduped) >= limit:
            break
    return deduped


def failed_pairwise_scores() -> dict[tuple[str, str], float]:
    path = FAILED_OUTPUT_DIR / "fixed_test_case_ranks.csv"
    # The failed model did not persist candidate scores; keep interface explicit.
    return {}


def make_expanded_candidates(
    base: pd.DataFrame,
    split: str,
    mondo: dict[str, Any],
    case_hpos: dict[str, set[str]],
    case_gold: dict[str, str],
    disease_hpos: dict[str, set[str]],
    inverted: dict[str, set[str]],
    train_case_hpos: dict[str, set[str]],
    train_case_gold: dict[str, str],
    max_candidates: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_by_case = {case_id: group.copy() for case_id, group in base.groupby("case_id", sort=False)}
    for case_id, group in base_by_case.items():
        current_candidates = group.sort_values("current_rank", kind="stable")["candidate_id"].astype(str).tolist()
        candidate_sources: dict[str, set[str]] = defaultdict(set)
        source_scores: dict[tuple[str, str], float] = {}
        source_shared: dict[tuple[str, str], int] = {}
        for cand in current_candidates:
            candidate_sources[cand].add("hgnn")
        for cand in mondo_neighbor_candidates(current_candidates, mondo):
            candidate_sources[cand].add("mondo_expansion")
        hpo_rows = hpo_retrieval(case_hpos.get(case_id, set()), disease_hpos, inverted, limit=220)
        for cand, score, shared in hpo_rows:
            candidate_sources[cand].add("hpo_expansion")
            source_scores[(cand, "hpo")] = score
            source_shared[(cand, "hpo")] = shared
        for cand, score in similar_case_candidates(case_id, case_hpos.get(case_id, set()), train_case_hpos, train_case_gold):
            candidate_sources[cand].add("similar_case")
            source_scores[(cand, "similar")] = score

        gold = case_gold.get(case_id, str(group["gold_id"].iloc[0]))
        base_lookup = {str(row.candidate_id): row for row in group.itertuples(index=False)}
        ordered_candidates = sorted(
            candidate_sources,
            key=lambda cand: (
                0 if "hgnn" in candidate_sources[cand] else 1,
                base_lookup[cand].current_rank if cand in base_lookup else 999,
                -source_scores.get((cand, "hpo"), 0.0),
                cand,
            ),
        )[:max_candidates]
        for cand in ordered_candidates:
            base_row = base_lookup.get(cand)
            cand_hpos = disease_hpos.get(cand, set())
            case_set = case_hpos.get(case_id, set())
            shared = int(len(case_set & cand_hpos))
            exact_ratio = float(shared / math.sqrt(max(len(case_set), 1) * max(len(cand_hpos), 1))) if cand_hpos else 0.0
            hpo_score = source_scores.get((cand, "hpo"), exact_ratio)
            row = {
                "split": split,
                "dataset": "DDD",
                "case_id": case_id,
                "gold_mondo": gold,
                "candidate_mondo": cand,
                "label_is_gold": int(cand == gold),
                "hgnn_score": float(getattr(base_row, "hgnn_score", 0.0)) if base_row is not None else 0.0,
                "hgnn_rank": int(getattr(base_row, "original_rank", 999)) if base_row is not None else 999,
                "current_score": float(getattr(base_row, "current_score", 0.0)) if base_row is not None else hpo_score * 0.25,
                "current_rank": int(getattr(base_row, "current_rank", 999)) if base_row is not None else 999,
                "failed_pairwise_score": 0.0,
                "exact_hpo_overlap_count": shared,
                "exact_hpo_overlap_ratio": exact_ratio,
                "ic_weighted_overlap": float(getattr(base_row, "ic_weighted_overlap", 0.0)) if base_row is not None else hpo_score,
                "semantic_overlap": float(getattr(base_row, "semantic_ic_overlap", 0.0)) if base_row is not None else hpo_score,
                "case_coverage": float(shared / max(len(case_set), 1)),
                "disease_coverage": float(shared / max(len(cand_hpos), 1)) if cand_hpos else 0.0,
                "jaccard_overlap": float(shared / max(len(case_set | cand_hpos), 1)) if cand_hpos else 0.0,
                "candidate_source_hgnn": int("hgnn" in candidate_sources[cand]),
                "candidate_source_mondo_expansion": int("mondo_expansion" in candidate_sources[cand]),
                "candidate_source_hpo_expansion": int("hpo_expansion" in candidate_sources[cand]),
                "candidate_source_similar_case": int("similar_case" in candidate_sources[cand]),
                "candidate_source_count": len(candidate_sources[cand]),
                "case_hpo_count": int(len(case_set)),
                "disease_hpo_count": int(len(cand_hpos)),
                "obsolete_label_flag": int(cand in mondo["deprecated"]),
            }
            pc, sib, shared_anc = relation_to_gold(cand, gold, mondo)
            row["mondo_parent_child_flag"] = pc
            row["mondo_sibling_flag"] = sib
            row["mondo_shared_ancestor_score"] = shared_anc
            row["synonym_or_replacement_flag"] = int(mondo["replacements"].get(cand) == gold or mondo["replacements"].get(gold) == cand)
            row["disease_hyperedge_similarity_to_gold"] = hpo_set_similarity(cand_hpos, disease_hpos.get(gold, set()))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["hgnn_rank_recip"] = 1.0 / pd.to_numeric(out["hgnn_rank"], errors="coerce").replace(0, np.nan).fillna(999)
    out["current_rank_recip"] = 1.0 / pd.to_numeric(out["current_rank"], errors="coerce").replace(0, np.nan).fillna(999)
    out["log1p_case_hpo_count"] = np.log1p(pd.to_numeric(out["case_hpo_count"], errors="coerce").fillna(0.0))
    out["log1p_disease_hpo_count"] = np.log1p(pd.to_numeric(out["disease_hpo_count"], errors="coerce").fillna(0.0))
    out["max_exact_overlap_in_case"] = out.groupby("case_id")["exact_hpo_overlap_count"].transform("max")
    out["max_ic_overlap_in_case"] = out.groupby("case_id")["ic_weighted_overlap"].transform("max")
    out["evidence_rank_by_ic"] = (
        out.sort_values(["case_id", "ic_weighted_overlap", "exact_hpo_overlap_count"], ascending=[True, False, False], kind="stable")
        .groupby("case_id")
        .cumcount()
        + 1
    )
    out["evidence_rank_by_ic_recip"] = 1.0 / out["evidence_rank_by_ic"].replace(0, np.nan).fillna(999)
    return out.sort_values(["case_id", "current_rank", "hgnn_rank", "candidate_mondo"], kind="stable").reset_index(drop=True)


def rank_from_candidates(df: pd.DataFrame, score_col: str, case_col: str = "case_id") -> pd.DataFrame:
    rows = []
    for case_id, group in df.groupby(case_col, sort=False):
        ordered = group.sort_values([score_col, "current_rank", "hgnn_rank"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
        hits = np.flatnonzero(ordered["label_is_gold"].to_numpy(dtype=int) == 1)
        rank = int(hits[0] + 1) if len(hits) else 999999
        current = group[group["label_is_gold"] == 1]["current_rank"]
        current_rank = int(current.min()) if not current.empty else 999999
        rows.append({"case_id": case_id, "current_rank_internal": current_rank, "new_rank": rank})
    return pd.DataFrame(rows)


def audit_expansion(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        for limit in [50, 100, 200]:
            ranks = []
            for _, group in df.groupby("case_id", sort=False):
                ordered = group.sort_values(
                    ["candidate_source_hgnn", "current_rank", "ic_weighted_overlap"],
                    ascending=[False, True, False],
                    kind="stable",
                ).head(limit)
                ranks.append(bool(ordered["label_is_gold"].max()))
            current_recall = float(df[df["candidate_source_hgnn"].eq(1)].groupby("case_id")["label_is_gold"].max().mean())
            rows.append(
                {
                    "split": split,
                    "candidate_limit": limit,
                    "num_cases": int(df["case_id"].nunique()),
                    "current_top50_recall": current_recall,
                    "expanded_recall": float(np.mean(ranks)),
                    "avg_candidates_per_case": float(df.groupby("case_id")["candidate_mondo"].nunique().mean()),
                    "gold_recovered_beyond_top50_cases": int(
                        sum(
                            (not bool(group[group["candidate_source_hgnn"].eq(1)]["label_is_gold"].max()))
                            and bool(group["label_is_gold"].max())
                            for _, group in df.groupby("case_id", sort=False)
                        )
                    ),
                    "hgnn_source_rows": int(df["candidate_source_hgnn"].sum()),
                    "mondo_expansion_rows": int(df["candidate_source_mondo_expansion"].sum()),
                    "hpo_expansion_rows": int(df["candidate_source_hpo_expansion"].sum()),
                    "similar_case_rows": int(df["candidate_source_similar_case"].sum()),
                }
            )
    return pd.DataFrame(rows)


def write_expansion_audit(audit: pd.DataFrame) -> None:
    write_csv(audit, REPORT_DIR / "ddd_candidate_expansion_audit.csv")
    lines = [
        "# DDD Candidate Expansion Audit",
        "",
        markdown_table(audit),
        "",
        "## 判断",
        "- expansion 保留 current top50，不丢弃已有候选。",
        "- 是否值得进入 listwise reranker：只有当 validation expanded recall 高于 current top50 recall，且新增候选没有显著稀释 gold 排序时才值得。",
        "- test expansion 只用于 fixed evaluation，不参与 topK/source/weight 选择。",
    ]
    write_text(REPORT_DIR / "ddd_candidate_expansion_audit.md", "\n".join(lines))


def final_mainline_ddd_ranks() -> pd.DataFrame:
    df = pd.read_csv(FINAL_CASE_RANKS, dtype={"case_id": str, "dataset": str})
    out = df[df["dataset"].eq("DDD")][["case_id", "final_rank"]].rename(columns={"final_rank": "current_rank"})
    out["current_rank"] = pd.to_numeric(out["current_rank"], errors="coerce").fillna(999999).astype(int)
    return out


def failed_pairwise_ranks() -> pd.DataFrame:
    path = FAILED_OUTPUT_DIR / "fixed_test_case_ranks.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["case_id", "failed_pairwise_rank"])
    df = pd.read_csv(path, dtype={"case_id": str})
    return df[["case_id", "new_rank"]].rename(columns={"new_rank": "failed_pairwise_rank"})


def write_failed_pairwise_audit() -> None:
    delta_path = FAILED_DIR / "case_level_delta.csv"
    delta = pd.read_csv(delta_path, dtype={"case_id": str}) if delta_path.is_file() else pd.DataFrame()
    if delta.empty:
        write_text(REPORT_DIR / "failed_pairwise_audit.md", "# Failed Pairwise Audit\n\nNo failed pairwise delta file found.")
        return
    write_csv(delta, REPORT_DIR / "failed_pairwise_case_delta.csv")
    top1_gain = int((delta["top1_delta"] == 1).sum())
    top1_loss = int((delta["top1_delta"] == -1).sum())
    top3_gain = int((delta["top3_delta"] == 1).sum())
    top3_loss = int((delta["top3_delta"] == -1).sum())
    top5_gain = int((delta["top5_delta"] == 1).sum())
    top5_loss = int((delta["top5_delta"] == -1).sum())
    baseline_1_5_harmed = int((delta["current_rank"].between(1, 5)) & (delta["new_rank"] > delta["current_rank"]).sum()) if False else int(((delta["current_rank"].between(1, 5)) & (delta["new_rank"] > delta["current_rank"])).sum())
    r6_20_improved = int(((delta["current_rank"].between(6, 20)) & (delta["new_rank"] < delta["current_rank"])).sum())
    r21_50_improved = int(((delta["current_rank"].between(21, 50)) & (delta["new_rank"] < delta["current_rank"])).sum())
    type_counts = Counter()
    for text in delta.loc[delta["rank_change"].eq("worsened"), "train_like_hard_negative_types_present_in_test_top50"].fillna(""):
        for part in str(text).split("|"):
            if part:
                type_counts[part] += 1
    lines = [
        "# Failed Pairwise Head Audit",
        "",
        f"- top1 gained/lost: {top1_gain}/{top1_loss}",
        f"- top3 gained/lost: {top3_gain}/{top3_loss}",
        f"- top5 gained/lost: {top5_gain}/{top5_loss}",
        f"- baseline/current rank 1-5 被推坏病例数: {baseline_1_5_harmed}",
        f"- rank 6-20 被推好病例数: {r6_20_improved}",
        f"- rank 21-50 被推好病例数: {r21_50_improved}",
        "",
        "## 误伤类型",
        markdown_table(pd.DataFrame(type_counts.most_common(), columns=["hard_negative_like_type", "worsened_case_count"])),
        "",
        "## 结论",
        "- pairwise head 能改善少量病例，但 lost cases 多于 gained cases，说明普通 pairwise objective 会误伤已靠前病例。",
        "- sibling / same-parent 近邻可能存在 label granularity 问题，不能无差别强惩罚。",
        "- frozen features 对 DDD 仍不足，尤其无法稳定保护 current top-k。",
        "- 下一步应使用 listwise/top-k-aware objective，并加入 current-score anchoring 与 top-k protection。",
    ]
    write_text(REPORT_DIR / "failed_pairwise_audit.md", "\n".join(lines))


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BASE_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def train_pointwise_models(train_df: pd.DataFrame, seed: int) -> list[SelectedModel]:
    train = feature_frame(train_df)
    x = train[BASE_FEATURES].to_numpy(dtype=float)
    y = train["label_is_gold"].to_numpy(dtype=int)
    current_rank = pd.to_numeric(train["current_rank"], errors="coerce").fillna(999)
    sample_weight = np.where(y == 1, np.where(current_rank <= 5, 8.0, np.where(current_rank <= 50, 15.0, 20.0)), 1.0)
    models: list[SelectedModel] = []
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    for c in [0.03, 0.1, 0.3, 1.0]:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(xs, y, sample_weight=sample_weight)
        models.append(SelectedModel(f"anchored_logistic_C{c}", "current-score anchored linear reranker", model, scaler, "predict_proba", 0.0, 1.0, {"C": c}))
    for alpha_ridge in [0.1, 1.0, 10.0]:
        model = RidgeClassifier(alpha=alpha_ridge, class_weight="balanced", random_state=seed)
        model.fit(xs, y, sample_weight=sample_weight)
        models.append(SelectedModel(f"anchored_ridge_A{alpha_ridge}", "current-score anchored linear reranker", model, scaler, "decision_function", 0.0, 1.0, {"alpha": alpha_ridge}))
    # GBDT is optional; skip automatically if local sklearn threading is blocked.
    for max_iter in [80]:
        try:
            model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, l2_regularization=0.05, random_state=seed)
            model.fit(x, y, sample_weight=sample_weight)
            models.append(SelectedModel(f"hist_gbdt_iter{max_iter}", "GBDT pointwise reranker", model, None, "predict_proba", 0.0, 1.0, {"max_iter": max_iter}))
        except (PermissionError, OSError, RuntimeError) as exc:
            print(f"[WARN] skip HistGradientBoostingClassifier: {exc}")
    try:
        sample = train.sample(n=min(len(train), 120_000), random_state=seed)
        xg = sample[BASE_FEATURES].to_numpy(dtype=float)
        yg = sample["label_is_gold"].to_numpy(dtype=int)
        wg = np.where(yg == 1, 15.0, 1.0)
        model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=seed)
        model.fit(xg, yg, sample_weight=wg)
        models.append(SelectedModel("gbdt_depth2", "GBDT pointwise reranker", model, None, "predict_proba", 0.0, 1.0, {"n_estimators": 80}))
    except (PermissionError, OSError, RuntimeError, ValueError) as exc:
        print(f"[WARN] skip GradientBoostingClassifier: {exc}")
    return models


def build_lambdarank_pairs(train_df: pd.DataFrame, seed: int, max_pairs_per_case: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    weights = []
    train = feature_frame(train_df)
    for _, group in train.groupby("case_id", sort=False):
        gold = group[group["label_is_gold"].eq(1)]
        if gold.empty:
            continue
        pos = gold.iloc[0]
        pos_rank = int(pos["current_rank"])
        neg = group[group["label_is_gold"].eq(0)].copy()
        neg["pair_priority"] = 0.0
        neg.loc[neg["current_rank"] < pos_rank, "pair_priority"] += 4.0
        neg.loc[neg["current_rank"] <= 5, "pair_priority"] += 3.0
        neg["pair_priority"] += pd.to_numeric(neg["ic_weighted_overlap"], errors="coerce").fillna(0.0)
        neg = neg.sort_values(["pair_priority", "current_rank"], ascending=[False, True], kind="stable").head(max_pairs_per_case)
        pos_x = pos[BASE_FEATURES].to_numpy(dtype=float)
        if pos_rank <= 3:
            w = 0.5
        elif pos_rank <= 5:
            w = 4.0
        elif pos_rank <= 50:
            w = 3.0
        else:
            w = 1.0
        for _, neg_row in neg.iterrows():
            neg_x = neg_row[BASE_FEATURES].to_numpy(dtype=float)
            diff = pos_x - neg_x
            rows.append(diff)
            labels.append(1)
            weights.append(w)
            rows.append(-diff)
            labels.append(0)
            weights.append(w)
    return np.vstack(rows), np.asarray(labels), np.asarray(weights)


def train_lambdarank_models(train_df: pd.DataFrame, seed: int) -> list[SelectedModel]:
    x, y, w = build_lambdarank_pairs(train_df, seed=seed)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    out = []
    for c in [0.03, 0.1, 0.3, 1.0]:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(xs, y, sample_weight=w)
        out.append(SelectedModel(f"lambdarank_linear_C{c}", "LambdaRank-style weighted pairwise reranker", model, scaler, "linear_coef", 0.0, 1.0, {"C": c}))
    return out


def raw_score(df: pd.DataFrame, model: SelectedModel) -> np.ndarray:
    x = feature_frame(df)[BASE_FEATURES].to_numpy(dtype=float)
    if model.scaler is not None:
        x = model.scaler.transform(x)
    if model.score_mode == "predict_proba":
        return model.model.predict_proba(x)[:, 1]
    if model.score_mode == "decision_function":
        return model.model.decision_function(x)
    if model.score_mode == "linear_coef":
        return x @ model.model.coef_.reshape(-1)
    raise ValueError(model.score_mode)


def norm_by_case(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    s = pd.Series(values, index=df.index, dtype=float)
    lo = s.groupby(df["case_id"]).transform("min")
    hi = s.groupby(df["case_id"]).transform("max")
    return ((s - lo) / (hi - lo).replace(0, 1.0)).to_numpy(dtype=float)


def score_variant(df: pd.DataFrame, model: SelectedModel, alpha: float, current_weight: float) -> pd.DataFrame:
    out = df.copy()
    model_norm = norm_by_case(out, raw_score(out, model))
    current_norm = norm_by_case(out, pd.to_numeric(out["current_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    out["listwise_score"] = current_weight * current_norm + alpha * model_norm
    return out


def current_final_metrics_for_validation(val_df: pd.DataFrame) -> dict[str, Any]:
    ranks = rank_from_candidates(val_df[val_df["candidate_source_hgnn"].eq(1)], "current_score")
    return metric_from_ranks(ranks["new_rank"])


def selection_objective(current: dict[str, Any], new: dict[str, Any], delta_cases: pd.DataFrame) -> tuple[float, str]:
    deltas = {m: float(new[m] - current[m]) for m in ["top1", "top3", "top5", "rank_le_50"]}
    score = 4 * deltas["top1"] + 3 * deltas["top3"] + 3 * deltas["top5"] + deltas["rank_le_50"]
    penalties = []
    for k, weight in [(1, 4), (3, 3), (5, 3)]:
        lost_rate = float((delta_cases[f"top{k}_delta"] == -1).mean())
        if lost_rate:
            score -= weight * lost_rate
            penalties.append(f"top{k}_lost_penalty:{weight * lost_rate:.4f}")
    if new["top5"] < current["top5"]:
        p = 10 * (current["top5"] - new["top5"])
        score -= p
        penalties.append(f"top5_below_current:{p:.4f}")
    if current["rank_le_50"] - new["rank_le_50"] > 0.005:
        p = 5 * (current["rank_le_50"] - new["rank_le_50"])
        score -= p
        penalties.append(f"rank50_drop:{p:.4f}")
    return score, "无硬约束惩罚" if not penalties else "; ".join(penalties)


def delta_table_from_ranks(ranks: pd.DataFrame, current_col: str = "current_rank", new_col: str = "new_rank") -> pd.DataFrame:
    out = ranks.copy()
    for k in [1, 3, 5]:
        out[f"current_top{k}"] = out[current_col] <= k
        out[f"new_top{k}"] = out[new_col] <= k
        out[f"top{k}_delta"] = out[f"new_top{k}"].astype(int) - out[f"current_top{k}"].astype(int)
    out["rank_delta"] = out[current_col] - out[new_col]
    out["rank_change"] = np.where(out["rank_delta"] > 0, "improved", np.where(out["rank_delta"] < 0, "worsened", "unchanged"))
    return out


def validation_grid(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, SelectedModel | None, dict[str, Any]]:
    models = train_pointwise_models(train_df, seed) + train_lambdarank_models(train_df, seed)
    current = current_final_metrics_for_validation(val_df)
    rows = []
    best: SelectedModel | None = None
    best_obj = -1e18
    for model in models:
        for alpha, cur_w in [(0.02, 0.98), (0.05, 0.95), (0.10, 0.90), (0.20, 0.80), (0.35, 0.65), (0.50, 0.50)]:
            scored = score_variant(val_df, model, alpha, cur_w)
            ranks = rank_from_candidates(scored, "listwise_score")
            metrics = metric_from_ranks(ranks["new_rank"])
            deltas = delta_table_from_ranks(ranks, "current_rank_internal", "new_rank")
            obj, reason = selection_objective(current, metrics, deltas)
            key = f"{model.model_key}_a{alpha}_cur{cur_w}"
            row = {
                "model_key": key,
                "model_type": model.model_type,
                "alpha_model": alpha,
                "weight_current": cur_w,
                "validation_objective": obj,
                "validation_selected_reason": reason,
                "current_top1": current["top1"],
                "new_top1": metrics["top1"],
                "delta_top1": metrics["top1"] - current["top1"],
                "current_top3": current["top3"],
                "new_top3": metrics["top3"],
                "delta_top3": metrics["top3"] - current["top3"],
                "current_top5": current["top5"],
                "new_top5": metrics["top5"],
                "delta_top5": metrics["top5"] - current["top5"],
                "current_rank_le_50": current["rank_le_50"],
                "new_rank_le_50": metrics["rank_le_50"],
                "delta_rank_le_50": metrics["rank_le_50"] - current["rank_le_50"],
                "top1_lost_cases": int((deltas["top1_delta"] == -1).sum()),
                "top3_lost_cases": int((deltas["top3_delta"] == -1).sum()),
                "top5_lost_cases": int((deltas["top5_delta"] == -1).sum()),
            }
            rows.append(row)
            if obj > best_obj:
                best_obj = obj
                best = SelectedModel(key, model.model_type, model.model, model.scaler, model.score_mode, alpha, cur_w, model.hyperparams)
    grid = pd.DataFrame(rows).sort_values("validation_objective", ascending=False)
    top = grid.iloc[0]
    if not (top["new_top1"] > top["current_top1"] or top["new_top3"] > top["current_top3"] or top["new_top5"] > top["current_top5"]):
        return grid, None, current
    if top["new_top5"] < top["current_top5"]:
        return grid, None, current
    return grid, best, current


def save_selected(best: SelectedModel | None, grid: pd.DataFrame) -> None:
    if best is None:
        payload = {"selected": False, "reason": "No validation model clearly improved current without Top5 drop."}
        (OUTPUT_DIR / "selected_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_text(REPORT_DIR / "validation_selected_config.md", "# Validation Selected Config\n\n未选择模型：validation 上没有模型明确优于 current 且保持 Top5 不下降。因此不执行 fixed test。")
        return
    payload = {
        "selected": True,
        "model_key": best.model_key,
        "model_type": best.model_type,
        "alpha_model": best.alpha_model,
        "weight_current": best.weight_current,
        "hyperparams": best.hyperparams,
        "features": BASE_FEATURES,
        "scope": "DDD-specific",
        "does_not_modify_encoder": True,
    }
    (OUTPUT_DIR / "selected_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "selected_model.pkl").open("wb") as f:
        pickle.dump({"config": payload, "model": best.model, "scaler": best.scaler, "score_mode": best.score_mode}, f)
    lines = [
        "# Validation Selected Config",
        "",
        f"- selected_model: `{best.model_key}`",
        f"- model_type: {best.model_type}",
        f"- alpha_model: {best.alpha_model}",
        f"- weight_current: {best.weight_current}",
        "",
        "## Top Validation Rows",
        markdown_table(grid.head(10)),
    ]
    write_text(REPORT_DIR / "validation_selected_config.md", "\n".join(lines))


def fixed_test(best: SelectedModel, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = score_variant(test_df, best, best.alpha_model, best.weight_current)
    internal = rank_from_candidates(scored, "listwise_score")
    current = final_mainline_ddd_ranks()
    failed = failed_pairwise_ranks()
    ranks = internal[["case_id", "new_rank"]].merge(current, on="case_id", how="left").merge(failed, on="case_id", how="left")
    ranks["current_rank"] = pd.to_numeric(ranks["current_rank"], errors="coerce").fillna(999999).astype(int)
    ranks["failed_pairwise_rank"] = pd.to_numeric(ranks["failed_pairwise_rank"], errors="coerce").fillna(999999).astype(int)
    cur_m = metric_from_ranks(ranks["current_rank"])
    fail_m = metric_from_ranks(ranks["failed_pairwise_rank"])
    new_m = metric_from_ranks(ranks["new_rank"])
    row = {
        "dataset": "DDD",
        "num_cases": int(cur_m["num_cases"]),
        "current_top1": cur_m["top1"],
        "failed_pairwise_top1": fail_m["top1"],
        "listwise_top1": new_m["top1"],
        "target_top1": DDD_TARGET[0],
        "delta_top1_vs_current": new_m["top1"] - cur_m["top1"],
        "current_top3": cur_m["top3"],
        "failed_pairwise_top3": fail_m["top3"],
        "listwise_top3": new_m["top3"],
        "target_top3": DDD_TARGET[1],
        "delta_top3_vs_current": new_m["top3"] - cur_m["top3"],
        "current_top5": cur_m["top5"],
        "failed_pairwise_top5": fail_m["top5"],
        "listwise_top5": new_m["top5"],
        "target_top5": DDD_TARGET[2],
        "delta_top5_vs_current": new_m["top5"] - cur_m["top5"],
        "current_rank_le_50": cur_m["rank_le_50"],
        "failed_pairwise_rank_le_50": fail_m["rank_le_50"],
        "listwise_rank_le_50": new_m["rank_le_50"],
        "delta_rank_le_50_vs_current": new_m["rank_le_50"] - cur_m["rank_le_50"],
    }
    for k, target in [(1, DDD_TARGET[0]), (3, DDD_TARGET[1]), (5, DDD_TARGET[2])]:
        row[f"case_gap_before_top{k}"] = int(math.ceil(max(target - row[f"current_top{k}"], 0) * row["num_cases"]))
        row[f"case_gap_after_top{k}"] = int(math.ceil(max(target - row[f"listwise_top{k}"], 0) * row["num_cases"]))
    row["deeprare_ddd_target_reached"] = bool(row["listwise_top1"] >= DDD_TARGET[0] and row["listwise_top3"] >= DDD_TARGET[1] and row["listwise_top5"] >= DDD_TARGET[2])
    return pd.DataFrame([row]), ranks


def write_fixed_md(results: pd.DataFrame, delta: pd.DataFrame) -> None:
    row = results.iloc[0]
    lines = [
        "# DDD Candidate Listwise Fixed Test Results",
        "",
        markdown_table(results),
        "",
        "## 误伤统计",
        f"- current top1 -> new not top1: {int(((delta['current_rank'] <= 1) & (delta['new_rank'] > 1)).sum())}",
        f"- current top3 -> new not top3: {int(((delta['current_rank'] <= 3) & (delta['new_rank'] > 3)).sum())}",
        f"- current top5 -> new not top5: {int(((delta['current_rank'] <= 5) & (delta['new_rank'] > 5)).sum())}",
        "",
        "## 结论",
        f"- 是否达到 DeepRare DDD target: {'是' if row['deeprare_ddd_target_reached'] else '否'}。",
    ]
    write_text(REPORT_DIR / "fixed_test_results.md", "\n".join(lines))


def write_case_delta(delta: pd.DataFrame, test_df: pd.DataFrame) -> None:
    source_by_case = (
        test_df.groupby("case_id")[["candidate_source_mondo_expansion", "candidate_source_hpo_expansion", "candidate_source_similar_case"]]
        .max()
        .reset_index()
    )
    out = delta.merge(source_by_case, on="case_id", how="left")
    harmed = out[out["rank_change"].eq("worsened")]
    summary = pd.DataFrame(
        [
            {
                "dataset": "DDD",
                "num_cases": int(len(out)),
                "top1_gained_cases": int((out["top1_delta"] == 1).sum()),
                "top1_lost_cases": int((out["top1_delta"] == -1).sum()),
                "top3_gained_cases": int((out["top3_delta"] == 1).sum()),
                "top3_lost_cases": int((out["top3_delta"] == -1).sum()),
                "top5_gained_cases": int((out["top5_delta"] == 1).sum()),
                "top5_lost_cases": int((out["top5_delta"] == -1).sum()),
                "rank_improved": int((out["rank_change"] == "improved").sum()),
                "rank_worsened": int((out["rank_change"] == "worsened").sum()),
                "rank_unchanged": int((out["rank_change"] == "unchanged").sum()),
                "current_rank_2_5_to_rank1": int(((out["current_rank"].between(2, 5)) & (out["new_rank"] <= 1)).sum()),
                "current_rank_4_5_to_top3": int(((out["current_rank"].between(4, 5)) & (out["new_rank"] <= 3)).sum()),
                "current_rank_6_50_to_top5": int(((out["current_rank"].between(6, 50)) & (out["new_rank"] <= 5)).sum()),
                "current_rank_gt50_to_top50": int(((out["current_rank"] > 50) & (out["new_rank"] <= 50)).sum()),
                "harmed_mean_current_rank": float(harmed["current_rank"].mean()) if not harmed.empty else 0.0,
            }
        ]
    )
    write_csv(out, REPORT_DIR / "case_level_delta.csv")
    write_csv(summary, REPORT_DIR / "case_level_delta_summary.csv")
    source_gain = out[out["rank_change"].eq("improved")][["candidate_source_mondo_expansion", "candidate_source_hpo_expansion", "candidate_source_similar_case"]].sum().reset_index()
    source_gain.columns = ["expansion_source", "improved_cases_with_source"]
    lines = [
        "# DDD Listwise Case-Level Delta Summary",
        "",
        markdown_table(summary),
        "",
        "## Expansion source 对改善病例的贡献",
        markdown_table(source_gain),
        "",
        "## Harmed cases 共同特征",
        f"- harmed cases count: {len(harmed)}",
        f"- harmed mean current rank: {float(harmed['current_rank'].mean()) if not harmed.empty else 0.0:.2f}",
    ]
    write_text(REPORT_DIR / "case_level_delta_summary.md", "\n".join(lines))


def bootstrap(delta: pd.DataFrame, failed_ranks: pd.Series, iters: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cur = delta["current_rank"].to_numpy(dtype=int)
    new = delta["new_rank"].to_numpy(dtype=int)
    failed = failed_ranks.to_numpy(dtype=int)
    n = len(delta)
    rows = []
    for k in [1, 3, 5]:
        vals = defaultdict(list)
        for _ in range(iters):
            idx = rng.integers(0, n, size=n)
            cur_v = float(np.mean(cur[idx] <= k))
            new_v = float(np.mean(new[idx] <= k))
            fail_v = float(np.mean(failed[idx] <= k))
            vals[f"top{k}"].append(new_v)
            vals[f"delta_top{k}_vs_current"].append(new_v - cur_v)
            vals[f"delta_top{k}_vs_failed_pairwise"].append(new_v - fail_v)
        for metric, arr_values in vals.items():
            arr = np.asarray(arr_values)
            rows.append(
                {
                    "dataset": "DDD",
                    "metric": metric,
                    "mean": float(arr.mean()),
                    "ci95_low": float(np.quantile(arr, 0.025)),
                    "ci95_high": float(np.quantile(arr, 0.975)),
                    "stable_positive": bool(metric.startswith("delta") and np.quantile(arr, 0.025) > 0),
                }
            )
    return pd.DataFrame(rows)


def write_bootstrap_md(ci: pd.DataFrame) -> None:
    lines = [
        "# DDD Candidate Listwise Bootstrap 95% CI",
        "",
        markdown_table(ci),
        "",
        "## 结论",
        f"- 是否稳定优于 current: {bool(ci[ci['metric'].str.contains('vs_current')]['stable_positive'].any())}",
        f"- 是否稳定优于 failed pairwise: {bool(ci[ci['metric'].str.contains('vs_failed_pairwise')]['stable_positive'].any())}",
        "- 是否接近或达到 DeepRare target：见 fixed_test_results.md。",
    ]
    write_text(REPORT_DIR / "bootstrap_ci.md", "\n".join(lines))


def write_paper_and_next(executed: bool, results: pd.DataFrame | None) -> None:
    if not executed or results is None or results.empty:
        reached = False
        can_main = False
    else:
        row = results.iloc[0]
        reached = bool(row["deeprare_ddd_target_reached"])
        can_main = bool(reached and row["delta_top1_vs_current"] >= 0 and row["delta_top3_vs_current"] >= 0 and row["delta_top5_vs_current"] >= 0)
    lines = [
        "# Recommended Paper Table",
        "",
        f"1. listwise reranker 是否能作为 DDD 新主线：{'是' if can_main else '否'}。",
        f"2. 是否达到 DeepRare DDD target：{'是' if reached else '否'}。",
        f"3. 是否可以进入论文主表：{'可以' if can_main else '不建议'}。",
        "4. failed pairwise hard-negative head 是否作为负结果附表：是。",
        "5. 是否仍需要 encoder-level hard negative fine-tuning：是，如果 listwise 未达标或 validation 未通过。",
        "6. 是否需要 label/mapping/outlier audit：是。",
        "7. 图对比学习是否仍后置：是。",
        "8. MIMIC 是否仍需要单独 top1-oriented listwise reranker：是。",
    ]
    write_text(REPORT_DIR / "recommended_paper_table.md", "\n".join(lines))
    if not reached:
        next_lines = [
            "# Next After Failed Listwise",
            "",
            "A. encoder-level hard negative fine-tuning：需要。保持 encoder 架构不变，新增可开关 hard-negative sampler 和 margin/listwise/supervised contrastive loss，独立 checkpoint，validation selected，test fixed once。",
            "B. DDD label/mapping/outlier audit：需要。检查 sibling/parent-child exact miss、MONDO 粒度、gold 是否过细或过粗、HPO overlap 异常。",
            "C. candidate generation 上游改造：如果 expanded recall 仍不能覆盖 rank>50，则需要继续。",
            "D. 图对比学习：仍作为 P4，不作为当前优先项。",
        ]
        write_text(REPORT_DIR / "next_after_failed_listwise.md", "\n".join(next_lines))


def write_feature_manifest() -> None:
    lines = [
        "# Listwise Feature Manifest",
        "",
        "- gold-relative fields such as `mondo_parent_child_flag`, `mondo_sibling_flag`, and `disease_hyperedge_similarity_to_gold` are retained for audit/supplementary analysis, not used as model scoring features.",
        "- scoring features:",
        *[f"  - `{feature}`" for feature in BASE_FEATURES],
    ]
    write_text(REPORT_DIR / "listwise_feature_manifest.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_failed_pairwise_audit()

    mondo = load_mondo_resource(MONDO_JSON)
    disease_hpos = load_disease_hpos()
    inverted = build_hpo_inverted(disease_hpos)
    case_hpos, case_gold = load_case_hpos()
    train_case_hpos = {case_id: hpos for case_id, hpos in case_hpos.items() if case_id.startswith("train::")}
    train_case_gold = {case_id: gold for case_id, gold in case_gold.items() if case_id.startswith("train::")}

    train_base = load_base_candidates(TRAIN_RAW, "train")
    val_base = load_base_candidates(VAL_RAW, "validation")
    test_base = load_base_candidates(TEST_RAW, "test")

    train_df = make_expanded_candidates(train_base, "train", mondo, case_hpos, case_gold, disease_hpos, inverted, train_case_hpos, train_case_gold, args.max_expanded_candidates)
    val_df = make_expanded_candidates(val_base, "validation", mondo, case_hpos, case_gold, disease_hpos, inverted, train_case_hpos, train_case_gold, args.max_expanded_candidates)
    test_df = make_expanded_candidates(test_base, "test", mondo, case_hpos, case_gold, disease_hpos, inverted, train_case_hpos, train_case_gold, args.max_expanded_candidates)

    write_csv(train_df, OUTPUT_DIR / "ddd_train_expanded_candidates.csv")
    write_csv(val_df, OUTPUT_DIR / "ddd_validation_expanded_candidates.csv")
    write_csv(test_df, OUTPUT_DIR / "ddd_test_expanded_candidates.csv")
    write_csv(train_df, OUTPUT_DIR / "listwise_train_candidates.csv")
    write_csv(val_df, OUTPUT_DIR / "listwise_validation_candidates.csv")
    write_csv(test_df, OUTPUT_DIR / "listwise_test_candidates.csv")
    write_feature_manifest()

    audit = audit_expansion(train_df, val_df, test_df)
    write_expansion_audit(audit)

    grid, best, current_val = validation_grid(train_df, val_df, args.random_seed)
    write_csv(grid, REPORT_DIR / "validation_grid.csv")
    save_selected(best, grid)

    executed = best is not None
    results = None
    if executed:
        results, ranks = fixed_test(best, test_df)
        delta = delta_table_from_ranks(ranks, "current_rank", "new_rank")
        write_csv(results, REPORT_DIR / "fixed_test_results.csv")
        write_csv(ranks, OUTPUT_DIR / "fixed_test_case_ranks.csv")
        write_fixed_md(results, delta)
        write_case_delta(delta, test_df)
        ci = bootstrap(delta, ranks["failed_pairwise_rank"], args.bootstrap_iters, args.random_seed)
        write_csv(ci, REPORT_DIR / "bootstrap_ci.csv")
        write_bootstrap_md(ci)
    else:
        write_text(REPORT_DIR / "fixed_test_results.md", "# Fixed Test Results\n\n未执行 fixed test：validation 没有 selected model 明确优于 current 且保持 Top5 不下降。")
        write_csv(pd.DataFrame(), REPORT_DIR / "fixed_test_results.csv")
        write_csv(pd.DataFrame(), OUTPUT_DIR / "fixed_test_case_ranks.csv")
        write_text(REPORT_DIR / "case_level_delta_summary.md", "# Case-Level Delta Summary\n\n未执行 fixed test。")
        write_csv(pd.DataFrame(), REPORT_DIR / "case_level_delta.csv")
        write_csv(pd.DataFrame(), REPORT_DIR / "bootstrap_ci.csv")
        write_text(REPORT_DIR / "bootstrap_ci.md", "# Bootstrap 95% CI\n\n未执行 fixed test。")

    write_paper_and_next(executed, results)
    manifest = {
        "executed_fixed_test": executed,
        "selected_model": best.model_key if best else None,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "report_dir": str(REPORT_DIR.resolve()),
        "test_side_tuning": False,
        "does_not_modify_encoder": True,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
