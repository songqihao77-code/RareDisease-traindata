from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import load_static_resources, load_test_cases, load_yaml_config
from src.data.dataset import read_case_table_file
from src.rerank.hpo_semantic import HpoSemanticMatcher


DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_finetune_attn_idf_main.yaml"
DEFAULT_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run read-only rank and mapping diagnostics.")
    parser.add_argument("--data-config-path", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--train-config-path", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--details-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def latest_details_path(train_config: dict[str, Any]) -> Path:
    save_dir = Path(train_config["paths"]["save_dir"])
    eval_dir = save_dir / "evaluation"
    candidates = sorted(eval_dir.glob("*_details.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No evaluation details CSV found under {eval_dir}")
    return candidates[0]


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Candidates CSV not found: {path}")
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    required = {
        "case_id",
        "dataset_name",
        "gold_id",
        "candidate_id",
        "original_rank",
        "hgnn_score",
        "exact_overlap",
        "ic_weighted_overlap",
        "semantic_ic_overlap",
        "disease_hpo_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for column in required - {"case_id", "dataset_name", "gold_id", "candidate_id"}:
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df.sort_values(["case_id", "original_rank"], kind="stable").reset_index(drop=True)


def metric_summary(ranks: pd.Series) -> dict[str, Any]:
    arr = ranks.astype(int).to_numpy()
    return {
        "num_cases": int(arr.size),
        "top1": float(np.mean(arr <= 1)) if arr.size else float("nan"),
        "top3": float(np.mean(arr <= 3)) if arr.size else float("nan"),
        "top5": float(np.mean(arr <= 5)) if arr.size else float("nan"),
        "rank_le_50": float(np.mean(arr <= 50)) if arr.size else float("nan"),
        "gold_in_top50": float(np.mean(arr <= 50)) if arr.size else float("nan"),
        "median_rank": float(np.median(arr)) if arr.size else float("nan"),
        "mean_rank": float(np.mean(arr)) if arr.size else float("nan"),
        "rank_gt_50_count": int(np.sum(arr > 50)),
        "top50_rank_gt5_count": int(np.sum((arr > 5) & (arr <= 50))),
        "rank_1_count": int(np.sum(arr <= 1)),
        "rank_2_3_count": int(np.sum((arr > 1) & (arr <= 3))),
        "rank_4_5_count": int(np.sum((arr > 3) & (arr <= 5))),
        "rank_6_10_count": int(np.sum((arr > 5) & (arr <= 10))),
        "rank_11_50_count": int(np.sum((arr > 10) & (arr <= 50))),
        "rank_gt_50_bucket_count": int(np.sum(arr > 50)),
    }


def build_rank_decomposition(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in details.groupby("dataset_name", sort=True):
        row = {"dataset": dataset, **metric_summary(group["true_rank"])}
        rows.append(row)
    rows.append({"dataset": "ALL", **metric_summary(details["true_rank"])})
    return pd.DataFrame(rows)


def labels_per_case_from_raw(data_config: dict[str, Any], data_config_path: Path, dataset_name: str) -> pd.DataFrame:
    bundle = load_test_cases(data_config, data_config_path)
    raw_df = bundle["raw_df"].copy()
    case_id_col = bundle["case_id_col"]
    label_col = bundle["label_col"]
    subset = raw_df.loc[raw_df["_source_file"].astype(str).apply(lambda value: Path(value).stem == dataset_name)]
    labels = (
        subset.dropna(subset=[case_id_col, label_col])
        .groupby(case_id_col)[label_col]
        .agg(lambda values: sorted(set(str(value) for value in values)))
        .reset_index()
        .rename(columns={case_id_col: "case_id", label_col: "gold_labels"})
    )
    labels["num_gold_labels"] = labels["gold_labels"].apply(len)
    return labels


def any_label_ranks(candidates: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_map = dict(zip(labels["case_id"], labels["gold_labels"], strict=True))
    rows: list[dict[str, Any]] = []
    for case_id, group in candidates.groupby("case_id", sort=False):
        gold_labels = set(label_map.get(str(case_id), []))
        hit_ranks = group.loc[group["candidate_id"].isin(gold_labels), "original_rank"]
        any_rank = int(hit_ranks.min()) if not hit_ranks.empty else 51
        rows.append({"case_id": str(case_id), "any_label_rank": any_rank})
    return pd.DataFrame(rows)


def build_mimic_multilabel_audit(details: pd.DataFrame, candidates: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    mimic_details = details.loc[details["dataset_name"] == "mimic_test", ["case_id", "true_rank"]].copy()
    mimic_candidates = candidates.loc[candidates["dataset_name"] == "mimic_test"].copy()
    any_ranks = any_label_ranks(mimic_candidates, labels)
    merged = labels.merge(mimic_details, on="case_id", how="left").merge(any_ranks, on="case_id", how="left")
    merged["subset"] = np.where(merged["num_gold_labels"] > 1, "multi_label", "single_label")
    rows: list[dict[str, Any]] = []
    for subset_name, group in merged.groupby("subset", sort=True):
        exact = metric_summary(group["true_rank"])
        any_label = metric_summary(group["any_label_rank"])
        rows.append(
            {
                "dataset": "mimic_test",
                "subset": subset_name,
                "num_cases": int(len(group)),
                "mean_gold_label_count": float(group["num_gold_labels"].mean()),
                "median_gold_label_count": float(group["num_gold_labels"].median()),
                "exact_top1": exact["top1"],
                "exact_top3": exact["top3"],
                "exact_top5": exact["top5"],
                "exact_rank_le_50": exact["rank_le_50"],
                "any_label_hit_at_1": any_label["top1"],
                "any_label_hit_at_3": any_label["top3"],
                "any_label_hit_at_5": any_label["top5"],
                "any_label_hit_at_50": any_label["rank_le_50"],
            }
        )
    exact_all = metric_summary(merged["true_rank"])
    any_all = metric_summary(merged["any_label_rank"])
    rows.append(
        {
            "dataset": "mimic_test",
            "subset": "ALL",
            "num_cases": int(len(merged)),
            "mean_gold_label_count": float(merged["num_gold_labels"].mean()),
            "median_gold_label_count": float(merged["num_gold_labels"].median()),
            "exact_top1": exact_all["top1"],
            "exact_top3": exact_all["top3"],
            "exact_top5": exact_all["top5"],
            "exact_rank_le_50": exact_all["rank_le_50"],
            "any_label_hit_at_1": any_all["top1"],
            "any_label_hit_at_3": any_all["top3"],
            "any_label_hit_at_5": any_all["top5"],
            "any_label_hit_at_50": any_all["rank_le_50"],
        }
    )
    return pd.DataFrame(rows)


def read_train_labels(train_config: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for raw_path in train_config["paths"].get("train_files", []):
        path = Path(raw_path)
        if not path.is_file():
            continue
        df = read_case_table_file(path)
        label_col = "mondo_label" if "mondo_label" in df.columns else "mondo_id" if "mondo_id" in df.columns else None
        if label_col is None:
            continue
        labels.update(df[label_col].dropna().astype(str).tolist())
    return labels


def nonzero_indices(matrix: Any, disease_idx: int) -> set[int]:
    col = matrix[:, int(disease_idx)]
    return {int(idx) for idx in col.nonzero()[0].tolist()}


def evidence_for_pair(
    *,
    case_hpo_idx: set[int],
    disease_hpo_idx: set[int],
    hpo_specificity: np.ndarray,
    semantic_matcher: HpoSemanticMatcher,
    idx_to_hpo: dict[int, str],
    hpo_specificity_by_id: dict[str, float],
) -> dict[str, Any]:
    shared = case_hpo_idx & disease_hpo_idx
    case_count = len(case_hpo_idx)
    disease_count = len(disease_hpo_idx)
    case_ic_total = float(sum(float(hpo_specificity[idx]) for idx in case_hpo_idx))
    shared_ic_total = float(sum(float(hpo_specificity[idx]) for idx in shared))
    semantic = semantic_matcher.score(
        case_hpos={idx_to_hpo[idx] for idx in case_hpo_idx if idx in idx_to_hpo},
        disease_hpos={idx_to_hpo[idx] for idx in disease_hpo_idx if idx in idx_to_hpo},
        hpo_specificity=hpo_specificity_by_id,
    )
    return {
        "case_hpo_count": int(case_count),
        "gold_disease_hpo_count": int(disease_count),
        "shared_hpo_count": int(len(shared)),
        "exact_overlap": float(len(shared) / math.sqrt(case_count * disease_count)) if case_count and disease_count else 0.0,
        "ic_weighted_overlap": float(shared_ic_total / case_ic_total) if case_ic_total > 0 else 0.0,
        "semantic_overlap": float(semantic["semantic_ic_overlap"]),
        "overlap_zero": bool(len(shared) == 0),
    }


def load_mondo_status(project_root: Path) -> dict[str, dict[str, bool]]:
    candidates = [
        project_root / "LLLdataset" / "knowledge" / "mondo-rare.json",
        project_root / "data" / "raw_data" / "mondo.json",
        project_root / "data" / "raw_data" / "mondo-base.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status: dict[str, dict[str, bool]] = {}
        for graph in payload.get("graphs", []):
            for node in graph.get("nodes", []):
                node_id = normalize_mondo_id(str(node.get("id", "")))
                if not node_id.startswith("MONDO:"):
                    continue
                meta = node.get("meta", {}) if isinstance(node.get("meta", {}), dict) else {}
                label = str(node.get("lbl", ""))
                deprecated = bool(meta.get("deprecated", False) or label.lower().startswith("obsolete"))
                status[node_id] = {"obsolete": deprecated}
        if status:
            return status
    return {}


def normalize_mondo_id(value: str) -> str:
    text = str(value).strip()
    if text.startswith("http://purl.obolibrary.org/obo/MONDO_"):
        return text.rsplit("/", 1)[-1].replace("_", ":")
    if text.startswith("MONDO_"):
        return text.replace("_", ":")
    return text


def build_case_hpo_map(data_config: dict[str, Any], data_config_path: Path, hpo_to_idx: dict[str, int]) -> dict[str, set[int]]:
    bundle = load_test_cases(data_config, data_config_path)
    valid_hpos = set(hpo_to_idx)
    case_hpo_map: dict[str, set[int]] = {}
    for row in bundle["case_table"].itertuples(index=False):
        hpo_ids = [str(hpo) for hpo in row.hpo_ids if str(hpo) in valid_hpos]
        case_hpo_map[str(row.case_id)] = {int(hpo_to_idx[hpo]) for hpo in hpo_ids}
    return case_hpo_map


def build_top50_miss_audit(
    details: pd.DataFrame,
    data_config: dict[str, Any],
    data_config_path: Path,
    train_config: dict[str, Any],
) -> pd.DataFrame:
    resources = load_static_resources(train_config)
    hpo_to_idx = resources["hpo_to_idx"]
    disease_to_idx = resources["disease_to_idx"]
    hpo_specificity = resources["hpo_specificity"]
    idx_to_hpo = {int(idx): str(hpo_id) for hpo_id, idx in hpo_to_idx.items()}
    hpo_specificity_by_id = {str(hpo_id): float(hpo_specificity[int(idx)]) for hpo_id, idx in hpo_to_idx.items()}
    semantic_matcher, semantic_meta = HpoSemanticMatcher.from_project(PROJECT_ROOT)
    del semantic_meta
    case_hpo_map = build_case_hpo_map(data_config, data_config_path, hpo_to_idx)
    seen_labels = read_train_labels(train_config)
    mondo_status = load_mondo_status(PROJECT_ROOT)

    rows: list[dict[str, Any]] = []
    for row in details.itertuples(index=False):
        rank = int(row.true_rank)
        if rank <= 50:
            continue
        gold_id = str(row.true_label)
        gold_idx = disease_to_idx.get(gold_id)
        case_hpo_idx = case_hpo_map.get(str(row.case_id), set())
        if gold_idx is None:
            evidence = {
                "case_hpo_count": int(len(case_hpo_idx)),
                "gold_disease_hpo_count": 0,
                "shared_hpo_count": 0,
                "exact_overlap": 0.0,
                "ic_weighted_overlap": 0.0,
                "semantic_overlap": 0.0,
                "overlap_zero": True,
            }
        else:
            evidence = evidence_for_pair(
                case_hpo_idx=case_hpo_idx,
                disease_hpo_idx=nonzero_indices(resources["H_disease"], int(gold_idx)),
                hpo_specificity=hpo_specificity,
                semantic_matcher=semantic_matcher,
                idx_to_hpo=idx_to_hpo,
                hpo_specificity_by_id=hpo_specificity_by_id,
            )
        rows.append(
            {
                "case_id": str(row.case_id),
                "dataset": str(row.dataset_name),
                "gold_id": gold_id,
                "true_rank": rank,
                **evidence,
                "seen_label": bool(gold_id in seen_labels),
                "unmapped_mondo": bool(gold_id not in disease_to_idx),
                "obsolete_mondo": bool(mondo_status.get(gold_id, {}).get("obsolete", False)),
            }
        )
    return pd.DataFrame(rows)


def bucket_value(value: float, edges: list[float]) -> str:
    prev = "-inf"
    for edge in edges:
        if value <= edge:
            return f"({prev},{edge}]"
        prev = str(edge)
    return f"({prev},inf)"


def build_failure_by_bucket(miss_audit: pd.DataFrame) -> pd.DataFrame:
    if miss_audit.empty:
        return pd.DataFrame()
    df = miss_audit.copy()
    df["case_hpo_bucket"] = df["case_hpo_count"].apply(lambda value: bucket_value(float(value), [3, 5, 10, 20]))
    df["gold_hpo_bucket"] = df["gold_disease_hpo_count"].apply(lambda value: bucket_value(float(value), [0, 3, 7, 15, 30]))
    df["ic_overlap_bucket"] = df["ic_weighted_overlap"].apply(lambda value: bucket_value(float(value), [0, 0.05, 0.1, 0.25, 0.5]))
    group_cols = ["dataset", "case_hpo_bucket", "gold_hpo_bucket", "ic_overlap_bucket", "seen_label", "overlap_zero"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            num_rank_gt50_cases=("case_id", "nunique"),
            mean_rank=("true_rank", "mean"),
            mean_case_hpo_count=("case_hpo_count", "mean"),
            mean_gold_disease_hpo_count=("gold_disease_hpo_count", "mean"),
            mean_exact_overlap=("exact_overlap", "mean"),
            mean_ic_weighted_overlap=("ic_weighted_overlap", "mean"),
            mean_semantic_overlap=("semantic_overlap", "mean"),
            obsolete_mondo_rate=("obsolete_mondo", "mean"),
            unmapped_mondo_rate=("unmapped_mondo", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "num_rank_gt50_cases"], ascending=[True, False], kind="stable")
    )
    return summary


def load_mondo_parent_map(project_root: Path) -> dict[str, set[str]]:
    candidates = [
        project_root / "LLLdataset" / "knowledge" / "mondo-rare.json",
        project_root / "data" / "raw_data" / "mondo.json",
        project_root / "data" / "raw_data" / "mondo-base.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        parent_map: dict[str, set[str]] = defaultdict(set)
        for graph in payload.get("graphs", []):
            for edge in graph.get("edges", []):
                pred = str(edge.get("pred", ""))
                if not pred.endswith(("is_a", "subClassOf")):
                    continue
                child = normalize_mondo_id(str(edge.get("sub", "")))
                parent = normalize_mondo_id(str(edge.get("obj", "")))
                if child.startswith("MONDO:") and parent.startswith("MONDO:"):
                    parent_map[child].add(parent)
        if parent_map:
            return dict(parent_map)
    return {}


def ancestor_map(parent_map: dict[str, set[str]]) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = {}
    for node in parent_map:
        seen: set[str] = set()
        queue: deque[str] = deque(parent_map.get(node, set()))
        while queue:
            parent = queue.popleft()
            if parent in seen:
                continue
            seen.add(parent)
            queue.extend(parent_map.get(parent, set()) - seen)
        ancestors[node] = seen
    return ancestors


def disease_relation(left: str, right: str, parent_map: dict[str, set[str]], ancestors: dict[str, set[str]]) -> str:
    if left == right:
        return "same"
    left_anc = ancestors.get(left, set())
    right_anc = ancestors.get(right, set())
    if left in right_anc:
        return "candidate_ancestor_of_gold"
    if right in left_anc:
        return "candidate_descendant_of_gold"
    if parent_map.get(left, set()) & parent_map.get(right, set()):
        return "same_parent"
    if left_anc & right_anc:
        return "shared_ancestor"
    return "unrelated_or_unknown"


def disease_pair_overlap(
    left_idx: int,
    right_idx: int,
    h_disease: Any,
    hpo_specificity: np.ndarray,
) -> dict[str, Any]:
    left = nonzero_indices(h_disease, left_idx)
    right = nonzero_indices(h_disease, right_idx)
    shared = left & right
    left_ic = float(sum(float(hpo_specificity[idx]) for idx in left))
    shared_ic = float(sum(float(hpo_specificity[idx]) for idx in shared))
    return {
        "candidate_disease_hpo_count": int(len(left)),
        "gold_disease_hpo_count": int(len(right)),
        "candidate_gold_shared_hpo_count": int(len(shared)),
        "candidate_gold_jaccard": float(len(shared) / len(left | right)) if left or right else 0.0,
        "candidate_gold_ic_coverage": float(shared_ic / left_ic) if left_ic > 0 else 0.0,
    }


def build_ddd_nearmiss(
    details: pd.DataFrame,
    candidates: pd.DataFrame,
    train_config: dict[str, Any],
) -> pd.DataFrame:
    resources = load_static_resources(train_config)
    disease_to_idx = resources["disease_to_idx"]
    parent_map = load_mondo_parent_map(PROJECT_ROOT)
    ancestors = ancestor_map(parent_map)
    ddd_wrong = details.loc[(details["dataset_name"] == "DDD") & (details["true_rank"] > 1), ["case_id", "true_label", "true_rank"]]
    wrong_case_ids = set(ddd_wrong["case_id"].astype(str))
    detail_map = ddd_wrong.set_index("case_id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for row in candidates.loc[
        (candidates["dataset_name"] == "DDD")
        & (candidates["case_id"].isin(wrong_case_ids))
        & (candidates["original_rank"] <= 10)
    ].itertuples(index=False):
        case_id = str(row.case_id)
        gold_id = str(detail_map[case_id]["true_label"])
        candidate_id = str(row.candidate_id)
        candidate_idx = disease_to_idx.get(candidate_id)
        gold_idx = disease_to_idx.get(gold_id)
        overlap = {}
        if candidate_idx is not None and gold_idx is not None:
            overlap = disease_pair_overlap(int(candidate_idx), int(gold_idx), resources["H_disease"], resources["hpo_specificity"])
        rows.append(
            {
                "case_id": case_id,
                "gold_id": gold_id,
                "case_true_rank": int(detail_map[case_id]["true_rank"]),
                "candidate_id": candidate_id,
                "candidate_rank": int(row.original_rank),
                "candidate_hgnn_score": float(row.hgnn_score),
                "candidate_case_exact_overlap": float(row.exact_overlap),
                "candidate_case_ic_weighted_overlap": float(row.ic_weighted_overlap),
                "candidate_case_semantic_overlap": float(row.semantic_ic_overlap),
                **overlap,
                "mondo_relation_to_gold": disease_relation(candidate_id, gold_id, parent_map, ancestors),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pair_counts = Counter(zip(df["gold_id"], df["candidate_id"]))
    df["confusion_pair_count"] = [
        int(pair_counts[(gold_id, candidate_id)]) for gold_id, candidate_id in zip(df["gold_id"], df["candidate_id"], strict=True)
    ]
    return df.sort_values(["confusion_pair_count", "gold_id", "case_id", "candidate_rank"], ascending=[False, True, True, True])


def write_recommended_next_experiments(output_path: Path) -> None:
    text = """# Recommended Next Experiments

## 可以进入论文主表的结果

- 原始 exact HGNN baseline：来自 `src.evaluation.evaluator` 的 top1/top3/top5/rank<=50，可进入主表。
- validation 选定、test 只评估一次的 top50 reranker：只有在使用 validation candidates 保存固定权重后，才能进入主表。

## 只能作为附表或 error analysis 的结果

- 当前 test-side rerank grid/gate 结果：只能作为 exploratory upper bound 或消融分析。
- `mimic_test` any-label hit：不覆盖原始 exact metric，只用于解释多标签病例的潜在低估。
- synonym / parent-child / obsolete relaxed evaluation：只能作为附加口径，不能替代 exact evaluation。

## 不需要训练即可优先修复

1. 补充 MONDO/HPO obsolete、alt_id、synonym、subclass 审计。
2. 对 `mimic_test` 多标签病例输出 exact 与 any-label 双口径。
3. 对 rank>50 样本补充 gold hyperedge coverage，并修复明显 unmapped/obsolete 标签。
4. 核对 HMS 25-case test split 与论文口径是否一致。

## 需要 reranker 的问题

1. gold 已在 top50 但 rank>5 的样本，尤其 DDD near-miss。
2. evidence 特征能解释的 top50 内排序错误，例如 exact/IC/semantic overlap 明显支持 gold。
3. validation-selected linear/listwise reranker，用固定权重或轻量模型只在 HGNN top50 内重排。

## 需要 hard negative training 的问题

1. DDD 中同父类、同祖先或 HPO 高重叠疾病反复混淆。
2. 当前在线 hard negatives 覆盖不到的 ontology sibling / disease family negatives。
3. top50 内候选分数差距小但语义上高度相近的病例。

## 暂不建议作为主线的方向

- 把 static HPO retrieval 作为硬候选池；现有诊断显示 static@50 整体弱且对 `mimic_test` 不稳。
- 在 test candidates 上直接选择 rerank 权重；必须改为 validation select -> fixed test。
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_yaml_config(args.data_config_path)
    train_config = load_yaml_config(args.train_config_path)
    details_path = args.details_path or latest_details_path(train_config)
    details = pd.read_csv(details_path, dtype={"case_id": str, "dataset_name": str, "true_label": str})
    details["true_rank"] = pd.to_numeric(details["true_rank"], errors="raise").astype(int)
    candidates = load_candidates(args.candidates_path)

    rank_decomposition = build_rank_decomposition(details)
    rank_decomposition.to_csv(output_dir / "dataset_rank_decomposition.csv", index=False, encoding="utf-8-sig")

    mimic_labels = labels_per_case_from_raw(data_config, args.data_config_path, "mimic_test")
    mimic_audit = build_mimic_multilabel_audit(details, candidates, mimic_labels)
    mimic_audit.to_csv(output_dir / "mimic_multilabel_audit.csv", index=False, encoding="utf-8-sig")

    miss_audit = build_top50_miss_audit(details, data_config, args.data_config_path, train_config)
    miss_audit.to_csv(output_dir / "top50_miss_audit.csv", index=False, encoding="utf-8-sig")
    bucket = build_failure_by_bucket(miss_audit)
    bucket.to_csv(output_dir / "dataset_failure_by_bucket.csv", index=False, encoding="utf-8-sig")

    ddd_nearmiss = build_ddd_nearmiss(details, candidates, train_config)
    ddd_nearmiss.to_csv(output_dir / "ddd_nearmiss_pairs.csv", index=False, encoding="utf-8-sig")

    write_recommended_next_experiments(output_dir / "recommended_next_experiments.md")

    manifest = {
        "details_path": str(details_path.resolve()),
        "candidates_path": str(args.candidates_path.resolve()),
        "outputs": {
            "dataset_rank_decomposition": str((output_dir / "dataset_rank_decomposition.csv").resolve()),
            "mimic_multilabel_audit": str((output_dir / "mimic_multilabel_audit.csv").resolve()),
            "top50_miss_audit": str((output_dir / "top50_miss_audit.csv").resolve()),
            "dataset_failure_by_bucket": str((output_dir / "dataset_failure_by_bucket.csv").resolve()),
            "ddd_nearmiss_pairs": str((output_dir / "ddd_nearmiss_pairs.csv").resolve()),
            "recommended_next_experiments": str((output_dir / "recommended_next_experiments.md").resolve()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
