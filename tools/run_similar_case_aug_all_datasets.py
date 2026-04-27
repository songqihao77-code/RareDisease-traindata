from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_case_files
from src.evaluation.evaluator import load_test_cases, load_yaml_config
from src.training.trainer import resolve_train_files, split_train_val_by_case
from tools.run_mimic_similar_case_aug import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    compute_similar_matches,
    df_to_markdown,
    hgnn_source,
    load_candidates,
    parse_float_list,
    parse_int_list,
    parse_score_types,
    select_on_validation,
    similar_source,
    combine,
    write_csv,
    write_text,
)


DEFAULT_DETAILS = PROJECT_ROOT / "outputs" / "attn_beta_sweep" / "edge_log_beta02" / "evaluation" / "best_20260426_165031_details.csv"
DEFAULT_PER_DATASET = PROJECT_ROOT / "outputs" / "attn_beta_sweep" / "edge_log_beta02" / "evaluation" / "best_20260426_165031_per_dataset.csv"
DEFAULT_VAL_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_validation.csv"
DEFAULT_TEST_CANDIDATES = PROJECT_ROOT / "outputs" / "mimic_next" / "top50_candidates_recleaned_test.csv"
DEFAULT_FROZEN_CONFIG = PROJECT_ROOT / "reports" / "mimic_next" / "frozen_similar_case_aug_config.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "similar_case_all"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "similar_case_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all-dataset fixed HGNN + SimilarCase-Aug evaluation.")
    parser.add_argument("--data-config-path", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--train-config-path", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--baseline-details-path", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--baseline-per-dataset-path", type=Path, default=DEFAULT_PER_DATASET)
    parser.add_argument("--validation-candidates-path", type=Path, default=DEFAULT_VAL_CANDIDATES)
    parser.add_argument("--test-candidates-path", type=Path, default=DEFAULT_TEST_CANDIDATES)
    parser.add_argument("--frozen-config-path", type=Path, default=DEFAULT_FROZEN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--candidate-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["frozen", "validation_select"], default="frozen")
    parser.add_argument("--similar-case-topk", default="3,5,10,20")
    parser.add_argument("--similar-case-weight", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--score-type", default="raw_similarity,rank_decay")
    return parser.parse_args()


def load_all_case_tables(
    data_config: dict[str, Any],
    data_config_path: Path,
    train_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))
    train_files = resolve_train_files(train_config["paths"])
    all_train = load_case_files(
        file_paths=[str(path) for path in train_files],
        case_id_col=case_id_col,
        label_col=label_col,
        disease_index_path=train_config["paths"]["disease_index_path"],
        split_namespace="train",
    )
    train_df, val_df = split_train_val_by_case(
        all_train,
        val_ratio=float(train_config["data"]["val_ratio"]),
        random_seed=int(train_config["data"]["random_seed"]),
        case_id_col=case_id_col,
    )
    test_bundle = load_test_cases(data_config, data_config_path)
    test_df = test_bundle["raw_df"].copy()

    def to_table(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for case_id, group in df.groupby(case_id_col, sort=False):
            labels = sorted(set(group[label_col].dropna().astype(str).tolist()))
            hpos = sorted(set(group[hpo_col].dropna().astype(str).tolist()))
            source_file = Path(str(group["_source_file"].iloc[0])).name if "_source_file" in group else ""
            dataset_name = Path(source_file).stem if source_file else ""
            rows.append(
                {
                    "case_id": str(case_id),
                    "dataset_name": dataset_name,
                    "source_file": source_file,
                    "primary_label": str(group[label_col].iloc[0]),
                    "label_set": labels,
                    "hpo_ids": hpos,
                    "hpo_count": len(hpos),
                    "label_count": len(labels),
                }
            )
        return pd.DataFrame(rows)

    return to_table(train_df), to_table(val_df), to_table(test_df)


def dataset_counts(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["file", "dataset_name", "rows", "cases"])
    dataset_col = "dataset_name" if "dataset_name" in df.columns else None
    case_col = "case_id" if "case_id" in df.columns else None
    rows = []
    if dataset_col:
        for dataset, group in df.groupby(dataset_col, dropna=False, sort=True):
            rows.append(
                {
                    "file": name,
                    "dataset_name": str(dataset),
                    "rows": int(len(group)),
                    "cases": int(group[case_col].nunique()) if case_col else "",
                }
            )
    return pd.DataFrame(rows)


def write_candidate_audit(
    report_dir: Path,
    test_candidates: pd.DataFrame,
    val_candidates: pd.DataFrame,
    baseline_details: pd.DataFrame,
    baseline_per_dataset: pd.DataFrame,
) -> None:
    frames = [
        dataset_counts(test_candidates, "test_candidates"),
        dataset_counts(val_candidates, "validation_candidates"),
        dataset_counts(baseline_details, "baseline_details"),
        dataset_counts(baseline_per_dataset, "baseline_per_dataset"),
    ]
    audit = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    write_csv(audit, report_dir / "candidate_file_audit.csv")
    test_datasets = set(test_candidates.get("dataset_name", pd.Series(dtype=str)).dropna().astype(str))
    baseline_datasets = set(baseline_per_dataset.get("dataset_name", pd.Series(dtype=str)).dropna().astype(str))
    val_datasets = set(val_candidates.get("dataset_name", pd.Series(dtype=str)).dropna().astype(str))
    md = [
        "# SimilarCase All-Dataset Candidate File Audit",
        "",
        f"- test candidates rows/cases: {len(test_candidates)} / {test_candidates['case_id'].nunique() if 'case_id' in test_candidates else '无法确认'}",
        f"- test candidates datasets: {', '.join(sorted(test_datasets))}",
        f"- baseline per-dataset datasets: {', '.join(sorted(baseline_datasets))}",
        f"- validation candidates rows/cases: {len(val_candidates)} / {val_candidates['case_id'].nunique() if 'case_id' in val_candidates else '无法确认'}",
        f"- validation candidates datasets: {', '.join(sorted(val_datasets))}",
        f"- test candidates 是否覆盖 baseline 全部数据集: {'是' if baseline_datasets <= test_datasets else '否'}",
        "- 每条 candidate 包含 `dataset_name` 和 `case_id` 字段，可直接按 dataset 聚合；`case_id` 也包含 split namespace。",
        "- 当前候选文件已包含全部 test dataset，因此本轮未重新导出 all-test top50 candidates。",
        "",
        df_to_markdown(audit),
    ]
    write_text(report_dir / "candidate_file_audit.md", "\n".join(md))


def load_frozen_config(path: Path) -> dict[str, Any]:
    config = json.load(open(path, encoding="utf-8"))
    return {
        "source_combination": "HGNN + similar_case",
        "status": "loaded_frozen_config",
        "similar_case_topk": int(config["similar_case_topk"]),
        "similar_case_weight": float(config["similar_case_weight"]),
        "similar_case_score_type": str(config["score_type"]),
        "selected_by": str(config.get("selected_by", "validation")),
        "test_protocol": str(config.get("test_protocol", "fixed_once")),
    }


def compute_similar_matches_by_dataset(test_table: pd.DataFrame, train_table: pd.DataFrame, topk: int) -> pd.DataFrame:
    frames = []
    for _, group in test_table.groupby("dataset_name", sort=True):
        frames.append(compute_similar_matches(group.copy(), train_table, topk))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_dataset_metrics(case_ranks: pd.DataFrame, case_table: pd.DataFrame, method: str) -> pd.DataFrame:
    meta = case_table[["case_id", "dataset_name", "source_file"]].copy()
    merged = case_ranks.merge(meta, on="case_id", how="left")
    rows = []
    for dataset, group in merged.groupby("dataset_name", sort=True):
        exact = pd.to_numeric(group["exact_rank"], errors="coerce").fillna(9999).to_numpy(dtype=int)
        any_rank = pd.to_numeric(group["any_label_rank"], errors="coerce").fillna(9999).to_numpy(dtype=int)
        rows.append(
            {
                "dataset_name": dataset,
                "source_file": str(group["source_file"].iloc[0]),
                "method": method,
                "num_cases": int(len(group)),
                "top1": float(np.mean(exact <= 1)),
                "top3": float(np.mean(exact <= 3)),
                "top5": float(np.mean(exact <= 5)),
                "rank_le_50": float(np.mean(exact <= 50)),
                "mean_rank": float(np.mean(exact)),
                "median_rank": float(np.median(exact)),
                "any_label_at_1": float(np.mean(any_rank <= 1)),
                "any_label_at_3": float(np.mean(any_rank <= 3)),
                "any_label_at_5": float(np.mean(any_rank <= 5)),
                "any_label_at_50": float(np.mean(any_rank <= 50)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_with_exact_gold(
    ranked: pd.DataFrame,
    case_table: pd.DataFrame,
    exact_gold_by_case: dict[str, str],
    method: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    label_sets = {row.case_id: set(row.label_set) for row in case_table.itertuples(index=False)}
    rows = []
    exact_ranks = []
    any_ranks = []
    for case_id in case_table["case_id"].astype(str).tolist():
        group = ranked[ranked["case_id"] == case_id]
        exact_gold = exact_gold_by_case.get(case_id)
        if exact_gold is None:
            exact_rank = 9999
        else:
            exact_hits = group[group["candidate_id"] == exact_gold]
            exact_rank = int(exact_hits["rank"].min()) if not exact_hits.empty else 9999
        any_hits = group[group["candidate_id"].isin(label_sets[case_id])]
        any_rank = int(any_hits["rank"].min()) if not any_hits.empty else 9999
        exact_ranks.append(exact_rank)
        any_ranks.append(any_rank)
        rows.append(
            {
                "case_id": case_id,
                "method": method,
                "exact_gold": exact_gold or "",
                "exact_rank": exact_rank,
                "any_label_rank": any_rank,
                "label_count": len(label_sets[case_id]),
            }
        )
    exact_arr = np.asarray(exact_ranks, dtype=int)
    any_arr = np.asarray(any_ranks, dtype=int)
    metrics = {
        "method": method,
        "total_cases": int(exact_arr.size),
        "top1": float(np.mean(exact_arr <= 1)),
        "top3": float(np.mean(exact_arr <= 3)),
        "top5": float(np.mean(exact_arr <= 5)),
        "rank_le_50": float(np.mean(exact_arr <= 50)),
        "rank_gt_50_cases": int(np.sum(exact_arr > 50)),
        "gold_in_top50_but_rank_gt5_cases": int(np.sum((exact_arr > 5) & (exact_arr <= 50))),
        "median_rank": float(np.median(exact_arr)),
        "mean_rank": float(np.mean(exact_arr)),
        "any_label_at_1": float(np.mean(any_arr <= 1)),
        "any_label_at_3": float(np.mean(any_arr <= 3)),
        "any_label_at_5": float(np.mean(any_arr <= 5)),
        "any_label_at_50": float(np.mean(any_arr <= 50)),
    }
    return metrics, pd.DataFrame(rows)


def baseline_overall_from_details(details: pd.DataFrame) -> dict[str, Any]:
    ranks = pd.to_numeric(details["true_rank"], errors="coerce").fillna(9999).to_numpy(dtype=int)
    return {
        "method": "HGNN_exact_baseline",
        "total_cases": int(len(ranks)),
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "rank_le_50": float(np.mean(ranks <= 50)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def compare_to_baseline(baseline_per_dataset: pd.DataFrame, aug_per_dataset: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["dataset_name", "top1", "top3", "top5", "rank_le_50", "mean_rank", "median_rank"]
    base = baseline_per_dataset[base_cols].copy()
    base = base.rename(columns={col: f"baseline_{col}" for col in base_cols if col != "dataset_name"})
    aug_cols = ["dataset_name", "top1", "top3", "top5", "rank_le_50", "mean_rank", "median_rank", "any_label_at_5"]
    aug = aug_per_dataset[aug_cols].copy()
    aug = aug.rename(columns={col: f"similar_case_{col}" for col in aug_cols if col != "dataset_name"})
    out = base.merge(aug, on="dataset_name", how="outer")
    for metric in ["top1", "top3", "top5", "rank_le_50"]:
        out[f"delta_{metric}"] = out[f"similar_case_{metric}"] - out[f"baseline_{metric}"]
    return out


def write_reports(
    report_dir: Path,
    selected: dict[str, Any],
    baseline_overall: dict[str, Any],
    aug_overall: dict[str, Any],
    baseline_per_dataset: pd.DataFrame,
    aug_per_dataset: pd.DataFrame,
    delta: pd.DataFrame,
) -> None:
    overall = pd.DataFrame([baseline_overall, aug_overall])
    write_csv(overall, report_dir / "similar_case_all_overall_metrics.csv")
    write_csv(aug_per_dataset, report_dir / "similar_case_all_per_dataset_metrics.csv")
    write_csv(delta, report_dir / "similar_case_all_delta_vs_baseline.csv")
    write_text(
        report_dir / "similar_case_all_per_dataset_metrics.md",
        "\n".join(
            [
                "# SimilarCase-Aug All-Dataset Per-Dataset Metrics",
                "",
                "## Overall",
                df_to_markdown(overall),
                "",
                "## SimilarCase-Aug per dataset",
                df_to_markdown(aug_per_dataset),
                "",
                "## Delta vs HGNN exact baseline",
                df_to_markdown(delta),
            ]
        ),
    )
    md = [
        "# SimilarCase-Aug All-Dataset Fixed Evaluation",
        "",
        "## 1. Protocol",
        f"- method: HGNN + SimilarCase-Aug",
        f"- mode: {selected.get('status', '')}",
        f"- selected_by: {selected.get('selected_by', 'validation')}",
        f"- test_protocol: {selected.get('test_protocol', 'fixed_once')}",
        f"- similar_case_topk: {selected['similar_case_topk']}",
        f"- similar_case_weight: {selected['similar_case_weight']}",
        f"- score_type: {selected['similar_case_score_type']}",
        "- candidate sources: HGNN top50 + SimilarCase only",
        "- SimilarCase library: train/library namespace only",
        "- test SimilarCase retrieval: 按 dataset 分组计算后合并，保持 mimic 子集与原主线口径一致",
        "- any-label metrics are supplementary only",
        "",
        "## 2. Overall Metrics",
        df_to_markdown(overall),
        "",
        "## 3. Per-Dataset Delta",
        df_to_markdown(delta),
        "",
        "## 4. Judgment",
        f"- overall top5 delta: {float(aug_overall['top5']) - float(baseline_overall['top5']):.4f}",
        f"- overall rank<=50 delta: {float(aug_overall['rank_le_50']) - float(baseline_overall['rank_le_50']):.4f}",
        "- 如果某些小数据集波动较大，应优先看 case count 和 exact metric，不用 any-label 替代正式结论。",
    ]
    write_text(report_dir / "final_similar_case_all_report.md", "\n".join(md))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.candidate_output_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_yaml_config(args.data_config_path)
    train_config = load_yaml_config(args.train_config_path)
    train_table, val_table, test_table = load_all_case_tables(data_config, args.data_config_path, train_config)
    val_candidates = load_candidates(args.validation_candidates_path)
    test_candidates = load_candidates(args.test_candidates_path)
    baseline_details = pd.read_csv(args.baseline_details_path)
    baseline_per_dataset = pd.read_csv(args.baseline_per_dataset_path)
    write_candidate_audit(args.output_dir, test_candidates, val_candidates, baseline_details, baseline_per_dataset)

    test_hgnn = hgnn_source(test_candidates, test_table)
    if args.mode == "frozen":
        selected = load_frozen_config(args.frozen_config_path)
    else:
        topks = parse_int_list(args.similar_case_topk)
        weights = parse_float_list(args.similar_case_weight)
        score_types = parse_score_types(args.score_type)
        val_hgnn = hgnn_source(val_candidates, val_table)
        val_sim = compute_similar_matches(val_table, train_table, max(topks))
        selected, val_selection = select_on_validation(val_hgnn, val_sim, val_table, topks, weights, score_types)
        selected["selected_by"] = "validation"
        selected["test_protocol"] = "fixed_once"
        write_csv(val_selection, args.output_dir / "similar_case_all_validation_selection.csv")

    test_sim = compute_similar_matches_by_dataset(test_table, train_table, int(selected["similar_case_topk"]))
    test_sim_source = similar_source(test_sim, int(selected["similar_case_topk"]), str(selected["similar_case_score_type"]), test_table)
    ranked = combine(test_hgnn, test_sim_source, similar_weight=float(selected["similar_case_weight"]))
    ranked_path = args.candidate_output_dir / "similar_case_all_fixed_ranked_candidates.csv"
    write_csv(ranked, ranked_path)

    exact_gold_by_case = dict(zip(baseline_details["case_id"].astype(str), baseline_details["true_label"].astype(str)))
    aug_overall, case_ranks = evaluate_with_exact_gold(ranked, test_table, exact_gold_by_case, "HGNN_SimilarCase_Aug")
    baseline_overall = baseline_overall_from_details(baseline_details)
    aug_per_dataset = per_dataset_metrics(case_ranks, test_table, "HGNN_SimilarCase_Aug")
    delta = compare_to_baseline(baseline_per_dataset, aug_per_dataset)
    write_reports(args.output_dir, selected, baseline_overall, aug_overall, baseline_per_dataset, aug_per_dataset, delta)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "selected": selected,
                "baseline_overall": baseline_overall,
                "similar_case_overall": aug_overall,
                "ranked_candidates": str(ranked_path.resolve()),
                "report_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
