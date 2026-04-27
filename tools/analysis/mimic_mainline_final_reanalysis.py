from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "reports" / "mimic_diagnosis"
FINAL_METRICS = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "mainline_final_metrics.csv"
FINAL_METRICS_WITH_SOURCES = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "mainline_final_metrics_with_sources.csv"
FINAL_CASE_RANKS = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "mainline_final_case_ranks.csv"
STAGE6_METRICS = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage6_mimic_similar_case" / "similar_case_fixed_test.csv"
STAGE6_SELECTION = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage6_mimic_similar_case" / "similar_case_val_selection.csv"
STAGE6_RECOVERED = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage6_mimic_similar_case" / "recovered_rank_gt50_cases.csv"
STAGE6_NEAR_MISS = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage6_mimic_similar_case" / "near_miss_top5_cases.csv"
RUN_MANIFEST = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "run_manifest.json"
DOC_FROZEN_CONFIG = PROJECT_ROOT / "reports" / "mimic_next" / "frozen_similar_case_aug_config.json"
BASELINE_OVERLAP_CASE_LEVEL = REPORT_DIR / "mimic_hpo_hyperedge_overlap_case_level.csv"


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无记录_"
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    view = view.fillna("").astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def metric_from_rank(series: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(series, errors="coerce").fillna(9999).astype(int).to_numpy()
    return {
        "num_cases": int(arr.size),
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "rank_le_10": float(np.mean(arr <= 10)),
        "rank_le_20": float(np.mean(arr <= 20)),
        "rank_le_50": float(np.mean(arr <= 50)),
        "rank_gt_50_count": int(np.sum(arr > 50)),
        "rank_gt_50_ratio": float(np.mean(arr > 50)),
        "gold_in_top50_but_rank_gt5_count": int(np.sum((arr > 5) & (arr <= 50))),
        "gold_in_top50_but_rank_gt5_ratio": float(np.mean((arr > 5) & (arr <= 50))),
        "gold_in_top5_but_not_top1_count": int(np.sum((arr > 1) & (arr <= 5))),
        "gold_in_top5_but_not_top1_ratio": float(np.mean((arr > 1) & (arr <= 5))),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final_metrics = pd.read_csv(FINAL_METRICS)
    final_sources = pd.read_csv(FINAL_METRICS_WITH_SOURCES)
    stage6 = pd.read_csv(STAGE6_METRICS)
    ranks = pd.read_csv(FINAL_CASE_RANKS, dtype={"case_id": str, "dataset": str, "gold_id": str})
    mimic = ranks[ranks["dataset"].eq("mimic_test_recleaned_mondo_hpo_rows")].copy()
    mimic["baseline_rank"] = pd.to_numeric(mimic["baseline_rank"], errors="coerce").fillna(9999).astype(int)
    mimic["final_rank"] = pd.to_numeric(mimic["final_rank"], errors="coerce").fillna(9999).astype(int)

    baseline_summary = {"rank_source": "HGNN exact baseline", **metric_from_rank(mimic["baseline_rank"])}
    final_summary = {"rank_source": "mainline final SimilarCase-Aug", **metric_from_rank(mimic["final_rank"])}
    summary = pd.DataFrame([baseline_summary, final_summary])
    delta = {
        "top1_delta": final_summary["top1"] - baseline_summary["top1"],
        "top3_delta": final_summary["top3"] - baseline_summary["top3"],
        "top5_delta": final_summary["top5"] - baseline_summary["top5"],
        "rank_le_50_delta": final_summary["rank_le_50"] - baseline_summary["rank_le_50"],
        "rank_gt_50_count_delta": final_summary["rank_gt_50_count"] - baseline_summary["rank_gt_50_count"],
        "top50_late_count_delta": final_summary["gold_in_top50_but_rank_gt5_count"] - baseline_summary["gold_in_top50_but_rank_gt5_count"],
    }

    transitions = {
        "baseline_gt50_to_final_le50": int(((mimic["baseline_rank"] > 50) & (mimic["final_rank"] <= 50)).sum()),
        "baseline_le50_to_final_gt50": int(((mimic["baseline_rank"] <= 50) & (mimic["final_rank"] > 50)).sum()),
        "baseline_gt50_to_final_le5": int(((mimic["baseline_rank"] > 50) & (mimic["final_rank"] <= 5)).sum()),
        "baseline_6_50_to_final_le5": int(((mimic["baseline_rank"].between(6, 50)) & (mimic["final_rank"] <= 5)).sum()),
        "baseline_gt5_to_final_le5": int(((mimic["baseline_rank"] > 5) & (mimic["final_rank"] <= 5)).sum()),
        "baseline_le5_to_final_gt5": int(((mimic["baseline_rank"] <= 5) & (mimic["final_rank"] > 5)).sum()),
        "baseline_not_top1_to_final_top1": int(((mimic["baseline_rank"] > 1) & (mimic["final_rank"] <= 1)).sum()),
        "baseline_top1_to_final_not_top1": int(((mimic["baseline_rank"] <= 1) & (mimic["final_rank"] > 1)).sum()),
        "improved_rank": int((mimic["final_rank"] < mimic["baseline_rank"]).sum()),
        "worsened_rank": int((mimic["final_rank"] > mimic["baseline_rank"]).sum()),
        "unchanged_rank": int((mimic["final_rank"] == mimic["baseline_rank"]).sum()),
    }
    transition_df = pd.DataFrame([transitions])

    doc_config = read_json(DOC_FROZEN_CONFIG)
    current_selected = stage6.iloc[0].to_dict() if not stage6.empty else {}
    config_compare = pd.DataFrame(
        [
            {
                "source": "docx / reports/mimic_next frozen config",
                "topk": doc_config.get("similar_case_topk"),
                "weight": doc_config.get("similar_case_weight"),
                "score_type": doc_config.get("score_type"),
                "top1": doc_config.get("fixed_test_metrics", {}).get("top1"),
                "top3": doc_config.get("fixed_test_metrics", {}).get("top3"),
                "top5": doc_config.get("fixed_test_metrics", {}).get("top5"),
                "rank_le_50": doc_config.get("fixed_test_metrics", {}).get("rank_le_50"),
            },
            {
                "source": "current outputs/mainline_full_pipeline",
                "topk": current_selected.get("selected_similar_case_topk"),
                "weight": current_selected.get("selected_similar_case_weight"),
                "score_type": current_selected.get("selected_similar_case_score_type"),
                "top1": current_selected.get("top1"),
                "top3": current_selected.get("top3"),
                "top5": current_selected.get("top5"),
                "rank_le_50": current_selected.get("rank_le_50"),
            },
        ]
    )

    bucket_rows: list[dict[str, Any]] = []
    if BASELINE_OVERLAP_CASE_LEVEL.is_file():
        overlap = pd.read_csv(BASELINE_OVERLAP_CASE_LEVEL, dtype={"namespaced_case_id": str})
        merged = overlap.merge(
            mimic[["case_id", "baseline_rank", "final_rank"]],
            left_on="namespaced_case_id",
            right_on="case_id",
            how="inner",
        )
        for bucket_col in ["exact_hpo_overlap_count_bucket", "case_hpo_count_bucket", "gold_in_top50_bucket"]:
            for bucket, group in merged.groupby(bucket_col, sort=False):
                base = metric_from_rank(group["baseline_rank"])
                final = metric_from_rank(group["final_rank"])
                bucket_rows.append(
                    {
                        "bucket_type": bucket_col,
                        "bucket": bucket,
                        "num_cases": int(len(group)),
                        "baseline_top5": base["top5"],
                        "final_top5": final["top5"],
                        "top5_delta": final["top5"] - base["top5"],
                        "baseline_rank_le_50": base["rank_le_50"],
                        "final_rank_le_50": final["rank_le_50"],
                        "rank_le_50_delta": final["rank_le_50"] - base["rank_le_50"],
                    }
                )
    bucket_df = pd.DataFrame(bucket_rows)

    summary.to_csv(REPORT_DIR / "mimic_mainline_final_rank_decomposition.csv", index=False, encoding="utf-8-sig")
    transition_df.to_csv(REPORT_DIR / "mimic_mainline_final_transition_summary.csv", index=False, encoding="utf-8-sig")
    config_compare.to_csv(REPORT_DIR / "mimic_similar_case_config_compare.csv", index=False, encoding="utf-8-sig")
    if not bucket_df.empty:
        bucket_df.to_csv(REPORT_DIR / "mimic_mainline_final_overlap_bucket_delta.csv", index=False, encoding="utf-8-sig")

    final_mimic_row = final_metrics[final_metrics["dataset"].eq("mimic_test_recleaned_mondo_hpo_rows")]
    final_source_row = final_sources[final_sources["dataset"].eq("mimic_test_recleaned_mondo_hpo_rows")]
    recovered_count = len(pd.read_csv(STAGE6_RECOVERED)) if STAGE6_RECOVERED.is_file() else None
    near_miss_count = len(pd.read_csv(STAGE6_NEAR_MISS)) if STAGE6_NEAR_MISS.is_file() else None
    run_manifest = read_json(RUN_MANIFEST)

    lines = [
        "# mimic_test Mainline Final Reanalysis",
        "",
        "## 口径修正",
        "- 本报告使用 `outputs/mainline_full_pipeline/mainline_final_case_ranks.csv` 的 `final_rank`，不是只看 `stage3_exact_eval/exact_details.csv` 的 baseline `true_rank`。",
        "- `mimic_test_recleaned_mondo_hpo_rows` 在最终汇总中应用的模块是 `similar_case_fixed_test`。",
        "- 没有重跑训练；没有覆盖 mainline 输出。",
        "",
        "## 当前 mainline final 指标",
        to_markdown(final_mimic_row),
        "",
        "## 指标来源",
        to_markdown(final_source_row),
        "",
        "## Baseline vs Final",
        to_markdown(summary),
        "",
        "## Delta",
        to_markdown(pd.DataFrame([delta])),
        "",
        "## Rank Transition",
        to_markdown(transition_df),
        "",
        "## 文档 frozen config vs 当前 mainline 输出",
        to_markdown(config_compare),
        "",
        "## Overlap Bucket Delta",
        to_markdown(bucket_df) if not bucket_df.empty else "_未找到 overlap case-level 诊断文件_",
        "",
        "## 机制判断",
        f"- 当前 mainline final 的 `rank<=50` 从 {baseline_summary['rank_le_50']:.4f} 提升到 {final_summary['rank_le_50']:.4f}，净提升 {delta['rank_le_50_delta']:.4f}。",
        f"- baseline rank>50 被拉回 final rank<=50 的样本为 {transitions['baseline_gt50_to_final_le50']}；但 baseline rank>50 直接进入 final top5 的样本为 {transitions['baseline_gt50_to_final_le5']}。",
        f"- baseline rank 6-50 被推入 final top5 的样本为 {transitions['baseline_6_50_to_final_le5']}，说明 Top5 提升主要来自 top50 内局部重排，而不是真正解决大规模候选召回。",
        f"- 当前仍有 final rank>50 的病例 {final_summary['rank_gt_50_count']} 个，占比 {final_summary['rank_gt_50_ratio']:.4f}；这部分仍不是 reranker 或 hard negative 单独能解决的问题。",
        f"- `stage6` recovered case 文件行数为 {recovered_count}，near-miss 文件行数为 {near_miss_count}。",
        "- 因此，SimilarCase-Aug 已经是有效的 no-train 多视图/病例库证据增强模块；后续第一优先级仍应沿着 candidate expansion + evidence rerank 做，而不是直接转图对比学习。",
        "",
        "## 复现命令",
        "- 已有 checkpoint 评估链路：`D:\\python\\python.exe tools\\run_full_mainline_pipeline.py --config-path configs\\mainline_full_pipeline.yaml --mode eval_only`",
        "- 完整训练链路：`D:\\python\\python.exe tools\\run_full_mainline_pipeline.py --config-path configs\\mainline_full_pipeline.yaml --mode full`",
        "- 本复核脚本：`D:\\python\\python.exe tools\\analysis\\mimic_mainline_final_reanalysis.py`",
        "",
        "## Checkpoint 一致性",
        f"- run_manifest finetune_checkpoint: `{run_manifest.get('finetune_checkpoint', '')}`",
        f"- test candidate checkpoint: `{run_manifest.get('test_candidates_metadata', {}).get('checkpoint_path', '')}`",
    ]
    write_md(REPORT_DIR / "mimic_mainline_final_reanalysis.md", lines)
    print(json.dumps({"report": str((REPORT_DIR / "mimic_mainline_final_reanalysis.md").resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
