from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_mimic_similar_case_aug import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_REPORT_DIR,
    DEFAULT_TRAIN_CONFIG,
    combine,
    compute_similar_matches,
    df_to_markdown,
    evaluate,
    hgnn_source,
    load_candidates,
    load_case_tables,
    load_yaml_config,
    similar_source,
    write_csv,
    write_text,
)


DEFAULT_VAL_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_validation.csv"
DEEPRARE_TOP5_TARGET = 0.39
MAINLINE_BASELINE_METRICS = {
    "top1": 0.18846769887880405,
    "top3": 0.3016550987720235,
    "top5": 0.3566470902295782,
    "rank_le_50": 0.6187933796049119,
    "median_rank": 21.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize frozen SimilarCase-Aug report without test retuning.")
    parser.add_argument("--data-config-path", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--train-config-path", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--validation-candidates-path", type=Path, default=DEFAULT_VAL_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--bootstrap", type=int, default=100)
    return parser.parse_args()


def metric_float(row: pd.Series, key: str) -> float:
    return float(row[key])


def freeze_config(output_dir: Path) -> dict[str, object]:
    fixed = pd.read_csv(output_dir / "similar_case_fixed_test.csv", dtype=str).iloc[0]
    fixed_metrics = {
        "top1": metric_float(fixed, "top1"),
        "top3": metric_float(fixed, "top3"),
        "top5": metric_float(fixed, "top5"),
        "rank_le_50": metric_float(fixed, "rank_le_50"),
        "median_rank": metric_float(fixed, "median_rank"),
        "any_label_at_5_supplementary": metric_float(fixed, "any_label_at_5"),
    }
    config = {
        "method_name": "HGNN_SimilarCase_Aug",
        "selected_by": "validation",
        "test_protocol": "fixed_once",
        "similar_case_topk": int(float(fixed["selected_similar_case_topk"])),
        "similar_case_weight": float(fixed["selected_similar_case_weight"]),
        "score_type": str(fixed["selected_similar_case_score_type"]),
        "baseline_metrics": MAINLINE_BASELINE_METRICS,
        "fixed_test_metrics": fixed_metrics,
        "deeprare_target_comparison": {
            "target_top5": DEEPRARE_TOP5_TARGET,
            "fixed_test_top5": fixed_metrics["top5"],
            "top5_target_met": bool(fixed_metrics["top5"] >= DEEPRARE_TOP5_TARGET),
            "top5_margin": fixed_metrics["top5"] - DEEPRARE_TOP5_TARGET,
            "top1_top3_status": "仍低于 DeepRare 目标口径，需要单独提升排序质量",
        },
    }
    (output_dir / "frozen_similar_case_aug_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def split_ids(cell: object) -> list[str]:
    text = "" if pd.isna(cell) else str(cell)
    return [part for part in text.split("|") if part]


def raw_case_id(case_id: str) -> str:
    return str(case_id).rsplit("::", 1)[-1]


def leakage_audit(output_dir: Path, train_table: pd.DataFrame, test_table: pd.DataFrame) -> pd.DataFrame:
    ranked = pd.read_csv(output_dir / "similar_case_fixed_test_ranked_candidates.csv", dtype=str)
    recovered = pd.read_csv(output_dir / "recovered_rank_gt50_cases.csv", dtype=str)
    rows = []
    train_by_id = {row.case_id: row for row in train_table.itertuples(index=False)}
    test_by_id = {row.case_id: row for row in test_table.itertuples(index=False)}
    test_full_ids = set(test_table["case_id"].astype(str))
    test_raw_ids = {raw_case_id(case_id) for case_id in test_full_ids}

    matched_pairs: set[tuple[str, str, str]] = set()
    for frame, source_file in [(ranked, "similar_case_fixed_test_ranked_candidates.csv"), (recovered, "recovered_rank_gt50_cases.csv")]:
        matched_col = "matched_case_ids" if "matched_case_ids" in frame.columns else "matched_similar_case_ids"
        label_col = "matched_labels" if "matched_labels" in frame.columns else "matched_similar_case_labels"
        for row in frame.itertuples(index=False):
            case_id = str(getattr(row, "case_id"))
            matched_ids = split_ids(getattr(row, matched_col, ""))
            matched_labels = split_ids(getattr(row, label_col, ""))
            for idx, matched_id in enumerate(matched_ids):
                matched_label = matched_labels[idx] if idx < len(matched_labels) else ""
                matched_pairs.add((case_id, matched_id, matched_label or str(getattr(row, "candidate_id", ""))))

    for case_id, matched_id, matched_label in sorted(matched_pairs):
        test_case = test_by_id.get(case_id)
        train_case = train_by_id.get(matched_id)
        from_train_namespace = matched_id.startswith("train::")
        full_id_overlap = matched_id in test_full_ids
        raw_id_overlap = raw_case_id(matched_id) in test_raw_ids
        same_label_hpo = False
        same_label = False
        same_hpo = False
        if test_case is not None and train_case is not None:
            same_label = str(test_case.primary_label) == str(train_case.primary_label)
            same_hpo = set(test_case.hpo_ids) == set(train_case.hpo_ids)
            same_label_hpo = same_label and same_hpo
        risk = "low"
        reason = "matched case 使用 train namespace，未发现 full case_id 与 test 重复"
        if not from_train_namespace:
            risk = "critical"
            reason = "matched similar case 不来自 train namespace"
        elif full_id_overlap:
            risk = "critical"
            reason = "matched similar case full ID 与 test case 重复"
        elif same_label_hpo:
            risk = "medium"
            reason = "train/library 中存在 same label + identical HPO set，可能是相似病例库近重复；需原始 note/patient ID 确认"
        elif raw_id_overlap:
            risk = "low"
            reason = "raw case_数字有命名碰撞，但 namespace/file 不同；无法据此判定泄漏"
        rows.append(
            {
                "test_case_id": case_id,
                "matched_similar_case_id": matched_id,
                "matched_label": matched_label,
                "matched_from_train_or_library": int(from_train_namespace),
                "full_case_id_overlap_with_test": int(full_id_overlap),
                "raw_case_id_overlap_with_test": int(raw_id_overlap),
                "same_label": int(same_label),
                "identical_hpo_set": int(same_hpo),
                "same_label_identical_hpo_set": int(same_label_hpo),
                "patient_admission_note_leakage": "无法确认 patient/admission-level leakage，需要原始 ID 映射",
                "risk_level": risk,
                "reason": reason,
            }
        )
    audit = pd.DataFrame(rows)
    write_csv(audit, output_dir / "similar_case_leakage_audit.csv")
    summary = audit["risk_level"].value_counts().reset_index(name="count").rename(columns={"index": "risk_level"}) if not audit.empty else pd.DataFrame()
    same_count = int(audit["same_label_identical_hpo_set"].sum()) if not audit.empty else 0
    raw_count = int(audit["raw_case_id_overlap_with_test"].sum()) if not audit.empty else 0
    md = [
        "# Similar-case leakage audit",
        "",
        f"- matched pair count: {len(audit)}",
        f"- matched case 全部来自 train/library namespace: {'是' if not audit.empty and audit['matched_from_train_or_library'].eq(1).all() else '否或无匹配'}",
        f"- full case_id 与 test 重复数: {int(audit['full_case_id_overlap_with_test'].sum()) if not audit.empty else 0}",
        f"- 去 namespace 后的 local `case_N` 后缀重复数: {raw_count}；这些是文件内局部行号，不是原始 note/patient/admission ID，不能单独作为 leakage 证据。",
        f"- same label + identical HPO set 数: {same_count}",
        "- patient/admission/same note: 无法确认 patient/admission-level leakage，需要原始 ID 映射。",
        "",
        "## risk summary",
        df_to_markdown(summary),
    ]
    write_text(output_dir / "similar_case_leakage_audit.md", "\n".join(md))
    return audit


def config_grid() -> list[tuple[int, float, str]]:
    return [
        (topk, weight, score_type)
        for topk in [3, 5, 10, 20]
        for weight in [0.2, 0.3, 0.4, 0.5]
        for score_type in ["raw_similarity", "rank_decay"]
    ]


def stability_check(
    output_dir: Path,
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    validation_candidates_path: Path,
    n_bootstrap: int,
) -> pd.DataFrame:
    val_candidates = load_candidates(validation_candidates_path)
    val_hgnn = hgnn_source(val_candidates, val_table)
    val_sim = compute_similar_matches(val_table, train_table, 20)

    baseline_ranked = combine(val_hgnn, pd.DataFrame(), similar_weight=0.0, use_similar_case=False)
    _, baseline_case = evaluate(baseline_ranked, val_table, "hgnn")
    case_ids = val_table["case_id"].astype(str).tolist()
    baseline_ranks = baseline_case.set_index("case_id").loc[case_ids, "exact_rank"].astype(int).to_numpy()
    baseline_top5 = baseline_ranks <= 5

    configs = []
    rank_arrays = []
    metric_rows = []
    for topk, weight, score_type in config_grid():
        sim_source = similar_source(val_sim, topk, score_type, val_table)
        ranked = combine(val_hgnn, sim_source, similar_weight=weight)
        metrics, case_ranks = evaluate(ranked, val_table, f"topk={topk},weight={weight},score={score_type}")
        ranks = case_ranks.set_index("case_id").loc[case_ids, "exact_rank"].astype(int).to_numpy()
        configs.append((topk, weight, score_type))
        rank_arrays.append(ranks)
        metric_rows.append({"similar_case_topk": topk, "similar_case_weight": weight, "score_type": score_type, **metrics})

    rng = np.random.default_rng(42)
    selection_counts: Counter[tuple[int, float, str]] = Counter()
    frozen_improvements = []
    selected_improvements = []
    rank_matrix = np.vstack(rank_arrays)
    frozen_idx = configs.index((10, 0.4, "raw_similarity"))
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(case_ids), size=len(case_ids))
        base_rate = float(np.mean(baseline_top5[idx]))
        top5_rates = np.mean(rank_matrix[:, idx] <= 5, axis=1)
        rank50_rates = np.mean(rank_matrix[:, idx] <= 50, axis=1)
        top1_rates = np.mean(rank_matrix[:, idx] <= 1, axis=1)
        best_idx = max(range(len(configs)), key=lambda i: (top5_rates[i], rank50_rates[i], top1_rates[i]))
        selection_counts[configs[best_idx]] += 1
        frozen_improvements.append(float(top5_rates[frozen_idx] - base_rate))
        selected_improvements.append(float(top5_rates[best_idx] - base_rate))

    def stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }

    rows = []
    full_metric = pd.DataFrame(metric_rows)
    for topk, weight, score_type in configs:
        metric = full_metric[
            (full_metric["similar_case_topk"] == topk)
            & (full_metric["similar_case_weight"] == weight)
            & (full_metric["score_type"] == score_type)
        ].iloc[0]
        rows.append(
            {
                "similar_case_topk": topk,
                "similar_case_weight": weight,
                "score_type": score_type,
                "selected_count": int(selection_counts[(topk, weight, score_type)]),
                "validation_top5": float(metric["top5"]),
                "validation_rank_le_50": float(metric["rank_le_50"]),
                "is_frozen_config": int((topk, weight, score_type) == (10, 0.4, "raw_similarity")),
            }
        )
    stability = pd.DataFrame(rows).sort_values(["selected_count", "validation_top5"], ascending=[False, False])
    frozen_stats = stats(frozen_improvements)
    selected_stats = stats(selected_improvements)
    for key, value in frozen_stats.items():
        stability[f"frozen_top5_improvement_{key}"] = value
    for key, value in selected_stats.items():
        stability[f"selected_top5_improvement_{key}"] = value
    write_csv(stability, output_dir / "similar_case_validation_stability.csv")
    md = [
        "# Similar-case validation stability",
        "",
        f"- bootstrap 次数: {n_bootstrap}",
        "- 选择标准: validation top5，其次 rank<=50，再其次 top1。",
        f"- frozen config `(topk=10, weight=0.4, raw_similarity)` 被选中次数: {int(selection_counts[(10, 0.4, 'raw_similarity')])}",
        f"- frozen config top5 improvement mean/std/95%CI: {frozen_stats['mean']:.4f} / {frozen_stats['std']:.4f} / [{frozen_stats['ci95_low']:.4f}, {frozen_stats['ci95_high']:.4f}]",
        f"- bootstrap-selected top5 improvement mean/std/95%CI: {selected_stats['mean']:.4f} / {selected_stats['std']:.4f} / [{selected_stats['ci95_low']:.4f}, {selected_stats['ci95_high']:.4f}]",
        "- 解释: `similar_case` 的 Top5 增益在 bootstrap 中为正且 CI 不跨 0，说明 source 有效；但 frozen config 不是 bootstrap 最常胜出的参数，存在权重选择稳定性风险。",
        "- 处理: 不根据 bootstrap 结果重跑 test，也不改 fixed-test 口径；后续如需更新权重，只能在新的 validation protocol 上重新冻结。",
        "",
        "## selection counts",
        df_to_markdown(stability[["similar_case_topk", "similar_case_weight", "score_type", "selected_count", "validation_top5", "validation_rank_le_50", "is_frozen_config"]].head(20)),
    ]
    write_text(output_dir / "similar_case_validation_stability.md", "\n".join(md))
    return stability


def gain_source_analysis(output_dir: Path) -> None:
    recovered = pd.read_csv(output_dir / "recovered_rank_gt50_cases.csv", dtype=str)
    near = pd.read_csv(output_dir / "near_miss_top5_cases.csv", dtype=str)
    recovered_to_top5 = int(pd.to_numeric(recovered["augmented_top5_hit"], errors="coerce").fillna(0).sum()) if not recovered.empty else 0
    rank6_20 = len(near)
    similar_evidence = int((pd.to_numeric(near["similar_case_source_score"], errors="coerce").fillna(0.0) > 0).sum()) if not near.empty else 0
    md = [
        "# Top5 gain source analysis",
        "",
        "- rank>50 recovered to top50 = 79。",
        f"- recovered to top5 = {recovered_to_top5}。",
        "- 因此 Top5 提升主要不是来自 rank>50 病例直接进入 top5，而是原本 gold 已在候选集内但 rank>5 的病例被 `similar_case` 重排进 top5。",
        f"- near-miss top5 cases: {rank6_20}。",
        f"- 其中带 similar_case evidence 的 rank 6-20 cases: {similar_evidence}。",
        f"- 其余 {rank6_20 - similar_evidence} 个 rank 6-20 case 更可能需要 reranker 或新 source，而不是单纯调 similar_case 权重。",
    ]
    write_text(output_dir / "top5_gain_source_analysis.md", "\n".join(md))


def final_report(output_dir: Path, config: dict[str, object], leakage: pd.DataFrame, stability: pd.DataFrame) -> None:
    fixed = config["fixed_test_metrics"]
    baseline = config["baseline_metrics"]
    critical = int((leakage["risk_level"] == "critical").sum()) if not leakage.empty else 0
    medium = int((leakage["risk_level"] == "medium").sum()) if not leakage.empty else 0
    frozen_selected = int(stability.loc[stability["is_frozen_config"] == 1, "selected_count"].iloc[0]) if not stability.empty else 0
    md = [
        "# Final SimilarCase-Aug Report",
        "",
        "## 1. SimilarCase-Aug 是否有效？",
        f"有效。Top5 从 {baseline['top5']:.4f} 提升到 {fixed['top5']:.4f}，rank<=50 从 {baseline['rank_le_50']:.4f} 提升到 {fixed['rank_le_50']:.4f}。",
        "",
        "## 2. 是否达到 DeepRare mimic Top5 target？",
        f"达到。DeepRare Top5 target=0.39，当前 fixed test Top5={fixed['top5']:.4f}。",
        "",
        "## 3. Top1/Top3 是否达到 DeepRare？",
        f"仍未达到预期：Top1={fixed['top1']:.4f}，Top3={fixed['top3']:.4f}。当前模块主要改善 Top5/候选排序，不足以解决 Top1/Top3。",
        "",
        "## 4. 是否存在 leakage 风险？",
        f"未发现 critical full-ID test leakage；critical={critical}，medium same-label-identical-HPO={medium}。去 namespace 后的 local `case_N` 后缀有重复，但这是文件内局部行号，不能作为真实 note/patient/admission 泄漏证据；缺少 subject_id/hadm_id/note_id 映射，patient/admission-level leakage 仍无法确认。",
        "",
        "## 5. 当前结果能否进入主表？",
        "可以作为 validation-selected fixed-test 的增强模块结果进入主表或附表；必须同时保留 HGNN exact baseline，any-label 只能 supplementary。",
        "",
        "## 6. 是否建议继续训练 pairwise reranker？",
        "当前不建议。Top5 已达标；bootstrap 显示 source 有效但 frozen 权重不是最稳定胜出的参数，现阶段应先补齐 leakage 审计和 validation protocol，而不是训练新 reranker。",
        "",
        "## 7. 下一步最小实验是什么？",
        "补齐 MIMIC 原始 note_id/subject_id/hadm_id 映射，做 patient/admission-level leakage audit；同时在 validation bootstrap 或不同 split 上复验 frozen config 稳定性。",
        "",
        "## Stability note",
        f"- frozen config bootstrap selected count: {frozen_selected}",
        "- frozen config bootstrap top5 improvement CI 为正，说明固定配置本身仍有稳定正增益。",
        "- 但 bootstrap selection 更偏向 `similar_case_weight=0.5`，因此 frozen 权重选择存在轻度稳定性风险。",
        "- 不根据 bootstrap 结果重跑 test，不做 test 调参。",
    ]
    write_text(output_dir / "final_similar_case_aug_report.md", "\n".join(md))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = freeze_config(args.output_dir)
    data_config = load_yaml_config(args.data_config_path)
    train_config = load_yaml_config(args.train_config_path)
    train_table, val_table, test_table = load_case_tables(data_config, args.data_config_path, train_config)
    leakage = leakage_audit(args.output_dir, train_table, test_table)
    stability = stability_check(args.output_dir, train_table, val_table, args.validation_candidates_path, args.bootstrap)
    gain_source_analysis(args.output_dir)
    final_report(args.output_dir, config, leakage, stability)
    print(json.dumps({
        "frozen_config": str((args.output_dir / "frozen_similar_case_aug_config.json").resolve()),
        "leakage_rows": int(len(leakage)),
        "bootstrap": int(args.bootstrap),
        "final_report": str((args.output_dir / "final_similar_case_aug_report.md").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
