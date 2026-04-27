from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "deeprare_target_gap"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "deeprare_target_gap"

MAINLINE_DIR = PROJECT_ROOT / "outputs" / "mainline_full_pipeline"
FINAL_METRICS_WITH_SOURCES = MAINLINE_DIR / "mainline_final_metrics_with_sources.csv"
FINAL_CASE_RANKS = MAINLINE_DIR / "mainline_final_case_ranks.csv"
RUN_MANIFEST = MAINLINE_DIR / "run_manifest.json"
MAINLINE_CONFIG = PROJECT_ROOT / "configs" / "mainline_full_pipeline.yaml"


DEEPRARE_TARGETS: list[dict[str, Any]] = [
    {
        "deeprare_dataset_name": "RareBench-MME",
        "target_recall1": 0.78,
        "target_recall3": 0.85,
        "target_recall5": 0.90,
        "project_dataset_candidate": "MME",
        "mapping_status": "exact_match",
        "notes": "当前测试配置中存在 MME.xlsx，按 RareBench-MME 的对应候选处理；样本量很小，解释时需标注 high_variance。",
    },
    {
        "deeprare_dataset_name": "RareBench-HMS",
        "target_recall1": 0.57,
        "target_recall3": 0.65,
        "target_recall5": 0.71,
        "project_dataset_candidate": "HMS",
        "mapping_status": "exact_match",
        "notes": "当前项目测试集中名称为 HMS，可与 RareBench-HMS 对齐；样本量很小，解释时需标注 high_variance。",
    },
    {
        "deeprare_dataset_name": "RareBench-LIRICAL",
        "target_recall1": 0.56,
        "target_recall3": 0.65,
        "target_recall5": 0.68,
        "project_dataset_candidate": "LIRICAL",
        "mapping_status": "exact_match",
        "notes": "当前项目测试集中名称为 LIRICAL，可与 RareBench-LIRICAL 对齐。",
    },
    {
        "deeprare_dataset_name": "RareBench-RAMEDIS",
        "target_recall1": 0.73,
        "target_recall3": 0.83,
        "target_recall5": 0.85,
        "project_dataset_candidate": "RAMEDIS",
        "mapping_status": "exact_match",
        "notes": "当前项目测试集中名称为 RAMEDIS，可与 RareBench-RAMEDIS 对齐。",
    },
    {
        "deeprare_dataset_name": "MIMIC-IV-Rare",
        "target_recall1": 0.29,
        "target_recall3": 0.37,
        "target_recall5": 0.39,
        "project_dataset_candidate": "mimic_test_recleaned_mondo_hpo_rows",
        "mapping_status": "exact_match",
        "notes": "当前 mainline 的 mimic test 数据集；只使用 strict exact Recall，不使用 supplementary any-label 指标。",
    },
    {
        "deeprare_dataset_name": "MyGene",
        "target_recall1": 0.76,
        "target_recall3": 0.80,
        "target_recall5": 0.81,
        "project_dataset_candidate": "MyGene2",
        "mapping_status": "approximate_match",
        "notes": "项目数据集名称为 MyGene2，作为当前可用的 MyGene 近似对应数据集，标记为 approximate_match。",
    },
    {
        "deeprare_dataset_name": "DDD",
        "target_recall1": 0.48,
        "target_recall3": 0.60,
        "target_recall5": 0.63,
        "project_dataset_candidate": "DDD",
        "mapping_status": "exact_match",
        "notes": "当前项目测试集中名称为 DDD，可直接对齐。",
    },
    {
        "deeprare_dataset_name": "Xinhua Hosp.",
        "target_recall1": 0.58,
        "target_recall3": 0.71,
        "target_recall5": 0.74,
        "project_dataset_candidate": "",
        "mapping_status": "unavailable",
        "notes": "configs/data_llldataset_eval.yaml 和当前 final mainline metrics 中没有 Xinhua Hosp. 对应数据集，不做硬映射。",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"

    rendered = df.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            if col == "num_cases" or col.endswith("_case_gap") or col.endswith("_count"):
                rendered[col] = rendered[col].map(
                    lambda x: "" if pd.isna(x) else str(int(float(x)))
                )
            else:
                rendered[col] = rendered[col].map(lambda x: fmt_float(x, float_digits))
        else:
            rendered[col] = rendered[col].fillna("").astype(str)

    headers = list(rendered.columns)
    rows = rendered.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def write_csv_and_md(
    df: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
    title: str,
    notes: list[str] | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [f"# {title}", ""]
    if notes:
        lines.extend(notes)
        lines.append("")
    lines.append(markdown_table(df))
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def case_gap(target: float, current: float | None, num_cases: int | None) -> int | None:
    if current is None or num_cases is None:
        return None
    return int(math.ceil(max(target - current, 0.0) * num_cases))


def gap_value(target: float, current: float | None) -> float | None:
    if current is None:
        return None
    return max(target - current, 0.0)


def target_status(row: dict[str, Any]) -> str:
    if row["mapping_status"] == "unavailable" or row.get("current_top1") is None:
        return "unavailable"
    top1 = row["current_top1"] >= row["target_top1"]
    top3 = row["current_top3"] >= row["target_top3"]
    top5 = row["current_top5"] >= row["target_top5"]
    if top1 and top3 and top5:
        return "all_reached"
    if top3 and top5:
        return "top3_top5_reached"
    if top5:
        return "top5_reached_only"
    return "far_from_target"


def priority_level(status: str, num_cases: int | None, dataset: str) -> str:
    if status in {"unavailable", "all_reached"}:
        return "P3"
    if dataset in {"MIMIC-IV-Rare", "DDD"}:
        return "P0"
    if num_cases is not None and num_cases < 60:
        return "P2"
    if status in {"top5_reached_only", "top3_top5_reached"}:
        return "P1"
    return "P1"


def normalize_mainline_metrics(metrics: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage_configs = manifest.get("stage_configs", {})

    for _, row in metrics.iterrows():
        dataset = str(row["dataset"])
        if dataset == "ALL":
            continue

        module = str(row.get("module_applied", ""))
        if module == "ddd_validation_selected_grid_rerank":
            validation_flag = "yes_validation_selected_fixed_test"
            notes = "DDD grid rerank 权重在 validation 上选择，然后在 test 上固定评估。"
        elif module == "similar_case_fixed_test":
            validation_flag = "yes_validation_selected_fixed_test"
            notes = "SimilarCase-Aug 配置在 validation 上选择，然后在 mimic test 上固定评估；这里只记录 strict exact 指标。"
        else:
            validation_flag = "yes_fixed_mainline_test"
            notes = "HGNN exact baseline 来自当前固定 mainline checkpoint；没有 dataset-specific test-side tuning。"

        rows.append(
            {
                "project_dataset": dataset,
                "num_cases": int(row["cases"]),
                "current_top1": float(row["top1"]),
                "current_top3": float(row["top3"]),
                "current_top5": float(row["top5"]),
                "current_rank_le_50": float(row["rank_le_50"]),
                "module_applied": module,
                "source_result_path": row.get("source_result_path", ""),
                "checkpoint_path": row.get("checkpoint_path", manifest.get("finetune_checkpoint", "")),
                "config_path": row.get("train_config_path", stage_configs.get("finetune", "")),
                "is_validation_selected_fixed_test": validation_flag,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


def build_gap_analysis(targets: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    current_by_dataset = {row["project_dataset"]: row for _, row in current.iterrows()}
    rows: list[dict[str, Any]] = []

    for _, target in targets.iterrows():
        project_dataset = target["project_dataset_candidate"]
        cur = current_by_dataset.get(project_dataset)
        num_cases = int(cur["num_cases"]) if cur is not None else None
        top1 = float(cur["current_top1"]) if cur is not None else None
        top3 = float(cur["current_top3"]) if cur is not None else None
        top5 = float(cur["current_top5"]) if cur is not None else None
        rank50 = float(cur["current_rank_le_50"]) if cur is not None else None

        row = {
            "deeprare_dataset_name": target["deeprare_dataset_name"],
            "project_dataset": project_dataset,
            "mapping_status": target["mapping_status"],
            "num_cases": num_cases,
            "current_top1": top1,
            "target_top1": float(target["target_recall1"]),
            "top1_gap": gap_value(float(target["target_recall1"]), top1),
            "top1_case_gap": case_gap(float(target["target_recall1"]), top1, num_cases),
            "current_top3": top3,
            "target_top3": float(target["target_recall3"]),
            "top3_gap": gap_value(float(target["target_recall3"]), top3),
            "top3_case_gap": case_gap(float(target["target_recall3"]), top3, num_cases),
            "current_top5": top5,
            "target_top5": float(target["target_recall5"]),
            "top5_gap": gap_value(float(target["target_recall5"]), top5),
            "top5_case_gap": case_gap(float(target["target_recall5"]), top5, num_cases),
            "current_rank_le_50": rank50,
        }
        row["target_status"] = target_status(row)
        row["priority_level"] = priority_level(row["target_status"], num_cases, row["deeprare_dataset_name"])

        notes: list[str] = []
        if row["mapping_status"] == "unavailable":
            notes.append("unavailable / not comparable")
        if num_cases is not None and num_cases < 60:
            notes.append("high_variance")
        if row["deeprare_dataset_name"] == "MIMIC-IV-Rare":
            notes.append("Top5 已达标；Top1/Top3 仍低于 target")
        if row["deeprare_dataset_name"] == "RareBench-LIRICAL":
            notes.append("strict target 未全达，但 Top3/Top5 只差约 1 case")
        if row["deeprare_dataset_name"] == "MyGene":
            notes.append("从 MyGene2 近似映射")
        row["notes"] = "; ".join(notes)
        rows.append(row)

    return pd.DataFrame(rows)


def bucket_rank(rank: float) -> str:
    if rank == 1:
        return "rank=1"
    if 2 <= rank <= 3:
        return "rank 2-3"
    if 4 <= rank <= 5:
        return "rank 4-5"
    if 6 <= rank <= 20:
        return "rank 6-20"
    if 21 <= rank <= 50:
        return "rank 21-50"
    return "rank>50"


def diagnose_gap(status: str, top1_gap: float | None, top3_gap: float | None, top5_gap: float | None, rank50: float | None) -> str:
    if status == "unavailable":
        return "当前项目没有可比较数据集。"
    if status == "all_reached":
        return "DeepRare Recall@1/@3/@5 均已达标；保持 current mainline，并关注小样本方差。"
    if top5_gap is not None and top5_gap <= 0 and top1_gap is not None and top1_gap > 0:
        return "Top5 已达标但 Top1/Top3 落后，主要瓶颈是已召回 top5/top10 候选内部排序。"
    if top5_gap is not None and top5_gap > 0:
        if rank50 is not None and rank50 < 0.70:
            return "Top5 和 Rank<=50 都偏低，单独 reranking 不足，需要 candidate expansion 或上游召回改进。"
        return "Top5 低于 target，但 Rank<=50 仍有可恢复空间；需要 candidate recall audit 与 hard-negative reranking 结合。"
    if top3_gap is not None and top3_gap > 0:
        return "Top5 接近或已达标但 Top3 仍不足；优先做 top5 局部排序。"
    return "剩余缺口很小；训练前先做 mapping/outlier audit。"


def build_rank_bucket_gap(case_ranks: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    comparable = gap[gap["target_status"] != "unavailable"].copy()
    rows: list[dict[str, Any]] = []

    gap_by_project = {row["project_dataset"]: row for _, row in comparable.iterrows()}
    for project_dataset, group in case_ranks.groupby("dataset"):
        if project_dataset not in gap_by_project:
            continue
        ranks = pd.to_numeric(group["final_rank"], errors="coerce").dropna()
        counts = Counter(bucket_rank(float(rank)) for rank in ranks)
        total = int(len(ranks))
        gap_row = gap_by_project[project_dataset]
        row = {
            "deeprare_dataset_name": gap_row["deeprare_dataset_name"],
            "project_dataset": project_dataset,
            "num_cases": total,
        }
        for bucket in ["rank=1", "rank 2-3", "rank 4-5", "rank 6-20", "rank 21-50", "rank>50"]:
            count = counts.get(bucket, 0)
            row[f"{bucket}_count"] = count
            row[f"{bucket}_rate"] = count / total if total else 0.0
        row["current_rank_le_50"] = gap_row["current_rank_le_50"]
        row["target_status"] = gap_row["target_status"]
        row["gap_diagnosis"] = diagnose_gap(
            gap_row["target_status"],
            gap_row["top1_gap"],
            gap_row["top3_gap"],
            gap_row["top5_gap"],
            gap_row["current_rank_le_50"],
        )
        if total < 60:
            row["notes"] = "high_variance"
        else:
            row["notes"] = ""
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["deeprare_dataset_name"]).reset_index(drop=True)


def metric_line(row: pd.Series) -> str:
    if row["target_status"] == "unavailable":
        return "unavailable / not comparable"
    return (
        f"Top1 {fmt_float(row['current_top1'])}/{fmt_float(row['target_top1'])} "
        f"(case gap {int(row['top1_case_gap'])}), "
        f"Top3 {fmt_float(row['current_top3'])}/{fmt_float(row['target_top3'])} "
        f"(case gap {int(row['top3_case_gap'])}), "
        f"Top5 {fmt_float(row['current_top5'])}/{fmt_float(row['target_top5'])} "
        f"(case gap {int(row['top5_case_gap'])}), "
        f"Rank<=50 {fmt_float(row['current_rank_le_50'])}"
    )


def write_strategy(gap: pd.DataFrame, bucket: pd.DataFrame) -> None:
    bucket_by_project = {row["project_dataset"]: row for _, row in bucket.iterrows()}
    gap_by_name = {row["deeprare_dataset_name"]: row for _, row in gap.iterrows()}

    lines = [
        "# 各数据集提升策略",
        "",
        "所有建议均以 strict exact Recall@1/@3/@5 作为主结果。any-label、relaxed MONDO、ancestor/sibling/synonym/replacement 分析只能作为 supplementary。",
        "",
    ]

    for name in [
        "MIMIC-IV-Rare",
        "DDD",
        "RareBench-HMS",
        "RareBench-LIRICAL",
        "RareBench-RAMEDIS",
        "MyGene",
        "RareBench-MME",
        "Xinhua Hosp.",
    ]:
        row = gap_by_name[name]
        project = row["project_dataset"]
        bucket_row = bucket_by_project.get(project)
        lines.extend([f"## {name}", "", f"- Current vs target: {metric_line(row)}"])
        if bucket_row is not None:
            lines.append(
                "- Rank bucket: "
                f"rank=1 {int(bucket_row['rank=1_count'])}, "
                f"2-3 {int(bucket_row['rank 2-3_count'])}, "
                f"4-5 {int(bucket_row['rank 4-5_count'])}, "
                f"6-20 {int(bucket_row['rank 6-20_count'])}, "
                f"21-50 {int(bucket_row['rank 21-50_count'])}, "
                f">50 {int(bucket_row['rank>50_count'])}."
            )

        if name == "MIMIC-IV-Rare":
            lines.extend(
                [
                    "- 策略：Top5 已达到 DeepRare target，重点提升 Top1/Top3。优先做 light-train reranker、pairwise/listwise reranker、HGNN top1 protection、gated SimilarCase。",
                    "- 不优先继续无门控扩大 SimilarCase topk；图对比学习继续后置。",
                ]
            )
        elif name == "DDD":
            lines.extend(
                [
                    "- 策略：DDD 三个 target 均未达到，但 Rank<=50 明显高于 Top5，说明很多 gold 已进入 top50 但排序不够靠前。",
                    "- 优先 ontology-aware hard negative training 与 pairwise/listwise reranking；负样本包括 same-parent、sibling、高 HPO-overlap、top50-above-gold。",
                ]
            )
        elif name == "RareBench-HMS":
            lines.extend(
                [
                    "- 策略：项目测试集只有 25 例，标注 high_variance，不建议作为唯一主结论。",
                    "- 如果要提升，优先查 label/mapping 和 top50 miss；如增加 relaxed 分析，必须与 exact 主结果分开报告。",
                ]
            )
        elif name == "RareBench-LIRICAL":
            lines.extend(
                [
                    "- 策略：Top3/Top5 只差约 1 case，属于 near-miss/high_variance，不应按大缺口处理。",
                    "- 优先 outlier audit、mapping audit、局部 rerank；不要盲目训练。",
                ]
            )
        elif name in {"RareBench-RAMEDIS", "MyGene"}:
            lines.extend(
                [
                    "- 策略：current mainline 已达到或超过 DeepRare targets，保持主线。",
                    "- 若后续追求 Top1 polish，可做 top5-to-top1 reranker；若未来 Top5 下降，先做 candidate recall audit。",
                ]
            )
        elif name == "RareBench-MME":
            lines.extend(
                [
                    "- 策略：当前 MME 已达到 DeepRare targets，但只有 10 例，必须标注 high_variance。",
                    "- 不要根据该数据集单独决定整体训练策略。",
                ]
            )
        elif name == "Xinhua Hosp.":
            lines.extend(
                [
                    "- 策略：当前项目中 unavailable / not comparable。",
                    "- 除非新增并记录明确对应的数据集，否则不要把其他项目数据集硬映射成 Xinhua Hosp.。",
                ]
            )
        lines.append("")

    (REPORT_DIR / "dataset_specific_strategy.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_experiment_plan() -> None:
    lines = [
        "# DeepRare Target 统一实验路线",
        "",
        "## P0：锁定指标口径",
        "- 确认所有 current mainline 数字均来自 validation-selected fixed test 或无 test-side tuning 的 fixed mainline test。",
        "- 表格中统一把 Recall@1/@3/@5 与 Top1/Top3/Top5 作为 strict exact 同一指标族。",
        "- 不混写 baseline exact、frozen config experiments、current mainline final results。",
        "",
        "## P1：target-aware no-train rerank",
        "- 对 Top5 未达标的数据集，先做 candidate recall audit 和 candidate expansion。",
        "- 对 Top5 已达标但 Top1/Top3 未达标的数据集，做 top5/top10 局部重排。",
        "- 所有权重只在 validation 上选择；test 只做 fixed evaluation。",
        "",
        "## P2：light-train reranker",
        "- 构造统一候选表。",
        "- 特征包括 HGNN score、SimilarCase score、HPO overlap、IC overlap、MONDO relation、source count、rank/margin features，以及经过 validation 证明有效的数据集指示特征。",
        "- 首选 linear / GBDT / pairwise reranker；目标函数优先优化 Recall@1 和 Recall@3，同时约束 Recall@5 不下降。",
        "",
        "## P3：ontology-aware hard negative training",
        "- 重点用于 DDD 以及其他 Rank<=50 高但 Top1/Top3 低的数据集。",
        "- 负样本包括 same-parent、sibling、高 HPO-overlap、top50-above-gold。",
        "- validation selection 与 fixed test reporting 必须分开。",
        "",
        "## P4：图对比学习",
        "- 仅在 label 清洗稳定、candidate recall 足够、正负对可靠后再做。",
        "- 不作为当前缺口的第一优先级。",
    ]
    (REPORT_DIR / "deeprare_target_experiment_plan.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_recommended_next_task(gap: pd.DataFrame) -> None:
    all_reached = gap[gap["target_status"] == "all_reached"]["deeprare_dataset_name"].tolist()
    top5_only = gap[gap["target_status"] == "top5_reached_only"]["deeprare_dataset_name"].tolist()
    far_rows = gap[gap["target_status"] == "far_from_target"].copy()
    near_miss = far_rows[
        (far_rows["top5_case_gap"].fillna(9999) <= 1)
        & (far_rows["top3_case_gap"].fillna(9999) <= 1)
    ]["deeprare_dataset_name"].tolist()
    far = far_rows[~far_rows["deeprare_dataset_name"].isin(near_miss)]["deeprare_dataset_name"].tolist()
    unavailable = gap[gap["target_status"] == "unavailable"]["deeprare_dataset_name"].tolist()

    lines = [
        "# 下一步 Codex 执行建议",
        "",
        "## 直接回答",
        f"1. 已经达到全部 DeepRare 标准的数据集：{', '.join(all_reached) if all_reached else 'none'}。",
        f"2. 只达到 Top5、但 Top1/Top3 未达标的数据集：{', '.join(top5_only) if top5_only else 'none'}。",
        f"3. 距离较远的数据集：{', '.join(far) if far else 'none'}。near-miss/high_variance：{', '.join(near_miss) if near_miss else 'none'}。unavailable：{', '.join(unavailable) if unavailable else 'none'}。",
        "4. 每个数据集还差多少 case 见下表。",
        "5. 第一优先级：MIMIC-IV-Rare 做 validation-selected light reranker 的 top5/top10 局部排序；DDD 同时做 reranking 与 candidate recall audit。Top5 未达标的数据集才继续 candidate expansion。",
        "6. 需要 hard negative training，尤其是 DDD，因为 Rank<=50 明显高于 Top5，说明很多 gold 可恢复但排序不够靠前。",
        "7. 图对比学习继续后置，等 label/mapping 与 candidate recall audit 稳定后再做。",
        "8. 最适合写入论文主表的结果：outputs/mainline_full_pipeline/mainline_final_metrics_with_sources.csv 中的 current mainline final strict exact fixed-test metrics。",
        "9. 只能做 supplementary：mimic any-label 指标、relaxed MONDO/ancestor/sibling/synonym/replacement 匹配、小样本 high_variance 分析，以及 unavailable/not-comparable mapping。",
        "",
        "## Case Gaps",
    ]

    display_cols = [
        "deeprare_dataset_name",
        "project_dataset",
        "num_cases",
        "top1_case_gap",
        "top3_case_gap",
        "top5_case_gap",
        "target_status",
        "priority_level",
        "notes",
    ]
    lines.append(markdown_table(gap[display_cols]))
    lines.extend(
        [
            "",
            "## 推荐下一步任务",
            "实现一个统一的 validation-selected light reranker 实验：",
            "- Input：stage4 当前 top50 candidate tables 加 current final ranks。",
            "- Scope：MIMIC-IV-Rare top5/top10 局部排序，以及 DDD top50 reranking。",
            "- Features：HGNN score/rank/margin、SimilarCase score、exact HPO overlap、IC overlap、semantic HPO coverage、MONDO relation features、source-count features。",
            "- Selection：validation-only objective，优先 Recall@1 和 Recall@3，并拒绝导致 Recall@5 下降的配置。",
            "- Output：fixed-test strict exact Recall@1/@3/@5；supplementary relaxed analyses 单独成表。",
        ]
    )
    (REPORT_DIR / "recommended_next_codex_task.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = read_json(RUN_MANIFEST)
    targets = pd.DataFrame(DEEPRARE_TARGETS)
    metrics = pd.read_csv(FINAL_METRICS_WITH_SOURCES)
    case_ranks = pd.read_csv(FINAL_CASE_RANKS)

    current = normalize_mainline_metrics(metrics, manifest)
    gap = build_gap_analysis(targets, current)
    bucket = build_rank_bucket_gap(case_ranks, gap)

    write_csv_and_md(
        targets,
        REPORT_DIR / "deeprare_target_table.csv",
        REPORT_DIR / "deeprare_target_table.md",
        "DeepRare Target 对照表",
        [
            "mapping_status 采用保守口径。unavailable 数据集不强行映射到无关项目数据集。",
        ],
    )
    write_csv_and_md(
        current,
        REPORT_DIR / "current_project_mainline_metrics.csv",
        REPORT_DIR / "current_project_mainline_metrics.md",
        "当前项目 Mainline Final 指标",
        [
            "指标来自 current mainline final outputs，口径为 strict exact Recall@1/@3/@5。",
            "本表不纳入 supplementary any-label 或 relaxed matching 指标。",
        ],
    )
    write_csv_and_md(
        gap,
        REPORT_DIR / "deeprare_gap_analysis.csv",
        REPORT_DIR / "deeprare_gap_analysis.md",
        "DeepRare Target Gap 分析",
        [
            "case_gap = ceil(max(target - current, 0) * num_cases).",
            "priority_level 反映当前可恢复性和实验优先级，不等同于论文重要性。",
        ],
    )
    write_csv_and_md(
        bucket,
        REPORT_DIR / "dataset_rank_bucket_gap.csv",
        REPORT_DIR / "dataset_rank_bucket_gap.md",
        "Dataset Rank Bucket Gap",
        [
            "rank bucket 基于 mainline_final_case_ranks.csv 的 final_rank 计算。",
        ],
    )

    write_strategy(gap, bucket)
    write_experiment_plan()
    write_recommended_next_task(gap)

    for filename in [
        "deeprare_target_table.csv",
        "current_project_mainline_metrics.csv",
        "deeprare_gap_analysis.csv",
        "dataset_rank_bucket_gap.csv",
    ]:
        (OUTPUT_DIR / filename).write_bytes((REPORT_DIR / filename).read_bytes())

    output_manifest = {
        "source_metrics": str(FINAL_METRICS_WITH_SOURCES),
        "source_case_ranks": str(FINAL_CASE_RANKS),
        "source_run_manifest": str(RUN_MANIFEST),
        "mainline_config": str(MAINLINE_CONFIG),
        "report_dir": str(REPORT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "strict_exact_main_result": True,
        "no_test_side_tuning": True,
        "generated_files": sorted(path.name for path in REPORT_DIR.iterdir() if path.is_file()),
    }
    (OUTPUT_DIR / "generation_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote reports to {REPORT_DIR}")
    print(f"Wrote output copies to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
