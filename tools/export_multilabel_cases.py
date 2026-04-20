from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


INPUT_PATHS = [
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\ddd_test.csv"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\MyGene2.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\RAMEDIS.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\HMS.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\LIRICAL.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\mimic_rag_0425.csv"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\MME.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\test\mimic_test.csv"),
]

OUTPUT_DIR = Path(r"D:\RareDisease-traindata\LLLdataset\dataset\duobiaoqian")


@dataclass(frozen=True)
class ExportResult:
    dataset_name: str
    source_path: Path
    extracted_path: Path
    case_stats_path: Path
    total_rows: int
    total_cases: int
    multilabel_cases: int
    multilabel_rows: int
    max_label_count: int


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, dtype=str)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(path, index=False)


def build_case_stats(df: pd.DataFrame) -> pd.DataFrame:
    labels_by_case = (
        df.groupby("case_id", sort=False)["mondo_label"]
        .agg(lambda values: list(dict.fromkeys(values.astype(str).tolist())))
        .reset_index(name="mondo_labels")
    )
    rows_by_case = (
        df.groupby("case_id", sort=False)
        .size()
        .reset_index(name="row_count")
    )
    stats_df = labels_by_case.merge(rows_by_case, on="case_id", how="left")
    stats_df["label_count"] = stats_df["mondo_labels"].apply(len)
    stats_df["mondo_labels_json"] = stats_df["mondo_labels"].apply(
        lambda values: json.dumps(values, ensure_ascii=False)
    )
    stats_df = stats_df.drop(columns=["mondo_labels"])
    return stats_df.sort_values(
        by=["label_count", "row_count", "case_id"],
        ascending=[False, False, True],
    )


def export_one_dataset(path: Path, output_dir: Path) -> ExportResult:
    df = read_table(path)
    required_cols = {"case_id", "mondo_label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_text = ", ".join(sorted(missing_cols))
        raise ValueError(f"{path} 缺少必要列: {missing_text}")

    total_rows = int(len(df))
    total_cases = int(df["case_id"].nunique(dropna=True))

    case_stats = build_case_stats(df)
    multilabel_case_stats = case_stats.loc[case_stats["label_count"] > 1].copy()
    multilabel_case_ids = multilabel_case_stats["case_id"].astype(str).tolist()

    multilabel_df = df.loc[df["case_id"].astype(str).isin(multilabel_case_ids)].copy()

    stem = path.stem
    suffix = path.suffix.lower()
    extracted_path = output_dir / f"{stem}_multilabel_cases{suffix}"
    case_stats_path = output_dir / f"{stem}_multilabel_case_stats.csv"

    # 保留原始文件中的所有列与原始行内容，只过滤出多标签病例对应的行。
    write_table(multilabel_df, extracted_path)
    multilabel_case_stats.to_csv(case_stats_path, index=False, encoding="utf-8-sig")

    max_label_count = int(multilabel_case_stats["label_count"].max()) if not multilabel_case_stats.empty else 0
    return ExportResult(
        dataset_name=path.name,
        source_path=path,
        extracted_path=extracted_path,
        case_stats_path=case_stats_path,
        total_rows=total_rows,
        total_cases=total_cases,
        multilabel_cases=int(len(multilabel_case_stats)),
        multilabel_rows=int(len(multilabel_df)),
        max_label_count=max_label_count,
    )


def build_prompt(results: list[ExportResult], output_dir: Path) -> str:
    lines = [
        "你是一名医学数据审计与罕见病标注分析专家。",
        "",
        "任务目标：",
        "请针对以下“多标签同 case_id”的病例，判断哪个疾病最可能是真实主疾病或最终应保留的唯一疾病标签。",
        "这些病例来自已经处理好的长表数据，每个病例由 case_id 标识；同一个 case_id 下出现了多个 mondo_label，且共享一组 hpo_id。",
        "",
        "请你对每个多标签病例完成以下工作：",
        "1. 阅读该病例的全部候选 mondo_label 列表与 hpo_id 列表。",
        "2. 判断哪个疾病最可能是真实主疾病。",
        "3. 如果候选标签之间属于同义词、上下位概念、基础病与急性发作、慢性病与危象关系，请明确指出。",
        "4. 如果无法可靠判断唯一主疾病，请输出“无法确定”。",
        "5. 给出简短但明确的理由，重点说明：",
        "   - 哪些 HPO 更支持哪个疾病",
        "   - 哪些标签可能只是并发症、共病、上下位概念或非主病",
        "   - 是否存在明显的标签歧义或冲突",
        "",
        "输出格式要求：",
        "请以结构化表格输出，每行一个 case_id，包含以下字段：",
        "- dataset_name",
        "- case_id",
        "- candidate_mondo_labels",
        "- selected_mondo_label",
        "- decision",
        "- confidence",
        "- rationale",
        "",
        "其中：",
        "- selected_mondo_label：填写你认为最应保留的唯一标签；如果无法判断则填“无法确定”",
        "- decision：只能填写“保留该标签”或“无法确定”",
        "- confidence：给出 high / medium / low",
        "",
        "分析数据文件如下：",
    ]
    for result in results:
        lines.append(f"- 数据集原文件：{result.source_path}")
        lines.append(f"  多标签病例明细：{result.extracted_path}")
        lines.append(f"  多标签病例统计：{result.case_stats_path}")
    lines.extend(
        [
            "",
            "分析原则：",
            "- 优先寻找更能解释核心 phenotype 的疾病，而不是泛化标签。",
            "- 如果多个标签本质上是同义、上下位或同一疾病不同阶段，请优先选择更稳定、更基础的疾病标签。",
            "- 如果多个标签明显是不同共病，且无法从 HPO 中区分主病，请输出“无法确定”，不要强行选择。",
            "- 不要因为标签顺序在前就默认它更正确。",
        ]
    )
    prompt_text = "\n".join(lines)
    (output_dir / "gpt_pro_multilabel_analysis_prompt.txt").write_text(prompt_text, encoding="utf-8")
    return prompt_text


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [export_one_dataset(path, OUTPUT_DIR) for path in INPUT_PATHS]

    summary_df = pd.DataFrame(
        [
            {
                "dataset_name": result.dataset_name,
                "source_path": str(result.source_path),
                "extracted_path": str(result.extracted_path),
                "case_stats_path": str(result.case_stats_path),
                "total_rows": result.total_rows,
                "total_cases": result.total_cases,
                "multilabel_cases": result.multilabel_cases,
                "multilabel_rows": result.multilabel_rows,
                "max_label_count": result.max_label_count,
            }
            for result in results
        ]
    )
    summary_df.to_csv(OUTPUT_DIR / "multilabel_summary.csv", index=False, encoding="utf-8-sig")
    build_prompt(results, OUTPUT_DIR)

    print(summary_df.to_string(index=False))
    print(f"\n输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
