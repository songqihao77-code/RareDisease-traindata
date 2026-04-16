from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed"
DEFAULT_REFERENCE_FILE = PROJECT_ROOT / "data" / "processed" / "knowledge" / "DiseaseHyperedge_data_v2.xlsx"
DEFAULT_DISEASE_INDEX_FILE = DEFAULT_INPUT_DIR / "Disease_index_v3.xlsx"
DEFAULT_HPO_INDEX_FILE = DEFAULT_INPUT_DIR / "HPO_index_v3.xlsx"
DEFAULT_OUTPUT_FILE = DEFAULT_INPUT_DIR / "DiseaseHyperedge_data_v3.xlsx"

REQUIRED_COLUMNS = ["case_id", "mondo_id", "hpo_id", "raw_weight"]
OUTPUT_COLUMNS = ["case_id", "mondo_id", "hpo_id", "raw_weight", "weight"]

SOURCE_FILES = OrderedDict(
    [
        ("orphanet", "orphanet_mondo_hpo_raw_weight.csv"),
        ("GARD", "GARD.csv"),
        ("HPOA", "HPOA.csv"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 orphanet -> GARD -> HPOA 的优先顺序重建疾病超边，"
            "并输出 DiseaseHyperedge_data_v3.xlsx。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"包含 orphanet/GARD/HPOA CSV 的目录。默认: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"输出 xlsx 文件路径。默认: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=DEFAULT_REFERENCE_FILE,
        help=f"用于校验输出字段格式的参考文件。默认: {DEFAULT_REFERENCE_FILE}",
    )
    parser.add_argument(
        "--disease-index-file",
        type=Path,
        default=DEFAULT_DISEASE_INDEX_FILE,
        help=f"用于校验 mondo_id 的索引文件。默认: {DEFAULT_DISEASE_INDEX_FILE}",
    )
    parser.add_argument(
        "--hpo-index-file",
        type=Path,
        default=DEFAULT_HPO_INDEX_FILE,
        help=f"用于校验 hpo_id 的索引文件。默认: {DEFAULT_HPO_INDEX_FILE}",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{file_path} 缺少必要列: {missing}")


def load_source_rows(file_path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    df = pd.read_csv(file_path)
    validate_columns(df, file_path)

    work = df[REQUIRED_COLUMNS].copy()
    work["mondo_id"] = work["mondo_id"].astype(str).str.strip()
    work["hpo_id"] = work["hpo_id"].astype(str).str.strip()
    work["raw_weight"] = pd.to_numeric(work["raw_weight"], errors="coerce")
    work = work.dropna(subset=["mondo_id", "hpo_id", "raw_weight"])
    work = work[
        work["mondo_id"].str.startswith("MONDO:")
        & work["hpo_id"].str.startswith("HP:")
    ].copy()

    ordered_mondo_ids = work["mondo_id"].drop_duplicates().tolist()
    mondo_to_hpo_weight: dict[str, dict[str, float]] = {}
    grouped = work.groupby(["mondo_id", "hpo_id"], sort=False, as_index=False)["raw_weight"].max()

    for row in grouped.itertuples(index=False):
        mondo_to_hpo_weight.setdefault(row.mondo_id, {})[row.hpo_id] = float(row.raw_weight)

    ordered_mondo_ids = [mondo_id for mondo_id in ordered_mondo_ids if mondo_id in mondo_to_hpo_weight]
    return ordered_mondo_ids, mondo_to_hpo_weight


def build_merged_rows(
    source_payloads: OrderedDict[str, tuple[list[str], dict[str, dict[str, float]]]]
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    ordered_mondo_ids: list[str] = []
    selected_source_by_mondo: dict[str, str] = {}
    selected_hpo_by_mondo: dict[str, dict[str, float]] = {}
    added_count_by_source: dict[str, int] = {source_name: 0 for source_name in source_payloads}

    for source_name, (source_mondo_ids, source_hpo_map) in source_payloads.items():
        for mondo_id in source_mondo_ids:
            if mondo_id in selected_source_by_mondo:
                continue
            selected_source_by_mondo[mondo_id] = source_name
            selected_hpo_by_mondo[mondo_id] = source_hpo_map[mondo_id]
            ordered_mondo_ids.append(mondo_id)
            added_count_by_source[source_name] += 1

    rows: list[dict[str, object]] = []
    for case_number, mondo_id in enumerate(ordered_mondo_ids, start=1):
        case_id = f"case_{case_number}"
        hpo_weights = selected_hpo_by_mondo[mondo_id]
        total_raw_weight = sum(hpo_weights.values())
        if total_raw_weight <= 0:
            raise ValueError(f"{mondo_id} 的 raw_weight 总和无效: {total_raw_weight}")

        for hpo_id in sorted(hpo_weights):
            raw_weight = float(hpo_weights[hpo_id])
            weight = raw_weight / total_raw_weight
            rows.append(
                {
                    "case_id": case_id,
                    "mondo_id": mondo_id,
                    "hpo_id": hpo_id,
                    "raw_weight": raw_weight,
                    "weight": weight,
                }
            )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), added_count_by_source, selected_source_by_mondo


def validate_against_reference(df: pd.DataFrame, reference_file: Path) -> None:
    reference_df = pd.read_excel(reference_file)
    reference_columns = list(reference_df.columns)
    if reference_columns != OUTPUT_COLUMNS:
        raise ValueError(
            f"参考文件 {reference_file} 的列顺序为 {reference_columns}，"
            f"与预期 {OUTPUT_COLUMNS} 不一致。"
        )
    if list(df.columns) != reference_columns:
        raise ValueError(f"输出列顺序异常: {list(df.columns)}")


def validate_weight_sum(df: pd.DataFrame) -> None:
    case_weight_sum = df.groupby("case_id", sort=False)["weight"].sum()
    invalid_case_ids = case_weight_sum[~case_weight_sum.round(10).eq(1.0)]
    if not invalid_case_ids.empty:
        sample = invalid_case_ids.head(10).to_dict()
        raise ValueError(f"以下 case_id 的权重和不为 1: {sample}")


def validate_index_coverage(df: pd.DataFrame, disease_index_file: Path, hpo_index_file: Path) -> tuple[int, int]:
    disease_index_df = pd.read_excel(disease_index_file)
    hpo_index_df = pd.read_excel(hpo_index_file)

    disease_vocab = set(disease_index_df["mondo_id"].astype(str))
    hpo_vocab = set(hpo_index_df["hpo_id"].astype(str))

    missing_mondo = sorted(set(df["mondo_id"]) - disease_vocab)
    missing_hpo = sorted(set(df["hpo_id"]) - hpo_vocab)

    if missing_mondo:
        raise ValueError(f"存在未收录到 Disease_index_v3 的 mondo_id，示例: {missing_mondo[:10]}")
    if missing_hpo:
        raise ValueError(f"存在未收录到 HPO_index_v3 的 hpo_id，示例: {missing_hpo[:10]}")

    return len(disease_vocab), len(hpo_vocab)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_file = args.output_file.resolve()
    reference_file = args.reference_file.resolve()
    disease_index_file = args.disease_index_file.resolve()
    hpo_index_file = args.hpo_index_file.resolve()

    source_payloads: OrderedDict[str, tuple[list[str], dict[str, dict[str, float]]]] = OrderedDict()
    for source_name, file_name in SOURCE_FILES.items():
        file_path = input_dir / file_name
        source_payloads[source_name] = load_source_rows(file_path)

    merged_df, added_count_by_source, selected_source_by_mondo = build_merged_rows(source_payloads)

    validate_against_reference(merged_df, reference_file)
    validate_weight_sum(merged_df)
    disease_vocab_count, hpo_vocab_count = validate_index_coverage(
        merged_df,
        disease_index_file=disease_index_file,
        hpo_index_file=hpo_index_file,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_excel(output_file, index=False)

    source_usage = pd.Series(selected_source_by_mondo).value_counts().reindex(SOURCE_FILES.keys(), fill_value=0)
    print(f"输出文件: {output_file}")
    print(f"输出行数: {len(merged_df)}")
    print(f"输出疾病数: {merged_df['mondo_id'].nunique()}")
    print(f"输出 HPO 数: {merged_df['hpo_id'].nunique()}")
    for source_name in SOURCE_FILES:
        print(f"{source_name} 新增疾病数: {added_count_by_source[source_name]}")
    print(f"source_usage={source_usage.to_dict()}")
    print(f"Disease_index_v3 覆盖数: {disease_vocab_count}")
    print(f"HPO_index_v3 覆盖数: {hpo_vocab_count}")


if __name__ == "__main__":
    main()
