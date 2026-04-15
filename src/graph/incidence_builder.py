from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: set[str], file_name: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{file_name} 缺少必要列: {missing_text}")


def validate_unique(df: pd.DataFrame, key_column: str, file_name: str) -> None:
    duplicated = df[df.duplicated(subset=[key_column], keep=False)]
    if not duplicated.empty:
        sample = duplicated[key_column].drop_duplicates().head(10).tolist()
        raise ValueError(f"{file_name} 的 {key_column} 存在重复值，示例: {sample}")


def build_sparse_triplets(
    data_file: Path,
    disease_index_file: Path,
    hpo_index_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    data_df = pd.read_excel(data_file)
    disease_index_df = pd.read_excel(disease_index_file)
    hpo_index_df = pd.read_excel(hpo_index_file)

    validate_columns(data_df, {"case_id", "mondo_id", "hpo_id", "weight"}, data_file.name)
    validate_columns(disease_index_df, {"mondo_id", "disease_idx"}, disease_index_file.name)
    validate_columns(hpo_index_df, {"hpo_id", "hpo_idx"}, hpo_index_file.name)

    validate_unique(disease_index_df, "mondo_id", disease_index_file.name)
    validate_unique(hpo_index_df, "hpo_id", hpo_index_file.name)

    merged_df = (
        data_df
        .merge(hpo_index_df[["hpo_id", "hpo_idx"]], on="hpo_id", how="left")
        .merge(disease_index_df[["mondo_id", "disease_idx"]], on="mondo_id", how="left")
    )

    missing_hpo = merged_df.loc[merged_df["hpo_idx"].isna(), "hpo_id"].drop_duplicates().tolist()
    if missing_hpo:
        raise ValueError(f"以下 hpo_id 未在索引中找到: {missing_hpo[:10]}")

    missing_disease = (
        merged_df.loc[merged_df["disease_idx"].isna(), "mondo_id"].drop_duplicates().tolist()
    )
    if missing_disease:
        raise ValueError(f"以下 mondo_id 未在索引中找到: {missing_disease[:10]}")

    rows = merged_df["hpo_idx"].astype(np.int64).to_numpy()
    cols = merged_df["disease_idx"].astype(np.int64).to_numpy()
    vals = merged_df["weight"].astype(np.float64).to_numpy()
    shape = np.array(
        [
            int(hpo_index_df["hpo_idx"].max()) + 1,
            int(disease_index_df["disease_idx"].max()) + 1,
        ],
        dtype=np.int64,
    )

    triplets_df = merged_df.copy()
    triplets_df["rows"] = rows
    triplets_df["cols"] = cols
    triplets_df["vals"] = vals

    ordered_columns = [
        "case_id",
        "mondo_id",
        "hpo_id",
        "raw_weight",
        "weight",
        "rows",
        "cols",
        "vals",
    ]
    existing_columns = [column for column in ordered_columns if column in triplets_df.columns]
    remaining_columns = [column for column in triplets_df.columns if column not in existing_columns]
    triplets_df = triplets_df[existing_columns + remaining_columns]

    return rows, cols, vals, shape, triplets_df, merged_df


def write_outputs(
    output_npz: Path,
    output_xlsx: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    shape: np.ndarray,
    triplets_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_npz, rows=rows, cols=cols, vals=vals, shape=shape)

    meta_df = pd.DataFrame(
        [
            {"metric": "triplet_count", "value": int(len(rows))},
            {"metric": "unique_hpo_count", "value": int(merged_df["hpo_id"].nunique())},
            {"metric": "unique_disease_count", "value": int(merged_df["mondo_id"].nunique())},
            {"metric": "shape_rows", "value": int(shape[0])},
            {"metric": "shape_cols", "value": int(shape[1])},
        ]
    )

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        triplets_df.to_excel(writer, sheet_name="triplets", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)


def parse_args() -> argparse.Namespace:
    default_data_file = Path(r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\DiseaseHyperedge_data_v4.xlsx")
    default_disease_index_file = Path(
        r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx"
    )
    default_hpo_index_file = Path(r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\HPO_index_v4.xlsx")
    default_output_dir = Path(r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed")

    parser = argparse.ArgumentParser(
        description="将 DiseaseHyperedge_data_v4.xlsx 转成 sparse triplets npz/xlsx。"
    )
    parser.add_argument("--data-file", type=Path, default=default_data_file)
    parser.add_argument("--disease-index-file", type=Path, default=default_disease_index_file)
    parser.add_argument("--hpo-index-file", type=Path, default=default_hpo_index_file)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--output-npz-name",
        default="DiseaseHyperedge_sparse_triplets_v4.npz",
    )
    parser.add_argument(
        "--output-xlsx-name",
        default="DiseaseHyperedge_sparse_triplets_v4.xlsx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, cols, vals, shape, triplets_df, merged_df = build_sparse_triplets(
        data_file=args.data_file,
        disease_index_file=args.disease_index_file,
        hpo_index_file=args.hpo_index_file,
    )
    output_npz = args.output_dir / args.output_npz_name
    output_xlsx = args.output_dir / args.output_xlsx_name
    write_outputs(
        output_npz=output_npz,
        output_xlsx=output_xlsx,
        rows=rows,
        cols=cols,
        vals=vals,
        shape=shape,
        triplets_df=triplets_df,
        merged_df=merged_df,
    )

    print(f"已生成 {output_npz}")
    print(f"已生成 {output_xlsx}")
    print(f"triplet_count={len(rows)}")
    print(f"shape=({int(shape[0])}, {int(shape[1])})")


if __name__ == "__main__":
    main()
