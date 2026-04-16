from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = [
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\LIRICAL.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\mimic_rag_0425.csv"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\mimic_test.csv"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\MME.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\MyGene2.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\RAMEDIS.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\HMS.xlsx"),
    Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\ddd_test.csv"),
]
DEFAULT_OUTPUT = Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\total.xlsx")
MONDO_COLUMN = "mondo_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate processed dataset rows into total.xlsx mondo frequency table."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Processed CSV/XLSX files that contain a mondo_label column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output XLSX path for the mondo frequency table.",
    )
    return parser.parse_args()


def read_processed_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if MONDO_COLUMN not in df.columns:
        raise ValueError(f"{path.name} is missing required column: {MONDO_COLUMN}")
    return df


def build_frequency_table(input_paths: list[Path]) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    stats: list[dict[str, int | str]] = []
    mondo_series_list: list[pd.Series] = []

    for path in input_paths:
        df = read_processed_table(path)
        mondo_values = df[MONDO_COLUMN].fillna("").astype(str).str.strip()
        mondo_values = mondo_values[mondo_values.ne("")]
        mondo_series_list.append(mondo_values)
        stats.append(
            {
                "file": path.name,
                "rows": int(len(df)),
                "usable_mondo_rows": int(len(mondo_values)),
                "unique_mondo_ids": int(mondo_values.nunique()),
            }
        )

    combined = pd.concat(mondo_series_list, ignore_index=True)
    frequency_df = (
        combined.value_counts(sort=True)
        .rename_axis("mondo_id")
        .reset_index(name="frequency")
        .sort_values(["frequency", "mondo_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    return frequency_df, stats


def main() -> None:
    args = parse_args()
    frequency_df, stats = build_frequency_table(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frequency_df.to_excel(args.output, index=False)

    for item in stats:
        print(
            f"{item['file']}: rows={item['rows']}, "
            f"usable_mondo_rows={item['usable_mondo_rows']}, "
            f"unique_mondo_ids={item['unique_mondo_ids']}"
        )
    print(
        f"Wrote {len(frequency_df)} unique mondo_ids and "
        f"{int(frequency_df['frequency'].sum())} total rows to {args.output}"
    )


if __name__ == "__main__":
    main()
