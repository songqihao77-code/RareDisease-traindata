from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANK_DECOMP = PROJECT_ROOT / "reports" / "diagnosis" / "dataset_rank_decomposition.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "deeprare_parity"

DEEPRARE_TARGETS: dict[str, dict[str, float | int | str]] = {
    "DDD": {"paper_cases": 2283, "top1": 0.48, "top3": 0.60, "top5": 0.63},
    "mimic_test": {"paper_name": "MIMIC-IV-Rare", "paper_cases": 1873, "top1": 0.29, "top3": 0.37, "top5": 0.39},
    "LIRICAL": {"paper_cases": 370, "top1": 0.56, "top3": 0.65, "top5": 0.68},
    "RAMEDIS": {"paper_cases": 624, "top1": 0.71, "top3": 0.83, "top5": 0.85},
    "HMS": {"paper_cases": 88, "top1": 0.57, "top3": 0.65, "top5": 0.71},
    "MME": {"paper_cases": 40, "top1": 0.78, "top3": 0.85, "top5": 0.90},
    "MyGene2": {"paper_cases": 146, "top1": 0.76, "top3": 0.80, "top5": 0.81},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exact HGNN metrics with DeepRare HPO-wise Recall targets.")
    parser.add_argument("--rank-decomposition-path", type=Path, default=DEFAULT_RANK_DECOMP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rank_decomposition(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Rank decomposition CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "num_cases", "top1", "top3", "top5", "rank_le_50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def build_gap_table(rank_df: pd.DataFrame) -> pd.DataFrame:
    current_by_dataset = rank_df.set_index("dataset").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for dataset, target in DEEPRARE_TARGETS.items():
        current = current_by_dataset.get(dataset)
        if current is None:
            rows.append({"dataset": dataset, "status": "missing_current_result"})
            continue
        paper_cases = int(target["paper_cases"])
        current_cases = int(current["num_cases"])
        rank_le_50 = float(current["rank_le_50"])
        rows.append(
            {
                "dataset": dataset,
                "paper_dataset_name": str(target.get("paper_name", dataset)),
                "current_cases": current_cases,
                "deeprare_paper_cases": paper_cases,
                "split_mismatch": bool(current_cases != paper_cases),
                "current_top1": float(current["top1"]),
                "current_top3": float(current["top3"]),
                "current_top5": float(current["top5"]),
                "deeprare_top1": float(target["top1"]),
                "deeprare_top3": float(target["top3"]),
                "deeprare_top5": float(target["top5"]),
                "gap_top1": float(current["top1"]) - float(target["top1"]),
                "gap_top3": float(current["top3"]) - float(target["top3"]),
                "gap_top5": float(current["top5"]) - float(target["top5"]),
                "rank_le_50_upper_bound": rank_le_50,
                "top50_rerank_can_reach_top1_target": bool(rank_le_50 >= float(target["top1"])),
                "top50_rerank_can_reach_top3_target": bool(rank_le_50 >= float(target["top3"])),
                "top50_rerank_can_reach_top5_target": bool(rank_le_50 >= float(target["top5"])),
                "reaches_deeprare_top1": bool(float(current["top1"]) >= float(target["top1"])),
                "reaches_deeprare_top3": bool(float(current["top3"]) >= float(target["top3"])),
                "reaches_deeprare_top5": bool(float(current["top5"]) >= float(target["top5"])),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_report(table: pd.DataFrame) -> str:
    display_cols = [
        "dataset",
        "current_cases",
        "deeprare_paper_cases",
        "split_mismatch",
        "current_top1",
        "current_top3",
        "current_top5",
        "deeprare_top1",
        "deeprare_top3",
        "deeprare_top5",
        "gap_top1",
        "gap_top3",
        "gap_top5",
        "rank_le_50_upper_bound",
        "top50_rerank_can_reach_top5_target",
    ]
    lines = [
        "# DeepRare Target Gap",
        "",
        "> 本表使用当前 exact HGNN baseline / rank decomposition 与 DeepRare 论文目标对齐。`rank<=50` 只是 top50 rerank 的理论上限，不代表正式可报告结果。",
        "",
        markdown_table(table, display_cols),
        "",
        "## Split Mismatch",
        "",
    ]
    split_mismatch = table.loc[table["split_mismatch"], ["dataset", "current_cases", "deeprare_paper_cases"]]
    if split_mismatch.empty:
        lines.append("- 未发现 split case 数不一致。")
    else:
        for row in split_mismatch.itertuples(index=False):
            lines.append(f"- `{row.dataset}`: 当前 {row.current_cases} vs 论文 {row.deeprare_paper_cases}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_df = load_rank_decomposition(args.rank_decomposition_path)
    table = build_gap_table(rank_df)
    csv_path = args.output_dir / "deeprare_target_gap.csv"
    md_path = args.output_dir / "deeprare_target_gap.md"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(markdown_report(table), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path.resolve()), "md": str(md_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
