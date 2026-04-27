from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_top50_evidence_rerank import (
    evaluate_fixed_payload,
    load_candidates,
    run_presets,
    select_best_row,
    to_matrix,
    weight_payload_from_row,
)


VALIDATION_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_validation.csv"
TEST_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rerank"
REPORT_DIR = PROJECT_ROOT / "reports" / "ddd_improvement"

GRID_WEIGHTS_PATH = OUTPUT_DIR / "ddd_val_selected_grid_weights.json"
FIXED_METRICS_PATH = OUTPUT_DIR / "ddd_rerank_fixed_test_metrics.csv"
FIXED_BY_DATASET_PATH = OUTPUT_DIR / "ddd_rerank_fixed_test_by_dataset.csv"
REPORT_PATH = REPORT_DIR / "ddd_validation_selected_rerank_report.md"
EXISTING_VALIDATION_GRID = OUTPUT_DIR / "rerank_validation_grid_results.csv"


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("|" + "|".join(_fmt(value).replace("|", "/") for value in row) + "|")
    return "\n".join(lines)


def payload_for(row: pd.Series, *, objective: str) -> dict[str, Any]:
    payload = weight_payload_from_row(
        row,
        objective=objective,
        source_candidates_path=VALIDATION_CANDIDATES,
        protocol="validation_select",
    )
    payload["selection_kind"] = "grid"
    return payload


def metric_row(df: pd.DataFrame, preset: str) -> pd.Series:
    rows = df.loc[df["preset"] == preset]
    if rows.empty:
        raise KeyError(f"Missing preset: {preset}")
    return rows.iloc[0]


def wide_to_long(metrics: pd.DataFrame) -> pd.DataFrame:
    datasets = ["DDD", "mimic_test", "HMS", "LIRICAL", "ALL"]
    rows: list[dict[str, Any]] = []
    for item in metrics.itertuples(index=False):
        record = item._asdict()
        for dataset in datasets:
            num_key = f"{dataset}_num_cases"
            if num_key not in record or pd.isna(record[num_key]):
                continue
            rows.append(
                {
                    "protocol": record["preset"],
                    "kind": record["kind"],
                    "dataset": dataset,
                    "num_cases": int(record[num_key]),
                    "top1": float(record[f"{dataset}_top1"]),
                    "top3": float(record[f"{dataset}_top3"]),
                    "top5": float(record[f"{dataset}_top5"]),
                    "median_rank": float(record[f"{dataset}_median_rank"]),
                    "mean_rank": float(record[f"{dataset}_mean_rank"]),
                    "rank_le_50": float(record[f"{dataset}_rank_le_50"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_matrix = to_matrix(load_candidates(TEST_CANDIDATES))

    if not EXISTING_VALIDATION_GRID.is_file():
        raise FileNotFoundError(f"Existing validation grid is required: {EXISTING_VALIDATION_GRID}")
    validation_grid = pd.read_csv(EXISTING_VALIDATION_GRID)

    grid_selected: list[dict[str, Any]] = []
    fixed_rows: list[pd.DataFrame] = []
    for objective in ("DDD_top1", "ALL_top1"):
        selected = select_best_row(validation_grid, objective)
        payload = payload_for(selected, objective=objective)
        payload["output_preset"] = f"validation_grid_{objective}"
        grid_selected.append(payload)
        fixed_rows.append(evaluate_fixed_payload(test_matrix, payload, preset=payload["output_preset"]))

    GRID_WEIGHTS_PATH.write_text(
        json.dumps(
            {
                "protocol": "validation_select",
                "selection_kind": "grid",
                "validation_candidates_path": str(VALIDATION_CANDIDATES),
                "test_candidates_path": str(TEST_CANDIDATES),
                "selections": grid_selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fixed_metrics = pd.concat(fixed_rows, ignore_index=True)
    fixed_metrics.to_csv(FIXED_METRICS_PATH, index=False, encoding="utf-8-sig")
    fixed_by_dataset = wide_to_long(fixed_metrics)
    fixed_by_dataset.to_csv(FIXED_BY_DATASET_PATH, index=False, encoding="utf-8-sig")

    baseline = run_presets(test_matrix)
    baseline_row = metric_row(baseline, "A_hgnn_only")
    ddd_rows = fixed_by_dataset.loc[fixed_by_dataset["dataset"] == "DDD"].copy()
    result_rows = [
        {
            "Protocol": "HGNN baseline",
            "Selection Objective": "",
            "DDD Top1": float(baseline_row["DDD_top1"]),
            "DDD Top3": float(baseline_row["DDD_top3"]),
            "DDD Top5": float(baseline_row["DDD_top5"]),
            "Recall@50": float(baseline_row["DDD_rank_le_50"]),
            "Top1 Delta": 0.0,
            "Top3 Delta": 0.0,
            "Top5 Delta": 0.0,
            "Paper Usability": "主表 baseline",
        }
    ]
    for row in ddd_rows.itertuples(index=False):
        result_rows.append(
            {
                "Protocol": row.protocol,
                "Selection Objective": row.protocol.replace("validation_grid_", ""),
                "DDD Top1": float(row.top1),
                "DDD Top3": float(row.top3),
                "DDD Top5": float(row.top5),
                "Recall@50": float(row.rank_le_50),
                "Top1 Delta": float(row.top1) - float(baseline_row["DDD_top1"]),
                "Top3 Delta": float(row.top3) - float(baseline_row["DDD_top3"]),
                "Top5 Delta": float(row.top5) - float(baseline_row["DDD_top5"]),
                "Paper Usability": "validation-selected grid fixed test，可作为论文主线候选",
            }
        )
    result_df = pd.DataFrame(result_rows)

    selected_lines = ["### Grid"]
    for payload in grid_selected:
        selected_lines.append(f"- `{payload['output_preset']}` objective=`{payload['selection_objective']}`")
        selected_lines.append(f"  - weights: `{payload['weights']}`")
        selected_lines.append(f"  - validation metrics: `{payload.get('selected_metrics', {})}`")

    report = [
        "# DDD Validation-selected Grid Rerank Report",
        "",
        "- protocol: validation candidates select grid weights, test candidates fixed evaluation once.",
        "- DDD final mainline uses `validation_grid_DDD_top1`.",
        "- gated rerank and test-side exploratory search are not mainline methods.",
        f"- validation candidates: `{VALIDATION_CANDIDATES}`",
        f"- test candidates: `{TEST_CANDIDATES}`",
        "",
        "## Fixed Test Metrics",
        markdown_table(
            result_df,
            [
                "Protocol",
                "Selection Objective",
                "DDD Top1",
                "DDD Top3",
                "DDD Top5",
                "Recall@50",
                "Top1 Delta",
                "Top3 Delta",
                "Top5 Delta",
                "Paper Usability",
            ],
        ),
        "",
        "## Selected Weights",
        "\n".join(selected_lines),
        "",
        "## Paper Boundary",
        "- HGNN exact baseline can enter the main table.",
        "- validation-selected grid fixed-test rerank is the final DDD mainline.",
        "- gated rerank, HN dry-run, and test-side grid/gate are not mainline results.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "grid_weights": str(GRID_WEIGHTS_PATH),
                "fixed_metrics": str(FIXED_METRICS_PATH),
                "fixed_by_dataset": str(FIXED_BY_DATASET_PATH),
                "report": str(REPORT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
