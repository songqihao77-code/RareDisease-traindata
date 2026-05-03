from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "mainline_full_pipeline_hybrid_tag_v5"
DEFAULT_DETAILS_PATH = DEFAULT_RUN_DIR / "stage3_exact_eval" / "exact_details.csv"
DEFAULT_CANDIDATES_PATH = DEFAULT_RUN_DIR / "stage4_candidates" / "top50_candidates_test.csv"
KS = (1, 5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Top-1/Top-5/Recall@10/Recall@20/Recall@50 from exact "
            "evaluation details or exported top50 candidates."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Run directory containing stage3_exact_eval/ and/or stage4_candidates/. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--details-path",
        type=Path,
        default=None,
        help="Path to exact_details.csv. Preferred source because it contains true full-disease ranks.",
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=None,
        help="Path to top50_candidates_test.csv. Used only when exact details are unavailable or --prefer-candidates is set.",
    )
    parser.add_argument(
        "--prefer-candidates",
        action="store_true",
        help="Use top50 candidates even if exact_details.csv exists.",
    )
    parser.add_argument(
        "--no-all",
        action="store_true",
        help="Do not append an ALL aggregate row.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_input(args: argparse.Namespace) -> tuple[Path, str]:
    run_dir = resolve_path(args.run_dir)
    details_path = resolve_path(args.details_path) if args.details_path else run_dir / "stage3_exact_eval" / "exact_details.csv"
    candidates_path = (
        resolve_path(args.candidates_path)
        if args.candidates_path
        else run_dir / "stage4_candidates" / "top50_candidates_test.csv"
    )

    if args.prefer_candidates:
        if candidates_path.is_file():
            return candidates_path, "candidates"
        raise FileNotFoundError(f"candidate file not found: {candidates_path}")

    if details_path.is_file():
        return details_path, "details"
    if candidates_path.is_file():
        return candidates_path, "candidates"

    fallback_details = DEFAULT_DETAILS_PATH
    fallback_candidates = DEFAULT_CANDIDATES_PATH
    if fallback_details.is_file():
        return fallback_details, "details"
    if fallback_candidates.is_file():
        return fallback_candidates, "candidates"

    raise FileNotFoundError(
        "No input file found. Tried "
        f"details={details_path}, candidates={candidates_path}, "
        f"default_details={fallback_details}, default_candidates={fallback_candidates}"
    )


def read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        for row in reader:
            yield row


def parse_rank(value: str, *, row_id: str) -> int:
    try:
        rank = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid rank for {row_id}: {value!r}") from exc
    if rank < 1:
        raise ValueError(f"rank must be >= 1 for {row_id}: {rank}")
    return rank


def load_ranks_from_details(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    required = {"dataset_name", "true_rank"}
    for idx, row in enumerate(read_csv_rows(path), start=2):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        dataset = str(row["dataset_name"]).strip()
        rank = parse_rank(row["true_rank"], row_id=f"line {idx}")
        if not dataset:
            raise ValueError(f"empty dataset_name at line {idx}")
        rows.append((dataset, rank))
    if not rows:
        raise ValueError(f"no evaluation rows found in {path}")
    return rows


def load_ranks_from_candidates(path: Path) -> list[tuple[str, int]]:
    required = {"case_id", "dataset_name", "gold_id", "candidate_id", "original_rank"}
    best_gold_rank: dict[tuple[str, str], int] = {}
    case_dataset: dict[tuple[str, str], str] = {}

    for idx, row in enumerate(read_csv_rows(path), start=2):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        dataset = str(row["dataset_name"]).strip()
        case_id = str(row["case_id"]).strip()
        if not dataset or not case_id:
            raise ValueError(f"empty dataset_name/case_id at line {idx}")

        key = (dataset, case_id)
        case_dataset[key] = dataset
        if str(row["gold_id"]).strip() == str(row["candidate_id"]).strip():
            rank = parse_rank(row["original_rank"], row_id=f"line {idx}")
            best_gold_rank[key] = min(best_gold_rank.get(key, rank), rank)

    if not case_dataset:
        raise ValueError(f"no candidate rows found in {path}")

    ranks: list[tuple[str, int]] = []
    absent_rank = 10**9
    for key, dataset in case_dataset.items():
        ranks.append((dataset, best_gold_rank.get(key, absent_rank)))
    return ranks


def display_dataset_name(dataset: str) -> str:
    lowered = dataset.lower()
    if lowered.startswith("mimic") or "mimic_test" in lowered:
        return "MIMIC"
    return dataset


def sort_key(dataset: str) -> tuple[int, str]:
    order = {
        "DDD": 0,
        "MIMIC": 1,
        "HMS": 2,
        "LIRICAL": 3,
        "MME": 4,
        "MyGene2": 5,
        "RAMEDIS": 6,
        "ALL": 99,
    }
    return order.get(dataset, 50), dataset


def summarize(ranks: list[tuple[str, int]], include_all: bool) -> list[dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for raw_dataset, rank in ranks:
        grouped[display_dataset_name(raw_dataset)].append(rank)

    rows: list[dict[str, object]] = []
    for dataset, dataset_ranks in grouped.items():
        rows.append(build_metric_row(dataset, dataset_ranks))

    if include_all:
        rows.append(build_metric_row("ALL", [rank for _, rank in ranks]))

    rows.sort(key=lambda row: sort_key(str(row["Dataset"])))
    return rows


def build_metric_row(dataset: str, ranks: list[int]) -> dict[str, object]:
    if not ranks:
        raise ValueError(f"dataset has no ranks: {dataset}")
    cases = len(ranks)
    return {
        "Dataset": dataset,
        "Cases": cases,
        "Top-1": sum(rank == 1 for rank in ranks) / cases,
        "Top-5": sum(rank <= 5 for rank in ranks) / cases,
        "Recall@10": sum(rank <= 10 for rank in ranks) / cases,
        "Recall@20": sum(rank <= 20 for rank in ranks) / cases,
        "Recall@50": sum(rank <= 50 for rank in ranks) / cases,
    }


def fmt_rate(value: object) -> str:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return str(value)
    return f"{float(value) * 100:.2f}%"


def print_markdown(rows: list[dict[str, object]]) -> None:
    print("| Dataset | Cases | Top-1 | Top-5 | Recall@10 | Recall@20 | Recall@50 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['Dataset']} | {row['Cases']} | "
            f"{fmt_rate(row['Top-1'])} | {fmt_rate(row['Top-5'])} | "
            f"{fmt_rate(row['Recall@10'])} | {fmt_rate(row['Recall@20'])} | "
            f"{fmt_rate(row['Recall@50'])} |"
        )


def main() -> None:
    args = parse_args()
    input_path, input_kind = resolve_input(args)
    if input_kind == "details":
        ranks = load_ranks_from_details(input_path)
    else:
        ranks = load_ranks_from_candidates(input_path)
    rows = summarize(ranks, include_all=not args.no_all)
    print_markdown(rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
