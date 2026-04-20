from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "hpo_similarity_by_mondo"
DATASET_PATHS = {
    "HMS": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "HMS.xlsx",
    "LIRICAL": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "LIRICAL.xlsx",
    "mimic_rag_0425": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "mimic_rag_0425.csv",
    "MME": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "MME.xlsx",
    "MyGene2": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "MyGene2.xlsx",
    "RAMEDIS": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "RAMEDIS.xlsx",
    "ddd_test": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "ddd_test.csv",
    "mimic_test": PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "test" / "mimic_test.csv",
}
REQUIRED_COLUMNS = ["case_id", "mondo_label", "hpo_id"]


@dataclass(frozen=True, slots=True)
class PairwiseStats:
    avg_jaccard: float
    min_jaccard: float
    max_jaccard: float
    avg_shared_hpo: float
    avg_diff_hpo: float
    pair_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze HPO overlap within each MONDO label by dataset and export "
            "dataset-level summaries plus per-MONDO details."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for exported results. Defaults to outputs/hpo_similarity_by_mondo/<timestamp>.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = (
        df[REQUIRED_COLUMNS]
        .dropna(subset=REQUIRED_COLUMNS)
        .astype({"case_id": "string", "mondo_label": "string", "hpo_id": "string"})
        .drop_duplicates()
    )
    return frame


def build_case_sets(df: pd.DataFrame) -> dict[str, list[set[str]]]:
    case_group = (
        df.groupby(["mondo_label", "case_id"], sort=True)["hpo_id"]
        .agg(lambda values: set(values.tolist()))
        .reset_index()
    )
    grouped: dict[str, list[set[str]]] = {}
    for mondo_label, group in case_group.groupby("mondo_label", sort=True):
        grouped[str(mondo_label)] = [set(hpo_set) for hpo_set in group["hpo_id"].tolist()]
    return grouped


def calc_pairwise_stats(case_sets: list[set[str]]) -> PairwiseStats:
    jaccards: list[float] = []
    shared_counts: list[int] = []
    diff_counts: list[int] = []
    pair_count = 0

    for left, right in combinations(case_sets, 2):
        pair_count += 1
        shared = left & right
        union = left | right
        diff = union - shared
        jaccards.append(len(shared) / len(union) if union else 1.0)
        shared_counts.append(len(shared))
        diff_counts.append(len(diff))

    if pair_count == 0:
        return PairwiseStats(
            avg_jaccard=1.0,
            min_jaccard=1.0,
            max_jaccard=1.0,
            avg_shared_hpo=0.0,
            avg_diff_hpo=0.0,
            pair_count=0,
        )

    return PairwiseStats(
        avg_jaccard=mean(jaccards),
        min_jaccard=min(jaccards),
        max_jaccard=max(jaccards),
        avg_shared_hpo=mean(shared_counts),
        avg_diff_hpo=mean(diff_counts),
        pair_count=pair_count,
    )


def analyze_dataset(dataset_name: str, path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = load_dataset(path)
    case_sets_by_mondo = build_case_sets(df)
    details: list[dict[str, object]] = []
    total_cases = df["case_id"].nunique()
    total_mondo = df["mondo_label"].nunique()

    for mondo_label, case_sets in case_sets_by_mondo.items():
        case_count = len(case_sets)
        union_hpo = set().union(*case_sets)
        intersection_hpo = set.intersection(*case_sets)
        unique_hpo_per_case = [len(case_set) for case_set in case_sets]
        pairwise = calc_pairwise_stats(case_sets)

        detail = {
            "dataset": dataset_name,
            "mondo_label": mondo_label,
            "case_count": case_count,
            "union_hpo_count": len(union_hpo),
            "common_hpo_count": len(intersection_hpo),
            "different_hpo_count": len(union_hpo - intersection_hpo),
            "strict_common_ratio": (len(intersection_hpo) / len(union_hpo)) if union_hpo else 1.0,
            "avg_hpo_per_case": mean(unique_hpo_per_case),
            "median_hpo_per_case": median(unique_hpo_per_case),
            "min_hpo_per_case": min(unique_hpo_per_case),
            "max_hpo_per_case": max(unique_hpo_per_case),
            **asdict(pairwise),
            "common_hpo_list": ";".join(sorted(intersection_hpo)),
            "different_hpo_list": ";".join(sorted(union_hpo - intersection_hpo)),
        }
        details.append(detail)

    detail_df = pd.DataFrame(details).sort_values(
        by=["case_count", "avg_jaccard", "strict_common_ratio", "mondo_label"],
        ascending=[False, False, False, True],
        kind="stable",
    )

    analyzable = detail_df[detail_df["case_count"] >= 2].copy()
    if analyzable.empty:
        summary = {
            "dataset": dataset_name,
            "source_path": str(path),
            "row_count": len(df),
            "case_count": total_cases,
            "mondo_count": total_mondo,
            "analyzable_mondo_count": 0,
            "single_case_mondo_count": int((detail_df["case_count"] == 1).sum()),
            "avg_case_count_per_analyzable_mondo": 0.0,
            "avg_strict_common_ratio": 0.0,
            "median_strict_common_ratio": 0.0,
            "weighted_strict_common_ratio": 0.0,
            "avg_pairwise_jaccard": 0.0,
            "median_pairwise_jaccard": 0.0,
            "pair_count_weighted_jaccard": 0.0,
            "total_common_hpo_count": 0,
            "total_different_hpo_count": 0,
            "zero_common_mondo_count": 0,
        }
        return detail_df, summary

    weighted_strict_common_ratio = analyzable["common_hpo_count"].sum() / analyzable["union_hpo_count"].sum()
    pair_weights = analyzable["pair_count"].replace(0, 1)
    pair_count_weighted_jaccard = (
        (analyzable["avg_jaccard"] * pair_weights).sum() / pair_weights.sum()
    )
    summary = {
        "dataset": dataset_name,
        "source_path": str(path),
        "row_count": len(df),
        "case_count": total_cases,
        "mondo_count": total_mondo,
        "analyzable_mondo_count": int(len(analyzable)),
        "single_case_mondo_count": int((detail_df["case_count"] == 1).sum()),
        "avg_case_count_per_analyzable_mondo": analyzable["case_count"].mean(),
        "avg_strict_common_ratio": analyzable["strict_common_ratio"].mean(),
        "median_strict_common_ratio": analyzable["strict_common_ratio"].median(),
        "weighted_strict_common_ratio": weighted_strict_common_ratio,
        "avg_pairwise_jaccard": analyzable["avg_jaccard"].mean(),
        "median_pairwise_jaccard": analyzable["avg_jaccard"].median(),
        "pair_count_weighted_jaccard": pair_count_weighted_jaccard,
        "total_common_hpo_count": int(analyzable["common_hpo_count"].sum()),
        "total_different_hpo_count": int(analyzable["different_hpo_count"].sum()),
        "zero_common_mondo_count": int((analyzable["common_hpo_count"] == 0).sum()),
    }
    return detail_df, summary


def export_results(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    preview_rows: list[dict[str, object]] = []

    for dataset_name, path in DATASET_PATHS.items():
        detail_df, summary = analyze_dataset(dataset_name, path)
        summary_rows.append(summary)

        detail_path = output_root / f"{dataset_name}_detail.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

        analyzable = detail_df[detail_df["case_count"] >= 2].copy()
        if not analyzable.empty:
            top_high = analyzable.nlargest(3, ["avg_jaccard", "strict_common_ratio", "case_count"])[
                [
                    "dataset",
                    "mondo_label",
                    "case_count",
                    "avg_jaccard",
                    "strict_common_ratio",
                    "common_hpo_count",
                    "different_hpo_count",
                ]
            ]
            top_low = analyzable.nsmallest(3, ["avg_jaccard", "strict_common_ratio", "case_count"])[
                [
                    "dataset",
                    "mondo_label",
                    "case_count",
                    "avg_jaccard",
                    "strict_common_ratio",
                    "common_hpo_count",
                    "different_hpo_count",
                ]
            ]
            preview_rows.extend(top_high.to_dict(orient="records"))
            preview_rows.extend(top_low.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset", kind="stable")
    summary_path = output_root / "summary_by_dataset.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    preview_df = pd.DataFrame(preview_rows).sort_values(
        by=["dataset", "avg_jaccard", "strict_common_ratio", "case_count"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    preview_path = output_root / "mondo_similarity_preview.csv"
    preview_df.to_csv(preview_path, index=False, encoding="utf-8-sig")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "detail_paths": {name: str(output_root / f"{name}_detail.csv") for name in DATASET_PATHS},
        "metric_definition": {
            "strict_common_ratio": "common_hpo_count / union_hpo_count within the same MONDO label",
            "avg_jaccard": "average pairwise Jaccard similarity across cases under the same MONDO label",
            "different_hpo_count": "union_hpo_count - common_hpo_count",
        },
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_OUTPUT_ROOT / timestamp
    else:
        output_root = args.output_root.resolve()

    metadata = export_results(output_root)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
