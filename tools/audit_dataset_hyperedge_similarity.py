from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"D:\RareDisease-traindata")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dataset_hyperedge_similarity"
HYPEREDGE_PATH = (
    PROJECT_ROOT
    / "LLLdataset"
    / "DiseaseHy"
    / "rare_disease_hgnn_clean_package_v59"
    / "v59_hyperedge_weighted_patched.csv"
)
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


def load_processed_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return (
        df[REQUIRED_COLUMNS]
        .dropna(subset=REQUIRED_COLUMNS)
        .astype({"case_id": "string", "mondo_label": "string", "hpo_id": "string"})
        .drop_duplicates()
    )


def load_hyperedge_table(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, float]], dict[str, float]]:
    df = pd.read_csv(path, dtype={"mondo_id": str, "hpo_id": str, "weight": float})
    required = {"mondo_id", "hpo_id", "weight"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    hpo_sets: dict[str, set[str]] = {}
    weight_maps: dict[str, dict[str, float]] = {}
    total_weights: dict[str, float] = {}

    for mondo_id, group in df.groupby("mondo_id", sort=False):
        ordered = (
            group[["hpo_id", "weight"]]
            .dropna(subset=["hpo_id", "weight"])
            .assign(weight=lambda x: x["weight"].astype(float))
            .groupby("hpo_id", sort=False)["weight"]
            .max()
            .sort_index()
        )
        weight_map = ordered.to_dict()
        hpo_sets[str(mondo_id)] = set(weight_map)
        weight_maps[str(mondo_id)] = weight_map
        total_weights[str(mondo_id)] = float(sum(weight_map.values()))
    return hpo_sets, weight_maps, total_weights


def build_case_level_table(
    dataset_name: str,
    df: pd.DataFrame,
    hyperedge_hpo_sets: dict[str, set[str]],
    hyperedge_weight_maps: dict[str, dict[str, float]],
    hyperedge_total_weights: dict[str, float],
) -> pd.DataFrame:
    case_df = (
        df.groupby(["case_id", "mondo_label"], sort=False)["hpo_id"]
        .agg(lambda values: sorted(set(values.tolist())))
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for row in case_df.itertuples(index=False):
        case_id = str(row.case_id)
        mondo_id = str(row.mondo_label)
        case_hpos = set(row.hpo_id)

        hyperedge_hpos = hyperedge_hpo_sets.get(mondo_id, set())
        weight_map = hyperedge_weight_maps.get(mondo_id, {})
        total_weight = float(hyperedge_total_weights.get(mondo_id, 0.0))

        shared = case_hpos & hyperedge_hpos
        union = case_hpos | hyperedge_hpos
        shared_weight = float(sum(weight_map.get(hpo_id, 0.0) for hpo_id in shared))

        rows.append(
            {
                "dataset": dataset_name,
                "case_id": case_id,
                "mondo_label": mondo_id,
                "case_hpo_count": len(case_hpos),
                "hyperedge_hpo_count": len(hyperedge_hpos),
                "shared_hpo_count": len(shared),
                "case_hit_ratio": len(shared) / len(case_hpos) if case_hpos else float("nan"),
                "hyperedge_cover_ratio": len(shared) / len(hyperedge_hpos) if hyperedge_hpos else float("nan"),
                "jaccard": len(shared) / len(union) if union else float("nan"),
                "weighted_hit_ratio": shared_weight / total_weight if total_weight > 0 else float("nan"),
                "shared_weight_sum": shared_weight,
                "hyperedge_weight_sum": total_weight,
                "shared_hpo_list": "|".join(sorted(shared)),
                "case_only_hpo_list": "|".join(sorted(case_hpos - hyperedge_hpos)),
                "hyperedge_only_hpo_list": "|".join(sorted(hyperedge_hpos - case_hpos)),
                "mapped_to_hyperedge": mondo_id in hyperedge_hpo_sets,
            }
        )

    return pd.DataFrame(rows)


def summarize_dataset(case_level_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for dataset_name, group in case_level_df.groupby("dataset", sort=False):
        mapped = group.loc[group["mapped_to_hyperedge"]].copy()
        summary_rows.append(
            {
                "dataset": dataset_name,
                "n_cases": int(group["case_id"].nunique()),
                "n_case_labels": int(len(group)),
                "n_mapped_case_labels": int(len(mapped)),
                "mapping_success_ratio": float(len(mapped) / len(group)) if len(group) else float("nan"),
                "mean_case_hit_ratio": mapped["case_hit_ratio"].mean(),
                "median_case_hit_ratio": mapped["case_hit_ratio"].median(),
                "mean_hyperedge_cover_ratio": mapped["hyperedge_cover_ratio"].mean(),
                "median_hyperedge_cover_ratio": mapped["hyperedge_cover_ratio"].median(),
                "mean_jaccard": mapped["jaccard"].mean(),
                "median_jaccard": mapped["jaccard"].median(),
                "mean_weighted_hit_ratio": mapped["weighted_hit_ratio"].mean(),
                "median_weighted_hit_ratio": mapped["weighted_hit_ratio"].median(),
                "ratio_weighted_hit_ge_0_3": (mapped["weighted_hit_ratio"] >= 0.3).mean() if not mapped.empty else float("nan"),
                "ratio_weighted_hit_ge_0_5": (mapped["weighted_hit_ratio"] >= 0.5).mean() if not mapped.empty else float("nan"),
                "ratio_weighted_hit_ge_0_8": (mapped["weighted_hit_ratio"] >= 0.8).mean() if not mapped.empty else float("nan"),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("mean_weighted_hit_ratio", ascending=False).reset_index(drop=True)


def summarize_dataset_mondo(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mapped_df = case_level_df.loc[case_level_df["mapped_to_hyperedge"]].copy()
    for (dataset_name, mondo_id), group in mapped_df.groupby(["dataset", "mondo_label"], sort=False):
        rows.append(
            {
                "dataset": dataset_name,
                "mondo_label": mondo_id,
                "n_case_labels": int(len(group)),
                "mean_case_hit_ratio": group["case_hit_ratio"].mean(),
                "mean_hyperedge_cover_ratio": group["hyperedge_cover_ratio"].mean(),
                "mean_jaccard": group["jaccard"].mean(),
                "mean_weighted_hit_ratio": group["weighted_hit_ratio"].mean(),
                "median_weighted_hit_ratio": group["weighted_hit_ratio"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "n_case_labels", "mean_weighted_hit_ratio", "mean_jaccard"],
        ascending=[True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperedge_hpo_sets, hyperedge_weight_maps, hyperedge_total_weights = load_hyperedge_table(HYPEREDGE_PATH)

    case_level_frames: list[pd.DataFrame] = []
    for dataset_name, dataset_path in DATASET_PATHS.items():
        dataset_df = load_processed_dataset(dataset_path)
        case_level_frames.append(
            build_case_level_table(
                dataset_name=dataset_name,
                df=dataset_df,
                hyperedge_hpo_sets=hyperedge_hpo_sets,
                hyperedge_weight_maps=hyperedge_weight_maps,
                hyperedge_total_weights=hyperedge_total_weights,
            )
        )

    case_level_df = pd.concat(case_level_frames, ignore_index=True)
    dataset_summary_df = summarize_dataset(case_level_df)
    dataset_mondo_df = summarize_dataset_mondo(case_level_df)

    case_level_path = output_dir / "case_level_similarity.csv"
    dataset_summary_path = output_dir / "dataset_summary.csv"
    dataset_mondo_path = output_dir / "dataset_mondo_summary.csv"
    metadata_path = output_dir / "metadata.json"

    case_level_df.to_csv(case_level_path, index=False, encoding="utf-8-sig")
    dataset_summary_df.to_csv(dataset_summary_path, index=False, encoding="utf-8-sig")
    dataset_mondo_df.to_csv(dataset_mondo_path, index=False, encoding="utf-8-sig")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hyperedge_path": str(HYPEREDGE_PATH),
        "output_dir": str(output_dir),
        "case_level_path": str(case_level_path),
        "dataset_summary_path": str(dataset_summary_path),
        "dataset_mondo_summary_path": str(dataset_mondo_path),
        "metrics": {
            "case_hit_ratio": "shared_hpo_count / case_hpo_count",
            "hyperedge_cover_ratio": "shared_hpo_count / hyperedge_hpo_count",
            "jaccard": "shared_hpo_count / union_hpo_count",
            "weighted_hit_ratio": "sum(weight of shared HPO in hyperedge) / sum(all hyperedge weights)",
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
