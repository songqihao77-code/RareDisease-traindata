from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATION_DIR = PROJECT_ROOT / "data" / "generation_data"
DEFAULT_INPUT_PATH = DEFAULT_GENERATION_DIR / "synthetic_low_count_mondo_cases_deepseek_v32.xlsx"
DEFAULT_OUTPUT_PATH = DEFAULT_GENERATION_DIR / "synthetic_low_count_mondo_cases_deepseek_v32_filtered.xlsx"


def split_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ()
    return tuple(part for part in text.split("|") if part)


def jaccard(ids_a: tuple[str, ...], ids_b: tuple[str, ...]) -> float:
    set_a = set(ids_a)
    set_b = set(ids_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter synthetic cases by quality rules.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--min_seed_jaccard", type=float, default=0.30)
    parser.add_argument("--keep_only_fully_filled_mondos", action="store_true", default=True)
    parser.add_argument("--keep_all_mondos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_only_fully_filled_mondos = bool(args.keep_only_fully_filled_mondos and not args.keep_all_mondos)

    generated_df = pd.read_excel(args.input_path, sheet_name="generated_cases", dtype=str)
    metadata_df = pd.read_excel(args.input_path, sheet_name="case_metadata", dtype=str)
    summary_df = pd.read_excel(args.input_path, sheet_name="generation_summary", dtype=str)
    params_df = pd.read_excel(args.input_path, sheet_name="run_params", dtype=str)
    audit_df = pd.read_excel(args.input_path, sheet_name="llm_audit", dtype=str)

    summary_df["generated_case_count"] = pd.to_numeric(summary_df["generated_case_count"], errors="coerce").fillna(0).astype(int)
    summary_df["requested_generated_case_count"] = pd.to_numeric(
        summary_df["requested_generated_case_count"], errors="coerce"
    ).fillna(0).astype(int)
    summary_df["original_total_case_pair_count"] = pd.to_numeric(
        summary_df["original_total_case_pair_count"], errors="coerce"
    ).fillna(0).astype(int)

    metadata_df["original_total_case_pair_count"] = pd.to_numeric(
        metadata_df["original_total_case_pair_count"], errors="coerce"
    ).fillna(0).astype(int)
    metadata_df["requested_target_total_case_count"] = pd.to_numeric(
        metadata_df["requested_target_total_case_count"], errors="coerce"
    ).fillna(0).astype(int)
    metadata_df["removed_non_core_count"] = pd.to_numeric(
        metadata_df["removed_non_core_count"], errors="coerce"
    ).fillna(0).astype(int)
    metadata_df["added_noise_count"] = pd.to_numeric(
        metadata_df["added_noise_count"], errors="coerce"
    ).fillna(0).astype(int)
    metadata_df["seed_hpo_count"] = pd.to_numeric(metadata_df["seed_hpo_count"], errors="coerce").fillna(0).astype(int)
    metadata_df["final_hpo_count"] = pd.to_numeric(metadata_df["final_hpo_count"], errors="coerce").fillna(0).astype(int)

    metadata_df["seed_hpo_ids_tuple"] = metadata_df["seed_hpo_ids"].map(split_ids)
    metadata_df["final_hpo_ids_tuple"] = metadata_df["final_hpo_ids"].map(split_ids)
    metadata_df["seed_final_jaccard"] = [
        jaccard(seed_ids, final_ids)
        for seed_ids, final_ids in zip(metadata_df["seed_hpo_ids_tuple"], metadata_df["final_hpo_ids_tuple"])
    ]

    if keep_only_fully_filled_mondos:
        keep_mondo_ids = set(
            summary_df.loc[
                summary_df["generated_case_count"] == summary_df["requested_generated_case_count"],
                "mondo_label",
            ].astype(str)
        )
    else:
        keep_mondo_ids = set(summary_df["mondo_label"].astype(str))

    metadata_df["quality_filter_reason"] = "keep"
    metadata_df.loc[~metadata_df["mondo_label"].isin(keep_mondo_ids), "quality_filter_reason"] = "drop_underfilled_mondo"
    metadata_df.loc[
        metadata_df["mondo_label"].isin(keep_mondo_ids)
        & (metadata_df["seed_final_jaccard"] < float(args.min_seed_jaccard)),
        "quality_filter_reason",
    ] = "drop_low_seed_jaccard"

    kept_metadata_df = metadata_df.loc[metadata_df["quality_filter_reason"] == "keep"].copy()
    kept_case_ids = set(kept_metadata_df["case_id"].astype(str))
    kept_generated_df = generated_df.loc[generated_df["case_id"].isin(kept_case_ids)].copy()
    kept_audit_df = audit_df.loc[audit_df["case_id"].astype(str).isin(kept_case_ids)].copy()

    filtered_summary_df = summary_df.loc[summary_df["mondo_label"].astype(str).isin(keep_mondo_ids)].copy()
    filtered_case_counts = kept_metadata_df.groupby("mondo_label")["case_id"].nunique().astype(int)
    filtered_summary_df["generated_case_count_before_filter"] = filtered_summary_df["generated_case_count"].astype(int)
    filtered_summary_df["generated_case_count"] = filtered_summary_df["mondo_label"].map(filtered_case_counts).fillna(0).astype(int)
    filtered_summary_df["removed_low_jaccard_case_count"] = (
        filtered_summary_df["generated_case_count_before_filter"] - filtered_summary_df["generated_case_count"]
    )
    filtered_summary_df["total_after_generation"] = (
        filtered_summary_df["original_total_case_pair_count"] + filtered_summary_df["generated_case_count"]
    )
    filtered_summary_df = filtered_summary_df.sort_values(
        ["original_total_case_pair_count", "mondo_label"], ascending=[True, True]
    ).reset_index(drop=True)

    filter_summary_df = pd.DataFrame(
        [
            {"metric": "input_path", "value": str(args.input_path)},
            {"metric": "output_path", "value": str(args.output_path)},
            {"metric": "min_seed_jaccard", "value": float(args.min_seed_jaccard)},
            {"metric": "keep_only_fully_filled_mondos", "value": keep_only_fully_filled_mondos},
            {"metric": "original_synthetic_case_count", "value": int(metadata_df["case_id"].nunique())},
            {"metric": "filtered_synthetic_case_count", "value": int(kept_metadata_df["case_id"].nunique())},
            {"metric": "removed_synthetic_case_count", "value": int(metadata_df["case_id"].nunique() - kept_metadata_df["case_id"].nunique())},
            {"metric": "original_target_mondo_count", "value": int(summary_df["mondo_label"].nunique())},
            {"metric": "filtered_target_mondo_count", "value": int(filtered_summary_df["mondo_label"].nunique())},
            {
                "metric": "removed_underfilled_mondo_count",
                "value": int((summary_df["generated_case_count"] != summary_df["requested_generated_case_count"]).sum())
                if keep_only_fully_filled_mondos
                else 0,
            },
            {
                "metric": "removed_low_seed_jaccard_case_count",
                "value": int((metadata_df["quality_filter_reason"] == "drop_low_seed_jaccard").sum()),
            },
        ]
    )

    params_df = pd.concat(
        [
            params_df,
            pd.DataFrame(
                [
                    {"parameter": "quality_filter_min_seed_jaccard", "value": float(args.min_seed_jaccard)},
                    {"parameter": "quality_filter_keep_only_fully_filled_mondos", "value": keep_only_fully_filled_mondos},
                    {"parameter": "quality_filter_input_path", "value": str(args.input_path)},
                ]
            ),
        ],
        ignore_index=True,
    )

    kept_metadata_df = kept_metadata_df.drop(columns=["seed_hpo_ids_tuple", "final_hpo_ids_tuple"])
    filter_decisions_df = metadata_df.drop(columns=["seed_hpo_ids_tuple", "final_hpo_ids_tuple"]).copy()
    filter_decisions_df = filter_decisions_df.sort_values(
        ["quality_filter_reason", "seed_final_jaccard", "mondo_label", "case_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_path, engine="openpyxl") as writer:
        kept_generated_df.to_excel(writer, sheet_name="generated_cases", index=False)
        kept_metadata_df.to_excel(writer, sheet_name="case_metadata", index=False)
        filtered_summary_df.to_excel(writer, sheet_name="generation_summary", index=False)
        params_df.to_excel(writer, sheet_name="run_params", index=False)
        kept_audit_df.to_excel(writer, sheet_name="llm_audit", index=False)
        filter_summary_df.to_excel(writer, sheet_name="filter_summary", index=False)
        filter_decisions_df.to_excel(writer, sheet_name="filter_decisions", index=False)

    print(f"input_path={args.input_path}")
    print(f"output_path={args.output_path}")
    print(f"min_seed_jaccard={args.min_seed_jaccard}")
    print(f"keep_only_fully_filled_mondos={keep_only_fully_filled_mondos}")
    print(f"original_synthetic_case_count={metadata_df['case_id'].nunique()}")
    print(f"filtered_synthetic_case_count={kept_metadata_df['case_id'].nunique()}")
    print(f"removed_underfilled_mondo_count={int((summary_df['generated_case_count'] != summary_df['requested_generated_case_count']).sum()) if keep_only_fully_filled_mondos else 0}")
    print(f"removed_low_seed_jaccard_case_count={int((metadata_df['quality_filter_reason'] == 'drop_low_seed_jaccard').sum())}")


if __name__ == "__main__":
    main()
