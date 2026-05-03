from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_namespaced_case_id, read_case_table_file
from src.evaluation.mondo_canonicalizer import MondoCanonicalizer
from src.evaluation.multilabel_metrics import (
    MISSING_RANK,
    compute_rank_metrics,
    ordered_unique_labels,
)
from tools.run_top50_evidence_rerank import load_candidates, score_matrix, to_matrix


DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "mainline_full_pipeline_hybrid_tag_v5"
DEFAULT_LOCKED_CONFIG = PROJECT_ROOT / "configs" / "mainline_full_pipeline_hybrid_tag_v5.locked.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "evaluator_multilabel_canonical_audit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluator_multilabel_canonical_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit strict, any-label, canonical MONDO, and obsolete-aware evaluation metrics."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--locked-config", type=Path, default=DEFAULT_LOCKED_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--out-report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mondo-json", type=Path, default=None)
    parser.add_argument("--obsolete-mondo-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_target_dir(base: Path, stamp: str) -> Path:
    base = resolve_path(base)
    target = base / f"run_{stamp}" if base.exists() else base
    target.mkdir(parents=True, exist_ok=False)
    return target


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (list, dict))).any():
            out[column] = out[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    out.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_files_from_data_config(data_config: dict[str, Any], data_config_path: Path) -> list[Path]:
    config_dir = data_config_path.resolve().parent
    files: list[Path] = []
    if data_config.get("test_files"):
        for item in data_config["test_files"]:
            path = Path(item)
            if not path.is_absolute():
                path = (config_dir / path).resolve()
            files.append(path)
    elif data_config.get("test_dir"):
        test_dir = Path(data_config["test_dir"])
        if not test_dir.is_absolute():
            test_dir = (config_dir / test_dir).resolve()
        files = sorted(
            path
            for path in test_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv"}
        )
    else:
        raise KeyError("data config must contain test_files or test_dir")
    return [path for path in files if not path.name.startswith("~$")]


def load_disease_index_from_locked_config(locked_config_path: Path) -> tuple[set[str], Path]:
    locked = load_yaml(locked_config_path)
    stage2_path = resolve_path(locked["paths"]["finetune_config"])
    stage2 = load_yaml(stage2_path)
    disease_index_path = resolve_path(stage2["paths"]["disease_index_path"])
    disease_df = pd.read_excel(disease_index_path, dtype={"mondo_id": str})
    return set(disease_df["mondo_id"].astype(str)), disease_index_path


def load_case_labels(
    data_config_path: Path,
    disease_ids: set[str],
    canonicalizer: MondoCanonicalizer,
) -> pd.DataFrame:
    data_config = load_yaml(data_config_path)
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))
    rows: list[dict[str, Any]] = []

    for path in test_files_from_data_config(data_config, data_config_path):
        df = read_case_table_file(path)
        if label_col not in df.columns and "mondo_id" in df.columns:
            df = df.rename(columns={"mondo_id": label_col})
        missing = {case_id_col, label_col, hpo_col} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df = df[[case_id_col, label_col, hpo_col]].copy()
        df[case_id_col] = df[case_id_col].astype(str).apply(
            lambda raw_case_id: build_namespaced_case_id(raw_case_id, path, "test")
        )
        for case_id, group in df.groupby(case_id_col, sort=False):
            labels = ordered_unique_labels(group[label_col].tolist())
            canonical_labels = canonicalizer.canonicalize_many(labels)
            obsolete_labels = [label for label in labels if canonicalizer.is_obsolete(label)]
            primary = labels[0] if labels else ""
            rows.append(
                {
                    "case_id": str(case_id),
                    "dataset": path.stem,
                    "source_file": path.name,
                    "source_path": str(path.resolve()),
                    "primary_label": primary,
                    "all_labels": labels,
                    "canonical_primary_label": canonicalizer.canonicalize(primary),
                    "canonical_all_labels": canonical_labels,
                    "label_count": int(len(labels)),
                    "is_multilabel": bool(len(labels) > 1),
                    "hpo_count": int(group[hpo_col].dropna().astype(str).nunique()),
                    "primary_in_disease_index": bool(primary in disease_ids),
                    "any_label_in_disease_index": bool(any(label in disease_ids for label in labels)),
                    "all_labels_in_disease_index_count": int(sum(label in disease_ids for label in labels)),
                    "primary_obsolete": bool(canonicalizer.is_obsolete(primary)),
                    "any_label_obsolete": bool(len(obsolete_labels) > 0),
                    "obsolete_labels": obsolete_labels,
                }
            )
    return pd.DataFrame(rows)


def min_rank_for_candidates(candidate_ids: list[str], target_ids: set[str]) -> int:
    if not target_ids:
        return MISSING_RANK
    for rank, candidate_id in enumerate(candidate_ids, start=1):
        if candidate_id in target_ids:
            return rank
    return MISSING_RANK


def rank_cases_from_ordered_candidates(
    ordered: pd.DataFrame,
    labels: pd.DataFrame,
    canonicalizer: MondoCanonicalizer,
    *,
    rank_col: str,
    candidate_col: str = "candidate_id",
) -> pd.DataFrame:
    label_map = labels.set_index("case_id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for case_id, group in ordered.sort_values(["case_id", rank_col], kind="stable").groupby("case_id", sort=False):
        meta = label_map.get(str(case_id))
        if not meta:
            continue
        candidate_ids = group[candidate_col].astype(str).tolist()
        canonical_candidate_ids = [canonicalizer.canonicalize(candidate_id) for candidate_id in candidate_ids]
        all_labels = list(meta["all_labels"])
        canonical_all_labels = list(meta["canonical_all_labels"])
        primary = str(meta["primary_label"])
        rows.append(
            {
                "case_id": str(case_id),
                "dataset": meta["dataset"],
                "primary_label": primary,
                "all_labels": all_labels,
                "canonical_all_labels": canonical_all_labels,
                "strict_primary_rank": min_rank_for_candidates(candidate_ids, {primary}),
                "any_label_rank": min_rank_for_candidates(candidate_ids, set(all_labels)),
                "canonical_primary_rank": min_rank_for_candidates(
                    canonical_candidate_ids,
                    {canonicalizer.canonicalize(primary)},
                ),
                "canonical_any_label_rank": min_rank_for_candidates(
                    canonical_candidate_ids,
                    set(canonical_all_labels),
                ),
                "top1_candidate": candidate_ids[0] if candidate_ids else "",
                "top1_candidate_canonical": canonical_candidate_ids[0] if canonical_candidate_ids else "",
                "top1_candidate_obsolete": bool(
                    canonicalizer.is_obsolete(candidate_ids[0]) if candidate_ids else False
                ),
                "candidate_count": int(len(candidate_ids)),
            }
        )
    return pd.DataFrame(rows)


def load_hgnn_top50_ranks(run_dir: Path, labels: pd.DataFrame, canonicalizer: MondoCanonicalizer) -> pd.DataFrame:
    path = run_dir / "stage4_candidates" / "top50_candidates_test.csv"
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    df["original_rank"] = pd.to_numeric(df["original_rank"], errors="raise").astype(int)
    return rank_cases_from_ordered_candidates(df, labels, canonicalizer, rank_col="original_rank")


def load_ddd_rerank_ranks(run_dir: Path, labels: pd.DataFrame, canonicalizer: MondoCanonicalizer) -> pd.DataFrame:
    candidate_path = run_dir / "stage4_candidates" / "top50_candidates_test.csv"
    weights_path = run_dir / "stage5_ddd_rerank" / "ddd_val_selected_grid_weights.json"
    if not weights_path.is_file():
        return pd.DataFrame()
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    candidates = load_candidates(candidate_path)
    matrix = to_matrix(candidates)
    scores = score_matrix(matrix, payload["weights"])
    candidate_ids = candidates["candidate_id"].to_numpy(dtype=str).reshape(scores.shape)
    order = np.lexsort((matrix.original_rank, -scores), axis=1)
    rows: list[dict[str, Any]] = []
    for row_idx, case_id in enumerate(matrix.case_ids):
        if matrix.dataset_names[row_idx] != "DDD":
            continue
        for new_rank, candidate_idx in enumerate(order[row_idx].tolist(), start=1):
            rows.append(
                {
                    "case_id": str(case_id),
                    "candidate_id": str(candidate_ids[row_idx, candidate_idx]),
                    "rerank_rank": int(new_rank),
                }
            )
    if not rows:
        return pd.DataFrame()
    return rank_cases_from_ordered_candidates(
        pd.DataFrame(rows),
        labels,
        canonicalizer,
        rank_col="rerank_rank",
    )


def load_mimic_similarcase_ranks(run_dir: Path, labels: pd.DataFrame, canonicalizer: MondoCanonicalizer) -> pd.DataFrame:
    path = run_dir / "stage6_mimic_similar_case" / "similar_case_fixed_test_ranked_candidates.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"case_id": str, "gold_id": str, "candidate_id": str})
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    return rank_cases_from_ordered_candidates(df, labels, canonicalizer, rank_col="rank")


def load_exact_strict_ranks(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "stage3_exact_eval" / "exact_details.csv"
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "true_label": str})
    df["true_rank"] = pd.to_numeric(df["true_rank"], errors="coerce").fillna(MISSING_RANK).astype(int)
    return df[["case_id", "dataset_name", "true_label", "pred_top1", "true_rank"]].rename(
        columns={"dataset_name": "dataset", "true_label": "primary_label", "true_rank": "strict_primary_rank"}
    )


def load_final_strict_ranks(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "mainline_final_case_ranks.csv"
    df = pd.read_csv(path, dtype={"case_id": str, "dataset": str, "gold_id": str})
    df["final_rank"] = pd.to_numeric(df["final_rank"], errors="coerce").fillna(MISSING_RANK).astype(int)
    return df[["case_id", "dataset", "gold_id", "final_rank", "module_applied"]].rename(
        columns={"gold_id": "primary_label", "final_rank": "strict_primary_rank"}
    )


def compose_final_multilabel_ranks(
    final_strict: pd.DataFrame,
    hgnn_top50: pd.DataFrame,
    ddd_rerank: pd.DataFrame,
    mimic_similar: pd.DataFrame,
) -> pd.DataFrame:
    hgnn = hgnn_top50.set_index("case_id")
    ddd = ddd_rerank.set_index("case_id") if not ddd_rerank.empty else pd.DataFrame().set_index(pd.Index([]))
    mimic = mimic_similar.set_index("case_id") if not mimic_similar.empty else pd.DataFrame().set_index(pd.Index([]))
    rows: list[dict[str, Any]] = []
    for row in final_strict.to_dict(orient="records"):
        case_id = str(row["case_id"])
        dataset = str(row["dataset"])
        source = hgnn
        source_name = "hgnn_top50"
        if dataset == "DDD" and case_id in ddd.index:
            source = ddd
            source_name = "ddd_validation_selected_grid_rerank"
        elif "mimic" in dataset.lower() and case_id in mimic.index:
            source = mimic
            source_name = "similar_case_fixed_test"
        if case_id not in source.index:
            continue
        source_row = source.loc[case_id]
        rows.append(
            {
                "case_id": case_id,
                "dataset": dataset,
                "primary_label": row["primary_label"],
                "strict_primary_rank": int(row["strict_primary_rank"]),
                "any_label_rank": int(source_row["any_label_rank"]),
                "canonical_primary_rank": int(source_row["canonical_primary_rank"]),
                "canonical_any_label_rank": int(source_row["canonical_any_label_rank"]),
                "metric_source": source_name,
                "module_applied": row.get("module_applied", ""),
            }
        )
    return pd.DataFrame(rows)


def metric_rows_for_rank_table(rank_df: pd.DataFrame, rank_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank_cols = [
        ("strict_primary", "strict_primary_rank"),
        ("any_label", "any_label_rank"),
        ("canonical_primary", "canonical_primary_rank"),
        ("canonical_any_label", "canonical_any_label_rank"),
    ]
    for scoring_mode, rank_col in rank_cols:
        if rank_col not in rank_df.columns:
            continue
        for dataset, group in rank_df.groupby("dataset", sort=True):
            metrics = compute_rank_metrics(group[rank_col].tolist())
            rows.append({"rank_source": rank_source, "dataset": dataset, "scoring_mode": scoring_mode, **metrics})
        metrics = compute_rank_metrics(rank_df[rank_col].tolist())
        rows.append({"rank_source": rank_source, "dataset": "ALL", "scoring_mode": scoring_mode, **metrics})
    return rows


def build_obsolete_cases(
    labels: pd.DataFrame,
    hgnn_top50: pd.DataFrame,
    final_mixed: pd.DataFrame,
    canonicalizer: MondoCanonicalizer,
) -> pd.DataFrame:
    hgnn_lookup = hgnn_top50.set_index("case_id")
    final_lookup = final_mixed.set_index("case_id") if not final_mixed.empty else pd.DataFrame().set_index(pd.Index([]))
    rows: list[dict[str, Any]] = []
    for case in labels.to_dict(orient="records"):
        case_id = case["case_id"]
        hgnn_row = hgnn_lookup.loc[case_id] if case_id in hgnn_lookup.index else {}
        final_row = final_lookup.loc[case_id] if case_id in final_lookup.index else {}
        pred_top1 = hgnn_row.get("top1_candidate", "") if isinstance(hgnn_row, pd.Series) else ""
        if not case["any_label_obsolete"] and not canonicalizer.is_obsolete(pred_top1):
            continue
        rows.append(
            {
                "case_id": case_id,
                "dataset": case["dataset"],
                "primary_label": case["primary_label"],
                "all_labels": case["all_labels"],
                "obsolete_labels": case["obsolete_labels"],
                "canonical_all_labels": case["canonical_all_labels"],
                "hgnn_top1": pred_top1,
                "hgnn_top1_obsolete": bool(canonicalizer.is_obsolete(pred_top1)),
                "hgnn_strict_rank": int(hgnn_row.get("strict_primary_rank", MISSING_RANK)) if isinstance(hgnn_row, pd.Series) else MISSING_RANK,
                "hgnn_any_label_rank": int(hgnn_row.get("any_label_rank", MISSING_RANK)) if isinstance(hgnn_row, pd.Series) else MISSING_RANK,
                "final_strict_rank": int(final_row.get("strict_primary_rank", MISSING_RANK)) if isinstance(final_row, pd.Series) else MISSING_RANK,
                "final_any_label_rank": int(final_row.get("any_label_rank", MISSING_RANK)) if isinstance(final_row, pd.Series) else MISSING_RANK,
            }
        )
    return pd.DataFrame(rows)


def build_sample_cases(labels: pd.DataFrame, hgnn_top50: pd.DataFrame, final_mixed: pd.DataFrame) -> pd.DataFrame:
    merged = labels.merge(
        hgnn_top50[["case_id", "strict_primary_rank", "any_label_rank", "canonical_any_label_rank", "top1_candidate"]],
        on="case_id",
        how="left",
        suffixes=("", "_hgnn"),
    )
    merged = merged.merge(
        final_mixed[["case_id", "strict_primary_rank", "any_label_rank", "canonical_any_label_rank", "metric_source"]],
        on="case_id",
        how="left",
        suffixes=("_hgnn", "_final"),
    )
    changed = merged[
        (merged["is_multilabel"])
        & (
            (merged["any_label_rank_hgnn"] < merged["strict_primary_rank_hgnn"])
            | (merged["canonical_any_label_rank_hgnn"] < merged["strict_primary_rank_hgnn"])
        )
    ].copy()
    obsolete = merged[merged["any_label_obsolete"]].copy()
    sample = pd.concat([changed.head(25), obsolete.head(25)], ignore_index=True).drop_duplicates("case_id")
    cols = [
        "case_id",
        "dataset",
        "primary_label",
        "all_labels",
        "canonical_all_labels",
        "is_multilabel",
        "any_label_obsolete",
        "obsolete_labels",
        "hpo_count",
        "top1_candidate",
        "strict_primary_rank_hgnn",
        "any_label_rank_hgnn",
        "canonical_any_label_rank_hgnn",
        "strict_primary_rank_final",
        "any_label_rank_final",
        "canonical_any_label_rank_final",
        "metric_source",
    ]
    return sample[[col for col in cols if col in sample.columns]].head(50)


def status_summary(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "NOT_FOUND"
    parts = []
    for column in columns:
        if column in df.columns:
            counts = Counter(df[column].astype(str))
            parts.append(f"{column}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "; ".join(parts)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    shown = df[columns].head(limit).copy() if not df.empty else pd.DataFrame(columns=columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown.to_dict(orient="records"):
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            values.append(str(value).replace("\n", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > limit:
        lines.append("| " + " | ".join(["..."] * len(columns)) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    audit_time: str,
    run_dir: Path,
    locked_config: Path,
    data_config: Path,
    disease_index_path: Path,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    obsolete_cases: pd.DataFrame,
    sample_cases: pd.DataFrame,
    output_manifest: Path,
    table_paths: dict[str, Path],
) -> None:
    mimic_metrics = metrics[
        (metrics["dataset"] == "mimic_test_recleaned_mondo_hpo_rows")
        & (metrics["rank_source"] == "final_mixed")
    ]
    final_strict = mimic_metrics[mimic_metrics["scoring_mode"] == "strict_primary"]
    final_any = mimic_metrics[mimic_metrics["scoring_mode"] == "any_label"]
    strict_top5 = float(final_strict["top5"].iloc[0]) if not final_strict.empty else float("nan")
    any_top5 = float(final_any["top5"].iloc[0]) if not final_any.empty else float("nan")
    multi_count = int(labels["is_multilabel"].sum())
    mimic_multi = int(labels.loc[labels["dataset"].str.contains("mimic", case=False), "is_multilabel"].sum())
    obsolete_label_cases = int(labels["any_label_obsolete"].sum())
    mimic_obsolete = int(
        labels.loc[labels["dataset"].str.contains("mimic", case=False), "any_label_obsolete"].sum()
    )

    lines = [
        "# Evaluator Multilabel / Canonical MONDO Audit",
        "",
        f"Generated at: `{audit_time}`",
        "",
        "## Executive Summary",
        "",
        "- No training was run; this audit re-reads frozen evaluation artifacts and raw test labels.",
        f"- Frozen run: `{run_dir}`",
        f"- Locked config: `{locked_config}`",
        f"- Disease index: `{disease_index_path}`",
        f"- Multi-label cases: `{multi_count}` total, `{mimic_multi}` in mimic-like datasets.",
        f"- Obsolete-label cases: `{obsolete_label_cases}` total, `{mimic_obsolete}` in mimic-like datasets.",
        f"- mimic final top5 strict primary vs any-label: `{strict_top5:.4f}` -> `{any_top5:.4f}`.",
        "- Primary metric remains strict primary-label; any-label and canonical/obsolete-aware metrics are supplementary.",
        "",
        "## Inputs",
        "",
        f"- data config: `{data_config}`",
        f"- output manifest: `{output_manifest}`",
        f"- exact details: `{run_dir / 'stage3_exact_eval' / 'exact_details.csv'}`",
        f"- candidate metadata: `{run_dir / 'stage4_candidates' / 'top50_candidates_test.metadata.json'}`",
        f"- final case ranks: `{run_dir / 'mainline_final_case_ranks.csv'}`",
        "",
        "## Evaluator Code Changes",
        "",
        "- `src/evaluation/evaluator.py` now preserves `mondo_labels` for every case while keeping `mondo_label` as the strict primary label.",
        "- Future evaluator runs keep existing `true_rank` / top-level strict metrics and add `any_label_rank`, `canonical_primary_rank`, and `canonical_any_label_rank` in details.",
        "- `src/evaluation/mondo_canonicalizer.py` resolves MONDO alternative IDs and curated obsolete replacements for supplementary relaxed metrics.",
        "",
        "## Label Coverage",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "n_cases": len(labels),
                        "multi_label_cases": multi_count,
                        "obsolete_any_label_cases": obsolete_label_cases,
                        "primary_in_index_rate": labels["primary_in_disease_index"].mean(),
                        "any_label_in_index_rate": labels["any_label_in_disease_index"].mean(),
                    }
                ]
            ),
            ["n_cases", "multi_label_cases", "obsolete_any_label_cases", "primary_in_index_rate", "any_label_in_index_rate"],
        ),
        "",
        "## Metric Comparison",
        "",
        f"Full table: `{table_paths['metric_comparison']}`",
        "",
        markdown_table(
            metrics[
                metrics["dataset"].isin(["DDD", "mimic_test_recleaned_mondo_hpo_rows", "ALL"])
            ],
            ["rank_source", "dataset", "scoring_mode", "n", "top1", "top3", "top5", "top10", "top30", "rank_le_50", "median_rank"],
            limit=60,
        ),
        "",
        "## Obsolete MONDO Cases",
        "",
        f"Full table: `{table_paths['obsolete_cases']}`",
        "",
        markdown_table(
            obsolete_cases,
            ["case_id", "dataset", "primary_label", "obsolete_labels", "hgnn_top1", "hgnn_any_label_rank", "final_any_label_rank"],
            limit=25,
        ),
        "",
        "## Sample Cases",
        "",
        f"Full table: `{table_paths['sample_cases']}`",
        "",
        markdown_table(
            sample_cases,
            ["case_id", "dataset", "primary_label", "all_labels", "top1_candidate", "strict_primary_rank_hgnn", "any_label_rank_hgnn", "metric_source"],
            limit=25,
        ),
        "",
        "## Generated Files",
        "",
        *[f"- {name}: `{path}`" for name, path in table_paths.items()],
        f"- manifest: `{output_manifest}`",
        "",
        "## Commands Run",
        "",
        "- `D:\\python\\python.exe -m compileall src\\evaluation\\evaluator.py src\\evaluation\\mondo_canonicalizer.py src\\evaluation\\multilabel_metrics.py tools\\audit_evaluator_multilabel_canonical.py`",
        "- `D:\\python\\python.exe tools\\audit_evaluator_multilabel_canonical.py --run-dir outputs\\mainline_full_pipeline_hybrid_tag_v5 --locked-config configs\\mainline_full_pipeline_hybrid_tag_v5.locked.yaml --data-config configs\\data_llldataset_eval.yaml --out-report-dir reports\\evaluator_multilabel_canonical_audit --out-output-dir outputs\\evaluator_multilabel_canonical_audit`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_time = datetime.now().isoformat(timespec="seconds")
    stamp = now_stamp()
    run_dir = resolve_path(args.run_dir)
    locked_config = resolve_path(args.locked_config)
    data_config = resolve_path(args.data_config)
    report_dir = prepare_target_dir(args.out_report_dir, stamp)
    output_dir = prepare_target_dir(args.out_output_dir, stamp)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    disease_ids, disease_index_path = load_disease_index_from_locked_config(locked_config)
    canonicalizer = MondoCanonicalizer.load(args.mondo_json, args.obsolete_mondo_csv)
    labels = load_case_labels(data_config, disease_ids, canonicalizer)
    exact_strict = load_exact_strict_ranks(run_dir)
    hgnn_top50 = load_hgnn_top50_ranks(run_dir, labels, canonicalizer)
    ddd_rerank = load_ddd_rerank_ranks(run_dir, labels, canonicalizer)
    mimic_similar = load_mimic_similarcase_ranks(run_dir, labels, canonicalizer)
    final_strict = load_final_strict_ranks(run_dir)
    final_mixed = compose_final_multilabel_ranks(final_strict, hgnn_top50, ddd_rerank, mimic_similar)

    metric_rows: list[dict[str, Any]] = []
    metric_rows.extend(metric_rows_for_rank_table(exact_strict, "exact_full_strict_primary"))
    metric_rows.extend(metric_rows_for_rank_table(hgnn_top50, "hgnn_top50_candidates"))
    if not ddd_rerank.empty:
        metric_rows.extend(metric_rows_for_rank_table(ddd_rerank, "ddd_rerank_top50"))
    if not mimic_similar.empty:
        metric_rows.extend(metric_rows_for_rank_table(mimic_similar, "mimic_similarcase_top50"))
    metric_rows.extend(metric_rows_for_rank_table(final_mixed, "final_mixed"))
    metrics = pd.DataFrame(metric_rows)

    obsolete_cases = build_obsolete_cases(labels, hgnn_top50, final_mixed, canonicalizer)
    sample_cases = build_sample_cases(labels, hgnn_top50, final_mixed)

    table_paths = {
        "case_label_audit": tables_dir / "case_label_audit.csv",
        "metric_comparison": tables_dir / "metric_comparison.csv",
        "hgnn_top50_multilabel_ranks": tables_dir / "hgnn_top50_multilabel_ranks.csv",
        "final_mixed_multilabel_ranks": tables_dir / "final_mixed_multilabel_ranks.csv",
        "obsolete_cases": tables_dir / "obsolete_mondo_cases.csv",
        "sample_cases": tables_dir / "sample_cases.csv",
    }
    write_csv(labels, table_paths["case_label_audit"])
    write_csv(metrics, table_paths["metric_comparison"])
    write_csv(hgnn_top50, table_paths["hgnn_top50_multilabel_ranks"])
    write_csv(final_mixed, table_paths["final_mixed_multilabel_ranks"])
    write_csv(obsolete_cases, table_paths["obsolete_cases"])
    write_csv(sample_cases, table_paths["sample_cases"])

    manifest = {
        "audit_time": audit_time,
        "run_dir": str(run_dir),
        "locked_config": str(locked_config),
        "data_config": str(data_config),
        "disease_index_path": str(disease_index_path),
        "canonicalizer_sources": canonicalizer.source_paths,
        "inputs": {
            "exact_details": str(run_dir / "stage3_exact_eval" / "exact_details.csv"),
            "top50_candidates_test": str(run_dir / "stage4_candidates" / "top50_candidates_test.csv"),
            "ddd_weights": str(run_dir / "stage5_ddd_rerank" / "ddd_val_selected_grid_weights.json"),
            "mimic_ranked_candidates": str(run_dir / "stage6_mimic_similar_case" / "similar_case_fixed_test_ranked_candidates.csv"),
            "final_case_ranks": str(run_dir / "mainline_final_case_ranks.csv"),
        },
        "outputs": {key: str(path) for key, path in table_paths.items()},
        "label_summary": {
            "n_cases": int(len(labels)),
            "multi_label_cases": int(labels["is_multilabel"].sum()),
            "obsolete_any_label_cases": int(labels["any_label_obsolete"].sum()),
            "mimic_multi_label_cases": int(labels.loc[labels["dataset"].str.contains("mimic", case=False), "is_multilabel"].sum()),
            "mimic_obsolete_any_label_cases": int(labels.loc[labels["dataset"].str.contains("mimic", case=False), "any_label_obsolete"].sum()),
        },
        "metric_status": status_summary(metrics, ["rank_source", "scoring_mode"]),
    }
    manifest_path = output_dir / "evaluator_multilabel_canonical_audit_manifest.json"
    write_json(manifest, manifest_path)
    report_path = report_dir / "evaluator_multilabel_canonical_audit.md"
    write_report(
        report_path,
        audit_time=audit_time,
        run_dir=run_dir,
        locked_config=locked_config,
        data_config=data_config,
        disease_index_path=disease_index_path,
        labels=labels,
        metrics=metrics,
        obsolete_cases=obsolete_cases,
        sample_cases=sample_cases,
        output_manifest=manifest_path,
        table_paths=table_paths,
    )
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "manifest_path": str(manifest_path),
                "n_cases": int(len(labels)),
                "multi_label_cases": int(labels["is_multilabel"].sum()),
                "obsolete_any_label_cases": int(labels["any_label_obsolete"].sum()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
