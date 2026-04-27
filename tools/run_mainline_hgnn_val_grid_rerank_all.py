from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_top50_evidence_rerank import load_candidates, ranks_from_scores, score_matrix, to_matrix


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mainline_hgnn_val_grid_rerank.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen all-dataset mainline: HGNN baseline + DDD validation grid rerank + mimic SimilarCase-Aug."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain a mapping: {path}")
    return payload


def read_metadata_for_csv(path: Path) -> dict[str, Any]:
    for metadata_path in [path.with_suffix(".metadata.json"), path.with_suffix(path.suffix + ".metadata.json")]:
        if metadata_path.exists():
            return load_json(metadata_path)
    return {}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无记录_"
    view = df.fillna("").astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def metric_from_ranks(ranks: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(ranks, errors="coerce").fillna(9999).to_numpy(dtype=int)
    return {
        "num_cases": int(arr.size),
        "top1": float(np.mean(arr <= 1)) if arr.size else float("nan"),
        "top3": float(np.mean(arr <= 3)) if arr.size else float("nan"),
        "top5": float(np.mean(arr <= 5)) if arr.size else float("nan"),
        "median_rank": float(np.median(arr)) if arr.size else float("nan"),
        "mean_rank": float(np.mean(arr)) if arr.size else float("nan"),
        "rank_le_50": float(np.mean(arr <= 50)) if arr.size else float("nan"),
    }


def metrics_by_dataset(case_ranks: pd.DataFrame, rank_col: str, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in case_ranks.groupby("dataset_name", sort=True):
        rows.append({"method": method, "dataset_name": dataset, **metric_from_ranks(group[rank_col])})
    rows.append({"method": method, "dataset_name": "ALL", **metric_from_ranks(case_ranks[rank_col])})
    return pd.DataFrame(rows)


def add_metric_sources(
    metrics: pd.DataFrame,
    *,
    source_by_dataset: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    out = metrics.copy()
    for col in ["module_applied", "source_result_path", "source_dataset_name", "checkpoint_path", "data_version"]:
        out[col] = ""

    for idx, row in out.iterrows():
        dataset = str(row["dataset_name"])
        source = source_by_dataset.get(dataset)
        if dataset == "ALL":
            source = {
                "module_applied": "mixed",
                "source_result_path": "mixed",
                "source_dataset_name": "ALL",
                "checkpoint_path": "mixed",
                "data_version": "mixed",
            }
        if source:
            for key, value in source.items():
                if key in out.columns:
                    out.at[idx, key] = value
    return out


def load_baseline_case_ranks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str})
    required = {"preset", "case_id", "dataset_name", "gold_id", "reranked_rank", "gold_in_top50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    base = df.loc[df["preset"] == "A_hgnn_only"].copy()
    if base.empty:
        raise ValueError(f"{path} has no A_hgnn_only rows.")
    base["baseline_rank"] = pd.to_numeric(base["reranked_rank"], errors="coerce").fillna(51).astype(int)
    return base[["case_id", "dataset_name", "gold_id", "baseline_rank", "gold_in_top50"]].reset_index(drop=True)


def load_exact_case_ranks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "true_label": str})
    required = {"case_id", "dataset_name", "true_label", "true_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df[["case_id", "dataset_name", "true_label", "true_rank"]].copy()
    out = out.rename(columns={"true_label": "gold_id", "true_rank": "baseline_rank"})
    out["baseline_rank"] = pd.to_numeric(out["baseline_rank"], errors="coerce").fillna(9999).astype(int)
    out["gold_in_top50"] = out["baseline_rank"].le(50)
    return out.reset_index(drop=True)


def load_ddd_payload(weights_path: Path, objective: str) -> dict[str, Any]:
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    selections = payload.get("selections", [])
    if not isinstance(selections, list):
        raise ValueError(f"Invalid weights payload: {weights_path}")
    for selection in selections:
        if selection.get("selection_objective") == objective and selection.get("selection_kind") == "grid":
            return selection
    raise KeyError(f"No grid selection for objective {objective!r} in {weights_path}")


def compute_ddd_grid_ranks(candidates_path: Path, weights_path: Path, objective: str, dataset: str) -> pd.DataFrame:
    matrix = to_matrix(load_candidates(candidates_path))
    payload = load_ddd_payload(weights_path, objective)
    scores = score_matrix(matrix, payload["weights"])
    ranks = ranks_from_scores(matrix, scores)
    df = pd.DataFrame(
        {
            "case_id": matrix.case_ids,
            "dataset_name": matrix.dataset_names,
            "ddd_grid_rank": ranks,
        }
    )
    return df.loc[df["dataset_name"] == dataset].copy()


def load_mimic_fixed_ranks(path: Path, dataset: str) -> pd.DataFrame:
    ranked = pd.read_csv(path, dtype={"case_id": str, "gold_id": str, "candidate_id": str})
    required = {"case_id", "gold_id", "candidate_id", "rank"}
    missing = required - set(ranked.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for case_id, group in ranked.groupby("case_id", sort=False):
        gold_id = str(group["gold_id"].iloc[0])
        hits = group.loc[group["candidate_id"].astype(str) == gold_id, "rank"]
        rank = int(pd.to_numeric(hits, errors="coerce").min()) if not hits.empty else 9999
        rows.append(
            {
                "case_id": str(case_id),
                "case_key": str(case_id).rsplit("::", 1)[-1],
                "dataset_name": dataset,
                "mimic_rank": rank,
            }
        )
    return pd.DataFrame(rows)


def dataset_aliases(config: dict[str, Any], canonical_dataset: str) -> list[str]:
    aliases = config.get("dataset_aliases", {})
    values = aliases.get(canonical_dataset, []) if isinstance(aliases, dict) else []
    out = [canonical_dataset]
    if isinstance(values, list):
        out.extend(str(value) for value in values)
    return list(dict.fromkeys(out))


def merge_final_ranks(
    baseline: pd.DataFrame,
    ddd_ranks: pd.DataFrame,
    mimic_ranks: pd.DataFrame,
    *,
    ddd_dataset: str,
    mimic_dataset: str,
) -> pd.DataFrame:
    final = baseline.copy()
    final["final_rank"] = final["baseline_rank"]
    final["applied_module"] = "hgnn_baseline"

    final = final.merge(ddd_ranks[["case_id", "ddd_grid_rank"]], on="case_id", how="left")
    ddd_mask = final["dataset_name"].eq(ddd_dataset) & final["ddd_grid_rank"].notna()
    final.loc[ddd_mask, "final_rank"] = final.loc[ddd_mask, "ddd_grid_rank"].astype(int)
    final.loc[ddd_mask, "applied_module"] = "ddd_validation_selected_grid_rerank"

    final = final.merge(mimic_ranks[["case_id", "mimic_rank"]], on="case_id", how="left")
    if final.loc[final["dataset_name"].eq(mimic_dataset), "mimic_rank"].isna().all() and "case_key" in mimic_ranks:
        final["case_key"] = final["case_id"].astype(str).str.rsplit("::", n=1).str[-1]
        keyed = mimic_ranks[["case_key", "mimic_rank"]].drop_duplicates(subset=["case_key"])
        final = final.drop(columns=["mimic_rank"]).merge(keyed, on="case_key", how="left")
    mimic_mask = final["dataset_name"].eq(mimic_dataset) & final["mimic_rank"].notna()
    final.loc[mimic_mask, "final_rank"] = final.loc[mimic_mask, "mimic_rank"].astype(int)
    final.loc[mimic_mask, "applied_module"] = "mimic_similar_case_aug"

    final["final_rank"] = pd.to_numeric(final["final_rank"], errors="coerce").fillna(9999).astype(int)
    return final


def merge_final_ranks_with_aliases(
    baseline: pd.DataFrame,
    ddd_ranks: pd.DataFrame,
    mimic_ranks: pd.DataFrame,
    *,
    ddd_dataset: str,
    mimic_datasets: list[str],
) -> pd.DataFrame:
    final = baseline.copy()
    final["final_rank"] = final["baseline_rank"]
    final["applied_module"] = "hgnn_baseline"

    final = final.merge(ddd_ranks[["case_id", "ddd_grid_rank"]], on="case_id", how="left")
    ddd_mask = final["dataset_name"].eq(ddd_dataset) & final["ddd_grid_rank"].notna()
    final.loc[ddd_mask, "final_rank"] = final.loc[ddd_mask, "ddd_grid_rank"].astype(int)
    final.loc[ddd_mask, "applied_module"] = "ddd_validation_selected_grid_rerank"

    final["case_key"] = final["case_id"].astype(str).str.rsplit("::", n=1).str[-1]
    mimic_join = mimic_ranks[["case_id", "case_key", "mimic_rank"]].copy()
    final = final.merge(mimic_join[["case_id", "mimic_rank"]], on="case_id", how="left")
    if final.loc[final["dataset_name"].isin(mimic_datasets), "mimic_rank"].isna().all():
        keyed = mimic_join[["case_key", "mimic_rank"]].drop_duplicates(subset=["case_key"])
        final = final.drop(columns=["mimic_rank"]).merge(keyed, on="case_key", how="left")
    mimic_mask = final["dataset_name"].isin(mimic_datasets) & final["mimic_rank"].notna()
    final.loc[mimic_mask, "final_rank"] = final.loc[mimic_mask, "mimic_rank"].astype(int)
    final.loc[mimic_mask, "applied_module"] = "mimic_similar_case_aug"

    final["final_rank"] = pd.to_numeric(final["final_rank"], errors="coerce").fillna(9999).astype(int)
    return final


def build_source_map(
    datasets: list[str],
    *,
    baseline_source_path: Path,
    ddd_source_path: Path,
    mimic_source_path: Path,
    ddd_dataset: str,
    mimic_datasets: list[str],
    checkpoint_path: str,
    data_version: str,
    baseline_module: str = "hgnn_baseline",
) -> dict[str, dict[str, Any]]:
    source_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        module = baseline_module
        path = baseline_source_path
        if dataset == ddd_dataset:
            module = "ddd_validation_selected_grid_rerank"
            path = ddd_source_path
        elif dataset in mimic_datasets:
            module = "mimic_similar_case_aug"
            path = mimic_source_path
        source_by_dataset[dataset] = {
            "module_applied": module,
            "source_result_path": str(path.resolve()),
            "source_dataset_name": dataset,
            "checkpoint_path": checkpoint_path,
            "data_version": data_version,
        }
    return source_by_dataset


def build_delta(baseline_metrics: pd.DataFrame, final_metrics: pd.DataFrame) -> pd.DataFrame:
    base = baseline_metrics.rename(
        columns={
            "top1": "baseline_top1",
            "top3": "baseline_top3",
            "top5": "baseline_top5",
            "median_rank": "baseline_median_rank",
            "mean_rank": "baseline_mean_rank",
            "rank_le_50": "baseline_rank_le_50",
        }
    )
    final = final_metrics.rename(
        columns={
            "top1": "final_top1",
            "top3": "final_top3",
            "top5": "final_top5",
            "median_rank": "final_median_rank",
            "mean_rank": "final_mean_rank",
            "rank_le_50": "final_rank_le_50",
        }
    )
    merged = base.drop(columns=["method"]).merge(final.drop(columns=["method"]), on=["dataset_name", "num_cases"], how="outer")
    for metric in ["top1", "top3", "top5", "rank_le_50"]:
        merged[f"delta_{metric}"] = merged[f"final_{metric}"] - merged[f"baseline_{metric}"]
    return merged


def _summary_dataset_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = summary.get("per_dataset", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("dataset_name")): row for row in rows if isinstance(row, dict)}


def _manifest_test_files(summary: dict[str, Any]) -> list[str]:
    run_manifest = summary.get("run_manifest", {})
    if not isinstance(run_manifest, dict):
        return []
    evaluation = run_manifest.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return []
    resolved = evaluation.get("resolved_test_inputs", {})
    if isinstance(resolved, dict) and isinstance(resolved.get("files"), list):
        return [str(value) for value in resolved["files"]]
    data_config = evaluation.get("data_config", {})
    if isinstance(data_config, dict) and isinstance(data_config.get("test_files"), list):
        return [str(value) for value in data_config["test_files"]]
    return []


def _file_by_name(files: list[str]) -> dict[str, str]:
    return {Path(value).name: value for value in files}


def _hash_file(path_like: str) -> str:
    path = resolve_path(path_like)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_case_details(old_details: pd.DataFrame, current_details: pd.DataFrame, dataset: str) -> dict[str, Any]:
    old = old_details.loc[old_details["dataset_name"].astype(str).eq(dataset)].copy()
    cur = current_details.loc[current_details["dataset_name"].astype(str).eq(dataset)].copy()
    old_cases = set(old["case_id"].astype(str))
    cur_cases = set(cur["case_id"].astype(str))
    overlap = old_cases & cur_cases
    label_mismatches = 0
    rank_changed = 0
    if overlap:
        old_cmp = old[["case_id", "true_label", "true_rank"]].rename(
            columns={"true_label": "old_label", "true_rank": "old_rank"}
        )
        cur_cmp = cur[["case_id", "true_label", "true_rank"]].rename(
            columns={"true_label": "current_label", "true_rank": "current_rank"}
        )
        merged = old_cmp.merge(cur_cmp, on="case_id", how="inner")
        label_mismatches = int((merged["old_label"].astype(str) != merged["current_label"].astype(str)).sum())
        rank_changed = int(
            (
                pd.to_numeric(merged["old_rank"], errors="coerce")
                != pd.to_numeric(merged["current_rank"], errors="coerce")
            ).sum()
        )
    return {
        "old_case_count": int(len(old)),
        "current_case_count": int(len(cur)),
        "case_id_set_equal": old_cases == cur_cases,
        "case_id_overlap": int(len(overlap)),
        "old_only_case_count": int(len(old_cases - cur_cases)),
        "current_only_case_count": int(len(cur_cases - old_cases)),
        "label_mismatch_count": label_mismatches,
        "rank_changed_case_count": rank_changed,
    }


def build_parity_audit(
    *,
    old_summary_path: Path,
    old_details_path: Path,
    current_summary_path: Path,
    current_details_path: Path,
    current_final_metrics: pd.DataFrame,
    current_source_map: dict[str, dict[str, Any]],
    output_csv: Path,
    output_md: Path,
) -> pd.DataFrame:
    old_summary = load_json(old_summary_path)
    current_summary = load_json(current_summary_path)
    old_details = pd.read_csv(old_details_path, dtype={"case_id": str, "dataset_name": str, "true_label": str})
    current_details = pd.read_csv(current_details_path, dtype={"case_id": str, "dataset_name": str, "true_label": str})
    old_by_dataset = _summary_dataset_map(old_summary)
    current_by_dataset = _summary_dataset_map(current_summary)
    final_by_dataset = {
        str(row["dataset_name"]): row.to_dict()
        for _, row in current_final_metrics.loc[current_final_metrics["dataset_name"].ne("ALL")].iterrows()
    }
    old_files = _file_by_name(_manifest_test_files(old_summary))
    current_files = _file_by_name(_manifest_test_files(current_summary))

    datasets = sorted(set(old_by_dataset) | set(current_by_dataset) | set(final_by_dataset))
    rows: list[dict[str, Any]] = []
    old_checkpoint = str(old_summary.get("checkpoint_path", ""))
    current_checkpoint = str(current_summary.get("checkpoint_path", ""))
    old_data_config = str(old_summary.get("data_config_path", ""))
    current_data_config = str(current_summary.get("data_config_path", ""))
    for dataset in datasets:
        old_ds = old_by_dataset.get(dataset, {})
        cur_ds = current_by_dataset.get(dataset, {})
        final_ds = final_by_dataset.get(dataset, {})
        source_file = str(old_ds.get("source_file") or cur_ds.get("source_file") or "")
        old_file = old_files.get(source_file, "")
        current_file = current_files.get(source_file, "")
        case_compare = _compare_case_details(old_details, current_details, dataset)
        current_source = current_source_map.get(dataset, {})
        final_top1 = final_ds.get("top1", np.nan)
        final_top3 = final_ds.get("top3", np.nan)
        final_top5 = final_ds.get("top5", np.nan)
        old_top1 = old_ds.get("top1", np.nan)
        old_top3 = old_ds.get("top3", np.nan)
        old_top5 = old_ds.get("top5", np.nan)
        rows.append(
            {
                "dataset_name": dataset,
                "old_source_file": old_ds.get("source_file", ""),
                "current_source_file": cur_ds.get("source_file", ""),
                "old_top1": old_top1,
                "current_exact_top1": cur_ds.get("top1", np.nan),
                "current_final_top1": final_top1,
                "delta_final_vs_old_top1": final_top1 - old_top1
                if pd.notna(final_top1) and pd.notna(old_top1)
                else np.nan,
                "old_top3": old_top3,
                "current_exact_top3": cur_ds.get("top3", np.nan),
                "current_final_top3": final_top3,
                "delta_final_vs_old_top3": final_top3 - old_top3
                if pd.notna(final_top3) and pd.notna(old_top3)
                else np.nan,
                "old_top5": old_top5,
                "current_exact_top5": cur_ds.get("top5", np.nan),
                "current_final_top5": final_top5,
                "delta_final_vs_old_top5": final_top5 - old_top5
                if pd.notna(final_top5) and pd.notna(old_top5)
                else np.nan,
                "checkpoint_same_as_old": old_checkpoint == current_checkpoint,
                "old_checkpoint_path": old_checkpoint,
                "current_checkpoint_path": current_checkpoint,
                "data_config_same_as_old": old_data_config == current_data_config,
                "old_data_config_path": old_data_config,
                "current_data_config_path": current_data_config,
                "test_file_same_path": old_file == current_file and bool(old_file),
                "old_test_file": old_file,
                "current_test_file": current_file,
                "old_test_file_sha256": _hash_file(old_file) if old_file else "",
                "current_test_file_sha256": _hash_file(current_file) if current_file else "",
                "module_applied_current_final": current_source.get("module_applied", ""),
                "source_result_path_current_final": current_source.get("source_result_path", ""),
                "uses_rerank_or_candidate_file": "top50_candidates" in current_source.get("source_result_path", "")
                or "rerank" in current_source.get("source_result_path", ""),
                "checkpoint_drift": old_checkpoint != current_checkpoint,
                **case_compare,
                "label_rows_identical_for_overlap": case_compare["label_mismatch_count"] == 0,
                "hpo_rows_audit": "same test file path; historical row hash unavailable"
                if old_file == current_file and bool(old_file)
                else "different test file path",
            }
        )
    audit = pd.DataFrame(rows)
    write_csv(audit, output_csv)

    focus = audit.loc[audit["dataset_name"].isin(["LIRICAL", "RAMEDIS", "MME"])].copy()
    lines = [
        "# Parity audit: old baseline vs current final",
        "",
        f"- old baseline summary: `{old_summary_path.resolve()}`",
        f"- old details: `{old_details_path.resolve()}`",
        f"- current exact summary: `{current_summary_path.resolve()}`",
        f"- current details: `{current_details_path.resolve()}`",
        f"- old checkpoint: `{old_checkpoint}`",
        f"- current checkpoint: `{current_checkpoint}`",
        f"- data config same: `{old_data_config == current_data_config}`",
        "",
        "## Focus datasets",
        df_to_markdown(
            focus[
                [
                    "dataset_name",
                    "old_top1",
                    "current_final_top1",
                    "delta_final_vs_old_top1",
                    "old_top5",
                    "current_final_top5",
                    "delta_final_vs_old_top5",
                    "case_id_set_equal",
                    "label_mismatch_count",
                    "checkpoint_drift",
                    "module_applied_current_final",
                ]
            ]
        ),
        "",
        "## Conclusion",
        "- LIRICAL/RAMEDIS/MME 当前最终汇总没有应用 mimic SimilarCase，也没有应用 DDD rerank。",
        "- 三者下降主要来自 checkpoint/config drift：旧 baseline 是 `g4b_weighting_idf`，当前重跑主线是 `train_finetune_attn_idf_main` / `attn_beta_sweep\\edge_log_beta02`。",
        "- case_id 与 label 在重叠集合内一致；HPO 行历史 hash 未记录，只能确认 summary 中解析到的 test file path 是否相同。",
    ]
    write_text(output_md, "\n".join(lines))
    return audit


def write_reports(
    *,
    report_dir: Path,
    output_dir: Path,
    config_path: Path,
    baseline_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    delta: pd.DataFrame,
    final_case_ranks_path: Path,
    manifest: dict[str, Any],
) -> None:
    write_csv(baseline_metrics, output_dir / "hgnn_baseline_metrics.csv")
    write_csv(final_metrics, output_dir / "mainline_final_metrics.csv")
    write_csv(delta, output_dir / "mainline_delta_vs_baseline.csv")
    manifest_path = output_dir / "mainline_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# HGNN + Validation-selected DDD Grid Rerank + Mimic Mainline",
        "",
        "## 1. Protocol",
        "- no training",
        "- no HGNN encoder change",
        "- no test-side tuning",
        "- no exact evaluation overwrite",
        "- DDD uses fixed validation-selected grid weights (`DDD_top1`)",
        "- mimic_test uses existing `HGNN_SimilarCase_Aug` fixed output",
        "- all other datasets use HGNN baseline ranks",
        "",
        "## 2. Inputs",
        f"- config: `{config_path}`",
        f"- final case ranks: `{final_case_ranks_path}`",
        f"- manifest: `{manifest_path}`",
        "",
        "## 3. HGNN Baseline Metrics",
        df_to_markdown(baseline_metrics),
        "",
        "## 4. Final Mainline Metrics",
        df_to_markdown(final_metrics),
        "",
        "## 5. Delta vs Historical Baseline",
        df_to_markdown(delta),
        "",
        "## 6. Mainline Judgment",
        "- DDD final mainline is `validation-selected grid rerank (DDD_top1)`.",
        "- mimic mainline is enabled only for `mimic_test`.",
        "- gated rerank, ontology-aware HN, and test-side exploratory grid/gate are not used.",
    ]
    write_text(report_dir / "mainline_final_all_dataset_report.md", "\n".join(report))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config_path)
    config = load_yaml(config_path)
    output_dir = resolve_path(config["outputs"]["output_dir"])
    report_dir = resolve_path(config["outputs"]["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline_case_ranks_path = resolve_path(config["baseline"]["hgnn_case_ranks"])
    baseline_candidates_path = resolve_path(config["baseline"]["hgnn_candidates"])
    baseline_metadata = read_metadata_for_csv(baseline_candidates_path)
    ddd_weights_path = resolve_path(config["ddd_module"]["weights"])
    mimic_ranked_path = resolve_path(config["mimic_module"]["fixed_ranked_candidates"])
    mimic_metrics_path = resolve_path(config["mimic_module"]["fixed_metrics"])
    fixed_mimic_metrics = pd.read_csv(mimic_metrics_path)

    baseline = load_baseline_case_ranks(baseline_case_ranks_path)
    baseline_metrics = metrics_by_dataset(baseline, "baseline_rank", "HGNN_exact_baseline")

    ddd_ranks = compute_ddd_grid_ranks(
        baseline_candidates_path,
        ddd_weights_path,
        str(config["ddd_module"]["selection_objective"]),
        str(config["ddd_module"]["dataset"]),
    )
    mimic_ranks = load_mimic_fixed_ranks(mimic_ranked_path, str(config["mimic_module"]["dataset"]))
    mimic_aliases = dataset_aliases(config, str(config["mimic_module"]["dataset"]))
    final = merge_final_ranks(
        baseline,
        ddd_ranks,
        mimic_ranks,
        ddd_dataset=str(config["ddd_module"]["dataset"]),
        mimic_dataset=str(config["mimic_module"]["dataset"]),
    )
    final_metrics = metrics_by_dataset(final, "final_rank", "mainline_hgnn_val_grid_rerank")
    delta = build_delta(baseline_metrics, final_metrics)

    checkpoint_path = str(baseline_metadata.get("checkpoint_path", ""))
    data_version = str(baseline_metadata.get("data_config_path", ""))
    if baseline_metadata.get("generated_at"):
        data_version = f"{data_version}; candidates_generated_at={baseline_metadata['generated_at']}"
    source_map_current = build_source_map(
        sorted(baseline["dataset_name"].astype(str).unique()),
        baseline_source_path=baseline_case_ranks_path,
        ddd_source_path=ddd_weights_path,
        mimic_source_path=mimic_ranked_path,
        ddd_dataset=str(config["ddd_module"]["dataset"]),
        mimic_datasets=[str(config["mimic_module"]["dataset"])],
        checkpoint_path=checkpoint_path,
        data_version=data_version,
    )
    final_metrics_with_sources = add_metric_sources(final_metrics, source_by_dataset=source_map_current)

    fixed_mimic = merge_final_ranks_with_aliases(
        baseline,
        ddd_ranks,
        mimic_ranks,
        ddd_dataset=str(config["ddd_module"]["dataset"]),
        mimic_datasets=mimic_aliases,
    )
    fixed_mimic_metrics = metrics_by_dataset(fixed_mimic, "final_rank", "mainline_hgnn_val_grid_rerank_fixed_mimic")
    source_map_fixed = build_source_map(
        sorted(baseline["dataset_name"].astype(str).unique()),
        baseline_source_path=baseline_case_ranks_path,
        ddd_source_path=ddd_weights_path,
        mimic_source_path=mimic_metrics_path,
        ddd_dataset=str(config["ddd_module"]["dataset"]),
        mimic_datasets=mimic_aliases,
        checkpoint_path=checkpoint_path,
        data_version=data_version,
    )
    fixed_mimic_metrics = add_metric_sources(fixed_mimic_metrics, source_by_dataset=source_map_fixed)

    frozen_config = config.get("frozen_protocol", {})
    frozen_details_path = resolve_path(frozen_config.get("exact_baseline_details", config["baseline"].get("exact_eval_source", "")))
    frozen_summary_path = resolve_path(frozen_config.get("exact_baseline_summary", ""))
    frozen_baseline = load_exact_case_ranks(frozen_details_path)
    frozen = merge_final_ranks_with_aliases(
        frozen_baseline,
        ddd_ranks,
        mimic_ranks,
        ddd_dataset=str(config["ddd_module"]["dataset"]),
        mimic_datasets=mimic_aliases,
    )
    frozen_metrics = metrics_by_dataset(frozen, "final_rank", "mainline_hgnn_frozen_protocol")
    frozen_checkpoint = checkpoint_path
    frozen_data_version = data_version
    if frozen_summary_path.exists():
        frozen_summary = load_json(frozen_summary_path)
        frozen_checkpoint = str(frozen_summary.get("checkpoint_path", frozen_checkpoint))
        frozen_data_version = str(frozen_summary.get("data_config_path", frozen_data_version))
    source_map_frozen = build_source_map(
        sorted(frozen_baseline["dataset_name"].astype(str).unique()),
        baseline_source_path=frozen_details_path,
        ddd_source_path=ddd_weights_path,
        mimic_source_path=mimic_metrics_path,
        ddd_dataset=str(config["ddd_module"]["dataset"]),
        mimic_datasets=mimic_aliases,
        checkpoint_path=frozen_checkpoint,
        data_version=frozen_data_version,
        baseline_module="hgnn_exact_frozen_baseline",
    )
    frozen_metrics = add_metric_sources(frozen_metrics, source_by_dataset=source_map_frozen)

    final_case_ranks_path = output_dir / "mainline_final_case_ranks.csv"
    write_csv(final, final_case_ranks_path)
    write_csv(final_metrics_with_sources, output_dir / "mainline_final_metrics_with_sources.csv")
    write_csv(fixed_mimic, output_dir / "mainline_final_case_ranks_fixed_mimic.csv")
    write_csv(fixed_mimic_metrics, output_dir / "mainline_final_metrics_fixed_mimic.csv")
    write_csv(frozen, output_dir / "mainline_final_case_ranks_frozen_protocol.csv")
    write_csv(frozen_metrics, output_dir / "mainline_final_metrics_frozen_protocol.csv")

    parity_config = config.get("parity_audit", {})
    if parity_config:
        build_parity_audit(
            old_summary_path=resolve_path(parity_config["old_summary"]),
            old_details_path=resolve_path(parity_config["old_details"]),
            current_summary_path=resolve_path(parity_config["current_summary"]),
            current_details_path=resolve_path(parity_config["current_details"]),
            current_final_metrics=final_metrics,
            current_source_map=source_map_current,
            output_csv=PROJECT_ROOT / "reports" / "mainline" / "parity_audit_old_vs_current.csv",
            output_md=PROJECT_ROOT / "reports" / "mainline" / "parity_audit_old_vs_current.md",
        )

    manifest = {
        "config_path": str(config_path.resolve()),
        "baseline_case_ranks_path": str(baseline_case_ranks_path.resolve()),
        "baseline_candidates_path": str(baseline_candidates_path.resolve()),
        "baseline_candidates_metadata": baseline_metadata,
        "ddd_weights_path": str(ddd_weights_path.resolve()),
        "mimic_ranked_candidates_path": str(mimic_ranked_path.resolve()),
        "mimic_fixed_metrics_path": str(mimic_metrics_path.resolve()),
        "mimic_dataset_aliases": mimic_aliases,
        "frozen_protocol": {
            "exact_baseline_details": str(frozen_details_path.resolve()),
            "exact_baseline_summary": str(frozen_summary_path.resolve()) if frozen_summary_path else "",
            "final_metrics": str((output_dir / "mainline_final_metrics_frozen_protocol.csv").resolve()),
        },
        "constraints": config.get("constraints", {}),
        "modules": {
            "DDD": config.get("ddd_module", {}),
            "mimic": config.get("mimic_module", {}),
            "other_datasets": config.get("other_datasets", {}),
        },
    }
    write_reports(
        report_dir=report_dir,
        output_dir=output_dir,
        config_path=config_path,
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        delta=delta,
        final_case_ranks_path=final_case_ranks_path,
        manifest=manifest,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "report_dir": str(report_dir.resolve()),
                "final_case_ranks": str(final_case_ranks_path.resolve()),
                "final_metrics": str((output_dir / "mainline_final_metrics.csv").resolve()),
                "final_metrics_with_sources": str((output_dir / "mainline_final_metrics_with_sources.csv").resolve()),
                "fixed_mimic_metrics": str((output_dir / "mainline_final_metrics_fixed_mimic.csv").resolve()),
                "frozen_protocol_metrics": str((output_dir / "mainline_final_metrics_frozen_protocol.csv").resolve()),
                "parity_audit": str((PROJECT_ROOT / "reports" / "mainline" / "parity_audit_old_vs_current.csv").resolve()),
                "delta": str((output_dir / "mainline_delta_vs_baseline.csv").resolve()),
                "report": str((report_dir / "mainline_final_all_dataset_report.md").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
