from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import load_test_cases, load_yaml_config


DEFAULT_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rerank"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "top50_evidence_rerank_v2_report.md"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"

METRIC_DATASETS = ["DDD", "mimic_test", "HMS", "LIRICAL", "ALL"]
FEATURE_COLUMNS = [
    "hgnn_score",
    "ic_weighted_overlap",
    "exact_overlap",
    "semantic_ic_overlap",
    "case_coverage",
    "disease_coverage",
    "size_penalty",
]
WEIGHT_COLUMNS = [
    "w_hgnn",
    "w_ic",
    "w_exact",
    "w_semantic",
    "w_case_cov",
    "w_dis_cov",
    "w_size",
]
V1_PRESETS: dict[str, dict[str, float]] = {
    "A_hgnn_only": {
        "w_hgnn": 1.0,
        "w_ic": 0.0,
        "w_exact": 0.0,
        "w_semantic": 0.0,
        "w_case_cov": 0.0,
        "w_dis_cov": 0.0,
        "w_size": 0.0,
    },
    "D_hgnn_ic_coverage_v1": {
        "w_hgnn": 0.65,
        "w_ic": 0.25,
        "w_exact": 0.0,
        "w_semantic": 0.0,
        "w_case_cov": 0.05,
        "w_dis_cov": 0.05,
        "w_size": 0.0,
    },
    "E_hgnn_ic_exact_coverage_v1": {
        "w_hgnn": 0.60,
        "w_ic": 0.20,
        "w_exact": 0.10,
        "w_semantic": 0.0,
        "w_case_cov": 0.05,
        "w_dis_cov": 0.05,
        "w_size": 0.0,
    },
}
GRID = {
    "w_hgnn": [0.70, 0.75, 0.80, 0.85, 0.90],
    "w_ic": [0.05, 0.10, 0.15, 0.20],
    "w_exact": [0.00, 0.05, 0.10, 0.15],
    "w_semantic": [0.00, 0.05, 0.10, 0.15],
    "w_case_cov": [0.00, 0.03, 0.05],
    "w_dis_cov": [0.00, 0.03, 0.05],
    "w_size": [0.00, 0.01, 0.02],
}
GATE_THRESHOLDS = {
    "max_exact_threshold": [0, 1, 2],
    "max_ic_threshold": [0.0, 0.05, 0.10, 0.15],
    "hgnn_margin_threshold": [None, 0.02, 0.05, 0.10],
}


@dataclass(slots=True)
class CandidateMatrix:
    case_ids: np.ndarray
    dataset_names: np.ndarray
    original_rank: np.ndarray
    gold_mask: np.ndarray
    feature_arrays: dict[str, np.ndarray]
    gate_arrays: dict[str, np.ndarray]
    top_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2 no-train top50 evidence reranking, grid search, and gated rerank."
    )
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--validation-candidates-path", type=Path, default=None)
    parser.add_argument("--test-candidates-path", type=Path, default=None)
    parser.add_argument("--data-config-path", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mode", choices=["all", "presets", "grid"], default="all")
    parser.add_argument(
        "--protocol",
        choices=["exploratory", "validation_select", "fixed_eval"],
        default="exploratory",
        help=(
            "exploratory keeps the historical test-side analysis; validation_select selects weights on "
            "validation candidates and optionally evaluates test once; fixed_eval loads saved weights."
        ),
    )
    parser.add_argument("--selected-weights-path", type=Path, default=None)
    parser.add_argument("--fixed-weights-path", type=Path, default=None)
    parser.add_argument("--selection-objective", type=str, default="ALL_top1")
    parser.add_argument("--selection-kind", choices=["grid", "gated"], default="gated")
    return parser.parse_args()


def required_columns() -> set[str]:
    return {
        "case_id",
        "dataset_name",
        "gold_id",
        "candidate_id",
        "original_rank",
        "hgnn_score",
        "exact_overlap",
        "ic_weighted_overlap",
        "case_coverage",
        "disease_coverage",
        "disease_hpo_count",
        "hgnn_margin",
        "max_exact_overlap_in_case",
        "max_ic_overlap_in_case",
        "evidence_rank_by_ic",
        "semantic_ic_overlap",
        "semantic_coverage_score",
    }


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Candidates CSV not found: {path}")
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    missing = required_columns() - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    numeric_cols = sorted(required_columns() - {"case_id", "dataset_name", "gold_id", "candidate_id"})
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df = df.sort_values(["case_id", "original_rank"], kind="stable").reset_index(drop=True)
    counts = df.groupby("case_id")["candidate_id"].size()
    if counts.nunique() != 1:
        raise ValueError("Every case must have the same number of top-k candidates.")
    return df


def minmax_by_case(values: np.ndarray) -> np.ndarray:
    mins = values.min(axis=1, keepdims=True)
    maxs = values.max(axis=1, keepdims=True)
    denom = maxs - mins
    return np.divide(values - mins, denom, out=np.ones_like(values, dtype=float), where=denom != 0)


def to_matrix(df: pd.DataFrame) -> CandidateMatrix:
    top_k = int(df.groupby("case_id")["candidate_id"].size().iloc[0])
    case_meta = df.groupby("case_id", sort=False)[["dataset_name"]].first().reset_index()
    num_cases = len(case_meta)
    shape = (num_cases, top_k)

    gold_mask = (df["candidate_id"].to_numpy(dtype=str) == df["gold_id"].to_numpy(dtype=str)).reshape(shape)
    hgnn = df["hgnn_score"].to_numpy(dtype=float).reshape(shape)
    ic = df["ic_weighted_overlap"].to_numpy(dtype=float).reshape(shape)
    exact = df["exact_overlap"].to_numpy(dtype=float).reshape(shape)
    semantic = df["semantic_ic_overlap"].to_numpy(dtype=float).reshape(shape)
    case_cov = df["case_coverage"].to_numpy(dtype=float).reshape(shape)
    dis_cov = df["disease_coverage"].to_numpy(dtype=float).reshape(shape)
    size_penalty = np.log1p(df["disease_hpo_count"].to_numpy(dtype=float).reshape(shape))
    original_rank = df["original_rank"].to_numpy(dtype=int).reshape(shape)

    feature_arrays = {
        "hgnn_score": minmax_by_case(hgnn),
        "ic_weighted_overlap": minmax_by_case(ic),
        "exact_overlap": minmax_by_case(exact),
        "semantic_ic_overlap": minmax_by_case(semantic),
        "case_coverage": minmax_by_case(case_cov),
        "disease_coverage": minmax_by_case(dis_cov),
        "size_penalty": minmax_by_case(size_penalty),
    }
    gate_arrays = {
        "max_exact_overlap_in_case": df["max_exact_overlap_in_case"].to_numpy(dtype=float).reshape(shape)[:, 0],
        "max_ic_overlap_in_case": df["max_ic_overlap_in_case"].to_numpy(dtype=float).reshape(shape)[:, 0],
        "hgnn_margin": df["hgnn_margin"].to_numpy(dtype=float).reshape(shape)[:, 0],
    }
    return CandidateMatrix(
        case_ids=case_meta["case_id"].to_numpy(dtype=str),
        dataset_names=case_meta["dataset_name"].to_numpy(dtype=str),
        original_rank=original_rank,
        gold_mask=gold_mask,
        feature_arrays=feature_arrays,
        gate_arrays=gate_arrays,
        top_k=top_k,
    )


def score_matrix(matrix: CandidateMatrix, weights: dict[str, float]) -> np.ndarray:
    return (
        weights["w_hgnn"] * matrix.feature_arrays["hgnn_score"]
        + weights["w_ic"] * matrix.feature_arrays["ic_weighted_overlap"]
        + weights["w_exact"] * matrix.feature_arrays["exact_overlap"]
        + weights["w_semantic"] * matrix.feature_arrays["semantic_ic_overlap"]
        + weights["w_case_cov"] * matrix.feature_arrays["case_coverage"]
        + weights["w_dis_cov"] * matrix.feature_arrays["disease_coverage"]
        - weights["w_size"] * matrix.feature_arrays["size_penalty"]
    )


def gate_mask(
    matrix: CandidateMatrix,
    *,
    max_exact_threshold: int,
    max_ic_threshold: float,
    hgnn_margin_threshold: float | None,
) -> np.ndarray:
    weak_evidence = (
        (matrix.gate_arrays["max_exact_overlap_in_case"] <= float(max_exact_threshold))
        & (matrix.gate_arrays["max_ic_overlap_in_case"] < float(max_ic_threshold))
    )
    if hgnn_margin_threshold is not None:
        confident_hgnn = matrix.gate_arrays["hgnn_margin"] >= float(hgnn_margin_threshold)
        weak_evidence = weak_evidence | confident_hgnn
    return weak_evidence


def ranks_from_scores(
    matrix: CandidateMatrix,
    scores: np.ndarray,
    *,
    keep_original_mask: np.ndarray | None = None,
) -> np.ndarray:
    effective_scores = scores.copy()
    if keep_original_mask is not None and keep_original_mask.any():
        original_scores = -matrix.original_rank.astype(float)
        effective_scores[keep_original_mask, :] = original_scores[keep_original_mask, :]

    order = np.lexsort((matrix.original_rank, -effective_scores), axis=1)
    sorted_gold = np.take_along_axis(matrix.gold_mask, order, axis=1)
    has_hit = sorted_gold.any(axis=1)
    ranks = np.full(matrix.gold_mask.shape[0], matrix.top_k + 1, dtype=int)
    ranks[has_hit] = np.argmax(sorted_gold[has_hit], axis=1) + 1
    return ranks


def metrics_for_ranks(ranks: np.ndarray) -> dict[str, float]:
    if ranks.size == 0:
        return {
            "num_cases": 0,
            "top1": float("nan"),
            "top3": float("nan"),
            "top5": float("nan"),
            "median_rank": float("nan"),
            "mean_rank": float("nan"),
            "rank_le_50": float("nan"),
        }
    return {
        "num_cases": int(ranks.size),
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "rank_le_50": float(np.mean(ranks <= 50)),
    }


def summarize(matrix: CandidateMatrix, ranks: np.ndarray) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for dataset_name in METRIC_DATASETS:
        if dataset_name == "ALL":
            subset = ranks
        else:
            subset = ranks[matrix.dataset_names == dataset_name]
        output[dataset_name] = metrics_for_ranks(subset)
    return output


def flatten_metrics(
    *,
    preset: str,
    kind: str,
    weights: dict[str, float],
    metrics: dict[str, dict[str, float]],
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"preset": preset, "kind": kind, **weights}
    if gate is not None:
        row.update(gate)
    else:
        row.update({"max_exact_threshold": None, "max_ic_threshold": None, "hgnn_margin_threshold": None})
    for dataset_name, metric_map in metrics.items():
        for metric_name, value in metric_map.items():
            row[f"{dataset_name}_{metric_name}"] = value
    return row


def grid_weight_iter() -> list[dict[str, float]]:
    keys = list(GRID)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(GRID[key] for key in keys))]


def gate_iter() -> list[dict[str, Any]]:
    keys = list(GATE_THRESHOLDS)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(GATE_THRESHOLDS[key] for key in keys))]


def select_pareto(df: pd.DataFrame) -> pd.DataFrame:
    objectives = ["DDD_top1", "DDD_top5", "ALL_top1", "mimic_test_top5", "LIRICAL_top5"]
    rows = df.reset_index(drop=True)
    keep = np.ones(len(rows), dtype=bool)
    values = rows[objectives].to_numpy(dtype=float)
    for idx in range(len(rows)):
        if not keep[idx]:
            continue
        dominated = (values >= values[idx]).all(axis=1) & (values > values[idx]).any(axis=1)
        dominated[idx] = False
        if dominated.any():
            keep[idx] = False
    return rows.loc[keep].sort_values(
        ["DDD_top1", "DDD_top5", "ALL_top1", "mimic_test_top5"],
        ascending=[False, False, False, False],
        kind="stable",
    )


def best_rows(results: pd.DataFrame, baseline_mimic_top5: float) -> dict[str, pd.Series]:
    constrained = results.loc[results["mimic_test_top5"] >= baseline_mimic_top5].copy()
    if constrained.empty:
        constrained = results.copy()
    return {
        "best_ddd_top1": results.sort_values(
            ["DDD_top1", "DDD_top5", "ALL_top1", "mimic_test_top5"],
            ascending=[False, False, False, False],
            kind="stable",
        ).iloc[0],
        "best_ddd_top5": results.sort_values(
            ["DDD_top5", "DDD_top1", "ALL_top1", "mimic_test_top5"],
            ascending=[False, False, False, False],
            kind="stable",
        ).iloc[0],
        "best_all_top1": results.sort_values(
            ["ALL_top1", "DDD_top1", "DDD_top5", "mimic_test_top5"],
            ascending=[False, False, False, False],
            kind="stable",
        ).iloc[0],
        "best_ddd_top1_mimic_safe": constrained.sort_values(
            ["DDD_top1", "DDD_top5", "ALL_top1", "mimic_test_top5"],
            ascending=[False, False, False, False],
            kind="stable",
        ).iloc[0],
        "best_all_top1_mimic_safe": constrained.sort_values(
            ["ALL_top1", "DDD_top1", "DDD_top5", "mimic_test_top5"],
            ascending=[False, False, False, False],
            kind="stable",
        ).iloc[0],
    }


def run_grid(matrix: CandidateMatrix) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    weights_list = grid_weight_iter()
    for index, weights in enumerate(weights_list, start=1):
        scores = score_matrix(matrix, weights)
        ranks = ranks_from_scores(matrix, scores)
        metrics = summarize(matrix, ranks)
        rows.append(flatten_metrics(preset=f"grid_{index:04d}", kind="grid", weights=weights, metrics=metrics))
    grid_df = pd.DataFrame(rows)
    return grid_df, select_pareto(grid_df)


def run_gated_grid(matrix: CandidateMatrix, candidate_weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weight_records = candidate_weights[WEIGHT_COLUMNS].drop_duplicates().to_dict(orient="records")
    gates = gate_iter()
    counter = 0
    for weights in weight_records:
        scores = score_matrix(matrix, {key: float(weights[key]) for key in WEIGHT_COLUMNS})
        for gate in gates:
            counter += 1
            keep_original = gate_mask(matrix, **gate)
            ranks = ranks_from_scores(matrix, scores, keep_original_mask=keep_original)
            metrics = summarize(matrix, ranks)
            gate_row = {
                **gate,
                "gated_case_ratio": float(np.mean(keep_original)),
            }
            rows.append(
                flatten_metrics(
                    preset=f"gated_{counter:04d}",
                    kind="gated_grid",
                    weights={key: float(weights[key]) for key in WEIGHT_COLUMNS},
                    metrics=metrics,
                    gate=gate_row,
                )
            )
    return pd.DataFrame(rows)


def _clean_gate(row: pd.Series) -> dict[str, Any] | None:
    required = ["max_exact_threshold", "max_ic_threshold", "hgnn_margin_threshold"]
    if any(key not in row for key in required):
        return None
    if pd.isna(row["max_exact_threshold"]) or pd.isna(row["max_ic_threshold"]):
        return None
    margin = row["hgnn_margin_threshold"]
    return {
        "max_exact_threshold": int(row["max_exact_threshold"]),
        "max_ic_threshold": float(row["max_ic_threshold"]),
        "hgnn_margin_threshold": None if pd.isna(margin) else float(margin),
    }


def select_best_row(results: pd.DataFrame, objective: str) -> pd.Series:
    if results.empty:
        raise ValueError("Cannot select weights from an empty results table.")
    if objective not in results.columns:
        raise KeyError(f"Selection objective {objective!r} not found in results columns.")
    sort_columns = [objective]
    for tie_breaker in ("ALL_top1", "ALL_top5", "DDD_top1", "mimic_test_top5"):
        if tie_breaker in results.columns and tie_breaker not in sort_columns:
            sort_columns.append(tie_breaker)
    return results.sort_values(sort_columns, ascending=[False] * len(sort_columns), kind="stable").iloc[0]


def weight_payload_from_row(
    row: pd.Series,
    *,
    objective: str,
    source_candidates_path: Path,
    protocol: str,
) -> dict[str, Any]:
    weights = {key: float(row[key]) for key in WEIGHT_COLUMNS}
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": protocol,
        "selection_objective": objective,
        "source_candidates_path": str(source_candidates_path.resolve()),
        "selected_preset": str(row.get("preset", "")),
        "selected_kind": str(row.get("kind", "")),
        "weights": weights,
        "gate": _clean_gate(row),
        "selected_metrics": {
            key: float(row[key])
            for key in row.index
            if any(key.endswith(f"_{metric}") for metric in ("top1", "top3", "top5", "rank_le_50"))
            and isinstance(row[key], (int, float, np.floating))
            and not pd.isna(row[key])
        },
        "warning": (
            "Use this payload only if it was selected on validation candidates. "
            "Do not present test-side exploratory weights as final test-set tuning."
        ),
    }


def save_weight_payload(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_weight_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Fixed weights JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = set(WEIGHT_COLUMNS) - set(payload.get("weights", {}))
    if missing:
        raise ValueError(f"Fixed weights JSON missing weights: {sorted(missing)}")
    return payload


def evaluate_fixed_payload(
    matrix: CandidateMatrix,
    payload: dict[str, Any],
    *,
    preset: str = "fixed_weights",
) -> pd.DataFrame:
    weights = {key: float(payload["weights"][key]) for key in WEIGHT_COLUMNS}
    scores = score_matrix(matrix, weights)
    gate = payload.get("gate")
    keep_original = None
    if isinstance(gate, dict):
        keep_original = gate_mask(
            matrix,
            max_exact_threshold=int(gate["max_exact_threshold"]),
            max_ic_threshold=float(gate["max_ic_threshold"]),
            hgnn_margin_threshold=gate.get("hgnn_margin_threshold"),
        )
    ranks = ranks_from_scores(matrix, scores, keep_original_mask=keep_original)
    metrics = summarize(matrix, ranks)
    return pd.DataFrame(
        [
            flatten_metrics(
                preset=preset,
                kind="fixed_eval",
                weights=weights,
                metrics=metrics,
                gate=gate if isinstance(gate, dict) else None,
            )
        ]
    )


def run_validation_select(
    *,
    validation_candidates_path: Path,
    test_candidates_path: Path | None,
    output_dir: Path,
    selected_weights_path: Path,
    objective: str,
    selection_kind: str,
) -> dict[str, Path]:
    validation_candidates = load_candidates(validation_candidates_path)
    validation_matrix = to_matrix(validation_candidates)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_df, pareto_df = run_grid(validation_matrix)
    selector_pool = pareto_df
    gated_df = pd.DataFrame()
    if selection_kind == "gated":
        gated_df = run_gated_grid(validation_matrix, pareto_df)
        selection_table = gated_df if not gated_df.empty else grid_df
    else:
        selection_table = grid_df

    selected = select_best_row(selection_table, objective)
    payload = weight_payload_from_row(
        selected,
        objective=objective,
        source_candidates_path=validation_candidates_path,
        protocol="validation_select",
    )
    save_weight_payload(payload, selected_weights_path)

    validation_grid_path = output_dir / "rerank_validation_grid_results.csv"
    validation_pareto_path = output_dir / "rerank_validation_pareto.csv"
    validation_gated_path = output_dir / "rerank_validation_gated_results.csv"
    grid_df.to_csv(validation_grid_path, index=False, encoding="utf-8-sig")
    pareto_df.to_csv(validation_pareto_path, index=False, encoding="utf-8-sig")
    gated_df.to_csv(validation_gated_path, index=False, encoding="utf-8-sig")

    paths = {
        "validation_grid_path": validation_grid_path,
        "validation_pareto_path": validation_pareto_path,
        "validation_gated_path": validation_gated_path,
        "selected_weights_path": selected_weights_path,
    }

    if test_candidates_path is not None:
        test_matrix = to_matrix(load_candidates(test_candidates_path))
        fixed_df = evaluate_fixed_payload(test_matrix, payload, preset="validation_selected_fixed_test")
        fixed_path = output_dir / "rerank_fixed_test_metrics.csv"
        fixed_df.to_csv(fixed_path, index=False, encoding="utf-8-sig")
        paths["fixed_test_metrics_path"] = fixed_path

    return paths


def run_fixed_eval(
    *,
    candidates_path: Path,
    fixed_weights_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    matrix = to_matrix(load_candidates(candidates_path))
    payload = load_weight_payload(fixed_weights_path)
    fixed_df = evaluate_fixed_payload(matrix, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = output_dir / "rerank_fixed_eval_metrics.csv"
    fixed_df.to_csv(fixed_path, index=False, encoding="utf-8-sig")
    return {"fixed_eval_metrics_path": fixed_path}


def load_mimic_labels_from_data_config(data_config_path: Path) -> pd.DataFrame:
    data_config = load_yaml_config(data_config_path)
    test_bundle = load_test_cases(data_config, data_config_path)
    raw_df = test_bundle["raw_df"].copy()
    case_id_col = test_bundle["case_id_col"]
    label_col = test_bundle["label_col"]
    mimic = raw_df.loc[raw_df["_source_file"].astype(str).apply(lambda value: Path(value).stem == "mimic_test")]
    labels_per_case = (
        mimic.dropna(subset=[case_id_col, label_col])
        .groupby(case_id_col)[label_col]
        .agg(lambda values: sorted(set(str(value) for value in values)))
        .reset_index()
        .rename(columns={case_id_col: "case_id", label_col: "labels"})
    )
    labels_per_case["num_labels"] = labels_per_case["labels"].apply(len)
    return labels_per_case


def run_presets(matrix: CandidateMatrix) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for preset, weights in V1_PRESETS.items():
        scores = score_matrix(matrix, weights)
        ranks = ranks_from_scores(matrix, scores)
        metrics = summarize(matrix, ranks)
        rows.append(flatten_metrics(preset=preset, kind="preset", weights=weights, metrics=metrics))
    return pd.DataFrame(rows)


def load_legacy_v1_metrics(output_dir: Path, fallback_df: pd.DataFrame) -> pd.DataFrame:
    legacy_path = output_dir / "top50_rerank_metrics.csv"
    fallback = fallback_df.loc[
        fallback_df["preset"].isin(["D_hgnn_ic_coverage_v1", "E_hgnn_ic_exact_coverage_v1"])
    ].copy()
    if not legacy_path.is_file():
        return fallback

    legacy = pd.read_csv(legacy_path)
    rename_map = {
        "D_hgnn_ic_coverage": "D_hgnn_ic_coverage_v1",
        "E_hgnn_ic_exact_coverage": "E_hgnn_ic_exact_coverage_v1",
    }
    selected = legacy.loc[legacy["preset"].isin(rename_map) & legacy["dataset_name"].isin(METRIC_DATASETS)].copy()
    rows: list[dict[str, Any]] = []
    for old_name, new_name in rename_map.items():
        group = selected.loc[selected["preset"] == old_name]
        if group.empty:
            return fallback
        row: dict[str, Any] = {
            "preset": new_name,
            "kind": "legacy_v1",
            "w_hgnn": np.nan,
            "w_ic": np.nan,
            "w_exact": np.nan,
            "w_semantic": 0.0,
            "w_case_cov": np.nan,
            "w_dis_cov": np.nan,
            "w_size": np.nan,
            "max_exact_threshold": None,
            "max_ic_threshold": None,
            "hgnn_margin_threshold": None,
        }
        for item in group.itertuples(index=False):
            dataset = str(item.dataset_name)
            row[f"{dataset}_num_cases"] = int(item.num_cases)
            row[f"{dataset}_top1"] = float(item.top1)
            row[f"{dataset}_top3"] = float(item.top3)
            row[f"{dataset}_top5"] = float(item.top5)
            row[f"{dataset}_median_rank"] = float(item.median_rank)
            row[f"{dataset}_mean_rank"] = float(item.mean_rank)
            row[f"{dataset}_rank_le_50"] = float(item.rank_le_50)
        rows.append(row)
    return pd.DataFrame(rows) if len(rows) == 2 else fallback


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in view[columns].itertuples(index=False):
        lines.append("|" + "|".join(_fmt(value) for value in row) + "|")
    return "\n".join(lines)


def row_summary(row: pd.Series) -> str:
    return (
        f"`{row['preset']}` "
        f"DDD top1/top3/top5={row['DDD_top1']:.4f}/{row['DDD_top3']:.4f}/{row['DDD_top5']:.4f}, "
        f"mimic_test top5={row['mimic_test_top5']:.4f}, "
        f"LIRICAL top5={row['LIRICAL_top5']:.4f}, ALL top1={row['ALL_top1']:.4f}"
    )


def write_report(
    *,
    report_path: Path,
    candidates_path: Path,
    metadata_path: Path,
    presets_df: pd.DataFrame,
    legacy_v1_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    semantic_summary: dict[str, Any],
    mimic_label_summary: dict[str, Any],
) -> None:
    baseline = presets_df.loc[presets_df["preset"] == "A_hgnn_only"].iloc[0]
    v1_d = legacy_v1_df.loc[legacy_v1_df["preset"] == "D_hgnn_ic_coverage_v1"].iloc[0]
    v1_e = legacy_v1_df.loc[legacy_v1_df["preset"] == "E_hgnn_ic_exact_coverage_v1"].iloc[0]
    grid_best = best_rows(grid_df, float(baseline["mimic_test_top5"]))
    gated_best = best_rows(gated_df, float(baseline["mimic_test_top5"]))
    recommended_main = gated_best["best_ddd_top1_mimic_safe"]
    recommended_appendix = grid_best["best_ddd_top1"]

    semantic_effective = bool(semantic_summary.get("semantic_nonzero_ratio", 0.0) > 0.0)
    best_semantic = grid_df.loc[grid_df["w_semantic"] > 0].sort_values(
        ["DDD_top1", "DDD_top5", "mimic_test_top5"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    best_no_semantic = grid_df.loc[grid_df["w_semantic"] == 0].sort_values(
        ["DDD_top1", "DDD_top5", "mimic_test_top5"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    mimic_protected = bool(recommended_main["mimic_test_top5"] >= baseline["mimic_test_top5"])
    ddd_improved = bool(
        recommended_main["DDD_top1"] > baseline["DDD_top1"]
        and recommended_main["DDD_top3"] >= baseline["DDD_top3"]
        and recommended_main["DDD_top5"] >= baseline["DDD_top5"]
    )
    lirical_improved = bool(recommended_main["LIRICAL_top5"] >= baseline["LIRICAL_top5"])
    recommend_hn = bool(ddd_improved and mimic_protected)

    key_cols = [
        "preset",
        "kind",
        "DDD_top1",
        "DDD_top3",
        "DDD_top5",
        "mimic_test_top5",
        "LIRICAL_top1",
        "LIRICAL_top5",
        "ALL_top1",
        "ALL_top5",
        "w_hgnn",
        "w_ic",
        "w_exact",
        "w_semantic",
        "w_case_cov",
        "w_dis_cov",
        "w_size",
        "max_exact_threshold",
        "max_ic_threshold",
        "hgnn_margin_threshold",
    ]
    combined_best = pd.DataFrame(
        [
            baseline,
            v1_d,
            v1_e,
            grid_best["best_ddd_top1"],
            grid_best["best_ddd_top5"],
            grid_best["best_ddd_top1_mimic_safe"],
            gated_best["best_ddd_top1"],
            gated_best["best_ddd_top1_mimic_safe"],
            gated_best["best_all_top1_mimic_safe"],
        ]
    ).drop_duplicates(subset=["preset", "kind"], keep="first")

    lines = [
        "# top50 evidence rerank v2 report",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- candidates_path: `{candidates_path}`",
        f"- candidates_metadata_path: `{metadata_path}`",
        "- analysis_scope: exploratory test-side analysis only; do not present these weights as final tuned test-set conclusions. For a paper, select weights again on validation data and report test once.",
        "- rank_policy: gold absent from top50 is counted as rank 51; mean_rank is top50-capped.",
        "",
        "## baseline and selected results",
        "",
        markdown_table(combined_best, key_cols),
        "",
        "## Pareto frontier preview",
        "",
        markdown_table(pareto_df, key_cols[:17], max_rows=20),
        "",
        "## Required answers",
        "",
        f"1. baseline A_hgnn_only: {row_summary(baseline)}.",
        f"2. v1 best D/E: D is {row_summary(v1_d)}; E is {row_summary(v1_e)}.",
        f"3. v2 grid search best by DDD top1: {row_summary(grid_best['best_ddd_top1'])}. Best mimic-safe DDD top1: {row_summary(grid_best['best_ddd_top1_mimic_safe'])}.",
        f"4. gated rerank best mimic-safe DDD top1: {row_summary(gated_best['best_ddd_top1_mimic_safe'])}.",
        f"5. DDD top1/top3/top5 是否继续提升: {'是' if ddd_improved else '否'}。推荐 gated 配置相对 baseline 为 "
        f"{recommended_main['DDD_top1'] - baseline['DDD_top1']:+.4f}/"
        f"{recommended_main['DDD_top3'] - baseline['DDD_top3']:+.4f}/"
        f"{recommended_main['DDD_top5'] - baseline['DDD_top5']:+.4f}。",
        f"6. mimic_test top5 是否被保护: {'是' if mimic_protected else '否'}。推荐配置 mimic_test top5={recommended_main['mimic_test_top5']:.4f}, baseline={baseline['mimic_test_top5']:.4f}。",
        f"7. LIRICAL 是否继续提升: {'是' if lirical_improved else '否'}。推荐配置 LIRICAL top5={recommended_main['LIRICAL_top5']:.4f}, baseline={baseline['LIRICAL_top5']:.4f}。",
        f"8. semantic_ic_overlap 是否有效: {'是' if semantic_effective else '否'}。nonzero_ratio={semantic_summary.get('semantic_nonzero_ratio', 0.0):.4f}, mean={semantic_summary.get('semantic_mean', 0.0):.4f}。",
        f"9. 是否建议进入 hard negative training: {'建议' if recommend_hn else '暂不建议'}。原因是当前 no-train gated rerank {'同时改善 DDD 并保护 mimic_test' if recommend_hn else '未稳定同时满足 DDD 提升和 mimic_test 保护'}。",
        f"10. 推荐最终论文主表配置: `{recommended_main['preset']}` 作为 validation 重新选择后的候选方案；附表配置: `{recommended_appendix['preset']}` 展示 test-side exploratory 上界和 ablation。正式论文必须在 validation set 上重新选权重。",
        "",
        "## mimic_test multi-label audit",
        "",
        f"- mimic_cases={mimic_label_summary['mimic_cases']}, multi_label_cases={mimic_label_summary['multi_label_cases']}, ratio={mimic_label_summary['multi_label_case_ratio']:.4f}.",
        "- 主 exact metric 未改变；multi-label 只用于审计潜在假错误。",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    candidates_path: Path,
    data_config_path: Path,
    output_dir: Path,
    report_path: Path,
    mode: str,
) -> dict[str, Path]:
    candidates = load_candidates(candidates_path)
    matrix = to_matrix(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)

    presets_df = run_presets(matrix)
    legacy_v1_df = load_legacy_v1_metrics(output_dir, presets_df)
    grid_df = pd.DataFrame()
    pareto_df = pd.DataFrame()
    gated_df = pd.DataFrame()
    if mode in {"all", "grid"}:
        grid_df, pareto_df = run_grid(matrix)
        baseline_mimic_top5 = float(presets_df.loc[presets_df["preset"] == "A_hgnn_only", "mimic_test_top5"].iloc[0])
        selector_pool = pd.concat(
            [
                pareto_df,
                pd.DataFrame(best_rows(grid_df, baseline_mimic_top5)).T,
            ],
            ignore_index=True,
        ).drop_duplicates(subset=WEIGHT_COLUMNS)
        gated_df = run_gated_grid(matrix, selector_pool)

    if mode == "presets":
        grid_df = presets_df.iloc[0:0].copy()
        pareto_df = presets_df.iloc[0:0].copy()
        gated_df = presets_df.iloc[0:0].copy()

    metadata_path = candidates_path.with_suffix(".metadata.json")
    semantic_summary = {
        "semantic_nonzero_ratio": float((candidates["semantic_ic_overlap"] > 0).mean()),
        "semantic_mean": float(candidates["semantic_ic_overlap"].mean()),
        "semantic_max": float(candidates["semantic_ic_overlap"].max()),
    }
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            semantic_summary["ontology_path"] = metadata.get("semantic", {}).get("ontology_path")
            semantic_summary["semantic_available"] = metadata.get("semantic", {}).get("available")
        except json.JSONDecodeError:
            semantic_summary["metadata_warning"] = "Could not parse candidate metadata JSON."

    labels = load_mimic_labels_from_data_config(data_config_path)
    mimic_label_summary = {
        "mimic_cases": int(labels["case_id"].nunique()),
        "multi_label_cases": int((labels["num_labels"] > 1).sum()),
        "multi_label_case_ratio": float((labels["num_labels"] > 1).mean()) if len(labels) else 0.0,
    }

    presets_path = output_dir / "rerank_v2_presets.csv"
    grid_path = output_dir / "rerank_v2_grid_results.csv"
    pareto_path = output_dir / "rerank_v2_pareto.csv"
    gated_path = output_dir / "rerank_v2_gated_results.csv"
    semantic_path = output_dir / "rerank_v2_semantic_summary.json"

    presets_df.to_csv(presets_path, index=False, encoding="utf-8-sig")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    pareto_df.to_csv(pareto_path, index=False, encoding="utf-8-sig")
    gated_df.to_csv(gated_path, index=False, encoding="utf-8-sig")
    semantic_path.write_text(
        json.dumps(
            {
                "semantic_summary": semantic_summary,
                "mimic_label_summary": mimic_label_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if mode == "all":
        write_report(
            report_path=report_path,
            candidates_path=candidates_path,
            metadata_path=metadata_path,
            presets_df=presets_df,
            legacy_v1_df=legacy_v1_df,
            grid_df=grid_df,
            pareto_df=pareto_df,
            gated_df=gated_df,
            semantic_summary=semantic_summary,
            mimic_label_summary=mimic_label_summary,
        )

    paths = {
        "presets_path": presets_path,
        "grid_path": grid_path,
        "pareto_path": pareto_path,
        "gated_path": gated_path,
        "semantic_path": semantic_path,
    }
    if mode == "all":
        paths["report_path"] = report_path
    return paths


def main() -> None:
    args = parse_args()
    if args.protocol == "validation_select":
        if args.validation_candidates_path is None:
            raise ValueError("--validation-candidates-path is required for --protocol validation_select.")
        selected_weights_path = args.selected_weights_path or (args.output_dir / "rerank_selected_weights.json")
        paths = run_validation_select(
            validation_candidates_path=args.validation_candidates_path,
            test_candidates_path=args.test_candidates_path,
            output_dir=args.output_dir,
            selected_weights_path=selected_weights_path,
            objective=args.selection_objective,
            selection_kind=args.selection_kind,
        )
    elif args.protocol == "fixed_eval":
        if args.fixed_weights_path is None:
            raise ValueError("--fixed-weights-path is required for --protocol fixed_eval.")
        eval_candidates_path = args.test_candidates_path or args.candidates_path
        paths = run_fixed_eval(
            candidates_path=eval_candidates_path,
            fixed_weights_path=args.fixed_weights_path,
            output_dir=args.output_dir,
        )
    else:
        paths = run(
            candidates_path=args.candidates_path,
            data_config_path=args.data_config_path,
            output_dir=args.output_dir,
            report_path=args.report_path,
            mode=args.mode,
        )
    print(json.dumps({key: str(path.resolve()) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
