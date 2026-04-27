from __future__ import annotations

import argparse
import itertools
import json
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


DEFAULT_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rerank"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"

METRIC_DATASETS = ["DDD", "mimic_test", "HMS", "LIRICAL", "ALL"]
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


@dataclass(slots=True)
class CandidateMatrix:
    case_ids: np.ndarray
    dataset_names: np.ndarray
    original_rank: np.ndarray
    gold_mask: np.ndarray
    feature_arrays: dict[str, np.ndarray]
    top_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-selected grid reranking or fixed evaluation on top50 candidates."
    )
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--validation-candidates-path", type=Path, default=None)
    parser.add_argument("--test-candidates-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--protocol",
        choices=["validation_select", "fixed_eval", "presets"],
        default="fixed_eval",
    )
    parser.add_argument("--selected-weights-path", type=Path, default=None)
    parser.add_argument("--fixed-weights-path", type=Path, default=None)
    parser.add_argument("--selection-objective", type=str, default="DDD_top1")
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
        "semantic_ic_overlap",
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

    return CandidateMatrix(
        case_ids=case_meta["case_id"].to_numpy(dtype=str),
        dataset_names=case_meta["dataset_name"].to_numpy(dtype=str),
        original_rank=original_rank,
        gold_mask=gold_mask,
        feature_arrays={
            "hgnn_score": minmax_by_case(hgnn),
            "ic_weighted_overlap": minmax_by_case(ic),
            "exact_overlap": minmax_by_case(exact),
            "semantic_ic_overlap": minmax_by_case(semantic),
            "case_coverage": minmax_by_case(case_cov),
            "disease_coverage": minmax_by_case(dis_cov),
            "size_penalty": minmax_by_case(size_penalty),
        },
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


def ranks_from_scores(matrix: CandidateMatrix, scores: np.ndarray) -> np.ndarray:
    order = np.lexsort((matrix.original_rank, -scores), axis=1)
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
        subset = ranks if dataset_name == "ALL" else ranks[matrix.dataset_names == dataset_name]
        output[dataset_name] = metrics_for_ranks(subset)
    return output


def flatten_metrics(
    *,
    preset: str,
    kind: str,
    weights: dict[str, float],
    metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "preset": preset,
        "kind": kind,
        **weights,
        "max_exact_threshold": None,
        "max_ic_threshold": None,
        "hgnn_margin_threshold": None,
    }
    for dataset_name, metric_map in metrics.items():
        for metric_name, value in metric_map.items():
            row[f"{dataset_name}_{metric_name}"] = value
    return row


def grid_weight_iter() -> list[dict[str, float]]:
    keys = list(GRID)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(GRID[key] for key in keys))]


def select_pareto(df: pd.DataFrame) -> pd.DataFrame:
    objectives = ["DDD_top1", "DDD_top5", "ALL_top1", "mimic_test_top5", "LIRICAL_top5"]
    rows = df.reset_index(drop=True)
    keep = np.ones(len(rows), dtype=bool)
    values = rows[objectives].to_numpy(dtype=float)
    for i in range(len(rows)):
        if not keep[i]:
            continue
        dominates_i = (values >= values[i]).all(axis=1) & (values > values[i]).any(axis=1)
        dominates_i[i] = False
        if dominates_i.any():
            keep[i] = False
    return rows.loc[keep].reset_index(drop=True)


def select_best_row(results: pd.DataFrame, objective: str) -> pd.Series:
    if results.empty:
        raise ValueError("Cannot select weights from an empty results table.")
    if objective not in results.columns:
        raise KeyError(f"Selection objective {objective!r} not found in results columns.")
    tie_breakers = [objective, "DDD_top5", "DDD_top3", "ALL_top1", "ALL_top5"]
    columns = [column for column in tie_breakers if column in results.columns]
    return results.sort_values(columns, ascending=[False] * len(columns), kind="stable").iloc[0]


def run_grid(matrix: CandidateMatrix) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for index, weights in enumerate(grid_weight_iter(), start=1):
        scores = score_matrix(matrix, weights)
        ranks = ranks_from_scores(matrix, scores)
        rows.append(flatten_metrics(preset=f"grid_{index:04d}", kind="grid", weights=weights, metrics=summarize(matrix, ranks)))
    grid_df = pd.DataFrame(rows)
    return grid_df, select_pareto(grid_df)


def weight_payload_from_row(
    row: pd.Series,
    *,
    objective: str,
    source_candidates_path: Path,
    protocol: str,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": protocol,
        "selection_objective": objective,
        "source_candidates_path": str(source_candidates_path.resolve()),
        "selected_preset": str(row.get("preset", "")),
        "selected_kind": "grid",
        "weights": {key: float(row[key]) for key in WEIGHT_COLUMNS},
        "gate": None,
        "selected_metrics": {
            key: float(row[key])
            for key in row.index
            if any(key.endswith(f"_{metric}") for metric in ("top1", "top3", "top5", "rank_le_50"))
        },
        "warning": "Use this payload only if it was selected on validation candidates. Do not use test-side selection as a final result.",
    }


def save_weight_payload(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _payload_from_container(payload: dict[str, Any], objective: str) -> dict[str, Any]:
    selections = payload.get("selections")
    if isinstance(selections, list):
        for selection in selections:
            if selection.get("selection_objective") == objective and selection.get("selection_kind", "grid") == "grid":
                return selection
        raise KeyError(f"No grid selection for objective {objective!r}.")
    return payload


def load_weight_payload(path: Path, objective: str = "DDD_top1") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Fixed weights JSON not found: {path}")
    payload = _payload_from_container(json.loads(path.read_text(encoding="utf-8")), objective)
    missing = set(WEIGHT_COLUMNS) - set(payload.get("weights", {}))
    if missing:
        raise ValueError(f"{path} missing weights: {sorted(missing)}")
    if payload.get("gate") not in (None, {}):
        raise ValueError("Gated rerank payloads are no longer supported by the frozen mainline.")
    return payload


def evaluate_fixed_payload(
    matrix: CandidateMatrix,
    payload: dict[str, Any],
    *,
    preset: str = "fixed_weights",
) -> pd.DataFrame:
    weights = {key: float(payload["weights"][key]) for key in WEIGHT_COLUMNS}
    ranks = ranks_from_scores(matrix, score_matrix(matrix, weights))
    metrics = summarize(matrix, ranks)
    return pd.DataFrame([flatten_metrics(preset=preset, kind="fixed_eval", weights=weights, metrics=metrics)])


def run_validation_select(
    *,
    validation_candidates_path: Path,
    test_candidates_path: Path | None,
    output_dir: Path,
    selected_weights_path: Path,
    objective: str,
) -> dict[str, Path]:
    validation_matrix = to_matrix(load_candidates(validation_candidates_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_df, pareto_df = run_grid(validation_matrix)
    selected = select_best_row(grid_df, objective)
    payload = weight_payload_from_row(
        selected,
        objective=objective,
        source_candidates_path=validation_candidates_path,
        protocol="validation_select",
    )
    save_weight_payload(payload, selected_weights_path)

    validation_grid_path = output_dir / "rerank_validation_grid_results.csv"
    validation_pareto_path = output_dir / "rerank_validation_pareto.csv"
    grid_df.to_csv(validation_grid_path, index=False, encoding="utf-8-sig")
    pareto_df.to_csv(validation_pareto_path, index=False, encoding="utf-8-sig")

    paths = {
        "validation_grid_path": validation_grid_path,
        "validation_pareto_path": validation_pareto_path,
        "selected_weights_path": selected_weights_path,
    }
    if test_candidates_path is not None:
        fixed_df = evaluate_fixed_payload(to_matrix(load_candidates(test_candidates_path)), payload, preset="validation_selected_fixed_test")
        fixed_path = output_dir / "rerank_fixed_test_metrics.csv"
        fixed_df.to_csv(fixed_path, index=False, encoding="utf-8-sig")
        paths["fixed_test_metrics_path"] = fixed_path
    return paths


def run_fixed_eval(
    *,
    candidates_path: Path,
    fixed_weights_path: Path,
    output_dir: Path,
    objective: str,
) -> dict[str, Path]:
    matrix = to_matrix(load_candidates(candidates_path))
    payload = load_weight_payload(fixed_weights_path, objective=objective)
    fixed_df = evaluate_fixed_payload(matrix, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = output_dir / "rerank_fixed_eval_metrics.csv"
    fixed_df.to_csv(fixed_path, index=False, encoding="utf-8-sig")
    return {"fixed_eval_metrics_path": fixed_path}


def run_presets(matrix: CandidateMatrix) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for preset, weights in V1_PRESETS.items():
        ranks = ranks_from_scores(matrix, score_matrix(matrix, weights))
        rows.append(flatten_metrics(preset=preset, kind="preset", weights=weights, metrics=summarize(matrix, ranks)))
    return pd.DataFrame(rows)


def run_preset_eval(*, candidates_path: Path, output_dir: Path) -> dict[str, Path]:
    df = run_presets(to_matrix(load_candidates(candidates_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rerank_preset_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {"preset_metrics_path": path}


def main() -> None:
    args = parse_args()
    if args.protocol == "validation_select":
        if args.validation_candidates_path is None:
            raise ValueError("--validation-candidates-path is required for validation_select.")
        selected_weights_path = args.selected_weights_path or (args.output_dir / "rerank_selected_grid_weights.json")
        paths = run_validation_select(
            validation_candidates_path=args.validation_candidates_path,
            test_candidates_path=args.test_candidates_path,
            output_dir=args.output_dir,
            selected_weights_path=selected_weights_path,
            objective=args.selection_objective,
        )
    elif args.protocol == "fixed_eval":
        if args.fixed_weights_path is None:
            raise ValueError("--fixed-weights-path is required for fixed_eval.")
        paths = run_fixed_eval(
            candidates_path=args.test_candidates_path or args.candidates_path,
            fixed_weights_path=args.fixed_weights_path,
            output_dir=args.output_dir,
            objective=args.selection_objective,
        )
    else:
        paths = run_preset_eval(candidates_path=args.candidates_path, output_dir=args.output_dir)

    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
