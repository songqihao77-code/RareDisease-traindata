from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_top50_pairwise_reranker import feature_frame

DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "rerank" / "pairwise_reranker.pkl"
DEFAULT_TEST_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "deeprare_parity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed test evaluation for the top50 pairwise reranker.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-candidates-path", type=Path, default=DEFAULT_TEST_CANDIDATES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def load_model(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pairwise reranker model not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    required = {"features", "scaler", "model"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} missing model payload keys: {sorted(missing)}")
    return payload


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Candidates CSV not found: {path}")
    return pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str}).sort_values(
        ["case_id", "original_rank"],
        kind="stable",
    )


def score_with_features(df: pd.DataFrame, payload: dict[str, Any], active_features: list[str]) -> pd.DataFrame:
    features = list(payload["features"])
    x_df, _ = feature_frame(
        df,
        allow_gold_leakage_features=bool(payload.get("gold_leakage_features_enabled", False)),
    )
    missing = [feature for feature in features if feature not in x_df.columns]
    if missing:
        raise ValueError(f"Test candidates missing reranker features: {missing}")
    x = x_df[features].to_numpy(dtype=float)
    x_scaled = payload["scaler"].transform(x)
    active = np.asarray([1.0 if feature in active_features else 0.0 for feature in features], dtype=float)
    coef = payload["model"].coef_.reshape(-1) * active
    scored = x_df.copy()
    scored["pairwise_score"] = x_scaled @ coef
    return scored


def evaluate(scored: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for dataset, dataset_df in scored.groupby("dataset_name", sort=True):
        ranks = []
        for _, group in dataset_df.groupby("case_id", sort=False):
            ordered = group.sort_values(["pairwise_score", "original_rank"], ascending=[False, True], kind="stable")
            hits = np.flatnonzero(ordered["candidate_id"].to_numpy(str) == ordered["gold_id"].to_numpy(str))
            ranks.append(int(hits[0] + 1) if len(hits) else 51)
        arr = np.asarray(ranks, dtype=int)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "num_cases": int(len(arr)),
                "top1": float((arr <= 1).mean()),
                "top3": float((arr <= 3).mean()),
                "top5": float((arr <= 5).mean()),
                "median_rank": float(np.median(arr)),
                "rank_le_50": float((arr <= 50).mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    relation = [feature for feature in features if feature.startswith("relation__")]
    return {
        "all_features": features,
        "no_hgnn_score": [feature for feature in features if feature != "hgnn_score"],
        "no_overlap_features": [
            feature
            for feature in features
            if feature not in {"exact_overlap", "ic_weighted_overlap", "semantic_ic_overlap"}
        ],
        "no_coverage_features": [feature for feature in features if feature not in {"case_coverage", "disease_coverage"}],
        "no_relation_features": [feature for feature in features if feature not in relation],
    }


def main() -> None:
    args = parse_args()
    payload = load_model(args.model_path)
    candidates = load_candidates(args.test_candidates_path)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    all_features = list(payload["features"])
    test_metrics = evaluate(score_with_features(candidates, payload, all_features), "pairwise_reranker_fixed_test")
    test_metrics.to_csv(args.report_dir / "pairwise_reranker_test.csv", index=False, encoding="utf-8-sig")

    ablation_frames = []
    for name, active_features in feature_groups(all_features).items():
        scored = score_with_features(candidates, payload, active_features)
        ablation_frames.append(evaluate(scored, name))
    ablation = pd.concat(ablation_frames, ignore_index=True)
    ablation.to_csv(args.report_dir / "pairwise_reranker_ablation.csv", index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "test": str((args.report_dir / "pairwise_reranker_test.csv").resolve()),
                "ablation": str((args.report_dir / "pairwise_reranker_ablation.csv").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
