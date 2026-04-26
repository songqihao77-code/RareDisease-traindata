from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "rerank" / "pairwise_reranker.pkl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "deeprare_parity"

BASE_FEATURES = [
    "hgnn_score",
    "hgnn_margin",
    "exact_overlap",
    "ic_weighted_overlap",
    "semantic_ic_overlap",
    "case_coverage",
    "disease_coverage",
    "log1p_disease_hpo_count",
]
LEAKAGE_FEATURE = "candidate_gold_shared_hpo_count"
RELATION_VALUES = [
    "same_parent",
    "sibling",
    "shared_ancestor",
    "ancestor",
    "descendant",
    "unrelated",
    "relation_unavailable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight linear pairwise reranker on HGNN top50 candidates.")
    parser.add_argument("--train-candidates-path", type=Path, required=True)
    parser.add_argument("--validation-candidates-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--max-negatives-per-case", type=int, default=20)
    parser.add_argument(
        "--allow-gold-leakage-features",
        action="store_true",
        help="Include candidate_gold_shared_hpo_count if present. This is for train-only diagnostics, not formal test use.",
    )
    return parser.parse_args()


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Candidates CSV not found: {path}")
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    required = {"case_id", "dataset_name", "gold_id", "candidate_id", "original_rank", "hgnn_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.sort_values(["case_id", "original_rank"], kind="stable").reset_index(drop=True)
    return df


def relation_bucket(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"same_parent", "sibling", "shared_ancestor", "ancestor", "descendant", "unrelated"}:
        return text
    if "same_parent" in text:
        return "same_parent"
    if "sibling" in text:
        return "sibling"
    if "shared_ancestor" in text:
        return "shared_ancestor"
    if "ancestor" in text:
        return "ancestor"
    if "descendant" in text:
        return "descendant"
    if text and text not in {"nan", "None"}:
        return "unrelated"
    return "relation_unavailable"


def feature_frame(df: pd.DataFrame, *, allow_gold_leakage_features: bool) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    for column in ["hgnn_margin", "exact_overlap", "ic_weighted_overlap", "semantic_ic_overlap", "case_coverage", "disease_coverage"]:
        if column not in out.columns:
            out[column] = 0.0
    if "disease_hpo_count" in out.columns:
        out["log1p_disease_hpo_count"] = np.log1p(pd.to_numeric(out["disease_hpo_count"], errors="coerce").fillna(0.0))
    else:
        out["log1p_disease_hpo_count"] = 0.0

    features = list(BASE_FEATURES)
    if allow_gold_leakage_features and LEAKAGE_FEATURE in out.columns:
        features.append(LEAKAGE_FEATURE)

    relation_col = "mondo_relation_to_gold" if "mondo_relation_to_gold" in out.columns else None
    relations = out[relation_col].map(relation_bucket) if relation_col else pd.Series(["relation_unavailable"] * len(out))
    for value in RELATION_VALUES:
        col = f"relation__{value}"
        out[col] = (relations == value).astype(float)
        features.append(col)

    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").fillna(0.0)
    return out, features


def build_pairwise_training(df: pd.DataFrame, features: list[str], max_negatives_per_case: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for _, group in df.groupby("case_id", sort=False):
        gold = group.loc[group["candidate_id"] == group["gold_id"]]
        if gold.empty:
            continue
        positive = gold.iloc[0][features].to_numpy(dtype=float)
        negatives = group.loc[group["candidate_id"] != group["gold_id"]].head(max_negatives_per_case)
        for _, neg in negatives.iterrows():
            negative = neg[features].to_numpy(dtype=float)
            diff = positive - negative
            x_rows.append(diff)
            y_rows.append(1)
            x_rows.append(-diff)
            y_rows.append(0)
    if not x_rows:
        raise ValueError("No pairwise examples were built. Check that gold appears in train top50 candidates.")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=int)


def score_candidates(df: pd.DataFrame, features: list[str], scaler: StandardScaler, model: LogisticRegression) -> pd.DataFrame:
    x = scaler.transform(df[features].to_numpy(dtype=float))
    coef = model.coef_.reshape(-1)
    scored = df.copy()
    scored["pairwise_score"] = x @ coef
    return scored


def evaluate(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_df in scored.groupby("dataset_name", sort=True):
        ranks = []
        for _, group in dataset_df.groupby("case_id", sort=False):
            ordered = group.sort_values(["pairwise_score", "original_rank"], ascending=[False, True], kind="stable")
            hits = np.flatnonzero((ordered["candidate_id"].to_numpy(str) == ordered["gold_id"].to_numpy(str)))
            ranks.append(int(hits[0] + 1) if len(hits) else 51)
        arr = np.asarray(ranks)
        rows.append(
            {
                "dataset": dataset,
                "num_cases": int(len(arr)),
                "top1": float((arr <= 1).mean()),
                "top3": float((arr <= 3).mean()),
                "top5": float((arr <= 5).mean()),
                "rank_le_50": float((arr <= 50).mean()),
                "median_rank": float(np.median(arr)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    train_raw = load_candidates(args.train_candidates_path)
    val_raw = load_candidates(args.validation_candidates_path)
    train_df, features = feature_frame(train_raw, allow_gold_leakage_features=args.allow_gold_leakage_features)
    val_df, _ = feature_frame(val_raw, allow_gold_leakage_features=args.allow_gold_leakage_features)
    x_train, y_train = build_pairwise_training(train_df, features, args.max_negatives_per_case)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    best: dict[str, Any] | None = None
    validation_rows = []
    for c_value in args.c_grid:
        model = LogisticRegression(C=float(c_value), class_weight="balanced", max_iter=2000, random_state=42)
        model.fit(x_train_scaled, y_train)
        metrics = evaluate(score_candidates(val_df, features, scaler, model))
        all_top1 = float(metrics["top1"].mean()) if not metrics.empty else 0.0
        row = {"C": float(c_value), "validation_macro_top1": all_top1}
        validation_rows.append(row)
        if best is None or all_top1 > float(best["validation_macro_top1"]):
            best = {"model": model, **row}
    if best is None:
        raise RuntimeError("No validation model was selected.")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "linear_pairwise_logistic",
        "features": features,
        "scaler": scaler,
        "model": best["model"],
        "validation_selection": {k: v for k, v in best.items() if k != "model"},
        "protocol": "train candidates for fitting, validation candidates for C selection, test fixed eval only",
        "gold_leakage_features_enabled": bool(args.allow_gold_leakage_features),
    }
    with args.model_path.open("wb") as f:
        pickle.dump(payload, f)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(args.report_dir / "pairwise_reranker_validation_selection.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"model": str(args.model_path.resolve()), "features": features}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
