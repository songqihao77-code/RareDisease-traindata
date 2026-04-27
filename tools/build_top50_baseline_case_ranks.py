from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DEFAULT_CASE_RANKS = PROJECT_ROOT / "outputs" / "rerank" / "top50_rerank_case_ranks.csv"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "rerank" / "top50_rerank_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HGNN baseline case ranks from exported top-k candidates."
    )
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-case-ranks", type=Path, default=DEFAULT_CASE_RANKS)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args()


def _metric_from_ranks(ranks: pd.Series) -> dict[str, Any]:
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


def _build_case_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "dataset_name", "gold_id", "candidate_id", "original_rank"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing required candidate columns: {sorted(missing)}")

    candidates = candidates.copy()
    candidates["original_rank"] = pd.to_numeric(
        candidates["original_rank"],
        errors="coerce",
    ).astype("Int64")
    rows: list[dict[str, Any]] = []
    for case_id, group in candidates.groupby("case_id", sort=False):
        group = group.sort_values("original_rank", kind="stable")
        dataset_name = str(group["dataset_name"].iloc[0])
        gold_id = str(group["gold_id"].iloc[0])
        max_rank = int(group["original_rank"].max())
        hits = group.loc[group["candidate_id"].astype(str) == gold_id, "original_rank"]
        gold_in_topk = not hits.empty
        rank = int(hits.min()) if gold_in_topk else max_rank + 1
        rows.append(
            {
                "preset": "A_hgnn_only",
                "case_id": str(case_id),
                "dataset_name": dataset_name,
                "gold_id": gold_id,
                "reranked_rank": rank,
                "gold_in_top50": bool(gold_in_topk),
            }
        )
    return pd.DataFrame(rows)


def _build_metrics(case_ranks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, group in case_ranks.groupby("dataset_name", sort=True):
        rows.append(
            {
                "preset": "A_hgnn_only",
                "dataset_name": dataset_name,
                **_metric_from_ranks(group["reranked_rank"]),
            }
        )
    rows.append(
        {
            "preset": "A_hgnn_only",
            "dataset_name": "ALL",
            **_metric_from_ranks(case_ranks["reranked_rank"]),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(
        args.candidates_path,
        dtype={
            "case_id": str,
            "dataset_name": str,
            "gold_id": str,
            "candidate_id": str,
        },
    )
    case_ranks = _build_case_ranks(candidates)
    metrics = _build_metrics(case_ranks)

    args.output_case_ranks.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    case_ranks.to_csv(args.output_case_ranks, index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_metrics, index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "candidates_path": str(args.candidates_path.resolve()),
                "output_case_ranks": str(args.output_case_ranks.resolve()),
                "output_metrics": str(args.output_metrics.resolve()),
                "num_cases": int(len(case_ranks)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
