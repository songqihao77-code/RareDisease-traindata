from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


MISSING_RANK = 10**9


def ordered_unique_labels(values: Iterable[Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            labels.append(text)
    return labels


def rank_for_label_set(
    ranked_indices: Iterable[int],
    labels: Iterable[str],
    disease_to_idx: dict[str, int],
) -> int:
    target_indices = {int(disease_to_idx[label]) for label in labels if label in disease_to_idx}
    if not target_indices:
        return MISSING_RANK
    for rank, idx in enumerate(ranked_indices, start=1):
        if int(idx) in target_indices:
            return int(rank)
    return MISSING_RANK


def rank_for_canonical_label_set(
    ranked_labels: Iterable[str],
    gold_labels: Iterable[str],
    canonicalize,
) -> int:
    target_labels = {canonicalize(label) for label in gold_labels if str(label).strip()}
    target_labels.discard("")
    if not target_labels:
        return MISSING_RANK
    for rank, candidate_label in enumerate(ranked_labels, start=1):
        if canonicalize(candidate_label) in target_labels:
            return int(rank)
    return MISSING_RANK


def compute_rank_metrics(
    ranks: Iterable[int | float],
    ks: tuple[int, ...] = (1, 3, 5, 10, 30, 50),
) -> dict[str, float | int]:
    arr = np.asarray([int(rank) for rank in ranks], dtype=np.int64)
    if arr.size == 0:
        out: dict[str, float | int] = {"n": 0}
        for k in ks:
            out[f"top{k}"] = float("nan")
        out["mean_rank"] = float("nan")
        out["median_rank"] = float("nan")
        out["rank_le_50"] = float("nan")
        return out
    out = {"n": int(arr.size)}
    for k in ks:
        out[f"top{k}"] = float(np.mean(arr <= k))
    out["mean_rank"] = float(np.mean(arr))
    out["median_rank"] = float(np.median(arr))
    out["rank_le_50"] = float(np.mean(arr <= 50))
    return out
