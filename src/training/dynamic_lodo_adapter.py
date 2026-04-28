from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


DEFAULT_LODO_DATASETS = ("DDD", "MIMIC-IV-Rare", "HMS", "LIRICAL", "MME")
MIMIC_CANONICAL_NAME = "MIMIC-IV-Rare"
EVIDENCE_COLUMNS = (
    "hgnn_score",
    "ic_weighted_overlap",
    "exact_overlap",
    "semantic_ic_overlap",
    "case_coverage",
)
META_FEATURE_COLUMNS = (
    "case_hpo_count_proxy",
    "mean_ic_overlap_top50",
    "max_ic_overlap_top50",
    "max_semantic_ic_overlap_top50",
    "mean_semantic_ic_overlap_top50",
    "hgnn_margin",
    "hgnn_score_std_top50",
)


@dataclass(slots=True)
class DynamicLodoCandidateData:
    case_ids: np.ndarray
    dataset_names: np.ndarray
    gold_ids: np.ndarray
    candidate_ids: np.ndarray
    original_rank: np.ndarray
    gold_mask: np.ndarray
    x_meta: np.ndarray
    x_evidence: np.ndarray
    top_k: int
    meta_feature_names: tuple[str, ...]
    evidence_feature_names: tuple[str, ...]
    source_frame: pd.DataFrame

    @property
    def num_cases(self) -> int:
        return int(self.case_ids.shape[0])


def canonical_dataset_name(dataset_name: str, mimic_aliases: Sequence[str] = ()) -> str:
    name = str(dataset_name)
    if name == MIMIC_CANONICAL_NAME:
        return MIMIC_CANONICAL_NAME
    lowered = name.lower()
    if name in set(map(str, mimic_aliases)) or lowered.startswith("mimic"):
        return MIMIC_CANONICAL_NAME
    return name


def load_dynamic_lodo_candidates(
    path: Path,
    *,
    mimic_aliases: Sequence[str] = (),
    include_datasets: Sequence[str] = DEFAULT_LODO_DATASETS,
) -> DynamicLodoCandidateData:
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    required = {
        "case_id",
        "dataset_name",
        "gold_id",
        "candidate_id",
        "original_rank",
        "shared_hpo_count",
        "hgnn_margin",
        *EVIDENCE_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required dynamic LODO columns: {sorted(missing)}")

    numeric_columns = sorted(required - {"case_id", "dataset_name", "gold_id", "candidate_id"})
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    include_set = {str(name) for name in include_datasets}
    df["dynamic_dataset"] = df["dataset_name"].map(lambda value: canonical_dataset_name(value, mimic_aliases))
    df = df.loc[df["dynamic_dataset"].isin(include_set)].copy()
    if df.empty:
        raise ValueError(f"No rows matched dynamic LODO datasets {sorted(include_set)} in {path}")

    df = df.sort_values(["case_id", "original_rank"], kind="stable").reset_index(drop=True)
    counts = df.groupby("case_id", sort=False)["candidate_id"].size()
    if counts.nunique() != 1:
        raise ValueError("Dynamic LODO expects every retained case to have the same top-k candidate count.")
    top_k = int(counts.iloc[0])

    case_meta = (
        df.groupby("case_id", sort=False)[["dynamic_dataset", "gold_id"]]
        .first()
        .reset_index()
        .rename(columns={"dynamic_dataset": "dataset_name"})
    )
    num_cases = len(case_meta)
    shape = (num_cases, top_k)

    evidence_raw = np.stack(
        [df[column].to_numpy(dtype=float).reshape(shape) for column in EVIDENCE_COLUMNS],
        axis=-1,
    )
    x_evidence = minmax_normalize_evidence_by_case(evidence_raw)

    original_rank = df["original_rank"].to_numpy(dtype=int).reshape(shape)
    candidate_ids = df["candidate_id"].to_numpy(dtype=str).reshape(shape)
    gold_ids_per_row = df["gold_id"].to_numpy(dtype=str).reshape(shape)
    gold_mask = candidate_ids == gold_ids_per_row

    x_meta = build_meta_feature_matrix(df, shape=shape)

    return DynamicLodoCandidateData(
        case_ids=case_meta["case_id"].to_numpy(dtype=str),
        dataset_names=case_meta["dataset_name"].to_numpy(dtype=str),
        gold_ids=case_meta["gold_id"].to_numpy(dtype=str),
        candidate_ids=candidate_ids,
        original_rank=original_rank,
        gold_mask=gold_mask,
        x_meta=x_meta,
        x_evidence=x_evidence,
        top_k=top_k,
        meta_feature_names=META_FEATURE_COLUMNS,
        evidence_feature_names=EVIDENCE_COLUMNS,
        source_frame=df,
    )


def minmax_normalize_evidence_by_case(values: np.ndarray) -> np.ndarray:
    mins = values.min(axis=1, keepdims=True)
    maxs = values.max(axis=1, keepdims=True)
    denom = maxs - mins
    return np.divide(values - mins, denom, out=np.zeros_like(values, dtype=float), where=denom > 0)


def build_meta_feature_matrix(df: pd.DataFrame, *, shape: tuple[int, int]) -> np.ndarray:
    shared = df["shared_hpo_count"].to_numpy(dtype=float).reshape(shape)
    coverage = df["case_coverage"].to_numpy(dtype=float).reshape(shape)
    inferred_hpo_count = np.divide(shared, coverage, out=np.zeros_like(shared), where=coverage > 0)
    case_hpo_count_proxy = np.max(inferred_hpo_count, axis=1)

    ic = df["ic_weighted_overlap"].to_numpy(dtype=float).reshape(shape)
    semantic = df["semantic_ic_overlap"].to_numpy(dtype=float).reshape(shape)
    hgnn = df["hgnn_score"].to_numpy(dtype=float).reshape(shape)
    hgnn_margin = df["hgnn_margin"].to_numpy(dtype=float).reshape(shape)[:, 0]

    return np.column_stack(
        [
            case_hpo_count_proxy,
            ic.mean(axis=1),
            ic.max(axis=1),
            semantic.max(axis=1),
            semantic.mean(axis=1),
            hgnn_margin,
            hgnn.std(axis=1),
        ]
    ).astype(np.float32)


def standardize_meta_features(
    x_meta: np.ndarray,
    train_indices: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_meta = x_meta[train_indices]
    mean = train_meta.mean(axis=0, keepdims=True)
    std = train_meta.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((x_meta - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def build_pairwise_tensor_dataset(
    data: DynamicLodoCandidateData,
    case_indices: np.ndarray,
    *,
    x_meta: np.ndarray,
    max_negatives_per_case: int = 10,
) -> TensorDataset:
    if max_negatives_per_case < 1:
        raise ValueError("max_negatives_per_case must be positive.")

    meta_rows: list[np.ndarray] = []
    pos_rows: list[np.ndarray] = []
    neg_rows: list[np.ndarray] = []

    for case_idx in case_indices:
        gold_positions = np.flatnonzero(data.gold_mask[case_idx])
        if gold_positions.size == 0:
            continue
        pos_idx = int(gold_positions[0])
        negative_positions = [
            int(idx)
            for idx in np.argsort(data.original_rank[case_idx], kind="stable")
            if idx != pos_idx
        ][:max_negatives_per_case]
        if not negative_positions:
            continue
        while len(negative_positions) < max_negatives_per_case:
            negative_positions.append(negative_positions[-1])

        meta_rows.append(x_meta[case_idx])
        pos_rows.append(data.x_evidence[case_idx, pos_idx])
        neg_rows.append(data.x_evidence[case_idx, negative_positions])

    if not meta_rows:
        raise ValueError("No pairwise examples were built. Check whether gold diseases appear in Top-50.")

    return TensorDataset(
        torch.tensor(np.stack(meta_rows), dtype=torch.float32),
        torch.tensor(np.stack(pos_rows), dtype=torch.float32),
        torch.tensor(np.stack(neg_rows), dtype=torch.float32),
    )


def hgnn_baseline_ranks(data: DynamicLodoCandidateData) -> np.ndarray:
    ranks = np.full(data.num_cases, data.top_k + 1, dtype=int)
    for row_idx in range(data.num_cases):
        gold_positions = np.flatnonzero(data.gold_mask[row_idx])
        if gold_positions.size:
            ranks[row_idx] = int(data.original_rank[row_idx, gold_positions[0]])
    return ranks


def ranks_from_candidate_scores(data: DynamicLodoCandidateData, scores: np.ndarray) -> np.ndarray:
    if scores.shape != data.original_rank.shape:
        raise ValueError(f"scores shape {scores.shape} does not match candidate shape {data.original_rank.shape}")
    ranks = np.full(data.num_cases, data.top_k + 1, dtype=int)
    for row_idx in range(data.num_cases):
        order = np.lexsort((data.original_rank[row_idx], -scores[row_idx]))
        sorted_gold = data.gold_mask[row_idx, order]
        hit_positions = np.flatnonzero(sorted_gold)
        if hit_positions.size:
            ranks[row_idx] = int(hit_positions[0] + 1)
    return ranks


def metrics_by_dataset(
    data: DynamicLodoCandidateData,
    ranks: np.ndarray,
    *,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name in DEFAULT_LODO_DATASETS:
        mask = data.dataset_names == dataset_name
        if not mask.any():
            continue
        rows.append({"dataset": dataset_name, "method": method, **rank_metrics(ranks[mask])})
    rows.append({"dataset": "ALL", "method": method, **rank_metrics(ranks)})
    return pd.DataFrame(rows)


def rank_metrics(ranks: np.ndarray) -> dict[str, object]:
    arr = np.asarray(ranks, dtype=int)
    if arr.size == 0:
        return {"cases": 0, "top1": np.nan, "top3": np.nan, "top5": np.nan, "recall_at_50": np.nan}
    return {
        "cases": int(arr.size),
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "recall_at_50": float(np.mean(arr <= 50)),
    }
