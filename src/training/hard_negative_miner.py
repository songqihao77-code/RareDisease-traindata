from __future__ import annotations

from collections.abc import Mapping

import torch


def mine_hard_negatives(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """屏蔽正类后，返回每个样本分数最高的 top-k 负类。"""
    if scores.ndim != 2:
        raise ValueError(f"scores 必须是 2 维张量，当前 shape={tuple(scores.shape)}。")
    if targets.ndim != 1:
        raise ValueError(f"targets 必须是 1 维张量，当前 shape={tuple(targets.shape)}。")
    if scores.shape[0] != targets.shape[0]:
        raise ValueError("scores 和 targets 的 batch 维必须一致。")

    real_k = min(int(k), max(scores.shape[1] - 1, 0))
    if real_k <= 0:
        return torch.empty((scores.shape[0], 0), dtype=torch.long, device=scores.device)

    if targets.device != scores.device:
        targets = targets.to(scores.device)

    masked_scores = scores.clone()
    masked_scores.scatter_(1, targets.unsqueeze(1), float('-inf'))
    return masked_scores.topk(real_k, dim=1).indices


def _normalize_strategy(strategy: str) -> str:
    normalized = str(strategy or "HN-current").strip()
    aliases = {
        "current": "HN-current",
        "overlap": "HN-overlap",
        "sibling": "HN-sibling",
        "shared_ancestor": "HN-shared-ancestor",
        "above_gold": "HN-above-gold",
        "mixed": "HN-mixed",
    }
    return aliases.get(normalized, normalized)


def _as_long_pool(pool: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if pool.ndim == 1:
        pool = pool.unsqueeze(1)
    if pool.ndim != 2:
        raise ValueError(f"candidate pool must be 1D or 2D, got shape={tuple(pool.shape)}")
    return pool.to(device=device, dtype=torch.long)


def _pool_for_strategy(
    *,
    strategy: str,
    candidate_pools: Mapping[str, torch.Tensor] | None,
) -> list[str]:
    if strategy == "HN-overlap":
        return ["overlap"]
    if strategy == "HN-sibling":
        return ["sibling", "same_parent"]
    if strategy == "HN-shared-ancestor":
        return ["shared_ancestor"]
    if strategy == "HN-above-gold":
        return ["above_gold"]
    if strategy == "HN-mixed":
        return ["current", "overlap", "sibling", "same_parent", "shared_ancestor", "above_gold"]
    if candidate_pools and strategy in candidate_pools:
        return [strategy]
    return ["current"]


def _quota_by_ratio(
    pool_names: list[str],
    k: int,
    sampling_ratios: Mapping[str, float] | None,
) -> dict[str, int]:
    if k <= 0:
        return {}
    if not pool_names:
        return {}
    if not sampling_ratios:
        base = max(k // len(pool_names), 1)
        quotas = {name: base for name in pool_names}
    else:
        total = sum(max(float(sampling_ratios.get(name, 0.0)), 0.0) for name in pool_names)
        if total <= 0:
            return _quota_by_ratio(pool_names, k, None)
        quotas = {
            name: int(round(k * max(float(sampling_ratios.get(name, 0.0)), 0.0) / total))
            for name in pool_names
        }
    while sum(quotas.values()) < k:
        for name in pool_names:
            quotas[name] = quotas.get(name, 0) + 1
            if sum(quotas.values()) >= k:
                break
    while sum(quotas.values()) > k:
        for name in reversed(pool_names):
            if quotas.get(name, 0) > 0:
                quotas[name] -= 1
                if sum(quotas.values()) <= k:
                    break
    return quotas


def _take_unique_candidates(
    *,
    row_candidates: torch.Tensor,
    target: int,
    limit: int,
    num_classes: int,
    used: set[int],
) -> list[int]:
    selected: list[int] = []
    for value in row_candidates.detach().cpu().tolist():
        idx = int(value)
        if idx < 0 or idx >= num_classes or idx == target or idx in used:
            continue
        used.add(idx)
        selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


def mine_configurable_hard_negatives(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    *,
    strategy: str = "HN-current",
    candidate_pools: Mapping[str, torch.Tensor] | None = None,
    sampling_ratios: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Mine hard negatives with optional ontology-aware candidate pools.

    The original score-based miner is preserved as ``HN-current``. Ontology
    pools are optional because the current training loop does not yet build
    per-batch MONDO relation pools; when a requested pool is unavailable this
    function fills the remaining quota with ``HN-current`` candidates.
    """
    current = mine_hard_negatives(scores=scores, targets=targets, k=k)
    strategy = _normalize_strategy(strategy)
    if strategy == "HN-current" or not candidate_pools:
        return current

    pool_names = _pool_for_strategy(strategy=strategy, candidate_pools=candidate_pools)
    available = []
    for name in pool_names:
        if name == "current":
            available.append(name)
        elif name in candidate_pools:
            available.append(name)
    if not available:
        return current

    quotas = _quota_by_ratio(available, int(k), sampling_ratios)
    rows: list[list[int]] = []
    num_classes = int(scores.shape[1])
    device = scores.device
    for row_idx in range(scores.shape[0]):
        target = int(targets[row_idx].detach().cpu().item())
        used: set[int] = set()
        row_selected: list[int] = []
        for name in available:
            quota = int(quotas.get(name, 0))
            if quota <= 0:
                continue
            if name == "current":
                pool_row = current[row_idx]
            else:
                pool = _as_long_pool(candidate_pools[name], device=device)
                if pool.shape[0] != scores.shape[0]:
                    raise ValueError(
                        f"candidate pool {name!r} batch mismatch: {pool.shape[0]} vs {scores.shape[0]}"
                    )
                pool_row = pool[row_idx]
            row_selected.extend(
                _take_unique_candidates(
                    row_candidates=pool_row,
                    target=target,
                    limit=quota,
                    num_classes=num_classes,
                    used=used,
                )
            )
        if len(row_selected) < int(k):
            row_selected.extend(
                _take_unique_candidates(
                    row_candidates=current[row_idx],
                    target=target,
                    limit=int(k) - len(row_selected),
                    num_classes=num_classes,
                    used=used,
                )
            )
        rows.append(row_selected[: int(k)])

    max_width = min(int(k), max(num_classes - 1, 0))
    if max_width <= 0:
        return torch.empty((scores.shape[0], 0), dtype=torch.long, device=device)
    out = torch.full((scores.shape[0], max_width), -1, dtype=torch.long, device=device)
    for row_idx, row_values in enumerate(rows):
        values = row_values[:max_width]
        if values:
            out[row_idx, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    return out


__all__ = ["mine_hard_negatives", "mine_configurable_hard_negatives"]
