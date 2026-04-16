from __future__ import annotations

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


__all__ = ["mine_hard_negatives"]
