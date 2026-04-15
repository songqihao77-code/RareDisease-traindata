from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_scores_and_targets(scores: torch.Tensor, targets: torch.Tensor) -> None:
    """检查 scores 和 targets 的基本合法性。"""
    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"scores 必须是 torch.Tensor，当前收到 {type(scores).__name__}。")
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"targets 必须是 torch.Tensor，当前收到 {type(targets).__name__}。")

    if scores.ndim != 2:
        raise ValueError(f"scores 必须是 2 维张量，当前 shape={tuple(scores.shape)}。")
    if targets.ndim != 1:
        raise ValueError(f"targets 必须是 1 维张量，当前 shape={tuple(targets.shape)}。")

    batch_size, num_disease = scores.shape
    if targets.shape[0] != batch_size:
        raise ValueError(
            f"scores 和 targets 的 batch 维必须一致，当前分别为 {batch_size} 和 {targets.shape[0]}。"
        )

    if targets.dtype != torch.long:
        raise TypeError(f"targets.dtype 必须是 torch.long/int64，当前为 {targets.dtype}。")

    if num_disease <= 0:
        raise ValueError("scores 的 disease 维长度必须大于 0。")

    if targets.numel() == 0:
        return

    min_target = int(targets.min().item())
    max_target = int(targets.max().item())
    if min_target < 0 or max_target >= num_disease:
        raise ValueError(
            f"targets 存在越界索引，合法范围应为 [0, {num_disease - 1}]，"
            f"当前最小值为 {min_target}，最大值为 {max_target}。"
        )


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction 只支持 'mean'、'sum'、'none'，当前为 {reduction!r}。")


def _compute_hard_rank_loss(
    scaled_scores: torch.Tensor,
    targets: torch.Tensor,
    hard_neg_indices: torch.Tensor,
    margin: float,
    top_m: int,
    reduction: str,
) -> torch.Tensor:
    if hard_neg_indices.ndim != 2:
        raise ValueError(
            f"hard_neg_indices 必须是 2 维张量，当前 shape={tuple(hard_neg_indices.shape)}。"
        )
    if hard_neg_indices.shape[0] != scaled_scores.shape[0]:
        raise ValueError("hard_neg_indices 和 scores 的 batch 维必须一致。")

    if hard_neg_indices.device != scaled_scores.device:
        hard_neg_indices = hard_neg_indices.to(scaled_scores.device)
    if hard_neg_indices.dtype != torch.long:
        hard_neg_indices = hard_neg_indices.long()

    if hard_neg_indices.numel() == 0 or top_m <= 0:
        zeros = scaled_scores.new_zeros(scaled_scores.shape[0])
        return _reduce_loss(zeros, reduction)

    neg_indices = hard_neg_indices[:, : min(top_m, hard_neg_indices.shape[1])]
    pos_scores = scaled_scores.gather(1, targets.unsqueeze(1))
    neg_scores = scaled_scores.gather(1, neg_indices)

    violations = F.relu(float(margin) - pos_scores + neg_scores)
    active_count = (violations > 0).sum(dim=1)
    sample_loss = violations.sum(dim=1) / active_count.clamp_min(1).to(violations.dtype)
    return _reduce_loss(sample_loss, reduction)


def compute_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    hard_neg_indices: torch.Tensor | None = None,
    tau: float = 1.0,
    margin: float = 0.1,
    hard_weight: float = 0.5,
    top_m: int = 3,
    poly_epsilon: float = 2.0,
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """先算 full-pool CE，再按需叠加 hard loss。"""
    if tau <= 0:
        raise ValueError(f"tau 必须大于 0，当前为 {tau}。")

    if targets.device != scores.device:
        targets = targets.to(scores.device)

    _validate_scores_and_targets(scores, targets)

    scaled_scores = scores / float(tau)
    
    ce_loss_none = F.cross_entropy(scaled_scores, targets, reduction="none")
    if poly_epsilon > 0.0:
        pt = torch.exp(-ce_loss_none)
        poly_loss = ce_loss_none + float(poly_epsilon) * (1.0 - pt)
    else:
        poly_loss = ce_loss_none
        
    ce_loss = _reduce_loss(poly_loss, reduction)

    if hard_neg_indices is None or hard_weight <= 0 or top_m <= 0:
        if reduction == "none":
            hard_rank_loss = ce_loss.new_zeros(ce_loss.shape)
        else:
            hard_rank_loss = ce_loss.new_zeros(())
    else:
        hard_rank_loss = _compute_hard_rank_loss(
            scaled_scores=scaled_scores,
            targets=targets,
            hard_neg_indices=hard_neg_indices,
            margin=margin,
            top_m=top_m,
            reduction=reduction,
        )

    loss = ce_loss + float(hard_weight) * hard_rank_loss

    pred = scores.argmax(dim=1)
    correct = pred == targets
    batch_acc = correct.float().mean()
    target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)

    return {
        "loss": loss,
        "ce_loss": ce_loss,
        "hard_rank_loss": hard_rank_loss,
        "pred": pred,
        "correct": correct,
        "batch_acc": batch_acc,
        "target_scores": target_scores,
    }


class FullPoolCrossEntropyLoss(nn.Module):
    """全池 softmax 交叉熵，可叠加 hard negative loss。"""

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "mean",
        hard_weight: float = 0.5,
        margin: float = 0.1,
        top_m: int = 3,
        poly_epsilon: float = 2.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature 必须大于 0，当前为 {temperature}。")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction 只支持 'mean'、'sum'、'none'，当前为 {reduction!r}。"
            )

        self.temperature = float(temperature)
        self.reduction = reduction
        self.hard_weight = float(hard_weight)
        self.margin = float(margin)
        self.top_m = int(top_m)
        self.poly_epsilon = float(poly_epsilon)

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        hard_neg_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return compute_loss(
            scores=scores,
            targets=targets,
            hard_neg_indices=hard_neg_indices,
            tau=self.temperature,
            margin=self.margin,
            hard_weight=self.hard_weight,
            top_m=self.top_m,
            poly_epsilon=self.poly_epsilon,
            reduction=self.reduction,
        )


def build_loss(
    loss_name: str = "full_pool_ce",
    temperature: float = 1.0,
    reduction: str = "mean",
    hard_weight: float = 0.5,
    margin: float = 0.1,
    top_m: int = 3,
    poly_epsilon: float = 2.0,
) -> nn.Module:
    """构建当前项目使用的损失模块。"""
    if loss_name != "full_pool_ce":
        raise ValueError(f"当前只支持 'full_pool_ce'，收到 {loss_name!r}。")

    return FullPoolCrossEntropyLoss(
        temperature=temperature,
        reduction=reduction,
        hard_weight=hard_weight,
        margin=margin,
        top_m=top_m,
        poly_epsilon=poly_epsilon,
    )


__all__ = ["FullPoolCrossEntropyLoss", "build_loss", "compute_loss"]
