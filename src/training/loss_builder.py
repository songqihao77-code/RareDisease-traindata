from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_scores_and_targets(scores: torch.Tensor, targets: torch.Tensor) -> None:
    """检查 scores 和 targets 的基础合法性。"""
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


class FullPoolCrossEntropyLoss(nn.Module):
    """最基础的全池 softmax 交叉熵损失。"""

    def __init__(self, temperature: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature 必须大于 0，当前为 {temperature}。")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction 只支持 'mean'、'sum'、'none'，当前为 {reduction!r}。"
            )

        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        # 保证 targets 和 scores 在同一设备上。
        if targets.device != scores.device:
            targets = targets.to(scores.device)

        _validate_scores_and_targets(scores, targets)

        logits = scores / self.temperature
        loss = F.cross_entropy(logits, targets, reduction=self.reduction)

        # 统计信息基于原始 scores，保持直观可解释。
        pred = scores.argmax(dim=1)
        correct = pred == targets
        batch_acc = correct.float().mean()
        target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)

        return {
            "loss": loss,
            "pred": pred,
            "correct": correct,
            "batch_acc": batch_acc,
            "target_scores": target_scores,
        }


def build_loss(
    loss_name: str = "full_pool_ce",
    temperature: float = 1.0,
    reduction: str = "mean",
) -> nn.Module:
    """构建当前项目使用的损失模块。"""
    if loss_name != "full_pool_ce":
        raise ValueError(f"当前只支持 'full_pool_ce'，收到 {loss_name!r}。")

    return FullPoolCrossEntropyLoss(
        temperature=temperature,
        reduction=reduction,
    )


__all__ = ["FullPoolCrossEntropyLoss", "build_loss"]
