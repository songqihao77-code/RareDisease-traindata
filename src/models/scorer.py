"""
Cosine Scorer
接收 readout.py 输出的 case_repr [B, d] 和 disease_repr [M, d]，
计算余弦相似度分数矩阵 scores [B, M]。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineScorer(nn.Module):
    """无可学习参数，纯余弦相似度打分。"""

    def __init__(self, **kwargs) -> None:
        # 统一配置入口会透传 hidden_dim 等参数，这里直接忽略即可。
        super().__init__()

    def forward(
        self,
        case_repr: torch.Tensor,
        disease_repr: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        del targets
        c = F.normalize(case_repr, p=2, dim=1)
        d = F.normalize(disease_repr, p=2, dim=1)
        scores = c @ d.t()
        return {"scores": scores}
