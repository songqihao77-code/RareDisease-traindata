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

    def forward(self, case_repr: torch.Tensor, disease_repr: torch.Tensor) -> dict:
        # L2 归一化（clamp 防止零向量除零）
        c = F.normalize(case_repr,    p=2, dim=1)   # [B, d]
        d = F.normalize(disease_repr, p=2, dim=1)   # [M, d]
        # scores[i, j] = cos(r_c_i, r_d_j)
        scores = c @ d.t()                           # [B, M]
        return {"scores": scores}
