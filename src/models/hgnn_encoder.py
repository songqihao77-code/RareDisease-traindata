"""
HGNN 编码器
接收统一超图 H（来自 src/data/build_hypergraph.py），
完成两层 HGNN 传播，输出 HPO 节点表示 Z。

传播公式：P = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
  第一层：X^(1) = ReLU(P X^(0) Θ^(0))
  第二层：X^(2) = P X^(1) Θ^(1)
  输出：  Z = X^(2)，形状 [num_hpo, hidden_dim]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse


def _to_sparse_tensor(H_scipy, device) -> torch.Tensor:
    """scipy sparse → torch sparse COO tensor，保持稀疏格式，不稠密化。"""
    H_coo = H_scipy.tocoo()
    idx = torch.tensor(np.vstack([H_coo.row, H_coo.col]), dtype=torch.long)
    val = torch.tensor(H_coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, H_coo.shape, device=device).coalesce()


class HGNNEncoder(nn.Module):
    """
    两层 HGNN 编码器。

    参数：
      num_hpo    : HPO 节点数量（由上游 build_hypergraph.py 的 len(hpo_to_idx) 给出）
      hidden_dim : 隐层维度，默认 128

    输入：
      H : 统一超图关联矩阵，形状 [num_hpo, num_edges]
          可为 scipy.sparse 矩阵或 torch.Tensor（float）
          前半列为病例超边（二值），后半列为疾病超边（加权），统一处理

    输出：
      Z : HPO 节点表示矩阵，形状 [num_hpo, hidden_dim]
    """

    def __init__(self, num_hpo: int, hidden_dim: int = 128):
        super().__init__()
        # 可学习初始节点嵌入 X^(0)，形状 [num_hpo, hidden_dim]
        self.X0     = nn.Parameter(torch.randn(num_hpo, hidden_dim) * 0.01)
        # 两层线性变换 Θ^(0) 和 Θ^(1)，均无偏置
        self.theta0 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _propagate(self, H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        高效计算 P @ X，不显式构建 N×N 传播矩阵。
        当前版本 W = I（单位矩阵），注释保留以说明此处是简化设定。

        步骤：
          1. dv^{-1/2} * X
          2. H^T @ (步骤1)
          3. de^{-1} * (步骤2)
          4. H @ (步骤3)
          5. dv^{-1/2} * (步骤4)
        """
        # 边度 δ(e) = Σ_v H_{ve}，形状 [E]；clamp 防止除零
        d_e = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1e-6)
        # 点度 d(v) = Σ_e H_{ve}（W=I），形状 [N]；clamp 防止除零
        d_v = torch.sparse.sum(H, dim=1).to_dense().clamp(min=1e-6)

        dv_inv_sqrt = d_v.pow(-0.5).unsqueeze(1)   # [N, 1]
        de_inv      = d_e.pow(-1.0).unsqueeze(1)   # [E, 1]

        Y = dv_inv_sqrt * X                    # D_v^{-1/2} X,              [N, d]
        Y = torch.sparse.mm(H.t(), Y)          # H^T D_v^{-1/2} X,          [E, d]
        Y = de_inv * Y                         # D_e^{-1} H^T D_v^{-1/2} X, [E, d]
        Y = torch.sparse.mm(H, Y)              # H D_e^{-1} H^T D_v^{-1/2} X,[N, d]
        Y = dv_inv_sqrt * Y                    # D_v^{-1/2} (...),           [N, d]
        return Y

    def forward(self, H) -> torch.Tensor:
        # scipy sparse → torch sparse COO，保持稀疏格式
        if scipy.sparse.issparse(H):
            H = _to_sparse_tensor(H, self.X0.device)
        else:
            H = H.float().to(self.X0.device)
            if not H.is_sparse:
                H = H.to_sparse().coalesce()

        # 第一层：X^(1) = ReLU(P X^(0) Θ^(0))
        X1 = F.relu(self.theta0(self._propagate(H, self.X0)))

        # 第二层：Z = X^(2) = P X^(1) Θ^(1)
        Z = self.theta1(self._propagate(H, X1))

        return Z  # [num_hpo, hidden_dim]
