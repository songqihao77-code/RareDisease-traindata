"""病例条件化细化模块。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn


def _to_sparse_tensor(matrix, device: torch.device) -> torch.Tensor:
    """把病例-HPO 关联矩阵统一转成 torch sparse COO。"""
    if scipy.sparse.issparse(matrix):
        matrix = matrix.tocoo()
        indices = torch.tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long, device=device)
        values = torch.tensor(matrix.data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, matrix.shape, device=device).coalesce()

    tensor = matrix.float().to(device)
    if tensor.is_sparse:
        return tensor.coalesce()
    return tensor.to_sparse().coalesce()


@dataclass(slots=True)
class RefinedCaseNodeState:
    """保存病例侧细化后的 active edge 表示，并按需转回 dense 张量。

    旧实现会先无差别物化 `[num_hpo, num_case, hidden_dim]` 再逐病例回写。
    训练热路径实际只会访问 active case-HPO 边，因此这里先保存：
    - 全局共享的原始节点表示 `base_node_repr`
    - active 边索引 `edge_index`
    - active 边权重 `edge_weight`
    - active 边细化后的表示 `edge_repr`

    只有在调试/测试明确需要 3D dense 张量时，才通过 `to_dense()` 再物化。
    这样不会改变数学定义，只是把中间表示从“总是 dense”改成“按需 dense”。
    """

    base_node_repr: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_repr: torch.Tensor
    num_case: int
    _dense_cache: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        num_hpo, hidden_dim = self.base_node_repr.shape
        return (int(num_hpo), int(self.num_case), int(hidden_dim))

    def to_dense(self) -> torch.Tensor:
        """按需恢复旧契约中的 3D dense 张量。"""
        if self._dense_cache is None:
            dense = self.base_node_repr.unsqueeze(1).expand(-1, self.num_case, -1).clone()
            if self.edge_repr.numel() > 0:
                dense[self.edge_index[0], self.edge_index[1], :] = self.edge_repr
            self._dense_cache = dense
        return self._dense_cache


class CaseConditionedRefiner(nn.Module):
    """按当前病例上下文轻量细化 HPO 节点表示。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        mlp_hidden_dim: int = 128,
        residual: float = 0.7,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 <= residual <= 1.0:
            raise ValueError(f"residual 必须位于 [0, 1]，当前为 {residual}")

        self.hidden_dim = int(hidden_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.residual = float(residual)
        self.eps = float(eps)

        # 先把病例上下文投影到节点表示空间，再由门控决定更新强度。
        self.ctx_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, node_repr: torch.Tensor, H_case) -> RefinedCaseNodeState:
        """
        输入：
        - node_repr: [num_hpo, hidden_dim]
        - H_case: [num_hpo, num_case]

        输出：
        - RefinedCaseNodeState

        数学含义与旧实现一致：
        - 每个病例先用其 active HPO 的加权平均得到 `case_ctx`
        - 再用 `z`、`ctx`、`z*ctx`、`|z-ctx|` 过门控 MLP
        - 最后做残差混合和 LayerNorm

        优化点：
        - 去掉逐病例 Python 循环
        - 去掉训练热路径里无差别物化整块 3D dense 张量
        - 改为按 active edge 一次性向量化计算
        """
        if node_repr.ndim != 2:
            raise ValueError(f"node_repr 必须是二维张量，当前 shape={tuple(node_repr.shape)}")

        num_hpo, hidden_dim = node_repr.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"node_repr hidden_dim 与 refiner 配置不一致，"
                f"得到 {hidden_dim}，期望 {self.hidden_dim}"
            )

        H_sparse = _to_sparse_tensor(H_case, node_repr.device)
        if H_sparse.shape[0] != num_hpo:
            raise ValueError(
                f"H_case 行数必须与 node_repr 的 num_hpo 一致，"
                f"得到 {H_sparse.shape[0]} 和 {num_hpo}"
            )

        num_case = int(H_sparse.shape[1])
        indices = H_sparse.indices()
        values = H_sparse.values().to(dtype=node_repr.dtype)

        if num_case == 0 or H_sparse._nnz() == 0:
            return RefinedCaseNodeState(
                base_node_repr=node_repr,
                edge_index=indices,
                edge_weight=values,
                edge_repr=node_repr.new_zeros((0, hidden_dim)),
                num_case=num_case,
            )

        hpo_idx = indices[0]
        case_idx = indices[1]

        # active edge 上的原始节点表示，形状 [E, hidden_dim]
        edge_node_repr = node_repr.index_select(0, hpo_idx)
        edge_weight = values.unsqueeze(1)

        # 先按病例聚合上下文，再回填到各条 active edge。
        case_sum = node_repr.new_zeros((num_case, hidden_dim))
        case_sum.index_add_(0, case_idx, edge_node_repr * edge_weight)

        case_weight_sum = node_repr.new_zeros((num_case,))
        case_weight_sum.index_add_(0, case_idx, values)
        case_ctx = case_sum / case_weight_sum.unsqueeze(1).clamp(min=self.eps)

        edge_ctx = case_ctx.index_select(0, case_idx)
        gate_input = torch.cat(
            [
                edge_node_repr,
                edge_ctx,
                edge_node_repr * edge_ctx,
                torch.abs(edge_node_repr - edge_ctx),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_input))

        case_ctx_update = self.ctx_proj(case_ctx)
        edge_ctx_update = case_ctx_update.index_select(0, case_idx)
        updated_edge = (1.0 - gate) * edge_node_repr + gate * edge_ctx_update
        refined_edge_repr = self.layer_norm(
            self.residual * edge_node_repr + (1.0 - self.residual) * updated_edge
        )

        return RefinedCaseNodeState(
            base_node_repr=node_repr,
            edge_index=indices,
            edge_weight=values,
            edge_repr=refined_edge_repr,
            num_case=num_case,
        )


__all__ = ["CaseConditionedRefiner", "RefinedCaseNodeState"]
