"""
Hyperedge Readout
接收 hgnn_encoder.py 输出的节点表示 Z [N_hpo, d]，
以及 build_hypergraph.py 输出的 H_case / H_disease，
聚合出 case_repr [N_case, d] 和 disease_repr [N_disease, d]。
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse


def _hmat(M, device) -> torch.Tensor:
    """scipy sparse 或稠密 Tensor → torch sparse COO（float32）。"""
    if scipy.sparse.issparse(M):
        M = M.tocoo()
        idx = torch.tensor(np.vstack([M.row, M.col]), dtype=torch.long)
        val = torch.tensor(M.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, M.shape, device=device).coalesce()
    M = M.float().to(device)
    return M.to_sparse().coalesce() if not M.is_sparse else M.coalesce()


def _spmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """统一的矩阵乘法：稀疏 A @ 稠密 B，或稠密 A @ 稠密 B。"""
    if A.is_sparse:
        return torch.sparse.mm(A.coalesce(), B)
    return A @ B


class HyperedgeReadout(nn.Module):
    """无可学习参数，纯聚合操作。"""

    def build_case_repr(self, Z: torch.Tensor, H_case) -> torch.Tensor:
        """
        病例平均 readout：
          case_repr = H_case^T @ Z / case_degree
          case_degree[j] = H_case 第 j 列之和（该病例包含的 HPO 数目）
        """
        H = _hmat(H_case, Z.device)                          # [N_hpo, N_case]
        out = _spmm(H.t().coalesce(), Z)                      # [N_case, d]
        # 列和：每个病例的 HPO 覆盖度；clamp 防除零
        deg = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1e-6)  # [N_case]
        return out / deg.unsqueeze(1)                         # [N_case, d]

    def build_disease_repr(self, Z: torch.Tensor, H_disease) -> torch.Tensor:
        """
        疾病加权 readout：
          disease_repr = H_disease^T @ Z
          保留 H_disease 原始权重语义，不做额外归一化。
        """
        H = _hmat(H_disease, Z.device)                       # [N_hpo, N_disease]
        return _spmm(H.t().coalesce(), Z)                     # [N_disease, d]

    def forward(self, Z: torch.Tensor, H_case, H_disease) -> dict:
        return {
            "case_repr":    self.build_case_repr(Z, H_case),
            "disease_repr": self.build_disease_repr(Z, H_disease),
        }
