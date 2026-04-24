"""
Hyperedge Readout

接收 encoder 输出的节点表示 `Z [num_hpo, hidden_dim]`，
以及 `H_case / H_disease`，构造：
- case_repr [num_case, hidden_dim]
- disease_repr [num_disease, hidden_dim]
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn

from src.models.case_refiner import RefinedCaseNodeState


def _hmat(M, device) -> torch.Tensor:
    """scipy sparse 或 torch Tensor -> torch sparse COO(float32)。"""
    if scipy.sparse.issparse(M):
        M = M.tocoo()
        idx = torch.tensor(np.vstack([M.row, M.col]), dtype=torch.long)
        val = torch.tensor(M.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, M.shape, device=device).coalesce()
    M = M.float().to(device)
    return M.to_sparse().coalesce() if not M.is_sparse else M.coalesce()


def _spmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """统一矩阵乘法：支持 sparse @ dense。"""
    if A.is_sparse:
        return torch.sparse.mm(A.coalesce(), B)
    return A @ B


class HyperedgeReadout(nn.Module):
    """病例侧 attention readout + 疾病侧线性 readout。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        attn_hidden_dim: int | None = None,
        attn_dropout: float = 0.1,
        context_mode: str = "leave_one_out",
        residual_uniform: float = 0.2,
        return_attention: bool = False,
        attn_prior_mode: str = "none",
        attn_prior_beta: float = 0.0,
        attn_prior_learnable: bool = False,
        attn_prior_normalize: str = "center",
        attn_prior_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        if context_mode != "leave_one_out":
            raise ValueError(
                f"context_mode 当前只支持 'leave_one_out'，当前为 {context_mode!r}"
            )

        attn_prior_mode = str(attn_prior_mode)
        attn_prior_normalize = str(attn_prior_normalize)
        attn_prior_eps = float(attn_prior_eps)
        if attn_prior_mode not in {"none", "edge_weight_log"}:
            raise ValueError(
                f"attn_prior_mode 只支持 'none' 或 'edge_weight_log'，当前为 {attn_prior_mode!r}"
            )
        if attn_prior_normalize not in {"none", "center", "zscore"}:
            raise ValueError(
                "attn_prior_normalize 只支持 'none'、'center' 或 'zscore'，"
                f"当前为 {attn_prior_normalize!r}"
            )
        if attn_prior_eps <= 0.0:
            raise ValueError(f"attn_prior_eps 必须大于 0，当前为 {attn_prior_eps}")

        self.hidden_dim = hidden_dim
        self.attn_hidden_dim = attn_hidden_dim if attn_hidden_dim is not None else hidden_dim
        self.context_mode = context_mode
        self.residual_uniform = residual_uniform
        self.return_attention = return_attention
        self.attn_prior_mode = attn_prior_mode
        self.attn_prior_learnable = bool(attn_prior_learnable)
        self.attn_prior_normalize = attn_prior_normalize
        self.attn_prior_eps = float(attn_prior_eps)
        if self.attn_prior_learnable:
            self.attn_prior_beta = nn.Parameter(torch.tensor(float(attn_prior_beta), dtype=torch.float32))
        else:
            self.attn_prior_beta = float(attn_prior_beta)

        self.attn_mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.attn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(self.attn_hidden_dim, 1),
        )

    def _apply_attention_prior(
        self,
        logits_1d: torch.Tensor,
        edge_weight: torch.Tensor,
        case_idx: torch.Tensor,
        num_case: int,
    ) -> torch.Tensor:
        if self.attn_prior_mode == "none":
            return logits_1d
        if self.attn_prior_mode == "edge_weight_log":
            if not self.attn_prior_learnable and float(self.attn_prior_beta) == 0.0:
                return logits_1d

            prior = torch.log(
                edge_weight.to(device=logits_1d.device, dtype=logits_1d.dtype).clamp_min(0.0)
                + float(self.attn_prior_eps)
            )
            if self.attn_prior_normalize in {"center", "zscore"}:
                counts = logits_1d.new_zeros((num_case,))
                counts.scatter_add_(0, case_idx, logits_1d.new_ones(logits_1d.shape))
                counts = counts.clamp_min(1.0)

                prior_sum = logits_1d.new_zeros((num_case,))
                prior_sum.scatter_add_(0, case_idx, prior)
                prior_mean = prior_sum / counts
                prior = prior - prior_mean.index_select(0, case_idx)

                if self.attn_prior_normalize == "zscore":
                    sq_sum = logits_1d.new_zeros((num_case,))
                    sq_sum.scatter_add_(0, case_idx, prior.square())
                    std = torch.sqrt(sq_sum / counts + float(self.attn_prior_eps))
                    prior = prior / std.index_select(0, case_idx)

            beta = self.attn_prior_beta
            if isinstance(beta, torch.Tensor):
                beta_value = beta.to(device=logits_1d.device, dtype=logits_1d.dtype)
            else:
                beta_value = logits_1d.new_tensor(float(beta))
            return logits_1d + beta_value * prior

        raise ValueError(f"未知 attn_prior_mode: {self.attn_prior_mode!r}")

    def _build_case_repr_from_edges(
        self,
        edge_repr: torch.Tensor,
        case_idx: torch.Tensor,
        edge_weight: torch.Tensor,
        num_case: int,
        edge_index: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """按 active edge 批量构造病例表示。

        这里保留原始 leave-one-out attention 的数学定义，只把实现改成
        “按 edge 一次性向量化”，避免 refined 分支里的逐病例 Python 循环。
        """
        if edge_repr.ndim != 2:
            raise ValueError(f"edge_repr 必须是二维张量，当前 shape={tuple(edge_repr.shape)}")
        if case_idx.ndim != 1:
            raise ValueError(f"case_idx 必须是一维张量，当前 shape={tuple(case_idx.shape)}")
        if edge_weight.ndim != 1:
            raise ValueError(f"edge_weight 必须是一维张量，当前 shape={tuple(edge_weight.shape)}")
        if edge_repr.shape[0] != case_idx.shape[0] or edge_repr.shape[0] != edge_weight.shape[0]:
            raise ValueError("edge_repr、case_idx、edge_weight 的边数维度不一致。")

        hidden_dim = edge_repr.shape[1]
        case_repr = edge_repr.new_zeros((num_case, hidden_dim))
        if edge_repr.shape[0] == 0:
            empty_extra = {
                "edge_attention": edge_repr.new_zeros((0,)),
                "case_context": edge_repr.new_zeros((0, hidden_dim)),
                "edge_index": edge_index,
            }
            return (case_repr, empty_extra) if self.return_attention else case_repr

        eps = 1e-6
        weighted_edge_repr = edge_repr * edge_weight.unsqueeze(1)

        case_deg = edge_weight.new_zeros((num_case,))
        case_deg.scatter_add_(0, case_idx, edge_weight)

        case_sum = edge_repr.new_zeros((num_case, hidden_dim))
        case_sum.index_add_(0, case_idx, weighted_edge_repr)
        base_case_repr = case_sum / case_deg.unsqueeze(1).clamp(min=eps)

        edge_case_deg = case_deg.index_select(0, case_idx).unsqueeze(1)
        edge_weight_2d = edge_weight.unsqueeze(1)
        remaining_deg = edge_case_deg - edge_weight_2d

        ctx_loocv = (case_sum.index_select(0, case_idx) - weighted_edge_repr) / remaining_deg.clamp(
            min=1e-8
        )
        fallback_mask = (remaining_deg <= eps).float()
        ctx_edge = fallback_mask * base_case_repr.index_select(0, case_idx) + (1.0 - fallback_mask) * ctx_loocv

        feat_edge = torch.cat(
            [
                edge_repr,
                ctx_edge,
                edge_repr * ctx_edge,
                torch.abs(edge_repr - ctx_edge),
            ],
            dim=-1,
        )
        logits_1d = self.attn_mlp(feat_edge).squeeze(-1)
        logits_1d = self._apply_attention_prior(
            logits_1d=logits_1d,
            edge_weight=edge_weight,
            case_idx=case_idx,
            num_case=num_case,
        )

        group_max = logits_1d.new_full((num_case,), float("-inf"))
        group_max.scatter_reduce_(0, case_idx, logits_1d, reduce="amax", include_self=False)
        group_max = torch.where(group_max == float("-inf"), torch.zeros_like(group_max), group_max)

        exp_val = torch.exp(logits_1d - group_max.index_select(0, case_idx))
        group_sum = exp_val.new_zeros((num_case,))
        group_sum.scatter_add_(0, case_idx, exp_val)
        alpha_edge = exp_val / (group_sum.index_select(0, case_idx) + 1e-8)

        uniform_alpha_edge = edge_weight / (case_deg.index_select(0, case_idx) + 1e-8)
        final_alpha_edge = (1.0 - self.residual_uniform) * alpha_edge + self.residual_uniform * uniform_alpha_edge

        case_repr.index_add_(0, case_idx, final_alpha_edge.unsqueeze(1) * edge_repr)

        if self.return_attention:
            return case_repr, {
                "edge_attention": final_alpha_edge,
                "case_context": ctx_edge,
                "edge_index": edge_index,
            }
        return case_repr

    def build_case_repr(
        self,
        Z: torch.Tensor,
        H_case: torch.Tensor | scipy.sparse.spmatrix,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """病例侧 leave-one-out attention readout。"""
        H_coalesced = _hmat(H_case, Z.device)
        indices = H_coalesced.indices()
        vals = H_coalesced.values()

        if indices.shape[1] == 0:
            out = torch.zeros((H_case.shape[1], Z.shape[1]), device=Z.device)
            empty_extra = {
                "edge_attention": Z.new_zeros((0,)),
                "case_context": Z.new_zeros((0, Z.shape[1])),
                "edge_index": indices,
            }
            return (out, empty_extra) if self.return_attention else out

        edge_repr = Z.index_select(0, indices[0])
        return self._build_case_repr_from_edges(
            edge_repr=edge_repr,
            case_idx=indices[1],
            edge_weight=vals,
            num_case=int(H_case.shape[1]),
            edge_index=indices,
        )

    def build_case_repr_from_refined(
        self,
        refined_case_node_repr: torch.Tensor | RefinedCaseNodeState,
        H_case: torch.Tensor | scipy.sparse.spmatrix,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """用病例条件化后的节点表示构建病例表示。"""
        if isinstance(refined_case_node_repr, RefinedCaseNodeState):
            num_hpo, num_case, hidden_dim = refined_case_node_repr.shape
            if getattr(H_case, "shape", None) != (num_hpo, num_case):
                raise ValueError(
                    "H_case 与 refined_case_node_repr 的前两维必须一致，"
                    f"得到 H_case={getattr(H_case, 'shape', None)}, refined={refined_case_node_repr.shape}"
                )
            if hidden_dim != self.hidden_dim:
                raise ValueError(
                    f"refined_case_node_repr hidden_dim 应为 {self.hidden_dim}，当前为 {hidden_dim}"
                )
            return self._build_case_repr_from_edges(
                edge_repr=refined_case_node_repr.edge_repr,
                case_idx=refined_case_node_repr.edge_index[1],
                edge_weight=refined_case_node_repr.edge_weight,
                num_case=num_case,
                edge_index=refined_case_node_repr.edge_index,
            )

        if refined_case_node_repr.ndim != 3:
            raise ValueError(
                "refined_case_node_repr 必须是三维张量，"
                f"当前 shape={tuple(refined_case_node_repr.shape)}"
            )

        H_coalesced = _hmat(H_case, refined_case_node_repr.device)
        num_hpo, num_case, hidden_dim = refined_case_node_repr.shape
        if H_coalesced.shape != (num_hpo, num_case):
            raise ValueError(
                "H_case 与 refined_case_node_repr 的前两维必须一致，"
                f"得到 H_case={H_coalesced.shape}, refined={tuple(refined_case_node_repr.shape)}"
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"refined_case_node_repr hidden_dim 应为 {self.hidden_dim}，当前为 {hidden_dim}"
            )

        indices = H_coalesced.indices()
        edge_repr = refined_case_node_repr[indices[0], indices[1], :]
        return self._build_case_repr_from_edges(
            edge_repr=edge_repr,
            case_idx=indices[1],
            edge_weight=H_coalesced.values(),
            num_case=num_case,
            edge_index=indices,
        )

    def build_disease_repr(self, Z: torch.Tensor, H_disease) -> torch.Tensor:
        """疾病侧 readout：`disease_repr = H_disease^T @ Z`。"""
        H = _hmat(H_disease, Z.device)
        return _spmm(H.t().coalesce(), Z)

    def forward(
        self,
        Z: torch.Tensor,
        H_case,
        H_disease,
        refined_case_node_repr: torch.Tensor | RefinedCaseNodeState | None = None,
        disease_repr_override: torch.Tensor | None = None,
    ) -> dict:
        if refined_case_node_repr is None:
            case_ret = self.build_case_repr(Z, H_case)
        else:
            case_ret = self.build_case_repr_from_refined(refined_case_node_repr, H_case)

        disease_repr = (
            disease_repr_override
            if disease_repr_override is not None
            else self.build_disease_repr(Z, H_disease)
        )

        if isinstance(case_ret, tuple):
            case_repr, extra = case_ret
            out_dict = {
                "case_repr": case_repr,
                "disease_repr": disease_repr,
            }
            out_dict.update(extra)
            return out_dict

        return {
            "case_repr": case_ret,
            "disease_repr": disease_repr,
        }
