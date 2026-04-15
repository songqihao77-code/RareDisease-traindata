try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project(allow_omp_duplicate=True)

import numpy as np
import pytest
import scipy.sparse
import torch
import torch.nn as nn

from src.models.readout import HyperedgeReadout

N_HPO = 6
N_CASE = 2
N_DISEASE = 4
D = 8

torch.manual_seed(42)

def test_readout():
    Z = torch.randn(N_HPO, D, requires_grad=True)
    # case 0 and case 1 share the first symptom (HPO=0) but have different contexts
    rows_c = [0, 1, 0, 2, 3]
    cols_c = [0, 0, 1, 1, 1] 
    vals_c = [1.0, 1.0, 1.0, 1.0, 1.0]

    H_case = scipy.sparse.csr_matrix(
        (np.array(vals_c, dtype=np.float32), (rows_c, cols_c)),
        shape=(N_HPO, N_CASE),
    )

    rows_d = [0, 1, 3]
    cols_d = [0, 0, 1]
    vals_d = [1.0, 1.0, 1.0]
    H_disease = scipy.sparse.csr_matrix(
        (np.array(vals_d, dtype=np.float32), (rows_d, cols_d)),
        shape=(N_HPO, N_DISEASE),
    )

    # 强制偏置使其产出固定区分度
    model = HyperedgeReadout(hidden_dim=D, residual_uniform=0.0, return_attention=True)
    with torch.no_grad():
        for layer in model.attn_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)

    out = model(Z, torch.tensor(H_case.toarray(), dtype=torch.float32), torch.tensor(H_disease.toarray(), dtype=torch.float32))
    
    case_repr = out["case_repr"]
    disease_repr = out["disease_repr"]
    edge_attention = out["edge_attention"]
    edge_index = out["edge_index"]

    # 检查 disease_repr
    expected_disease = torch.tensor(H_disease.toarray().T, dtype=torch.float32) @ Z
    assert torch.allclose(disease_repr, expected_disease, atol=1e-5)

    assert case_repr.shape == (N_CASE, D)
    assert not torch.isnan(case_repr).any()

    # 验证梯度
    loss = case_repr.sum()
    loss.backward()
    assert Z.grad is not None
    
    # 验证同一个 HPO(0) 在不同 Case 下的注意力是有差异的
    # HPO 0 在 case 0 (边索引 0) 和 case 1 (边索引 2)
    alpha_0_case_0 = edge_attention[(edge_index[0] == 0) & (edge_index[1] == 0)][0]
    alpha_0_case_1 = edge_attention[(edge_index[0] == 0) & (edge_index[1] == 1)][0]
    assert torch.abs(alpha_0_case_0 - alpha_0_case_1) > 1e-5


def test_readout_rejects_unknown_context_mode() -> None:
    with pytest.raises(ValueError, match="context_mode"):
        HyperedgeReadout(hidden_dim=D, context_mode="global")


def test_build_case_repr_from_refined_matches_base_when_inputs_same() -> None:
    Z = torch.randn(N_HPO, D)
    rows_c = [0, 1, 0, 2, 3]
    cols_c = [0, 0, 1, 1, 1]
    vals_c = [1.0, 1.0, 1.0, 1.0, 1.0]
    H_case = scipy.sparse.csr_matrix(
        (np.array(vals_c, dtype=np.float32), (rows_c, cols_c)),
        shape=(N_HPO, N_CASE),
    )

    model = HyperedgeReadout(hidden_dim=D, residual_uniform=0.0, return_attention=False)
    refined = Z.unsqueeze(1).expand(-1, N_CASE, -1).clone()

    base_case_repr = model.build_case_repr(Z, H_case)
    refined_case_repr = model.build_case_repr_from_refined(refined, H_case)

    assert refined_case_repr.shape == (N_CASE, D)
    assert torch.allclose(refined_case_repr, base_case_repr, atol=1e-5)
    assert torch.isfinite(refined_case_repr).all()

if __name__ == "__main__":
    test_readout()
    print("test_readout.py passed")
