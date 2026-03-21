import os
import sys

import numpy as np
import scipy.sparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.readout import HyperedgeReadout

# ── 测试参数 ─────────────────────────────────────────────────────────────────
N_HPO     = 6   # HPO 节点数
N_CASE    = 3   # 病例数
N_DISEASE = 4   # 疾病数
D         = 8   # 节点表示维度

torch.manual_seed(42)


def make_data():
    Z = torch.randn(N_HPO, D)

    # H_case：二值稀疏矩阵 [N_HPO, N_CASE]
    #   病例0 → HPO {0,1,2}
    #   病例1 → HPO {2,3}
    #   病例2 → HPO {0,4,5}
    rows_c = [0, 1, 2,  2, 3,  0, 4, 5]
    cols_c = [0, 0, 0,  1, 1,  2, 2, 2]
    H_case = scipy.sparse.csr_matrix(
        (np.ones(len(rows_c), dtype=np.float32), (rows_c, cols_c)),
        shape=(N_HPO, N_CASE),
    )

    # H_disease：加权稀疏矩阵 [N_HPO, N_DISEASE]
    #   用不同权重模拟知识库中 HPO→疾病的关联强度
    rows_d = [0, 1, 3, 2, 4, 5, 1, 3]
    cols_d = [0, 0, 1, 1, 2, 2, 3, 3]
    vals_d = [0.8, 0.6, 0.9, 0.5, 1.0, 0.7, 0.4, 0.3]
    H_disease = scipy.sparse.csr_matrix(
        (np.array(vals_d, dtype=np.float32), (rows_d, cols_d)),
        shape=(N_HPO, N_DISEASE),
    )

    return Z, H_case, H_disease


def manual_case_repr(Z_np, H_case_dense):
    out = H_case_dense.T @ Z_np
    deg = H_case_dense.sum(axis=0, keepdims=True).T
    return out / np.maximum(deg, 1e-6)


def manual_disease_repr(Z_np, H_disease_dense):
    return H_disease_dense.T @ Z_np


def main():
    Z, H_case, H_disease = make_data()
    model = HyperedgeReadout()
    model.eval()

    with torch.no_grad():
        out = model(Z, H_case, H_disease)

    case_repr = out["case_repr"]
    disease_repr = out["disease_repr"]

    ref_case = manual_case_repr(Z.numpy(), H_case.toarray())
    ref_disease = manual_disease_repr(Z.numpy(), H_disease.toarray())

    diff_case = np.abs(case_repr.numpy() - ref_case).max()
    diff_disease = np.abs(disease_repr.numpy() - ref_disease).max()

    assert case_repr.shape == (N_CASE, D), f"case_repr shape 异常: {case_repr.shape}"
    assert disease_repr.shape == (N_DISEASE, D), f"disease_repr shape 异常: {disease_repr.shape}"
    assert diff_case < 1e-5, f"case_repr 数值不一致: {diff_case}"
    assert diff_disease < 1e-5, f"disease_repr 数值不一致: {diff_disease}"

    H_case_t = torch.tensor(H_case.toarray(), dtype=torch.float32)
    H_disease_t = torch.tensor(H_disease.toarray(), dtype=torch.float32)
    with torch.no_grad():
        out_dense = model(Z, H_case_t, H_disease_t)

    diff_dense = max(
        (out_dense["case_repr"] - case_repr).abs().max().item(),
        (out_dense["disease_repr"] - disease_repr).abs().max().item(),
    )
    assert diff_dense < 1e-5, f"dense 输入结果不一致: {diff_dense}"

    print("test_readout.py passed")
    print(f"case_repr_shape={tuple(case_repr.shape)}")
    print(f"disease_repr_shape={tuple(disease_repr.shape)}")
    print(f"max_case_diff={diff_case:.2e}")
    print(f"max_disease_diff={diff_disease:.2e}")


if __name__ == "__main__":
    main()
