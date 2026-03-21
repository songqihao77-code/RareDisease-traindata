"""
测试 readout.py 的正确性与输出可读性。
构造小规模假数据，运行 HyperedgeReadout，打印中间值和最终结果。
"""

import numpy as np
import torch
import scipy.sparse

import sys
import os
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
    """构造可复现的假数据，返回 Z、H_case、H_disease（scipy sparse）。"""

    # Z：HPO 节点表示，模拟 hgnn_encoder 输出
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


def print_sparse(name, M):
    """打印 scipy sparse 矩阵的稠密形式，便于人工核查。"""
    print(f"\n{name}（形状 {M.shape}）：")
    print(np.array2string(M.toarray(), precision=2, suppress_small=True))


def print_tensor(name, T):
    """打印 Tensor，保留 4 位小数。"""
    print(f"\n{name}（形状 {list(T.shape)}）：")
    print(torch.round(T, decimals=4).detach().numpy())


# ── 手工验证辅助：逐列手算 case_repr ────────────────────────────────────────
def manual_case_repr(Z_np, H_case_dense):
    """用 numpy 手工计算 case_repr，供与模型输出对比。"""
    out = H_case_dense.T @ Z_np                          # [N_case, D]
    deg = H_case_dense.sum(axis=0, keepdims=True).T      # [N_case, 1]
    return out / np.maximum(deg, 1e-6)


def manual_disease_repr(Z_np, H_disease_dense):
    """用 numpy 手工计算 disease_repr，供与模型输出对比。"""
    return H_disease_dense.T @ Z_np                      # [N_disease, D]


# ── 主测试流程 ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HyperedgeReadout 测试")
    print("=" * 60)

    Z, H_case, H_disease = make_data()

    # ── 打印输入 ──────────────────────────────────────────────────
    print_tensor("Z（HPO 节点表示）", Z)
    print_sparse("H_case（病例超边矩阵）", H_case)
    print_sparse("H_disease（疾病超边矩阵）", H_disease)

    # ── 每个病例的 HPO 覆盖度（即 case_degree）───────────────────
    case_deg = np.array(H_case.sum(axis=0)).flatten()
    print(f"\ncase_degree（每个病例的 HPO 数目）：{case_deg}")

    # ── 运行 readout 模块 ─────────────────────────────────────────
    model = HyperedgeReadout()
    model.eval()
    with torch.no_grad():
        out = model(Z, H_case, H_disease)

    case_repr    = out["case_repr"]
    disease_repr = out["disease_repr"]

    print_tensor("case_repr（病例表示，平均 readout）", case_repr)
    print_tensor("disease_repr（疾病表示，加权 readout）", disease_repr)

    # ── 与 numpy 手工结果对比（逐元素差）────────────────────────
    Z_np = Z.numpy()
    H_c_dense = H_case.toarray()
    H_d_dense = H_disease.toarray()

    ref_case    = manual_case_repr(Z_np, H_c_dense)
    ref_disease = manual_disease_repr(Z_np, H_d_dense)

    diff_case    = np.abs(case_repr.numpy()    - ref_case).max()
    diff_disease = np.abs(disease_repr.numpy() - ref_disease).max()

    print(f"\n── 精度验证 ──────────────────────────────────")
    print(f"case_repr    最大绝对误差：{diff_case:.2e}  {'PASS' if diff_case < 1e-5 else 'FAIL'}")
    print(f"disease_repr 最大绝对误差：{diff_disease:.2e}  {'PASS' if diff_disease < 1e-5 else 'FAIL'}")

    # ── 输出形状断言 ─────────────────────────────────────────────
    assert case_repr.shape    == (N_CASE,    D), f"形状异常：{case_repr.shape}"
    assert disease_repr.shape == (N_DISEASE, D), f"形状异常：{disease_repr.shape}"
    print("\n形状断言：PASS")

    # ── 测试 Tensor 输入（非 scipy sparse）─────────────────────
    H_case_t    = torch.tensor(H_c_dense, dtype=torch.float32)
    H_disease_t = torch.tensor(H_d_dense, dtype=torch.float32)
    with torch.no_grad():
        out2 = model(Z, H_case_t, H_disease_t)
    diff2 = max(
        (out2["case_repr"]    - case_repr).abs().max().item(),
        (out2["disease_repr"] - disease_repr).abs().max().item(),
    )
    print(f"稠密 Tensor 输入一致性误差：{diff2:.2e}  {'PASS' if diff2 < 1e-5 else 'FAIL'}")

    print("\n全部测试完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
