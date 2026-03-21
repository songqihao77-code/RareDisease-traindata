"""
测试 scorer.py 的正确性与输出可读性。
构造小规模假数据，运行 CosineScorer，打印中间值和最终分数矩阵。
"""

import numpy as np
import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.scorer import CosineScorer

# ── 测试参数 ──────────────────────────────────────────────────────────────────
B = 3    # 病例数
M = 5    # 疾病数
D = 8    # 表示维度

torch.manual_seed(0)


def print_tensor(name, T, decimals=4):
    print(f"\n{name}  (形状 {list(T.shape)}):")
    print(np.round(T.detach().numpy(), decimals))


def main():
    print("=" * 60)
    print("  CosineScorer 测试")
    print("=" * 60)

    # ── 构造随机输入（模拟 readout 输出）────────────────────────
    case_repr    = torch.randn(B, D)
    disease_repr = torch.randn(M, D)

    print_tensor("case_repr（病例表示，来自 readout）", case_repr)
    print_tensor("disease_repr（疾病表示，来自 readout）", disease_repr)

    # ── 运行 scorer ───────────────────────────────────────────
    scorer = CosineScorer()
    scorer.eval()
    with torch.no_grad():
        out = scorer(case_repr, disease_repr)

    scores = out["scores"]
    print_tensor("scores（余弦相似度矩阵，行=病例，列=疾病）", scores)

    # ── 逐行显示每个病例的最高得分疾病 ───────────────────────
    print("\n── 每个病例的 Top-1 疾病 ──────────────────────────────")
    for i in range(B):
        top_j   = scores[i].argmax().item()
        top_val = scores[i, top_j].item()
        print(f"  病例 {i}  →  疾病 {top_j}  (cos={top_val:.4f})")

    # ── 精度验证：与 numpy 手工余弦相似度对比 ────────────────
    def np_cosine(A, B_):
        A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B_n = B_ / (np.linalg.norm(B_, axis=1, keepdims=True) + 1e-12)
        return A_n @ B_n.T

    ref = np_cosine(case_repr.numpy(), disease_repr.numpy())
    diff = np.abs(scores.numpy() - ref).max()

    print(f"\n── 精度验证 ────────────────────────────────────────────")
    print(f"与 numpy 手算余弦最大绝对误差：{diff:.2e}  {'PASS' if diff < 1e-5 else 'FAIL'}")

    # ── 值域验证：余弦相似度应在 [-1, 1] ──────────────────────
    s_min, s_max = scores.min().item(), scores.max().item()
    in_range = (-1.0 - 1e-5) <= s_min and s_max <= (1.0 + 1e-5)
    print(f"分数值域 [{s_min:.4f}, {s_max:.4f}]，应在 [-1, 1]：{'PASS' if in_range else 'FAIL'}")

    # ── 自身余弦（单位向量与自身的余弦应为 1.0）─────────────
    same_repr = torch.randn(4, D)
    with torch.no_grad():
        self_out = scorer(same_repr, same_repr)
    diag = torch.diag(self_out["scores"])
    diag_err = (diag - 1.0).abs().max().item()
    print(f"自身余弦（对角线应全为 1.0），最大误差：{diag_err:.2e}  {'PASS' if diag_err < 1e-5 else 'FAIL'}")

    # ── 输出形状断言 ───────────────────────────────────────────
    assert scores.shape == (B, M), f"形状异常：{scores.shape}"
    print(f"\n形状断言 {list(scores.shape)} == [{B}, {M}]：PASS")

    print("\n全部测试完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
