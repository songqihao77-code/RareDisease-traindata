import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.scorer import CosineScorer


B = 3
M = 5
D = 8

torch.manual_seed(0)


def np_cosine(A, B_):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B_ / (np.linalg.norm(B_, axis=1, keepdims=True) + 1e-12)
    return A_n @ B_n.T


def main():
    case_repr = torch.randn(B, D)
    disease_repr = torch.randn(M, D)

    scorer = CosineScorer()
    scorer.eval()
    with torch.no_grad():
        out = scorer(case_repr, disease_repr)

    scores = out["scores"]
    ref = np_cosine(case_repr.numpy(), disease_repr.numpy())
    diff = np.abs(scores.numpy() - ref).max()

    assert scores.shape == (B, M), f"scores shape 异常: {scores.shape}"
    assert diff < 1e-5, f"scores 与 numpy 结果不一致: {diff}"

    s_min = scores.min().item()
    s_max = scores.max().item()
    assert -1.0 - 1e-5 <= s_min <= 1.0 + 1e-5
    assert -1.0 - 1e-5 <= s_max <= 1.0 + 1e-5

    same_repr = torch.randn(4, D)
    with torch.no_grad():
        self_scores = scorer(same_repr, same_repr)["scores"]
    diag_err = (torch.diag(self_scores) - 1.0).abs().max().item()
    assert diag_err < 1e-5, f"自相似对角线不为 1: {diag_err}"

    print("test_scorer.py passed")
    print(f"scores_shape={tuple(scores.shape)}")
    print(f"max_diff={diff:.2e}")
    print(f"score_range=({s_min:.4f}, {s_max:.4f})")


if __name__ == "__main__":
    main()

