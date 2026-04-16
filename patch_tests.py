import re
import os

# 1. 修复 test_trainer.py
with open("test/test_trainer.py", "r", encoding="utf-8") as f:
    text = f.read()

test_trainer_code = """
def test_build_model_config_readout_passthrough():
    from src.training.trainer import build_model_config
    cfg = {
        "model": {
            "hidden_dim": 256,
            "readout": {
                "attn_hidden_dim": 128,
                "context_mode": "leave_one_out"
            }
        }
    }
    out = build_model_config(cfg, num_hpo=10)
    assert out["model"]["readout"]["attn_hidden_dim"] == 128
    assert out["model"]["readout"]["context_mode"] == "leave_one_out"
    assert out["model"]["readout"]["hidden_dim"] == 256

"""
if "test_build_model_config_readout_passthrough" not in text:
    with open("test/test_trainer.py", "a", encoding="utf-8") as f:
        f.write(test_trainer_code)

# 2. 修复 test_full_chain_e2e.py
with open("test/test_full_chain_e2e.py", "r", encoding="utf-8") as f:
    text = f.read()

# Fix _build_model_config
text = re.sub(r'\"readout\": \{\"type\": \"hyperedge\"\},', '"readout": {"type": "hyperedge", "hidden_dim": hidden_dim, "attn_hidden_dim": hidden_dim},', text)

# Fix _prepare_batch train_files
prepare_batch_fix = """def _prepare_batch(batch_size: int = SMOKE_BATCH_SIZE) -> dict:
    cfg = load_config()
    train_files = cfg.get("paths", {}).get("train_files")
    if train_files is None:
        train_dir = Path(cfg["paths"]["train_dir"])
        train_files = [str(next(train_dir.glob("*.xlsx")))]
    df = load_case_files([train_files[0]])"""
text = re.sub(r'def _prepare_batch\(batch_size: int = SMOKE_BATCH_SIZE\) -> dict:\n    cfg = load_config\(\)\n    df = load_case_files\(\[cfg\["train_files"\]\[0\]\]\)', prepare_batch_fix, text)

with open("test/test_full_chain_e2e.py", "w", encoding="utf-8") as f:
    f.write(text)

# 3. 重写 test_readout.py
readout_code = '''try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project(allow_omp_duplicate=True)

import numpy as np
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

if __name__ == "__main__":
    test_readout()
    print("test_readout.py passed")
'''

with open("test/test_readout.py", "w", encoding="utf-8") as f:
    f.write(readout_code)

print("Tests patched successfully")
