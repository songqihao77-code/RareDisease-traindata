try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project(allow_omp_duplicate=True)

import torch

from src.data.build_hypergraph import build_batch_hypergraph, load_static_graph
from src.data.dataset import CaseBatchLoader, load_case_files, load_config
from src.models.hgnn_encoder import HGNNEncoder


HIDDEN_DIM = 128


def _print_section(title: str) -> None:
    print()
    print("=" * 50)
    print(title)
    print("=" * 50)


def run_hgnn_encoder_check() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print_section("1. 加载静态图和配置")
    cfg = load_config()
    static = load_static_graph()
    print(f"  运行设备      : {device}")
    if device.type == "cuda":
        print(f"  GPU 型号      : {torch.cuda.get_device_name(0)}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存总量      : {total_memory_gb:.1f} GB")
    print(f"  batch_size    : {cfg['batch_size']}")
    print(f"  H_disease shape: {static['H_disease'].shape}")

    _print_section("2. 加载训练文件，构建 batch（CPU，scipy sparse）")
    df = load_case_files([cfg["train_files"][0]])
    loader = CaseBatchLoader(df, batch_size=cfg["batch_size"])
    batch0 = loader.get_batch(0)
    result = build_batch_hypergraph(
        batch0,
        static["hpo_to_idx"],
        static["disease_to_idx"],
        static["H_disease"],
    )
    H = result["H"]
    num_hpo = H.shape[0]
    print(f"  H type  : {type(H)}")
    print(f"  H shape : {H.shape}")
    print(f"  H nnz   : {H.nnz}  (稀疏度 {H.nnz / (H.shape[0] * H.shape[1]) * 100:.3f}%)")
    print(f"  case_ids: {result['case_ids']}")

    _print_section(f"3. 初始化 HGNNEncoder 并移到 {device}")
    model = HGNNEncoder(num_hpo=num_hpo, hidden_dim=HIDDEN_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  num_hpo    = {num_hpo}")
    print(f"  hidden_dim = {HIDDEN_DIM}")
    print(f"  总参数量   = {total_params:,}")
    print(f"  X0 device  : {model.X0.device}")
    if device.type == "cuda":
        print(f"  显存已用   : {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    _print_section(f"4. 前向传播（H 从 CPU scipy sparse 自动转到 {device}）")
    Z = model(H)
    print(f"  Z shape  : {Z.shape}  (期望 [{num_hpo}, {HIDDEN_DIM}])")
    print(f"  Z device : {Z.device}")
    print(f"  Z mean   : {Z.mean().item():.6f}")
    print(f"  Z std    : {Z.std().item():.6f}")
    print(f"  含 NaN   : {Z.isnan().any().item()}")
    print(f"  含 Inf   : {Z.isinf().any().item()}")
    if device.type == "cuda":
        print(f"  显存已用 : {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    _print_section("5. 梯度回传")
    Z.sum().backward()
    print(f"  X0     grad norm : {model.X0.grad.norm().item():.4f}")
    print(f"  theta0 grad norm : {model.theta0.weight.grad.norm().item():.4f}")
    print(f"  theta1 grad norm : {model.theta1.weight.grad.norm().item():.4f}")

    _print_section("6. 断言检查")
    assert Z.shape == (num_hpo, HIDDEN_DIM), "FAIL: Z shape 不符"
    assert str(Z.device).startswith(device.type), f"FAIL: Z 不在 {device}"
    assert not Z.isnan().any(), "FAIL: Z 含 NaN"
    assert not Z.isinf().any(), "FAIL: Z 含 Inf"
    assert model.X0.grad is not None, "FAIL: X0 无梯度"
    assert model.theta0.weight.grad is not None, "FAIL: theta0 无梯度"
    assert model.theta1.weight.grad is not None, "FAIL: theta1 无梯度"
    print("  所有断言通过")

    _print_section("测试完成")
    return {
        "passed": True,
        "device": str(device),
        "num_hpo": num_hpo,
        "hidden_dim": HIDDEN_DIM,
        "z_shape": tuple(Z.shape),
        "case_ids": list(result["case_ids"]),
    }


def test_hgnn_encoder_smoke() -> None:
    result = run_hgnn_encoder_check()
    assert result["passed"]


if __name__ == "__main__":
    run_hgnn_encoder_check()
