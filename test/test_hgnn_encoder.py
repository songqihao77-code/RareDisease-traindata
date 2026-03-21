# 手动测试脚本：HGNNEncoder（含 GPU 测试）
# 运行方式：在 D:/RareDisease 目录下执行 python test_hgnn_encoder.py

import torch
from src.data.build_hypergraph import load_static_graph, build_batch_hypergraph
from src.data.dataset import load_config, load_case_files, CaseBatchLoader
from src.models.hgnn_encoder import HGNNEncoder

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. 加载静态图和配置 ───────────────────────────────────────────────────────
print("=" * 50)
print("1. 加载静态图和配置")
print("=" * 50)
cfg    = load_config()
static = load_static_graph()
print(f"  运行设备      : {device}")
if device.type == "cuda":
    print(f"  GPU 型号      : {torch.cuda.get_device_name(0)}")
    print(f"  显存总量      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"  batch_size    : {cfg['batch_size']}")
print(f"  H_disease shape: {static['H_disease'].shape}")

# ── 2. 构建一个 batch 的 H ────────────────────────────────────────────────────
print()
print("=" * 50)
print("2. 加载训练文件，构建 batch（CPU，scipy sparse）")
print("=" * 50)
df     = load_case_files([cfg["train_files"][0]])
loader = CaseBatchLoader(df, batch_size=cfg["batch_size"])
batch0 = loader.get_batch(0)
result = build_batch_hypergraph(
    batch0,
    static["hpo_to_idx"],
    static["disease_to_idx"],
    static["H_disease"],
)
H       = result["H"]
num_hpo = H.shape[0]
print(f"  H type  : {type(H)}")
print(f"  H shape : {H.shape}")
print(f"  H nnz   : {H.nnz}  (稀疏度 {H.nnz / (H.shape[0] * H.shape[1]) * 100:.3f}%)")
print(f"  case_ids: {result['case_ids']}")

# ── 3. 初始化模型并移到目标设备 ───────────────────────────────────────────────
print()
print("=" * 50)
print(f"3. 初始化 HGNNEncoder 并移到 {device}")
print("=" * 50)
model = HGNNEncoder(num_hpo=num_hpo, hidden_dim=128).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  num_hpo    = {num_hpo}")
print(f"  hidden_dim = 128")
print(f"  总参数量   = {total_params:,}")
print(f"  X0 device  : {model.X0.device}")
if device.type == "cuda":
    print(f"  显存已用   : {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# ── 4. 前向传播（H 在 CPU，模型在 GPU，forward 内自动转移）────────────────────
print()
print("=" * 50)
print(f"4. 前向传播（H 从 CPU scipy sparse 自动转到 {device}）")
print("=" * 50)
Z = model(H)   # forward 内部自动把 H 转到 model 所在设备
print(f"  Z shape  : {Z.shape}  (期望 [{num_hpo}, 128])")
print(f"  Z device : {Z.device}")
print(f"  Z mean   : {Z.mean().item():.6f}")
print(f"  Z std    : {Z.std().item():.6f}")
print(f"  含 NaN   : {Z.isnan().any().item()}")
print(f"  含 Inf   : {Z.isinf().any().item()}")
if device.type == "cuda":
    print(f"  显存已用 : {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# ── 5. 梯度回传 ───────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("5. 梯度回传")
print("=" * 50)
Z.sum().backward()
print(f"  X0     grad norm : {model.X0.grad.norm().item():.4f}")
print(f"  theta0 grad norm : {model.theta0.weight.grad.norm().item():.4f}")
print(f"  theta1 grad norm : {model.theta1.weight.grad.norm().item():.4f}")

# ── 6. 断言 ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("6. 断言检查")
print("=" * 50)
assert Z.shape == (num_hpo, 128),              "FAIL: Z shape 不符"
assert str(Z.device).startswith(device.type),  f"FAIL: Z 不在 {device}"
assert not Z.isnan().any(),                    "FAIL: Z 含 NaN"
assert not Z.isinf().any(),                    "FAIL: Z 含 Inf"
assert model.X0.grad is not None,              "FAIL: X0 无梯度"
assert model.theta0.weight.grad is not None,   "FAIL: theta0 无梯度"
assert model.theta1.weight.grad is not None,   "FAIL: theta1 无梯度"
print("  所有断言通过")

print()
print("=" * 50)
print("测试完成")
print("=" * 50)
