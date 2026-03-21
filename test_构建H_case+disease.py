from src.data.dataset import load_config, load_case_files, CaseBatchLoader
from src.data.build_hypergraph import load_static_graph, build_batch_hypergraph

cfg = load_config()
static = load_static_graph()
df = load_case_files(cfg["train_files"])  # 加载全部训练文件
loader = CaseBatchLoader(df, batch_size=cfg["batch_size"])

print(f"总 batch 数：{len(loader)}")

for i, batch_df in enumerate(loader):
    r = build_batch_hypergraph(
        batch_df, static["hpo_to_idx"], static["disease_to_idx"], static["H_disease"]
    )
    print(f"batch {i}: H={r['H'].shape}, cases={r['case_ids']}")
    if i == 2:  # 只看前3个 batch
        break