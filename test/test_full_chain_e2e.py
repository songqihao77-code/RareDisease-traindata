import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_hypergraph import build_batch_hypergraph, load_static_graph
from src.data.dataset import CaseBatchLoader, load_case_files, load_config
from src.models.model_pipeline import ModelPipeline


SEED = 7
SMOKE_BATCH_SIZE = 8
OVERFIT_STEPS = 100
OVERFIT_LR = 0.05


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_batch(batch_size: int = SMOKE_BATCH_SIZE) -> dict:
    cfg = load_config()
    df = load_case_files([cfg["train_files"][0]])
    loader = CaseBatchLoader(df, batch_size=batch_size)
    batch_df = loader.get_batch(0)
    static = load_static_graph()
    return {
        "cfg": cfg,
        "df": df,
        "loader": loader,
        "batch_df": batch_df,
        "static": static,
    }


def _select_gold_diseases(batch_df, disease_to_idx_full: dict[str, int]) -> list[str]:
    ordered_labels = (
        batch_df.groupby("case_id", sort=False)["mondo_label"].first().tolist()
    )
    deduped = list(dict.fromkeys(label for label in ordered_labels if label in disease_to_idx_full))
    if not deduped:
        raise AssertionError("当前 8 条样本在疾病索引中没有可用标签。")
    return deduped


def _prepare_compact_graph(batch_df, static: dict) -> dict:
    selected_diseases = _select_gold_diseases(batch_df, static["disease_to_idx"])
    subset_cols = [static["disease_to_idx"][disease_id] for disease_id in selected_diseases]
    H_disease_subset = static["H_disease"][:, subset_cols]
    disease_to_idx_subset = {
        disease_id: idx for idx, disease_id in enumerate(selected_diseases)
    }
    return build_batch_hypergraph(
        batch_df,
        static["hpo_to_idx"],
        disease_to_idx_subset,
        H_disease_subset,
    )


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} 含有 nan 或 inf"


def run_full_chain_checks() -> dict:
    set_seed()
    prepared = _prepare_batch(batch_size=SMOKE_BATCH_SIZE)

    loader = prepared["loader"]
    batch_df = prepared["batch_df"]
    static = prepared["static"]

    batch_graph = build_batch_hypergraph(
        batch_df,
        static["hpo_to_idx"],
        static["disease_to_idx"],
        static["H_disease"],
    )
    pipeline = ModelPipeline(num_hpo=batch_graph["H"].shape[0], hidden_dim=128)
    forward_out = pipeline(batch_graph)

    batch_size = len(batch_graph["case_ids"])
    disease_count = batch_graph["H_disease"].shape[1]

    check_results = [
        {
            "id": 1,
            "name": "dataset.py 能正确产出一个 batch",
            "passed": (
                batch_df is not None
                and batch_df.shape[0] > 0
                and batch_df["case_id"].nunique() == SMOKE_BATCH_SIZE
                and set(loader.case_ids[:SMOKE_BATCH_SIZE]) == set(batch_graph["case_ids"])
            ),
            "detail": f"batch_rows={len(batch_df)}, batch_unique_cases={batch_df['case_id'].nunique()}",
        },
        {
            "id": 2,
            "name": "build_hypergraph.py 返回完整字典且 H_case / H_disease 维度正确",
            "passed": (
                {
                    "H",
                    "H_case",
                    "H_disease",
                    "case_ids",
                    "case_labels",
                    "case_cols_global",
                    "disease_cols_global",
                    "gold_disease_cols_global",
                }.issubset(batch_graph.keys())
                and batch_graph["H_case"].shape == (static["num_hpo"], batch_size)
                and batch_graph["H_disease"].shape == (static["num_hpo"], static["num_disease"])
                and batch_graph["H"].shape == (static["num_hpo"], batch_size + static["num_disease"])
            ),
            "detail": (
                f"H_case={batch_graph['H_case'].shape}, "
                f"H_disease={batch_graph['H_disease'].shape}, H={batch_graph['H'].shape}"
            ),
        },
        {
            "id": 3,
            "name": "hgnn_encoder.py 输出 Z，且 Z.shape[1] == 128",
            "passed": (
                "Z" in forward_out
                and forward_out["Z"].shape[0] == static["num_hpo"]
                and forward_out["Z"].shape[1] == 128
            ),
            "detail": f"Z_shape={tuple(forward_out['Z'].shape)}",
        },
        {
            "id": 4,
            "name": "readout.py 输出 case_repr.shape[0] == batch_size",
            "passed": forward_out["case_repr"].shape[0] == batch_size,
            "detail": f"case_repr_shape={tuple(forward_out['case_repr'].shape)}, batch_size={batch_size}",
        },
        {
            "id": 5,
            "name": "readout.py 输出 disease_repr.shape[0] == 疾病总数",
            "passed": forward_out["disease_repr"].shape[0] == disease_count,
            "detail": (
                f"disease_repr_shape={tuple(forward_out['disease_repr'].shape)}, "
                f"disease_count={disease_count}"
            ),
        },
        {
            "id": 6,
            "name": "scorer.py 输出 scores.shape == (batch_size, disease_count)",
            "passed": tuple(forward_out["scores_local"].shape) == (batch_size, disease_count),
            "detail": f"scores_local_shape={tuple(forward_out['scores_local'].shape)}",
        },
        {
            "id": 7,
            "name": "scores、Z、loss 中都没有 nan / inf",
            "passed": bool(
                torch.isfinite(forward_out["scores"]).all()
                and torch.isfinite(forward_out["Z"]).all()
                and torch.isfinite(forward_out["loss"]).all()
            ),
            "detail": (
                f"scores_finite={bool(torch.isfinite(forward_out['scores']).all())}, "
                f"Z_finite={bool(torch.isfinite(forward_out['Z']).all())}, "
                f"loss_finite={bool(torch.isfinite(forward_out['loss']).all())}"
            ),
        },
        {
            "id": 8,
            "name": "gold_disease_cols_global 全部落在 scores 的列范围内",
            "passed": all(
                0 <= col < forward_out["scores"].shape[1]
                for col in batch_graph["gold_disease_cols_global"]
            ),
            "detail": (
                f"gold_range=[{min(batch_graph['gold_disease_cols_global'])}, "
                f"{max(batch_graph['gold_disease_cols_global'])}], "
                f"scores_cols={forward_out['scores'].shape[1]}"
            ),
        },
        {
            "id": 9,
            "name": "loss.backward() 成功",
            "passed": False,
            "detail": "",
        },
    ]

    pipeline.zero_grad(set_to_none=True)
    forward_out["loss"].backward()
    grad_tensors = [
        pipeline.encoder.X0.grad,
        pipeline.encoder.theta0.weight.grad,
        pipeline.encoder.theta1.weight.grad,
    ]
    grads_ok = all(grad is not None and torch.isfinite(grad).all() for grad in grad_tensors)
    check_results[8]["passed"] = grads_ok
    check_results[8]["detail"] = (
        f"X0_grad={grad_tensors[0] is not None}, "
        f"theta0_grad={grad_tensors[1] is not None}, "
        f"theta1_grad={grad_tensors[2] is not None}"
    )

    for item in check_results:
        assert item["passed"], f"检查 {item['id']} 失败: {item['detail']}"

    return {
        "check_results": check_results,
        "summary": {
            "batch_size": batch_size,
            "disease_count": disease_count,
            "loss": float(forward_out["loss"].item()),
            "scores_shape": list(forward_out["scores_local"].shape),
            "global_scores_shape": list(forward_out["scores"].shape),
        },
    }


def run_small_sample_overfit() -> dict:
    set_seed()
    prepared = _prepare_batch(batch_size=SMOKE_BATCH_SIZE)
    compact_graph = _prepare_compact_graph(prepared["batch_df"], prepared["static"])

    pipeline = ModelPipeline(num_hpo=compact_graph["H"].shape[0], hidden_dim=128)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=OVERFIT_LR)

    reached_full_acc_at = None
    best_acc = 0.0
    best_loss = float("inf")
    last_out = None

    for step in range(1, OVERFIT_STEPS + 1):
        optimizer.zero_grad()
        out = pipeline(compact_graph)
        loss = out["loss"]
        _assert_finite("overfit_scores", out["scores"])
        _assert_finite("overfit_Z", out["Z"])
        _assert_finite("overfit_loss", loss.unsqueeze(0))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = out["scores_local"].argmax(dim=1)
            acc = (preds == out["gold_disease_cols_local"]).float().mean().item()
            best_acc = max(best_acc, acc)
            best_loss = min(best_loss, float(loss.item()))
            if reached_full_acc_at is None and acc == 1.0:
                reached_full_acc_at = step
            last_out = {
                "step": step,
                "loss": float(loss.item()),
                "acc": float(acc),
            }

    result = {
        "passed": reached_full_acc_at is not None,
        "check_result": {
            "id": 10,
            "name": "8 条样本能快速过拟合",
            "passed": reached_full_acc_at is not None,
            "detail": (
                f"batch_size={len(compact_graph['case_ids'])}, "
                f"disease_count={compact_graph['H_disease'].shape[1]}, "
                f"reached_full_acc_at={reached_full_acc_at}, "
                f"best_acc={best_acc:.4f}, best_loss={best_loss:.4f}"
            ),
        },
        "batch_size": len(compact_graph["case_ids"]),
        "disease_count": compact_graph["H_disease"].shape[1],
        "reached_full_acc_at": reached_full_acc_at,
        "best_acc": best_acc,
        "best_loss": best_loss,
        "last_step": last_out,
    }
    assert result["passed"], f"8 条样本未在 {OVERFIT_STEPS} 步内达到 100% 训练准确率"
    return result


def test_full_chain_e2e_smoke() -> None:
    run_full_chain_checks()


def test_small_sample_can_overfit_quickly() -> None:
    run_small_sample_overfit()


if __name__ == "__main__":
    smoke = run_full_chain_checks()
    overfit = run_small_sample_overfit()
    report = {
        "checks": smoke["check_results"] + [overfit["check_result"]],
        "smoke_summary": smoke["summary"],
        "overfit": overfit,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))





