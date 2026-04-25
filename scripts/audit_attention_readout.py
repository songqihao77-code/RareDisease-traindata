from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_hypergraph import build_batch_hypergraph, load_static_graph
from src.data.dataset import CaseBatchLoader, load_case_files
from src.evaluation.evaluator import load_checkpoint_model
from src.training.trainer import resolve_train_files, split_train_val_by_case


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def resolve_device(config: dict[str, Any], requested: str) -> torch.device:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)
    configured = str(config.get("train", {}).get("device", "cpu"))
    if configured.startswith("cuda") and torch.cuda.is_available():
        return torch.device(configured)
    return torch.device("cpu")


def build_audit_batch(
    config: dict[str, Any],
    *,
    split: str,
    max_cases: int,
) -> pd.DataFrame:
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    case_files = resolve_train_files(paths_cfg)
    all_df = load_case_files(
        file_paths=[str(path) for path in case_files],
        disease_index_path=paths_cfg["disease_index_path"],
        split_namespace="train",
    )
    train_df, val_df = split_train_val_by_case(
        df=all_df,
        val_ratio=float(data_cfg["val_ratio"]),
        random_seed=int(data_cfg["random_seed"]),
    )
    audit_df = val_df if split == "val" else train_df
    if audit_df.empty:
        raise ValueError(f"Selected split is empty: {split}")

    case_ids = audit_df["case_id"].drop_duplicates().head(max_cases).tolist()
    return audit_df[audit_df["case_id"].isin(case_ids)].copy()


def entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    clipped = np.clip(values.astype(np.float64, copy=False), 1e-12, None)
    return float(-(clipped * np.log(clipped)).sum())


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for item_idx in order[pos:end]:
            ranks[item_idx] = avg_rank
        pos = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def summarize_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    alpha = [float(row["attention_alpha"]) for row in rows]
    weights = [float(row["h_case_weight"]) for row in rows]
    top_row = min(rows, key=lambda row: int(row["alpha_rank"]))
    return {
        "case_id": rows[0]["case_id"],
        "gold_disease_id": rows[0]["gold_disease_id"],
        "top1_disease_id": rows[0]["top1_disease_id"],
        "is_correct": bool(rows[0]["is_correct"]),
        "num_active_hpo": len(rows),
        "attention_entropy": entropy(np.asarray(alpha, dtype=np.float64)),
        "attention_weight_pearson": pearson(alpha, weights),
        "attention_weight_spearman": spearman(alpha, weights),
        "top_attended_hpo_id": top_row["hpo_id"],
        "top_attended_in_gold": bool(top_row["in_gold_disease_hyperedge"]),
        "top_attended_in_top1": bool(top_row["in_top1_disease_hyperedge"]),
    }


def mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def summarize_groups(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, expected_correct in (("correct", True), ("incorrect", False), ("all", None)):
        selected = [
            row
            for row in case_rows
            if expected_correct is None or bool(row["is_correct"]) is expected_correct
        ]
        out[name] = {
            "num_cases": len(selected),
            "attention_entropy_mean": mean_optional([row["attention_entropy"] for row in selected]),
            "attention_entropy_var": (
                None
                if not selected
                else float(np.var([float(row["attention_entropy"]) for row in selected]))
            ),
            "attention_weight_pearson_mean": mean_optional(
                [row["attention_weight_pearson"] for row in selected]
            ),
            "attention_weight_spearman_mean": mean_optional(
                [row["attention_weight_spearman"] for row in selected]
            ),
            "top_attended_in_gold_rate": mean_optional(
                [1.0 if row["top_attended_in_gold"] else 0.0 for row in selected]
            ),
            "top_attended_in_top1_rate": mean_optional(
                [1.0 if row["top_attended_in_top1"] else 0.0 for row in selected]
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit HyperedgeReadout attention on a small case batch."
    )
    parser.add_argument(
        "--train_config_path",
        default="configs/train_finetune_attn_idf_main.yaml",
    )
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--max_cases", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.train_config_path)
    device = resolve_device(config, args.device)
    static_graph = load_static_graph(
        hpo_index_path=config["paths"]["hpo_index_path"],
        disease_index_path=config["paths"]["disease_index_path"],
        disease_incidence_path=config["paths"]["disease_incidence_path"],
    )
    idx_to_hpo = {int(idx): str(hpo_id) for hpo_id, idx in static_graph["hpo_to_idx"].items()}
    idx_to_disease = {
        int(idx): str(disease_id) for disease_id, idx in static_graph["disease_to_idx"].items()
    }

    model, resolved_checkpoint_path, _checkpoint = load_checkpoint_model(
        train_config=config,
        checkpoint_path=args.checkpoint_path,
        num_hpo=int(static_graph["num_hpo"]),
        device=device,
    )
    if not hasattr(model.readout, "return_attention"):
        raise TypeError("Model readout does not expose return_attention.")
    model.readout.return_attention = True

    audit_df = build_audit_batch(config, split=args.split, max_cases=int(args.max_cases))
    loader = CaseBatchLoader(
        df=audit_df,
        batch_size=int(args.max_cases),
        sampler_mode="natural",
    )
    batch_df = loader.get_batch(0)
    batch_graph = build_batch_hypergraph(
        case_df=batch_df,
        hpo_to_idx=static_graph["hpo_to_idx"],
        disease_to_idx=static_graph["disease_to_idx"],
        H_disease=static_graph["H_disease"],
        top_50_hpos=static_graph.get("top_50_hpos", []),
        hpo_dropout_prob=0.0,
        hpo_corruption_prob=0.0,
        include_combined_h=False,
        case_noise_control=config.get("case_noise_control"),
        hpo_specificity=static_graph.get("hpo_specificity"),
        verbose=False,
    )

    with torch.inference_mode():
        outputs = model(batch_graph, return_intermediate=True)

    edge_index = outputs["edge_index"].detach().cpu().numpy()
    edge_attention = outputs["edge_attention"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu()
    top1_indices = scores.argmax(dim=1).tolist()
    gold_indices = outputs["gold_disease_idx_in_score_pool"].detach().cpu().tolist()

    h_case_csc = batch_graph["H_case"].tocsc()
    h_disease_csr = static_graph["H_disease"].tocsr()
    hpo_specificity = static_graph.get("hpo_specificity")

    edge_lookup = {
        (int(case_idx), int(hpo_idx)): edge_pos
        for edge_pos, (hpo_idx, case_idx) in enumerate(zip(edge_index[0], edge_index[1], strict=True))
    }

    detail_rows: list[dict[str, Any]] = []
    case_summary_rows: list[dict[str, Any]] = []
    for case_idx, case_id in enumerate(batch_graph["case_ids"]):
        gold_idx = int(gold_indices[case_idx])
        top1_idx = int(top1_indices[case_idx])
        col = h_case_csc.getcol(case_idx)
        case_rows: list[dict[str, Any]] = []
        for hpo_idx, h_case_weight in zip(col.indices.tolist(), col.data.tolist(), strict=True):
            edge_pos = edge_lookup[(case_idx, int(hpo_idx))]
            alpha = float(edge_attention[edge_pos])
            row = {
                "case_id": str(case_id),
                "case_local_idx": int(case_idx),
                "hpo_idx": int(hpo_idx),
                "hpo_id": idx_to_hpo[int(hpo_idx)],
                "h_case_weight": float(h_case_weight),
                "attention_alpha": alpha,
                "idf_specificity": (
                    None if hpo_specificity is None else float(hpo_specificity[int(hpo_idx)])
                ),
                "gold_disease_idx": gold_idx,
                "gold_disease_id": idx_to_disease[gold_idx],
                "top1_disease_idx": top1_idx,
                "top1_disease_id": idx_to_disease[top1_idx],
                "is_correct": bool(top1_idx == gold_idx),
                "in_gold_disease_hyperedge": bool(h_disease_csr[int(hpo_idx), gold_idx] != 0),
                "in_top1_disease_hyperedge": bool(h_disease_csr[int(hpo_idx), top1_idx] != 0),
            }
            case_rows.append(row)

        case_rows.sort(key=lambda row: float(row["attention_alpha"]), reverse=True)
        for rank, row in enumerate(case_rows, start=1):
            row["alpha_rank"] = rank
            detail_rows.append(row)
        case_summary_rows.append(summarize_case(case_rows))

    summary = {
        "train_config_path": str(Path(args.train_config_path).resolve()),
        "checkpoint_path": str(resolved_checkpoint_path.resolve()),
        "split": args.split,
        "max_cases": int(args.max_cases),
        "num_cases": len(case_summary_rows),
        "readout": {
            "return_attention": bool(model.readout.return_attention),
            "residual_uniform": float(model.readout.residual_uniform),
            "attn_prior_mode": str(model.readout.attn_prior_mode),
            "attn_prior_beta": (
                float(model.readout.attn_prior_beta.detach().cpu().item())
                if isinstance(model.readout.attn_prior_beta, torch.Tensor)
                else float(model.readout.attn_prior_beta)
            ),
            "attn_prior_normalize": str(model.readout.attn_prior_normalize),
        },
        "case_noise_control": config.get("case_noise_control"),
        "groups": summarize_groups(case_summary_rows),
    }

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(config["paths"]["save_dir"]) / "attention_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "attention_edges.csv"
    case_path = output_dir / "attention_cases.csv"
    summary_path = output_dir / "attention_summary.json"

    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(case_summary_rows).to_csv(case_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"attention_edges_csv={detail_path}")
    print(f"attention_cases_csv={case_path}")
    print(f"attention_summary_json={summary_path}")
    print(json.dumps(summary["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
