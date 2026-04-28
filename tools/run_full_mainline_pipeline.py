from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dynamic_lodo_adapter import (
    DEFAULT_LODO_DATASETS,
    DynamicLodoCandidateData,
    build_pairwise_tensor_dataset,
    hgnn_baseline_ranks,
    load_dynamic_lodo_candidates,
    metrics_by_dataset as dynamic_metrics_by_dataset,
    ranks_from_candidate_scores,
    standardize_meta_features,
)
from src.training.dynamic_router_lodo import DynamicMetaRouter, PairwiseMarginRankingLoss
from tools.run_top50_evidence_rerank import load_candidates, load_weight_payload, ranks_from_scores, score_matrix, to_matrix


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mainline_full_pipeline.yaml"
MIMIC_DEFAULT_ALIASES = ["mimic_test", "mimic_test_recleaned", "mimic_test_recleaned_mondo_hpo_rows"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full RareDisease HGNN mainline pipeline.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["full", "eval_only", "dynamic_lodo"], default="full")
    parser.add_argument("--dynamic-epochs", type=int, default=8)
    parser.add_argument("--dynamic-batch-size", type=int, default=32)
    parser.add_argument("--dynamic-lr", type=float, default=1e-3)
    parser.add_argument("--dynamic-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dynamic-margin", type=float, default=0.1)
    parser.add_argument("--dynamic-negatives", type=int, default=10)
    parser.add_argument("--dynamic-hidden-dim", type=int, default=16)
    parser.add_argument("--dynamic-dropout", type=float, default=0.3)
    parser.add_argument("--dynamic-hgnn-prior-blend", type=float, default=0.75)
    parser.add_argument("--dynamic-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dynamic-seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(command: list[str], *, cwd: Path, manifest: dict[str, Any], step: str) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[{step}] {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    finished_at = datetime.now().isoformat(timespec="seconds")
    manifest.setdefault("commands", []).append(
        {
            "step": step,
            "command": command,
            "returncode": int(completed.returncode),
            "started_at": started_at,
            "finished_at": finished_at,
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {step}, returncode={completed.returncode}")


def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {directory}")
    return files[-1]


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".metadata.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain a mapping: {path}")
    return payload


def normalize_objective(value: str) -> str:
    normalized = str(value).strip()
    if normalized.lower() == "ddd_top1":
        return "DDD_top1"
    return normalized


def metric_from_ranks(ranks: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(ranks, errors="coerce").fillna(9999).to_numpy(dtype=int)
    return {
        "cases": int(arr.size),
        "top1": float(np.mean(arr <= 1)) if arr.size else float("nan"),
        "top3": float(np.mean(arr <= 3)) if arr.size else float("nan"),
        "top5": float(np.mean(arr <= 5)) if arr.size else float("nan"),
        "rank_le_50": float(np.mean(arr <= 50)) if arr.size else float("nan"),
    }


def metrics_by_dataset(case_ranks: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in case_ranks.groupby("dataset", sort=True):
        rows.append({"dataset": dataset, **metric_from_ranks(group[rank_col])})
    rows.append({"dataset": "ALL", **metric_from_ranks(case_ranks[rank_col])})
    return pd.DataFrame(rows)


def load_exact_case_ranks(details_path: Path) -> pd.DataFrame:
    df = pd.read_csv(details_path, dtype={"case_id": str, "dataset_name": str, "true_label": str})
    required = {"case_id", "dataset_name", "true_label", "true_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{details_path} missing required columns: {sorted(missing)}")
    out = df[["case_id", "dataset_name", "true_label", "true_rank"]].copy()
    out = out.rename(columns={"dataset_name": "dataset", "true_label": "gold_id", "true_rank": "baseline_rank"})
    out["baseline_rank"] = pd.to_numeric(out["baseline_rank"], errors="coerce").fillna(9999).astype(int)
    return out


def load_ddd_ranks(test_candidates_path: Path, weights_path: Path, objective: str, dataset: str = "DDD") -> pd.DataFrame:
    matrix = to_matrix(load_candidates(test_candidates_path))
    payload = load_weight_payload(weights_path, objective=objective)
    ranks = ranks_from_scores(matrix, score_matrix(matrix, payload["weights"]))
    out = pd.DataFrame({"case_id": matrix.case_ids, "dataset": matrix.dataset_names, "ddd_rank": ranks})
    return out.loc[out["dataset"].eq(dataset)].copy()


def load_mimic_ranks(ranked_candidates_path: Path) -> pd.DataFrame:
    ranked = pd.read_csv(ranked_candidates_path, dtype={"case_id": str, "gold_id": str, "candidate_id": str})
    required = {"case_id", "gold_id", "candidate_id", "rank"}
    missing = required - set(ranked.columns)
    if missing:
        raise ValueError(f"{ranked_candidates_path} missing required columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for case_id, group in ranked.groupby("case_id", sort=False):
        gold_id = str(group["gold_id"].iloc[0])
        hits = group.loc[group["candidate_id"].astype(str).eq(gold_id), "rank"]
        rank = int(pd.to_numeric(hits, errors="coerce").min()) if not hits.empty else 9999
        rows.append({"case_id": str(case_id), "case_key": str(case_id).rsplit("::", 1)[-1], "mimic_rank": rank})
    return pd.DataFrame(rows)


def source_for_dataset(
    dataset: str,
    *,
    ddd_dataset: str,
    mimic_aliases: list[str],
    exact_details_path: Path,
    ddd_weights_path: Path,
    mimic_metrics_path: Path,
    checkpoint_path: Path,
    data_config_path: Path,
    train_config_path: Path,
) -> dict[str, Any]:
    module = "hgnn_exact_baseline"
    source_path = exact_details_path
    if dataset == ddd_dataset:
        module = "ddd_validation_selected_grid_rerank"
        source_path = ddd_weights_path
    elif dataset in mimic_aliases:
        module = "similar_case_fixed_test"
        source_path = mimic_metrics_path
    return {
        "module_applied": module,
        "source_result_path": str(source_path.resolve()),
        "source_dataset_name": dataset,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "data_config_path": str(data_config_path.resolve()),
        "train_config_path": str(train_config_path.resolve()),
    }


def aggregate_final_metrics(
    *,
    exact_details_path: Path,
    exact_summary_path: Path,
    test_candidates_path: Path,
    ddd_weights_path: Path,
    mimic_ranked_path: Path,
    mimic_metrics_path: Path,
    output_dir: Path,
    data_config_path: Path,
    train_config_path: Path,
    ddd_objective: str,
    mimic_aliases: list[str],
) -> dict[str, Path]:
    summary = load_json(exact_summary_path)
    checkpoint_path = resolve_path(summary["checkpoint_path"])
    baseline = load_exact_case_ranks(exact_details_path)
    baseline["final_rank"] = baseline["baseline_rank"]
    baseline["module_applied"] = "hgnn_exact_baseline"

    ddd_ranks = load_ddd_ranks(test_candidates_path, ddd_weights_path, objective=ddd_objective)
    final = baseline.merge(ddd_ranks[["case_id", "ddd_rank"]], on="case_id", how="left")
    ddd_mask = final["dataset"].eq("DDD") & final["ddd_rank"].notna()
    final.loc[ddd_mask, "final_rank"] = final.loc[ddd_mask, "ddd_rank"].astype(int)
    final.loc[ddd_mask, "module_applied"] = "ddd_validation_selected_grid_rerank"

    mimic_ranks = load_mimic_ranks(mimic_ranked_path)
    final["case_key"] = final["case_id"].astype(str).str.rsplit("::", n=1).str[-1]
    final = final.merge(mimic_ranks[["case_id", "mimic_rank"]], on="case_id", how="left")
    if final.loc[final["dataset"].isin(mimic_aliases), "mimic_rank"].isna().all():
        keyed = mimic_ranks[["case_key", "mimic_rank"]].drop_duplicates(subset=["case_key"])
        final = final.drop(columns=["mimic_rank"]).merge(keyed, on="case_key", how="left")
    mimic_mask = final["dataset"].isin(mimic_aliases) & final["mimic_rank"].notna()
    final.loc[mimic_mask, "final_rank"] = final.loc[mimic_mask, "mimic_rank"].astype(int)
    final.loc[mimic_mask, "module_applied"] = "similar_case_fixed_test"
    final["final_rank"] = pd.to_numeric(final["final_rank"], errors="coerce").fillna(9999).astype(int)

    metrics = metrics_by_dataset(final, "final_rank")
    source_rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        dataset = str(row["dataset"])
        if dataset == "ALL":
            source_rows.append(
                {
                    **row.to_dict(),
                    "module_applied": "mixed",
                    "source_result_path": "mixed",
                    "source_dataset_name": "ALL",
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "data_config_path": str(data_config_path.resolve()),
                    "train_config_path": str(train_config_path.resolve()),
                }
            )
            continue
        source_rows.append(
            {
                **row.to_dict(),
                **source_for_dataset(
                    dataset,
                    ddd_dataset="DDD",
                    mimic_aliases=mimic_aliases,
                    exact_details_path=exact_details_path,
                    ddd_weights_path=ddd_weights_path,
                    mimic_metrics_path=mimic_metrics_path,
                    checkpoint_path=checkpoint_path,
                    data_config_path=data_config_path,
                    train_config_path=train_config_path,
                ),
            }
        )
    metrics_with_sources = pd.DataFrame(source_rows)

    metrics_path = output_dir / "mainline_final_metrics.csv"
    metrics_sources_path = output_dir / "mainline_final_metrics_with_sources.csv"
    case_ranks_path = output_dir / "mainline_final_case_ranks.csv"
    write_csv(metrics, metrics_path)
    write_csv(metrics_with_sources, metrics_sources_path)
    write_csv(final, case_ranks_path)
    return {
        "metrics": metrics_path,
        "metrics_with_sources": metrics_sources_path,
        "case_ranks": case_ranks_path,
    }


def resolve_dynamic_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[dynamic_lodo] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def router_fused_scores(weights: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    if evidence.dim() == 2:
        return torch.sum(weights * evidence, dim=-1, keepdim=True)
    if evidence.dim() == 3:
        return torch.sum(weights.unsqueeze(1) * evidence, dim=-1)
    raise ValueError(f"Unsupported evidence tensor shape: {tuple(evidence.shape)}")


def apply_hgnn_prior_blend(weights: torch.Tensor, blend: float) -> torch.Tensor:
    blend = min(max(float(blend), 0.0), 1.0)
    if blend <= 0.0:
        return weights
    prior = torch.zeros_like(weights)
    prior[:, 0] = 1.0
    return (1.0 - blend) * weights + blend * prior


def train_dynamic_router(
    *,
    dataset: torch.utils.data.TensorDataset,
    meta_dim: int,
    evidence_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    fold_name: str,
) -> tuple[DynamicMetaRouter, list[dict[str, float]]]:
    generator = torch.Generator()
    generator.manual_seed(int(args.dynamic_seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.dynamic_batch_size),
        shuffle=True,
        generator=generator,
    )
    model = DynamicMetaRouter(
        meta_feat_dim=meta_dim,
        hidden_dim=int(args.dynamic_hidden_dim),
        num_evidence=evidence_dim,
        dropout=float(args.dynamic_dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.dynamic_lr),
        weight_decay=float(args.dynamic_weight_decay),
    )
    criterion = PairwiseMarginRankingLoss(margin=float(args.dynamic_margin))
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(1, int(args.dynamic_epochs) + 1):
        total_loss = 0.0
        total_batches = 0
        weight_vars: list[float] = []
        for meta_features, pos_evidence, neg_evidence in loader:
            meta_features = meta_features.to(device)
            pos_evidence = pos_evidence.to(device)
            neg_evidence = neg_evidence.to(device)

            optimizer.zero_grad(set_to_none=True)
            weights = apply_hgnn_prior_blend(model(meta_features), float(args.dynamic_hgnn_prior_blend))
            pos_scores = router_fused_scores(weights, pos_evidence)
            neg_scores = router_fused_scores(weights, neg_evidence)
            loss = criterion(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            weight_vars.append(float(torch.var(weights.detach(), dim=0).mean().item()) if weights.shape[0] > 1 else 0.0)

        avg_loss = total_loss / max(total_batches, 1)
        avg_weight_var = float(np.mean(weight_vars)) if weight_vars else 0.0
        history.append({"epoch": float(epoch), "loss": avg_loss, "weight_variance": avg_weight_var})
        print(
            f"[dynamic_lodo] fold={fold_name} epoch={epoch}/{args.dynamic_epochs} "
            f"loss={avg_loss:.6f} weight_var={avg_weight_var:.8f}"
        )
        if avg_weight_var < 1e-5:
            print(f"[dynamic_lodo] WARNING fold={fold_name}: router weights are close to static.")

    return model, history


def infer_dynamic_weights(
    *,
    model: DynamicMetaRouter,
    x_meta: np.ndarray,
    case_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    hgnn_prior_blend: float,
) -> np.ndarray:
    model.eval()
    weights: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(case_indices), batch_size):
            batch_indices = case_indices[start : start + batch_size]
            meta_tensor = torch.tensor(x_meta[batch_indices], dtype=torch.float32, device=device)
            batch_weights = apply_hgnn_prior_blend(model(meta_tensor), hgnn_prior_blend)
            weights.append(batch_weights.cpu().numpy())
    return np.concatenate(weights, axis=0) if weights else np.empty((0, 0), dtype=np.float32)


def static_grid_ranks_for_dynamic_cases(
    *,
    candidates_path: Path,
    weights_path: Path,
    objective: str,
    dynamic_data: DynamicLodoCandidateData,
) -> np.ndarray:
    matrix = to_matrix(load_candidates(candidates_path))
    payload = load_weight_payload(weights_path, objective=objective)
    ranks = ranks_from_scores(matrix, score_matrix(matrix, payload["weights"]))
    rank_by_case = {str(case_id): int(rank) for case_id, rank in zip(matrix.case_ids, ranks, strict=True)}
    return np.asarray([rank_by_case.get(str(case_id), dynamic_data.top_k + 1) for case_id in dynamic_data.case_ids], dtype=int)


def build_dynamic_ranked_candidates(
    *,
    data: DynamicLodoCandidateData,
    scores: np.ndarray,
    weights_by_case: np.ndarray,
    fold_by_case: dict[str, str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for case_idx, case_id in enumerate(data.case_ids):
        order = np.lexsort((data.original_rank[case_idx], -scores[case_idx]))
        offsets = case_idx * data.top_k + order
        case_rows = data.source_frame.iloc[offsets].copy()
        case_rows["dynamic_lodo_score"] = scores[case_idx, order]
        case_rows["dynamic_lodo_rank"] = np.arange(1, len(order) + 1, dtype=int)
        case_rows["dynamic_lodo_fold"] = fold_by_case[str(case_id)]
        for weight_idx, evidence_name in enumerate(data.evidence_feature_names):
            case_rows[f"dynamic_w_{evidence_name}"] = float(weights_by_case[case_idx, weight_idx])
        rows.append(case_rows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metrics_to_markdown(df: pd.DataFrame) -> str:
    columns = ["dataset", "method", "cases", "top1", "top3", "top5", "recall_at_50"]
    header = "| " + " | ".join(["Dataset", "Method", "Cases", "Top-1", "Top-3", "Top-5", "Recall@50"]) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in df[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    str(int(row["cases"])),
                    f"{float(row['top1']):.4f}",
                    f"{float(row['top3']):.4f}",
                    f"{float(row['top5']):.4f}",
                    f"{float(row['recall_at_50']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run_dynamic_lodo_pipeline(
    *,
    test_candidates_path: Path,
    static_weights_path: Path,
    output_dir: Path,
    objective: str,
    mimic_aliases: list[str],
    args: argparse.Namespace,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_dynamic_device(str(args.dynamic_device))
    torch.manual_seed(int(args.dynamic_seed))
    np.random.seed(int(args.dynamic_seed))

    data = load_dynamic_lodo_candidates(
        test_candidates_path,
        mimic_aliases=mimic_aliases,
        include_datasets=DEFAULT_LODO_DATASETS,
    )
    print(
        "[dynamic_lodo] loaded cases: "
        + ", ".join(
            f"{dataset}={int(np.sum(data.dataset_names == dataset))}"
            for dataset in DEFAULT_LODO_DATASETS
            if np.any(data.dataset_names == dataset)
        )
    )
    print(f"[dynamic_lodo] meta_features={list(data.meta_feature_names)}")
    print(f"[dynamic_lodo] evidence_features={list(data.evidence_feature_names)}")
    print(f"[dynamic_lodo] hgnn_prior_blend={float(args.dynamic_hgnn_prior_blend):.3f}")

    dynamic_ranks = np.full(data.num_cases, data.top_k + 1, dtype=int)
    dynamic_scores = np.zeros((data.num_cases, data.top_k), dtype=np.float32)
    dynamic_weights = np.zeros((data.num_cases, len(data.evidence_feature_names)), dtype=np.float32)
    fold_by_case: dict[str, str] = {}
    fold_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for test_dataset in DEFAULT_LODO_DATASETS:
        test_indices = np.flatnonzero(data.dataset_names == test_dataset)
        if test_indices.size == 0:
            continue
        train_indices = np.flatnonzero(data.dataset_names != test_dataset)
        x_meta_scaled, meta_mean, meta_std = standardize_meta_features(data.x_meta, train_indices)
        pairwise_dataset = build_pairwise_tensor_dataset(
            data,
            train_indices,
            x_meta=x_meta_scaled,
            max_negatives_per_case=int(args.dynamic_negatives),
        )
        print(
            f"[dynamic_lodo] fold={test_dataset} train_cases={len(train_indices)} "
            f"pairwise_cases={len(pairwise_dataset)} test_cases={len(test_indices)} device={device}"
        )
        model, history = train_dynamic_router(
            dataset=pairwise_dataset,
            meta_dim=len(data.meta_feature_names),
            evidence_dim=len(data.evidence_feature_names),
            args=args,
            device=device,
            fold_name=test_dataset,
        )
        for item in history:
            history_rows.append({"fold": test_dataset, **item})

        fold_weights = infer_dynamic_weights(
            model=model,
            x_meta=x_meta_scaled,
            case_indices=test_indices,
            device=device,
            batch_size=int(args.dynamic_batch_size),
            hgnn_prior_blend=float(args.dynamic_hgnn_prior_blend),
        )
        fold_scores = np.einsum("ce,cke->ck", fold_weights, data.x_evidence[test_indices])
        score_buffer = np.zeros((data.num_cases, data.top_k), dtype=np.float32)
        score_buffer[test_indices] = fold_scores
        fold_ranks = ranks_from_candidate_scores(data, score_buffer)[test_indices]

        dynamic_ranks[test_indices] = fold_ranks
        dynamic_scores[test_indices] = fold_scores
        dynamic_weights[test_indices] = fold_weights
        for case_id in data.case_ids[test_indices]:
            fold_by_case[str(case_id)] = test_dataset

        fold_rows.append(
            {
                "fold": test_dataset,
                "train_cases": int(len(train_indices)),
                "pairwise_cases": int(len(pairwise_dataset)),
                "test_cases": int(len(test_indices)),
                "test_top1": float(np.mean(fold_ranks <= 1)),
                "test_top3": float(np.mean(fold_ranks <= 3)),
                "test_top5": float(np.mean(fold_ranks <= 5)),
                "test_recall_at_50": float(np.mean(fold_ranks <= 50)),
                "test_weight_variance": float(np.var(fold_weights, axis=0).mean()) if len(fold_weights) > 1 else 0.0,
                "mean_weights": {
                    name: float(value)
                    for name, value in zip(data.evidence_feature_names, fold_weights.mean(axis=0), strict=True)
                },
                "meta_mean": {name: float(value) for name, value in zip(data.meta_feature_names, meta_mean, strict=True)},
                "meta_std": {name: float(value) for name, value in zip(data.meta_feature_names, meta_std, strict=True)},
            }
        )

    hgnn_ranks = hgnn_baseline_ranks(data)
    static_ranks = static_grid_ranks_for_dynamic_cases(
        candidates_path=test_candidates_path,
        weights_path=static_weights_path,
        objective=objective,
        dynamic_data=data,
    )
    baseline_metrics = dynamic_metrics_by_dataset(data, hgnn_ranks, method="HGNN Baseline")
    static_metrics = dynamic_metrics_by_dataset(data, static_ranks, method="Static Grid")
    dynamic_metrics = dynamic_metrics_by_dataset(data, dynamic_ranks, method="Dynamic LODO")
    metrics = pd.concat([baseline_metrics, static_metrics, dynamic_metrics], ignore_index=True)

    ranked_candidates = build_dynamic_ranked_candidates(
        data=data,
        scores=dynamic_scores,
        weights_by_case=dynamic_weights,
        fold_by_case=fold_by_case,
    )
    case_ranks = pd.DataFrame(
        {
            "case_id": data.case_ids,
            "dataset": data.dataset_names,
            "gold_id": data.gold_ids,
            "hgnn_rank": hgnn_ranks,
            "static_grid_rank": static_ranks,
            "dynamic_lodo_rank": dynamic_ranks,
        }
    )

    metrics_path = output_dir / "dynamic_lodo_metrics.csv"
    case_ranks_path = output_dir / "dynamic_lodo_case_ranks.csv"
    ranked_candidates_path = output_dir / "dynamic_lodo_ranked_candidates.csv"
    folds_path = output_dir / "dynamic_lodo_folds.json"
    history_path = output_dir / "dynamic_lodo_training_history.csv"
    markdown_path = output_dir / "dynamic_lodo_metrics.md"

    write_csv(metrics, metrics_path)
    write_csv(case_ranks, case_ranks_path)
    write_csv(ranked_candidates, ranked_candidates_path)
    write_csv(pd.DataFrame(history_rows), history_path)
    write_json({"folds": fold_rows}, folds_path)
    markdown = metrics_to_markdown(metrics)
    write_text(markdown_path, markdown + "\n")
    print("\n" + markdown)

    return {
        "metrics": metrics_path,
        "case_ranks": case_ranks_path,
        "ranked_candidates": ranked_candidates_path,
        "folds": folds_path,
        "training_history": history_path,
        "markdown": markdown_path,
    }


def assert_same_checkpoint(candidate_path: Path, checkpoint_path: Path) -> None:
    meta_path = metadata_path(candidate_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Candidate metadata not found: {meta_path}")
    metadata = load_json(meta_path)
    candidate_checkpoint = resolve_path(metadata["checkpoint_path"])
    if candidate_checkpoint.resolve() != checkpoint_path.resolve():
        raise ValueError(
            f"Checkpoint mismatch for {candidate_path}: "
            f"{candidate_checkpoint} != {checkpoint_path}"
        )


def copy_stable_eval_outputs(eval_dir: Path, stage_dir: Path) -> dict[str, Path]:
    details = latest_file(eval_dir, "*_details.csv")
    summary = latest_file(eval_dir, "*_summary.json")
    per_dataset = latest_file(eval_dir, "*_per_dataset.csv")
    stable = {
        "details": stage_dir / "exact_details.csv",
        "summary": stage_dir / "exact_summary.json",
        "per_dataset": stage_dir / "exact_per_dataset.csv",
    }
    shutil.copy2(details, stable["details"])
    shutil.copy2(summary, stable["summary"])
    shutil.copy2(per_dataset, stable["per_dataset"])
    return stable


def build_stage_configs(config: dict[str, Any], output_dir: Path, mode: str) -> dict[str, Path]:
    paths_cfg = config["paths"]
    config_dir = output_dir / "configs"
    stage1 = output_dir / "stage1_pretrain"
    stage2 = output_dir / "stage2_finetune"
    stage3 = output_dir / "stage3_exact_eval"

    pretrain = load_yaml(resolve_path(paths_cfg["pretrain_config"]))
    pretrain.setdefault("paths", {})["save_dir"] = str(stage1.resolve())
    pretrain_config_path = config_dir / "stage1_pretrain.yaml"
    write_yaml(pretrain, pretrain_config_path)

    finetune = load_yaml(resolve_path(paths_cfg["finetune_config"]))
    finetune.setdefault("paths", {})["save_dir"] = str(stage2.resolve())
    resume = config.get("resume", {}) if isinstance(config.get("resume", {}), dict) else {}
    pretrain_checkpoint = resume.get("pretrain_checkpoint") or str((stage1 / "checkpoints" / "best.pt").resolve())
    finetune.setdefault("train", {})["init_checkpoint_path"] = str(resolve_path(pretrain_checkpoint))
    finetune_config_path = config_dir / "stage2_finetune.yaml"
    write_yaml(finetune, finetune_config_path)

    eval_train = dict(finetune)
    eval_train["paths"] = dict(finetune["paths"])
    eval_train["paths"]["save_dir"] = str(stage3.resolve())
    eval_train_config_path = config_dir / "stage3_exact_eval_train.yaml"
    write_yaml(eval_train, eval_train_config_path)

    return {
        "pretrain": pretrain_config_path,
        "finetune": finetune_config_path,
        "eval_train": eval_train_config_path,
    }


def write_readme(path: Path, output_dir: Path) -> None:
    text = f"""# RareDisease HGNN full mainline pipeline

## 一键运行命令

```cmd
D:\\python\\python.exe tools\\run_full_mainline_pipeline.py --config-path configs\\mainline_full_pipeline.yaml --mode full
```

已有 finetune checkpoint 时只重跑评估和后处理：

```cmd
D:\\python\\python.exe tools\\run_full_mainline_pipeline.py --config-path configs\\mainline_full_pipeline.yaml --mode eval_only
```

## 每一步做了什么

1. `stage1_pretrain`: 调用 `python -m src.training.trainer` 运行 pretrain，并把 `save_dir` 写到 `{output_dir / "stage1_pretrain"}`。
2. `stage2_finetune`: 调用 `python -m src.training.trainer` 运行 finetune，`init_checkpoint_path` 指向 stage1 的 `best.pt`。
3. `stage3_exact_eval`: 调用 `python -m src.evaluation.evaluator`，使用 stage2 的 `best.pt` 生成 exact evaluation。
4. `stage4_candidates`: 调用 `tools/export_top50_candidates.py`，分别导出 validation/test top50 candidates。
5. `stage5_ddd_rerank`: 调用 `tools/run_top50_evidence_rerank.py --protocol validation_select`，只在 validation candidates 上选权重，再对 test candidates 固定评估一次。
6. `stage6_mimic_similar_case`: 调用 `tools/run_mimic_similar_case_aug.py`，使用 validation-selected fixed SimilarCase 参数，对 test candidates 固定评估一次。
7. final aggregation: DDD 使用 rerank，mimic 使用 SimilarCase，其他数据集使用 exact baseline。

## 为什么 DDD 是评估后 rerank

DDD 模块只读取 evaluation 后导出的 top50 candidates 和 validation 选择出的固定权重，不改变 HGNN encoder、loss、sampler 或训练 checkpoint。因此它是 dataset-specific post-processing，不属于 `trainer.py` 的训练逻辑。

## 为什么 mimic 是评估后 SimilarCase

mimic SimilarCase 模块只在 HGNN candidates 上融合相似病例证据，并且参数由 validation 选择后在 test 固定评估。它不反向传播、不更新 checkpoint，也不改变训练数据采样，因此属于 evaluation 后模块。

## 最终主表

论文主表读取：

`outputs/mainline_full_pipeline/mainline_final_metrics_with_sources.csv`

简版指标表是：

`outputs/mainline_full_pipeline/mainline_final_metrics.csv`

## 如何确认没有 checkpoint 混用

检查 `outputs/mainline_full_pipeline/run_manifest.json` 中的 `finetune_checkpoint`、`validation_candidates_metadata.checkpoint_path`、`test_candidates_metadata.checkpoint_path` 和 final table 的 `checkpoint_path`。这些路径必须一致。

## 如何确认 mimic alias 已正确匹配

检查 `mainline_final_metrics_with_sources.csv` 中 `mimic_test`、`mimic_test_recleaned` 或 `mimic_test_recleaned_mondo_hpo_rows` 对应行，`module_applied` 应为 `similar_case_fixed_test`。

## 如何确认 DDD 和 mimic 已进入最终表

检查 `mainline_final_metrics_with_sources.csv`：

- `DDD` 的 `module_applied` 应为 `ddd_validation_selected_grid_rerank`。
- mimic alias 数据集的 `module_applied` 应为 `similar_case_fixed_test`。
- `HMS/LIRICAL/MME/MyGene2/RAMEDIS` 的 `module_applied` 应为 `hgnn_exact_baseline`。
"""
    write_text(path, text)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config_path)
    config = load_yaml(config_path)
    output_dir = resolve_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_dirs = {
        "stage1": output_dir / "stage1_pretrain",
        "stage2": output_dir / "stage2_finetune",
        "stage3": output_dir / "stage3_exact_eval",
        "stage4": output_dir / "stage4_candidates",
        "stage5": output_dir / "stage5_ddd_rerank",
        "stage6": output_dir / "stage6_mimic_similar_case",
        "stage7": output_dir / "stage7_dynamic_lodo",
    }
    for stage_dir in stage_dirs.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "commands": [],
    }
    pipeline = config.get("pipeline", {})
    resume = config.get("resume", {}) if isinstance(config.get("resume", {}), dict) else {}
    stage_configs = build_stage_configs(config, output_dir, args.mode)

    if args.mode == "full" and pipeline.get("run_pretrain", True):
        run_command(
            [sys.executable, "-m", "src.training.trainer", "--config", str(stage_configs["pretrain"])],
            cwd=PROJECT_ROOT,
            manifest=manifest,
            step="stage1_pretrain",
        )

    pretrain_best = stage_dirs["stage1"] / "checkpoints" / "best.pt"
    if args.mode == "full" and pipeline.get("run_finetune", True):
        if not pretrain_best.is_file():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_best}")
        run_command(
            [sys.executable, "-m", "src.training.trainer", "--config", str(stage_configs["finetune"])],
            cwd=PROJECT_ROOT,
            manifest=manifest,
            step="stage2_finetune",
        )

    if args.mode in {"eval_only", "dynamic_lodo"}:
        resume_checkpoint = resume.get("finetune_checkpoint")
        finetune_checkpoint = resolve_path(resume_checkpoint) if resume_checkpoint else stage_dirs["stage2"] / "checkpoints" / "best.pt"
    else:
        finetune_checkpoint = stage_dirs["stage2"] / "checkpoints" / "best.pt"
    if not finetune_checkpoint.is_file():
        raise FileNotFoundError(f"Finetune checkpoint not found: {finetune_checkpoint}")
    manifest["finetune_checkpoint"] = str(finetune_checkpoint.resolve())

    data_config_path = resolve_path(config["paths"]["data_eval_config"])
    validation_candidates = stage_dirs["stage4"] / "top50_candidates_validation.csv"
    test_candidates = stage_dirs["stage4"] / "top50_candidates_test.csv"
    ddd_cfg = config.get("postprocess", {}).get("ddd", {})
    ddd_objective = normalize_objective(ddd_cfg.get("objective", "DDD_top1"))
    ddd_weights = stage_dirs["stage5"] / "ddd_val_selected_grid_weights.json"
    mimic_cfg = config.get("postprocess", {}).get("mimic", {})
    mimic_aliases = [str(value) for value in mimic_cfg.get("dataset_aliases", MIMIC_DEFAULT_ALIASES)]

    if args.mode == "dynamic_lodo":
        for case_source, output_path, step_name in [
            ("validation", validation_candidates, "stage4_candidates_validation"),
            ("test", test_candidates, "stage4_candidates_test"),
        ]:
            if output_path.is_file():
                continue
            run_command(
                [
                    sys.executable,
                    "tools/export_top50_candidates.py",
                    "--data-config-path",
                    str(data_config_path),
                    "--train-config-path",
                    str(stage_configs["finetune"]),
                    "--checkpoint-path",
                    str(finetune_checkpoint),
                    "--output-path",
                    str(output_path),
                    "--top-k",
                    "50",
                    "--case-source",
                    case_source,
                ],
                cwd=PROJECT_ROOT,
                manifest=manifest,
                step=step_name,
            )
        assert_same_checkpoint(validation_candidates, finetune_checkpoint)
        assert_same_checkpoint(test_candidates, finetune_checkpoint)
        manifest["validation_candidates_metadata"] = load_json(metadata_path(validation_candidates))
        manifest["test_candidates_metadata"] = load_json(metadata_path(test_candidates))

        if not ddd_weights.is_file():
            run_command(
                [
                    sys.executable,
                    "tools/run_top50_evidence_rerank.py",
                    "--protocol",
                    "validation_select",
                    "--validation-candidates-path",
                    str(validation_candidates),
                    "--test-candidates-path",
                    str(test_candidates),
                    "--output-dir",
                    str(stage_dirs["stage5"]),
                    "--selected-weights-path",
                    str(ddd_weights),
                    "--selection-objective",
                    ddd_objective,
                ],
                cwd=PROJECT_ROOT,
                manifest=manifest,
                step="stage5_ddd_rerank",
            )

        dynamic_paths = run_dynamic_lodo_pipeline(
            test_candidates_path=test_candidates,
            static_weights_path=ddd_weights,
            output_dir=stage_dirs["stage7"],
            objective=ddd_objective,
            mimic_aliases=mimic_aliases,
            args=args,
        )
        manifest["dynamic_lodo_outputs"] = {key: str(value.resolve()) for key, value in dynamic_paths.items()}
        manifest["stage_configs"] = {key: str(value.resolve()) for key, value in stage_configs.items()}
        manifest["stage_dirs"] = {key: str(value.resolve()) for key, value in stage_dirs.items()}
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(manifest, output_dir / "run_manifest.json")
        print(
            json.dumps(
                {
                    "output_dir": str(stage_dirs["stage7"].resolve()),
                    "dynamic_lodo_metrics": str(dynamic_paths["metrics"].resolve()),
                    "dynamic_lodo_markdown": str(dynamic_paths["markdown"].resolve()),
                    "run_manifest": str((output_dir / "run_manifest.json").resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    exact_outputs: dict[str, Path] = {
        "details": stage_dirs["stage3"] / "exact_details.csv",
        "summary": stage_dirs["stage3"] / "exact_summary.json",
        "per_dataset": stage_dirs["stage3"] / "exact_per_dataset.csv",
    }
    if pipeline.get("run_exact_eval", True):
        run_command(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluator",
                "--data_config_path",
                str(data_config_path),
                "--train_config_path",
                str(stage_configs["eval_train"]),
                "--checkpoint_path",
                str(finetune_checkpoint),
            ],
            cwd=PROJECT_ROOT,
            manifest=manifest,
            step="stage3_exact_eval",
        )
        exact_outputs = copy_stable_eval_outputs(stage_dirs["stage3"] / "evaluation", stage_dirs["stage3"])

    if pipeline.get("run_candidate_export", True):
        for case_source, output_path, step_name in [
            ("validation", validation_candidates, "stage4_candidates_validation"),
            ("test", test_candidates, "stage4_candidates_test"),
        ]:
            run_command(
                [
                    sys.executable,
                    "tools/export_top50_candidates.py",
                    "--data-config-path",
                    str(data_config_path),
                    "--train-config-path",
                    str(stage_configs["finetune"]),
                    "--checkpoint-path",
                    str(finetune_checkpoint),
                    "--output-path",
                    str(output_path),
                    "--top-k",
                    "50",
                    "--case-source",
                    case_source,
                ],
                cwd=PROJECT_ROOT,
                manifest=manifest,
                step=step_name,
            )

    assert_same_checkpoint(validation_candidates, finetune_checkpoint)
    assert_same_checkpoint(test_candidates, finetune_checkpoint)
    manifest["validation_candidates_metadata"] = load_json(metadata_path(validation_candidates))
    manifest["test_candidates_metadata"] = load_json(metadata_path(test_candidates))

    if ddd_cfg.get("enabled", True) and pipeline.get("run_ddd_rerank", True):
        run_command(
            [
                sys.executable,
                "tools/run_top50_evidence_rerank.py",
                "--protocol",
                "validation_select",
                "--validation-candidates-path",
                str(validation_candidates),
                "--test-candidates-path",
                str(test_candidates),
                "--output-dir",
                str(stage_dirs["stage5"]),
                "--selected-weights-path",
                str(ddd_weights),
                "--selection-objective",
                ddd_objective,
            ],
            cwd=PROJECT_ROOT,
            manifest=manifest,
            step="stage5_ddd_rerank",
        )

    mimic_ranked = stage_dirs["stage6"] / "similar_case_fixed_test_ranked_candidates.csv"
    mimic_metrics = stage_dirs["stage6"] / "similar_case_fixed_test.csv"
    if mimic_cfg.get("enabled", True) and pipeline.get("run_mimic_similar_case", True):
        run_command(
            [
                sys.executable,
                "tools/run_mimic_similar_case_aug.py",
                "--data-config-path",
                str(data_config_path),
                "--train-config-path",
                str(stage_configs["finetune"]),
                "--validation-candidates-path",
                str(validation_candidates),
                "--test-candidates-path",
                str(test_candidates),
                "--output-dir",
                str(stage_dirs["stage6"]),
                "--similarity-device",
                str(mimic_cfg.get("similarity_device", "auto")),
                "--similarity-batch-size",
                str(mimic_cfg.get("similarity_batch_size", 256)),
            ],
            cwd=PROJECT_ROOT,
            manifest=manifest,
            step="stage6_mimic_similar_case",
        )

    if pipeline.get("run_final_aggregation", True):
        final_paths = aggregate_final_metrics(
            exact_details_path=exact_outputs["details"],
            exact_summary_path=exact_outputs["summary"],
            test_candidates_path=test_candidates,
            ddd_weights_path=ddd_weights,
            mimic_ranked_path=mimic_ranked,
            mimic_metrics_path=mimic_metrics,
            output_dir=output_dir,
            data_config_path=data_config_path,
            train_config_path=stage_configs["finetune"],
            ddd_objective=ddd_objective,
            mimic_aliases=mimic_aliases,
        )
        manifest["final_outputs"] = {key: str(value.resolve()) for key, value in final_paths.items()}

    manifest["stage_configs"] = {key: str(value.resolve()) for key, value in stage_configs.items()}
    manifest["stage_dirs"] = {key: str(value.resolve()) for key, value in stage_dirs.items()}
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest, output_dir / "run_manifest.json")
    write_readme(PROJECT_ROOT / "reports" / "mainline" / "full_pipeline_readme.md", output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "final_metrics": str((output_dir / "mainline_final_metrics.csv").resolve()),
                "final_metrics_with_sources": str((output_dir / "mainline_final_metrics_with_sources.csv").resolve()),
                "run_manifest": str((output_dir / "run_manifest.json").resolve()),
                "readme": str((PROJECT_ROOT / "reports" / "mainline" / "full_pipeline_readme.md").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
