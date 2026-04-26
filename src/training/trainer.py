from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_hypergraph import build_batch_hypergraph, load_static_graph
from src.data.dataset import (
    REAL_SOURCE_NAMES,
    SYNTHETIC_SOURCE_NAMES,
    CaseBatchLoader,
    load_case_files,
    normalize_source_name,
)
from src.models.model_pipeline import ModelPipeline
from src.runtime_config import (
    TRUSTED_MAINLINE,
    build_model_pipeline_config,
    build_training_effective_config,
    print_effective_config,
    resolve_loss_config,
    save_yaml,
)
from src.training.hard_negative_miner import mine_configurable_hard_negatives
from src.training.loss_builder import build_loss


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("配置文件内容必须是字典。")
    return config


def list_case_files(train_dir: str | Path) -> list[Path]:
    """扫描训练目录。"""
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")

    file_paths: list[Path] = []
    skipped_files: list[str] = []
    for path in sorted(train_dir.glob("*.xlsx")):
        # 跳过 Excel 临时锁文件
        if path.name.startswith("~$"):
            skipped_files.append(path.name)
            continue
        file_paths.append(path)

    if not file_paths:
        raise FileNotFoundError(f"训练目录下未找到 .xlsx 文件: {train_dir}")
    if skipped_files:
        print(f"跳过临时文件: {', '.join(skipped_files)}")
    return file_paths


def resolve_train_files(paths_cfg: dict[str, Any]) -> list[Path]:
    """解析训练文件列表，优先使用 train_files，否则扫描 train_dir。"""
    if not isinstance(paths_cfg, dict):
        raise TypeError(f"paths_cfg 必须是 dict，当前收到 {type(paths_cfg).__name__}。")

    raw_train_files = paths_cfg.get("train_files")
    if raw_train_files:
        if not isinstance(raw_train_files, list):
            raise TypeError("paths.train_files 必须是列表。")

        file_paths: list[Path] = []
        skipped_files: list[str] = []
        for item in raw_train_files:
            path = Path(item)
            if path.name.startswith("~$"):
                skipped_files.append(path.name)
                continue
            if not path.is_file():
                raise FileNotFoundError(f"训练文件不存在: {path}")
            file_paths.append(path)

        if not file_paths:
            raise FileNotFoundError("paths.train_files 中没有可用的 .xlsx 文件。")
        if skipped_files:
            print(f"跳过临时文件: {', '.join(skipped_files)}")
        return file_paths

    train_dir = paths_cfg.get("train_dir")
    if not train_dir:
        raise KeyError("paths 中必须提供 train_files 或 train_dir。")
    return list_case_files(train_dir)


def load_init_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
) -> Path | None:
    """仅加载初始 checkpoint 的模型权重，不恢复优化器状态。"""
    if checkpoint_path is None:
        return None

    resolved_path = Path(checkpoint_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"init_checkpoint_path 不存在: {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"无法识别 checkpoint 结构: {resolved_path}")

    model_state = model.state_dict()
    compatible_state_dict: dict[str, Any] = {}
    skipped_keys: list[str] = []
    unexpected_keys: list[str] = []

    # readout 升级后，允许跳过这部分不兼容的旧权重。
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if getattr(model_state[key], "shape", None) != getattr(value, "shape", None):
            skipped_keys.append(key)
            continue
        compatible_state_dict[key] = value

    missing_keys, load_unexpected_keys = model.load_state_dict(compatible_state_dict, strict=False)
    unexpected_keys.extend(load_unexpected_keys)

    allowed_prefixes = ("readout.", "case_refiner.")
    disallowed_missing = [key for key in missing_keys if not key.startswith(allowed_prefixes)]
    disallowed_unexpected = [
        key for key in [*unexpected_keys, *skipped_keys] if not key.startswith(allowed_prefixes)
    ]
    if disallowed_missing or disallowed_unexpected:
        raise ValueError(
            f"加载初始 checkpoint 失败，missing_keys={disallowed_missing}, "
            f"unexpected_keys={disallowed_unexpected}"
        )

    relaxed_missing = [key for key in missing_keys if key.startswith(allowed_prefixes)]
    relaxed_skipped = [
        key for key in [*unexpected_keys, *skipped_keys] if key.startswith(allowed_prefixes)
    ]
    if relaxed_missing or relaxed_skipped:
        print(
            "[WARN] init checkpoint 与当前 readout 结构不完全一致，"
            "已跳过相关权重并继续加载。"
        )
    return resolved_path


def split_train_val_by_case(
    df: pd.DataFrame,
    val_ratio: float,
    random_seed: int,
    case_id_col: str = "case_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 case_id 切分训练集和验证集。"""
    if case_id_col not in df.columns:
        raise KeyError(f"数据中缺少列: {case_id_col}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 0 到 1 之间（可以为 0，表示关闭验证集全量训练）。")

    if val_ratio == 0.0:
        return df.copy(), pd.DataFrame(columns=df.columns)

    case_ids = df[case_id_col].dropna().drop_duplicates().tolist()
    if len(case_ids) < 2:
        raise ValueError("病例数量过少，无法切分 train/val。")

    rng = np.random.default_rng(random_seed)
    rng.shuffle(case_ids)

    val_case_count = int(len(case_ids) * val_ratio)
    if val_case_count <= 0:
        raise ValueError("验证集为空，请调大 val_ratio 或检查数据量。")
    if val_case_count >= len(case_ids):
        raise ValueError("训练集为空，请调小 val_ratio 或检查数据量。")

    val_case_ids = set(case_ids[:val_case_count])
    train_case_ids = set(case_ids[val_case_count:])
    train_df = df[df[case_id_col].isin(train_case_ids)].reset_index(drop=True)
    val_df = df[df[case_id_col].isin(val_case_ids)].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("train_df 为空，请检查切分结果。")

    return train_df, val_df


def compute_topk_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """计算 top-k 指标。"""
    if scores.ndim != 2:
        raise ValueError(f"scores 必须是二维张量，当前 shape={tuple(scores.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets 必须是一维张量，当前 shape={tuple(targets.shape)}")
    if scores.shape[0] != targets.shape[0]:
        raise ValueError("scores 和 targets 的 batch 维度不一致。")
    if scores.shape[0] == 0:
        raise ValueError("空 batch 无法计算 top-k。")

    metrics: dict[str, float] = {}
    max_k = min(max(ks), scores.shape[1])
    topk_indices = scores.topk(max_k, dim=1).indices
    expanded_targets = targets.view(-1, 1)

    for k in ks:
        real_k = min(k, scores.shape[1])
        hits = (topk_indices[:, :real_k] == expanded_targets).any(dim=1)
        metrics[f"top{k}"] = hits.float().mean().item()
    return metrics


def _parse_source_list(raw_sources: Any, default_sources: tuple[str, ...]) -> list[str]:
    """解析 validation 配置里的数据源列表。"""
    if raw_sources is None:
        return list(default_sources)
    if not isinstance(raw_sources, list):
        raise TypeError("validation 数据源列表必须是 list。")
    return [str(item) for item in raw_sources]


def resolve_validation_config(config: dict[str, Any]) -> dict[str, list[str]]:
    """解析训练期 source-aware 验证所需的配置。"""
    validation_cfg = config.get("validation", {})
    if validation_cfg is None:
        validation_cfg = {}
    if not isinstance(validation_cfg, dict):
        raise TypeError("validation 配置必须是 dict。")

    return {
        "real_sources": _parse_source_list(validation_cfg.get("real_sources"), REAL_SOURCE_NAMES),
        "synthetic_sources": _parse_source_list(
            validation_cfg.get("synthetic_sources"),
            SYNTHETIC_SOURCE_NAMES,
        ),
    }


def resolve_train_sampler_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """解析训练阶段的 sampler 配置。"""
    sampler_cfg = train_cfg.get("sampler", {})
    if sampler_cfg is None:
        sampler_cfg = {}
    if not isinstance(sampler_cfg, dict):
        raise TypeError("train.sampler 配置必须是 dict。")

    sampler_mode = str(sampler_cfg.get("sampler_mode", "natural"))
    if sampler_mode not in {"natural", "source_balanced"}:
        raise ValueError(
            f"train.sampler.sampler_mode 只支持 'natural' 或 'source_balanced'，当前为 {sampler_mode!r}"
        )

    source_balanced_target_cases = sampler_cfg.get("source_balanced_target_cases")
    if source_balanced_target_cases is not None:
        source_balanced_target_cases = int(source_balanced_target_cases)
        if source_balanced_target_cases <= 0:
            raise ValueError("train.sampler.source_balanced_target_cases 必须大于 0。")

    return {
        "sampler_mode": sampler_mode,
        "source_balanced_target_cases": source_balanced_target_cases,
    }


def resolve_phase2_real_heavy_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """解析 phase2 real-heavy finetune 配置。"""
    phase2_cfg = train_cfg.get("phase2_real_heavy", {})
    if phase2_cfg is None:
        phase2_cfg = {}
    if not isinstance(phase2_cfg, dict):
        raise TypeError("train.phase2_real_heavy 配置必须是 dict。")

    keep_sources_default = [*REAL_SOURCE_NAMES, "mimic_rag_0425"]
    drop_sources_default = ["FakeDisease"]
    keep_sources = _parse_source_list(phase2_cfg.get("keep_sources"), tuple(keep_sources_default))
    drop_sources = _parse_source_list(phase2_cfg.get("drop_sources"), tuple(drop_sources_default))

    overlap_sources = sorted(set(keep_sources) & set(drop_sources))
    if overlap_sources:
        raise ValueError(
            "train.phase2_real_heavy.keep_sources 和 drop_sources 不能重叠: "
            + ", ".join(overlap_sources)
        )

    start_epoch = int(phase2_cfg.get("start_epoch", 7))
    if start_epoch < 1:
        raise ValueError("train.phase2_real_heavy.start_epoch 必须大于等于 1。")

    lr_scale = float(phase2_cfg.get("lr_scale", 0.5))
    if lr_scale <= 0.0:
        raise ValueError("train.phase2_real_heavy.lr_scale 必须大于 0。")

    return {
        "enabled": bool(phase2_cfg.get("enabled", False)),
        "start_epoch": start_epoch,
        "lr_scale": lr_scale,
        "keep_sources": keep_sources,
        "drop_sources": drop_sources,
    }


def build_case_source_counts(
    df: pd.DataFrame,
    case_id_col: str = "case_id",
) -> dict[str, int]:
    """按 case 级 source 统计当前数据集的病例数。"""
    if case_id_col not in df.columns:
        raise KeyError(f"数据中缺少列: {case_id_col}")

    if "_source_name" in df.columns:
        source_series = df.groupby(case_id_col, sort=False)["_source_name"].first()
    elif "_source_file" in df.columns:
        source_series = df.groupby(case_id_col, sort=False)["_source_file"].first().map(normalize_source_name)
    else:
        source_series = pd.Series(["unknown_source"] * df[case_id_col].nunique())

    source_series = source_series.fillna("unknown_source").astype(str)
    return {str(source_name): int(count) for source_name, count in source_series.value_counts().sort_index().items()}


def build_phase2_real_heavy_train_df(
    train_df: pd.DataFrame,
    keep_sources: list[str],
    drop_sources: list[str],
    case_id_col: str = "case_id",
) -> pd.DataFrame:
    """基于 source_name 过滤出 phase2 real-heavy 子集。"""
    if case_id_col not in train_df.columns:
        raise KeyError(f"train_df 缺少列: {case_id_col}")

    if "_source_name" in train_df.columns:
        case_source_series = train_df.groupby(case_id_col, sort=False)["_source_name"].first()
    elif "_source_file" in train_df.columns:
        case_source_series = (
            train_df.groupby(case_id_col, sort=False)["_source_file"].first().map(normalize_source_name)
        )
    else:
        raise KeyError("train_df 缺少 _source_name/_source_file，无法构建 phase2 real-heavy 子集。")

    keep_source_set = {str(source_name) for source_name in keep_sources}
    drop_source_set = {str(source_name) for source_name in drop_sources}
    selected_case_ids = [
        str(case_id)
        for case_id, source_name in case_source_series.fillna("unknown_source").astype(str).items()
        if source_name in keep_source_set and source_name not in drop_source_set
    ]
    if not selected_case_ids:
        raise ValueError("phase2 real-heavy 过滤后没有可用病例，请检查 keep_sources/drop_sources。")

    phase2_train_df = train_df[train_df[case_id_col].isin(selected_case_ids)].reset_index(drop=True)
    if phase2_train_df.empty:
        raise ValueError("phase2 real-heavy 过滤后 train_df 为空，请检查配置。")
    return phase2_train_df


def build_train_loader(
    train_df: pd.DataFrame,
    batch_size: int,
    train_sampler_cfg: dict[str, Any],
) -> CaseBatchLoader:
    """按当前主线 sampler 配置构建训练集 loader。"""
    return CaseBatchLoader(
        df=train_df,
        batch_size=batch_size,
        sampler_mode=train_sampler_cfg["sampler_mode"],
        source_balanced_target_cases=train_sampler_cfg["source_balanced_target_cases"],
    )


def format_source_counts(source_counts: dict[str, int]) -> str:
    """把 source 计数摘要格式化成适合日志打印的文本。"""
    if not source_counts:
        return "<empty>"
    return " ".join(
        f"{source_name}={int(source_counts[source_name])}"
        for source_name in sorted(source_counts)
    )


def _resolve_batch_source_names(
    batch_df: pd.DataFrame,
    case_ids: list[str],
    case_id_col: str,
) -> list[str]:
    """根据 batch 内的 case_id 解析与输出顺序对齐的数据源名称。"""
    if "_source_name" in batch_df.columns:
        source_series = batch_df.groupby(case_id_col, sort=False)["_source_name"].first()
        case_source_map = {str(case_id): str(source_name) for case_id, source_name in source_series.items()}
    elif "_source_file" in batch_df.columns:
        source_series = batch_df.groupby(case_id_col, sort=False)["_source_file"].first()
        case_source_map = {
            str(case_id): normalize_source_name(source_file)
            for case_id, source_file in source_series.items()
        }
    else:
        case_source_map = {}

    return [case_source_map.get(str(case_id), "unknown_source") for case_id in case_ids]


def compute_grouped_topk_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    source_names: list[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """按数据源分组统计当前 batch 的 top-k 命中情况。"""
    if scores.ndim != 2:
        raise ValueError(f"scores 必须是二维张量，当前 shape={tuple(scores.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets 必须是一维张量，当前 shape={tuple(targets.shape)}")
    if scores.shape[0] != targets.shape[0]:
        raise ValueError("scores 和 targets 的 batch 维度不一致。")
    if len(source_names) != scores.shape[0]:
        raise ValueError("source_names 长度必须与 batch 样本数一致。")

    max_k = min(max(ks), scores.shape[1])
    topk_indices = scores.topk(max_k, dim=1).indices
    expanded_targets = targets.view(-1, 1)
    hits_by_k = {
        f"top{k}": (topk_indices[:, : min(k, scores.shape[1])] == expanded_targets).any(dim=1)
        for k in ks
    }

    group_hits: dict[str, dict[str, int]] = {}
    group_counts: dict[str, int] = {}
    for row_idx, raw_source_name in enumerate(source_names):
        source_name = str(raw_source_name or "unknown_source")
        group_counts[source_name] = group_counts.get(source_name, 0) + 1
        source_hit_store = group_hits.setdefault(
            source_name,
            {f"top{k}": 0 for k in ks},
        )
        for k in ks:
            source_hit_store[f"top{k}"] += int(hits_by_k[f"top{k}"][row_idx].item())

    group_metrics: dict[str, dict[str, float]] = {}
    for source_name, count in group_counts.items():
        group_metrics[source_name] = {
            f"top{k}": group_hits[source_name][f"top{k}"] / float(count)
            for k in ks
        }

    return {
        "group_metrics": group_metrics,
        "group_hits": group_hits,
        "group_counts": group_counts,
    }


def build_source_metric_fields(
    grouped_metrics: dict[str, dict[str, float]],
    real_sources: list[str],
    synthetic_sources: list[str],
    prefix: str = "val",
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float | None]:
    """把分组指标摊平成 history 和日志可直接使用的字段。"""
    fields: dict[str, float | None] = {}
    ordered_sources: list[str] = []
    for source_name in [*real_sources, *synthetic_sources]:
        if source_name not in ordered_sources:
            ordered_sources.append(source_name)
    for source_name in sorted(grouped_metrics):
        if source_name not in ordered_sources:
            ordered_sources.append(source_name)

    for source_name in ordered_sources:
        metrics = grouped_metrics.get(source_name)
        for k in ks:
            field_name = f"{prefix}_{source_name}_top{k}"
            fields[field_name] = None if metrics is None else float(metrics[f"top{k}"])

    available_real_sources = [source_name for source_name in real_sources if source_name in grouped_metrics]
    for k in ks:
        macro_field_name = f"{prefix}_real_macro_top{k}"
        if not available_real_sources:
            fields[macro_field_name] = None
            continue
        fields[macro_field_name] = float(
            sum(grouped_metrics[source_name][f"top{k}"] for source_name in available_real_sources)
            / len(available_real_sources)
        )

    return fields


def format_metric(value: float | None) -> str:
    """把指标值格式化成适合控制台打印的文本。"""
    if value is None:
        return "None"
    return f"{value:.4f}"


def _init_case_noise_aggregate() -> dict[str, Any]:
    return {
        "present": False,
        "enabled": False,
        "log_stats": False,
        "mode": None,
        "weighting": None,
        "alpha": None,
        "normalize_weights": None,
        "num_cases": 0,
        "pruned_case_count": 0,
        "raw_hpo_total": 0,
        "kept_hpo_total": 0,
        "dropped_hpo_total": 0,
        "weight_entropy_total": 0.0,
    }


def _accumulate_case_noise_aggregate(
    aggregate: dict[str, Any],
    batch_stats: dict[str, Any] | None,
) -> None:
    if not isinstance(batch_stats, dict):
        return

    num_cases = int(batch_stats.get("num_cases", 0))
    aggregate["present"] = True
    aggregate["enabled"] = bool(aggregate["enabled"] or batch_stats.get("enabled", False))
    aggregate["log_stats"] = bool(aggregate["log_stats"] or batch_stats.get("log_stats", False))
    aggregate["mode"] = batch_stats.get("mode", aggregate["mode"])
    aggregate["weighting"] = batch_stats.get("weighting", aggregate["weighting"])
    aggregate["alpha"] = batch_stats.get("alpha", aggregate["alpha"])
    aggregate["normalize_weights"] = batch_stats.get(
        "normalize_weights",
        aggregate["normalize_weights"],
    )
    aggregate["num_cases"] += num_cases
    aggregate["pruned_case_count"] += int(batch_stats.get("pruned_case_count", 0))
    aggregate["raw_hpo_total"] += int(batch_stats.get("raw_hpo_total", 0))
    aggregate["kept_hpo_total"] += int(batch_stats.get("kept_hpo_total", 0))
    aggregate["dropped_hpo_total"] += int(batch_stats.get("dropped_hpo_total", 0))
    aggregate["weight_entropy_total"] += float(batch_stats.get("mean_case_weight_entropy", 0.0)) * num_cases


def _finalize_case_noise_aggregate(aggregate: dict[str, Any]) -> dict[str, Any] | None:
    if not bool(aggregate.get("present")):
        return None

    num_cases = int(aggregate["num_cases"])
    raw_hpo_total = int(aggregate["raw_hpo_total"])
    kept_hpo_total = int(aggregate["kept_hpo_total"])
    dropped_hpo_total = int(aggregate["dropped_hpo_total"])
    return {
        "enabled": bool(aggregate["enabled"]),
        "log_stats": bool(aggregate["log_stats"]),
        "mode": aggregate["mode"],
        "weighting": aggregate["weighting"],
        "alpha": aggregate["alpha"],
        "normalize_weights": aggregate["normalize_weights"],
        "num_cases": num_cases,
        "pruned_case_count": int(aggregate["pruned_case_count"]),
        "raw_hpo_total": raw_hpo_total,
        "kept_hpo_total": kept_hpo_total,
        "dropped_hpo_total": dropped_hpo_total,
        "drop_ratio": float(dropped_hpo_total / float(raw_hpo_total)) if raw_hpo_total else 0.0,
        "mean_raw_hpo_per_case": float(raw_hpo_total / float(num_cases)) if num_cases else 0.0,
        "mean_kept_hpo_per_case": float(kept_hpo_total / float(num_cases)) if num_cases else 0.0,
        "mean_case_weight_entropy": (
            float(aggregate["weight_entropy_total"] / float(num_cases)) if num_cases else 0.0
        ),
    }


def _format_case_noise_summary(summary: dict[str, Any]) -> str:
    alpha_text = ""
    if summary.get("alpha") is not None:
        alpha_text = f"alpha={float(summary['alpha']):.4f} "
    return (
        f"mode={summary.get('mode')} "
        f"weighting={summary.get('weighting')} "
        f"{alpha_text}"
        f"raw_hpo={int(summary.get('raw_hpo_total', 0))} "
        f"kept_hpo={int(summary.get('kept_hpo_total', 0))} "
        f"drop_ratio={float(summary.get('drop_ratio', 0.0)):.4f} "
        f"mean_kept_hpo={float(summary.get('mean_kept_hpo_per_case', 0.0)):.2f} "
        f"mean_weight_entropy={float(summary.get('mean_case_weight_entropy', 0.0)):.4f}"
    )


def run_one_epoch(
    epoch: int,
    model: ModelPipeline,
    loader: CaseBatchLoader,
    static_graph: dict[str, Any],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    is_train: bool,
    shuffle: bool,
    random_seed: int,
    grad_clip_norm: float | None,
    log_every: int,
    hpo_dropout_prob: float = 0.0,
    hpo_corruption_prob: float = 0.0,
    case_noise_control: dict[str, Any] | None = None,
    hard_negative_cfg: dict[str, Any] | None = None,
    real_sources: list[str] | None = None,
) -> dict[str, Any]:
    """执行一轮训练或验证。"""
    if is_train:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()

    loader.set_epoch(
        epoch=epoch,
        shuffle=(is_train and shuffle),
        random_seed=random_seed,
    )
    if is_train:
        sampling_summary = loader.get_sampling_summary()
        print(
            f"Epoch {epoch} Train Sampler "
            f"mode={sampling_summary['sampler_mode']} "
            f"num_cases={sampling_summary['num_cases']} "
            f"target_cases_per_source={sampling_summary['source_balanced_target_cases']} "
            f"source_counts={format_source_counts(sampling_summary['source_counts'])}"
        )
    total_steps = len(loader)
    hard_negative_cfg = hard_negative_cfg or {}
    use_hard_negative = bool(hard_negative_cfg.get("use_hard_negative", True))
    hard_negative_start_epoch = int(hard_negative_cfg.get("start_epoch", 2))
    hard_negative_k = int(hard_negative_cfg.get("k", 10))
    hard_negative_strategy = str(hard_negative_cfg.get("strategy", "HN-current"))
    hard_negative_sampling_ratios = hard_negative_cfg.get("sampling_ratios")
    metric_ks = (1, 3, 5)

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top5 = 0.0
    total_samples = 0
    used_batches = 0
    grouped_hit_totals: dict[str, dict[str, int]] = {}
    grouped_sample_counts: dict[str, int] = {}
    case_noise_aggregate = _init_case_noise_aggregate()

    with context:
        for step, batch_idx in enumerate(range(total_steps), start=1):
            batch_df = loader.get_batch(batch_idx)
            if batch_df is None or batch_df.empty:
                print(f"Epoch {epoch} Step {step}/{total_steps} 空 batch，跳过。")
                continue

            try:
                batch_graph = build_batch_hypergraph(
                    case_df=batch_df,
                    hpo_to_idx=static_graph["hpo_to_idx"],
                    disease_to_idx=static_graph["disease_to_idx"],
                    H_disease=static_graph["H_disease"],
                    top_50_hpos=static_graph.get("top_50_hpos", []),
                    hpo_dropout_prob=hpo_dropout_prob if is_train else 0.0,
                    hpo_corruption_prob=hpo_corruption_prob if is_train else 0.0,
                    case_noise_control=case_noise_control,
                    hpo_specificity=static_graph.get("hpo_specificity"),
                    # 训练热路径只依赖 H_case 和 H_disease。
                    # 这里跳过 H=[H_case|H_disease] 的冗余拼接，不改变任何前向数值。
                    include_combined_h=False,
                    verbose=False,
                )
            except ValueError as exc:
                print(f"Epoch {epoch} Step {step}/{total_steps} {exc}，跳过。")
                continue

            # H_case 为空表示当前 batch 没有有效病例；H_disease 是静态图，只需确认疾病列非空。
            if batch_graph["H_case"].shape[1] == 0 or batch_graph["H_disease"].shape[1] == 0:
                print(f"Epoch {epoch} Step {step}/{total_steps} 无有效图，跳过。")
                continue

            batch_case_noise_stats = batch_graph.get("case_noise_stats")
            _accumulate_case_noise_aggregate(case_noise_aggregate, batch_case_noise_stats)
            if (
                isinstance(batch_case_noise_stats, dict)
                and bool(batch_case_noise_stats.get("log_stats", False))
                and (step % log_every == 0 or step == total_steps)
            ):
                print(
                    f"Epoch {epoch} Step {step}/{total_steps} CaseNoise "
                    f"{_format_case_noise_summary(batch_case_noise_stats)}"
                )

            outputs = model(batch_graph)
            if "scores" not in outputs:
                raise KeyError("model 输出缺少 scores。")
            if "gold_disease_idx_in_score_pool" not in outputs:
                raise KeyError("model 输出缺少 gold_disease_idx_in_score_pool。")

            scores = outputs["scores"]
            targets = outputs["gold_disease_idx_in_score_pool"]
            if scores.shape[0] == 0 or scores.shape[1] == 0 or targets.numel() == 0:
                print(f"Epoch {epoch} Step {step}/{total_steps} 无有效病例或标签，跳过。")
                continue

            hard_neg_indices = None
            if is_train and use_hard_negative and epoch >= hard_negative_start_epoch:
                # 训练时在线挖 top-k 难负例。
                with torch.no_grad():
                    hard_neg_indices = mine_configurable_hard_negatives(
                        scores=scores.detach(),
                        targets=targets.detach(),
                        k=hard_negative_k,
                        strategy=hard_negative_strategy,
                        sampling_ratios=hard_negative_sampling_ratios,
                    )

            loss_output = loss_fn(scores, targets, hard_neg_indices=hard_neg_indices)
            if "loss" not in loss_output:
                raise KeyError("loss_fn 输出缺少 loss。")

            loss = loss_output["loss"]
            if is_train:
                if optimizer is None:
                    raise ValueError("训练阶段 optimizer 不能为空。")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

                if step % log_every == 0 or step == total_steps:
                    print(f"Epoch {epoch} Step {step}/{total_steps} Loss {loss.detach().item():.6f}")

            batch_size = int(targets.shape[0])
            detached_scores = scores.detach()
            detached_targets = targets.detach()
            batch_metrics = compute_topk_metrics(detached_scores, detached_targets, ks=metric_ks)

            if not is_train:
                batch_source_names = _resolve_batch_source_names(
                    batch_df=batch_df,
                    case_ids=[str(case_id) for case_id in batch_graph["case_ids"]],
                    case_id_col=loader.case_id_col,
                )
                grouped_batch_metrics = compute_grouped_topk_metrics(
                    detached_scores,
                    detached_targets,
                    batch_source_names,
                    ks=metric_ks,
                )
                for source_name, sample_count in grouped_batch_metrics["group_counts"].items():
                    grouped_sample_counts[source_name] = (
                        grouped_sample_counts.get(source_name, 0) + int(sample_count)
                    )
                    source_hit_totals = grouped_hit_totals.setdefault(
                        source_name,
                        {f"top{k}": 0 for k in metric_ks},
                    )
                    for k in metric_ks:
                        source_hit_totals[f"top{k}"] += int(
                            grouped_batch_metrics["group_hits"][source_name][f"top{k}"]
                        )

            total_loss += float(loss.detach().item()) * batch_size
            total_top1 += batch_metrics["top1"] * batch_size
            total_top3 += batch_metrics["top3"] * batch_size
            total_top5 += batch_metrics["top5"] * batch_size
            total_samples += batch_size
            used_batches += 1

    if total_samples == 0:
        stage = "训练" if is_train else "验证"
        raise RuntimeError(f"{stage}阶段所有 batch 都被跳过，请检查数据或配置。")

    result: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top3": total_top3 / total_samples,
        "top5": total_top5 / total_samples,
        "num_samples": float(total_samples),
        "num_batches": float(used_batches),
    }
    if not is_train:
        grouped_metrics: dict[str, dict[str, float]] = {}
        for source_name, sample_count in grouped_sample_counts.items():
            grouped_metrics[source_name] = {
                f"top{k}": grouped_hit_totals[source_name][f"top{k}"] / float(sample_count)
                for k in metric_ks
            }
        result["group_metrics"] = grouped_metrics
        result["group_counts"] = {source_name: int(count) for source_name, count in grouped_sample_counts.items()}

        available_real_sources = [
            source_name for source_name in (real_sources or list(REAL_SOURCE_NAMES)) if source_name in grouped_metrics
        ]
        for k in metric_ks:
            metric_name = f"real_macro_top{k}"
            if not available_real_sources:
                result[metric_name] = None
                continue
            result[metric_name] = float(
                sum(grouped_metrics[source_name][f"top{k}"] for source_name in available_real_sources)
                / len(available_real_sources)
            )

    case_noise_summary = _finalize_case_noise_aggregate(case_noise_aggregate)
    if case_noise_summary is not None:
        result["case_noise_summary"] = case_noise_summary
        if case_noise_summary["enabled"] or case_noise_summary["log_stats"]:
            stage_name = "Train" if is_train else "Eval"
            print(f"Epoch {epoch} {stage_name} CaseNoise {_format_case_noise_summary(case_noise_summary)}")

    return result


def save_history(
    history: list[dict[str, Any]],
    log_dir: str | Path,
    run_timestamp: str,
) -> None:
    """保存训练历史。"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        log_dir / f"history_{run_timestamp}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(log_dir / f"history_{run_timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    save_path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float | None,
    config: dict[str, Any],
) -> None:
    """保存 checkpoint。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        },
        save_path,
    )


def resolve_device(device_name: str) -> torch.device:
    """解析训练设备。"""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("cuda 不可用，自动切换到 cpu。")
        return torch.device("cpu")
    return torch.device(device_name)


def is_metric_improved(current: float, best: float | None, mode: str) -> bool:
    """判断监控指标是否提升。"""
    if best is None:
        return True
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError("monitor_mode 只能是 max 或 min。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "训练 HGNN 模型。当前唯一可信主线不是隐式默认 train.yaml，"
            "而是 run_full_train.cmd 显式串起的 staged pipeline。"
        )
    )
    parser.add_argument("--config", type=str, required=True, help="显式指定训练配置路径。")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    loss_cfg = config["loss"]
    optimizer_cfg = config["optimizer"]
    train_cfg = config["train"]
    validation_cfg = resolve_validation_config(config)
    train_sampler_cfg = resolve_train_sampler_config(train_cfg)
    phase2_real_heavy_cfg = resolve_phase2_real_heavy_config(train_cfg)

    random_seed = int(data_cfg["random_seed"])
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = resolve_device(str(train_cfg["device"]))
    save_dir = Path(paths_cfg["save_dir"])
    checkpoint_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    case_files = resolve_train_files(paths_cfg)
    print(f"训练文件数: {len(case_files)}")
    print(f"主线说明: {TRUSTED_MAINLINE['description']}")

    all_df = load_case_files(
        file_paths=[str(path) for path in case_files],
        disease_index_path=paths_cfg["disease_index_path"],
        split_namespace="train",
    )
    train_df, val_df = split_train_val_by_case(
        df=all_df,
        val_ratio=float(data_cfg["val_ratio"]),
        random_seed=random_seed,
    )
    print(
        f"切分完成: train_case={train_df['case_id'].nunique()} "
        f"val_case={val_df['case_id'].nunique()}"
    )

    phase1_train_case_count = int(train_df["case_id"].nunique())
    phase1_source_counts = build_case_source_counts(train_df)
    phase2_train_df = None
    phase2_train_case_count = None
    phase2_source_counts: dict[str, int] | None = None
    if phase2_real_heavy_cfg["enabled"]:
        phase2_train_df = build_phase2_real_heavy_train_df(
            train_df=train_df,
            keep_sources=phase2_real_heavy_cfg["keep_sources"],
            drop_sources=phase2_real_heavy_cfg["drop_sources"],
        )
        phase2_train_case_count = int(phase2_train_df["case_id"].nunique())
        phase2_source_counts = build_case_source_counts(phase2_train_df)
        print(
            "Phase2 real-heavy prepared: "
            f"start_epoch={phase2_real_heavy_cfg['start_epoch']} "
            f"phase1_train_case={phase1_train_case_count} "
            f"phase2_train_case={phase2_train_case_count} "
            f"keep_sources={phase2_real_heavy_cfg['keep_sources']} "
            f"drop_sources={phase2_real_heavy_cfg['drop_sources']}"
        )
        print(
            "Phase2 source summary: "
            f"phase1_source_counts={format_source_counts(phase1_source_counts)} "
            f"phase2_source_counts={format_source_counts(phase2_source_counts)}"
        )

    train_loader = build_train_loader(
        train_df=train_df,
        batch_size=int(data_cfg["batch_size"]),
        train_sampler_cfg=train_sampler_cfg,
    )
    print(
        "Train sampler config: "
        f"mode={train_sampler_cfg['sampler_mode']}, "
        f"source_balanced_target_cases={train_sampler_cfg['source_balanced_target_cases']}"
    )
    
    if val_df.empty:
        val_loader = None
        train_cfg["eval_every"] = 0
        train_cfg["early_stop"] = False
        print("注意: val_ratio 设置为了 0，程序已自动临时关闭评估 (eval_every=0) 与早停机制，只保存最后的 last.pt。")
    else:
        val_loader = CaseBatchLoader(
            df=val_df,
            batch_size=int(data_cfg["batch_size"]),
            sampler_mode="natural",
        )

    static_graph = load_static_graph(
        hpo_index_path=paths_cfg["hpo_index_path"],
        disease_index_path=paths_cfg["disease_index_path"],
        disease_incidence_path=paths_cfg["disease_incidence_path"],
    )

    model_pipeline_config = build_model_pipeline_config(config, static_graph["num_hpo"])
    resolved_loss_cfg = resolve_loss_config(loss_cfg)
    for warning_text in resolved_loss_cfg["warnings"]:
        print(f"[WARN] {warning_text}")

    effective_config = build_training_effective_config(
        config_path=args.config,
        raw_config=config,
        resolved_train_files=case_files,
        model_pipeline_config=model_pipeline_config,
        resolved_loss_config=resolved_loss_cfg,
    )
    effective_config["validation"] = validation_cfg
    effective_config["train"]["sampler"] = train_sampler_cfg
    effective_config["train"]["phase2_real_heavy"] = phase2_real_heavy_cfg
    effective_config_path = save_yaml(effective_config, save_dir / "effective_config.yaml")
    print_effective_config("Training Effective Config", effective_config)
    print(f"训练 effective config 已保存: {effective_config_path}")

    model = ModelPipeline(model_pipeline_config).to(device)
    init_checkpoint_path = load_init_checkpoint(
        model=model,
        checkpoint_path=train_cfg.get("init_checkpoint_path"),
    )
    if init_checkpoint_path is not None:
        print(f"已加载初始 checkpoint: {init_checkpoint_path}")
    hard_negative_cfg = dict(resolved_loss_cfg["hard_negative"])
    loss_fn = build_loss(
        loss_name=resolved_loss_cfg["loss_name"],
        temperature=float(resolved_loss_cfg["temperature"]),
        hard_weight=hard_negative_cfg["weight"],
        margin=hard_negative_cfg["margin"],
        top_m=hard_negative_cfg["top_m"],
        poly_epsilon=float(resolved_loss_cfg["poly_epsilon"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_epoch: int | None = None
    no_improve_count = 0
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_every = int(train_cfg["eval_every"])
    num_epochs = int(train_cfg["num_epochs"])
    monitor = str(train_cfg["monitor"])
    monitor_mode = str(train_cfg["monitor_mode"])
    early_stop = bool(train_cfg["early_stop"])
    patience = int(train_cfg["early_stop_patience"])
    shuffle = bool(data_cfg["shuffle"])
    grad_clip_norm = train_cfg["grad_clip_norm"]
    log_every = max(1, int(train_cfg.get("log_every", 96)))
    hpo_dropout_prob = float(train_cfg.get("hpo_dropout_prob", 0.0))
    hpo_corruption_prob = float(train_cfg.get("hpo_corruption_prob", 0.0))
    case_noise_control_cfg = config.get("case_noise_control")

    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    current_train_phase = "phase1"
    phase2_switched = False
    for epoch in range(1, num_epochs + 1):
        if (
            phase2_real_heavy_cfg["enabled"]
            and not phase2_switched
            and epoch >= phase2_real_heavy_cfg["start_epoch"]
        ):
            if phase2_train_df is None or phase2_source_counts is None or phase2_train_case_count is None:
                raise ValueError("phase2 real-heavy 已启用，但 phase2_train_df 未正确构建。")

            old_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = float(group["lr"]) * float(phase2_real_heavy_cfg["lr_scale"])
            new_lrs = [float(group["lr"]) for group in optimizer.param_groups]

            train_loader = build_train_loader(
                train_df=phase2_train_df,
                batch_size=int(data_cfg["batch_size"]),
                train_sampler_cfg=train_sampler_cfg,
            )
            current_train_phase = "phase2_real_heavy"
            phase2_switched = True
            print(
                "Phase2 real-heavy switch: "
                f"epoch={epoch} "
                f"keep_sources={phase2_real_heavy_cfg['keep_sources']} "
                f"drop_sources={phase2_real_heavy_cfg['drop_sources']} "
                f"phase1_train_case={phase1_train_case_count} "
                f"phase2_train_case={phase2_train_case_count} "
                f"lr_old={old_lrs} "
                f"lr_new={new_lrs}"
            )
            print(
                "Phase2 real-heavy source_counts: "
                f"{format_source_counts(phase2_source_counts)}"
            )

        train_metrics = run_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            static_graph=static_graph,
            loss_fn=loss_fn,
            optimizer=optimizer,
            is_train=True,
            shuffle=shuffle,
            random_seed=random_seed,
            grad_clip_norm=grad_clip_norm,
            log_every=log_every,
            hpo_dropout_prob=hpo_dropout_prob,
            hpo_corruption_prob=hpo_corruption_prob,
            case_noise_control=case_noise_control_cfg,
            hard_negative_cfg=hard_negative_cfg,
        )

        do_eval = eval_every > 0 and epoch % eval_every == 0
        if do_eval:
            val_metrics = run_one_epoch(
                epoch=epoch,
                model=model,
                loader=val_loader,
                static_graph=static_graph,
                loss_fn=loss_fn,
                optimizer=None,
                is_train=False,
                shuffle=False,
                random_seed=random_seed,
                grad_clip_norm=None,
                log_every=log_every,
                hpo_dropout_prob=0.0,
                hpo_corruption_prob=0.0,
                case_noise_control=case_noise_control_cfg,
                hard_negative_cfg=hard_negative_cfg,
                real_sources=validation_cfg["real_sources"],
            )
        else:
            val_metrics = None

        record = {
            "epoch": epoch,
            "train_phase": current_train_phase,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top3": train_metrics["top3"],
            "train_top5": train_metrics["top5"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_top1": None if val_metrics is None else val_metrics["top1"],
            "val_top3": None if val_metrics is None else val_metrics["top3"],
            "val_top5": None if val_metrics is None else val_metrics["top5"],
            "val_real_macro_top1": None if val_metrics is None else val_metrics.get("real_macro_top1"),
            "val_real_macro_top3": None if val_metrics is None else val_metrics.get("real_macro_top3"),
            "val_real_macro_top5": None if val_metrics is None else val_metrics.get("real_macro_top5"),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_metrics is not None:
            record.update(
                build_source_metric_fields(
                    grouped_metrics=val_metrics.get("group_metrics", {}),
                    real_sources=validation_cfg["real_sources"],
                    synthetic_sources=validation_cfg["synthetic_sources"],
                    prefix="val",
                )
            )
        history.append(record)
        save_history(history, log_dir, run_timestamp)

        print(
            f"Epoch {epoch} Train "
            f"phase={record['train_phase']} "
            f"train_loss={record['train_loss']:.6f} "
            f"train_top1={record['train_top1']:.4f} "
            f"train_top3={record['train_top3']:.4f} "
            f"train_top5={record['train_top5']:.4f}"
        )
        if do_eval and val_metrics is not None:
            print(
                f"Epoch {epoch} Eval "
                f"val_loss={record['val_loss']:.6f} "
                f"val_top1={format_metric(record['val_top1'])} "
                f"val_top3={format_metric(record['val_top3'])} "
                f"val_top5={format_metric(record['val_top5'])} "
                f"val_real_macro_top1={format_metric(record['val_real_macro_top1'])} "
                f"val_real_macro_top3={format_metric(record['val_real_macro_top3'])} "
                f"val_real_macro_top5={format_metric(record['val_real_macro_top5'])}"
            )
            present_sources = list(val_metrics.get("group_metrics", {}).keys())
            if present_sources:
                ordered_sources: list[str] = []
                for source_name in [*validation_cfg["real_sources"], *validation_cfg["synthetic_sources"]]:
                    if source_name in present_sources and source_name not in ordered_sources:
                        ordered_sources.append(source_name)
                for source_name in sorted(present_sources):
                    if source_name not in ordered_sources:
                        ordered_sources.append(source_name)
                source_metric_text = " ".join(
                    f"val_{source_name}_top1={format_metric(record.get(f'val_{source_name}_top1'))}"
                    for source_name in ordered_sources
                )
                print(f"Epoch {epoch} Eval Sources {source_metric_text}")

        if do_eval and bool(train_cfg["save_every_eval"]):
            save_checkpoint(
                save_path=checkpoint_dir / f"epoch_{epoch:03d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_metric=best_metric,
                config=config,
            )

        if do_eval:
            if monitor not in record:
                raise KeyError(f"监控指标不存在: {monitor}")
            current_metric = record[monitor]
            if current_metric is None:
                raise ValueError(f"监控指标为空: {monitor}")

            if is_metric_improved(float(current_metric), best_metric, monitor_mode):
                best_metric = float(current_metric)
                best_epoch = epoch
                no_improve_count = 0
                if bool(train_cfg["save_best"]):
                    save_checkpoint(
                        save_path=checkpoint_dir / "best.pt",
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        best_metric=best_metric,
                        config=config,
                    )
            else:
                no_improve_count += 1

            if early_stop and no_improve_count >= patience:
                print(f"早停触发: best_epoch={best_epoch}, best_{monitor}={best_metric:.6f}")
                break

    if bool(train_cfg["save_last"]):
        save_checkpoint(
            save_path=checkpoint_dir / "last.pt",
            epoch=history[-1]["epoch"],
            model=model,
            optimizer=optimizer,
            best_metric=best_metric,
            config=config,
        )

    save_history(history, log_dir, run_timestamp)
    print(f"训练结束: best_epoch={best_epoch}, best_{monitor}={best_metric}")


if __name__ == "__main__":
    main()
