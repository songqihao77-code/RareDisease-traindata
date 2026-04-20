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
from src.data.dataset import CaseBatchLoader, load_case_files
from src.models.model_pipeline import ModelPipeline
from src.training.hard_negative_miner import mine_hard_negatives
from src.training.loss_builder import build_loss

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


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
    hard_negative_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
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
    total_steps = len(loader)
    hard_negative_cfg = hard_negative_cfg or {}
    use_hard_negative = bool(hard_negative_cfg.get("use_hard_negative", True))
    hard_negative_start_epoch = int(hard_negative_cfg.get("start_epoch", 2))
    hard_negative_k = int(hard_negative_cfg.get("k", 10))

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top5 = 0.0
    total_samples = 0
    used_batches = 0

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

            outputs = model(batch_graph)
            if "scores" not in outputs:
                raise KeyError("model 输出缺少 scores。")
            if "gold_disease_cols_local" not in outputs:
                raise KeyError("model 输出缺少 gold_disease_cols_local。")

            scores = outputs["scores"]
            targets = outputs["gold_disease_cols_local"]
            if scores.shape[0] == 0 or scores.shape[1] == 0 or targets.numel() == 0:
                print(f"Epoch {epoch} Step {step}/{total_steps} 无有效病例或标签，跳过。")
                continue

            hard_neg_indices = None
            if is_train and use_hard_negative and epoch >= hard_negative_start_epoch:
                # 训练时在线挖 top-k 难负例。
                with torch.no_grad():
                    hard_neg_indices = mine_hard_negatives(
                        scores=scores.detach(),
                        targets=targets.detach(),
                        k=hard_negative_k,
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
            batch_metrics = compute_topk_metrics(scores.detach(), targets.detach(), ks=(1, 3, 5))

            total_loss += float(loss.detach().item()) * batch_size
            total_top1 += batch_metrics["top1"] * batch_size
            total_top3 += batch_metrics["top3"] * batch_size
            total_top5 += batch_metrics["top5"] * batch_size
            total_samples += batch_size
            used_batches += 1

    if total_samples == 0:
        stage = "训练" if is_train else "验证"
        raise RuntimeError(f"{stage}阶段所有 batch 都被跳过，请检查数据或配置。")

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top3": total_top3 / total_samples,
        "top5": total_top5 / total_samples,
        "num_samples": float(total_samples),
        "num_batches": float(used_batches),
    }


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


def build_model_config(config: dict[str, Any], num_hpo: int) -> dict[str, Any]:
    """组装 ModelPipeline 配置。"""
    hidden_dim = int(config["model"]["hidden_dim"])
    model_cfg = config.get("model", {})
    readout_cfg = {"type": "hyperedge", **model_cfg.get("readout", {}), "hidden_dim": hidden_dim}

    pipeline_model_cfg: dict[str, Any] = {
        "encoder": {
            "type": "hgnn",
            "num_hpo": num_hpo,
            "hidden_dim": hidden_dim,
        },
        "readout": readout_cfg,
        "scorer": {"type": "cosine"},
        "outputs": {
            "include_metadata": True,
            "include_global_scores": False,
            "return_intermediate": False,
        },
    }

    case_refiner_cfg = model_cfg.get("case_refiner", {})
    if isinstance(case_refiner_cfg, dict):
        pipeline_model_cfg["case_refiner"] = {
            "type": "case_conditioned",
            "enabled": bool(case_refiner_cfg.get("enabled", False)),
            "hidden_dim": hidden_dim,
            "mlp_hidden_dim": int(case_refiner_cfg.get("mlp_hidden_dim", hidden_dim)),
            "residual": float(case_refiner_cfg.get("residual", 0.7)),
        }

    return {
        "model": pipeline_model_cfg
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    loss_cfg = config["loss"]
    optimizer_cfg = config["optimizer"]
    train_cfg = config["train"]

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

    all_df = load_case_files(
        file_paths=[str(path) for path in case_files],
        disease_index_path=paths_cfg["disease_index_path"],
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

    train_loader = CaseBatchLoader(df=train_df, batch_size=int(data_cfg["batch_size"]))
    
    if val_df.empty:
        val_loader = None
        train_cfg["eval_every"] = 0
        train_cfg["early_stop"] = False
        print("注意: val_ratio 设置为了 0，程序已自动临时关闭评估 (eval_every=0) 与早停机制，只保存最后的 last.pt。")
    else:
        val_loader = CaseBatchLoader(df=val_df, batch_size=int(data_cfg["batch_size"]))

    static_graph = load_static_graph(
        hpo_index_path=paths_cfg["hpo_index_path"],
        disease_index_path=paths_cfg["disease_index_path"],
        disease_incidence_path=paths_cfg["disease_incidence_path"],
    )

    model = ModelPipeline(build_model_config(config, static_graph["num_hpo"])).to(device)
    init_checkpoint_path = load_init_checkpoint(
        model=model,
        checkpoint_path=train_cfg.get("init_checkpoint_path"),
    )
    if init_checkpoint_path is not None:
        print(f"已加载初始 checkpoint: {init_checkpoint_path}")
    hard_loss_cfg = loss_cfg.get("hard_negative", {})
    hard_negative_cfg = {
        "use_hard_negative": bool(
            hard_loss_cfg.get("use_hard_negative", loss_cfg.get("use_hard_negative", True))
        ),
        "k": int(hard_loss_cfg.get("k", loss_cfg.get("hard_negative_k", 10))),
        "top_m": int(hard_loss_cfg.get("top_m", loss_cfg.get("hard_negative_top_m", 3))),
        "start_epoch": int(
            hard_loss_cfg.get("start_epoch", loss_cfg.get("hard_negative_start_epoch", 2))
        ),
        "weight": float(hard_loss_cfg.get("weight", loss_cfg.get("hard_negative_weight", 0.5))),
        "margin": float(hard_loss_cfg.get("margin", loss_cfg.get("hard_negative_margin", 0.1))),
    }
    loss_fn = build_loss(
        loss_name=loss_cfg["loss_name"],
        temperature=float(loss_cfg["temperature"]),
        hard_weight=hard_negative_cfg["weight"],
        margin=hard_negative_cfg["margin"],
        top_m=hard_negative_cfg["top_m"],
        poly_epsilon=float(loss_cfg.get("poly_epsilon", 2.0)),
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

    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    for epoch in range(1, num_epochs + 1):
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
                hard_negative_cfg=hard_negative_cfg,
            )
        else:
            val_metrics = None

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top3": train_metrics["top3"],
            "train_top5": train_metrics["top5"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_top1": None if val_metrics is None else val_metrics["top1"],
            "val_top3": None if val_metrics is None else val_metrics["top3"],
            "val_top5": None if val_metrics is None else val_metrics["top5"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        save_history(history, log_dir, run_timestamp)

        print(
            f"Epoch {epoch} Train "
            f"train_loss={record['train_loss']:.6f} "
            f"train_top1={record['train_top1']:.4f} "
            f"train_top3={record['train_top3']:.4f} "
            f"train_top5={record['train_top5']:.4f}"
        )
        if do_eval and val_metrics is not None:
            print(
                f"Epoch {epoch} Eval "
                f"val_loss={record['val_loss']:.6f} "
                f"val_top1={record['val_top1']:.4f} "
                f"val_top3={record['val_top3']:.4f} "
                f"val_top5={record['val_top5']:.4f}"
            )

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
