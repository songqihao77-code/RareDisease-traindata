from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_hypergraph import build_batch_hypergraph, load_static_graph
from src.data.dataset import CaseBatchLoader, read_case_table_file
from src.models.model_pipeline import ModelPipeline

DEFAULT_DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """读取 yaml 配置。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件内容必须是字典: {path}")
    return config


def _resolve_config_dir(config_path: str | Path | None) -> Path:
    """Return the directory used to resolve relative paths in configs."""
    if config_path is None:
        return PROJECT_ROOT
    return Path(config_path).resolve().parent


def _resolve_case_files_from_paths(
    paths_cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    files_key: str = "train_files",
    dir_key: str = "train_dir",
) -> list[Path]:
    """Resolve case files from either an explicit file list or a directory."""
    if not isinstance(paths_cfg, dict):
        raise TypeError("paths config must be a dict")

    config_dir = _resolve_config_dir(config_path)
    resolved_files: list[Path] = []

    raw_files = paths_cfg.get(files_key)
    if raw_files:
        if not isinstance(raw_files, list):
            raise TypeError(f"paths.{files_key} must be a list")
        for item in raw_files:
            path = Path(item)
            if not path.is_absolute():
                path = (config_dir / path).resolve()
            if path.name.startswith("~$"):
                continue
            resolved_files.append(path)
        return resolved_files

    raw_dir = paths_cfg.get(dir_key)
    if not raw_dir:
        return []

    directory = Path(raw_dir)
    if not directory.is_absolute():
        directory = (config_dir / directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{dir_key} does not exist: {directory}")

    return sorted(
        path for path in directory.glob("*.xlsx") if not path.name.startswith("~$")
    )


def _build_file_manifest(file_paths: list[Path]) -> dict[str, Any]:
    """Build a JSON-friendly manifest for resolved case files."""
    return {
        "count": int(len(file_paths)),
        "files": [str(path.resolve()) for path in file_paths],
    }


def _load_checkpoint_metadata(checkpoint_path: str | Path | None) -> dict[str, Any] | None:
    """Load lightweight metadata from a training checkpoint."""
    if checkpoint_path is None:
        return None

    path = Path(checkpoint_path)
    if not path.is_file():
        return {
            "checkpoint_path": str(path),
            "exists": False,
        }

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {
            "checkpoint_path": str(path.resolve()),
            "exists": True,
            "checkpoint_type": type(checkpoint).__name__,
        }

    config = checkpoint.get("config")
    metadata: dict[str, Any] = {
        "checkpoint_path": str(path.resolve()),
        "exists": True,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_metric": checkpoint.get("best_metric"),
        "config": config,
    }

    if isinstance(config, dict):
        train_files = _resolve_case_files_from_paths(
            config.get("paths", {}),
            files_key="train_files",
            dir_key="train_dir",
        )
        metadata["resolved_train_inputs"] = _build_file_manifest(train_files)

    return metadata


def _build_run_manifest(
    *,
    data_config: dict[str, Any],
    data_config_path: str | Path,
    train_config: dict[str, Any],
    train_config_path: str | Path,
    resolved_checkpoint_path: Path,
    checkpoint: dict[str, Any],
    test_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Collect the full config context used for this evaluation run."""
    finetune_train_files = _resolve_case_files_from_paths(
        train_config.get("paths", {}),
        config_path=train_config_path,
        files_key="train_files",
        dir_key="train_dir",
    )
    init_checkpoint_path = train_config.get("train", {}).get("init_checkpoint_path")

    return {
        "evaluation": {
            "data_config_path": str(Path(data_config_path).resolve()),
            "data_config": data_config,
            "resolved_test_inputs": {
                **_build_file_manifest([Path(path) for path in test_bundle["test_files"]]),
                "batch_size": int(test_bundle["batch_size"]),
                "case_id_col": str(test_bundle["case_id_col"]),
                "label_col": str(test_bundle["label_col"]),
                "hpo_col": str(test_bundle["hpo_col"]),
            },
        },
        "finetune": {
            "train_config_path": str(Path(train_config_path).resolve()),
            "train_config": train_config,
            "resolved_train_inputs": _build_file_manifest(finetune_train_files),
            "checkpoint": {
                "checkpoint_path": str(resolved_checkpoint_path.resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "best_metric": checkpoint.get("best_metric"),
                "embedded_train_config": checkpoint.get("config"),
            },
        },
        "pretrain": _load_checkpoint_metadata(init_checkpoint_path),
    }


def _resolve_device(train_config: dict[str, Any]) -> torch.device:
    """自动选择设备。"""
    device_name = str(train_config.get("train", {}).get("device", "auto")).strip().lower()
    if device_name in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("cuda 不可用，自动切换到 cpu。")
        return torch.device("cpu")
    return torch.device(device_name)


def _resolve_checkpoint_path(
    train_config: dict[str, Any],
    checkpoint_path: str | Path | None,
) -> Path:
    """解析 checkpoint 文件路径。"""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint_path 必须是存在的文件: {path}")
        return path

    save_dir = train_config.get("paths", {}).get("save_dir")
    if not save_dir:
        raise KeyError("train.yaml 缺少 paths.save_dir，无法自动查找 checkpoint。")

    checkpoint_dir = Path(save_dir) / "checkpoints"
    candidates = [checkpoint_dir / "best.pt", checkpoint_dir / "last.pt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    pt_files = sorted(checkpoint_dir.glob("*.pt"))
    if pt_files:
        return pt_files[-1]

    raise FileNotFoundError(f"未找到可用 checkpoint 文件: {checkpoint_dir}")


def _build_model_config(train_config: dict[str, Any], num_hpo: int) -> dict[str, Any]:
    """按训练配置组装模型配置。"""
    model_cfg = train_config.get("model")
    if not isinstance(model_cfg, dict):
        raise KeyError("train.yaml 缺少 model 配置。")
    if "hidden_dim" not in model_cfg:
        raise KeyError("train.yaml 缺少 model.hidden_dim。")

    hidden_dim = int(model_cfg["hidden_dim"])
    pipeline_model_cfg: dict[str, Any] = {
        "encoder": {
            "type": "hgnn",
            "num_hpo": int(num_hpo),
            "hidden_dim": hidden_dim,
        },
        "readout": {"type": "hyperedge", **model_cfg.get("readout", {}), "hidden_dim": hidden_dim},
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


def load_checkpoint_model(
    train_config: dict[str, Any],
    checkpoint_path: str | Path | None,
    num_hpo: int,
    device: torch.device,
) -> tuple[ModelPipeline, Path, dict[str, Any]]:
    """实例化模型并加载 checkpoint。"""
    resolved_path = _resolve_checkpoint_path(train_config, checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"无法识别 checkpoint 结构: {resolved_path}")

    model = ModelPipeline(_build_model_config(train_config, num_hpo)).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "checkpoint 与模型结构不匹配，"
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    model.eval()
    return model, resolved_path, checkpoint if isinstance(checkpoint, dict) else {}


def _load_index_frame(path: Path, id_col: str, idx_col: str) -> pd.DataFrame:
    """读取并校验索引表。"""
    if not path.exists():
        raise FileNotFoundError(f"索引文件不存在: {path}")

    df = pd.read_excel(path, dtype={id_col: str})
    required_cols = {id_col, idx_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path.name} 缺少必要列: {', '.join(sorted(missing_cols))}")

    df = df[[id_col, idx_col]].copy()
    df[idx_col] = df[idx_col].astype(int)
    df = df.sort_values(idx_col).reset_index(drop=True)
    expected = list(range(len(df)))
    actual = df[idx_col].tolist()
    if actual != expected:
        raise ValueError(f"{path.name} 的 {idx_col} 必须从 0 开始连续编号。")
    return df


def load_static_resources(train_config: dict[str, Any]) -> dict[str, Any]:
    """加载 H_disease、HPO 索引和疾病索引。"""
    paths_cfg = train_config.get("paths")
    if not isinstance(paths_cfg, dict):
        raise KeyError("train.yaml 缺少 paths 配置。")

    required_paths = {
        "hpo_index_path",
        "disease_index_path",
        "disease_incidence_path",
    }
    missing_paths = required_paths - set(paths_cfg)
    if missing_paths:
        raise KeyError(f"train.yaml 缺少必要路径配置: {', '.join(sorted(missing_paths))}")

    hpo_index_path = Path(paths_cfg["hpo_index_path"])
    disease_index_path = Path(paths_cfg["disease_index_path"])
    disease_incidence_path = Path(paths_cfg["disease_incidence_path"])
    if not disease_incidence_path.exists():
        raise FileNotFoundError(f"H_disease 文件不存在: {disease_incidence_path}")

    try:
        static_graph = load_static_graph(
            hpo_index_path=str(hpo_index_path),
            disease_index_path=str(disease_index_path),
            disease_incidence_path=str(disease_incidence_path),
        )
    except Exception as exc:
        raise RuntimeError(f"加载 H_disease 失败: {disease_incidence_path}") from exc

    hpo_index_df = _load_index_frame(hpo_index_path, id_col="hpo_id", idx_col="hpo_idx")
    disease_index_df = _load_index_frame(
        disease_index_path,
        id_col="mondo_id",
        idx_col="disease_idx",
    )

    if static_graph["H_disease"].shape != (len(hpo_index_df), len(disease_index_df)):
        raise ValueError(
            "H_disease 与索引表维度不一致，"
            f"H_disease={static_graph['H_disease'].shape}, "
            f"hpo={len(hpo_index_df)}, disease={len(disease_index_df)}"
        )

    static_graph["hpo_index_df"] = hpo_index_df
    static_graph["disease_index_df"] = disease_index_df
    static_graph["disease_labels"] = disease_index_df["mondo_id"].astype(str).tolist()
    return static_graph


def _resolve_test_files(data_config: dict[str, Any], data_config_path: str | Path) -> list[Path]:
    """从 data.yaml 解析测试文件列表。"""
    config_dir = Path(data_config_path).resolve().parent
    files: list[Path] = []

    if data_config.get("test_files"):
        raw_files = data_config["test_files"]
        if not isinstance(raw_files, list):
            raise ValueError("data.yaml 中的 test_files 必须是列表。")
        for item in raw_files:
            path = Path(item)
            if not path.is_absolute():
                path = (config_dir / path).resolve()
            files.append(path)
    elif data_config.get("test_dir"):
        test_dir = Path(data_config["test_dir"])
        if not test_dir.is_absolute():
            test_dir = (config_dir / test_dir).resolve()
        if not test_dir.is_dir():
            raise FileNotFoundError(f"测试目录不存在: {test_dir}")
        files = sorted(
            path
            for path in test_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".xlsx", ".xls", ".csv"}
            and not path.name.startswith("~$")
        )
    else:
        raise KeyError("data.yaml 必须提供 test_files 或 test_dir。")

    cleaned_files = [path for path in files if not path.name.startswith("~$")]
    if not cleaned_files:
        raise FileNotFoundError("未找到可用测试文件。")

    for path in cleaned_files:
        if not path.is_file():
            raise FileNotFoundError(f"测试文件不存在: {path}")
    return cleaned_files


def load_test_cases(
    data_config: dict[str, Any],
    data_config_path: str | Path,
) -> dict[str, Any]:
    """读取测试病例并整理为统一结构。"""
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))
    batch_size = int(data_config.get("batch_size", 8))

    file_paths = _resolve_test_files(data_config, data_config_path)
    frames: list[pd.DataFrame] = []

    for path in file_paths:
        df = read_case_table_file(path)
        required_cols = {case_id_col, label_col, hpo_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"{path.name} 缺少必要列: {', '.join(sorted(missing_cols))}")

        df = df[[case_id_col, label_col, hpo_col]].copy()
        df[case_id_col] = path.stem + "_" + df[case_id_col].astype(str)
        df["_source_file"] = path.name
        frames.append(df)

    raw_df = pd.concat(frames, ignore_index=True)
    if raw_df.empty:
        raise ValueError("测试集为空。")

    case_records: list[dict[str, Any]] = []
    for case_id, group_df in raw_df.groupby(case_id_col, sort=False):
        case_records.append(
            {
                "case_id": str(case_id),
                "mondo_label": str(group_df[label_col].iloc[0]),
                "hpo_ids": [str(hpo) for hpo in group_df[hpo_col].dropna().unique().tolist()],
                "source_file": str(group_df["_source_file"].iloc[0]),
            }
        )

    case_table = pd.DataFrame(case_records)
    if case_table.empty:
        raise ValueError("测试集中没有可识别的病例。")

    return {
        "raw_df": raw_df,
        "case_table": case_table,
        "batch_size": batch_size,
        "case_id_col": case_id_col,
        "label_col": label_col,
        "hpo_col": hpo_col,
        "test_files": [str(path) for path in file_paths],
    }


def compute_topk_metrics(
    true_ranks: list[int],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """根据真实排名计算 top-k。"""
    if not true_ranks:
        raise ValueError("没有可评估病例，无法计算 top-k 指标。")

    total = float(len(true_ranks))
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"top{k}"] = sum(rank <= k for rank in true_ranks) / total
    return metrics


def _build_per_dataset_summary(
    case_table: pd.DataFrame,
    detailed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按测试集汇总 top-k。"""
    details_df = pd.DataFrame(detailed_results)
    summary_rows: list[dict[str, Any]] = []

    for source_file, group_df in case_table.groupby("source_file", sort=True):
        dataset_name = Path(str(source_file)).stem
        num_cases = int(len(group_df))
        num_evaluable = int(group_df["skip_reason"].isna().sum())
        num_skipped = int(group_df["skip_reason"].notna().sum())
        num_skipped_missing_label = int((group_df["skip_reason"] == "label_not_in_disease_index").sum())
        num_skipped_no_valid_hpo = int((group_df["skip_reason"] == "no_valid_hpo").sum())

        row = {
            "dataset_name": dataset_name,
            "source_file": str(source_file),
            "num_cases": num_cases,
            "num_evaluable": num_evaluable,
            "num_skipped": num_skipped,
            "num_skipped_missing_label": num_skipped_missing_label,
            "num_skipped_no_valid_hpo": num_skipped_no_valid_hpo,
            "top1": None,
            "top3": None,
            "top5": None,
        }

        if num_evaluable > 0:
            group_ranks = (
                details_df.loc[details_df["source_file"] == str(source_file), "true_rank"]
                .astype(int)
                .tolist()
            )
            if len(group_ranks) != num_evaluable:
                raise RuntimeError(
                    f"测试集 {dataset_name} 的评估病例数不一致，"
                    f"期望 {num_evaluable}，实际 {len(group_ranks)}。"
                )
            group_metrics = compute_topk_metrics(group_ranks, ks=(1, 3, 5))
            row["top1"] = float(group_metrics["top1"])
            row["top3"] = float(group_metrics["top3"])
            row["top5"] = float(group_metrics["top5"])

        summary_rows.append(row)

    return summary_rows


def evaluate(
    data_config_path: str | Path,
    train_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行完整评估流程。"""
    data_config = load_yaml_config(data_config_path)
    train_config = load_yaml_config(train_config_path)
    device = _resolve_device(train_config)

    resources = load_static_resources(train_config)
    test_bundle = load_test_cases(data_config, data_config_path)

    case_table = test_bundle["case_table"].copy()
    hpo_to_idx = resources["hpo_to_idx"]
    disease_to_idx = resources["disease_to_idx"]

    case_table["valid_hpo_ids"] = case_table["hpo_ids"].apply(
        lambda hpo_ids: [hpo_id for hpo_id in hpo_ids if hpo_id in hpo_to_idx]
    )
    case_table["valid_hpo_count"] = case_table["valid_hpo_ids"].apply(len)
    case_table["skip_reason"] = None
    case_table.loc[~case_table["mondo_label"].isin(disease_to_idx), "skip_reason"] = (
        "label_not_in_disease_index"
    )
    case_table.loc[
        case_table["skip_reason"].isna() & (case_table["valid_hpo_count"] == 0),
        "skip_reason",
    ] = "no_valid_hpo"

    skipped_cases = case_table[case_table["skip_reason"].notna()].copy()
    evaluable_cases = case_table[case_table["skip_reason"].isna()].copy()
    if evaluable_cases.empty:
        raise ValueError("测试集中没有可评估病例。")

    raw_df = test_bundle["raw_df"].copy()
    case_id_col = test_bundle["case_id_col"]
    label_col = test_bundle["label_col"]
    hpo_col = test_bundle["hpo_col"]
    evaluable_case_ids = set(evaluable_cases["case_id"].tolist())
    valid_hpo_ids = set(hpo_to_idx)

    eval_df = raw_df[raw_df[case_id_col].isin(evaluable_case_ids)].copy()
    eval_df = eval_df[eval_df[hpo_col].isin(valid_hpo_ids)].copy()
    if eval_df.empty:
        raise ValueError("过滤后的测试集为空，无法评估。")

    loader = CaseBatchLoader(
        df=eval_df,
        batch_size=int(test_bundle["batch_size"]),
        case_id_col=case_id_col,
    )
    model, resolved_checkpoint_path, checkpoint = load_checkpoint_model(
        train_config=train_config,
        checkpoint_path=checkpoint_path,
        num_hpo=resources["num_hpo"],
        device=device,
    )
    case_source_map = dict(zip(case_table["case_id"], case_table["source_file"], strict=True))

    detailed_results: list[dict[str, Any]] = []
    true_ranks: list[int] = []
    processed_case_ids: set[str] = set()

    print(f"测试文件数: {len(test_bundle['test_files'])}")
    print(f"测试病例数: {len(case_table)}")
    print(f"可评估病例数: {len(evaluable_cases)}，跳过: {len(skipped_cases)}")
    print(f"使用 checkpoint: {resolved_checkpoint_path}")
    print(f"评估设备: {device}")

    with torch.inference_mode():
        for batch_idx in range(len(loader)):
            batch_df = loader.get_batch(batch_idx)
            batch_graph = build_batch_hypergraph(
                case_df=batch_df,
                hpo_to_idx=hpo_to_idx,
                disease_to_idx=disease_to_idx,
                H_disease=resources["H_disease"],
                top_50_hpos=resources.get("top_50_hpos", []),
                case_id_col=case_id_col,
                label_col=label_col,
                hpo_col=hpo_col,
                verbose=False,
            )

            num_case = batch_graph["H_case"].shape[1]
            if num_case == 0:
                continue

            outputs = model(batch_graph, return_intermediate=True)
            node_repr = outputs["node_repr"]
            if not isinstance(node_repr, torch.Tensor) or node_repr.ndim != 2:
                raise ValueError("encoder 输出必须是二维张量 [num_hpo, hidden_dim]。")
            if node_repr.shape[0] != resources["num_hpo"]:
                raise ValueError(
                    "encoder 输出的 HPO 维度不正确，"
                    f"得到 {tuple(node_repr.shape)}，期望首维 {resources['num_hpo']}。"
                )

            if model.case_refiner is not None and "refined_case_node_repr" not in outputs:
                raise ValueError("评估路径未返回 refined_case_node_repr，说明未经过 refiner。")

            case_repr = outputs["case_repr"]
            if case_repr.ndim != 2 or case_repr.shape[0] != num_case:
                raise ValueError(
                    "case readout 输出维度不正确，"
                    f"得到 {tuple(case_repr.shape)}，期望首维 {num_case}。"
                )

            disease_repr = outputs["disease_repr"]
            if disease_repr.ndim != 2 or disease_repr.shape[0] != resources["num_disease"]:
                raise ValueError(
                    "disease readout 输出维度不正确，"
                    f"得到 {tuple(disease_repr.shape)}，期望首维 {resources['num_disease']}。"
                )

            scores = outputs["scores"]
            if not isinstance(scores, torch.Tensor):
                raise ValueError("scorer 输出缺少 scores 张量。")
            if scores.shape != (num_case, resources["num_disease"]):
                raise ValueError(
                    "scores 维度不正确，"
                    f"得到 {tuple(scores.shape)}，期望 {(num_case, resources['num_disease'])}。"
                )

            topk_k = min(5, scores.shape[1])
            topk = torch.topk(scores, k=topk_k, dim=1)
            ranked_indices = torch.argsort(scores, dim=1, descending=True)

            case_ids = batch_graph["case_ids"]
            case_labels = batch_graph["case_labels"]
            gold_indices = outputs["gold_disease_cols_local"].tolist()
            if not (len(case_ids) == len(case_labels) == len(gold_indices) == num_case):
                raise ValueError("评估 batch 中的病例元信息长度不一致。")

            for row_idx, (case_id, true_label, gold_idx) in enumerate(
                zip(case_ids, case_labels, gold_indices, strict=True)
            ):
                gold_idx = int(gold_idx)
                source_file = str(case_source_map[str(case_id)])
                dataset_name = Path(source_file).stem
                rank_position = (ranked_indices[row_idx] == gold_idx).nonzero(as_tuple=False)
                if rank_position.numel() != 1:
                    raise ValueError(f"无法定位真实疾病排名: case_id={case_id}")
                true_rank = int(rank_position.item()) + 1

                top_indices = topk.indices[row_idx].tolist()
                top_scores = [float(value) for value in topk.values[row_idx].tolist()]
                top_labels = [resources["disease_labels"][idx] for idx in top_indices]

                detailed_results.append(
                    {
                        "case_id": str(case_id),
                        "dataset_name": dataset_name,
                        "source_file": source_file,
                        "true_label": str(true_label),
                        "pred_top1": top_labels[0],
                        "pred_top1_score": top_scores[0],
                        "top5_labels": top_labels,
                        "top5_scores": top_scores,
                        "true_rank": true_rank,
                    }
                )
                true_ranks.append(true_rank)
                processed_case_ids.add(str(case_id))

    expected_case_ids = set(evaluable_cases["case_id"].tolist())
    missing_case_ids = sorted(expected_case_ids - processed_case_ids)
    if missing_case_ids:
        preview = ", ".join(missing_case_ids[:10])
        raise RuntimeError(f"以下病例未完成评估: {preview}")

    metrics = compute_topk_metrics(true_ranks, ks=(1, 3, 5))
    per_dataset_summary = _build_per_dataset_summary(case_table, detailed_results)
    run_manifest = _build_run_manifest(
        data_config=data_config,
        data_config_path=data_config_path,
        train_config=train_config,
        train_config_path=train_config_path,
        resolved_checkpoint_path=resolved_checkpoint_path,
        checkpoint=checkpoint,
        test_bundle=test_bundle,
    )
    summary = {
        "data_config_path": str(Path(data_config_path).resolve()),
        "train_config_path": str(Path(train_config_path).resolve()),
        "checkpoint_path": str(resolved_checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "num_cases": int(len(case_table)),
        "num_evaluable": int(len(detailed_results)),
        "num_skipped": int(len(skipped_cases)),
        "num_skipped_missing_label": int(
            (skipped_cases["skip_reason"] == "label_not_in_disease_index").sum()
        ),
        "num_skipped_no_valid_hpo": int((skipped_cases["skip_reason"] == "no_valid_hpo").sum()),
        "top1": float(metrics["top1"]),
        "top3": float(metrics["top3"]),
        "top5": float(metrics["top5"]),
        "per_dataset": per_dataset_summary,
        "run_manifest": run_manifest,
    }

    print(
        f"评估完成: top1={summary['top1']:.4f}, "
        f"top3={summary['top3']:.4f}, top5={summary['top5']:.4f}"
    )
    for row in per_dataset_summary:
        if row["top1"] is None:
            print(f"{row['dataset_name']}: 无可评估病例")
            continue
        print(
            f"{row['dataset_name']}: "
            f"top1={row['top1']:.4f}, top3={row['top3']:.4f}, top5={row['top5']:.4f}"
        )

    return {
        "summary": summary,
        "details": detailed_results,
        "per_dataset_summary": per_dataset_summary,
        "skipped_cases": skipped_cases[["case_id", "mondo_label", "skip_reason", "source_file"]]
        .copy()
        .to_dict(orient="records"),
    }


def save_results(results: dict[str, Any], train_config: dict[str, Any]) -> dict[str, str]:
    """保存详细结果和汇总结果。"""
    paths_cfg = train_config.get("paths")
    if not isinstance(paths_cfg, dict) or "save_dir" not in paths_cfg:
        raise KeyError("train.yaml 缺少 paths.save_dir，无法保存结果。")

    eval_dir = Path(paths_cfg["save_dir"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_stem = Path(results["summary"]["checkpoint_path"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = eval_dir / f"{checkpoint_stem}_{timestamp}_details.csv"
    summary_path = eval_dir / f"{checkpoint_stem}_{timestamp}_summary.json"
    per_dataset_path = eval_dir / f"{checkpoint_stem}_{timestamp}_per_dataset.csv"
    skipped_path = eval_dir / f"{checkpoint_stem}_{timestamp}_skipped.csv"

    details_df = pd.DataFrame(results["details"])
    if not details_df.empty:
        details_df = details_df.copy()
        details_df["top5_labels"] = details_df["top5_labels"].apply(
            lambda values: json.dumps(values, ensure_ascii=False)
        )
        details_df["top5_scores"] = details_df["top5_scores"].apply(
            lambda values: json.dumps(values, ensure_ascii=False)
        )
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    per_dataset_df = pd.DataFrame(
        results["per_dataset_summary"],
        columns=[
            "dataset_name",
            "source_file",
            "num_cases",
            "num_evaluable",
            "num_skipped",
            "num_skipped_missing_label",
            "num_skipped_no_valid_hpo",
            "top1",
            "top3",
            "top5",
        ],
    )
    per_dataset_df.to_csv(per_dataset_path, index=False, encoding="utf-8-sig")

    skipped_df = pd.DataFrame(
        results["skipped_cases"],
        columns=["case_id", "mondo_label", "skip_reason", "source_file"],
    )
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results["summary"], f, ensure_ascii=False, indent=2)

    print(f"详细结果已保存: {details_path}")
    print(f"汇总结果已保存: {summary_path}")
    print(f"分测试集结果已保存: {per_dataset_path}")
    print(f"跳过病例已保存: {skipped_path}")

    return {
        "details_path": str(details_path),
        "summary_path": str(summary_path),
        "per_dataset_path": str(per_dataset_path),
        "skipped_path": str(skipped_path),
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="评估 HGNN 模型在测试集上的疾病排序表现。")
    parser.add_argument(
        "--data_config_path",
        type=str,
        default=str(DEFAULT_DATA_CONFIG_PATH),
        help="data.yaml 路径。",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default=str(DEFAULT_TRAIN_CONFIG_PATH),
        help="train.yaml 路径。",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="checkpoint 文件路径；为空时自动优先查找 best.pt。",
    )
    args = parser.parse_args()

    train_config = load_yaml_config(args.train_config_path)
    results = evaluate(
        data_config_path=args.data_config_path,
        train_config_path=args.train_config_path,
        checkpoint_path=args.checkpoint_path,
    )
    save_results(results, train_config)


if __name__ == "__main__":
    main()
