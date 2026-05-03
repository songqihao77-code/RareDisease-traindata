from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRUSTED_MAINLINE = {
    "description": (
        "当前唯一可信主线是 run_full_train.cmd 串起的 staged pipeline："
        "train_pretrain.yaml -> train_finetune_attn_idf_main.yaml -> data_llldataset_eval.yaml。"
    ),
    "entry_script": "run_full_train.cmd",
    "train_configs": [
        "configs/train_pretrain.yaml",
        "configs/train_finetune_attn_idf_main.yaml",
    ],
    "eval_config": "configs/data_llldataset_eval.yaml",
}

_TRUSTED_TRAIN_CONFIGS = {
    (PROJECT_ROOT / rel_path).resolve()
    for rel_path in TRUSTED_MAINLINE["train_configs"]
}
_TRUSTED_EVAL_CONFIG = (PROJECT_ROOT / TRUSTED_MAINLINE["eval_config"]).resolve()


def to_serializable(value: Any) -> Any:
    """递归转换为可安全写入 YAML/JSON 的基础类型。"""
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    return value


def dump_yaml_text(payload: Mapping[str, Any]) -> str:
    """格式化 YAML 文本。"""
    return yaml.safe_dump(
        to_serializable(dict(payload)),
        allow_unicode=True,
        sort_keys=False,
    )


def save_yaml(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """保存 YAML 文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_yaml_text(payload), encoding="utf-8")
    return output_path


def print_effective_config(title: str, payload: Mapping[str, Any]) -> None:
    """打印 effective config。"""
    print(f"===== {title} =====")
    print(dump_yaml_text(payload).rstrip())
    print(f"===== End of {title} =====")


def is_trusted_training_config(config_path: str | Path) -> bool:
    return Path(config_path).resolve() in _TRUSTED_TRAIN_CONFIGS


def is_trusted_eval_config(config_path: str | Path) -> bool:
    return Path(config_path).resolve() == _TRUSTED_EVAL_CONFIG


def build_model_pipeline_config(config: Mapping[str, Any], num_hpo: int) -> dict[str, Any]:
    """按训练配置组装 ModelPipeline 配置。"""
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        raise KeyError("配置缺少 model 配置。")
    if "hidden_dim" not in model_cfg:
        raise KeyError("配置缺少 model.hidden_dim。")

    hidden_dim = int(model_cfg["hidden_dim"])
    readout_cfg = {
        "type": "hyperedge",
        **dict(model_cfg.get("readout", {})),
        "hidden_dim": hidden_dim,
    }


    encoder_cfg = dict(model_cfg.get("encoder", {}))
    encoder_type = str(encoder_cfg.get("type", "hgnn"))
    if bool(encoder_cfg.get("use_tag_encoder", False)) or encoder_type in {"tag_hgnn", "hybrid_tag_hgnn"}:
        raise ValueError("TAG encoder has been removed from the active framework; use model.encoder.type='hgnn'.")
    encoder_cfg["type"] = "hgnn"
    encoder_cfg.pop("use_tag_encoder", None)
    encoder_cfg.pop("pretrained_embed_path", None)
    encoder_cfg.pop("hpo_embed_path", None)

    pipeline_model_cfg: dict[str, Any] = {
        "encoder": {
            **encoder_cfg,
            "num_hpo": int(num_hpo),
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
    if isinstance(case_refiner_cfg, Mapping):
        pipeline_model_cfg["case_refiner"] = {
            "type": "case_conditioned",
            "enabled": bool(case_refiner_cfg.get("enabled", False)),
            "hidden_dim": hidden_dim,
            "mlp_hidden_dim": int(case_refiner_cfg.get("mlp_hidden_dim", hidden_dim)),
            "residual": float(case_refiner_cfg.get("residual", 0.7)),
        }

    return {"model": pipeline_model_cfg}


def _disabled_hard_negative_config() -> dict[str, Any]:
    return {
        "use_hard_negative": False,
        "k": 0,
        "top_m": 0,
        "start_epoch": 0,
        "weight": 0.0,
        "margin": 0.0,
        "strategy": "HN-current",
        "sampling_ratios": {},
    }


def _parse_legacy_hard_negative(loss_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    legacy_keys = (
        "use_hard_negative",
        "hard_negative_k",
        "hard_negative_top_m",
        "hard_negative_start_epoch",
        "hard_negative_weight",
        "hard_negative_margin",
    )
    if not any(key in loss_cfg for key in legacy_keys):
        return None

    return {
        "use_hard_negative": loss_cfg.get("use_hard_negative", False),
        "k": loss_cfg.get("hard_negative_k"),
        "top_m": loss_cfg.get("hard_negative_top_m"),
        "start_epoch": loss_cfg.get("hard_negative_start_epoch"),
        "weight": loss_cfg.get("hard_negative_weight"),
        "margin": loss_cfg.get("hard_negative_margin"),
    }


def resolve_loss_config(loss_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """把 loss 配置解析成不会再发生 hidden default 漂移的显式语义。"""
    if not isinstance(loss_cfg, Mapping):
        raise TypeError(f"loss 配置必须是 Mapping，当前为 {type(loss_cfg).__name__}。")

    if "loss_name" not in loss_cfg:
        raise KeyError("loss.loss_name 必须显式声明。")
    if "temperature" not in loss_cfg:
        raise KeyError("loss.temperature 必须显式声明。")

    warnings: list[str] = []
    if "poly_epsilon" in loss_cfg:
        poly_epsilon = float(loss_cfg["poly_epsilon"])
    else:
        poly_epsilon = 0.0
        warnings.append("loss.poly_epsilon 未声明，按 0.0 处理。")

    hard_negative_block = loss_cfg.get("hard_negative")
    if hard_negative_block is None:
        legacy_hard_negative = _parse_legacy_hard_negative(loss_cfg)
        if legacy_hard_negative is None:
            hard_negative_cfg = _disabled_hard_negative_config()
            warnings.append("loss.hard_negative 未声明，按关闭处理。")
        else:
            hard_negative_block = legacy_hard_negative
            warnings.append(
                "检测到 legacy hard_negative 顶层字段，已显式解析；建议迁移到 loss.hard_negative。"
            )

    if hard_negative_block is not None:
        if not isinstance(hard_negative_block, Mapping):
            raise TypeError("loss.hard_negative 必须是 Mapping。")

        use_hard_negative = bool(hard_negative_block.get("use_hard_negative", False))
        if not use_hard_negative:
            hard_negative_cfg = {
                "use_hard_negative": False,
                "k": int(hard_negative_block.get("k", 0) or 0),
                "top_m": int(hard_negative_block.get("top_m", 0) or 0),
                "start_epoch": int(hard_negative_block.get("start_epoch", 0) or 0),
                "weight": float(hard_negative_block.get("weight", 0.0) or 0.0),
                "margin": float(hard_negative_block.get("margin", 0.0) or 0.0),
                "strategy": str(hard_negative_block.get("strategy", "HN-current")),
                "sampling_ratios": dict(hard_negative_block.get("sampling_ratios", {}) or {}),
            }
            if "use_hard_negative" not in hard_negative_block:
                warnings.append("loss.hard_negative.use_hard_negative 未声明，按 false 处理。")
        else:
            required_keys = ("k", "top_m", "start_epoch", "weight", "margin")
            missing_keys = [key for key in required_keys if key not in hard_negative_block]
            if missing_keys:
                raise KeyError(
                    "loss.hard_negative 已启用，但缺少必要字段: "
                    + ", ".join(missing_keys)
                )
            hard_negative_cfg = {
                "use_hard_negative": True,
                "k": int(hard_negative_block["k"]),
                "top_m": int(hard_negative_block["top_m"]),
                "start_epoch": int(hard_negative_block["start_epoch"]),
                "weight": float(hard_negative_block["weight"]),
                "margin": float(hard_negative_block["margin"]),
                "strategy": str(hard_negative_block.get("strategy", "HN-current")),
                "sampling_ratios": dict(hard_negative_block.get("sampling_ratios", {}) or {}),
            }

    return {
        "loss_name": str(loss_cfg["loss_name"]),
        "temperature": float(loss_cfg["temperature"]),
        "poly_epsilon": float(poly_epsilon),
        "hard_negative": hard_negative_cfg,
        "warnings": warnings,
    }


def build_training_effective_config(
    *,
    config_path: str | Path,
    raw_config: Mapping[str, Any],
    resolved_train_files: list[Path],
    model_pipeline_config: Mapping[str, Any],
    resolved_loss_config: Mapping[str, Any],
) -> dict[str, Any]:
    train_cfg = dict(raw_config.get("train", {}))
    data_cfg = dict(raw_config.get("data", {}))
    optimizer_cfg = dict(raw_config.get("optimizer", {}))
    paths_cfg = dict(raw_config.get("paths", {}))

    return {
        "mainline_contract": {
            **TRUSTED_MAINLINE,
            "active_config_path": str(Path(config_path).resolve()),
            "is_trusted_training_config": is_trusted_training_config(config_path),
        },
        "paths": {
            "save_dir": str(Path(paths_cfg["save_dir"]).resolve()) if "save_dir" in paths_cfg else None,
            "hpo_index_path": paths_cfg.get("hpo_index_path"),
            "disease_index_path": paths_cfg.get("disease_index_path"),
            "disease_incidence_path": paths_cfg.get("disease_incidence_path"),
            "resolved_train_files": [str(path.resolve()) for path in resolved_train_files],
        },
        "data": {
            "batch_size": int(data_cfg["batch_size"]),
            "val_ratio": float(data_cfg["val_ratio"]),
            "shuffle": bool(data_cfg["shuffle"]),
            "random_seed": int(data_cfg["random_seed"]),
        },
        "model_pipeline": to_serializable(model_pipeline_config["model"]),
        "module_switches": {
            "encoder_type": model_pipeline_config["model"]["encoder"]["type"],
            "readout_type": model_pipeline_config["model"]["readout"]["type"],
            "scorer_type": model_pipeline_config["model"]["scorer"]["type"],
            "case_refiner_enabled": bool(
                model_pipeline_config["model"].get("case_refiner", {}).get("enabled", False)
            ),
        },
        "loss": to_serializable(resolved_loss_config),
        "optimizer": to_serializable(optimizer_cfg),
        "train": {
            "num_epochs": int(train_cfg["num_epochs"]),
            "device": str(train_cfg["device"]),
            "init_checkpoint_path": train_cfg.get("init_checkpoint_path"),
            "hpo_dropout_prob": float(train_cfg.get("hpo_dropout_prob", 0.0)),
            "hpo_corruption_prob": float(train_cfg.get("hpo_corruption_prob", 0.0)),
            "eval_every": int(train_cfg["eval_every"]),
            "log_every": int(train_cfg.get("log_every", 96)),
            "early_stop": bool(train_cfg["early_stop"]),
            "early_stop_patience": int(train_cfg["early_stop_patience"]),
            "monitor": str(train_cfg["monitor"]),
            "monitor_mode": str(train_cfg["monitor_mode"]),
            "save_best": bool(train_cfg["save_best"]),
            "save_last": bool(train_cfg["save_last"]),
            "save_every_eval": bool(train_cfg["save_every_eval"]),
            "grad_clip_norm": train_cfg.get("grad_clip_norm"),
            "num_workers": int(train_cfg.get("num_workers", 0)),
        },
    }


def build_evaluation_effective_config(
    *,
    data_config_path: str | Path,
    train_config_path: str | Path,
    checkpoint_path: str | Path,
    resolved_train_files: list[Path],
    resolved_test_files: list[Path],
    model_pipeline_config: Mapping[str, Any],
    resolved_loss_config: Mapping[str, Any],
    overlap_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "mainline_contract": {
            **TRUSTED_MAINLINE,
            "active_train_config_path": str(Path(train_config_path).resolve()),
            "active_data_config_path": str(Path(data_config_path).resolve()),
            "is_trusted_training_config": is_trusted_training_config(train_config_path),
            "is_trusted_eval_config": is_trusted_eval_config(data_config_path),
        },
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "resolved_train_files": [str(path.resolve()) for path in resolved_train_files],
        "resolved_test_files": [str(path.resolve()) for path in resolved_test_files],
        "model_pipeline": to_serializable(model_pipeline_config["model"]),
        "module_switches": {
            "encoder_type": model_pipeline_config["model"]["encoder"]["type"],
            "readout_type": model_pipeline_config["model"]["readout"]["type"],
            "scorer_type": model_pipeline_config["model"]["scorer"]["type"],
            "case_refiner_enabled": bool(
                model_pipeline_config["model"].get("case_refiner", {}).get("enabled", False)
            ),
        },
        "loss": to_serializable(resolved_loss_config),
        "case_id_overlap_check": to_serializable(overlap_summary),
    }
