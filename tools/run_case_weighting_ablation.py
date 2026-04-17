from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "train_finetune.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_PRETRAIN_CHECKPOINT = PROJECT_ROOT / "outputs" / "stage1_pretrain" / "checkpoints" / "best.pt"


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    case_weighting_enabled: bool
    residual_uniform: float


EXPERIMENTS = [
    ExperimentSpec(name="baseline_cw0_ru020", case_weighting_enabled=False, residual_uniform=0.2),
    ExperimentSpec(name="fixed_ic_cw1_ru020", case_weighting_enabled=True, residual_uniform=0.2),
    ExperimentSpec(name="fixed_ic_cw1_ru010", case_weighting_enabled=True, residual_uniform=0.1),
    ExperimentSpec(name="fixed_ic_cw1_ru000", case_weighting_enabled=True, residual_uniform=0.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 case weighting 四组微调 + 评估对照实验。"
    )
    parser.add_argument(
        "--finetune-config",
        type=Path,
        default=DEFAULT_FINETUNE_CONFIG,
        help="微调基础配置文件路径。",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="评估数据配置文件路径。",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=DEFAULT_PRETRAIN_CHECKPOINT,
        help="第一阶段预训练 checkpoint 路径。",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="总输出目录；留空时自动按时间戳创建。",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="执行训练/评估时使用的 Python 解释器。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成配置与执行计划，不真正启动训练和评估。",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任意一组失败时立即停止；默认继续后续实验。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件内容必须是字典: {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def dump_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_run_root(run_root: Path | None) -> Path:
    if run_root is not None:
        return run_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / "case_weighting_ablation" / timestamp).resolve()


def build_case_weighting_block(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": "fixed_ic",
        "ic_source": "h_disease_binary_df",
        "eps": 1e-8,
        "clip_quantile_low": 0.05,
        "clip_quantile_high": 0.95,
        "weight_min": 0.5,
        "weight_max": 1.5,
        "zero_support_policy": "identity",
    }


def make_experiment_config(
    *,
    base_config: dict[str, Any],
    experiment: ExperimentSpec,
    pretrain_checkpoint: Path,
    save_dir: Path,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("paths", {})
    config["paths"]["save_dir"] = str(save_dir)

    model_cfg = config.setdefault("model", {})
    readout_cfg = model_cfg.setdefault("readout", {})
    readout_cfg["residual_uniform"] = float(experiment.residual_uniform)
    model_cfg["case_weighting"] = build_case_weighting_block(experiment.case_weighting_enabled)

    train_cfg = config.setdefault("train", {})
    train_cfg["init_checkpoint_path"] = str(pretrain_checkpoint)
    return config


def run_and_tee(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()


def rename_evaluation_outputs(eval_dir: Path, prefix: str) -> dict[str, str]:
    renamed: dict[str, str] = {}
    if not eval_dir.is_dir():
        return renamed

    for path in sorted(eval_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(f"{prefix}__"):
            renamed[path.name] = str(path)
            continue

        target = path.with_name(f"{prefix}__{path.name}")
        path.rename(target)
        renamed[target.name] = str(target)
    return renamed


def find_summary_file(renamed_outputs: dict[str, str]) -> Path | None:
    for name, path_str in renamed_outputs.items():
        if name.endswith("_summary.json"):
            return Path(path_str)
    return None


def build_commands(
    *,
    python_executable: str,
    config_path: Path,
    data_config_path: Path,
    checkpoint_path: Path,
) -> dict[str, list[str]]:
    return {
        "train": [
            python_executable,
            "-m",
            "src.training.trainer",
            "--config",
            str(config_path),
        ],
        "evaluate": [
            python_executable,
            "-m",
            "src.evaluation.evaluator",
            "--data_config_path",
            str(data_config_path),
            "--train_config_path",
            str(config_path),
            "--checkpoint_path",
            str(checkpoint_path),
        ],
    }


def update_summary_csv(run_root: Path, experiments: list[dict[str, Any]]) -> None:
    csv_path = run_root / "case_weighting_ablation_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "case_weighting_enabled",
                "residual_uniform",
                "status",
                "top1",
                "top3",
                "top5",
                "save_dir",
                "config_path",
                "summary_json",
            ],
        )
        writer.writeheader()
        for exp in experiments:
            metrics = exp.get("metrics", {})
            writer.writerow(
                {
                    "experiment": exp["experiment"],
                    "case_weighting_enabled": exp["case_weighting_enabled"],
                    "residual_uniform": exp["residual_uniform"],
                    "status": exp["status"],
                    "top1": metrics.get("top1"),
                    "top3": metrics.get("top3"),
                    "top5": metrics.get("top5"),
                    "save_dir": exp["save_dir"],
                    "config_path": exp["config_path"],
                    "summary_json": exp.get("summary_json"),
                }
            )


def main() -> None:
    args = parse_args()
    finetune_config_path = args.finetune_config.resolve()
    data_config_path = args.data_config.resolve()
    pretrain_checkpoint = args.pretrain_checkpoint.resolve()

    if not finetune_config_path.is_file():
        raise FileNotFoundError(f"微调配置不存在: {finetune_config_path}")
    if not data_config_path.is_file():
        raise FileNotFoundError(f"评估配置不存在: {data_config_path}")
    if not pretrain_checkpoint.is_file():
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {pretrain_checkpoint}")

    run_root = resolve_run_root(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    generated_config_dir = run_root / "generated_configs"
    manifest_path = run_root / "case_weighting_ablation_manifest.json"

    base_config = load_yaml(finetune_config_path)
    experiments_state: list[dict[str, Any]] = []

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "python": args.python,
        "finetune_config": str(finetune_config_path),
        "data_config": str(data_config_path),
        "pretrain_checkpoint": str(pretrain_checkpoint),
        "dry_run": bool(args.dry_run),
        "experiments": experiments_state,
    }
    dump_json(manifest_path, manifest)

    print(f"[INFO] 总输出目录: {run_root}")
    print(f"[INFO] 预训练 checkpoint: {pretrain_checkpoint}")

    for experiment in EXPERIMENTS:
        exp_root = run_root / experiment.name
        config_path = generated_config_dir / f"train_finetune_{experiment.name}.yaml"
        checkpoint_path = exp_root / "checkpoints" / "best.pt"
        train_log = exp_root / "logs" / f"{experiment.name}__train.log"
        eval_log = exp_root / "logs" / f"{experiment.name}__evaluate.log"

        config = make_experiment_config(
            base_config=base_config,
            experiment=experiment,
            pretrain_checkpoint=pretrain_checkpoint,
            save_dir=exp_root,
        )
        dump_yaml(config_path, config)

        commands = build_commands(
            python_executable=args.python,
            config_path=config_path,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint_path,
        )

        exp_state: dict[str, Any] = {
            "experiment": experiment.name,
            "case_weighting_enabled": experiment.case_weighting_enabled,
            "residual_uniform": experiment.residual_uniform,
            "save_dir": str(exp_root),
            "config_path": str(config_path),
            "train_log": str(train_log),
            "eval_log": str(eval_log),
            "status": "planned" if args.dry_run else "pending",
            "commands": commands,
        }
        experiments_state.append(exp_state)
        dump_json(manifest_path, manifest)
        update_summary_csv(run_root, experiments_state)

        print("=" * 80)
        print(
            f"[INFO] {experiment.name}: "
            f"case_weighting={experiment.case_weighting_enabled}, "
            f"residual_uniform={experiment.residual_uniform:.1f}"
        )
        print(f"[INFO] 配置文件: {config_path}")
        print(f"[INFO] 结果目录: {exp_root}")

        if args.dry_run:
            continue

        train_code = run_and_tee(commands["train"], cwd=PROJECT_ROOT, log_path=train_log)
        exp_state["train_return_code"] = train_code
        if train_code != 0:
            exp_state["status"] = "train_failed"
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, experiments_state)
            print(f"[ERROR] {experiment.name} 训练失败，返回码={train_code}")
            if args.stop_on_error:
                raise SystemExit(train_code)
            continue

        if not checkpoint_path.is_file():
            exp_state["status"] = "missing_best_checkpoint"
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, experiments_state)
            print(f"[ERROR] {experiment.name} 未找到 best checkpoint: {checkpoint_path}")
            if args.stop_on_error:
                raise SystemExit(1)
            continue

        eval_code = run_and_tee(commands["evaluate"], cwd=PROJECT_ROOT, log_path=eval_log)
        exp_state["evaluate_return_code"] = eval_code
        if eval_code != 0:
            exp_state["status"] = "evaluate_failed"
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, experiments_state)
            print(f"[ERROR] {experiment.name} 评估失败，返回码={eval_code}")
            if args.stop_on_error:
                raise SystemExit(eval_code)
            continue

        renamed_outputs = rename_evaluation_outputs(exp_root / "evaluation", experiment.name)
        exp_state["renamed_evaluation_outputs"] = renamed_outputs

        summary_file = find_summary_file(renamed_outputs)
        if summary_file is not None and summary_file.is_file():
            with summary_file.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            exp_state["summary_json"] = str(summary_file)
            exp_state["metrics"] = {
                "top1": summary.get("top1"),
                "top3": summary.get("top3"),
                "top5": summary.get("top5"),
                "num_cases": summary.get("num_cases"),
                "num_evaluable": summary.get("num_evaluable"),
                "checkpoint_path": summary.get("checkpoint_path"),
            }

        exp_state["status"] = "completed"
        dump_json(manifest_path, manifest)
        update_summary_csv(run_root, experiments_state)

    print("=" * 80)
    print("[DONE] 四组 case weighting 对照实验处理完成。")
    print(f"[DONE] 总结果目录: {run_root}")
    print(f"[DONE] 汇总 CSV: {run_root / 'case_weighting_ablation_results.csv'}")
    print(f"[DONE] 汇总 JSON: {manifest_path}")
    for exp_state in experiments_state:
        print(
            f"[DONE] {exp_state['experiment']} -> {exp_state['status']} -> {exp_state['save_dir']}"
        )


if __name__ == "__main__":
    main()
