from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_pretrain.yaml"
DEFAULT_FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "train_finetune_attn_ru020.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_RESIDUALS = [0.1, 0.3, 0.5, 0.7, 0.9]


@dataclass(slots=True)
class ExperimentPaths:
    """单组 residual 实验涉及到的关键路径。"""

    residual: float
    label: str
    root_dir: Path
    pretrain_save_dir: Path
    finetune_save_dir: Path
    pretrain_config_path: Path
    finetune_config_path: Path
    pretrain_log_path: Path
    finetune_log_path: Path
    eval_log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 case_refiner residual 对比实验：预训练 -> 微调 -> 评估。"
    )
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        default=DEFAULT_PRETRAIN_CONFIG,
        help="预训练基础配置文件路径。",
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
        "--residuals",
        type=float,
        nargs="+",
        default=DEFAULT_RESIDUALS,
        help="需要对比的 case_refiner.residual 列表。",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="实验总输出目录；留空时自动按时间戳创建。",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="执行训练/评估所使用的 Python 解释器。",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任意一组失败时立即停止；默认会记录失败并继续跑下一组。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成配置和执行计划，不真正启动训练/评估。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件内容必须是字典：{path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # 保留字段顺序，便于第二天人工查看具体配置。
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def dump_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_label(residual: float) -> str:
    """把 0.7 之类的值转换成适合目录名的标签。"""
    return f"res_{residual:.1f}".replace(".", "p")


def ensure_case_refiner_enabled(config: dict[str, Any], residual: float) -> None:
    """在不改动原始配置文件的前提下，给临时配置显式打开 case_refiner。"""
    model_cfg = config.setdefault("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    case_refiner_cfg = model_cfg.setdefault("case_refiner", {})
    case_refiner_cfg["enabled"] = True
    case_refiner_cfg["mlp_hidden_dim"] = int(case_refiner_cfg.get("mlp_hidden_dim", hidden_dim))
    case_refiner_cfg["residual"] = float(residual)


def resolve_run_root(run_root: Path | None) -> Path:
    if run_root is not None:
        return run_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / "case_refiner_residual_sweep" / timestamp).resolve()


def build_experiment_paths(run_root: Path, residual: float) -> ExperimentPaths:
    label = make_label(residual)
    exp_root = run_root / label
    generated_cfg_dir = exp_root / "generated_configs"
    logs_dir = exp_root / "logs"
    return ExperimentPaths(
        residual=float(residual),
        label=label,
        root_dir=exp_root,
        pretrain_save_dir=exp_root / "pretrain",
        finetune_save_dir=exp_root / "finetune",
        pretrain_config_path=generated_cfg_dir / f"train_pretrain_{label}.yaml",
        finetune_config_path=generated_cfg_dir / f"train_finetune_{label}.yaml",
        pretrain_log_path=logs_dir / "pretrain.log",
        finetune_log_path=logs_dir / "finetune.log",
        eval_log_path=logs_dir / "evaluate.log",
    )


def prepare_configs(
    *,
    pretrain_config_path: Path,
    finetune_config_path: Path,
    paths: ExperimentPaths,
) -> dict[str, Any]:
    pretrain_cfg = load_yaml(pretrain_config_path)
    finetune_cfg = load_yaml(finetune_config_path)

    # 预训练也显式开启 case_refiner，并把 residual 绑定到当前 sweep 值。
    ensure_case_refiner_enabled(pretrain_cfg, paths.residual)
    ensure_case_refiner_enabled(finetune_cfg, paths.residual)

    pretrain_cfg.setdefault("paths", {})
    pretrain_cfg["paths"]["save_dir"] = str(paths.pretrain_save_dir)

    finetune_cfg.setdefault("paths", {})
    finetune_cfg["paths"]["save_dir"] = str(paths.finetune_save_dir)
    finetune_cfg.setdefault("train", {})
    finetune_cfg["train"]["init_checkpoint_path"] = str(
        paths.pretrain_save_dir / "checkpoints" / "best.pt"
    )

    dump_yaml(paths.pretrain_config_path, pretrain_cfg)
    dump_yaml(paths.finetune_config_path, finetune_cfg)

    return {
        "pretrain": pretrain_cfg,
        "finetune": finetune_cfg,
    }


def build_stage_commands(
    *,
    python_executable: str,
    data_config_path: Path,
    paths: ExperimentPaths,
) -> dict[str, list[str]]:
    pretrain_best_ckpt = paths.pretrain_save_dir / "checkpoints" / "best.pt"
    finetune_best_ckpt = paths.finetune_save_dir / "checkpoints" / "best.pt"
    return {
        "pretrain": [
            python_executable,
            "-m",
            "src.training.trainer",
            "--config",
            str(paths.pretrain_config_path),
        ],
        "finetune": [
            python_executable,
            "-m",
            "src.training.trainer",
            "--config",
            str(paths.finetune_config_path),
        ],
        "evaluate": [
            python_executable,
            "-m",
            "src.evaluation.evaluator",
            "--data_config_path",
            str(data_config_path),
            "--train_config_path",
            str(paths.finetune_config_path),
            "--checkpoint_path",
            str(finetune_best_ckpt),
        ],
        "pretrain_best_ckpt": [str(pretrain_best_ckpt)],
        "finetune_best_ckpt": [str(finetune_best_ckpt)],
    }


def run_and_tee(command: list[str], *, cwd: Path, log_path: Path) -> int:
    """把命令输出同时写到终端和日志，方便第二天追溯。"""
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


def find_latest_summary_json(finetune_save_dir: Path) -> Path | None:
    eval_dir = finetune_save_dir / "evaluation"
    if not eval_dir.is_dir():
        return None
    summary_files = sorted(eval_dir.glob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return summary_files[0] if summary_files else None


def update_csv(run_root: Path, experiments: list[dict[str, Any]]) -> None:
    csv_path = run_root / "sweep_results.csv"
    rows = []
    for exp in experiments:
        rows.append(
            {
                "label": exp["label"],
                "residual": exp["residual"],
                "status": exp["status"],
                "pretrain_status": exp["stages"]["pretrain"]["status"],
                "finetune_status": exp["stages"]["finetune"]["status"],
                "evaluate_status": exp["stages"]["evaluate"]["status"],
                "top1": exp.get("metrics", {}).get("top1"),
                "top3": exp.get("metrics", {}).get("top3"),
                "top5": exp.get("metrics", {}).get("top5"),
                "summary_path": exp.get("evaluation_summary_path"),
                "pretrain_save_dir": exp["pretrain_save_dir"],
                "finetune_save_dir": exp["finetune_save_dir"],
            }
        )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "residual",
                "status",
                "pretrain_status",
                "finetune_status",
                "evaluate_status",
                "top1",
                "top3",
                "top5",
                "summary_path",
                "pretrain_save_dir",
                "finetune_save_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def update_state(run_root: Path, state: dict[str, Any]) -> None:
    dump_json(run_root / "sweep_state.json", state)
    update_csv(run_root, state["experiments"])


def main() -> None:
    args = parse_args()
    run_root = resolve_run_root(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    residuals = [float(value) for value in args.residuals]
    experiments: list[dict[str, Any]] = []

    state: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "pretrain_config": str(args.pretrain_config.resolve()),
        "finetune_config": str(args.finetune_config.resolve()),
        "data_config": str(args.data_config.resolve()),
        "python": args.python,
        "dry_run": bool(args.dry_run),
        "residuals": residuals,
        "experiments": experiments,
    }
    update_state(run_root, state)

    print(f"[INFO] 实验总目录: {run_root}")
    print(f"[INFO] residual 列表: {', '.join(f'{value:.1f}' for value in residuals)}")

    for residual in residuals:
        paths = build_experiment_paths(run_root, residual)
        prepared_cfgs = prepare_configs(
            pretrain_config_path=args.pretrain_config.resolve(),
            finetune_config_path=args.finetune_config.resolve(),
            paths=paths,
        )
        commands = build_stage_commands(
            python_executable=args.python,
            data_config_path=args.data_config.resolve(),
            paths=paths,
        )

        exp_state: dict[str, Any] = {
            "label": paths.label,
            "residual": residual,
            "status": "pending",
            "root_dir": str(paths.root_dir),
            "pretrain_save_dir": str(paths.pretrain_save_dir),
            "finetune_save_dir": str(paths.finetune_save_dir),
            "pretrain_config_path": str(paths.pretrain_config_path),
            "finetune_config_path": str(paths.finetune_config_path),
            "stages": {
                "pretrain": {
                    "status": "pending",
                    "command": commands["pretrain"],
                    "log_path": str(paths.pretrain_log_path),
                },
                "finetune": {
                    "status": "pending",
                    "command": commands["finetune"],
                    "log_path": str(paths.finetune_log_path),
                },
                "evaluate": {
                    "status": "pending",
                    "command": commands["evaluate"],
                    "log_path": str(paths.eval_log_path),
                },
            },
            "generated_case_refiner": {
                "pretrain": prepared_cfgs["pretrain"]["model"]["case_refiner"],
                "finetune": prepared_cfgs["finetune"]["model"]["case_refiner"],
            },
        }
        experiments.append(exp_state)
        update_state(run_root, state)

        print("=" * 80)
        print(f"[INFO] 开始实验 {paths.label} (case_refiner.residual={residual:.1f})")
        print(f"[INFO] 预训练输出目录: {paths.pretrain_save_dir}")
        print(f"[INFO] 微调输出目录: {paths.finetune_save_dir}")

        if args.dry_run:
            exp_state["status"] = "dry_run_only"
            for stage_name in ("pretrain", "finetune", "evaluate"):
                exp_state["stages"][stage_name]["status"] = "planned"
            update_state(run_root, state)
            continue

        failed = False
        for stage_name, log_path in (
            ("pretrain", paths.pretrain_log_path),
            ("finetune", paths.finetune_log_path),
            ("evaluate", paths.eval_log_path),
        ):
            print("-" * 80)
            print(f"[INFO] {paths.label} -> {stage_name} 开始")
            exp_state["stages"][stage_name]["status"] = "running"
            update_state(run_root, state)

            return_code = run_and_tee(
                exp_state["stages"][stage_name]["command"],
                cwd=PROJECT_ROOT,
                log_path=log_path,
            )

            if return_code != 0:
                exp_state["stages"][stage_name]["status"] = "failed"
                exp_state["stages"][stage_name]["return_code"] = return_code
                exp_state["status"] = "failed"
                failed = True
                update_state(run_root, state)
                print(
                    f"[ERROR] {paths.label} -> {stage_name} 失败，返回码={return_code}。"
                )
                if args.stop_on_error:
                    raise SystemExit(return_code)
                break

            exp_state["stages"][stage_name]["status"] = "completed"
            exp_state["stages"][stage_name]["return_code"] = return_code
            update_state(run_root, state)

        if failed:
            continue

        summary_path = find_latest_summary_json(paths.finetune_save_dir)
        if summary_path is not None:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            exp_state["evaluation_summary_path"] = str(summary_path)
            exp_state["metrics"] = {
                "top1": summary.get("top1"),
                "top3": summary.get("top3"),
                "top5": summary.get("top5"),
                "num_cases": summary.get("num_cases"),
                "num_evaluable": summary.get("num_evaluable"),
                "checkpoint_path": summary.get("checkpoint_path"),
            }

        exp_state["status"] = "completed"
        update_state(run_root, state)

    print("=" * 80)
    print("[DONE] 所有 residual 实验已处理完成。")
    print(f"[DONE] 总目录: {run_root}")
    print(f"[DONE] 汇总文件: {run_root / 'sweep_results.csv'}")
    print(f"[DONE] 状态文件: {run_root / 'sweep_state.json'}")


if __name__ == "__main__":
    main()
