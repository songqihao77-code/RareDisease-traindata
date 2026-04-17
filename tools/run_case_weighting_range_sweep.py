from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "train_finetune.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_PRETRAIN_CHECKPOINT = PROJECT_ROOT / "outputs" / "stage1_pretrain" / "checkpoints" / "best.pt"
DEFAULT_SEEDS = [42, 43]


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    case_weighting_enabled: bool
    residual_uniform: float
    weight_min: float
    weight_max: float


EXPERIMENTS = [
    ExperimentSpec(
        name="baseline_cw0_ru020",
        case_weighting_enabled=False,
        residual_uniform=0.2,
        weight_min=0.5,
        weight_max=1.5,
    ),
    ExperimentSpec(
        name="fixed_ic_w080_120_cw1_ru020",
        case_weighting_enabled=True,
        residual_uniform=0.2,
        weight_min=0.8,
        weight_max=1.2,
    ),
    ExperimentSpec(
        name="fixed_ic_w090_110_cw1_ru020",
        case_weighting_enabled=True,
        residual_uniform=0.2,
        weight_min=0.9,
        weight_max=1.1,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 baseline 与 fixed-IC 不同权重范围的多 seed 微调评估实验。"
    )
    parser.add_argument("--finetune-config", type=Path, default=DEFAULT_FINETUNE_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--pretrain-checkpoint", type=Path, default=DEFAULT_PRETRAIN_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
    return (PROJECT_ROOT / "outputs" / "case_weighting_range_sweep" / timestamp).resolve()


def build_case_weighting_block(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "enabled": bool(spec.case_weighting_enabled),
        "mode": "fixed_ic",
        "ic_source": "h_disease_binary_df",
        "eps": 1e-8,
        "clip_quantile_low": 0.05,
        "clip_quantile_high": 0.95,
        "weight_min": float(spec.weight_min),
        "weight_max": float(spec.weight_max),
        "zero_support_policy": "identity",
    }


def build_experiment_config(
    *,
    base_config: dict[str, Any],
    spec: ExperimentSpec,
    seed: int,
    pretrain_checkpoint: Path,
    save_dir: Path,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("paths", {})
    config["paths"]["save_dir"] = str(save_dir)

    data_cfg = config.setdefault("data", {})
    data_cfg["random_seed"] = int(seed)

    model_cfg = config.setdefault("model", {})
    readout_cfg = model_cfg.setdefault("readout", {})
    readout_cfg["residual_uniform"] = float(spec.residual_uniform)
    model_cfg["case_weighting"] = build_case_weighting_block(spec)

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


def find_output_by_suffix(renamed_outputs: dict[str, str], suffix: str) -> str | None:
    for name, path_str in renamed_outputs.items():
        if name.endswith(suffix):
            return path_str
    return None


def update_summary_csv(run_root: Path, experiments: list[dict[str, Any]]) -> None:
    csv_path = run_root / "range_sweep_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "experiment",
                "case_weighting_enabled",
                "residual_uniform",
                "weight_min",
                "weight_max",
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
                    "seed": exp["seed"],
                    "experiment": exp["experiment"],
                    "case_weighting_enabled": exp["case_weighting_enabled"],
                    "residual_uniform": exp["residual_uniform"],
                    "weight_min": exp["weight_min"],
                    "weight_max": exp["weight_max"],
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

    base_config = load_yaml(finetune_config_path)
    generated_config_dir = run_root / "generated_configs"
    manifest_path = run_root / "range_sweep_manifest.json"
    experiments_state: list[dict[str, Any]] = []

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "python": args.python,
        "finetune_config": str(finetune_config_path),
        "data_config": str(data_config_path),
        "pretrain_checkpoint": str(pretrain_checkpoint),
        "seeds": [int(seed) for seed in args.seeds],
        "experiments": [asdict(spec) for spec in EXPERIMENTS],
        "dry_run": bool(args.dry_run),
        "runs": experiments_state,
    }
    dump_json(manifest_path, manifest)

    print(f"[INFO] 总输出目录: {run_root}")
    print(f"[INFO] 随机种子: {', '.join(str(int(seed)) for seed in args.seeds)}")
    print("[INFO] 实验列表: baseline_cw0_ru020, fixed_ic_w080_120_cw1_ru020, fixed_ic_w090_110_cw1_ru020")

    for seed in args.seeds:
        seed = int(seed)
        for spec in EXPERIMENTS:
            run_name = f"{spec.name}_seed{seed:03d}"
            exp_root = run_root / run_name
            config_path = generated_config_dir / f"train_finetune_{run_name}.yaml"
            checkpoint_path = exp_root / "checkpoints" / "best.pt"
            train_log = exp_root / "logs" / f"{run_name}__train.log"
            eval_log = exp_root / "logs" / f"{run_name}__evaluate.log"

            config = build_experiment_config(
                base_config=base_config,
                spec=spec,
                seed=seed,
                pretrain_checkpoint=pretrain_checkpoint,
                save_dir=exp_root,
            )
            dump_yaml(config_path, config)

            train_cmd = [args.python, "-m", "src.training.trainer", "--config", str(config_path)]
            eval_cmd = [
                args.python,
                "-m",
                "src.evaluation.evaluator",
                "--data_config_path",
                str(data_config_path),
                "--train_config_path",
                str(config_path),
                "--checkpoint_path",
                str(checkpoint_path),
            ]

            exp_state: dict[str, Any] = {
                "seed": seed,
                "experiment": spec.name,
                "run_name": run_name,
                "case_weighting_enabled": spec.case_weighting_enabled,
                "residual_uniform": spec.residual_uniform,
                "weight_min": spec.weight_min,
                "weight_max": spec.weight_max,
                "save_dir": str(exp_root),
                "config_path": str(config_path),
                "train_log": str(train_log),
                "eval_log": str(eval_log),
                "status": "planned" if args.dry_run else "pending",
            }
            experiments_state.append(exp_state)
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, experiments_state)

            print("=" * 80)
            print(
                f"[INFO] {run_name}: "
                f"case_weighting={spec.case_weighting_enabled}, "
                f"residual_uniform={spec.residual_uniform:.1f}, "
                f"weight_range=[{spec.weight_min:.1f}, {spec.weight_max:.1f}]"
            )
            print(f"[INFO] 配置文件: {config_path}")
            print(f"[INFO] 结果目录: {exp_root}")

            if args.dry_run:
                continue

            train_code = run_and_tee(train_cmd, cwd=PROJECT_ROOT, log_path=train_log)
            exp_state["train_return_code"] = train_code
            if train_code != 0:
                exp_state["status"] = "train_failed"
                dump_json(manifest_path, manifest)
                update_summary_csv(run_root, experiments_state)
                print(f"[ERROR] {run_name} 训练失败，返回码={train_code}")
                if args.stop_on_error:
                    raise SystemExit(train_code)
                continue

            if not checkpoint_path.is_file():
                exp_state["status"] = "missing_best_checkpoint"
                dump_json(manifest_path, manifest)
                update_summary_csv(run_root, experiments_state)
                print(f"[ERROR] {run_name} 未找到 best checkpoint: {checkpoint_path}")
                if args.stop_on_error:
                    raise SystemExit(1)
                continue

            eval_code = run_and_tee(eval_cmd, cwd=PROJECT_ROOT, log_path=eval_log)
            exp_state["evaluate_return_code"] = eval_code
            if eval_code != 0:
                exp_state["status"] = "evaluate_failed"
                dump_json(manifest_path, manifest)
                update_summary_csv(run_root, experiments_state)
                print(f"[ERROR] {run_name} 评估失败，返回码={eval_code}")
                if args.stop_on_error:
                    raise SystemExit(eval_code)
                continue

            renamed_outputs = rename_evaluation_outputs(exp_root / "evaluation", run_name)
            exp_state["renamed_evaluation_outputs"] = renamed_outputs
            summary_json = find_output_by_suffix(renamed_outputs, "_summary.json")
            if summary_json is not None:
                exp_state["summary_json"] = summary_json
                with Path(summary_json).open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                exp_state["metrics"] = {
                    "top1": summary.get("top1"),
                    "top3": summary.get("top3"),
                    "top5": summary.get("top5"),
                    "num_cases": summary.get("num_cases"),
                    "num_evaluable": summary.get("num_evaluable"),
                }

            exp_state["status"] = "completed"
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, experiments_state)

    print("=" * 80)
    print("[DONE] 权重范围 sweep 已完成。")
    print(f"[DONE] 总目录: {run_root}")
    print(f"[DONE] 汇总 CSV: {run_root / 'range_sweep_results.csv'}")
    print(f"[DONE] 汇总 JSON: {manifest_path}")


if __name__ == "__main__":
    main()
