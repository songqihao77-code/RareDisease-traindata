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
DEFAULT_PRETRAIN_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "stage1_pretrain" / "checkpoints" / "best.pt"
)


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    case_weighting_enabled: bool
    apply_to: str
    residual_uniform: float
    weight_min: float
    weight_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 baseline / fixed_ic_both / fixed_ic_readout_only 三组微调与评估。"
    )
    parser.add_argument("--finetune-config", type=Path, default=DEFAULT_FINETUNE_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--pretrain-checkpoint", type=Path, default=DEFAULT_PRETRAIN_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=None, help="留空时使用 finetune config 中的 data.random_seed。")
    parser.add_argument("--weight-min", type=float, default=0.8)
    parser.add_argument("--weight-max", type=float, default=1.2)
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
    return (PROJECT_ROOT / "outputs" / "case_weighting_apply_to_ablation" / timestamp).resolve()


def resolve_seed(base_config: dict[str, Any], seed_arg: int | None) -> int:
    if seed_arg is not None:
        return int(seed_arg)
    return int(base_config.get("data", {}).get("random_seed", 42))


def build_experiments(seed: int, weight_min: float, weight_max: float) -> list[ExperimentSpec]:
    weight_tag = f"w{int(round(weight_min * 100)):03d}_{int(round(weight_max * 100)):03d}"
    seed_tag = f"seed{seed:03d}"
    return [
        ExperimentSpec(
            name=f"baseline_cw0_ru020_{seed_tag}",
            case_weighting_enabled=False,
            apply_to="both",
            residual_uniform=0.2,
            weight_min=weight_min,
            weight_max=weight_max,
        ),
        ExperimentSpec(
            name=f"fixed_ic_both_{weight_tag}_cw1_ru020_{seed_tag}",
            case_weighting_enabled=True,
            apply_to="both",
            residual_uniform=0.2,
            weight_min=weight_min,
            weight_max=weight_max,
        ),
        ExperimentSpec(
            name=f"fixed_ic_readout_only_{weight_tag}_cw1_ru020_{seed_tag}",
            case_weighting_enabled=True,
            apply_to="readout_only",
            residual_uniform=0.2,
            weight_min=weight_min,
            weight_max=weight_max,
        ),
    ]


def build_case_weighting_block(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "enabled": bool(spec.case_weighting_enabled),
        "mode": "fixed_ic",
        "apply_to": spec.apply_to,
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


def collect_evaluation_outputs(eval_dir: Path) -> dict[str, str]:
    collected: dict[str, str] = {}
    if not eval_dir.is_dir():
        return collected

    for path in sorted(eval_dir.iterdir()):
        if not path.is_file():
            continue
        collected[path.name] = str(path)
    return collected


def find_output_by_suffix(collected_outputs: dict[str, str], suffix: str) -> str | None:
    for name, path_str in collected_outputs.items():
        if name.endswith(suffix):
            return path_str
    return None


def update_summary_csv(run_root: Path, runs: list[dict[str, Any]]) -> None:
    csv_path = run_root / "apply_to_ablation_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "case_weighting_enabled",
                "apply_to",
                "residual_uniform",
                "weight_min",
                "weight_max",
                "seed",
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
        for run in runs:
            metrics = run.get("metrics", {})
            writer.writerow(
                {
                    "experiment": run["experiment"],
                    "case_weighting_enabled": run["case_weighting_enabled"],
                    "apply_to": run["apply_to"],
                    "residual_uniform": run["residual_uniform"],
                    "weight_min": run["weight_min"],
                    "weight_max": run["weight_max"],
                    "seed": run["seed"],
                    "status": run["status"],
                    "top1": metrics.get("top1"),
                    "top3": metrics.get("top3"),
                    "top5": metrics.get("top5"),
                    "save_dir": run["save_dir"],
                    "config_path": run["config_path"],
                    "summary_json": run.get("summary_json"),
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
    if args.weight_min > args.weight_max:
        raise ValueError("weight_min 不能大于 weight_max。")

    base_config = load_yaml(finetune_config_path)
    seed = resolve_seed(base_config, args.seed)
    experiments = build_experiments(seed, args.weight_min, args.weight_max)

    run_root = resolve_run_root(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    generated_config_dir = run_root / "generated_configs"
    manifest_path = run_root / "apply_to_ablation_manifest.json"
    runs_state: list[dict[str, Any]] = []

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "python": args.python,
        "finetune_config": str(finetune_config_path),
        "data_config": str(data_config_path),
        "pretrain_checkpoint": str(pretrain_checkpoint),
        "seed": seed,
        "weight_min": float(args.weight_min),
        "weight_max": float(args.weight_max),
        "dry_run": bool(args.dry_run),
        "experiments": [],
        "runs": runs_state,
    }
    dump_json(manifest_path, manifest)

    print(f"[INFO] 总输出目录: {run_root}")
    print(f"[INFO] 使用 seed: {seed}")
    print(f"[INFO] weight_range=[{args.weight_min:.1f}, {args.weight_max:.1f}]")

    for spec in experiments:
        exp_root = run_root / spec.name
        config_path = generated_config_dir / f"train_finetune_{spec.name}.yaml"
        checkpoint_path = exp_root / "checkpoints" / "best.pt"
        train_log = exp_root / "logs" / f"{spec.name}__train.log"
        eval_log = exp_root / "logs" / f"{spec.name}__evaluate.log"

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

        run_state: dict[str, Any] = {
            "experiment": spec.name,
            "case_weighting_enabled": spec.case_weighting_enabled,
            "apply_to": spec.apply_to,
            "residual_uniform": spec.residual_uniform,
            "weight_min": spec.weight_min,
            "weight_max": spec.weight_max,
            "seed": seed,
            "save_dir": str(exp_root),
            "config_path": str(config_path),
            "train_log": str(train_log),
            "eval_log": str(eval_log),
            "train_command": train_cmd,
            "eval_command": eval_cmd,
            "status": "planned" if args.dry_run else "pending",
        }
        manifest["experiments"].append(
            {
                "name": spec.name,
                "case_weighting_enabled": spec.case_weighting_enabled,
                "apply_to": spec.apply_to,
                "residual_uniform": spec.residual_uniform,
                "weight_min": spec.weight_min,
                "weight_max": spec.weight_max,
            }
        )
        runs_state.append(run_state)
        dump_json(manifest_path, manifest)
        update_summary_csv(run_root, runs_state)

        print("=" * 80)
        print(
            f"[INFO] {spec.name}: "
            f"case_weighting={spec.case_weighting_enabled}, "
            f"apply_to={spec.apply_to}, "
            f"weight_range=[{spec.weight_min:.1f}, {spec.weight_max:.1f}], "
            f"residual_uniform={spec.residual_uniform:.1f}"
        )
        print(f"[INFO] 配置文件: {config_path}")
        print(f"[INFO] 结果目录: {exp_root}")

        if args.dry_run:
            continue

        train_code = run_and_tee(train_cmd, cwd=PROJECT_ROOT, log_path=train_log)
        run_state["train_return_code"] = train_code
        if train_code != 0:
            run_state["status"] = "train_failed"
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, runs_state)
            print(f"[ERROR] {spec.name} 训练失败，返回码={train_code}")
            if args.stop_on_error:
                raise SystemExit(train_code)
            continue

        if not checkpoint_path.is_file():
            run_state["status"] = "missing_best_checkpoint"
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, runs_state)
            print(f"[ERROR] {spec.name} 未找到 best checkpoint: {checkpoint_path}")
            if args.stop_on_error:
                raise SystemExit(1)
            continue

        eval_code = run_and_tee(eval_cmd, cwd=PROJECT_ROOT, log_path=eval_log)
        run_state["evaluate_return_code"] = eval_code
        if eval_code != 0:
            run_state["status"] = "evaluate_failed"
            dump_json(manifest_path, manifest)
            update_summary_csv(run_root, runs_state)
            print(f"[ERROR] {spec.name} 评估失败，返回码={eval_code}")
            if args.stop_on_error:
                raise SystemExit(eval_code)
            continue

        evaluation_outputs = collect_evaluation_outputs(exp_root / "evaluation")
        run_state["evaluation_outputs"] = evaluation_outputs

        summary_json = find_output_by_suffix(evaluation_outputs, "_summary.json")
        if summary_json is not None:
            run_state["summary_json"] = summary_json
            with Path(summary_json).open("r", encoding="utf-8") as f:
                summary = json.load(f)
            run_state["metrics"] = {
                "top1": summary.get("top1"),
                "top3": summary.get("top3"),
                "top5": summary.get("top5"),
                "num_cases": summary.get("num_cases"),
                "num_evaluable": summary.get("num_evaluable"),
            }

        run_state["status"] = "completed"
        dump_json(manifest_path, manifest)
        update_summary_csv(run_root, runs_state)

    print("=" * 80)
    print("[DONE] 三组 apply_to 对照实验已处理完成。")
    print(f"[DONE] 总目录: {run_root}")
    print(f"[DONE] 汇总 CSV: {run_root / 'apply_to_ablation_results.csv'}")
    print(f"[DONE] 汇总 JSON: {manifest_path}")


if __name__ == "__main__":
    main()
