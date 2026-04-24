from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIRS = [
    PROJECT_ROOT / "outputs" / "case_noise_ablation" / "g4b_weighting_idf_seed42",
    PROJECT_ROOT / "outputs" / "case_noise_ablation" / "g4b_weighting_idf_seed123",
    PROJECT_ROOT / "outputs" / "case_noise_ablation" / "g4b_weighting_idf_seed3407",
]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "case_noise_ablation" / "g4b_weighting_idf_3seed_summary.json"
)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / float(len(values) - 1))


def _latest_summary_path(run_dir: Path) -> Path:
    evaluation_dir = run_dir / "evaluation"
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"evaluation 目录不存在: {evaluation_dir}")

    summary_files = sorted(
        evaluation_dir.glob("*_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not summary_files:
        raise FileNotFoundError(f"未找到 summary JSON: {evaluation_dir}")
    return summary_files[0]


def _load_summary(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary JSON 顶层必须是对象: {summary_path}")
    return payload


def _get_mimic_metrics(summary: dict[str, Any], summary_path: Path) -> dict[str, float]:
    per_dataset = summary.get("per_dataset")
    if not isinstance(per_dataset, list):
        raise ValueError(f"per_dataset 缺失或格式错误: {summary_path}")

    for row in per_dataset:
        if isinstance(row, dict) and row.get("dataset_name") == "mimic_test":
            return {
                "top1": float(row["top1"]),
                "top3": float(row["top3"]),
                "top5": float(row["top5"]),
            }
    raise ValueError(f"per_dataset 中未找到 dataset_name='mimic_test': {summary_path}")


def aggregate(run_dirs: list[Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    overall_top1: list[float] = []
    overall_top3: list[float] = []
    overall_top5: list[float] = []
    mimic_top1: list[float] = []
    mimic_top3: list[float] = []
    mimic_top5: list[float] = []

    for run_dir in run_dirs:
        summary_path = _latest_summary_path(run_dir)
        summary = _load_summary(summary_path)
        mimic_metrics = _get_mimic_metrics(summary, summary_path)

        overall = {
            "top1": float(summary["top1"]),
            "top3": float(summary["top3"]),
            "top5": float(summary["top5"]),
        }
        checkpoint_epoch = summary.get("checkpoint_epoch")
        run_record = {
            "run_dir": str(run_dir.resolve()),
            "summary_path": str(summary_path.resolve()),
            "checkpoint_epoch": None if checkpoint_epoch is None else int(checkpoint_epoch),
            "overall": overall,
            "mimic_test": mimic_metrics,
        }
        runs.append(run_record)

        overall_top1.append(overall["top1"])
        overall_top3.append(overall["top3"])
        overall_top5.append(overall["top5"])
        mimic_top1.append(mimic_metrics["top1"])
        mimic_top3.append(mimic_metrics["top3"])
        mimic_top5.append(mimic_metrics["top5"])

    return {
        "num_runs": len(runs),
        "runs": runs,
        "aggregate": {
            "overall": {
                "top1": {"mean": _mean(overall_top1), "std": _std(overall_top1)},
                "top3": {"mean": _mean(overall_top3), "std": _std(overall_top3)},
                "top5": {"mean": _mean(overall_top5), "std": _std(overall_top5)},
            },
            "mimic_test": {
                "top1": {"mean": _mean(mimic_top1), "std": _std(mimic_top1)},
                "top3": {"mean": _mean(mimic_top3), "std": _std(mimic_top3)},
                "top5": {"mean": _mean(mimic_top5), "std": _std(mimic_top5)},
            },
            "checkpoint_epochs": [
                run["checkpoint_epoch"] for run in runs if run["checkpoint_epoch"] is not None
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 g4b idf 3-seed 实验结果。")
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        help="实验输出目录；默认使用 g4b_weighting_idf_seed42/123/3407 三个目录。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="聚合 JSON 输出路径。",
    )
    args = parser.parse_args()

    run_dirs = [Path(path).resolve() for path in (args.run_dirs or [str(path) for path in DEFAULT_RUN_DIRS])]
    try:
        result = aggregate(run_dirs)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
