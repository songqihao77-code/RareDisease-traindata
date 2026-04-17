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

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "train_finetune.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_PRETRAIN_CHECKPOINT = PROJECT_ROOT / "outputs" / "stage1_pretrain" / "checkpoints" / "best.pt"
DEFAULT_SEEDS = [42, 43, 44]
FOCUS_DATASETS = {"mimic_test", "RAMEDIS"}


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    case_weighting_enabled: bool
    residual_uniform: float


EXPERIMENTS = [
    ExperimentSpec(
        name="baseline_cw0_ru020",
        case_weighting_enabled=False,
        residual_uniform=0.2,
    ),
    ExperimentSpec(
        name="fixed_ic_cw1_ru020",
        case_weighting_enabled=True,
        residual_uniform=0.2,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 baseline/fixed-IC 的多随机种子实验，并产出病例级改进分析。"
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
    return (PROJECT_ROOT / "outputs" / "case_weighting_seed_compare" / timestamp).resolve()


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


def build_experiment_config(
    *,
    base_config: dict[str, Any],
    experiment: ExperimentSpec,
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


def find_output_by_suffix(renamed_outputs: dict[str, str], suffix: str) -> Path | None:
    for name, path_str in renamed_outputs.items():
        if name.endswith(suffix):
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


def load_disease_incidence(path: Path):
    try:
        return __import__("scipy.sparse").sparse.load_npz(path)
    except Exception:
        pass

    npz = np.load(path, allow_pickle=False)
    row_key = "rows" if "rows" in npz else "row"
    col_key = "cols" if "cols" in npz else "col"
    value_key = "vals" if "vals" in npz else "data"
    return __import__("scipy.sparse").sparse.csr_matrix(
        (npz[value_key], (npz[row_key], npz[col_key])),
        shape=tuple(npz["shape"]),
    )


def compute_ic_tables(train_config: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    paths_cfg = train_config["paths"]
    hpo_index_path = Path(paths_cfg["hpo_index_path"])
    disease_incidence_path = Path(paths_cfg["disease_incidence_path"])

    hpo_index_df = pd.read_excel(hpo_index_path, dtype={"hpo_id": str})
    hpo_index_df["hpo_idx"] = hpo_index_df["hpo_idx"].astype(int)
    idx_to_hpo = dict(zip(hpo_index_df["hpo_idx"], hpo_index_df["hpo_id"]))

    H_disease = load_disease_incidence(disease_incidence_path)
    num_hpo, num_disease = H_disease.shape
    eps = 1e-8

    H_coo = H_disease.tocoo()
    positive_mask = H_coo.data > 0
    df = np.bincount(H_coo.row[positive_mask], minlength=num_hpo).astype(np.float64, copy=False)
    support_mask = df > 0

    ic_raw = np.zeros(num_hpo, dtype=np.float64)
    ic_weight = np.ones(num_hpo, dtype=np.float64)

    if support_mask.any():
        p = (df[support_mask] + eps) / (float(num_disease) + eps)
        ic_raw_support = -np.log(p)
        clip_low = float(np.quantile(ic_raw_support, 0.05))
        clip_high = float(np.quantile(ic_raw_support, 0.95))
        clipped = np.clip(ic_raw_support, clip_low, clip_high)
        ic_raw[support_mask] = ic_raw_support

        clipped_min = float(clipped.min())
        clipped_max = float(clipped.max())
        if clipped_max - clipped_min <= eps:
            normalized = np.full_like(clipped, 0.5, dtype=np.float64)
        else:
            normalized = (clipped - clipped_min) / (clipped_max - clipped_min)
        ic_weight[support_mask] = 0.5 + normalized

    raw_map: dict[str, float] = {}
    weight_map: dict[str, float] = {}
    for idx, hpo_id in idx_to_hpo.items():
        raw_map[str(hpo_id)] = float(ic_raw[int(idx)])
        weight_map[str(hpo_id)] = float(ic_weight[int(idx)])
    return raw_map, weight_map


def read_case_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError(f"不支持的病例文件格式: {path}")


def build_case_hpo_table(
    data_config: dict[str, Any],
    ic_raw_map: dict[str, float],
    ic_weight_map: dict[str, float],
) -> pd.DataFrame:
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))

    rows: list[dict[str, Any]] = []
    for path_str in data_config["test_files"]:
        path = Path(path_str)
        df = read_case_table_file(path)
        df = df[[case_id_col, label_col, hpo_col]].copy()
        df[case_id_col] = path.stem + "_" + df[case_id_col].astype(str)

        for case_id, group_df in df.groupby(case_id_col, sort=False):
            hpo_ids = [str(v) for v in group_df[hpo_col].dropna().unique().tolist()]
            valid_hpo_ids = [hpo for hpo in hpo_ids if hpo in ic_weight_map]
            sorted_hpos = sorted(
                valid_hpo_ids,
                key=lambda hpo: (ic_weight_map.get(hpo, 1.0), ic_raw_map.get(hpo, 0.0), hpo),
                reverse=True,
            )
            top_hpo_payload = [
                {
                    "hpo_id": hpo,
                    "ic_weight": round(float(ic_weight_map[hpo]), 6),
                    "ic_raw": round(float(ic_raw_map[hpo]), 6),
                }
                for hpo in sorted_hpos[:10]
            ]
            rows.append(
                {
                    "case_id": str(case_id),
                    "dataset_name": path.stem,
                    "source_file": path.name,
                    "true_label": str(group_df[label_col].iloc[0]),
                    "hpo_ids": json.dumps(hpo_ids, ensure_ascii=False),
                    "valid_hpo_ids": json.dumps(valid_hpo_ids, ensure_ascii=False),
                    "top_ic_hpos": json.dumps(top_hpo_payload, ensure_ascii=False),
                    "num_hpo": int(len(hpo_ids)),
                    "num_valid_hpo": int(len(valid_hpo_ids)),
                    "max_ic_weight": float(max((ic_weight_map[h] for h in valid_hpo_ids), default=1.0)),
                    "mean_ic_weight": float(
                        np.mean([ic_weight_map[h] for h in valid_hpo_ids]) if valid_hpo_ids else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_details_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["case_id"] = df["case_id"].astype(str)
    df["dataset_name"] = df["dataset_name"].astype(str)
    df["true_label"] = df["true_label"].astype(str)
    df["pred_top1"] = df["pred_top1"].astype(str)
    df["true_rank"] = df["true_rank"].astype(int)
    return df


def update_run_summary_csv(run_root: Path, experiments: list[dict[str, Any]]) -> None:
    csv_path = run_root / "seed_experiment_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
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
                "details_csv",
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
                    "status": exp["status"],
                    "top1": metrics.get("top1"),
                    "top3": metrics.get("top3"),
                    "top5": metrics.get("top5"),
                    "save_dir": exp["save_dir"],
                    "config_path": exp["config_path"],
                    "summary_json": exp.get("summary_json"),
                    "details_csv": exp.get("details_csv"),
                }
            )


def build_case_level_comparisons(
    run_root: Path,
    experiments_state: list[dict[str, Any]],
    case_hpo_df: pd.DataFrame,
) -> dict[str, Path]:
    focus_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    completed_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for exp in experiments_state:
        if exp.get("status") == "completed":
            completed_lookup[(int(exp["seed"]), str(exp["experiment"]))] = exp

    for seed in sorted({int(exp["seed"]) for exp in experiments_state}):
        baseline_state = completed_lookup.get((seed, "baseline_cw0_ru020"))
        fixed_state = completed_lookup.get((seed, "fixed_ic_cw1_ru020"))
        if baseline_state is None or fixed_state is None:
            continue

        baseline_details = load_details_table(Path(baseline_state["details_csv"]))
        fixed_details = load_details_table(Path(fixed_state["details_csv"]))

        baseline_focus = baseline_details[baseline_details["dataset_name"].isin(FOCUS_DATASETS)].copy()
        fixed_focus = fixed_details[fixed_details["dataset_name"].isin(FOCUS_DATASETS)].copy()

        merged = baseline_focus.merge(
            fixed_focus,
            on=["case_id", "dataset_name", "source_file", "true_label"],
            how="inner",
            suffixes=("_baseline", "_fixed_ic"),
        )
        if merged.empty:
            continue

        merged["seed"] = seed
        merged["baseline_true_rank"] = merged["true_rank_baseline"].astype(int)
        merged["fixed_ic_true_rank"] = merged["true_rank_fixed_ic"].astype(int)
        merged["rank_delta"] = merged["baseline_true_rank"] - merged["fixed_ic_true_rank"]
        merged["wrong_to_correct"] = (
            (merged["baseline_true_rank"] > 1) & (merged["fixed_ic_true_rank"] == 1)
        )
        merged["correct_to_wrong"] = (
            (merged["baseline_true_rank"] == 1) & (merged["fixed_ic_true_rank"] > 1)
        )
        merged["improved"] = merged["rank_delta"] > 0
        merged["worsened"] = merged["rank_delta"] < 0

        merged = merged.merge(
            case_hpo_df,
            on=["case_id", "dataset_name", "source_file", "true_label"],
            how="left",
        )
        focus_rows.append(merged)

        metric_rows.append(
            {
                "seed": seed,
                "baseline_top1": baseline_state.get("metrics", {}).get("top1"),
                "fixed_ic_top1": fixed_state.get("metrics", {}).get("top1"),
                "baseline_top3": baseline_state.get("metrics", {}).get("top3"),
                "fixed_ic_top3": fixed_state.get("metrics", {}).get("top3"),
                "baseline_top5": baseline_state.get("metrics", {}).get("top5"),
                "fixed_ic_top5": fixed_state.get("metrics", {}).get("top5"),
                "focus_case_count": int(len(merged)),
                "improved_case_count": int(merged["improved"].sum()),
                "wrong_to_correct_count": int(merged["wrong_to_correct"].sum()),
                "worsened_case_count": int(merged["worsened"].sum()),
            }
        )

    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metric_df = pd.DataFrame(metric_rows)
    seed_metric_path = analysis_dir / "focus_dataset_seed_metrics.csv"
    metric_df.to_csv(seed_metric_path, index=False, encoding="utf-8-sig")

    per_seed_df = pd.concat(focus_rows, ignore_index=True) if focus_rows else pd.DataFrame()
    per_seed_path = analysis_dir / "focus_case_comparison_per_seed.csv"
    per_seed_df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")

    aggregate_path = analysis_dir / "focus_case_comparison_aggregate.csv"
    notable_path = analysis_dir / "focus_case_notable_improvements.csv"
    report_path = analysis_dir / "focus_case_report.md"

    if per_seed_df.empty:
        pd.DataFrame().to_csv(aggregate_path, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(notable_path, index=False, encoding="utf-8-sig")
        report_path.write_text(
            "# Case Weighting 多随机种子病例级分析\n\n未找到可分析的完整 baseline/fixed_ic 结果。\n",
            encoding="utf-8",
        )
        return {
            "seed_metric_path": seed_metric_path,
            "per_seed_path": per_seed_path,
            "aggregate_path": aggregate_path,
            "notable_path": notable_path,
            "report_path": report_path,
        }

    aggregate_rows: list[dict[str, Any]] = []
    grouped = per_seed_df.groupby(["dataset_name", "source_file", "case_id", "true_label"], sort=True)
    for (dataset_name, source_file, case_id, true_label), group_df in grouped:
        baseline_ranks = group_df["baseline_true_rank"].astype(int)
        fixed_ranks = group_df["fixed_ic_true_rank"].astype(int)
        rank_deltas = group_df["rank_delta"].astype(int)
        representative = group_df.iloc[0]

        aggregate_rows.append(
            {
                "dataset_name": dataset_name,
                "source_file": source_file,
                "case_id": case_id,
                "true_label": true_label,
                "seed_count": int(len(group_df)),
                "baseline_mean_rank": float(baseline_ranks.mean()),
                "fixed_ic_mean_rank": float(fixed_ranks.mean()),
                "mean_rank_delta": float(rank_deltas.mean()),
                "median_rank_delta": float(rank_deltas.median()),
                "best_rank_delta": int(rank_deltas.max()),
                "worst_rank_delta": int(rank_deltas.min()),
                "improved_seed_count": int(group_df["improved"].sum()),
                "wrong_to_correct_seed_count": int(group_df["wrong_to_correct"].sum()),
                "worsened_seed_count": int(group_df["worsened"].sum()),
                "baseline_top1_seed_count": int((baseline_ranks == 1).sum()),
                "fixed_ic_top1_seed_count": int((fixed_ranks == 1).sum()),
                "num_hpo": representative.get("num_hpo"),
                "num_valid_hpo": representative.get("num_valid_hpo"),
                "max_ic_weight": representative.get("max_ic_weight"),
                "mean_ic_weight": representative.get("mean_ic_weight"),
                "top_ic_hpos": representative.get("top_ic_hpos"),
                "baseline_pred_top1_by_seed": json.dumps(
                    dict(zip(group_df["seed"].astype(str), group_df["pred_top1_baseline"].astype(str), strict=True)),
                    ensure_ascii=False,
                ),
                "fixed_ic_pred_top1_by_seed": json.dumps(
                    dict(zip(group_df["seed"].astype(str), group_df["pred_top1_fixed_ic"].astype(str), strict=True)),
                    ensure_ascii=False,
                ),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["dataset_name", "wrong_to_correct_seed_count", "mean_rank_delta", "improved_seed_count", "case_id"],
        ascending=[True, False, False, False, True],
    )
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    notable_df = aggregate_df[
        (aggregate_df["wrong_to_correct_seed_count"] > 0)
        | (aggregate_df["mean_rank_delta"] >= 5.0)
        | (aggregate_df["improved_seed_count"] >= 2)
    ].copy()
    notable_df.to_csv(notable_path, index=False, encoding="utf-8-sig")

    report_lines = [
        "# Case Weighting 多随机种子病例级分析",
        "",
        "比较对象：`baseline_cw0_ru020` vs `fixed_ic_cw1_ru020`",
        f"随机种子：{', '.join(str(int(v)) for v in sorted(per_seed_df['seed'].unique().tolist()))}",
        "",
        "## 按种子统计",
        "",
    ]
    if not metric_df.empty:
        for row in metric_df.sort_values("seed").to_dict(orient="records"):
            report_lines.append(
                f"- seed={row['seed']}: "
                f"baseline_top1={row['baseline_top1']}, fixed_ic_top1={row['fixed_ic_top1']}, "
                f"improved_cases={row['improved_case_count']}, "
                f"wrong_to_correct={row['wrong_to_correct_count']}, "
                f"worsened={row['worsened_case_count']}"
            )
    else:
        report_lines.append("- 无完整种子对照结果。")

    for dataset_name in ["mimic_test", "RAMEDIS"]:
        report_lines.extend(["", f"## {dataset_name}", ""])
        dataset_df = notable_df[notable_df["dataset_name"] == dataset_name].head(20)
        if dataset_df.empty:
            report_lines.append("- 未找到满足筛选条件的病例。")
            continue
        for row in dataset_df.to_dict(orient="records"):
            report_lines.append(
                f"- {row['case_id']}: "
                f"baseline_mean_rank={row['baseline_mean_rank']:.2f}, "
                f"fixed_ic_mean_rank={row['fixed_ic_mean_rank']:.2f}, "
                f"mean_rank_delta={row['mean_rank_delta']:.2f}, "
                f"wrong_to_correct_seed_count={row['wrong_to_correct_seed_count']}, "
                f"improved_seed_count={row['improved_seed_count']}, "
                f"max_ic_weight={row['max_ic_weight']}, "
                f"top_ic_hpos={row['top_ic_hpos']}"
            )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "seed_metric_path": seed_metric_path,
        "per_seed_path": per_seed_path,
        "aggregate_path": aggregate_path,
        "notable_path": notable_path,
        "report_path": report_path,
    }


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
    data_config = load_yaml(data_config_path)
    ic_raw_map, ic_weight_map = compute_ic_tables(base_config)
    case_hpo_df = build_case_hpo_table(data_config, ic_raw_map=ic_raw_map, ic_weight_map=ic_weight_map)

    generated_config_dir = run_root / "generated_configs"
    manifest_path = run_root / "seed_compare_manifest.json"
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
        "dry_run": bool(args.dry_run),
        "experiments": experiments_state,
    }
    dump_json(manifest_path, manifest)

    print(f"[INFO] 总输出目录: {run_root}")
    print("[INFO] 对照实验: baseline_cw0_ru020 vs fixed_ic_cw1_ru020")
    print(f"[INFO] 随机种子: {', '.join(str(int(seed)) for seed in args.seeds)}")

    for seed in args.seeds:
        seed = int(seed)
        for experiment in EXPERIMENTS:
            exp_name = f"{experiment.name}_seed{seed:03d}"
            exp_root = run_root / exp_name
            config_path = generated_config_dir / f"train_finetune_{exp_name}.yaml"
            checkpoint_path = exp_root / "checkpoints" / "best.pt"
            train_log = exp_root / "logs" / f"{exp_name}__train.log"
            eval_log = exp_root / "logs" / f"{exp_name}__evaluate.log"

            config = build_experiment_config(
                base_config=base_config,
                experiment=experiment,
                seed=seed,
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
                "seed": seed,
                "experiment": experiment.name,
                "experiment_run_name": exp_name,
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
            update_run_summary_csv(run_root, experiments_state)

            print("=" * 80)
            print(
                f"[INFO] {exp_name}: "
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
                update_run_summary_csv(run_root, experiments_state)
                print(f"[ERROR] {exp_name} 训练失败，返回码={train_code}")
                if args.stop_on_error:
                    raise SystemExit(train_code)
                continue

            if not checkpoint_path.is_file():
                exp_state["status"] = "missing_best_checkpoint"
                dump_json(manifest_path, manifest)
                update_run_summary_csv(run_root, experiments_state)
                print(f"[ERROR] {exp_name} 未找到 best checkpoint: {checkpoint_path}")
                if args.stop_on_error:
                    raise SystemExit(1)
                continue

            eval_code = run_and_tee(commands["evaluate"], cwd=PROJECT_ROOT, log_path=eval_log)
            exp_state["evaluate_return_code"] = eval_code
            if eval_code != 0:
                exp_state["status"] = "evaluate_failed"
                dump_json(manifest_path, manifest)
                update_run_summary_csv(run_root, experiments_state)
                print(f"[ERROR] {exp_name} 评估失败，返回码={eval_code}")
                if args.stop_on_error:
                    raise SystemExit(eval_code)
                continue

            renamed_outputs = rename_evaluation_outputs(exp_root / "evaluation", exp_name)
            exp_state["renamed_evaluation_outputs"] = renamed_outputs

            summary_path = find_output_by_suffix(renamed_outputs, "_summary.json")
            details_path = find_output_by_suffix(renamed_outputs, "_details.csv")
            per_dataset_path = find_output_by_suffix(renamed_outputs, "_per_dataset.csv")

            if summary_path is not None and summary_path.is_file():
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                exp_state["summary_json"] = str(summary_path)
                exp_state["metrics"] = {
                    "top1": summary.get("top1"),
                    "top3": summary.get("top3"),
                    "top5": summary.get("top5"),
                    "num_cases": summary.get("num_cases"),
                    "num_evaluable": summary.get("num_evaluable"),
                    "checkpoint_path": summary.get("checkpoint_path"),
                }
            if details_path is not None:
                exp_state["details_csv"] = str(details_path)
            if per_dataset_path is not None:
                exp_state["per_dataset_csv"] = str(per_dataset_path)

            exp_state["status"] = "completed"
            dump_json(manifest_path, manifest)
            update_run_summary_csv(run_root, experiments_state)

    analysis_outputs = build_case_level_comparisons(
        run_root=run_root,
        experiments_state=experiments_state,
        case_hpo_df=case_hpo_df,
    )
    manifest["analysis_outputs"] = {k: str(v) for k, v in analysis_outputs.items()}
    dump_json(manifest_path, manifest)

    print("=" * 80)
    print("[DONE] 多随机种子对照实验与病例级分析已完成。")
    print(f"[DONE] 总目录: {run_root}")
    print(f"[DONE] 逐实验汇总: {run_root / 'seed_experiment_results.csv'}")
    print(f"[DONE] 病例级逐种子对比: {analysis_outputs['per_seed_path']}")
    print(f"[DONE] 病例级聚合对比: {analysis_outputs['aggregate_path']}")
    print(f"[DONE] 重点病例筛选: {analysis_outputs['notable_path']}")
    print(f"[DONE] Markdown 报告: {analysis_outputs['report_path']}")


if __name__ == "__main__":
    main()
