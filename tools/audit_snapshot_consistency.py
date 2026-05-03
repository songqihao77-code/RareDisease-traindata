from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - runtime audit fallback
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "mainline_full_pipeline_hybrid_tag_v5"
DEFAULT_TOP_CONFIG = PROJECT_ROOT / "configs" / "mainline_full_pipeline.yaml"
DEFAULT_LOCKED_CONFIG = PROJECT_ROOT / "configs" / "mainline_full_pipeline_hybrid_tag_v5.locked.yaml"
DEFAULT_RUNNER = PROJECT_ROOT / "tools" / "run_full_mainline_pipeline.py"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "snapshot_consistency_audit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "snapshot_consistency_audit"
HASH_LARGE_FILE_LIMIT = 64 * 1024 * 1024
CSV_FIELD_LIMIT = 1024 * 1024
UNSUPPORTED_PIPELINE_CONFIG_KEYS = {
    "resume.skip_pretrain": "Use pipeline.run_pretrain or --mode instead.",
    "resume.skip_finetune": "Use pipeline.run_finetune or --mode instead.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit consistency of a RareDisease HGNN output snapshot without mutating it."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--top-config", type=Path, default=DEFAULT_TOP_CONFIG)
    parser.add_argument("--locked-config", type=Path, default=None)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--out-report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true", help="Exit 2 on High/Critical failures.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def rel_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_target_dir(base: Path, stamp: str) -> Path:
    base = resolve_path(base)
    if base.exists():
        target = base / f"run_{stamp}"
    else:
        target = base
    target.mkdir(parents=True, exist_ok=False)
    return target


def to_text(value: Any) -> str:
    if value is None:
        return "NOT_FOUND"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def run_cmd(command: list[str]) -> tuple[str, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return "NOT_FOUND", "COMMAND_NOT_FOUND"
    except Exception as exc:
        return f"COMMAND_FAILED: {exc}", "COMMAND_FAILED"
    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    if proc.returncode != 0:
        return output or f"COMMAND_FAILED returncode={proc.returncode}", "COMMAND_FAILED"
    return output, "OK"


def sha256_file(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "NOT_FOUND", "NOT_FOUND"
    size = path.stat().st_size
    if size > HASH_LARGE_FILE_LIMIT:
        return "SKIPPED_HASH_LARGE_FILE", "SKIPPED_HASH_LARGE_FILE"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "OK"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_text(row.get(key, "")) for key in fieldnames})


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    csv.field_size_limit(CSV_FIELD_LIMIT)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(dict(row))
    return rows


def read_csv_header(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def flatten_config(payload: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_config(value, child))
    elif isinstance(payload, list):
        if not payload:
            flat[prefix] = []
        else:
            for index, value in enumerate(payload):
                child = f"{prefix}[{index}]"
                flat.update(flatten_config(value, child))
    else:
        flat[prefix] = payload
    return flat


def get_path(payload: Any, dotted: str) -> Any:
    cur = payload
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def has_path(payload: Any, dotted: str) -> bool:
    cur = payload
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False
    return True


def norm_path_text(value: Any) -> str:
    if value in (None, "", "NOT_FOUND"):
        return "NOT_FOUND"
    try:
        return str(resolve_path(str(value)))
    except Exception:
        return str(value)


def same_path(a: Any, b: Any) -> bool:
    if a in (None, "", "NOT_FOUND") or b in (None, "", "NOT_FOUND"):
        return False
    try:
        return resolve_path(str(a)) == resolve_path(str(b))
    except Exception:
        return str(a) == str(b)


def is_inside(path: Any, parent: Path) -> bool:
    if path in (None, "", "NOT_FOUND", "mixed"):
        return False
    try:
        Path(path).resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def path_exists_text(path: Any) -> str:
    if path in (None, "", "NOT_FOUND", "mixed"):
        return "false"
    try:
        return bool_text(Path(path).is_file() or Path(path).is_dir())
    except Exception:
        return "false"


def build_repository_state(
    run_dir: Path,
    top_config: Path,
    runner: Path,
    audit_time: str,
    locked_config: Path | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    git_status, git_status_status = run_cmd(["git", "status", "--short"])
    git_head, git_head_status = run_cmd(["git", "rev-parse", "HEAD"])
    rows = [
        {"item": "cwd", "value": str(PROJECT_ROOT), "status": "OK", "notes": ""},
        {"item": "audit_time", "value": audit_time, "status": "OK", "notes": ""},
        {"item": "python_version", "value": sys.version.replace("\n", " "), "status": "OK", "notes": platform.platform()},
        {"item": "git_status_short", "value": git_status, "status": git_status_status, "notes": ""},
        {"item": "git_rev_parse_HEAD", "value": git_head, "status": git_head_status, "notes": ""},
        {"item": "run_dir", "value": str(run_dir), "status": "OK" if run_dir.is_dir() else "NOT_FOUND", "notes": ""},
        {"item": "top_config", "value": str(top_config), "status": "OK" if top_config.is_file() else "NOT_FOUND", "notes": ""},
        {
            "item": "locked_config",
            "value": str(locked_config) if locked_config else "NOT_PROVIDED",
            "status": "OK" if locked_config and locked_config.is_file() else ("NOT_APPLICABLE" if locked_config is None else "NOT_FOUND"),
            "notes": "",
        },
        {"item": "runner", "value": str(runner), "status": "OK" if runner.is_file() else "NOT_FOUND", "notes": ""},
    ]
    return rows, git_head if git_head_status == "OK" else "NOT_FOUND", git_status if git_status_status == "OK" else git_status


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def build_snapshot_inventory(run_dir: Path) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path, str]] = [
        ("manifest", run_dir / "run_manifest.json", "run manifest"),
        ("config", run_dir / "configs" / "stage1_pretrain.yaml", "stage1 pretrain config"),
        ("config", run_dir / "configs" / "stage2_finetune.yaml", "stage2 finetune config"),
        ("config", run_dir / "configs" / "stage3_exact_eval_train.yaml", "stage3 eval train config"),
        ("checkpoint", run_dir / "stage1_pretrain" / "checkpoints" / "best.pt", "stage1 best checkpoint"),
        ("checkpoint", run_dir / "stage2_finetune" / "checkpoints" / "best.pt", "stage2 best checkpoint"),
        ("result", run_dir / "stage3_exact_eval" / "exact_details.csv", "exact eval details"),
        ("result", run_dir / "stage3_exact_eval" / "exact_per_dataset.csv", "exact eval per dataset"),
        ("result", run_dir / "stage3_exact_eval" / "exact_summary.json", "exact eval summary"),
        ("candidate", run_dir / "stage4_candidates" / "top50_candidates_validation.csv", "validation top50 candidates"),
        ("candidate", run_dir / "stage4_candidates" / "top50_candidates_test.csv", "test top50 candidates"),
        ("candidate_metadata", run_dir / "stage4_candidates" / "top50_candidates_validation.metadata.json", "validation candidate metadata"),
        ("candidate_metadata", run_dir / "stage4_candidates" / "top50_candidates_test.metadata.json", "test candidate metadata"),
        ("result", run_dir / "stage5_ddd_rerank" / "rerank_fixed_test_metrics.csv", "DDD rerank fixed test metrics"),
        ("result", run_dir / "stage5_ddd_rerank" / "ddd_val_selected_grid_weights.json", "DDD selected weights"),
        ("result", run_dir / "stage6_mimic_similar_case" / "similar_case_fixed_test.csv", "mimic SimilarCase metrics"),
        ("result", run_dir / "stage6_mimic_similar_case" / "similar_case_fixed_test_ranked_candidates.csv", "mimic SimilarCase ranked candidates"),
        ("result", run_dir / "stage6_mimic_similar_case" / "manifest.json", "mimic SimilarCase manifest"),
        ("result", run_dir / "mainline_final_metrics.csv", "final metrics"),
        ("result", run_dir / "mainline_final_metrics_with_sources.csv", "final metrics with sources"),
        ("result", run_dir / "mainline_final_case_ranks.csv", "final case ranks"),
    ]

    for pattern, file_type in [
        ("stage3_exact_eval/*.json", "result"),
        ("stage3_exact_eval/evaluation/*.json", "result"),
        ("stage4_candidates/*.json", "candidate_metadata"),
        ("stage5_ddd_rerank/*.json", "result"),
        ("stage6_mimic_similar_case/*.json", "result"),
    ]:
        for path in run_dir.glob(pattern):
            paths.append((file_type, path, f"glob:{pattern}"))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for file_type, path, notes in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        exists = path.is_file() or path.is_dir()
        sha = "NOT_FOUND"
        hash_note = ""
        size = ""
        mtime = ""
        if exists and path.is_file():
            stat = path.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            sha, hash_note = sha256_file(path)
        elif exists:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            hash_note = "DIRECTORY"
        out.append(
            {
                "type": file_type,
                "path": str(path.resolve()) if exists else str(path),
                "exists": bool_text(exists),
                "size_bytes": size,
                "sha256": sha,
                "mtime": mtime,
                "notes": "; ".join(part for part in (notes, hash_note) if part and part != "OK"),
            }
        )
    return out


def command_arg(commands: list[dict[str, Any]], step: str, option: str) -> str:
    for command in commands:
        if command.get("step") != step:
            continue
        argv = command.get("command")
        if not isinstance(argv, list):
            continue
        for idx, token in enumerate(argv):
            if token == option and idx + 1 < len(argv):
                return str(argv[idx + 1])
    return "NOT_FOUND"


def add_check(
    rows: list[dict[str, Any]],
    check_id: str,
    check_name: str,
    expected: Any,
    actual: Any,
    status: str,
    severity: str,
    evidence_path: Any,
    notes: str = "",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "check_name": check_name,
            "expected": expected,
            "actual": actual,
            "status": status,
            "severity": severity,
            "evidence_path": evidence_path,
            "notes": notes,
        }
    )


def build_manifest_checks(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_path = run_dir / "run_manifest.json"
    expected_stage2_ckpt = run_dir / "stage2_finetune" / "checkpoints" / "best.pt"
    if not manifest:
        add_check(rows, "M001", "run_manifest.json exists and parses", str(manifest_path), "NOT_FOUND", "NOT_FOUND", "High", manifest_path)
        return rows

    manifest_output_dir = manifest.get("output_dir")
    add_check(
        rows,
        "M001",
        "manifest output_dir equals audited run-dir",
        str(run_dir.resolve()),
        manifest_output_dir or "NOT_FOUND",
        "PASS" if same_path(manifest_output_dir, run_dir) else "FAIL",
        "High",
        manifest_path,
    )

    finetune_checkpoint = manifest.get("finetune_checkpoint")
    add_check(
        rows,
        "M002",
        "manifest finetune checkpoint exists",
        str(expected_stage2_ckpt.resolve()),
        finetune_checkpoint or "NOT_FOUND",
        "PASS" if finetune_checkpoint and Path(finetune_checkpoint).is_file() else "FAIL",
        "Critical",
        manifest_path,
    )
    add_check(
        rows,
        "M003",
        "manifest finetune checkpoint is run-dir stage2 best.pt",
        str(expected_stage2_ckpt.resolve()),
        finetune_checkpoint or "NOT_FOUND",
        "PASS" if same_path(finetune_checkpoint, expected_stage2_ckpt) else "FAIL",
        "High",
        manifest_path,
    )

    stage_configs = manifest.get("stage_configs") if isinstance(manifest.get("stage_configs"), dict) else {}
    for key, expected_path in {
        "pretrain": run_dir / "configs" / "stage1_pretrain.yaml",
        "finetune": run_dir / "configs" / "stage2_finetune.yaml",
        "eval_train": run_dir / "configs" / "stage3_exact_eval_train.yaml",
    }.items():
        actual = stage_configs.get(key, "NOT_FOUND")
        add_check(
            rows,
            f"M10{len(rows)}",
            f"manifest stage_configs.{key} points inside audited run",
            str(expected_path.resolve()),
            actual,
            "PASS" if same_path(actual, expected_path) and Path(actual).is_file() else "FAIL",
            "High" if key == "finetune" else "Medium",
            manifest_path,
        )

    commands = manifest.get("commands") if isinstance(manifest.get("commands"), list) else []
    stage3_checkpoint = command_arg(commands, "stage3_exact_eval", "--checkpoint_path")
    add_check(
        rows,
        "M020",
        "stage3 exact eval command uses stage2 best checkpoint",
        str(expected_stage2_ckpt.resolve()),
        stage3_checkpoint,
        "PASS" if same_path(stage3_checkpoint, expected_stage2_ckpt) else "FAIL",
        "Critical",
        manifest_path,
    )

    stage3_data_config = command_arg(commands, "stage3_exact_eval", "--data_config_path")
    metadata_data_config = (manifest.get("test_candidates_metadata") or {}).get("data_config_path")
    add_check(
        rows,
        "M021",
        "stage3 eval and candidate metadata use same data config",
        stage3_data_config,
        metadata_data_config or "NOT_FOUND",
        "PASS" if same_path(stage3_data_config, metadata_data_config) else "FAIL",
        "High",
        manifest_path,
    )

    for source_name, metadata_key in [("validation", "validation_candidates_metadata"), ("test", "test_candidates_metadata")]:
        meta = manifest.get(metadata_key) if isinstance(manifest.get(metadata_key), dict) else {}
        checkpoint = meta.get("checkpoint_path")
        output_path = meta.get("output_path")
        add_check(
            rows,
            f"M03{source_name}",
            f"{source_name} candidate metadata checkpoint matches stage2 best",
            str(expected_stage2_ckpt.resolve()),
            checkpoint or "NOT_FOUND",
            "PASS" if same_path(checkpoint, expected_stage2_ckpt) else "FAIL",
            "Critical",
            manifest_path,
            f"metadata key={metadata_key}",
        )
        add_check(
            rows,
            f"M04{source_name}",
            f"{source_name} candidate metadata output file exists",
            str(run_dir / "stage4_candidates"),
            output_path or "NOT_FOUND",
            "PASS" if output_path and Path(output_path).is_file() and is_inside(output_path, run_dir) else "FAIL",
            "High",
            manifest_path,
        )

    final_outputs = manifest.get("final_outputs") if isinstance(manifest.get("final_outputs"), dict) else {}
    for key in ("metrics", "metrics_with_sources", "case_ranks"):
        actual = final_outputs.get(key, "NOT_FOUND")
        add_check(
            rows,
            f"M05{key}",
            f"manifest final_outputs.{key} exists inside run-dir",
            str(run_dir),
            actual,
            "PASS" if actual != "NOT_FOUND" and Path(actual).is_file() and is_inside(actual, run_dir) else "FAIL",
            "High",
            manifest_path,
        )

    final_sources_path = run_dir / "mainline_final_metrics_with_sources.csv"
    rows_csv = read_csv_rows(final_sources_path)
    if not rows_csv:
        add_check(rows, "M060", "final metrics with sources readable", str(final_sources_path), "NOT_FOUND", "NOT_FOUND", "High", final_sources_path)
    else:
        for dataset, expected_module in [
            ("DDD", "ddd_validation_selected_grid_rerank"),
            ("mimic_test_recleaned_mondo_hpo_rows", "similar_case_fixed_test"),
        ]:
            row = next((r for r in rows_csv if r.get("dataset") == dataset), None)
            if row is None:
                add_check(rows, f"M06{dataset}", f"{dataset} final source row exists", dataset, "NOT_FOUND", "NOT_FOUND", "High", final_sources_path)
                continue
            module = row.get("module_applied", "")
            source = row.get("source_result_path", "")
            status = "PASS" if module == expected_module and source and source != "mixed" and Path(source).exists() and is_inside(source, run_dir) else "FAIL"
            add_check(
                rows,
                f"M06{dataset}",
                f"{dataset} final source is expected postprocess inside run-dir",
                expected_module,
                {"module_applied": module, "source_result_path": source},
                status,
                "High",
                final_sources_path,
            )
    return rows


def compare_values(top_value: Any, snapshot_value: Any) -> str:
    if top_value is None or snapshot_value is None:
        return "NOT_FOUND"
    if isinstance(top_value, (str, Path)) and isinstance(snapshot_value, (str, Path)):
        if ("/" in str(top_value) or "\\" in str(top_value) or ":" in str(top_value)) and (
            "/" in str(snapshot_value) or "\\" in str(snapshot_value) or ":" in str(snapshot_value)
        ):
            return "PASS" if same_path(top_value, snapshot_value) else "FAIL"
    return "PASS" if top_value == snapshot_value else "FAIL"


def build_top_config_comparison(top_config: dict[str, Any], manifest: dict[str, Any], stage1: dict[str, Any], stage2: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(key_path: str, top_value: Any, snapshot_value: Any, severity: str, notes: str = "") -> None:
        status = compare_values(top_value, snapshot_value)
        rows.append(
            {
                "key_path": key_path,
                "top_config_value": "NOT_FOUND" if top_value is None else top_value,
                "snapshot_value": "NOT_FOUND" if snapshot_value is None else snapshot_value,
                "status": status,
                "severity": severity if status == "FAIL" else ("Info" if status == "PASS" else severity),
                "notes": notes,
            }
        )

    add("paths.output_dir", norm_path_text(get_path(top_config, "paths.output_dir")), norm_path_text(manifest.get("output_dir") or run_dir), "High")
    add("tag_encoder.enabled", get_path(top_config, "tag_encoder.enabled"), get_path(manifest, "tag_encoder.enabled"), "High")
    add("paths.data_eval_config", norm_path_text(get_path(top_config, "paths.data_eval_config")), norm_path_text(get_path(manifest, "test_candidates_metadata.data_config_path")), "Medium")
    add("resume.finetune_checkpoint", norm_path_text(get_path(top_config, "resume.finetune_checkpoint")), norm_path_text(manifest.get("finetune_checkpoint")), "High")
    add("resume.skip_pretrain", get_path(top_config, "resume.skip_pretrain"), None, "High", "snapshot manifest has no equivalent field; runner audit checks whether this is ignored")
    add("resume.skip_finetune", get_path(top_config, "resume.skip_finetune"), None, "High", "snapshot manifest has no equivalent field; runner audit checks whether this is ignored")

    commands = manifest.get("commands") if isinstance(manifest.get("commands"), list) else []
    command_steps = {str(cmd.get("step")) for cmd in commands if isinstance(cmd, dict)}
    pipeline_map = {
        "pipeline.run_pretrain": "stage1_pretrain",
        "pipeline.run_finetune": "stage2_finetune",
        "pipeline.run_exact_eval": "stage3_exact_eval",
        "pipeline.run_candidate_export": "stage4_candidates_test",
        "pipeline.run_ddd_rerank": "stage5_ddd_rerank",
        "pipeline.run_mimic_similar_case": "stage6_mimic_similar_case",
        "pipeline.run_final_aggregation": None,
    }
    for key_path, step in pipeline_map.items():
        top_value = get_path(top_config, key_path)
        snapshot_value = True if step is None else step in command_steps
        add(key_path, top_value, snapshot_value, "Medium", "snapshot value inferred from run_manifest commands")

    add("data.random_seed", None, get_path(stage2, "data.random_seed"), "Low", "top-level config does not expose training seed")
    add("model.hidden_dim", None, get_path(stage2, "model.hidden_dim"), "Low", "top-level config does not expose model hidden_dim")
    add("model.encoder.use_tag_encoder", get_path(top_config, "tag_encoder.enabled"), get_path(stage2, "model.encoder.use_tag_encoder"), "High", "top-level tag flag compared to effective stage2 model encoder")
    add("model.case_refiner.enabled", None, get_path(stage2, "model.case_refiner.enabled"), "Low", "top-level config does not expose case_refiner")
    add("dual_stream.enabled", get_path(top_config, "dual_stream.enabled"), get_path(stage2, "dual_stream.enabled"), "Low")
    add("stage2.train.init_checkpoint_path", norm_path_text(get_path(top_config, "resume.finetune_checkpoint")), norm_path_text(get_path(stage2, "train.init_checkpoint_path")), "Medium", "top resume checkpoint is not the same concept as stage2 pretrain init checkpoint")
    add("paths.pretrain_config", norm_path_text(get_path(top_config, "paths.pretrain_config")), norm_path_text(get_path(manifest, "stage_configs.pretrain")), "Low", "top config points to source template, snapshot points to generated effective config")
    add("paths.finetune_config", norm_path_text(get_path(top_config, "paths.finetune_config")), norm_path_text(get_path(manifest, "stage_configs.finetune")), "Low", "top config points to source template, snapshot points to generated effective config")
    return rows


def config_file_hash(value: Any) -> str:
    if value in (None, "", "NOT_FOUND"):
        return "NOT_FOUND"
    try:
        path = resolve_path(str(value))
    except Exception:
        return "NOT_FOUND"
    digest, _ = sha256_file(path)
    return digest


def build_locked_config_comparison(
    locked_config: dict[str, Any],
    locked_config_path: Path | None,
    manifest: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if locked_config_path is None:
        return [
            {
                "key_path": "locked_config",
                "locked_config_value": "NOT_PROVIDED",
                "snapshot_value": str(run_dir),
                "status": "NOT_APPLICABLE",
                "severity": "Info",
                "notes": "pass --locked-config to verify a frozen reproduction config",
            }
        ]
    if not locked_config:
        return [
            {
                "key_path": "locked_config",
                "locked_config_value": str(locked_config_path),
                "snapshot_value": str(run_dir),
                "status": "NOT_FOUND",
                "severity": "High",
                "notes": "locked config missing or unreadable",
            }
        ]

    def add(key_path: str, locked_value: Any, snapshot_value: Any, severity: str, notes: str = "") -> None:
        status = compare_values(locked_value, snapshot_value)
        rows.append(
            {
                "key_path": key_path,
                "locked_config_value": "NOT_FOUND" if locked_value is None else locked_value,
                "snapshot_value": "NOT_FOUND" if snapshot_value is None else snapshot_value,
                "status": status,
                "severity": severity if status == "FAIL" else ("Info" if status == "PASS" else severity),
                "notes": notes,
            }
        )

    add("paths.output_dir", norm_path_text(get_path(locked_config, "paths.output_dir")), norm_path_text(manifest.get("output_dir") or run_dir), "High")
    add("tag_encoder.enabled", get_path(locked_config, "tag_encoder.enabled"), get_path(manifest, "tag_encoder.enabled"), "High")
    add("model.encoder.use_tag_encoder", get_path(locked_config, "model.encoder.use_tag_encoder"), get_path(stage2, "model.encoder.use_tag_encoder"), "High")
    add("resume.finetune_checkpoint", norm_path_text(get_path(locked_config, "resume.finetune_checkpoint")), norm_path_text(manifest.get("finetune_checkpoint")), "High")
    add("resume.pretrain_checkpoint", norm_path_text(get_path(locked_config, "resume.pretrain_checkpoint")), norm_path_text(get_path(stage2, "train.init_checkpoint_path")), "Medium")
    add("paths.data_eval_config", norm_path_text(get_path(locked_config, "paths.data_eval_config")), norm_path_text(get_path(manifest, "test_candidates_metadata.data_config_path")), "Medium")
    add("paths.pretrain_config", norm_path_text(get_path(locked_config, "paths.pretrain_config")), norm_path_text(get_path(manifest, "stage_configs.pretrain")), "Medium")
    add("paths.finetune_config", norm_path_text(get_path(locked_config, "paths.finetune_config")), norm_path_text(get_path(manifest, "stage_configs.finetune")), "Medium")
    add("paths.pretrain_config.sha256", config_file_hash(get_path(locked_config, "paths.pretrain_config")), config_file_hash(get_path(manifest, "stage_configs.pretrain")), "Medium", "hash comparison of effective stage1 config")
    add("paths.finetune_config.sha256", config_file_hash(get_path(locked_config, "paths.finetune_config")), config_file_hash(get_path(manifest, "stage_configs.finetune")), "Medium", "hash comparison of effective stage2 config")
    add("tag_encoder.pretrained_embed_path", norm_path_text(get_path(locked_config, "tag_encoder.pretrained_embed_path")), norm_path_text(get_path(manifest, "tag_encoder.pretrained_embed_path")), "Medium")

    for key_path in UNSUPPORTED_PIPELINE_CONFIG_KEYS:
        present = has_path(locked_config, key_path)
        rows.append(
            {
                "key_path": key_path,
                "locked_config_value": get_path(locked_config, key_path) if present else "ABSENT",
                "snapshot_value": "ABSENT_EXPECTED",
                "status": "FAIL" if present else "PASS",
                "severity": "High" if present else "Info",
                "notes": "locked config must not contain runner-validated unsupported control-flow keys",
            }
        )
    return rows


def build_runner_config_validation(
    config_paths: list[tuple[str, Path | None]],
    runner: Path,
) -> list[dict[str, Any]]:
    runner_text = runner.read_text(encoding="utf-8", errors="ignore") if runner.is_file() else ""
    runner_has_function = "def validate_pipeline_config_keys" in runner_text
    runner_has_strict_arg = "--strict-config-keys" in runner_text
    rows: list[dict[str, Any]] = [
        {
            "config_path": str(runner),
            "key_path": "runner.validate_pipeline_config_keys",
            "value": bool_text(runner_has_function),
            "status": "PASS" if runner_has_function else "FAIL",
            "strict_would_fail": "false",
            "severity": "Info" if runner_has_function else "High",
            "notes": "runner exposes config-key validation function",
        },
        {
            "config_path": str(runner),
            "key_path": "runner.--strict-config-keys",
            "value": bool_text(runner_has_strict_arg),
            "status": "PASS" if runner_has_strict_arg else "FAIL",
            "strict_would_fail": "false",
            "severity": "Info" if runner_has_strict_arg else "High",
            "notes": "runner exposes strict CLI guard",
        },
    ]
    for label, config_path in config_paths:
        if config_path is None:
            continue
        cfg = load_yaml(config_path)
        if not cfg:
            rows.append(
                {
                    "config_path": str(config_path),
                    "key_path": "config",
                    "value": "NOT_FOUND",
                    "status": "NOT_FOUND",
                    "strict_would_fail": "false",
                    "severity": "High",
                    "notes": f"{label} config missing or unreadable",
                }
            )
            continue
        is_locked = "locked" in label.lower()
        for key_path, guidance in UNSUPPORTED_PIPELINE_CONFIG_KEYS.items():
            present = has_path(cfg, key_path)
            status = "PASS"
            severity = "Info"
            notes = "key absent; strict validation passes for this key"
            if present:
                status = "FAIL" if is_locked else "EXPECTED_STRICT_FAIL"
                severity = "High"
                notes = f"unsupported control-flow key present; {guidance}"
            rows.append(
                {
                    "config_path": str(config_path),
                    "key_path": key_path,
                    "value": get_path(cfg, key_path) if present else "ABSENT",
                    "status": status,
                    "strict_would_fail": bool_text(present),
                    "severity": severity,
                    "notes": notes,
                }
            )
    return rows


def leaf_key_for_path(key_path: str) -> str:
    key = key_path.split(".")[-1]
    if "[" in key:
        key = key.split("[", 1)[0]
    return key


def build_suspicious_keys(top_config: dict[str, Any], runner: Path) -> list[dict[str, Any]]:
    runner_text = runner.read_text(encoding="utf-8", errors="ignore") if runner.is_file() else ""
    flat = flatten_config(top_config)
    rows: list[dict[str, Any]] = []
    control_words = ("run_", "skip", "enabled", "checkpoint", "output_dir", "config", "source", "mode", "fixed", "select", "objective", "tuning")
    for key_path, value in sorted(flat.items()):
        leaf = leaf_key_for_path(key_path)
        runner_mentions_leaf = bool(leaf and leaf in runner_text)
        runner_mentions_full = bool(key_path in runner_text)
        status = "CONSUMED_OR_MENTIONED" if runner_mentions_leaf or runner_mentions_full else "SUSPICIOUS"
        severity = "Low"
        notes = ""
        if key_path in {"resume.skip_pretrain", "resume.skip_finetune"}:
            if not runner_mentions_leaf and not runner_mentions_full:
                status = "CONFIRMED_IGNORED"
                severity = "High"
                notes = f"runner does not contain {leaf!r}; field can mislead rerun expectations"
            else:
                severity = "Medium"
        elif status == "SUSPICIOUS" and any(word in key_path for word in control_words):
            severity = "Medium"
            notes = "control-flow-looking key is not directly mentioned in runner source; verify manually"
        rows.append(
            {
                "key_path": key_path,
                "value": value,
                "runner_mentions_leaf_key": bool_text(runner_mentions_leaf),
                "runner_mentions_full_key": bool_text(runner_mentions_full),
                "status": status,
                "severity": severity,
                "notes": notes,
            }
        )
    return rows


def metadata_for_candidate(candidate: Path) -> Path | None:
    candidates = [
        candidate.with_suffix(".metadata.json"),
        candidate.parent / f"{candidate.stem}.metadata.json",
        candidate.parent / f"{candidate.stem}.meta.json",
    ]
    candidates.extend(sorted(candidate.parent.glob(f"{candidate.stem}*metadata*.json")))
    candidates.extend(sorted(candidate.parent.glob(f"{candidate.stem}*meta*.json")))
    for path in unique_paths(candidates):
        if path.is_file():
            return path
    return None


def audit_candidate_csv(candidate: Path, expected_top_k: int = 50) -> dict[str, Any]:
    if not candidate.is_file():
        return {
            "n_cases": 0,
            "n_rows": 0,
            "candidate_count_issue_cases": 0,
            "duplicate_candidate_cases": 0,
            "rank_issue_cases": 0,
            "has_score": False,
            "status": "NOT_FOUND",
            "notes": "candidate file missing",
        }
    header = read_csv_header(candidate)
    case_col = "case_id" if "case_id" in header else None
    candidate_col = "candidate_id" if "candidate_id" in header else ("candidate_mondo" if "candidate_mondo" in header else None)
    rank_col = "original_rank" if "original_rank" in header else ("rank" if "rank" in header else None)
    score_col = "hgnn_score" if "hgnn_score" in header else ("score" if "score" in header else None)
    if not case_col or not candidate_col:
        return {
            "n_cases": 0,
            "n_rows": 0,
            "candidate_count_issue_cases": 0,
            "duplicate_candidate_cases": 0,
            "rank_issue_cases": 0,
            "has_score": bool(score_col),
            "status": "FAIL",
            "notes": f"missing case/candidate columns; header={header}",
        }
    counts: Counter[str] = Counter()
    candidate_sets: dict[str, set[str]] = defaultdict(set)
    ranks: dict[str, list[int]] = defaultdict(list)
    n_rows = 0
    with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n_rows += 1
            case_id = str(row.get(case_col, ""))
            candidate_id = str(row.get(candidate_col, ""))
            counts[case_id] += 1
            candidate_sets[case_id].add(candidate_id)
            if rank_col:
                try:
                    ranks[case_id].append(int(float(row.get(rank_col, ""))))
                except Exception:
                    ranks[case_id].append(-1)
    issue_count = sum(1 for count in counts.values() if count != expected_top_k)
    duplicate_count = sum(1 for case_id, count in counts.items() if len(candidate_sets[case_id]) != count)
    rank_issue = 0
    if rank_col:
        expected_ranks = list(range(1, expected_top_k + 1))
        for case_id, case_ranks in ranks.items():
            if sorted(case_ranks) != expected_ranks:
                rank_issue += 1
    status = "PASS" if issue_count == 0 and duplicate_count == 0 and (not rank_col or rank_issue == 0) and score_col else "WARN"
    notes = []
    if not rank_col:
        notes.append("rank column NOT_FOUND")
    if not score_col:
        notes.append("score column NOT_FOUND")
    return {
        "n_cases": len(counts),
        "n_rows": n_rows,
        "candidate_count_issue_cases": issue_count,
        "duplicate_candidate_cases": duplicate_count,
        "rank_issue_cases": rank_issue,
        "has_score": bool(score_col),
        "status": status,
        "notes": "; ".join(notes),
    }


def build_candidate_consistency(run_dir: Path, expected_checkpoint: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage4 = run_dir / "stage4_candidates"
    candidates = sorted(stage4.glob("*candidate*.csv")) if stage4.is_dir() else []
    if not candidates:
        candidates = [stage4 / "top50_candidates_validation.csv", stage4 / "top50_candidates_test.csv"]
    for candidate in unique_paths(candidates):
        metadata = metadata_for_candidate(candidate)
        meta_payload = load_json(metadata) if metadata else {}
        checkpoint = (
            meta_payload.get("checkpoint_path")
            or meta_payload.get("checkpoint")
            or meta_payload.get("model_checkpoint")
            or "NOT_FOUND"
        )
        top_k = int(meta_payload.get("top_k") or 50)
        csv_audit = audit_candidate_csv(candidate, expected_top_k=top_k)
        checkpoint_match = bool(same_path(checkpoint, expected_checkpoint))
        status = csv_audit["status"]
        if checkpoint == "NOT_FOUND":
            status = "WARN"
        elif not checkpoint_match:
            status = "FAIL"
        rows.append(
            {
                "candidate_file": str(candidate.resolve()) if candidate.exists() else str(candidate),
                "metadata_file": str(metadata.resolve()) if metadata else "NOT_FOUND",
                "checkpoint_in_metadata": checkpoint,
                "expected_checkpoint": str(expected_checkpoint.resolve()),
                "checkpoint_match": bool_text(checkpoint_match),
                "n_cases": meta_payload.get("num_cases", csv_audit["n_cases"]),
                "n_rows": meta_payload.get("num_rows", csv_audit["n_rows"]),
                "candidate_count_issue_cases": csv_audit["candidate_count_issue_cases"],
                "duplicate_candidate_cases": csv_audit["duplicate_candidate_cases"],
                "status": status,
                "notes": f"rank_issue_cases={csv_audit['rank_issue_cases']}; has_score={csv_audit['has_score']}; {csv_audit['notes']}",
            }
        )
    return rows


def expected_source_type(dataset: str) -> str:
    if dataset == "DDD":
        return "ddd_validation_selected_grid_rerank"
    if "mimic" in dataset.lower():
        return "similar_case_fixed_test"
    if dataset == "ALL":
        return "mixed"
    return "hgnn_exact_baseline"


def count_case_rank_datasets(case_ranks_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not case_ranks_path.is_file():
        return counts
    header = read_csv_header(case_ranks_path)
    dataset_col = "dataset" if "dataset" in header else ("dataset_name" if "dataset_name" in header else None)
    if dataset_col is None:
        return counts
    with case_ranks_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts[str(row.get(dataset_col, ""))] += 1
    return counts


def build_final_traceability(run_dir: Path) -> list[dict[str, Any]]:
    final_sources_path = run_dir / "mainline_final_metrics_with_sources.csv"
    case_ranks_path = run_dir / "mainline_final_case_ranks.csv"
    case_rank_counts = count_case_rank_datasets(case_ranks_path)
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(final_sources_path):
        dataset = str(row.get("dataset", ""))
        source_path = row.get("source_result_path") or row.get("source_path") or "NOT_FOUND"
        metric_source = row.get("module_applied") or row.get("metric_source") or "NOT_FOUND"
        expected = expected_source_type(dataset)
        source_exists = source_path == "mixed" or (source_path != "NOT_FOUND" and Path(source_path).exists())
        inside = source_path != "mixed" and is_inside(source_path, run_dir)
        source_ok = source_exists and (inside or source_path == "mixed")
        module_ok = metric_source == expected or (dataset == "ALL" and metric_source == "mixed")
        case_rank_ok = dataset == "ALL" or case_rank_counts.get(dataset, 0) > 0
        status = "PASS" if source_ok and module_ok and case_rank_ok else "FAIL"
        rows.append(
            {
                "dataset": dataset,
                "metric_source": metric_source,
                "source_path": source_path,
                "source_exists": bool_text(source_exists),
                "inside_run_dir": bool_text(inside),
                "expected_source_type": expected,
                "status": status,
                "notes": f"case_rank_rows={case_rank_counts.get(dataset, 0)}; final_case_ranks={case_ranks_path}",
            }
        )
    if not rows:
        rows.append(
            {
                "dataset": "NOT_FOUND",
                "metric_source": "NOT_FOUND",
                "source_path": str(final_sources_path),
                "source_exists": "false",
                "inside_run_dir": "false",
                "expected_source_type": "NOT_FOUND",
                "status": "NOT_FOUND",
                "notes": "mainline_final_metrics_with_sources.csv missing or empty",
            }
        )
    return rows


def get_metric(rows: list[dict[str, str]], dataset: str, metric: str) -> str:
    for row in rows:
        if row.get("dataset") == dataset or row.get("dataset_name") == dataset:
            return row.get(metric) or row.get(metric.lower()) or ""
    return ""


def build_snapshot_comparison() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_dir in sorted((PROJECT_ROOT / "outputs").glob("mainline_full_pipeline*")):
        if not output_dir.is_dir():
            continue
        final_sources = output_dir / "mainline_final_metrics_with_sources.csv"
        final_metrics = final_sources if final_sources.is_file() else output_dir / "mainline_final_metrics.csv"
        metrics_rows = read_csv_rows(final_metrics) if final_metrics.is_file() else []
        mimic_row = next((r for r in metrics_rows if "mimic" in str(r.get("dataset", r.get("dataset_name", ""))).lower()), {})
        ddd_row = next((r for r in metrics_rows if str(r.get("dataset", r.get("dataset_name", ""))) == "DDD"), {})
        all_row = next((r for r in metrics_rows if str(r.get("dataset", r.get("dataset_name", ""))) == "ALL"), {})
        notes = []
        if not final_metrics.is_file():
            notes.append("final metrics NOT_FOUND")
        rows.append(
            {
                "output_dir": str(output_dir.resolve()),
                "has_manifest": bool_text((output_dir / "run_manifest.json").is_file()),
                "has_stage2_checkpoint": bool_text((output_dir / "stage2_finetune" / "checkpoints" / "best.pt").is_file()),
                "has_final_metrics": bool_text(final_metrics.is_file()),
                "ALL_top1": all_row.get("top1", ""),
                "DDD_top1": ddd_row.get("top1", ""),
                "mimic_top1": mimic_row.get("top1", ""),
                "mimic_top5": mimic_row.get("top5", ""),
                "mimic_rank_le_50": mimic_row.get("rank_le_50", ""),
                "notes": "; ".join(notes),
            }
        )
    return rows


def summarize_failures(*tables: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for table in tables:
        for row in table:
            status = str(row.get("status", ""))
            severity = str(row.get("severity", "Info"))
            if status in {"FAIL", "CONFIRMED_IGNORED"} and severity in counts:
                counts[severity] += 1
            if status == "WARN" and severity in {"Medium", "High", "Critical"}:
                counts[severity] += 1
    return counts


def table_path_map(report_dir: Path) -> dict[str, str]:
    names = [
        "repository_state",
        "snapshot_inventory",
        "manifest_consistency_checks",
        "top_config_vs_snapshot",
        "locked_config_vs_snapshot",
        "suspicious_config_keys",
        "runner_config_validation",
        "candidate_consistency",
        "final_result_traceability",
        "output_snapshot_comparison",
    ]
    return {name: str((report_dir / "tables" / f"{name}.csv").resolve()) for name in names}


def status_counts(rows: list[dict[str, Any]]) -> str:
    counter = Counter(str(row.get("status", "")) for row in rows)
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items())) or "none"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    shown = rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown:
        values = [to_text(row.get(col, "")).replace("\n", "<br>") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > limit:
        lines.append("| " + " | ".join(["..."] * len(columns)) + " |")
    return "\n".join(lines)


def problem_rows(
    top_comparison: list[dict[str, Any]],
    locked_comparison: list[dict[str, Any]],
    suspicious: list[dict[str, Any]],
    runner_validation: list[dict[str, Any]],
    snapshot_comparison: list[dict[str, Any]],
) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    high_mismatches = [row for row in top_comparison if row.get("status") == "FAIL" and row.get("severity") in {"High", "Critical"}]
    if high_mismatches:
        keys = ", ".join(str(row["key_path"]) for row in high_mismatches[:6])
        problems.append(
            {
                "severity": "High",
                "problem": "Current top-level config does not reproduce audited snapshot",
                "evidence": keys,
                "recommended_fix": "Create a locked config/manifest for the audited snapshot and do not reuse mutable top config as proof of reproduction.",
            }
        )
    locked_failures = [row for row in locked_comparison if row.get("status") == "FAIL" and row.get("severity") in {"High", "Critical"}]
    if locked_failures:
        keys = ", ".join(str(row["key_path"]) for row in locked_failures[:6])
        problems.append(
            {
                "severity": "High",
                "problem": "Locked config does not match audited snapshot",
                "evidence": keys,
                "recommended_fix": "Fix the locked config from run_manifest.json and generated stage configs before using it as a formal baseline.",
            }
        )
    ignored = [row for row in suspicious if row.get("status") == "CONFIRMED_IGNORED"]
    for row in ignored:
        problems.append(
            {
                "severity": str(row.get("severity", "High")),
                "problem": f"Confirmed ignored config key: {row.get('key_path')}",
                "evidence": "runner source does not mention the key leaf",
                "recommended_fix": "Remove the key or implement explicit warning/strict validation in the runner before using the config for formal runs.",
            }
        )
    runner_failures = [
        row
        for row in runner_validation
        if row.get("status") == "FAIL" and str(row.get("key_path", "")).startswith("runner.")
    ]
    if runner_failures:
        problems.append(
            {
                "severity": "High",
                "problem": "Runner config validation guard is missing",
                "evidence": ", ".join(str(row.get("key_path")) for row in runner_failures),
                "recommended_fix": "Add validate_pipeline_config_keys and --strict-config-keys before relying on strict config checks.",
            }
        )
    complete_snapshots = [row for row in snapshot_comparison if row.get("has_final_metrics") == "true"]
    if len(complete_snapshots) > 1:
        problems.append(
            {
                "severity": "Medium",
                "problem": "Multiple mainline output snapshots contain final metrics",
                "evidence": f"{len(complete_snapshots)} mainline_full_pipeline* dirs with final metrics",
                "recommended_fix": "Reference results by frozen manifest path and output dir in every report.",
            }
        )
    return problems


def write_markdown_report(
    path: Path,
    *,
    audit_time: str,
    run_dir: Path,
    top_config: Path,
    locked_config: Path | None,
    runner: Path,
    repo_rows: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    manifest_checks: list[dict[str, Any]],
    top_comparison: list[dict[str, Any]],
    locked_comparison: list[dict[str, Any]],
    suspicious: list[dict[str, Any]],
    runner_validation: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    traceability: list[dict[str, Any]],
    snapshot_comparison: list[dict[str, Any]],
    frozen_manifest_path: Path,
    tables: dict[str, str],
    commands_run: list[str],
) -> None:
    internal_bad = [
        row
        for row in [*manifest_checks, *candidates, *traceability]
        if row.get("status") == "FAIL" and row.get("severity", "High") in {"Critical", "High"}
    ]
    top_bad = [row for row in top_comparison if row.get("status") == "FAIL" and row.get("severity") in {"Critical", "High"}]
    locked_bad = [row for row in locked_comparison if row.get("status") == "FAIL" and row.get("severity") in {"Critical", "High"}]
    ignored = [row for row in suspicious if row.get("status") == "CONFIRMED_IGNORED"]
    locked_ignored = [
        row
        for row in runner_validation
        if locked_config
        and same_path(row.get("config_path"), locked_config)
        and row.get("strict_would_fail") == "true"
    ]
    runner_guard_ok = all(
        row.get("status") == "PASS"
        for row in runner_validation
        if str(row.get("key_path", "")).startswith("runner.")
    )
    candidate_bad = [row for row in candidates if row.get("status") == "FAIL"]
    trace_bad = [row for row in traceability if row.get("status") == "FAIL"]
    multiple_snapshots = len([row for row in snapshot_comparison if row.get("has_final_metrics") == "true"]) > 1
    problems = problem_rows(top_comparison, locked_comparison, suspicious, runner_validation, snapshot_comparison)

    executive = [
        "# Snapshot Consistency Audit",
        "",
        f"Generated at: `{audit_time}`",
        "",
        "## 1. Executive Summary",
        "",
        f"- Audited run internal consistency: {'PASS' if not internal_bad else 'FAIL'} ({len(internal_bad)} High/Critical internal failures).",
        f"- Current top-level config can reproduce audited run: {'NO' if top_bad else 'YES'} ({len(top_bad)} High/Critical mismatches).",
        f"- Locked config can serve as frozen reproduction config: {'YES' if locked_config and not locked_bad else 'NO'} ({len(locked_bad)} High/Critical locked-config mismatches).",
        f"- Checkpoint / candidate / final result mix detected inside audited run: {'NO' if not candidate_bad and not trace_bad else 'YES'}.",
        f"- Locked config contains ignored keys: {'YES' if locked_ignored else 'NO'}.",
        f"- Runner warning/strict config validation available: {'YES' if runner_guard_ok else 'NO'}.",
        f"- Confirmed silently ignored config keys after runner validation: {', '.join(row['key_path'] for row in ignored) if ignored else 'none'}.",
        f"- Multiple output snapshot mix risk: {'YES' if multiple_snapshots else 'NO'}.",
        "- Baseline recommendation: use the locked config plus frozen manifest for traceability; do not use the mutable top-level config as the reproduction contract.",
        "",
        "The high-risk state has moved from unguarded config drift to a locked snapshot plus runner-level warning/strict validation. The mutable config still does not reproduce the audited snapshot.",
        "",
        "## 2. Audited Snapshot",
        "",
        f"- run_dir: `{run_dir}`",
        f"- top_config: `{top_config}`",
        f"- locked_config: `{locked_config if locked_config else 'NOT_PROVIDED'}`",
        f"- runner: `{runner}`",
        f"- frozen manifest: `{frozen_manifest_path}`",
        "",
        "## 3. Repository State",
        "",
        markdown_table(repo_rows, ["item", "value", "status", "notes"], limit=12),
        "",
        "## 4. Snapshot Inventory",
        "",
        f"Full table: `{tables['snapshot_inventory']}`",
        "",
        markdown_table(inventory, ["type", "path", "exists", "size_bytes", "sha256", "notes"], limit=20),
        "",
        "## 5. Manifest Consistency Checks",
        "",
        f"Status summary: {status_counts(manifest_checks)}",
        "",
        markdown_table(manifest_checks, ["check_id", "check_name", "expected", "actual", "status", "severity", "notes"], limit=30),
        "",
        "## 6. Top Config vs Snapshot",
        "",
        f"Status summary: {status_counts(top_comparison)}",
        "",
        markdown_table(top_comparison, ["key_path", "top_config_value", "snapshot_value", "status", "severity", "notes"], limit=30),
        "",
        "## 7. Locked Config Verification",
        "",
        f"Status summary: {status_counts(locked_comparison)}",
        "",
        markdown_table(locked_comparison, ["key_path", "locked_config_value", "snapshot_value", "status", "severity", "notes"], limit=30),
        "",
        "## 8. Suspicious / Ignored Config Keys",
        "",
        f"Status summary: {status_counts(suspicious)}",
        "",
        markdown_table(suspicious, ["key_path", "value", "status", "severity", "notes"], limit=40),
        "",
        "## 9. Runner Config Validation",
        "",
        f"Status summary: {status_counts(runner_validation)}",
        "",
        markdown_table(runner_validation, ["config_path", "key_path", "value", "status", "strict_would_fail", "severity", "notes"], limit=30),
        "",
        "## 10. Candidate-Checkpoint Consistency",
        "",
        f"Status summary: {status_counts(candidates)}",
        "",
        markdown_table(candidates, ["candidate_file", "metadata_file", "checkpoint_match", "n_cases", "n_rows", "candidate_count_issue_cases", "duplicate_candidate_cases", "status", "notes"], limit=20),
        "",
        "## 11. Final Result Traceability",
        "",
        f"Status summary: {status_counts(traceability)}",
        "",
        markdown_table(traceability, ["dataset", "metric_source", "source_path", "source_exists", "inside_run_dir", "expected_source_type", "status", "notes"], limit=20),
        "",
        "## 12. Output Snapshot Comparison",
        "",
        markdown_table(snapshot_comparison, ["output_dir", "has_manifest", "has_stage2_checkpoint", "has_final_metrics", "ALL_top1", "DDD_top1", "mimic_top1", "mimic_top5", "mimic_rank_le_50", "notes"], limit=30),
        "",
        "## 13. Confirmed Problems",
        "",
    ]
    if problems:
        executive.append(markdown_table(problems, ["severity", "problem", "evidence", "recommended_fix"], limit=20))
    else:
        executive.append("No Critical/High confirmed problems were found.")
    executive.extend(
        [
            "",
            "## 14. Risks / Warnings",
            "",
            "- The repository worktree is dirty; treat existing unrelated changes as separate user work.",
            "- Several `mainline_full_pipeline*` output directories contain metrics; reports must cite the exact output directory and manifest.",
            "- Config-template paths and generated stage config paths are intentionally different; this is acceptable only when documented by a frozen manifest.",
            "",
            "## 15. Recommended Fixes",
            "",
            "1. Treat `configs/mainline_full_pipeline_hybrid_tag_v5.locked.yaml` as the frozen config for this audited snapshot.",
            "2. Keep `--strict-config-keys` enabled for formal config checks before starting any expensive run.",
            "3. Require every final report to cite `run_manifest.json`, candidate metadata, final metrics source table, locked config, and frozen audit manifest.",
            "4. Keep future audit outputs outside `outputs/mainline_full_pipeline*` to avoid overwriting experiment artifacts.",
            "",
            "## 16. Commands Run",
            "",
            *[f"- `{command}`" for command in commands_run],
            "",
            "## 17. Generated Files",
            "",
            f"- Markdown report: `{path}`",
            f"- frozen manifest: `{frozen_manifest_path}`",
            *[f"- {name}: `{table_path}`" for name, table_path in tables.items()],
            "",
        ]
    )
    path.write_text("\n".join(executive), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    top_config_path = resolve_path(args.top_config)
    locked_config_path = resolve_path(args.locked_config) if args.locked_config else None
    runner_path = resolve_path(args.runner)
    audit_time = datetime.now().isoformat(timespec="seconds")
    stamp = now_stamp()
    report_dir = prepare_target_dir(args.out_report_dir, stamp)
    output_dir = prepare_target_dir(args.out_output_dir, stamp)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    repo_rows, git_commit, git_status = build_repository_state(run_dir, top_config_path, runner_path, audit_time, locked_config_path)
    manifest = load_json(run_dir / "run_manifest.json")
    top_config = load_yaml(top_config_path)
    locked_config = load_yaml(locked_config_path) if locked_config_path else {}
    stage1 = load_yaml(run_dir / "configs" / "stage1_pretrain.yaml")
    stage2 = load_yaml(run_dir / "configs" / "stage2_finetune.yaml")

    inventory = build_snapshot_inventory(run_dir)
    manifest_checks = build_manifest_checks(run_dir, manifest)
    top_comparison = build_top_config_comparison(top_config, manifest, stage1, stage2, run_dir)
    locked_comparison = build_locked_config_comparison(locked_config, locked_config_path, manifest, stage1, stage2, run_dir)
    suspicious = build_suspicious_keys(top_config, runner_path)
    runner_validation = build_runner_config_validation(
        [("top_config", top_config_path), ("locked_config", locked_config_path)],
        runner_path,
    )
    expected_checkpoint = Path(manifest.get("finetune_checkpoint") or (run_dir / "stage2_finetune" / "checkpoints" / "best.pt"))
    candidates = build_candidate_consistency(run_dir, expected_checkpoint)
    traceability = build_final_traceability(run_dir)
    snapshot_comparison = build_snapshot_comparison()
    fail_counts = summarize_failures(manifest_checks, top_comparison, locked_comparison, suspicious, runner_validation, candidates, traceability)
    tables = table_path_map(report_dir)

    write_csv(repo_rows, Path(tables["repository_state"]), ["item", "value", "status", "notes"])
    write_csv(inventory, Path(tables["snapshot_inventory"]), ["type", "path", "exists", "size_bytes", "sha256", "mtime", "notes"])
    write_csv(manifest_checks, Path(tables["manifest_consistency_checks"]), ["check_id", "check_name", "expected", "actual", "status", "severity", "evidence_path", "notes"])
    write_csv(top_comparison, Path(tables["top_config_vs_snapshot"]), ["key_path", "top_config_value", "snapshot_value", "status", "severity", "notes"])
    write_csv(locked_comparison, Path(tables["locked_config_vs_snapshot"]), ["key_path", "locked_config_value", "snapshot_value", "status", "severity", "notes"])
    write_csv(suspicious, Path(tables["suspicious_config_keys"]), ["key_path", "value", "runner_mentions_leaf_key", "runner_mentions_full_key", "status", "severity", "notes"])
    write_csv(runner_validation, Path(tables["runner_config_validation"]), ["config_path", "key_path", "value", "status", "strict_would_fail", "severity", "notes"])
    write_csv(candidates, Path(tables["candidate_consistency"]), ["candidate_file", "metadata_file", "checkpoint_in_metadata", "expected_checkpoint", "checkpoint_match", "n_cases", "n_rows", "candidate_count_issue_cases", "duplicate_candidate_cases", "status", "notes"])
    write_csv(traceability, Path(tables["final_result_traceability"]), ["dataset", "metric_source", "source_path", "source_exists", "inside_run_dir", "expected_source_type", "status", "notes"])
    write_csv(snapshot_comparison, Path(tables["output_snapshot_comparison"]), ["output_dir", "has_manifest", "has_stage2_checkpoint", "has_final_metrics", "ALL_top1", "DDD_top1", "mimic_top1", "mimic_top5", "mimic_rank_le_50", "notes"])

    key_files = [row for row in inventory if row.get("exists") == "true"]
    frozen_manifest = {
        "audit_time": audit_time,
        "repo_root": str(PROJECT_ROOT),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "audited_run_dir": str(run_dir),
        "top_config_path": str(top_config_path),
        "locked_config_path": str(locked_config_path) if locked_config_path else "NOT_PROVIDED",
        "runner_path": str(runner_path),
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "key_files": key_files,
        "key_file_hashes": {row["path"]: row["sha256"] for row in key_files},
        "inferred_checkpoint_path": str(expected_checkpoint),
        "inferred_exact_eval_path": str((run_dir / "stage3_exact_eval" / "exact_details.csv").resolve()),
        "inferred_candidate_paths": [
            str((run_dir / "stage4_candidates" / "top50_candidates_validation.csv").resolve()),
            str((run_dir / "stage4_candidates" / "top50_candidates_test.csv").resolve()),
        ],
        "inferred_final_metrics_path": str((run_dir / "mainline_final_metrics_with_sources.csv").resolve()),
        "consistency_summary": {
            "manifest_checks": status_counts(manifest_checks),
            "top_config_vs_snapshot": status_counts(top_comparison),
            "locked_config_vs_snapshot": status_counts(locked_comparison),
            "suspicious_config_keys": status_counts(suspicious),
            "runner_config_validation": status_counts(runner_validation),
            "candidate_consistency": status_counts(candidates),
            "final_result_traceability": status_counts(traceability),
        },
        "critical_fail_count": fail_counts["Critical"],
        "high_fail_count": fail_counts["High"],
        "medium_warn_count": fail_counts["Medium"],
    }
    frozen_manifest_path = output_dir / "frozen_snapshot_manifest.json"
    write_json(frozen_manifest, frozen_manifest_path)

    commands_run = [
        "D:\\python\\python.exe -m compileall tools/audit_snapshot_consistency.py tools/run_full_mainline_pipeline.py",
        "D:\\python\\python.exe -c \"from tools.run_full_mainline_pipeline import validate_pipeline_config_keys; import yaml; cfg=yaml.safe_load(open('configs/mainline_full_pipeline.yaml', encoding='utf-8')); validate_pipeline_config_keys(cfg, strict=False)\"",
        "D:\\python\\python.exe -c \"from tools.run_full_mainline_pipeline import validate_pipeline_config_keys; import yaml; cfg=yaml.safe_load(open('configs/mainline_full_pipeline_hybrid_tag_v5.locked.yaml', encoding='utf-8')); validate_pipeline_config_keys(cfg, strict=True)\"",
        "D:\\python\\python.exe tools/audit_snapshot_consistency.py --run-dir outputs/mainline_full_pipeline_hybrid_tag_v5 --top-config configs/mainline_full_pipeline.yaml --locked-config configs/mainline_full_pipeline_hybrid_tag_v5.locked.yaml --runner tools/run_full_mainline_pipeline.py --out-report-dir reports/snapshot_consistency_lock_audit --out-output-dir outputs/snapshot_consistency_lock_audit",
    ]
    report_path = report_dir / "snapshot_consistency_audit.md"
    write_markdown_report(
        report_path,
        audit_time=audit_time,
        run_dir=run_dir,
        top_config=top_config_path,
        locked_config=locked_config_path,
        runner=runner_path,
        repo_rows=repo_rows,
        inventory=inventory,
        manifest_checks=manifest_checks,
        top_comparison=top_comparison,
        locked_comparison=locked_comparison,
        suspicious=suspicious,
        runner_validation=runner_validation,
        candidates=candidates,
        traceability=traceability,
        snapshot_comparison=snapshot_comparison,
        frozen_manifest_path=frozen_manifest_path,
        tables=tables,
        commands_run=commands_run,
    )

    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "frozen_manifest_path": str(frozen_manifest_path),
                "critical_fail_count": fail_counts["Critical"],
                "high_fail_count": fail_counts["High"],
                "medium_warn_count": fail_counts["Medium"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict and (fail_counts["Critical"] or fail_counts["High"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
