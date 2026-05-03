from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "mainline_full_pipeline_hybrid_tag_v5"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "current_framework_accuracy_audit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "current_framework_accuracy_audit"

GENERIC_HPO_IDS = {
    "HP:0000001",
    "HP:0000118",
    "HP:0000707",
    "HP:0001507",
    "HP:0001263",
    "HP:0004322",
    "HP:0001250",
    "HP:0001249",
    "HP:0001252",
    "HP:0002011",
}


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, **kwargs)
    raise ValueError(f"Unsupported table suffix: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def normalize_dataset_name(path_or_name: str | Path) -> str:
    stem = Path(str(path_or_name)).stem
    known = {
        "ddd": "DDD",
        "hms": "HMS",
        "lirical": "LIRICAL",
        "mme": "MME",
        "mygene2": "MyGene2",
        "ramedis": "RAMEDIS",
        "mimic_test_recleaned_mondo_hpo_rows": "mimic_test_recleaned_mondo_hpo_rows",
        "mimic_rag_0425": "mimic_rag_0425",
        "fakedisease": "FakeDisease",
    }
    return known.get(stem.lower(), stem)


def case_namespace(source_path: Path, split: str) -> str:
    return f"{split}::{rel_path(source_path)}"


def namespaced_case_id(raw_case_id: Any, source_path: Path, split: str) -> str:
    return f"{case_namespace(source_path, split)}::{str(raw_case_id)}"


def load_case_long(path: Path, split: str, case_id_col: str, label_col: str, hpo_col: str) -> pd.DataFrame:
    df = read_table(path)
    if label_col not in df.columns and "mondo_id" in df.columns:
        df = df.rename(columns={"mondo_id": label_col})
    missing = [col for col in (case_id_col, label_col, hpo_col) if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[[case_id_col, label_col, hpo_col]].copy()
    out[case_id_col] = out[case_id_col].astype(str).map(lambda cid: namespaced_case_id(cid, path, split))
    out["_raw_case_id"] = df[case_id_col].astype(str)
    out["_source_file"] = str(path.resolve())
    out["_source_name"] = normalize_dataset_name(path)
    return out


def split_by_case(df: pd.DataFrame, val_ratio: float, random_seed: int, case_id_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_ids = df[case_id_col].dropna().astype(str).unique().tolist()
    if val_ratio == 0.0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    rng = np.random.default_rng(int(random_seed))
    case_ids = list(case_ids)
    rng.shuffle(case_ids)
    val_case_count = int(len(case_ids) * float(val_ratio))
    val_ids = set(case_ids[:val_case_count])
    train_ids = set(case_ids[val_case_count:])
    return (
        df[df[case_id_col].isin(train_ids)].reset_index(drop=True),
        df[df[case_id_col].isin(val_ids)].reset_index(drop=True),
    )


def load_hp_resources() -> tuple[dict[str, str], set[str], dict[str, str]]:
    candidates = [PROJECT_ROOT / "raw_data" / "hp.json", PROJECT_ROOT / "data" / "raw_data" / "hp-base.json"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}, set(), {}
    graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    names: dict[str, str] = {}
    obsolete: set[str] = set()
    alt_to_primary: dict[str, str] = {}
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", ""))
        if "HP_" not in node_id:
            continue
        hp_id = "HP:" + node_id.rsplit("HP_", 1)[1]
        if node.get("lbl"):
            names[hp_id] = str(node["lbl"])
        meta = node.get("meta", {}) or {}
        if meta.get("deprecated"):
            obsolete.add(hp_id)
        for bpv in meta.get("basicPropertyValues", []) or []:
            pred = str(bpv.get("pred", ""))
            val = str(bpv.get("val", ""))
            if pred.endswith("hasAlternativeId") and val.startswith("HP:"):
                alt_to_primary[val] = hp_id
            if meta.get("deprecated") and pred.endswith("IAO_0100001") and val.startswith("HP:"):
                alt_to_primary[hp_id] = val
    return names, obsolete, alt_to_primary


def load_mondo_names() -> tuple[dict[str, str], set[str]]:
    candidates = [PROJECT_ROOT / "data" / "raw_data" / "mondo.json", PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}, set()
    graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    names: dict[str, str] = {}
    obsolete: set[str] = set()
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", ""))
        if "MONDO_" not in node_id:
            continue
        mondo_id = "MONDO:" + node_id.rsplit("MONDO_", 1)[1]
        if node.get("lbl"):
            names[mondo_id] = str(node["lbl"])
        if (node.get("meta", {}) or {}).get("deprecated"):
            obsolete.add(mondo_id)
    return names, obsolete


def load_static_indices(train_config: dict[str, Any]) -> dict[str, Any]:
    paths = train_config["paths"]
    hpo_index_path = resolve_path(paths["hpo_index_path"])
    disease_index_path = resolve_path(paths["disease_index_path"])
    incidence_path = resolve_path(paths["disease_incidence_path"])

    hpo_df = pd.read_excel(hpo_index_path, dtype={"hpo_id": str})
    disease_df = pd.read_excel(disease_index_path, dtype={"mondo_id": str})
    hpo_to_idx = dict(zip(hpo_df["hpo_id"].astype(str), hpo_df["hpo_idx"].astype(int)))
    idx_to_hpo = dict(zip(hpo_df["hpo_idx"].astype(int), hpo_df["hpo_id"].astype(str)))
    disease_to_idx = dict(zip(disease_df["mondo_id"].astype(str), disease_df["disease_idx"].astype(int)))
    idx_to_disease = dict(zip(disease_df["disease_idx"].astype(int), disease_df["mondo_id"].astype(str)))

    try:
        h_disease = load_npz(incidence_path)
    except Exception:
        npz = np.load(incidence_path, allow_pickle=False)
        h_disease = csr_matrix((npz["vals"], (npz["rows"], npz["cols"])), shape=tuple(npz["shape"]))
    h_disease = h_disease.tocsc()
    disease_with_hpo_idx = set(int(i) for i in np.flatnonzero(np.diff(h_disease.indptr) > 0))
    disease_with_hpo = {idx_to_disease[i] for i in disease_with_hpo_idx if i in idx_to_disease}
    hpo_df_counts = np.diff(h_disease.tocsr().indptr).astype(np.float64)
    num_disease = h_disease.shape[1]
    hpo_specificity = np.log((num_disease + 1.0) / (hpo_df_counts + 1.0))
    disease_hpo_sets: dict[str, set[str]] = {}
    for disease_idx in range(h_disease.shape[1]):
        disease = idx_to_disease.get(disease_idx)
        if disease is None:
            continue
        disease_hpo_sets[disease] = {
            idx_to_hpo[int(hpo_idx)]
            for hpo_idx in h_disease[:, disease_idx].indices
            if int(hpo_idx) in idx_to_hpo
        }
    return {
        "hpo_index_path": hpo_index_path,
        "disease_index_path": disease_index_path,
        "incidence_path": incidence_path,
        "hpo_df": hpo_df,
        "disease_df": disease_df,
        "hpo_to_idx": hpo_to_idx,
        "disease_to_idx": disease_to_idx,
        "disease_with_hpo": disease_with_hpo,
        "disease_hpo_sets": disease_hpo_sets,
        "hpo_specificity": hpo_specificity,
        "idx_to_hpo": idx_to_hpo,
        "idx_to_disease": idx_to_disease,
        "H_disease_shape": tuple(int(v) for v in h_disease.shape),
        "H_disease_nnz": int(h_disease.nnz),
    }


def case_table(df: pd.DataFrame, case_id_col: str, label_col: str, hpo_col: str) -> pd.DataFrame:
    rows = []
    for case_id, group in df.groupby(case_id_col, sort=False):
        labels_in_order = group[label_col].dropna().astype(str).tolist()
        label_set = sorted(set(labels_in_order))
        hpos = sorted(set(group[hpo_col].dropna().astype(str)))
        rows.append(
            {
                "case_id": str(case_id),
                "dataset": str(group["_source_name"].iloc[0]) if "_source_name" in group else "",
                "source_file": str(group["_source_file"].iloc[0]) if "_source_file" in group else "",
                "gold_id": labels_in_order[0] if labels_in_order else "",
                "label_set": "|".join(label_set),
                "label_count": len(label_set),
                "hpo_ids": hpos,
                "hpo_count": len(hpos),
            }
        )
    return pd.DataFrame(rows)


def metrics_from_ranks(ranks: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = pd.to_numeric(pd.Series(ranks), errors="coerce").fillna(10**9).to_numpy(dtype=np.int64)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "top1": np.nan,
            "top3": np.nan,
            "top5": np.nan,
            "top10": np.nan,
            "top30": np.nan,
            "rank_le_50": np.nan,
            "median_rank": np.nan,
            "mean_rank": np.nan,
            "mrr": np.nan,
        }
    return {
        "n": n,
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "top10": float(np.mean(arr <= 10)),
        "top30": float(np.mean(arr <= 30)),
        "rank_le_50": float(np.mean(arr <= 50)),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
        "mrr": float(np.mean(1.0 / np.maximum(arr, 1))),
    }


def build_inventory(run_dir: Path, report_dir: Path, output_dir: Path) -> pd.DataFrame:
    paths = {
        "config": [
            PROJECT_ROOT / "configs" / "mainline_full_pipeline.yaml",
            PROJECT_ROOT / "configs" / "train_pretrain.yaml",
            PROJECT_ROOT / "configs" / "train_finetune_attn_idf_main.yaml",
            PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml",
            run_dir / "configs" / "stage1_pretrain.yaml",
            run_dir / "configs" / "stage2_finetune.yaml",
            run_dir / "configs" / "stage3_exact_eval_train.yaml",
        ],
        "script": [
            PROJECT_ROOT / "tools" / "run_full_mainline_pipeline.py",
            PROJECT_ROOT / "tools" / "export_top50_candidates.py",
            PROJECT_ROOT / "tools" / "run_top50_evidence_rerank.py",
            PROJECT_ROOT / "tools" / "run_mimic_similar_case_aug.py",
            Path(__file__).resolve(),
        ],
        "src": [
            PROJECT_ROOT / "src" / "data" / "dataset.py",
            PROJECT_ROOT / "src" / "data" / "build_hypergraph.py",
            PROJECT_ROOT / "src" / "models" / "hgnn_encoder.py",
            PROJECT_ROOT / "src" / "hgnn_encoder_tag.py",
            PROJECT_ROOT / "src" / "models" / "model_pipeline.py",
            PROJECT_ROOT / "src" / "training" / "trainer.py",
            PROJECT_ROOT / "src" / "evaluation" / "evaluator.py",
            PROJECT_ROOT / "src" / "runtime_config.py",
        ],
        "checkpoint": [
            run_dir / "stage1_pretrain" / "checkpoints" / "best.pt",
            run_dir / "stage2_finetune" / "checkpoints" / "best.pt",
        ],
        "result": [
            run_dir / "stage3_exact_eval" / "exact_per_dataset.csv",
            run_dir / "stage3_exact_eval" / "exact_details.csv",
            run_dir / "stage5_ddd_rerank" / "rerank_fixed_test_metrics.csv",
            run_dir / "stage6_mimic_similar_case" / "similar_case_fixed_test.csv",
            run_dir / "mainline_final_metrics.csv",
            run_dir / "mainline_final_metrics_with_sources.csv",
            run_dir / "mainline_final_case_ranks.csv",
        ],
        "candidate": [
            run_dir / "stage4_candidates" / "top50_candidates_validation.csv",
            run_dir / "stage4_candidates" / "top50_candidates_test.csv",
        ],
        "dataset": [
            PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "test" / "DDD.csv",
            PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "test" / "mimic_test_recleaned_mondo_hpo_rows.csv",
            PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx",
            PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "HPO_index_v4.xlsx",
            PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "rare_disease_hgnn_clean_package_v59" / "v59DiseaseHy.npz",
            PROJECT_ROOT / "v59_rare_disease_authoritative_diagnostic_guide_final",
        ],
        "audit_output": [report_dir, output_dir],
    }
    rows = []
    notes = {
        "mainline_full_pipeline.yaml": "top-level staged pipeline config",
        "stage2_finetune.yaml": "effective finetune config captured by run",
        "top50_candidates_test.csv": "test top50 candidate export",
        "mainline_final_metrics.csv": "final mixed metrics after DDD/mimic postprocess",
        "v59DiseaseHy.npz": "v59 disease-HPO incidence matrix",
    }
    for kind, items in paths.items():
        for path in items:
            exists = path.exists()
            rows.append(
                {
                    "type": kind,
                    "path": str(path.resolve()),
                    "purpose": notes.get(path.name, kind),
                    "exists": bool(exists),
                    "notes": (
                        f"size={path.stat().st_size}; mtime={path.stat().st_mtime}"
                        if exists and path.is_file()
                        else ("directory" if exists and path.is_dir() else "NOT_FOUND")
                    ),
                }
            )
    return pd.DataFrame(rows)


def git_state() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            out = subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.STDOUT, text=True)
            return out.strip()
        except Exception as exc:
            return f"NOT_FOUND_OR_ERROR: {exc}"

    return {
        "status_short": run_git(["status", "--short"]),
        "status_branch": run_git(["status", "--branch", "--short"]),
    }


def load_eval_tables(data_config: dict[str, Any], train_config: dict[str, Any]) -> dict[str, Any]:
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))

    test_frames = []
    for raw_path in data_config.get("test_files", []):
        path = resolve_path(raw_path)
        test_frames.append(load_case_long(path, "test", case_id_col, label_col, hpo_col))
    test_df = pd.concat(test_frames, ignore_index=True)

    train_paths = [resolve_path(path) for path in train_config["paths"]["train_files"]]
    train_frames = [load_case_long(path, "train", case_id_col, label_col, hpo_col) for path in train_paths]
    all_train_df = pd.concat(train_frames, ignore_index=True)
    train_split_df, val_split_df = split_by_case(
        all_train_df,
        val_ratio=float(train_config["data"]["val_ratio"]),
        random_seed=int(train_config["data"]["random_seed"]),
        case_id_col=case_id_col,
    )
    return {
        "case_id_col": case_id_col,
        "label_col": label_col,
        "hpo_col": hpo_col,
        "test_df": test_df,
        "all_train_df": all_train_df,
        "train_split_df": train_split_df,
        "val_split_df": val_split_df,
    }


def summarize_dataset_quality(
    tables: dict[str, Any],
    static: dict[str, Any],
    hp_names: dict[str, str],
    hp_obsolete: set[str],
    mondo_obsolete: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hpo_to_idx = static["hpo_to_idx"]
    disease_to_idx = static["disease_to_idx"]
    disease_with_hpo = static["disease_with_hpo"]
    disease_hpo_sets = static["disease_hpo_sets"]
    hpo_specificity = static["hpo_specificity"]
    case_id_col = tables["case_id_col"]
    label_col = tables["label_col"]
    hpo_col = tables["hpo_col"]

    groups = {
        "test": tables["test_df"],
        "validation": tables["val_split_df"],
        "train_split": tables["train_split_df"],
    }
    rows = []
    top_hpo_rows = []
    overlap_rows = []
    train_labels = set(case_table(tables["train_split_df"], case_id_col, label_col, hpo_col)["gold_id"].astype(str))
    train_hpos = set(tables["train_split_df"][hpo_col].dropna().astype(str))

    for split, df in groups.items():
        ctab = case_table(df, case_id_col, label_col, hpo_col)
        for dataset, cgroup in ctab.groupby("dataset", sort=True):
            if cgroup.empty:
                continue
            hpo_counts = cgroup["hpo_count"].astype(int)
            all_hpos = [h for hpos in cgroup["hpo_ids"] for h in hpos]
            unique_hpos = set(all_hpos)
            mapped_hpos = {h for h in unique_hpos if h in hpo_to_idx}
            obsolete_hpos = {h for h in unique_hpos if h in hp_obsolete}
            not_train_hpos = {h for h in unique_hpos if h not in train_hpos}
            golds = cgroup["gold_id"].astype(str)
            valid_gold_mask = golds.isin(disease_to_idx)
            v59_mask = golds.isin(disease_with_hpo)
            label_in_train = golds.isin(train_labels)
            generic_case_mask = cgroup["hpo_ids"].apply(lambda xs: bool(set(xs) & GENERIC_HPO_IDS))

            overlap_counts = []
            jaccards = []
            gold_coverages = []
            case_coverages = []
            case_ic_means = []
            for row in cgroup.itertuples(index=False):
                case_hpos = set(row.hpo_ids)
                disease_hpos = disease_hpo_sets.get(str(row.gold_id), set())
                shared = case_hpos & disease_hpos
                union = case_hpos | disease_hpos
                overlap_counts.append(len(shared))
                jaccards.append(len(shared) / len(union) if union else 0.0)
                case_coverages.append(len(shared) / len(case_hpos) if case_hpos else 0.0)
                gold_coverages.append(len(shared) / len(disease_hpos) if disease_hpos else 0.0)
                idxs = [hpo_to_idx[h] for h in case_hpos if h in hpo_to_idx]
                case_ic_means.append(float(np.mean(hpo_specificity[idxs])) if idxs else 0.0)
                overlap_rows.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "case_id": row.case_id,
                        "gold_id": row.gold_id,
                        "hpo_count": int(row.hpo_count),
                        "gold_in_index": bool(str(row.gold_id) in disease_to_idx),
                        "gold_in_v59": bool(str(row.gold_id) in disease_with_hpo),
                        "gold_label_in_train_split": bool(str(row.gold_id) in train_labels),
                        "gold_disease_hpo_count": int(len(disease_hpos)),
                        "exact_hpo_overlap_count": int(len(shared)),
                        "case_hpo_coverage": float(case_coverages[-1]),
                        "gold_hpo_coverage": float(gold_coverages[-1]),
                        "jaccard": float(jaccards[-1]),
                        "mean_case_hpo_ic": float(case_ic_means[-1]),
                        "generic_hpo_present": bool(case_hpos & GENERIC_HPO_IDS),
                    }
                )

            q = hpo_counts.quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "n_cases": int(len(cgroup)),
                    "unique_gold": int(golds.nunique()),
                    "hpo_mean": float(hpo_counts.mean()),
                    "hpo_median": float(hpo_counts.median()),
                    "hpo_p25": float(q.loc[0.25]),
                    "hpo_p75": float(q.loc[0.75]),
                    "hpo_min": int(hpo_counts.min()),
                    "hpo_max": int(hpo_counts.max()),
                    "hpo_le_2_rate": float((hpo_counts <= 2).mean()),
                    "hpo_le_5_rate": float((hpo_counts <= 5).mean()),
                    "total_hpo_mentions": int(len(all_hpos)),
                    "unique_hpo": int(len(unique_hpos)),
                    "mapped_hpo": int(len(mapped_hpos)),
                    "unmapped_hpo": int(len(unique_hpos - mapped_hpos)),
                    "obsolete_hpo": int(len(obsolete_hpos)),
                    "hpo_mapped_rate": float(len(mapped_hpos) / len(unique_hpos)) if unique_hpos else 0.0,
                    "unique_gold_in_index": int(golds[golds.isin(disease_to_idx)].nunique()),
                    "gold_in_index_rate": float(valid_gold_mask.mean()),
                    "gold_in_v59_rate": float(v59_mask.mean()),
                    "obsolete_mondo_cases": int(golds.isin(mondo_obsolete).sum()),
                    "non_mondo_label_cases": int((~golds.str.startswith("MONDO:")).sum()),
                    "gold_label_in_train_case_rate": float(label_in_train.mean()),
                    "unique_gold_in_train_rate": float(len(set(golds) & train_labels) / golds.nunique()) if golds.nunique() else 0.0,
                    "generic_hpo_case_rate": float(generic_case_mask.mean()),
                    "gold_case_overlap_zero_rate": float(np.mean(np.asarray(overlap_counts) == 0)),
                    "gold_case_overlap_mean": float(np.mean(overlap_counts)),
                    "gold_case_jaccard_mean": float(np.mean(jaccards)),
                    "case_hpo_coverage_mean": float(np.mean(case_coverages)),
                    "gold_hpo_coverage_mean": float(np.mean(gold_coverages)),
                    "mean_case_hpo_ic": float(np.mean(case_ic_means)),
                    "not_in_train_unique_hpo": int(len(not_train_hpos)),
                }
            )

            top_counts = Counter(all_hpos)
            for hpo_id, count in top_counts.most_common(20):
                top_hpo_rows.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "hpo_id": hpo_id,
                        "hpo_name": hp_names.get(hpo_id, ""),
                        "case_or_row_mentions": int(count),
                        "mention_rate": float(count / max(len(all_hpos), 1)),
                        "is_generic_watchlist": bool(hpo_id in GENERIC_HPO_IDS),
                        "in_hpo_index": bool(hpo_id in hpo_to_idx),
                        "obsolete": bool(hpo_id in hp_obsolete),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(top_hpo_rows), pd.DataFrame(overlap_rows)


def summarize_metrics(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = pd.read_csv(run_dir / "stage3_exact_eval" / "exact_details.csv", dtype=str)
    exact["true_rank"] = pd.to_numeric(exact["true_rank"], errors="coerce")
    rows = []
    for dataset, group in exact.groupby("dataset_name", sort=True):
        row = {"dataset": dataset, "split": "test_exact_baseline", **metrics_from_ranks(group["true_rank"])}
        row["result_path"] = str((run_dir / "stage3_exact_eval" / "exact_details.csv").resolve())
        rows.append(row)
    row = {"dataset": "ALL", "split": "test_exact_baseline", **metrics_from_ranks(exact["true_rank"])}
    row["result_path"] = str((run_dir / "stage3_exact_eval" / "exact_details.csv").resolve())
    rows.append(row)

    final_path = run_dir / "mainline_final_case_ranks.csv"
    if final_path.is_file():
        final = pd.read_csv(final_path, dtype=str)
        rank_col = "final_rank" if "final_rank" in final.columns else "true_rank"
        final[rank_col] = pd.to_numeric(final[rank_col], errors="coerce")
        dataset_col = "dataset" if "dataset" in final.columns else "dataset_name"
        for dataset, group in final.groupby(dataset_col, sort=True):
            row = {"dataset": dataset, "split": "test_final_mixed", **metrics_from_ranks(group[rank_col])}
            row["result_path"] = str(final_path.resolve())
            rows.append(row)
        row = {"dataset": "ALL", "split": "test_final_mixed", **metrics_from_ranks(final[rank_col])}
        row["result_path"] = str(final_path.resolve())
        rows.append(row)

    final_metrics_path = run_dir / "mainline_final_metrics_with_sources.csv"
    final_metrics = pd.read_csv(final_metrics_path, dtype=str) if final_metrics_path.is_file() else pd.DataFrame()
    return pd.DataFrame(rows), final_metrics


def summarize_candidate_recall(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    issues = []
    for split, path in {
        "validation": run_dir / "stage4_candidates" / "top50_candidates_validation.csv",
        "test": run_dir / "stage4_candidates" / "top50_candidates_test.csv",
    }.items():
        if not path.is_file():
            continue
        cand = pd.read_csv(path, dtype=str)
        cand["original_rank"] = pd.to_numeric(cand["original_rank"], errors="coerce").astype(int)
        grouped = cand.groupby(["dataset_name", "case_id"], sort=False)
        per_case_rows = []
        for (dataset, case_id), group in grouped:
            gold = str(group["gold_id"].iloc[0])
            gold_ranks = group.loc[group["candidate_id"].astype(str) == gold, "original_rank"].tolist()
            candidate_count = len(group)
            unique_count = group["candidate_id"].astype(str).nunique()
            rank = min(gold_ranks) if gold_ranks else 10**9
            per_case_rows.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "case_id": case_id,
                    "gold_id": gold,
                    "candidate_count": int(candidate_count),
                    "unique_candidate_count": int(unique_count),
                    "duplicate_candidate_count": int(candidate_count - unique_count),
                    "gold_rank_in_top50_file": int(rank),
                    "gold_in_candidates": bool(rank <= 50),
                }
            )
        per_case = pd.DataFrame(per_case_rows)
        issues.append(per_case)
        for dataset, group in per_case.groupby("dataset", sort=True):
            arr = group["gold_rank_in_top50_file"].to_numpy(dtype=np.int64)
            rows.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "n": int(len(group)),
                    "gold@1": float(np.mean(arr <= 1)),
                    "gold@3": float(np.mean(arr <= 3)),
                    "gold@5": float(np.mean(arr <= 5)),
                    "gold@10": float(np.mean(arr <= 10)),
                    "gold@30": float(np.mean(arr <= 30)),
                    "gold@50": float(np.mean(arr <= 50)),
                    "gold_not_top50": int(np.sum(arr > 50)),
                    "gold_missing_index": "not_applicable_candidate_file",
                    "cases_with_duplicate_mondo": int((group["duplicate_candidate_count"] > 0).sum()),
                    "cases_candidate_count_lt50": int((group["candidate_count"] < 50).sum()),
                    "candidate_file": str(path.resolve()),
                }
            )
        arr = per_case["gold_rank_in_top50_file"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "split": split,
                "dataset": "ALL",
                "n": int(len(per_case)),
                "gold@1": float(np.mean(arr <= 1)),
                "gold@3": float(np.mean(arr <= 3)),
                "gold@5": float(np.mean(arr <= 5)),
                "gold@10": float(np.mean(arr <= 10)),
                "gold@30": float(np.mean(arr <= 30)),
                "gold@50": float(np.mean(arr <= 50)),
                "gold_not_top50": int(np.sum(arr > 50)),
                "gold_missing_index": "not_applicable_candidate_file",
                "cases_with_duplicate_mondo": int((per_case["duplicate_candidate_count"] > 0).sum()),
                "cases_candidate_count_lt50": int((per_case["candidate_count"] < 50).sum()),
                "candidate_file": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows), pd.concat(issues, ignore_index=True) if issues else pd.DataFrame()


def build_failure_cases(
    run_dir: Path,
    quality_case: pd.DataFrame,
    mondo_names: dict[str, str],
) -> pd.DataFrame:
    details_path = run_dir / "stage3_exact_eval" / "exact_details.csv"
    cand_path = run_dir / "stage4_candidates" / "top50_candidates_test.csv"
    details = pd.read_csv(details_path, dtype=str)
    details["true_rank"] = pd.to_numeric(details["true_rank"], errors="coerce").fillna(10**9).astype(int)
    candidates = pd.read_csv(cand_path, dtype=str)
    candidates["original_rank"] = pd.to_numeric(candidates["original_rank"], errors="coerce").fillna(10**9).astype(int)
    candidates["shared_hpo_count"] = pd.to_numeric(candidates.get("shared_hpo_count", 0), errors="coerce").fillna(0)
    candidates["jaccard_overlap"] = pd.to_numeric(candidates.get("jaccard_overlap", 0), errors="coerce").fillna(0.0)
    top1 = candidates[candidates["original_rank"] == 1].set_index("case_id")
    qcase = quality_case.set_index("case_id")

    rows = []
    mimic = details[details["dataset_name"].astype(str).str.startswith("mimic")].copy()
    buckets = [
        mimic[mimic["true_rank"] > 50].head(8),
        mimic[(mimic["true_rank"] > 5) & (mimic["true_rank"] <= 50)].head(6),
        mimic[(mimic["true_rank"] > 1) & (mimic["true_rank"] <= 5)].head(6),
        mimic.merge(qcase[["hpo_count"]], left_on="case_id", right_index=True, how="left")
        .query("hpo_count <= 5 and true_rank > 5")
        .head(6),
        mimic.merge(qcase[["hpo_count"]], left_on="case_id", right_index=True, how="left")
        .query("hpo_count >= 20 and true_rank > 5")
        .head(6),
    ]
    selected = pd.concat(buckets, ignore_index=True).drop_duplicates("case_id").head(30)
    for row in selected.itertuples(index=False):
        case_id = str(row.case_id)
        q = qcase.loc[case_id] if case_id in qcase.index else None
        t1 = top1.loc[case_id] if case_id in top1.index else None
        gold = str(row.true_label)
        top1_id = str(row.pred_top1)
        hpo_count = int(q["hpo_count"]) if q is not None and not pd.isna(q["hpo_count"]) else -1
        overlap_count = int(q["exact_hpo_overlap_count"]) if q is not None and not pd.isna(q["exact_hpo_overlap_count"]) else -1
        mean_ic = float(q["mean_case_hpo_ic"]) if q is not None and not pd.isna(q["mean_case_hpo_ic"]) else 0.0
        generic = bool(q["generic_hpo_present"]) if q is not None else False
        jaccard_top1 = float(t1["jaccard_overlap"]) if t1 is not None and "jaccard_overlap" in t1 else 0.0

        rank = int(row.true_rank)
        failure_type = "UNKNOWN"
        reasons = []
        if rank > 50:
            failure_type = "GOLD_NOT_IN_TOP50"
            reasons.append("gold outside HGNN top50 candidate set")
        elif rank > 5:
            failure_type = "RERANK_SORTING_FAILURE"
            reasons.append("gold is recallable but ranked below top5")
        elif rank > 1:
            failure_type = "SIMILAR_DISEASE_CONFUSION"
            reasons.append("gold is near top but not top1")
        if hpo_count <= 5:
            failure_type = "LOW_HPO_COUNT" if rank > 5 else failure_type
            reasons.append("low case HPO count")
        if overlap_count == 0:
            failure_type = "HPO_MAPPING_LOSS" if rank > 50 else failure_type
            reasons.append("zero overlap with gold disease hyperedge")
        if generic or mean_ic < 4.5:
            reasons.append("generic/low-specificity HPO profile")
        if jaccard_top1 >= 0.3 and rank > 1:
            failure_type = "SIMILAR_DISEASE_CONFUSION"
            reasons.append("top1 has substantial HPO overlap with case/gold context")

        rows.append(
            {
                "case_id": case_id,
                "gold_mondo": gold,
                "gold_name": mondo_names.get(gold, ""),
                "hpo_count": hpo_count,
                "gold_rank": rank,
                "top1_mondo": top1_id,
                "top1_name": mondo_names.get(top1_id, ""),
                "failure_type": failure_type,
                "possible_reason": "; ".join(dict.fromkeys(reasons)),
            }
        )
    return pd.DataFrame(rows)


def build_gap_table(quality: pd.DataFrame, candidate_recall: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    def get_quality(dataset: str) -> pd.Series:
        subset = quality[(quality["split"] == "test") & (quality["dataset"] == dataset)]
        return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)

    def get_recall(dataset: str) -> pd.Series:
        subset = candidate_recall[(candidate_recall["split"] == "test") & (candidate_recall["dataset"] == dataset)]
        return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)

    def get_metric(dataset: str) -> pd.Series:
        subset = metrics[(metrics["split"] == "test_exact_baseline") & (metrics["dataset"] == dataset)]
        return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)

    pairs = []
    for label, ddd_val, mimic_val, impact in [
        ("n_cases", get_quality("DDD").get("n_cases"), get_quality("mimic_test_recleaned_mondo_hpo_rows").get("n_cases"), "sample size and CI width"),
        ("hpo_mean", get_quality("DDD").get("hpo_mean"), get_quality("mimic_test_recleaned_mondo_hpo_rows").get("hpo_mean"), "case evidence density"),
        ("hpo_le_5_rate", get_quality("DDD").get("hpo_le_5_rate"), get_quality("mimic_test_recleaned_mondo_hpo_rows").get("hpo_le_5_rate"), "sparse case risk"),
        ("gold_case_overlap_zero_rate", get_quality("DDD").get("gold_case_overlap_zero_rate"), get_quality("mimic_test_recleaned_mondo_hpo_rows").get("gold_case_overlap_zero_rate"), "candidate recall risk"),
        ("gold_label_in_train_case_rate", get_quality("DDD").get("gold_label_in_train_case_rate"), get_quality("mimic_test_recleaned_mondo_hpo_rows").get("gold_label_in_train_case_rate"), "direct training signal"),
        ("unique_gold_in_train_rate", get_quality("DDD").get("unique_gold_in_train_rate"), get_quality("mimic_test_recleaned_mondo_hpo_rows").get("unique_gold_in_train_rate"), "disease coverage in train split"),
        ("gold@50", get_recall("DDD").get("gold@50"), get_recall("mimic_test_recleaned_mondo_hpo_rows").get("gold@50"), "top50 rerank upper bound"),
        ("top1_exact", get_metric("DDD").get("top1"), get_metric("mimic_test_recleaned_mondo_hpo_rows").get("top1"), "baseline exact ranking"),
        ("median_rank_exact", get_metric("DDD").get("median_rank"), get_metric("mimic_test_recleaned_mondo_hpo_rows").get("median_rank"), "ranking tail severity"),
    ]:
        diff = None
        try:
            diff = float(mimic_val) - float(ddd_val)
        except Exception:
            diff = ""
        pairs.append(
            {
                "comparison_item": label,
                "DDD": ddd_val,
                "mimic_test": mimic_val,
                "difference_mimic_minus_DDD": diff,
                "possible_impact": impact,
            }
        )
    return pd.DataFrame(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit for current RareDisease HGNN framework accuracy.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    report_dir = resolve_path(args.report_dir)
    output_dir = resolve_path(args.output_dir)
    tables_dir = report_dir / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    stage2_config_path = run_dir / "configs" / "stage2_finetune.yaml"
    data_config_path = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
    train_config = load_yaml(stage2_config_path)
    data_config = load_yaml(data_config_path)

    hp_names, hp_obsolete, hp_alt = load_hp_resources()
    mondo_names, mondo_obsolete = load_mondo_names()
    static = load_static_indices(train_config)
    tables = load_eval_tables(data_config, train_config)

    inventory = build_inventory(run_dir, report_dir, output_dir)
    metrics, final_metric_sources = summarize_metrics(run_dir)
    candidate_recall, candidate_case = summarize_candidate_recall(run_dir)
    quality, top_hpo, quality_case = summarize_dataset_quality(tables, static, hp_names, hp_obsolete, mondo_obsolete)
    failure_cases = build_failure_cases(run_dir, quality_case, mondo_names)
    gap = build_gap_table(quality, candidate_recall, metrics)

    write_csv(inventory, tables_dir / "repository_inventory.csv")
    write_csv(metrics, tables_dir / "current_metrics_summary.csv")
    write_csv(final_metric_sources, tables_dir / "final_metrics_with_sources.csv")
    write_csv(candidate_recall, tables_dir / "candidate_recall_summary.csv")
    write_csv(candidate_case, output_dir / "candidate_recall_case_level.csv")
    write_csv(quality, tables_dir / "dataset_quality_summary.csv")
    write_csv(top_hpo, tables_dir / "top_hpo_by_dataset.csv")
    write_csv(quality_case, output_dir / "dataset_quality_case_level.csv")
    write_csv(failure_cases, tables_dir / "failure_cases_mimic.csv")
    write_csv(gap, tables_dir / "ddd_vs_mimic_gap.csv")

    manifest = {
        "audit_script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "data_config_path": str(data_config_path.resolve()),
        "train_config_path": str(stage2_config_path.resolve()),
        "static_resources": {
            "hpo_index_path": str(static["hpo_index_path"]),
            "disease_index_path": str(static["disease_index_path"]),
            "incidence_path": str(static["incidence_path"]),
            "H_disease_shape": static["H_disease_shape"],
            "H_disease_nnz": static["H_disease_nnz"],
            "hpo_obsolete_count": len(hp_obsolete),
            "hpo_alt_id_count": len(hp_alt),
            "mondo_obsolete_count": len(mondo_obsolete),
        },
        "git": git_state(),
        "tables": {
            "repository_inventory": str((tables_dir / "repository_inventory.csv").resolve()),
            "current_metrics_summary": str((tables_dir / "current_metrics_summary.csv").resolve()),
            "candidate_recall_summary": str((tables_dir / "candidate_recall_summary.csv").resolve()),
            "dataset_quality_summary": str((tables_dir / "dataset_quality_summary.csv").resolve()),
            "top_hpo_by_dataset": str((tables_dir / "top_hpo_by_dataset.csv").resolve()),
            "ddd_vs_mimic_gap": str((tables_dir / "ddd_vs_mimic_gap.csv").resolve()),
            "failure_cases_mimic": str((tables_dir / "failure_cases_mimic.csv").resolve()),
        },
        "notes": [
            "This script reads existing data, configs, checkpoints metadata, result CSV/JSON, and candidate CSV files.",
            "It writes only to reports/current_framework_accuracy_audit and outputs/current_framework_accuracy_audit by default.",
            "It does not train, modify source data, overwrite mainline outputs, or mutate checkpoints.",
        ],
    }
    write_json(manifest, report_dir / "audit_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
