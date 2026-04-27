from __future__ import annotations

import ast
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "diagnosis"
RERANK_REPORT_DIR = PROJECT_ROOT / "reports" / "rerank"
RERANK_CONFIG_DIR = PROJECT_ROOT / "configs" / "rerank"
EXPERIMENT_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"

TEST_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
VAL_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_validation.csv"
BASELINE_RANKS = PROJECT_ROOT / "outputs" / "rerank" / "top50_rerank_case_ranks.csv"
BASELINE_METRICS = PROJECT_ROOT / "outputs" / "rerank" / "top50_rerank_metrics.csv"
FIXED_RERANK_METRICS = PROJECT_ROOT / "outputs" / "rerank" / "rerank_fixed_test_metrics.csv"
DDD_FIXED_RERANK_METRICS = PROJECT_ROOT / "outputs" / "rerank" / "ddd_rerank_fixed_test_metrics.csv"
VAL_SELECTED_WEIGHTS = PROJECT_ROOT / "outputs" / "rerank" / "val_selected_weights.json"
DDD_VAL_SELECTED_WEIGHTS = PROJECT_ROOT / "outputs" / "rerank" / "ddd_val_selected_grid_weights.json"
EXACT_DETAILS = (
    PROJECT_ROOT
    / "outputs"
    / "attn_beta_sweep"
    / "edge_log_beta02"
    / "evaluation"
    / "best_20260425_224439_details.csv"
)
EXACT_PER_DATASET = EXACT_DETAILS.with_name("best_20260425_224439_per_dataset.csv")

DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "train_finetune_attn_idf_main.yaml"
PRETRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_pretrain.yaml"
ONTOLOGY_HN_CONFIG = PROJECT_ROOT / "configs" / "train_finetune_ontology_hn.yaml"

MONDO_JSON = PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json"
HPO_OBO = PROJECT_ROOT / "data" / "raw_data" / "hp-base.obo"
HYPEREDGE_CSV = (
    PROJECT_ROOT
    / "LLLdataset"
    / "DiseaseHy"
    / "rare_disease_hgnn_clean_package_v59"
    / "v59_hyperedge_weighted_patched.csv"
)
DISEASE_INDEX = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx"
HPO_INDEX = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "HPO_index_v4.xlsx"
INCIDENCE_NPZ = (
    PROJECT_ROOT
    / "LLLdataset"
    / "DiseaseHy"
    / "rare_disease_hgnn_clean_package_v59"
    / "v59DiseaseHy.npz"
)


def ensure_dirs() -> None:
    for path in [REPORT_DIR, RERANK_REPORT_DIR, RERANK_CONFIG_DIR, EXPERIMENT_CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def to_markdown(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 4) -> str:
    if df.empty:
        return "_无记录_"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}f}")
    view = view.fillna("").astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def dataset_alias(name: str) -> str:
    stem = Path(str(name)).stem
    if stem in {"mimic_test_recleaned_mondo_hpo_rows", "MIMIC-Rare", "mimic_test"}:
        return "mimic_test"
    return stem


def raw_case_id(case_id: str) -> str:
    return str(case_id).rsplit("::", 1)[-1]


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(
        TEST_CANDIDATES,
        dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str},
    )
    df["dataset_name"] = df["dataset_name"].map(dataset_alias)
    df["original_rank"] = pd.to_numeric(df["original_rank"], errors="coerce").astype(int)
    return df.sort_values(["case_id", "original_rank"], kind="stable").reset_index(drop=True)


def load_exact_details() -> pd.DataFrame:
    df = pd.read_csv(EXACT_DETAILS, dtype={"case_id": str, "dataset_name": str, "true_label": str})
    df["dataset_name"] = df["dataset_name"].map(dataset_alias)
    df["true_rank"] = pd.to_numeric(df["true_rank"], errors="coerce").astype(int)
    df["raw_case_id"] = df["case_id"].map(raw_case_id)
    return df


def build_case_ranks(candidates: pd.DataFrame, exact_details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in candidates.groupby("case_id", sort=False):
        group = group.sort_values("original_rank", kind="stable")
        gold_id = str(group["gold_id"].iloc[0])
        hits = group.loc[group["candidate_id"].astype(str) == gold_id, "original_rank"]
        pool_rank = int(hits.min()) if not hits.empty else 51
        rows.append(
            {
                "case_id": case_id,
                "raw_case_id": raw_case_id(case_id),
                "dataset_name": dataset_alias(str(group["dataset_name"].iloc[0])),
                "gold_id": gold_id,
                "pool_rank": pool_rank,
                "gold_in_top50": bool(pool_rank <= 50),
            }
        )
    rank_df = pd.DataFrame(rows)
    details = exact_details[["case_id", "true_rank", "pred_top1"]].copy()
    return rank_df.merge(details, on="case_id", how="left")


def metric_row(ranks: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(ranks, errors="coerce").dropna().to_numpy(dtype=int)
    n = int(arr.size)
    if n == 0:
        return {
            "num_cases": 0,
            "top1": np.nan,
            "top3": np.nan,
            "top5": np.nan,
            "top10": np.nan,
            "top50": np.nan,
            "median_rank": np.nan,
            "mean_rank": np.nan,
            "rank_le_50_count": 0,
            "rank_le_50_ratio": np.nan,
            "rank_gt_50_count": 0,
            "rank_gt_50_ratio": np.nan,
            "top50_rank_gt5_count": 0,
            "top50_rank_gt5_ratio": np.nan,
            "gold_not_in_candidate_pool_count": 0,
            "gold_not_in_candidate_pool_ratio": np.nan,
        }
    rank_le_50 = arr <= 50
    rank_gt_50 = arr > 50
    top50_gt5 = (arr > 5) & (arr <= 50)
    return {
        "num_cases": n,
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "top10": float(np.mean(arr <= 10)),
        "top50": float(np.mean(rank_le_50)),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
        "rank_le_50_count": int(rank_le_50.sum()),
        "rank_le_50_ratio": float(np.mean(rank_le_50)),
        "rank_gt_50_count": int(rank_gt_50.sum()),
        "rank_gt_50_ratio": float(np.mean(rank_gt_50)),
        "top50_rank_gt5_count": int(top50_gt5.sum()),
        "top50_rank_gt5_ratio": float(np.mean(top50_gt5)),
        "gold_not_in_candidate_pool_count": int(rank_gt_50.sum()),
        "gold_not_in_candidate_pool_ratio": float(np.mean(rank_gt_50)),
    }


def load_case_tables_from_config(config_path: Path, key: str) -> pd.DataFrame:
    config = read_yaml(config_path)
    paths = config.get("test_files") if key == "test_files" else config.get("paths", {}).get(key)
    if not isinstance(paths, list):
        raise ValueError(f"Missing list {key} in {config_path}")
    frames: list[pd.DataFrame] = []
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        df = read_table(path, dtype=str)
        required = {"case_id", "mondo_label", "hpo_id"}
        if not required.issubset(df.columns):
            continue
        view = df[list(required)].copy()
        view["dataset_name"] = dataset_alias(path.name)
        view["source_file"] = str(path)
        frames.append(view)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train_case_tables() -> pd.DataFrame:
    return load_case_tables_from_config(FINETUNE_CONFIG, "train_files")


def case_feature_table(test_rows: pd.DataFrame, train_rows: pd.DataFrame) -> pd.DataFrame:
    train_labels = set(train_rows["mondo_label"].dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for (dataset, cid), group in test_rows.groupby(["dataset_name", "case_id"], sort=False):
        labels = sorted(group["mondo_label"].dropna().astype(str).unique().tolist())
        hpos = sorted(group["hpo_id"].dropna().astype(str).unique().tolist())
        exact_gold = labels[0] if labels else ""
        rows.append(
            {
                "dataset_name": dataset_alias(str(dataset)),
                "raw_case_id": str(cid),
                "gold_labels": labels,
                "exact_gold_id": exact_gold,
                "gold_label_count": int(len(labels)),
                "hpo_ids": hpos,
                "hpo_count": int(len(hpos)),
                "seen_bucket": "seen" if exact_gold in train_labels else "unseen",
                "hpo_count_bucket": hpo_bucket(len(hpos)),
            }
        )
    return pd.DataFrame(rows)


def hpo_bucket(count: int) -> str:
    if count <= 3:
        return "1-3"
    if count <= 6:
        return "4-6"
    if count <= 10:
        return "7-10"
    return ">10"


def mondo_id_from_iri(iri: str) -> str | None:
    if "MONDO_" not in iri:
        return None
    tail = iri.rsplit("/", 1)[-1].replace("MONDO_", "MONDO:")
    if tail.startswith("MONDO:"):
        return tail
    return None


def hp_id_from_iri(iri: str) -> str | None:
    if "HP_" not in iri:
        return None
    tail = iri.rsplit("/", 1)[-1].replace("HP_", "HP:")
    if tail.startswith("HP:"):
        return tail
    return None


def load_mondo() -> dict[str, Any]:
    data = json.loads(MONDO_JSON.read_text(encoding="utf-8"))
    graph = data["graphs"][0]
    names: dict[str, str] = {}
    synonyms: dict[str, set[str]] = defaultdict(set)
    deprecated: set[str] = set()
    replacements: dict[str, str] = {}
    xref_to_mondos: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)

    for node in graph.get("nodes", []):
        mondo = mondo_id_from_iri(str(node.get("id", "")))
        if not mondo:
            continue
        if node.get("lbl"):
            names[mondo] = str(node["lbl"])
        meta = node.get("meta", {}) or {}
        if meta.get("deprecated"):
            deprecated.add(mondo)
        for item in meta.get("basicPropertyValues", []) or []:
            pred = str(item.get("pred", ""))
            val = str(item.get("val", ""))
            replacement = mondo_id_from_iri(val)
            if pred.endswith("IAO_0100001") and replacement:
                replacements[mondo] = replacement
        for syn in meta.get("synonyms", []) or []:
            val = str(syn.get("val", "")).strip().lower()
            if val:
                synonyms[mondo].add(val)
        for xref in meta.get("xrefs", []) or []:
            val = str(xref.get("val", "")).strip()
            if val:
                xref_to_mondos[val].add(mondo)

    for edge in graph.get("edges", []) or []:
        if edge.get("pred") != "is_a":
            continue
        child = mondo_id_from_iri(str(edge.get("sub", "")))
        parent = mondo_id_from_iri(str(edge.get("obj", "")))
        if child and parent:
            parents[child].add(parent)

    @lru_cache(maxsize=None)
    def ancestors(mondo: str) -> frozenset[str]:
        out: set[str] = set()
        for parent in parents.get(mondo, set()):
            out.add(parent)
            out.update(ancestors(parent))
        return frozenset(out)

    one_to_many = {xref: sorted(vals) for xref, vals in xref_to_mondos.items() if len(vals) > 1}
    return {
        "names": names,
        "synonyms": synonyms,
        "deprecated": deprecated,
        "replacements": replacements,
        "parents": parents,
        "ancestors": ancestors,
        "xref_one_to_many": one_to_many,
    }


def load_hpo() -> dict[str, Any]:
    parents: dict[str, set[str]] = defaultdict(set)
    obsolete: set[str] = set()
    replacements: dict[str, str] = {}
    alt_to_id: dict[str, str] = {}
    current: dict[str, Any] | None = None

    def flush(term: dict[str, Any] | None) -> None:
        if not term or not term.get("id"):
            return
        tid = str(term["id"])
        if term.get("obsolete"):
            obsolete.add(tid)
        if term.get("replaced_by"):
            replacements[tid] = str(term["replaced_by"])
        for alt in term.get("alt_ids", []):
            alt_to_id[str(alt)] = tid
        for parent in term.get("parents", []):
            parents[tid].add(str(parent))

    for raw in HPO_OBO.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line == "[Term]":
            flush(current)
            current = {"alt_ids": [], "parents": []}
        elif current is None:
            continue
        elif line.startswith("id: "):
            current["id"] = line[4:].strip()
        elif line.startswith("alt_id: "):
            current["alt_ids"].append(line[8:].strip())
        elif line.startswith("is_obsolete: true"):
            current["obsolete"] = True
        elif line.startswith("replaced_by: "):
            current["replaced_by"] = line[13:].strip()
        elif line.startswith("is_a: "):
            current["parents"].append(line[6:].split(" ! ", 1)[0].strip())
    flush(current)

    @lru_cache(maxsize=None)
    def ancestors(hpo: str) -> frozenset[str]:
        out: set[str] = set()
        for parent in parents.get(hpo, set()):
            out.add(parent)
            out.update(ancestors(parent))
        return frozenset(out)

    return {
        "parents": parents,
        "obsolete": obsolete,
        "replacements": replacements,
        "alt_to_id": alt_to_id,
        "ancestors": ancestors,
    }


def load_indices_and_hyperedges() -> dict[str, Any]:
    disease_index = pd.read_excel(DISEASE_INDEX, dtype={"mondo_id": str})
    hpo_index = pd.read_excel(HPO_INDEX, dtype={"hpo_id": str})
    hyper = pd.read_csv(HYPEREDGE_CSV, dtype={"mondo_id": str, "hpo_id": str})
    disease_hpos = {
        mondo: set(group["hpo_id"].dropna().astype(str).unique().tolist())
        for mondo, group in hyper.groupby("mondo_id", sort=False)
    }
    hpo_df = hyper[["mondo_id", "hpo_id"]].drop_duplicates()
    hpo_disease_counts = hpo_df.groupby("hpo_id")["mondo_id"].nunique()
    n_disease = max(1, int(hyper["mondo_id"].nunique()))
    specificity = {
        str(hpo): float(math.log((1.0 + n_disease) / (1.0 + count)) + 1.0)
        for hpo, count in hpo_disease_counts.items()
    }
    npz = np.load(INCIDENCE_NPZ, allow_pickle=True)
    return {
        "disease_index": disease_index,
        "hpo_index": hpo_index,
        "disease_ids": set(disease_index["mondo_id"].astype(str)),
        "hpo_ids": set(hpo_index["hpo_id"].astype(str)),
        "disease_hpos": disease_hpos,
        "hpo_specificity": specificity,
        "npz_shape": tuple(int(x) for x in npz["shape"].tolist()),
    }


def overlap_metrics(
    case_hpos: set[str],
    disease_hpos: set[str],
    specificity: dict[str, float],
    hpo: dict[str, Any],
) -> dict[str, Any]:
    shared = case_hpos & disease_hpos
    denom = math.sqrt(max(1, len(case_hpos)) * max(1, len(disease_hpos)))
    case_ic_total = sum(float(specificity.get(h, 1.0)) for h in case_hpos)
    shared_ic = sum(float(specificity.get(h, 1.0)) for h in shared)
    semantic_shared = set(shared)
    disease_ancestor_union = set().union(*(set(hpo["ancestors"](dh)) for dh in disease_hpos)) if disease_hpos else set()
    for ch in case_hpos:
        ch_anc = set(hpo["ancestors"](ch))
        if ch in disease_ancestor_union or ch_anc & disease_hpos:
            semantic_shared.add(ch)
    return {
        "case_hpo_count": int(len(case_hpos)),
        "disease_hpo_count": int(len(disease_hpos)),
        "shared_hpo_count": int(len(shared)),
        "exact_overlap": float(len(shared) / denom) if denom else 0.0,
        "ic_weighted_overlap": float(shared_ic / case_ic_total) if case_ic_total > 0 else 0.0,
        "semantic_overlap": float(len(semantic_shared) / max(1, len(case_hpos))),
        "mean_case_hpo_ic": float(case_ic_total / max(1, len(case_hpos))),
    }


def relation_to_gold(candidate: str, gold: str, mondo: dict[str, Any]) -> str:
    if candidate == gold:
        return "same_disease"
    cand_name = mondo["names"].get(candidate, "").lower()
    gold_name = mondo["names"].get(gold, "").lower()
    if cand_name and (cand_name == gold_name or cand_name in mondo["synonyms"].get(gold, set())):
        return "synonym_or_name_match"
    if gold_name and gold_name in mondo["synonyms"].get(candidate, set()):
        return "synonym_or_name_match"
    cand_anc = set(mondo["ancestors"](candidate))
    gold_anc = set(mondo["ancestors"](gold))
    if candidate in gold_anc:
        return "candidate_ancestor_of_gold"
    if gold in cand_anc:
        return "candidate_descendant_of_gold"
    if mondo["parents"].get(candidate, set()) & mondo["parents"].get(gold, set()):
        return "same_parent"
    if cand_anc & gold_anc:
        return "shared_ancestor"
    return "unrelated_or_unknown"


def build_dataset_rank_decomposition(case_ranks: pd.DataFrame, case_features: pd.DataFrame) -> pd.DataFrame:
    merged = case_ranks.merge(case_features, on=["dataset_name", "raw_case_id"], how="left")
    rows: list[dict[str, Any]] = []
    for dataset, group in merged.groupby("dataset_name", sort=True):
        rows.append({"dataset": dataset, "bucket_type": "all", "bucket": "ALL", **metric_row(group["true_rank"])})
        for bucket, sub in group.groupby("seen_bucket", dropna=False):
            rows.append({"dataset": dataset, "bucket_type": "seen_disease", "bucket": str(bucket), **metric_row(sub["true_rank"])})
        for bucket, sub in group.groupby("hpo_count_bucket", dropna=False):
            rows.append({"dataset": dataset, "bucket_type": "hpo_count", "bucket": str(bucket), **metric_row(sub["true_rank"])})
    rows.append({"dataset": "ALL", "bucket_type": "all", "bucket": "ALL", **metric_row(merged["true_rank"])})
    return pd.DataFrame(rows)


def write_dataset_rank_report(df: pd.DataFrame) -> None:
    df.to_csv(REPORT_DIR / "dataset_rank_decomposition.csv", index=False, encoding="utf-8-sig")
    main = df[(df["bucket_type"] == "all") & (df["bucket"] == "ALL")].copy()
    lines = [
        "# Dataset-level Rank Decomposition",
        "",
        "本表使用 `outputs/rerank/top50_candidates_v2.csv` 与对应 exact evaluation 明细；`top50` 表示 gold 是否进入 HGNN top50 candidate pool。",
        "",
        "## Overall",
        to_markdown(
            main[
                [
                    "dataset",
                    "num_cases",
                    "top1",
                    "top3",
                    "top5",
                    "top10",
                    "top50",
                    "median_rank",
                    "mean_rank",
                    "rank_gt_50_count",
                    "rank_gt_50_ratio",
                    "top50_rank_gt5_count",
                    "top50_rank_gt5_ratio",
                ]
            ]
        ),
        "",
        "## Seen / Unseen 与 HPO 数量分桶",
        "详见 `dataset_rank_decomposition.csv` 中 `bucket_type=seen_disease` 和 `bucket_type=hpo_count` 行。",
    ]
    write_md(REPORT_DIR / "dataset_rank_decomposition.md", lines)


def build_mimic_audits(
    candidates: pd.DataFrame,
    case_ranks: pd.DataFrame,
    case_features: pd.DataFrame,
    resources: dict[str, Any],
    mondo: dict[str, Any],
    hpo: dict[str, Any],
) -> None:
    mimic_features = case_features[case_features["dataset_name"] == "mimic_test"].copy()
    mimic_ranks = case_ranks[case_ranks["dataset_name"] == "mimic_test"].copy()
    cand = candidates[candidates["dataset_name"] == "mimic_test"].copy()
    rank_map = dict(zip(mimic_ranks["raw_case_id"], mimic_ranks["true_rank"], strict=False))
    pred_map = dict(zip(mimic_ranks["raw_case_id"], mimic_ranks["pred_top1"], strict=False))
    rows: list[dict[str, Any]] = []
    candidate_by_case = {raw_case_id(k): g for k, g in cand.groupby("case_id", sort=False)}
    for rec in mimic_features.itertuples(index=False):
        labels = list(rec.gold_labels)
        group = candidate_by_case.get(str(rec.raw_case_id), pd.DataFrame())
        any_rank = 51
        if not group.empty:
            hit_ranks = group.loc[group["candidate_id"].astype(str).isin(set(labels)), "original_rank"]
            if not hit_ranks.empty:
                any_rank = int(hit_ranks.min())
        exact_gold = str(rec.exact_gold_id)
        disease_hpos = resources["disease_hpos"].get(exact_gold, set())
        overlap = overlap_metrics(set(rec.hpo_ids), disease_hpos, resources["hpo_specificity"], hpo)
        rows.append(
            {
                "case_id": str(rec.raw_case_id),
                "gold_labels": "|".join(labels),
                "gold_label_count": int(rec.gold_label_count),
                "label_subset": "multi_label" if int(rec.gold_label_count) > 1 else "single_label",
                "exact_gold_id": exact_gold,
                "exact_rank": int(rank_map.get(str(rec.raw_case_id), 9999)),
                "any_label_rank": int(any_rank),
                "any_label_hit_at_1": bool(any_rank <= 1),
                "any_label_hit_at_3": bool(any_rank <= 3),
                "any_label_hit_at_5": bool(any_rank <= 5),
                "any_label_hit_at_50": bool(any_rank <= 50),
                "hpo_count": int(rec.hpo_count),
                "seen_bucket": str(rec.seen_bucket),
                "pred_top1": pred_map.get(str(rec.raw_case_id), ""),
                "obsolete_gold": bool(exact_gold in mondo["deprecated"]),
                "replacement_mondo_id": mondo["replacements"].get(exact_gold, ""),
                "unmapped_gold": bool(exact_gold not in resources["disease_ids"]),
                "gold_hyperedge_missing": bool(exact_gold not in resources["disease_hpos"]),
                **overlap,
            }
        )
    audit = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for subset, sub in [
        ("multi_label", audit[audit["gold_label_count"] > 1]),
        ("single_label", audit[audit["gold_label_count"] == 1]),
        ("ALL", audit),
    ]:
        ranks = sub["exact_rank"].to_numpy(dtype=int)
        any_ranks = sub["any_label_rank"].to_numpy(dtype=int)
        summary_rows.append(
            {
                "dataset": "mimic_test",
                "subset": subset,
                "num_cases": int(len(sub)),
                "mean_gold_label_count": float(sub["gold_label_count"].mean()) if len(sub) else np.nan,
                "median_gold_label_count": float(sub["gold_label_count"].median()) if len(sub) else np.nan,
                "exact_top1": float(np.mean(ranks <= 1)) if len(ranks) else np.nan,
                "exact_top3": float(np.mean(ranks <= 3)) if len(ranks) else np.nan,
                "exact_top5": float(np.mean(ranks <= 5)) if len(ranks) else np.nan,
                "exact_top50": float(np.mean(ranks <= 50)) if len(ranks) else np.nan,
                "any_label_hit_at_1": float(np.mean(any_ranks <= 1)) if len(any_ranks) else np.nan,
                "any_label_hit_at_3": float(np.mean(any_ranks <= 3)) if len(any_ranks) else np.nan,
                "any_label_hit_at_5": float(np.mean(any_ranks <= 5)) if len(any_ranks) else np.nan,
                "any_label_hit_at_50": float(np.mean(any_ranks <= 50)) if len(any_ranks) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "mimic_multilabel_audit.csv", index=False, encoding="utf-8-sig")

    ic_q25 = float(audit["mean_case_hpo_ic"].quantile(0.25)) if not audit.empty else 0.0

    def buckets(row: pd.Series) -> str:
        out: list[str] = []
        if int(row["hpo_count"]) <= 3:
            out.append("HPO_too_few")
        if float(row["mean_case_hpo_ic"]) <= ic_q25:
            out.append("HPO_too_generic")
        if int(row["shared_hpo_count"]) == 0:
            out.append("overlap=0")
        if bool(row["unmapped_gold"]):
            out.append("unmapped_disease")
        if bool(row["obsolete_gold"]):
            out.append("obsolete_disease")
        if int(row["gold_label_count"]) > 1 and int(row["exact_rank"]) > 50 and int(row["any_label_rank"]) <= 50:
            out.append("multi-label_misevaluation")
        if row["seen_bucket"] == "unseen":
            out.append("unseen_disease")
        return "|".join(out) if out else "other"

    misses = audit[audit["exact_rank"] > 50].copy()
    misses["failure_buckets"] = misses.apply(buckets, axis=1)
    misses.to_csv(REPORT_DIR / "mimic_top50_miss_audit.csv", index=False, encoding="utf-8-sig")

    bucket_counts = Counter()
    for text in misses["failure_buckets"].fillna("other"):
        for bucket in str(text).split("|"):
            bucket_counts[bucket] += 1
    bucket_df = pd.DataFrame(
        [
            {
                "bucket": bucket,
                "num_cases": count,
                "ratio_among_rank_gt50": count / max(1, len(misses)),
            }
            for bucket, count in bucket_counts.most_common()
        ]
    )
    lines = [
        "# mimic_test Failure Bucket Summary",
        "",
        "## Multi-label Summary",
        to_markdown(summary),
        "",
        "## Rank > 50 Buckets",
        f"- rank>50 cases: {len(misses)}",
        f"- HPO generic threshold: mean_case_hpo_ic <= {ic_q25:.4f}",
        to_markdown(bucket_df),
        "",
        "## 结论",
        "- `any-label@k` 已单独输出，未覆盖原始 exact metric。",
        "- multi-label 病例的 any-label 命中明显高于 exact，说明 mimic_test 存在 formal exact 低估风险，但只能作为 supplementary。",
        "- rank>50 中 `overlap=0` 和低信息量 HPO 是主要候选召回瓶颈，单纯 top50 rerank 无法解决这部分样本。",
    ]
    write_md(REPORT_DIR / "mimic_failure_bucket_summary.md", lines)


def build_ddd_reports(
    candidates: pd.DataFrame,
    case_ranks: pd.DataFrame,
    case_features: pd.DataFrame,
    resources: dict[str, Any],
    mondo: dict[str, Any],
) -> None:
    ddd = case_ranks[(case_ranks["dataset_name"] == "DDD") & (case_ranks["true_rank"] > 1) & (case_ranks["true_rank"] <= 50)].copy()
    ddd_features = case_features[case_features["dataset_name"] == "DDD"].set_index("raw_case_id")
    rows: list[dict[str, Any]] = []
    pair_counter: Counter[tuple[str, str, str]] = Counter()
    above_gold_counts: list[int] = []
    for rec in ddd.itertuples(index=False):
        group = candidates[candidates["case_id"] == rec.case_id].sort_values("original_rank", kind="stable")
        if group.empty:
            continue
        gold = str(rec.gold_id)
        top1 = str(group.iloc[0]["candidate_id"])
        top10 = group.head(10).copy()
        top_ids = top10["candidate_id"].astype(str).tolist()
        relations = [relation_to_gold(cid, gold, mondo) for cid in top_ids]
        gold_hpos = resources["disease_hpos"].get(gold, set())
        shared_counts = []
        jaccards = []
        for cid in top_ids:
            cand_hpos = resources["disease_hpos"].get(cid, set())
            shared = gold_hpos & cand_hpos
            union = gold_hpos | cand_hpos
            shared_counts.append(len(shared))
            jaccards.append(float(len(shared) / len(union)) if union else 0.0)
        above = max(0, int(rec.true_rank) - 1)
        above_gold_counts.append(above)
        relation = relation_to_gold(top1, gold, mondo)
        pair_counter[(gold, top1, relation)] += 1
        feature = ddd_features.loc[rec.raw_case_id] if rec.raw_case_id in ddd_features.index else None
        rows.append(
            {
                "case_id": rec.case_id,
                "raw_case_id": rec.raw_case_id,
                "gold_disease_id": gold,
                "gold_disease_name": mondo["names"].get(gold, ""),
                "gold_rank": int(rec.true_rank),
                "top50_above_gold_negative_count": above,
                "top1_candidate_id": top1,
                "top1_candidate_name": mondo["names"].get(top1, ""),
                "top1_relation_to_gold": relation,
                "top10_candidate_ids": "|".join(top_ids),
                "top10_candidate_names": "|".join(mondo["names"].get(cid, "") for cid in top_ids),
                "top10_relations_to_gold": "|".join(relations),
                "top10_gold_hpo_shared_counts": "|".join(str(v) for v in shared_counts),
                "top10_gold_hpo_jaccards": "|".join(f"{v:.4f}" for v in jaccards),
                "case_hpo_count": int(feature["hpo_count"]) if feature is not None else np.nan,
                "gold_disease_hpo_count": int(len(gold_hpos)),
            }
        )
    near = pd.DataFrame(rows)
    near.to_csv(REPORT_DIR / "ddd_nearmiss_cases.csv", index=False, encoding="utf-8-sig")

    pairs = []
    for (gold, pred, relation), count in pair_counter.most_common():
        subset = near[(near["gold_disease_id"] == gold) & (near["top1_candidate_id"] == pred)]
        pairs.append(
            {
                "gold_disease_id": gold,
                "gold_disease_name": mondo["names"].get(gold, ""),
                "wrong_top1_disease_id": pred,
                "wrong_top1_disease_name": mondo["names"].get(pred, ""),
                "ontology_relation": relation,
                "confusion_count": int(count),
                "mean_gold_rank": float(subset["gold_rank"].mean()) if not subset.empty else np.nan,
                "mean_top50_above_gold_negative_count": float(subset["top50_above_gold_negative_count"].mean()) if not subset.empty else np.nan,
            }
        )
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(REPORT_DIR / "ddd_confusion_pairs.csv", index=False, encoding="utf-8-sig")

    relation_counts = near["top1_relation_to_gold"].value_counts().rename_axis("relation").reset_index(name="num_cases")
    recommendation = [
        "# DDD Hard Negative Recommendation",
        "",
        f"- top1 错误但 gold 在 top50 的 near-miss cases: {len(near)}",
        f"- top50 中排在 gold 前面的 negative 平均数: {float(np.mean(above_gold_counts)) if above_gold_counts else 0.0:.2f}",
        "",
        "## Top1 Wrong Relation Distribution",
        to_markdown(relation_counts),
        "",
        "## 推荐 negative 类型",
        "- `top50-above-gold negative`: 优先级最高，直接对应当前 DDD gold 被压在 top50 内的问题。",
        "- `high HPO-overlap negative`: 适合提升 top1/top3 排序，但需要避免把语义等价或合理鉴别诊断当成强负例。",
        "- `MONDO sibling/same-parent negative`: DDD 中常见同父类混淆，适合做 margin 较小的 hard negative。",
        "- `current top-k negative`: 保留为基础组，作为 HN-current 对照。",
        "- `mixed hard negative`: 论文实验最完整，但必须先在训练 loop 中真正构造并传入 ontology/overlap/top50 candidate pools。",
        "",
        "## False Negative 风险",
        "same-parent、sibling、ancestor-descendant 候选可能是临床上合理的近似诊断；建议降低 margin 或使用 soft label/低权重，不建议无差别强惩罚。",
    ]
    write_md(REPORT_DIR / "ddd_hard_negative_recommendation.md", recommendation)


def build_mapping_audit(test_rows: pd.DataFrame, train_rows: pd.DataFrame, resources: dict[str, Any], mondo: dict[str, Any], hpo: dict[str, Any]) -> None:
    all_rows = pd.concat([test_rows.assign(split="test"), train_rows.assign(split="train")], ignore_index=True)
    issues: list[dict[str, Any]] = []
    for (split, dataset), group in all_rows.groupby(["split", "dataset_name"], sort=True):
        disease_labels = set(group["mondo_label"].dropna().astype(str))
        hpo_labels = set(group["hpo_id"].dropna().astype(str))
        for label in sorted(disease_labels):
            if label not in resources["disease_ids"]:
                issues.append({"split": split, "dataset": dataset, "issue_type": "disease_not_in_Disease_index", "entity_id": label, "detail": ""})
            if label in mondo["deprecated"]:
                issues.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "issue_type": "obsolete_MONDO",
                        "entity_id": label,
                        "detail": mondo["replacements"].get(label, ""),
                    }
                )
        for hpo_id in sorted(hpo_labels):
            if hpo_id not in resources["hpo_ids"]:
                issue_type = "hpo_alt_id_not_normalized" if hpo_id in hpo["alt_to_id"] else "hpo_not_in_HPO_index"
                issues.append({"split": split, "dataset": dataset, "issue_type": issue_type, "entity_id": hpo_id, "detail": hpo["alt_to_id"].get(hpo_id, "")})
            if hpo_id in hpo["obsolete"]:
                issues.append({"split": split, "dataset": dataset, "issue_type": "obsolete_HPO", "entity_id": hpo_id, "detail": hpo["replacements"].get(hpo_id, "")})

    disease_shape_ok = resources["npz_shape"][1] == len(resources["disease_index"])
    hpo_shape_ok = resources["npz_shape"][0] == len(resources["hpo_index"])
    issues.append(
        {
            "split": "resource",
            "dataset": "DiseaseHy_v59",
            "issue_type": "npz_disease_dimension_check",
            "entity_id": str(resources["npz_shape"][1]),
            "detail": f"Disease_index rows={len(resources['disease_index'])}; ok={disease_shape_ok}",
        }
    )
    issues.append(
        {
            "split": "resource",
            "dataset": "DiseaseHy_v59",
            "issue_type": "npz_hpo_dimension_check",
            "entity_id": str(resources["npz_shape"][0]),
            "detail": f"HPO_index rows={len(resources['hpo_index'])}; ok={hpo_shape_ok}",
        }
    )
    issues.append(
        {
            "split": "resource",
            "dataset": "MONDO",
            "issue_type": "xref_one_to_many_mapping_count",
            "entity_id": str(len(mondo["xref_one_to_many"])),
            "detail": "OMIM/ORPHA/ICD 等 xref 映射到多个 MONDO 的数量；需要人工规则消歧。",
        }
    )
    out = pd.DataFrame(issues)
    out.to_csv(REPORT_DIR / "mondo_hpo_mapping_audit.csv", index=False, encoding="utf-8-sig")

    summary = out["issue_type"].value_counts().rename_axis("issue_type").reset_index(name="count")
    lines = [
        "# Mapping Issue Summary",
        "",
        "## Issue Counts",
        to_markdown(summary),
        "",
        "## 关键判断",
        f"- `Disease_index_v4.xlsx` 行数与 `v59DiseaseHy.npz` disease 维度一致: {disease_shape_ok}",
        f"- `HPO_index_v4.xlsx` 行数与 `v59DiseaseHy.npz` HPO 维度一致: {hpo_shape_ok}",
        "- processed dataset 的 disease/HPO 映射问题见 `mondo_hpo_mapping_audit.csv`。",
        "- MONDO xref 存在 one-to-many 风险，OMIM/ORPHA/ICD 到 MONDO 的自动映射不应无人工规则直接用于 exact gold 改写。",
        "- synonym 或 parent-child 命中可以解释 near miss，但不能混入正式 exact evaluation。",
    ]
    write_md(REPORT_DIR / "mapping_issue_summary.md", lines)


def yaml_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_rerank_reports() -> None:
    selected = json.loads(VAL_SELECTED_WEIGHTS.read_text(encoding="utf-8")) if VAL_SELECTED_WEIGHTS.is_file() else {}
    best_yaml = {
        "protocol": "validation_selected_fixed_test",
        "selection_source": str(VAL_SELECTED_WEIGHTS),
        "validation_candidates_path": str(VAL_CANDIDATES),
        "test_candidates_path": str(TEST_CANDIDATES),
        "selection_objective": selected.get("selection_objective", "ALL_top1"),
        "selected_preset": selected.get("selected_preset"),
        "weights": selected.get("weights", {}),
        "gate": selected.get("gate"),
        "fixed_test_only": True,
        "do_not_tune_on_test": True,
    }
    yaml_dump(RERANK_CONFIG_DIR / "best_val_selected_rerank.yaml", best_yaml)

    fixed = pd.read_csv(FIXED_RERANK_METRICS) if FIXED_RERANK_METRICS.is_file() else pd.DataFrame()
    ddd_fixed = pd.read_csv(DDD_FIXED_RERANK_METRICS) if DDD_FIXED_RERANK_METRICS.is_file() else pd.DataFrame()
    lines = [
        "# Fixed Test Rerank Result",
        "",
        "## Protocol",
        "- validation candidates exported separately via `tools/export_top50_candidates.py --case-source validation`。",
        "- validation grid search is implemented in `tools/run_top50_evidence_rerank.py --protocol validation_select`。",
        "- fixed test evaluation is implemented in `tools/run_top50_evidence_rerank.py --protocol fixed_eval`。",
        "- 原始 exact evaluation 不覆盖；rerank 结果单独保存。",
        "",
        "## ALL-objective Fixed Test",
        to_markdown(fixed),
        "",
        "## DDD-objective Fixed Test / Exploratory Comparison",
        to_markdown(ddd_fixed),
        "",
        "## 论文可用性",
        "- `validation_selected_fixed_test` 可进入论文主表，前提是明确权重来自 validation。",
        "- `test_side_exploratory_upper_bound`、test-side grid/gate/search 只能作为附表或 error analysis，不能作为主表结果。",
    ]
    write_md(RERANK_REPORT_DIR / "fixed_test_result.md", lines)

    plan = [
        "# Rerank Validationization Plan",
        "",
        "## 当前状态",
        "- `tools/export_top50_candidates.py` 支持 `--case-source validation`，可导出 validation top50 candidates。",
        "- `tools/run_top50_evidence_rerank.py` 支持 `validation_select`、`fixed_eval`、`presets` 三种协议。",
        "- validation grid search、best config JSON 保存、fixed test evaluation 已具备。",
        "- 旧的 `outputs/rerank/rerank_v2_grid_results.csv` / `rerank_v2_gated_results.csv` 属于 test-side exploratory，不应作为正式 test 结果。",
        "",
        "## 缺口",
        "- 当前正式脚本只冻结 linear grid；gated/mimic-safe gate 如果要进主表，需要同样从 validation 选择阈值并固定 test 跑一次。",
        "- best config 以前主要保存为 JSON，本次补充生成 `configs/rerank/best_val_selected_rerank.yaml`。",
        "",
        "## 推荐表述",
        "- Main table: `HGNN exact baseline` 与 `validation-selected fixed-test rerank`。",
        "- Supplementary: test-side exploratory upper bound、gated ablation、any-label 或 relaxed diagnostics。",
    ]
    write_md(REPORT_DIR / "rerank_validationization_plan.md", plan)


def build_hard_negative_reports() -> None:
    base = read_yaml(FINETUNE_CONFIG)
    variants = {
        "hgnn_hn_current.yaml": ("HN-current", {"current": 1.0}),
        "hgnn_hn_overlap.yaml": ("HN-overlap", {"overlap": 1.0}),
        "hgnn_hn_sibling.yaml": ("HN-sibling", {"sibling": 1.0}),
        "hgnn_hn_top50_above_gold.yaml": ("HN-above-gold", {"above_gold": 1.0}),
        "hgnn_hn_mixed.yaml": (
            "HN-mixed",
            {"current": 0.40, "overlap": 0.20, "sibling": 0.15, "shared_ancestor": 0.15, "above_gold": 0.10},
        ),
    }
    for filename, (strategy, ratios) in variants.items():
        cfg = json.loads(json.dumps(base))
        cfg.setdefault("paths", {})["save_dir"] = str(PROJECT_ROOT / "outputs" / "experiments" / Path(filename).stem)
        hn = cfg.setdefault("loss", {}).setdefault("hard_negative", {})
        hn["use_hard_negative"] = True
        hn["strategy"] = strategy
        hn["sampling_ratios"] = ratios
        cfg["diagnosis_note"] = (
            "Ontology/overlap/top50 strategies need trainer-side candidate_pools. "
            "Without that code path, src.training.hard_negative_miner falls back to HN-current."
        )
        yaml_dump(EXPERIMENT_CONFIG_DIR / filename, cfg)

    lines = [
        "# Hard Negative Code Audit",
        "",
        "## 当前实现",
        "- `src/training/hard_negative_miner.py::mine_hard_negatives` 是 score-based current model top-k negative。",
        "- `mine_configurable_hard_negatives` 已有 `HN-overlap`、`HN-sibling`、`HN-above-gold`、`HN-mixed` 的接口和 fallback 逻辑。",
        "- 但 `src/training/trainer.py` 调用时没有传入 `candidate_pools`，因此除 `HN-current` 外的策略目前会退化为 current top-k negative。",
        "- 当前训练没有真正使用 ontology-aware negative、HPO-overlap negative、MONDO sibling/same-parent negative、top50-above-gold negative。",
        "",
        "## False Negative 风险",
        "- sibling/same-parent/ancestor-descendant 疾病可能是合理鉴别诊断或标注粒度差异，强负例可能损伤泛化。",
        "- 建议先以较小 margin/weight 使用 ontology negative，并输出 relation bucket 指标。",
        "",
        "## 最小改动方案",
        "1. 在数据预处理阶段生成 disease-level pools：`overlap_pool`、`sibling_pool`、`same_parent_pool`。",
        "2. 从 train/validation top50 candidates 生成 `above_gold_pool`，只用于训练/validation，不从 test 构造训练池。",
        "3. 在 `build_batch_hypergraph` 或 trainer batch 侧按 gold disease idx 取 pool，传给 `mine_configurable_hard_negatives(candidate_pools=...)`。",
        "4. 对 relation-aware negatives 使用较低权重或 soft margin，保留 `HN-current` 对照。",
        "",
        "## 建议实验组",
        "- `HGNN baseline`: 当前 exact baseline。",
        "- `HN-current`: current top-k negative 对照。",
        "- `HN-overlap`: 高 HPO-overlap negative。",
        "- `HN-sibling`: MONDO sibling/same-parent negative。",
        "- `HN-top50-above-gold`: validation/train top50 中排在 gold 前面的 negative。",
        "- `HN-mixed`: current + overlap + sibling + shared_ancestor + above_gold。",
        "- `HN-mixed + val-selected rerank`: 训练后固定 validation rerank 权重再 test 一次。",
        "",
        "## 已生成配置",
        "- `configs/experiments/hgnn_hn_current.yaml`",
        "- `configs/experiments/hgnn_hn_overlap.yaml`",
        "- `configs/experiments/hgnn_hn_sibling.yaml`",
        "- `configs/experiments/hgnn_hn_top50_above_gold.yaml`",
        "- `configs/experiments/hgnn_hn_mixed.yaml`",
        "",
        "注意：这些配置先作为独立实验占位；在 trainer 传入 candidate pools 之前，ontology-aware 组不能解释为真正的 ontology-aware hard negative。",
    ]
    write_md(REPORT_DIR / "hard_negative_code_audit.md", lines)


def build_framework_report() -> list[str]:
    return [
        "## 当前实验框架审计",
        "",
        "| 阶段 | 入口脚本 | 主要输入 | 主要输出 | 说明 |",
        "| --- | --- | --- | --- | --- |",
        f"| Pretrain | `python -m src.training.trainer --config {PRETRAIN_CONFIG.relative_to(PROJECT_ROOT)}` | `configs/train_pretrain.yaml` 中的 train_files、Disease_index、HPO_index、v59DiseaseHy | `outputs/stage1_pretrain_v59` | 使用 validation split 监控 `val_top1` 保存 best/last |",
        f"| Finetune | `python -m src.training.trainer --config {FINETUNE_CONFIG.relative_to(PROJECT_ROOT)}` | processed train、pretrain checkpoint、v59DiseaseHy | `outputs/attn_beta_sweep/edge_log_beta02` | 当前 mainline，监控 `val_real_macro_top5`，hard negative 为 current top-k |",
        f"| Exact eval | `python -m src.evaluation.evaluator --data_config_path {DATA_CONFIG.relative_to(PROJECT_ROOT)} --train_config_path {FINETUNE_CONFIG.relative_to(PROJECT_ROOT)}` | data eval config、finetune config、checkpoint | `outputs/attn_beta_sweep/edge_log_beta02/evaluation/*` | exact rank 明细和 per-dataset metrics；未与 relaxed 混合 |",
        "| Candidate export | `tools/export_top50_candidates.py` | checkpoint、test/validation case source | `outputs/rerank/top50_candidates*.csv` | 支持 test / train / validation candidate export |",
        "| Rerank | `tools/run_top50_evidence_rerank.py` | top50 candidates、validation candidates、weights JSON/YAML | `outputs/rerank/*`, `reports/rerank/*` | 支持 validation grid selection 与 fixed test eval |",
        "| Similar-case aug | `tools/run_mimic_similar_case_aug.py` | mimic top50 candidates、validation candidates | `reports/mimic_next/*` | 只能作为固定 protocol 或 exploratory，不能 test 调参 |",
        "",
        "### 关键风险",
        "- 旧 rerank v2/grid/gate 文件存在 test-side exploratory search，应标记为 upper bound/附表。",
        "- 当前 validation candidate 已存在并可用于选权重；正式结果应只加载固定权重 test 一次。",
        "- exact evaluation 由 `src.evaluation.evaluator` 的 `true_rank` 产生，relaxed/any-label/synonym/parent-child 只应作为 error analysis。",
        "- Disease_index、HPO_index、v59DiseaseHy 维度一致性见 `mapping_issue_summary.md`；不同 dataset 均通过同一 v4/v59 资源评估。",
    ]


def build_final_report(rank_df: pd.DataFrame) -> None:
    main = rank_df[(rank_df["bucket_type"] == "all") & (rank_df["bucket"] == "ALL")].copy()
    lookup = {row["dataset"]: row for _, row in main.iterrows()}
    lines = [
        "# Recommended Accuracy Improvement Plan",
        "",
        *build_framework_report(),
        "",
        "## 当前准确率低的主要原因分解",
        "- `DDD`: Recall@50 约 0.745，说明大量 gold 已在 top50 内但排序不足；适合 validation-selected rerank 和 hard negative training。",
        "- `mimic_test`: Top1 低且 Recall@50 约 0.615，rank>50 比例高；主要是 candidate recall、multi-label exact 低估、HPO/KB overlap 不足。",
        "- `ALL`: 被 DDD 和 mimic_test 两个大数据集主导，Top1 提升优先看这两个 dataset。",
        "- `HMS/LIRICAL`: case 数较小且与论文 split 可能不一致，应先确认协议 parity。",
        "",
        "## Dataset 瓶颈",
        to_markdown(
            main[
                [
                    "dataset",
                    "num_cases",
                    "top1",
                    "top3",
                    "top5",
                    "top50",
                    "rank_gt_50_ratio",
                    "top50_rank_gt5_ratio",
                ]
            ]
        ),
        "",
        "## 不需要训练即可提升或解释的问题",
        "- 固定 validation-selected rerank 可改善 DDD/LIRICAL/HMS 的 top-k，但必须与 test-side exploratory 区分。",
        "- mimic_test any-label、multi-label、synonym/parent-child 命中只能解释 exact 低估，不能覆盖 exact 主指标。",
        "- MONDO/HPO obsolete、replacement、alt_id、xref one-to-many 应做 mapping audit 和数据清洗。",
        "",
        "## 适合 no-train reranker 的问题",
        "- gold 已在 top50 但 rank>5 的 DDD/LIRICAL/HMS 样本。",
        "- HPO exact/IC/semantic overlap 强但 HGNN score 未排前的 near miss。",
        "- 不适合解决 gold 不在 candidate pool 的 mimic rank>50 样本。",
        "",
        "## 适合 hard negative training 的问题",
        "- DDD same-parent/shared-ancestor/top50-above-gold confusion。",
        "- HPO 高重叠 hard negative 造成的 top1 错误。",
        "- mixed HN 可作为主实验，但需要先补 trainer candidate_pools。",
        "",
        "## 需要数据清洗或 label mapping 修复的问题",
        "- mimic_test multi-label exact gold 选择规则需明确；any-label 只做 supplementary。",
        "- obsolete MONDO/HPO、replacement、alt_id、OMIM/ORPHA/ICD one-to-many 映射需单独修复并冻结版本。",
        "- HPO parent-child/synonym 可解释命中不能混入 exact evaluation。",
        "",
        "## 可进入论文主表",
        "- `HGNN exact baseline`。",
        "- `validation-selected fixed-test rerank`。",
        "- 完成 candidate_pools 后的 `HN-current/HN-overlap/HN-sibling/HN-top50-above-gold/HN-mixed` 独立训练结果。",
        "",
        "## 只能作为附表或 error analysis",
        "- test-side grid/gate/weight search upper bound。",
        "- mimic any-label hit@k、relaxed synonym/parent-child 命中。",
        "- mapping audit、near-miss case list、hard negative candidate 类型统计。",
        "",
        "## 推荐下一步实验顺序与预期",
        "1. 固定 rerank protocol：预期 DDD Top1 +0.03~0.07，ALL Top1 +0.005~0.02；mimic Top1 不一定提升。",
        "2. 修复并冻结 mapping/obsolete/alt_id：预期主要提升可解释性和 exact 可信度，Recall@50 小幅改善。",
        "3. mimic candidate recovery / similar-case aug：预期 mimic Recall@50 和 Top5 改善，需 validation 选 source weights。",
        "4. 实现 trainer candidate_pools 后跑 HN-current/overlap/sibling/top50-above-gold/mixed：预期 DDD Top1/Top3 提升，需监控 false negative。",
        "5. HN-mixed + val-selected rerank：作为最终主表候选，预期 DDD Top1 和 ALL Top1 叠加提升。",
    ]
    write_md(REPORT_DIR / "recommended_accuracy_improvement_plan.md", lines)


def main() -> None:
    ensure_dirs()
    candidates = load_candidates()
    exact_details = load_exact_details()
    case_ranks = build_case_ranks(candidates, exact_details)
    test_rows = load_case_tables_from_config(DATA_CONFIG, "test_files")
    train_rows = load_train_case_tables()
    case_features = case_feature_table(test_rows, train_rows)
    mondo = load_mondo()
    hpo = load_hpo()
    resources = load_indices_and_hyperedges()

    rank_decomp = build_dataset_rank_decomposition(case_ranks, case_features)
    write_dataset_rank_report(rank_decomp)
    build_mimic_audits(candidates, case_ranks, case_features, resources, mondo, hpo)
    build_ddd_reports(candidates, case_ranks, case_features, resources, mondo)
    build_mapping_audit(test_rows, train_rows, resources, mondo, hpo)
    build_rerank_reports()
    build_hard_negative_reports()
    build_final_report(rank_decomp)

    manifest = {
        "inputs": {
            "test_candidates": str(TEST_CANDIDATES),
            "validation_candidates": str(VAL_CANDIDATES),
            "exact_details": str(EXACT_DETAILS),
            "data_config": str(DATA_CONFIG),
            "finetune_config": str(FINETUNE_CONFIG),
            "mondo_json": str(MONDO_JSON),
            "hpo_obo": str(HPO_OBO),
            "hyperedge_csv": str(HYPEREDGE_CSV),
        },
        "outputs": [
            "reports/diagnosis/dataset_rank_decomposition.csv",
            "reports/diagnosis/dataset_rank_decomposition.md",
            "reports/diagnosis/mimic_multilabel_audit.csv",
            "reports/diagnosis/mimic_top50_miss_audit.csv",
            "reports/diagnosis/mimic_failure_bucket_summary.md",
            "reports/diagnosis/ddd_nearmiss_cases.csv",
            "reports/diagnosis/ddd_confusion_pairs.csv",
            "reports/diagnosis/ddd_hard_negative_recommendation.md",
            "reports/diagnosis/rerank_validationization_plan.md",
            "configs/rerank/best_val_selected_rerank.yaml",
            "reports/rerank/fixed_test_result.md",
            "reports/diagnosis/hard_negative_code_audit.md",
            "configs/experiments/hgnn_hn_current.yaml",
            "configs/experiments/hgnn_hn_overlap.yaml",
            "configs/experiments/hgnn_hn_sibling.yaml",
            "configs/experiments/hgnn_hn_top50_above_gold.yaml",
            "configs/experiments/hgnn_hn_mixed.yaml",
            "reports/diagnosis/mondo_hpo_mapping_audit.csv",
            "reports/diagnosis/mapping_issue_summary.md",
            "reports/diagnosis/recommended_accuracy_improvement_plan.md",
        ],
    }
    (REPORT_DIR / "accuracy_diagnosis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
