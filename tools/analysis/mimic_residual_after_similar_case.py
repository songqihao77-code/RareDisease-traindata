from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_mimic_similar_case_aug import (
    compute_similar_matches,
    hgnn_source,
    load_candidates,
    load_case_tables,
    load_yaml_config,
    similar_source,
)


REPORT_DIR = PROJECT_ROOT / "reports" / "mimic_residual_after_similar_case"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mimic_residual_after_similar_case"

DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
TRAIN_CONFIG = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "configs" / "stage2_finetune.yaml"
MAINLINE_CONFIG = PROJECT_ROOT / "configs" / "mainline_full_pipeline.yaml"

FINAL_METRICS = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "mainline_final_metrics.csv"
FINAL_METRICS_WITH_SOURCES = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "mainline_final_metrics_with_sources.csv"
FINAL_CASE_RANKS = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "mainline_final_case_ranks.csv"
RUN_MANIFEST = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "run_manifest.json"

TEST_CANDIDATES = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage4_candidates" / "top50_candidates_test.csv"
VAL_CANDIDATES = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage4_candidates" / "top50_candidates_validation.csv"
SIMILAR_FIXED = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage6_mimic_similar_case" / "similar_case_fixed_test.csv"
SIMILAR_SELECTION = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage6_mimic_similar_case" / "similar_case_val_selection.csv"
SIMILAR_RANKED = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage6_mimic_similar_case" / "similar_case_fixed_test_ranked_candidates.csv"

DOC_FROZEN_CONFIG = PROJECT_ROOT / "reports" / "mimic_next" / "frozen_similar_case_aug_config.json"
PREV_REANALYSIS = PROJECT_ROOT / "reports" / "mimic_diagnosis" / "mimic_mainline_final_reanalysis.md"
OVERLAP_CASE_LEVEL = PROJECT_ROOT / "reports" / "mimic_diagnosis" / "mimic_hpo_hyperedge_overlap_case_level.csv"
LABEL_AUDIT = PROJECT_ROOT / "reports" / "mimic_diagnosis" / "mimic_label_mapping_audit.csv"
MIMIC_ROWS = PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "test" / "mimic_test_recleaned_mondo_hpo_rows.csv"

DISEASE_INDEX = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx"
HYPEREDGE_CSV = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "rare_disease_hgnn_clean_package_v59" / "v59_hyperedge_weighted_patched.csv"
MONDO_JSON = PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json"

MIMIC_DATASET = "mimic_test_recleaned_mondo_hpo_rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual analysis after current mimic SimilarCase-Aug mainline.")
    parser.add_argument("--similarity-device", default="auto")
    parser.add_argument("--similarity-batch-size", type=int, default=256)
    parser.add_argument("--skip-similar-top50-compute", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无记录_"
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    view = view.fillna("").astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def normalize_mondo(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text.replace("MONDO_", "MONDO:", 1) if text.startswith("MONDO_") else text


def raw_case_id(case_id: str) -> str:
    return str(case_id).rsplit("::", 1)[-1]


def metric_from_ranks(series: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(series, errors="coerce").fillna(9999).astype(int).to_numpy()
    if arr.size == 0:
        return {
            "num_cases": 0,
            "top1": np.nan,
            "top3": np.nan,
            "top5": np.nan,
            "rank_le_50": np.nan,
            "median_rank": np.nan,
            "mean_rank": np.nan,
        }
    return {
        "num_cases": int(arr.size),
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "rank_le_50": float(np.mean(arr <= 50)),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
    }


def mondo_from_iri(value: str) -> str | None:
    text = str(value)
    if "MONDO_" not in text:
        return None
    tail = text.rsplit("/", 1)[-1].replace("MONDO_", "MONDO:")
    return tail if tail.startswith("MONDO:") else None


def load_mondo_graph() -> dict[str, Any]:
    data = json.loads(MONDO_JSON.read_text(encoding="utf-8"))
    graph = data.get("graphs", [{}])[0]
    names: dict[str, str] = {}
    synonyms: dict[str, set[str]] = defaultdict(set)
    deprecated: set[str] = set()
    replacements: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    for node in graph.get("nodes", []) or []:
        mondo = mondo_from_iri(str(node.get("id", "")))
        if not mondo:
            continue
        if node.get("lbl"):
            names[mondo] = str(node["lbl"])
        meta = node.get("meta", {}) or {}
        if meta.get("deprecated"):
            deprecated.add(mondo)
        for syn in meta.get("synonyms", []) or []:
            val = str(syn.get("val", "")).strip().lower()
            if val:
                synonyms[mondo].add(val)
        for item in meta.get("basicPropertyValues", []) or []:
            replacement = mondo_from_iri(str(item.get("val", "")))
            if str(item.get("pred", "")).endswith("IAO_0100001") and replacement:
                replacements[mondo] = replacement
    for edge in graph.get("edges", []) or []:
        if edge.get("pred") != "is_a":
            continue
        child = mondo_from_iri(str(edge.get("sub", "")))
        parent = mondo_from_iri(str(edge.get("obj", "")))
        if child and parent:
            parents[child].add(parent)
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent_set in parents.items():
        for parent in parent_set:
            children[parent].add(child)

    @lru_cache(maxsize=None)
    def ancestors(mondo: str) -> frozenset[str]:
        out: set[str] = set()
        for parent in parents.get(mondo, set()):
            out.add(parent)
            out.update(ancestors(parent))
        return frozenset(out)

    return {
        "names": names,
        "synonyms": synonyms,
        "deprecated": deprecated,
        "replacements": replacements,
        "parents": parents,
        "children": children,
        "ancestors": ancestors,
    }


def relation_to_gold(candidate: str, gold: str, mondo: dict[str, Any]) -> str:
    candidate = normalize_mondo(candidate)
    gold = normalize_mondo(gold)
    if candidate == gold:
        return "same_disease"
    if mondo["replacements"].get(gold) == candidate or mondo["replacements"].get(candidate) == gold:
        return "replacement"
    candidate_name = str(mondo["names"].get(candidate, "")).lower()
    gold_name = str(mondo["names"].get(gold, "")).lower()
    if candidate_name and candidate_name == gold_name:
        return "synonym_or_name_match"
    if candidate_name and candidate_name in mondo["synonyms"].get(gold, set()):
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


def relation_priority(relation: str) -> int:
    order = {
        "same_disease": 0,
        "replacement": 1,
        "synonym_or_name_match": 2,
        "candidate_ancestor_of_gold": 3,
        "candidate_descendant_of_gold": 4,
        "same_parent": 5,
        "shared_ancestor": 6,
        "unrelated_or_unknown": 99,
    }
    return order.get(relation, 99)


def load_disease_resources() -> dict[str, Any]:
    disease_index = pd.read_excel(DISEASE_INDEX, dtype={"mondo_id": str})
    disease_ids = set(disease_index["mondo_id"].astype(str))
    hyper = pd.read_csv(HYPEREDGE_CSV, dtype={"mondo_id": str, "hpo_id": str})
    hyper["mondo_id"] = hyper["mondo_id"].map(normalize_mondo)
    disease_hpos = {
        disease: set(group["hpo_id"].dropna().astype(str).tolist())
        for disease, group in hyper.groupby("mondo_id", sort=False)
    }
    hpo_disease_counts = hyper[["mondo_id", "hpo_id"]].drop_duplicates().groupby("hpo_id")["mondo_id"].nunique()
    n_disease = max(1, int(hyper["mondo_id"].nunique()))
    hpo_ic = {
        str(hpo): float(math.log((1.0 + n_disease) / (1.0 + count)) + 1.0)
        for hpo, count in hpo_disease_counts.items()
    }
    return {
        "disease_ids": disease_ids,
        "disease_hpos": disease_hpos,
        "hpo_ic": hpo_ic,
    }


def overlap_score(case_hpos: set[str], disease_hpos: set[str], hpo_ic: dict[str, float]) -> dict[str, Any]:
    shared = case_hpos & disease_hpos
    case_ic = sum(float(hpo_ic.get(h, 1.0)) for h in case_hpos)
    shared_ic = sum(float(hpo_ic.get(h, 1.0)) for h in shared)
    return {
        "shared_hpo_count": int(len(shared)),
        "exact_overlap_ratio": float(len(shared) / len(case_hpos)) if case_hpos else 0.0,
        "ic_weighted_overlap": float(shared_ic / case_ic) if case_ic > 0 else 0.0,
        "disease_hpo_count": int(len(disease_hpos)),
        "score": float(len(shared) + 2.0 * (shared_ic / case_ic if case_ic > 0 else 0.0)),
    }


def load_case_labels() -> pd.DataFrame:
    rows = pd.read_csv(MIMIC_ROWS, dtype=str)
    rows["mondo_label"] = rows["mondo_label"].map(normalize_mondo)
    out = []
    for case_id, group in rows.groupby("case_id", sort=False):
        labels = sorted(set(group["mondo_label"].dropna().astype(str).tolist()))
        hpos = sorted(set(group["hpo_id"].dropna().astype(str).tolist()))
        out.append(
            {
                "case_key": str(case_id),
                "all_gold_labels": labels,
                "multilabel_case_flag": len(labels) > 1,
                "case_hpos": hpos,
                "case_hpo_count_from_rows": len(hpos),
            }
        )
    return pd.DataFrame(out)


def split_pipe(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part for part in str(value).split("|") if part]


def current_mainline_audit() -> dict[str, Any]:
    final_metrics = pd.read_csv(FINAL_METRICS)
    final_sources = pd.read_csv(FINAL_METRICS_WITH_SOURCES)
    similar_fixed = pd.read_csv(SIMILAR_FIXED)
    selection = pd.read_csv(SIMILAR_SELECTION)
    manifest = read_json(RUN_MANIFEST)
    frozen = read_json(DOC_FROZEN_CONFIG)

    current = similar_fixed.iloc[0].to_dict()
    key_cols = ["top5", "rank_le_50", "top1", "mean_rank"]
    selection_eval = selection[selection["status"].eq("evaluated")].copy()
    for col in key_cols:
        selection_eval[col] = pd.to_numeric(selection_eval[col], errors="coerce")
    selection_eval["_sort_mean_rank"] = -selection_eval["mean_rank"]
    best = selection_eval.sort_values(
        ["top5", "rank_le_50", "top1", "_sort_mean_rank"],
        ascending=[False, False, False, False],
        kind="stable",
    ).head(1)
    best_row = best.iloc[0].to_dict() if not best.empty else {}
    is_validation_selected = (
        int(float(current.get("selected_similar_case_topk", -1))) == int(float(best_row.get("similar_case_topk", -2)))
        and abs(float(current.get("selected_similar_case_weight", -1)) - float(best_row.get("similar_case_weight", -2))) < 1e-9
        and str(current.get("selected_similar_case_score_type")) == str(best_row.get("similar_case_score_type"))
    )
    mimic_final = final_metrics[final_metrics["dataset"].eq(MIMIC_DATASET)]
    mimic_source = final_sources[final_sources["dataset"].eq(MIMIC_DATASET)]
    frozen_row = {
        "config": "docx frozen config",
        "topk": frozen.get("similar_case_topk"),
        "weight": frozen.get("similar_case_weight"),
        "score_type": frozen.get("score_type"),
        "top1": frozen.get("fixed_test_metrics", {}).get("top1"),
        "top3": frozen.get("fixed_test_metrics", {}).get("top3"),
        "top5": frozen.get("fixed_test_metrics", {}).get("top5"),
        "rank_le_50": frozen.get("fixed_test_metrics", {}).get("rank_le_50"),
    }
    current_row = {
        "config": "current mainline",
        "topk": current.get("selected_similar_case_topk"),
        "weight": current.get("selected_similar_case_weight"),
        "score_type": current.get("selected_similar_case_score_type"),
        "top1": current.get("top1"),
        "top3": current.get("top3"),
        "top5": current.get("top5"),
        "rank_le_50": current.get("rank_le_50"),
    }
    lines = [
        "# Current mimic mainline audit",
        "",
        "## 当前主线结果",
        to_markdown(mimic_final),
        "",
        "## 来源",
        to_markdown(mimic_source),
        "",
        "## docx frozen config vs current mainline",
        to_markdown(pd.DataFrame([frozen_row, current_row])),
        "",
        "## validation selection check",
        f"- current mainline selected config: topk={current.get('selected_similar_case_topk')}, weight={current.get('selected_similar_case_weight')}, score_type={current.get('selected_similar_case_score_type')}",
        f"- validation best config by script key `(top5, rank_le_50, top1, -mean_rank)`: topk={best_row.get('similar_case_topk')}, weight={best_row.get('similar_case_weight')}, score_type={best_row.get('similar_case_score_type')}",
        f"- current mainline topk=20, weight=0.5 是否来自 validation-selected fixed test：{'是' if is_validation_selected else '不能确认'}",
        "",
        "## 结论",
        "- 如果采用当前 `outputs/mainline_full_pipeline` 作为正式主线，则主表建议采用 current mainline：Top1=0.2093, Top3=0.3422, Top5=0.4026, Rank<=50=0.6556。",
        "- docx frozen config Top5=0.3940 是较早 frozen 配置；它与当前输出配置不同，不应和 current mainline 混写为同一个实验。",
        "- 当前检查可确认 topk=20, weight=0.5 是现有 `similar_case_val_selection.csv` 按脚本选择规则得到的 validation-selected fixed-test 配置。",
        "",
        "## 可复现命令与路径",
        "- full: `D:\\python\\python.exe tools\\run_full_mainline_pipeline.py --config-path configs\\mainline_full_pipeline.yaml --mode full`",
        "- eval_only: `D:\\python\\python.exe tools\\run_full_mainline_pipeline.py --config-path configs\\mainline_full_pipeline.yaml --mode eval_only`",
        f"- final metrics: `{FINAL_METRICS}`",
        f"- final case ranks: `{FINAL_CASE_RANKS}`",
        f"- SimilarCase fixed test: `{SIMILAR_FIXED}`",
        f"- run_manifest checkpoint: `{manifest.get('finetune_checkpoint', '')}`",
    ]
    write_md(REPORT_DIR / "current_mimic_mainline_audit.md", lines)
    return {
        "current": current,
        "best_validation": best_row,
        "is_validation_selected": is_validation_selected,
    }


def build_residual_cases(mondo: dict[str, Any]) -> pd.DataFrame:
    ranks = pd.read_csv(FINAL_CASE_RANKS, dtype={"case_id": str, "dataset": str, "gold_id": str})
    mimic = ranks[ranks["dataset"].eq(MIMIC_DATASET)].copy()
    mimic["baseline_rank"] = pd.to_numeric(mimic["baseline_rank"], errors="coerce").fillna(9999).astype(int)
    mimic["final_rank"] = pd.to_numeric(mimic["final_rank"], errors="coerce").fillna(9999).astype(int)
    mimic["rank_delta"] = mimic["baseline_rank"] - mimic["final_rank"]
    mimic["case_key"] = mimic["case_id"].map(raw_case_id)

    final_ranked = pd.read_csv(SIMILAR_RANKED, dtype={"case_id": str, "gold_id": str, "candidate_id": str})
    for col in ["rank", "score", "hgnn_component", "similar_component", "similar_case_source_score"]:
        final_ranked[col] = pd.to_numeric(final_ranked[col], errors="coerce").fillna(0.0)
    final_top = final_ranked[final_ranked["rank"] <= 50].copy()
    baseline_candidates = pd.read_csv(TEST_CANDIDATES, dtype={"case_id": str, "candidate_id": str, "gold_id": str})
    baseline_candidates = baseline_candidates[baseline_candidates["dataset_name"].eq(MIMIC_DATASET)].copy()
    baseline_candidates["original_rank"] = pd.to_numeric(baseline_candidates["original_rank"], errors="coerce").fillna(9999).astype(int)

    overlap = pd.read_csv(OVERLAP_CASE_LEVEL, dtype={"namespaced_case_id": str}) if OVERLAP_CASE_LEVEL.is_file() else pd.DataFrame()
    label_case = load_case_labels()
    label_audit = pd.read_csv(LABEL_AUDIT, dtype=str) if LABEL_AUDIT.is_file() else pd.DataFrame()
    obsolete = set(label_audit.loc[label_audit.get("is_obsolete_mondo", "") == "True", "mondo_label"].astype(str)) if not label_audit.empty else set()

    rows = []
    grouped_final = {case_id: group.sort_values("rank") for case_id, group in final_ranked.groupby("case_id", sort=False)}
    grouped_base = {case_id: group.sort_values("original_rank") for case_id, group in baseline_candidates.groupby("case_id", sort=False)}
    overlap_map = overlap.set_index("namespaced_case_id").to_dict(orient="index") if not overlap.empty else {}
    label_map = label_case.set_index("case_key").to_dict(orient="index")

    for row in mimic.itertuples(index=False):
        case_id = str(row.case_id)
        gold = str(row.gold_id)
        group = grouped_final.get(case_id, pd.DataFrame())
        base_group = grouped_base.get(case_id, pd.DataFrame())
        top1 = str(group.iloc[0]["candidate_id"]) if not group.empty else ""
        top3 = group.head(3)["candidate_id"].astype(str).tolist() if not group.empty else []
        top5 = group.head(5)["candidate_id"].astype(str).tolist() if not group.empty else []
        final_top50_has_gold = bool((group["candidate_id"].astype(str) == gold).any()) if not group.empty else False
        baseline_top50_has_gold = bool((base_group["candidate_id"].astype(str) == gold).any()) if not base_group.empty else False
        gold_rows = group[group["candidate_id"].astype(str).eq(gold)] if not group.empty else pd.DataFrame()
        gold_row = gold_rows.sort_values("rank").head(1)
        evidence_count = 0
        best_similarity = 0.0
        if not gold_row.empty:
            evidence_count = len(split_pipe(gold_row.iloc[0].get("matched_case_ids", "")))
            best_similarity = float(gold_row.iloc[0].get("similar_case_source_score", 0.0))
        top1_row = group.head(1)
        top1_sim = float(top1_row.iloc[0].get("similar_component", 0.0)) if not top1_row.empty else 0.0
        top1_source_score = float(top1_row.iloc[0].get("similar_case_source_score", 0.0)) if not top1_row.empty else 0.0
        top_relations = [relation_to_gold(candidate, gold, mondo) for candidate in top5]
        best_relation = sorted(top_relations, key=relation_priority)[0] if top_relations else "unrelated_or_unknown"
        ov = overlap_map.get(case_id, {})
        labels = label_map.get(str(row.case_key), {})
        rows.append(
            {
                "case_id": case_id,
                "case_key": str(row.case_key),
                "gold_mondo": gold,
                "final_rank": int(row.final_rank),
                "baseline_rank": int(row.baseline_rank),
                "rank_delta": int(row.rank_delta),
                "final_top1_prediction": top1,
                "final_top3_candidates": "|".join(top3),
                "final_top5_candidates": "|".join(top5),
                "final_top50_has_gold": final_top50_has_gold,
                "baseline_top50_has_gold": baseline_top50_has_gold,
                "similar_case_evidence_count": evidence_count,
                "similar_case_best_similarity": best_similarity,
                "similar_case_gold_seen_flag": bool(evidence_count > 0),
                "similar_case_predicted_gold_flag": bool(best_similarity > 0.0),
                "final_top1_similar_component": top1_sim,
                "final_top1_similar_case_source_score": top1_source_score,
                "case_hpo_count": int(float(ov.get("case_hpo_count", labels.get("case_hpo_count_from_rows", 0)) or 0)),
                "gold_disease_hpo_count": int(float(ov.get("gold_disease_hpo_count", 0) or 0)),
                "exact_hpo_overlap_count": int(float(ov.get("exact_hpo_overlap_count", 0) or 0)),
                "exact_hpo_overlap_ratio": float(ov.get("exact_hpo_overlap_ratio", 0.0) or 0.0),
                "ic_weighted_overlap": float(ov.get("ic_weighted_overlap", 0.0) or 0.0),
                "semantic_overlap": float(ov.get("semantic_overlap", 0.0) or 0.0),
                "top1_relation_to_gold": relation_to_gold(top1, gold, mondo) if top1 else "",
                "top5_best_relation_to_gold": best_relation,
                "top5_relations_to_gold": "|".join(top_relations),
                "obsolete_label_flag": bool(gold in obsolete or gold in mondo["deprecated"]),
                "multilabel_case_flag": bool(labels.get("multilabel_case_flag", False)),
                "all_gold_labels": "|".join(labels.get("all_gold_labels", [])),
            }
        )
    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "final_residual_cases.csv", index=False, encoding="utf-8-sig")
    return out


def bucket_name(row: pd.Series) -> str:
    final_rank = int(row["final_rank"])
    baseline_rank = int(row["baseline_rank"])
    if baseline_rank == 1 and final_rank > 1:
        return "baseline_rank=1 -> final_rank>1 harmed"
    if baseline_rank <= 5 and final_rank > 5:
        return "baseline_rank<=5 -> final_rank>5 harmed"
    if baseline_rank > 50 and final_rank <= 50:
        return "baseline_rank>50 -> final_rank<=50"
    if 6 <= baseline_rank <= 50 and final_rank <= 5:
        return "baseline_rank 6-50 -> final_rank<=5"
    if final_rank > 50:
        return "final_rank>50"
    if 21 <= final_rank <= 50:
        return "final_rank 21-50"
    if 6 <= final_rank <= 20:
        return "final_rank 6-20"
    if 2 <= final_rank <= 5:
        return "final_rank 2-5"
    if final_rank == 1:
        return "final_rank=1"
    return "other"


def write_bucket_summary(residual: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "final_rank>50": residual["final_rank"] > 50,
        "final_rank 21-50": residual["final_rank"].between(21, 50),
        "final_rank 6-20": residual["final_rank"].between(6, 20),
        "final_rank 2-5": residual["final_rank"].between(2, 5),
        "final_rank=1": residual["final_rank"] == 1,
        "baseline_rank>50 -> final_rank<=50": (residual["baseline_rank"] > 50) & (residual["final_rank"] <= 50),
        "baseline_rank 6-50 -> final_rank<=5": residual["baseline_rank"].between(6, 50) & (residual["final_rank"] <= 5),
        "baseline_rank<=5 -> final_rank>5 harmed": (residual["baseline_rank"] <= 5) & (residual["final_rank"] > 5),
        "baseline_rank=1 -> final_rank>1 harmed": (residual["baseline_rank"] == 1) & (residual["final_rank"] > 1),
    }
    rows = []
    total = len(residual)
    for name, mask in definitions.items():
        group = residual[mask].copy()
        metrics = metric_from_ranks(group["final_rank"]) if not group.empty else metric_from_ranks(pd.Series(dtype=int))
        rows.append(
            {
                "bucket": name,
                "num_cases": int(len(group)),
                "case_ratio": float(len(group) / total) if total else np.nan,
                **{f"final_{key}": value for key, value in metrics.items() if key != "num_cases"},
                "mean_baseline_rank": float(group["baseline_rank"].mean()) if not group.empty else np.nan,
                "mean_rank_delta_baseline_minus_final": float(group["rank_delta"].mean()) if not group.empty else np.nan,
                "similar_gold_evidence_ratio": float(group["similar_case_gold_seen_flag"].mean()) if not group.empty else np.nan,
                "obsolete_label_ratio": float(group["obsolete_label_flag"].mean()) if not group.empty else np.nan,
                "multilabel_ratio": float(group["multilabel_case_flag"].mean()) if not group.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "residual_bucket_summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Residual bucket summary after SimilarCase-Aug",
        "",
        to_markdown(out),
        "",
        "## 结论",
        "- `final_rank>50` 是 SimilarCase-Aug 后仍无法由 rerank 直接解决的主要 residual recall 问题。",
        "- `final_rank 6-20` 和 `21-50` 是 stronger rerank / gated multiview evidence 的主要目标。",
        "- harmed bucket 说明 SimilarCase-Aug 存在误伤，需要 confidence gate 或 HGNN top1 protection。",
    ]
    write_md(REPORT_DIR / "residual_bucket_summary.md", lines)
    return out


def write_failure_types(residual: pd.DataFrame) -> pd.DataFrame:
    top_relaxed = ~residual["top5_best_relation_to_gold"].isin(["same_disease", "unrelated_or_unknown", ""])
    definitions = {
        "A_similar_case_no_or_weak_evidence": (residual["final_rank"] > 50) & (residual["similar_case_best_similarity"] < 0.1),
        "B_evidence_but_weight_insufficient": residual["final_rank"].between(6, 50) & (residual["similar_case_best_similarity"] >= 0.1),
        "C_wrong_evidence_interference": (
            ((residual["baseline_rank"] <= 5) & (residual["final_rank"] > 5))
            | ((residual["baseline_rank"] == 1) & (residual["final_rank"] > 1))
        ) & (residual["final_top1_similar_component"] > 0) & (residual["final_top1_prediction"] != residual["gold_mondo"]),
        "D_candidate_recall_missing": (residual["final_rank"] > 50) & (~residual["final_top50_has_gold"]),
        "E_label_ontology_relaxed_hit": (residual["final_rank"] > 1) & top_relaxed,
    }
    total = len(residual)
    rows = []
    for name, mask in definitions.items():
        group = residual[mask].copy()
        rows.append(
            {
                "failure_type": name,
                "num_cases": int(len(group)),
                "case_ratio": float(len(group) / total) if total else np.nan,
                "mean_final_rank": float(group["final_rank"].mean()) if not group.empty else np.nan,
                "median_final_rank": float(group["final_rank"].median()) if not group.empty else np.nan,
                "mean_baseline_rank": float(group["baseline_rank"].mean()) if not group.empty else np.nan,
                "similar_gold_evidence_ratio": float(group["similar_case_gold_seen_flag"].mean()) if not group.empty else np.nan,
                "mean_gold_similarity": float(group["similar_case_best_similarity"].mean()) if not group.empty else np.nan,
                "multilabel_ratio": float(group["multilabel_case_flag"].mean()) if not group.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "failure_type_summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Failure type summary after SimilarCase-Aug",
        "",
        to_markdown(out),
        "",
        "## 类型解释",
        "- A：final rank>50 且 gold 的 SimilarCase evidence 很弱或没有，优先看候选扩展或上游数据。",
        "- B：gold 有 evidence 但仍在 6-50，适合 gated rerank / stronger light reranker。",
        "- C：SimilarCase 把错误疾病推高造成误伤，适合 confidence gate 和 HGNN top1 protection。",
        "- D：final top50 不含 gold，candidate recall 缺失，rerank/hard negative 不能单独解决。",
        "- E：Top candidates 与 gold 有 ontology 近邻关系，只能作为 relaxed supplementary。",
    ]
    write_md(REPORT_DIR / "failure_type_summary.md", lines)
    return out


def relation_expansion(candidates: list[str], mondo: dict[str, Any], disease_ids: set[str], max_new: int = 100) -> list[tuple[str, str]]:
    seen = set(candidates)
    out: list[tuple[str, str]] = []
    def add(candidate: str, relation: str) -> None:
        if candidate in disease_ids and candidate not in seen:
            seen.add(candidate)
            out.append((candidate, relation))
    for cand in candidates:
        for parent in sorted(mondo["parents"].get(cand, set())):
            add(parent, "parent")
        for child in sorted(mondo["children"].get(cand, set())):
            add(child, "child")
        for parent in sorted(mondo["parents"].get(cand, set())):
            for sibling in sorted(mondo["children"].get(parent, set())):
                add(sibling, "sibling")
        for ancestor in sorted(list(mondo["ancestors"](cand)))[:3]:
            for relative in sorted(mondo["children"].get(ancestor, set())):
                add(relative, "shared_ancestor")
        if len(out) >= max_new:
            break
    return out[:max_new]


def top_hpo_expansion(case_hpos: set[str], resources: dict[str, Any], existing: set[str], max_new: int = 200) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for disease, disease_hpos in resources["disease_hpos"].items():
        if disease in existing:
            continue
        metrics = overlap_score(case_hpos, disease_hpos, resources["hpo_ic"])
        if metrics["shared_hpo_count"] <= 0 and metrics["ic_weighted_overlap"] <= 0:
            continue
        rows.append((disease, metrics))
    rows.sort(key=lambda item: (item[1]["score"], item[1]["ic_weighted_overlap"], item[1]["shared_hpo_count"]), reverse=True)
    return rows[:max_new]


def expansion_audit(
    residual: pd.DataFrame,
    mondo: dict[str, Any],
    resources: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    final_ranked = pd.read_csv(SIMILAR_RANKED, dtype={"case_id": str, "candidate_id": str, "gold_id": str})
    final_ranked["rank"] = pd.to_numeric(final_ranked["rank"], errors="coerce").fillna(9999).astype(int)
    final_top50 = {case_id: group.sort_values("rank").head(50)["candidate_id"].astype(str).tolist() for case_id, group in final_ranked.groupby("case_id", sort=False)}
    case_labels = load_case_labels().set_index("case_key").to_dict(orient="index")

    test_rows = []
    target = residual[(residual["final_rank"] > 50) | (residual["final_rank"].between(21, 50))].copy()
    for row in target.itertuples(index=False):
        case_id = str(row.case_id)
        gold = str(row.gold_mondo)
        existing = set(final_top50.get(case_id, []))
        relation_rows = relation_expansion(final_top50.get(case_id, []), mondo, resources["disease_ids"], max_new=100)
        labels = case_labels.get(str(row.case_key), {})
        case_hpos = set(labels.get("case_hpos", []))
        hpo_rows = top_hpo_expansion(case_hpos, resources, existing, max_new=200)
        for rank, (candidate, relation) in enumerate(relation_rows, start=1):
            test_rows.append(
                {
                    "split": "test_analysis_only",
                    "case_id": case_id,
                    "gold_mondo": gold,
                    "expansion_type": "mondo_relation",
                    "candidate_id": candidate,
                    "expansion_rank": rank,
                    "source_relation": relation,
                    "score": "",
                    "is_exact_gold": candidate == gold,
                    "candidate_already_in_final_top50": candidate in existing,
                }
            )
        for rank, (candidate, metrics) in enumerate(hpo_rows, start=1):
            test_rows.append(
                {
                    "split": "test_analysis_only",
                    "case_id": case_id,
                    "gold_mondo": gold,
                    "expansion_type": "hpo_hyperedge",
                    "candidate_id": candidate,
                    "expansion_rank": rank,
                    "source_relation": "",
                    "score": metrics["score"],
                    "shared_hpo_count": metrics["shared_hpo_count"],
                    "ic_weighted_overlap": metrics["ic_weighted_overlap"],
                    "is_exact_gold": candidate == gold,
                    "candidate_already_in_final_top50": candidate in existing,
                }
            )
    test_expanded = pd.DataFrame(test_rows)
    test_expanded.to_csv(OUTPUT_DIR / "residual_expanded_candidates_test_analysis_only.csv", index=False, encoding="utf-8-sig")

    val_expanded = pd.DataFrame()
    similar_validation_summary: dict[str, Any] = {"validation_similar_top30_new_gold": np.nan, "validation_similar_top50_new_gold": np.nan}
    similar_test_summary: dict[str, Any] = {"test_similar_top30_new_gold": np.nan, "test_similar_top50_new_gold": np.nan}
    existing_val_expansion = OUTPUT_DIR / "residual_expanded_candidates_validation.csv"
    existing_test_expansion = OUTPUT_DIR / "residual_expanded_candidates_test_analysis_only.csv"
    if args.skip_similar_top50_compute and existing_val_expansion.is_file() and existing_test_expansion.is_file():
        val_expanded = pd.read_csv(existing_val_expansion)
        old_test = pd.read_csv(existing_test_expansion)
        if "similar_case_top30" in set(old_test.get("expansion_type", pd.Series(dtype=str)).astype(str)):
            test_expanded = old_test
        if not val_expanded.empty and "expansion_type" in val_expanded.columns:
            similar_validation_summary = {
                "validation_similar_top30_gold_cases": int(val_expanded.loc[val_expanded["expansion_type"].eq("similar_case_top30"), "case_id"].nunique()),
                "validation_similar_top50_gold_cases": int(val_expanded.loc[val_expanded["expansion_type"].eq("similar_case_top50"), "case_id"].nunique()),
            }
        if not test_expanded.empty and "expansion_type" in test_expanded.columns:
            similar_test_summary = {
                "test_analysis_only_similar_top30_gold_cases": int(test_expanded.loc[test_expanded["expansion_type"].eq("similar_case_top30"), "case_id"].nunique()),
                "test_analysis_only_similar_top50_gold_cases": int(test_expanded.loc[test_expanded["expansion_type"].eq("similar_case_top50"), "case_id"].nunique()),
            }
    elif not args.skip_similar_top50_compute:
        data_config = load_yaml_config(DATA_CONFIG)
        train_config = load_yaml_config(TRAIN_CONFIG)
        train_table, val_table, test_table = load_case_tables(data_config, DATA_CONFIG, train_config)
        val_sim = compute_similar_matches(val_table, train_table, 50, device_name=args.similarity_device, batch_size=args.similarity_batch_size, label="validation_residual_top50")
        test_sim = compute_similar_matches(test_table, train_table, 50, device_name=args.similarity_device, batch_size=args.similarity_batch_size, label="test_residual_top50_analysis_only")

        def sim_gold_counts(sim_df: pd.DataFrame, table: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
            primary = dict(zip(table["case_id"].astype(str), table["primary_label"].astype(str)))
            records = []
            for case_id, group in sim_df.groupby("case_id", sort=False):
                gold = primary.get(str(case_id), "")
                ranks = group.loc[group["matched_label"].astype(str).eq(gold), "similar_rank"]
                best_rank = int(ranks.min()) if not ranks.empty else 9999
                for k in [30, 50]:
                    if best_rank <= k:
                        records.append(
                            {
                                "split": split,
                                "case_id": case_id,
                                "gold_mondo": gold,
                                "expansion_type": f"similar_case_top{k}",
                                "candidate_id": gold,
                                "expansion_rank": best_rank,
                                "source_relation": "matched_label",
                                "score": float(group.loc[group["matched_label"].astype(str).eq(gold), "raw_similarity"].max()),
                                "is_exact_gold": True,
                                "candidate_already_in_final_top50": "",
                            }
                        )
            counts = {
                f"{split}_similar_top30_gold_cases": int(sum(1 for _, g in sim_df.groupby("case_id") if not g.loc[g["matched_label"].astype(str).eq(primary.get(str(g['case_id'].iloc[0]), '')) & (g["similar_rank"] <= 30)].empty)),
                f"{split}_similar_top50_gold_cases": int(sum(1 for _, g in sim_df.groupby("case_id") if not g.loc[g["matched_label"].astype(str).eq(primary.get(str(g['case_id'].iloc[0]), '')) & (g["similar_rank"] <= 50)].empty)),
            }
            return pd.DataFrame(records), counts

        val_expanded, similar_validation_summary = sim_gold_counts(val_sim, val_table, "validation")
        test_sim_expanded, similar_test_summary = sim_gold_counts(test_sim, test_table, "test_analysis_only")
        val_expanded.to_csv(OUTPUT_DIR / "residual_expanded_candidates_validation.csv", index=False, encoding="utf-8-sig")
        if not test_sim_expanded.empty:
            test_expanded = pd.concat([test_expanded, test_sim_expanded], ignore_index=True, sort=False)
            test_expanded.to_csv(OUTPUT_DIR / "residual_expanded_candidates_test_analysis_only.csv", index=False, encoding="utf-8-sig")
    else:
        val_expanded.to_csv(OUTPUT_DIR / "residual_expanded_candidates_validation.csv", index=False, encoding="utf-8-sig")

    final_gt50_ids = set(residual.loc[residual["final_rank"] > 50, "case_id"])
    final_gt50_exp = test_expanded[test_expanded["case_id"].isin(final_gt50_ids)].copy() if not test_expanded.empty else pd.DataFrame()
    summary = {
        "final_rank_gt50_cases": int(len(final_gt50_ids)),
        "mondo_relation_recovered_gold_cases": int(final_gt50_exp.loc[(final_gt50_exp["expansion_type"] == "mondo_relation") & (final_gt50_exp["is_exact_gold"] == True), "case_id"].nunique()) if not final_gt50_exp.empty else 0,
        "hpo_hyperedge_recovered_gold_cases_top100": int(final_gt50_exp.loc[(final_gt50_exp["expansion_type"] == "hpo_hyperedge") & (final_gt50_exp["is_exact_gold"] == True) & (pd.to_numeric(final_gt50_exp["expansion_rank"], errors="coerce") <= 100), "case_id"].nunique()) if not final_gt50_exp.empty else 0,
        "hpo_hyperedge_recovered_gold_cases_top200": int(final_gt50_exp.loc[(final_gt50_exp["expansion_type"] == "hpo_hyperedge") & (final_gt50_exp["is_exact_gold"] == True) & (pd.to_numeric(final_gt50_exp["expansion_rank"], errors="coerce") <= 200), "case_id"].nunique()) if not final_gt50_exp.empty else 0,
        "similar_case_top30_recovered_gold_cases_analysis_only": int(final_gt50_exp.loc[(final_gt50_exp["expansion_type"] == "similar_case_top30") & (final_gt50_exp["is_exact_gold"] == True), "case_id"].nunique()) if not final_gt50_exp.empty else 0,
        "similar_case_top50_recovered_gold_cases_analysis_only": int(final_gt50_exp.loc[(final_gt50_exp["expansion_type"] == "similar_case_top50") & (final_gt50_exp["is_exact_gold"] == True), "case_id"].nunique()) if not final_gt50_exp.empty else 0,
        **similar_validation_summary,
        **similar_test_summary,
    }
    recovered_union = set()
    if not final_gt50_exp.empty:
        recovered_union = set(final_gt50_exp.loc[final_gt50_exp["is_exact_gold"] == True, "case_id"].astype(str))
    summary["still_unrecovered_by_any_analysis_expansion"] = int(len(final_gt50_ids - recovered_union))
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(REPORT_DIR / "residual_expansion_recall_audit.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Residual expansion recall audit",
        "",
        "## Summary",
        to_markdown(summary_df),
        "",
        "## 结论",
        "- test expansion 文件只作为 analysis-only，不用于选择 topk 或权重。",
        "- MONDO relation expansion 若只能找回少量 gold，说明 strict exact miss 不是简单 parent/child/sibling 扩展即可解决。",
        "- HPO hyperedge expansion 找回的 gold 如果主要在 top100/top200，而不是 top5，说明下一步需要 light reranker，而不是只做 candidate generation。",
        "- SimilarCase topk 30/50 只有 validation 有稳定证据时才建议扩大；不能用 test analysis-only 反向选择。",
    ]
    write_md(REPORT_DIR / "residual_expansion_recall_audit.md", lines)
    return summary_df


def rank_candidates(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    ranked = df.sort_values(["case_id", score_col, "hgnn_component"], ascending=[True, False, False], kind="stable").copy()
    ranked["rank"] = ranked.groupby("case_id").cumcount() + 1
    return ranked


def evaluate_ranked(ranked: pd.DataFrame, gold_by_case: dict[str, str]) -> dict[str, Any]:
    gold_df = pd.DataFrame(
        {"case_id": list(gold_by_case.keys()), "candidate_id": list(gold_by_case.values())}
    )
    hit = gold_df.merge(
        ranked[["case_id", "candidate_id", "rank"]],
        on=["case_id", "candidate_id"],
        how="left",
    )
    return metric_from_ranks(hit["rank"].fillna(9999))


def add_candidate_overlap_features(df: pd.DataFrame, table: pd.DataFrame, resources: dict[str, Any]) -> pd.DataFrame:
    case_hpos = {row.case_id: set(row.hpo_ids) for row in table.itertuples(index=False)}
    rows = []
    for row in df.itertuples(index=False):
        hpos = case_hpos.get(str(row.case_id), set())
        disease_hpos = resources["disease_hpos"].get(str(row.candidate_id), set())
        metrics = overlap_score(hpos, disease_hpos, resources["hpo_ic"])
        rows.append(metrics)
    metrics_df = pd.DataFrame(rows)
    out = df.reset_index(drop=True).copy()
    for col in ["shared_hpo_count", "exact_overlap_ratio", "ic_weighted_overlap", "disease_hpo_count"]:
        if col not in out.columns or out[col].isna().all():
            out[col] = metrics_df[col]
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(metrics_df[col])
    return out


def run_gated_validation(args: argparse.Namespace, resources: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    data_config = load_yaml_config(DATA_CONFIG)
    train_config = load_yaml_config(TRAIN_CONFIG)
    train_table, val_table, test_table = load_case_tables(data_config, DATA_CONFIG, train_config)
    val_candidates = load_candidates(VAL_CANDIDATES)
    test_candidates = load_candidates(TEST_CANDIDATES)
    test_candidates = test_candidates[test_candidates["dataset_name"].astype(str).str.startswith("mimic_test")].copy()
    val_hgnn = hgnn_source(val_candidates, val_table)
    test_hgnn = hgnn_source(test_candidates, test_table)

    val_sim = compute_similar_matches(val_table, train_table, 20, device_name=args.similarity_device, batch_size=args.similarity_batch_size, label="validation_gated")
    val_sim_source = similar_source(val_sim, 20, "raw_similarity", val_table)
    current_weight = 0.5
    current = pd.concat(
        [
            val_hgnn,
            val_sim_source.assign(hgnn_component=0.0, hgnn_score=0.0, ic_weighted_overlap=0.0, exact_overlap=0.0),
        ],
        ignore_index=True,
        sort=False,
    )
    for col in ["hgnn_component", "similar_component", "similar_case_source_score", "ic_weighted_overlap", "exact_overlap"]:
        if col not in current.columns:
            current[col] = 0.0
        current[col] = pd.to_numeric(current[col], errors="coerce").fillna(0.0)
    current = current.groupby(["case_id", "gold_id", "candidate_id"], as_index=False).agg(
        hgnn_component=("hgnn_component", "max"),
        similar_component=("similar_component", "max"),
        similar_case_source_score=("similar_case_source_score", "max"),
        ic_weighted_overlap=("ic_weighted_overlap", "max"),
        exact_overlap=("exact_overlap", "max"),
    )
    current = add_candidate_overlap_features(current, val_table, resources)
    gold_by_val = dict(zip(val_table["case_id"].astype(str), val_table["primary_label"].astype(str)))
    current["score"] = current["hgnn_component"] + current_weight * current["similar_component"]
    current_metrics = evaluate_ranked(rank_candidates(current), gold_by_val)

    rows = []
    best_payload: dict[str, Any] | None = None
    for sim_weight in [0.2, 0.3, 0.4, 0.5]:
        for ic_weight in [0.0, 0.05, 0.1]:
            for agree_boost in [0.0, 0.05, 0.1]:
                for protect_bonus in [0.0, 0.05, 0.1]:
                    scored = current.copy()
                    agreement = (scored["hgnn_component"] > 0) & (scored["similar_component"] > 0)
                    low_overlap = scored["ic_weighted_overlap"] <= 0.02
                    sim_eff = scored["similar_component"] * sim_weight
                    sim_eff = np.where(low_overlap & (scored["hgnn_component"] == 0), sim_eff * 0.5, sim_eff)
                    scored["score"] = (
                        scored["hgnn_component"]
                        + sim_eff
                        + ic_weight * scored["ic_weighted_overlap"]
                        + np.where(agreement, agree_boost, 0.0)
                    )
                    top1_idx = scored.sort_values(["case_id", "hgnn_component"], ascending=[True, False]).groupby("case_id").head(1).index
                    scored.loc[top1_idx, "score"] = scored.loc[top1_idx, "score"] + protect_bonus
                    metrics = evaluate_ranked(rank_candidates(scored), gold_by_val)
                    row = {
                        "sim_weight": sim_weight,
                        "ic_weight": ic_weight,
                        "agree_boost": agree_boost,
                        "protect_bonus": protect_bonus,
                        **{f"val_{key}": value for key, value in metrics.items()},
                        "baseline_current_val_top5": current_metrics["top5"],
                        "top5_delta_vs_current": metrics["top5"] - current_metrics["top5"],
                        "top1_delta_vs_current": metrics["top1"] - current_metrics["top1"],
                    }
                    rows.append(row)
                    key = (metrics["top5"], metrics["top3"], metrics["top1"], metrics["rank_le_50"], -metrics["mean_rank"])
                    if best_payload is None or key > best_payload["key"]:
                        best_payload = {"key": key, "row": row, "scored": scored}
    grid = pd.DataFrame(rows).sort_values(["val_top5", "val_top3", "val_top1", "val_rank_le_50"], ascending=[False, False, False, False])
    grid.to_csv(REPORT_DIR / "validation_gated_rerank_grid.csv", index=False, encoding="utf-8-sig")
    selected = best_payload["row"] if best_payload else None
    improved = bool(selected and selected["val_top5"] > current_metrics["top5"])
    config = {
        "selected_by": "validation",
        "current_simple_validation_metrics": current_metrics,
        "selected_config": selected,
        "validation_improved_top5": improved,
        "do_not_use_if_validation_improved_top5_false": not improved,
    }
    (OUTPUT_DIR / "gated_selected_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Validation gated rerank selected",
        "",
        f"- current simple SimilarCase validation top5: {current_metrics['top5']:.4f}",
        f"- best gated validation top5: {selected['val_top5']:.4f}" if selected else "- no selected config",
        f"- validation 是否提升：{'是' if improved else '否'}",
        "",
        "## Selected",
        to_markdown(pd.DataFrame([selected])) if selected else "_无_",
        "",
        "## Top grid rows",
        to_markdown(grid.head(20)),
    ]
    write_md(REPORT_DIR / "validation_gated_rerank_selected.md", lines)

    if not improved:
        return grid, None

    test_sim = compute_similar_matches(test_table, train_table, 20, device_name=args.similarity_device, batch_size=args.similarity_batch_size, label="test_gated_fixed")
    test_sim_source = similar_source(test_sim, 20, "raw_similarity", test_table)
    test_current = pd.concat(
        [
            test_hgnn,
            test_sim_source.assign(hgnn_component=0.0, hgnn_score=0.0, ic_weighted_overlap=0.0, exact_overlap=0.0),
        ],
        ignore_index=True,
        sort=False,
    )
    for col in ["hgnn_component", "similar_component", "similar_case_source_score", "ic_weighted_overlap", "exact_overlap"]:
        if col not in test_current.columns:
            test_current[col] = 0.0
        test_current[col] = pd.to_numeric(test_current[col], errors="coerce").fillna(0.0)
    test_current = test_current.groupby(["case_id", "gold_id", "candidate_id"], as_index=False).agg(
        hgnn_component=("hgnn_component", "max"),
        similar_component=("similar_component", "max"),
        similar_case_source_score=("similar_case_source_score", "max"),
        ic_weighted_overlap=("ic_weighted_overlap", "max"),
        exact_overlap=("exact_overlap", "max"),
    )
    test_current = add_candidate_overlap_features(test_current, test_table, resources)
    sw = float(selected["sim_weight"])
    iw = float(selected["ic_weight"])
    ab = float(selected["agree_boost"])
    pb = float(selected["protect_bonus"])
    agreement = (test_current["hgnn_component"] > 0) & (test_current["similar_component"] > 0)
    low_overlap = test_current["ic_weighted_overlap"] <= 0.02
    sim_eff = test_current["similar_component"] * sw
    sim_eff = np.where(low_overlap & (test_current["hgnn_component"] == 0), sim_eff * 0.5, sim_eff)
    test_current["score"] = test_current["hgnn_component"] + sim_eff + iw * test_current["ic_weighted_overlap"] + np.where(agreement, ab, 0.0)
    top1_idx = test_current.sort_values(["case_id", "hgnn_component"], ascending=[True, False]).groupby("case_id").head(1).index
    test_current.loc[top1_idx, "score"] = test_current.loc[top1_idx, "score"] + pb
    gold_by_test = dict(zip(test_table["case_id"].astype(str), test_table["primary_label"].astype(str)))
    test_metrics = evaluate_ranked(rank_candidates(test_current), gold_by_test)
    test_out = pd.DataFrame([{**selected, **{f"test_{key}": value for key, value in test_metrics.items()}}])
    test_out.to_csv(REPORT_DIR / "fixed_test_gated_rerank_results.csv", index=False, encoding="utf-8-sig")
    current_mainline = pd.read_csv(FINAL_METRICS)
    current_mimic = current_mainline[current_mainline["dataset"].eq(MIMIC_DATASET)].iloc[0].to_dict()
    write_md(
        REPORT_DIR / "fixed_test_gated_rerank_results.md",
        [
            "# Fixed test gated rerank results",
            "",
            "- 该结果只因为 validation top5 提升才执行 fixed test 一次。",
            (
                "- 与 current mainline `similar_case_fixed_test` 对比："
                f"Top1 {float(current_mimic['top1']):.4f} -> {test_metrics['top1']:.4f}（{test_metrics['top1'] - float(current_mimic['top1']):+.4f}），"
                f"Top3 {float(current_mimic['top3']):.4f} -> {test_metrics['top3']:.4f}（{test_metrics['top3'] - float(current_mimic['top3']):+.4f}），"
                f"Top5 {float(current_mimic['top5']):.4f} -> {test_metrics['top5']:.4f}（{test_metrics['top5'] - float(current_mimic['top5']):+.4f}），"
                f"Rank<=50 {float(current_mimic['rank_le_50']):.4f} -> {test_metrics['rank_le_50']:.4f}（{test_metrics['rank_le_50'] - float(current_mimic['rank_le_50']):+.4f}）。"
            ),
            "- 结论：gated rerank 是可进入对比表的 validation-selected fixed-test 候选，但 Top5 增益很小且 Rank<=50 有轻微下降，不建议未经更多验证就直接替换 current mainline。",
            to_markdown(test_out),
        ],
    )
    return grid, test_metrics


def write_recommendation(
    mainline: dict[str, Any],
    residual: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    expansion_summary: pd.DataFrame,
    gated_grid: pd.DataFrame,
    fixed_test_metrics: dict[str, Any] | None,
) -> None:
    final_gt50 = int((residual["final_rank"] > 50).sum())
    final_late = int((residual["final_rank"].between(6, 50)).sum())
    harmed_le5 = int(((residual["baseline_rank"] <= 5) & (residual["final_rank"] > 5)).sum())
    harmed_top1 = int(((residual["baseline_rank"] == 1) & (residual["final_rank"] > 1)).sum())
    validation_best = gated_grid.iloc[0].to_dict() if not gated_grid.empty else {}
    validation_improved = bool(validation_best and float(validation_best.get("top5_delta_vs_current", 0.0)) > 0)
    current_mimic = pd.read_csv(FINAL_METRICS)
    current_mimic_row = current_mimic[current_mimic["dataset"].eq(MIMIC_DATASET)].iloc[0].to_dict()
    fixed_lines: list[str]
    if fixed_test_metrics is not None:
        fixed_lines = [
            f"- fixed test 结果：Top1={fixed_test_metrics['top1']:.4f}、Top3={fixed_test_metrics['top3']:.4f}、Top5={fixed_test_metrics['top5']:.4f}、Rank<=50={fixed_test_metrics['rank_le_50']:.4f}。",
            (
                "- 相比 current mainline："
                f"Top1 {fixed_test_metrics['top1'] - float(current_mimic_row['top1']):+.4f}、"
                f"Top3 {fixed_test_metrics['top3'] - float(current_mimic_row['top3']):+.4f}、"
                f"Top5 {fixed_test_metrics['top5'] - float(current_mimic_row['top5']):+.4f}、"
                f"Rank<=50 {fixed_test_metrics['rank_le_50'] - float(current_mimic_row['rank_le_50']):+.4f}。"
            ),
            "- 因此 gated rerank 值得作为 validation-selected fixed-test 候选进入对比表；但 Top5 增益很小且召回略降，不建议未经更多 validation/bootstrap 稳定性验证就替换 current mainline。",
        ]
    else:
        fixed_lines = ["- validation 没有提升，已停止，未执行 fixed test。"]
    lines = [
        "# Recommended next step after SimilarCase-Aug residual diagnosis",
        "",
        "## 1. 当前 mimic 正式主线结果采用哪个配置",
        "- 当前 `outputs/mainline_full_pipeline` 中应采用 current mainline：`topk=20`, `weight=0.5`, `score_type=raw_similarity`, Top5=0.4026。",
        "- docx frozen config `topk=10`, `weight=0.4`, Top5=0.3940 是较早配置；如果论文以当前 mainline 为准，应在方法和表格中同步更新配置。",
        "",
        "## 2. SimilarCase-Aug 已经解决了什么",
        "- Top1/Top3/Top5/Rank<=50 都高于 HGNN exact baseline。",
        "- 它主要把 baseline rank 6-50 的病例推入 Top5，并把一部分 rank>50 拉回 top50。",
        "",
        "## 3. SimilarCase-Aug 没解决什么",
        f"- final rank>50 仍有 {final_gt50}/1873，说明 candidate recall residual 仍明显存在。",
        f"- final rank 6-50 仍有 {final_late}/1873，说明 top50 内排序 residual 仍存在。",
        "",
        "## 4. 是否存在 SimilarCase 误伤",
        f"- baseline rank<=5 -> final rank>5 的病例数为 {harmed_le5}。",
        f"- baseline rank=1 -> final rank>1 的病例数为 {harmed_top1}。",
        "- 因此存在误伤，需要 gate / HGNN top1 protection，而不是继续无门控增加 SimilarCase 权重。",
        "",
        "## 5. residual candidate expansion 是否有新增价值",
        to_markdown(expansion_summary),
        "- 如果扩展只能把 gold 拉到 top100/top200，下一步需要 light reranker 才可能转化为 Top5。",
        "",
        "## 6. gated rerank 是否值得进入 fixed test",
        f"- validation gated top5 delta vs current: {float(validation_best.get('top5_delta_vs_current', 0.0)):.4f}" if validation_best else "- validation gated 未产生结果。",
        f"- 是否执行 fixed test：{'是' if fixed_test_metrics is not None else '否'}。",
        *fixed_lines,
        "",
        "## 7. 图对比学习是否仍不作为第一优先级",
        "- 仍不作为第一优先级。当前 residual 同时包含 candidate recall 缺失、top50 内排序、SimilarCase 误伤、low-overlap/label noise。",
        "- 只有在 residual candidates 能稳定召回 gold、正负对可从 train/validation 构建、low-overlap 样本可过滤或降权后，才建议图对比学习。",
        "",
        "## 8. 下一步最推荐实验",
        "1. P0：固定 current mainline 配置与文档口径，避免 0.3940/0.4026 混写。",
        "2. P1：保留 validation-selected gated SimilarCase + multiview evidence rerank 作为候选，对 Top1/Top3 有小幅收益；下一步先做稳定性验证，不直接替换主线。",
        "3. P2：对 final rank>50 做 residual-targeted candidate expansion，重点看 MONDO/HPO expansion 能否提高 recall@100/200。",
        "4. P3：如果 expansion 提升 recall@100/200 但不能进 Top5，再做 light-train reranker。",
        "5. P4：最后再考虑图对比学习或 hard negative training。",
        "",
        "## 论文主表与 supplementary",
        "- strict exact current mainline 可进入主表，前提是说明配置来自 validation-selected fixed test。",
        "- any-label、relaxed MONDO、ancestor/sibling/synonym/replacement 命中只能 supplementary。",
        "- test analysis-only expansion 不能作为主表提升，只能说明潜在上限和下一步方向。",
    ]
    write_md(REPORT_DIR / "recommended_next_step_after_residual.md", lines)


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mondo = load_mondo_graph()
    resources = load_disease_resources()
    mainline = current_mainline_audit()
    residual = build_residual_cases(mondo)
    bucket_summary = write_bucket_summary(residual)
    failure_summary = write_failure_types(residual)
    expansion_summary = expansion_audit(residual, mondo, resources, args)
    gated_grid, fixed_test_metrics = run_gated_validation(args, resources)
    write_recommendation(mainline, residual, bucket_summary, failure_summary, expansion_summary, gated_grid, fixed_test_metrics)
    manifest = {
        "command": "D:\\python\\python.exe tools\\analysis\\mimic_residual_after_similar_case.py",
        "outputs": [
            str((REPORT_DIR / "current_mimic_mainline_audit.md").resolve()),
            str((OUTPUT_DIR / "final_residual_cases.csv").resolve()),
            str((REPORT_DIR / "residual_bucket_summary.csv").resolve()),
            str((REPORT_DIR / "residual_bucket_summary.md").resolve()),
            str((REPORT_DIR / "failure_type_summary.csv").resolve()),
            str((REPORT_DIR / "failure_type_summary.md").resolve()),
            str((OUTPUT_DIR / "residual_expanded_candidates_validation.csv").resolve()),
            str((OUTPUT_DIR / "residual_expanded_candidates_test_analysis_only.csv").resolve()),
            str((REPORT_DIR / "residual_expansion_recall_audit.csv").resolve()),
            str((REPORT_DIR / "residual_expansion_recall_audit.md").resolve()),
            str((REPORT_DIR / "validation_gated_rerank_grid.csv").resolve()),
            str((REPORT_DIR / "validation_gated_rerank_selected.md").resolve()),
            str((OUTPUT_DIR / "gated_selected_config.json").resolve()),
            str((REPORT_DIR / "recommended_next_step_after_residual.md").resolve()),
        ],
    }
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
