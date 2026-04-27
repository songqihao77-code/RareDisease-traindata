from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rerank.hpo_semantic import HpoSemanticMatcher


DEFAULT_MIMIC = PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "test" / "mimic_test_recleaned_mondo_hpo_rows.csv"
DEFAULT_EXACT = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage3_exact_eval" / "exact_details.csv"
DEFAULT_EXACT_SUMMARY = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage3_exact_eval" / "exact_summary.json"
DEFAULT_CANDIDATES = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "stage4_candidates" / "top50_candidates_test.csv"
DEFAULT_CANDIDATE_META = DEFAULT_CANDIDATES.with_suffix(".metadata.json")
DEFAULT_RUN_MANIFEST = PROJECT_ROOT / "outputs" / "mainline_full_pipeline" / "run_manifest.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "mimic_diagnosis"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mimic_diagnosis"

DISEASE_INDEX = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx"
HPO_INDEX = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "HPO_index_v4.xlsx"
HYPEREDGE_CSV = (
    PROJECT_ROOT
    / "LLLdataset"
    / "DiseaseHy"
    / "rare_disease_hgnn_clean_package_v59"
    / "v59_hyperedge_weighted_patched.csv"
)
INCIDENCE_NPZ = (
    PROJECT_ROOT
    / "LLLdataset"
    / "DiseaseHy"
    / "rare_disease_hgnn_clean_package_v59"
    / "v59DiseaseHy.npz"
)
MONDO_JSON = PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json"
DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
MAINLINE_CONFIG = PROJECT_ROOT / "configs" / "mainline_full_pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only mimic_test HGNN diagnosis.")
    parser.add_argument("--mimic-path", type=Path, default=DEFAULT_MIMIC)
    parser.add_argument("--exact-details-path", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--exact-summary-path", type=Path, default=DEFAULT_EXACT_SUMMARY)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate-metadata-path", type=Path, default=DEFAULT_CANDIDATE_META)
    parser.add_argument("--run-manifest-path", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def to_markdown(df: pd.DataFrame, max_rows: int | None = None, digits: int = 4) -> str:
    if df.empty:
        return "_无记录_"
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
    view = view.fillna("").astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def raw_case_id(case_id: str) -> str:
    return str(case_id).rsplit("::", 1)[-1]


def dataset_alias(value: str) -> str:
    stem = Path(str(value)).stem
    if stem in {"mimic_test", "mimic_test_recleaned", "mimic_test_recleaned_mondo_hpo_rows"}:
        return "mimic_test_recleaned_mondo_hpo_rows"
    return stem


def normalize_mondo(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.startswith("MONDO_"):
        return text.replace("MONDO_", "MONDO:", 1)
    return text


def mondo_from_iri(value: str) -> str | None:
    text = str(value)
    if "MONDO_" not in text:
        return None
    tail = text.rsplit("/", 1)[-1].replace("MONDO_", "MONDO:")
    return tail if tail.startswith("MONDO:") else None


def load_mondo_graph(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
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
            value = str(syn.get("val", "")).strip().lower()
            if value:
                synonyms[mondo].add(value)
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

    @lru_cache(maxsize=None)
    def ancestors(mondo: str) -> frozenset[str]:
        out: set[str] = set()
        for parent in parents.get(mondo, set()):
            out.add(parent)
            out.update(ancestors(parent))
        return frozenset(out)

    children: dict[str, set[str]] = defaultdict(set)
    for child, parent_set in parents.items():
        for parent in parent_set:
            children[parent].add(child)

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
    candidate_name = str(mondo["names"].get(candidate, "")).lower()
    gold_name = str(mondo["names"].get(gold, "")).lower()
    if candidate_name and candidate_name == gold_name:
        return "synonym_or_name_match"
    if candidate_name and candidate_name in mondo["synonyms"].get(gold, set()):
        return "synonym_or_name_match"
    if gold_name and gold_name in mondo["synonyms"].get(candidate, set()):
        return "synonym_or_name_match"
    candidate_ancestors = set(mondo["ancestors"](candidate))
    gold_ancestors = set(mondo["ancestors"](gold))
    if candidate in gold_ancestors:
        return "candidate_ancestor_of_gold"
    if gold in candidate_ancestors:
        return "candidate_descendant_of_gold"
    if mondo["parents"].get(candidate, set()) & mondo["parents"].get(gold, set()):
        return "same_parent"
    if candidate_ancestors & gold_ancestors:
        return "shared_ancestor"
    return "unrelated_or_unknown"


def relation_priority(relation: str) -> int:
    order = {
        "same_disease": 0,
        "synonym_or_name_match": 1,
        "candidate_ancestor_of_gold": 2,
        "candidate_descendant_of_gold": 3,
        "same_parent": 4,
        "shared_ancestor": 5,
        "unrelated_or_unknown": 6,
    }
    return order.get(relation, 99)


def parse_jsonish_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    mimic = pd.read_csv(args.mimic_path, dtype=str)
    mimic["mondo_label"] = mimic["mondo_label"].map(normalize_mondo)
    exact = pd.read_csv(args.exact_details_path, dtype=str)
    exact["dataset_name"] = exact["dataset_name"].map(dataset_alias)
    exact = exact[exact["dataset_name"] == "mimic_test_recleaned_mondo_hpo_rows"].copy()
    exact["true_rank"] = pd.to_numeric(exact["true_rank"], errors="coerce").astype(int)
    exact["raw_case_id"] = exact["case_id"].map(raw_case_id)
    candidates = pd.read_csv(args.candidates_path, dtype=str)
    candidates["dataset_name"] = candidates["dataset_name"].map(dataset_alias)
    candidates = candidates[candidates["dataset_name"] == "mimic_test_recleaned_mondo_hpo_rows"].copy()
    candidates["original_rank"] = pd.to_numeric(candidates["original_rank"], errors="coerce").astype(int)
    numeric_cols = [
        "hgnn_score",
        "exact_overlap",
        "ic_weighted_overlap",
        "case_coverage",
        "disease_coverage",
        "disease_hpo_count",
        "shared_hpo_count",
        "jaccard_overlap",
        "semantic_ic_overlap",
        "semantic_coverage_score",
        "hgnn_margin",
        "max_exact_overlap_in_case",
        "max_ic_overlap_in_case",
        "evidence_rank_by_ic",
    ]
    for col in numeric_cols:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates["raw_case_id"] = candidates["case_id"].map(raw_case_id)
    return {
        "mimic": mimic,
        "exact": exact,
        "candidates": candidates.sort_values(["case_id", "original_rank"], kind="stable"),
        "exact_summary": read_json(args.exact_summary_path),
        "candidate_meta": read_json(args.candidate_metadata_path),
        "run_manifest": read_json(args.run_manifest_path),
        "data_config": read_yaml(DATA_CONFIG),
        "mainline_config": read_yaml(MAINLINE_CONFIG),
    }


def build_case_table(mimic: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in mimic.groupby("case_id", sort=False):
        labels = [normalize_mondo(x) for x in group["mondo_label"].dropna().astype(str).tolist()]
        unique_labels = sorted(set(labels), key=labels.index)
        hpos = sorted(set(group["hpo_id"].dropna().astype(str).tolist()))
        rows.append(
            {
                "raw_case_id": str(case_id),
                "exact_gold_id": unique_labels[0] if unique_labels else "",
                "all_gold_labels": unique_labels,
                "gold_label_count": int(len(unique_labels)),
                "label_subset": "multi-label" if len(unique_labels) > 1 else "single-label",
                "case_hpos": hpos,
                "case_hpo_count": int(len(hpos)),
            }
        )
    case_table = pd.DataFrame(rows)
    exact_view = exact[["raw_case_id", "case_id", "true_label", "pred_top1", "true_rank", "top5_labels"]].copy()
    exact_view["top5_labels_parsed"] = exact_view["top5_labels"].map(parse_jsonish_list)
    return case_table.merge(exact_view, on="raw_case_id", how="left")


def metric_from_ranks(ranks: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(ranks, errors="coerce").dropna().to_numpy(dtype=int)
    total = int(arr.size)
    if total == 0:
        return {k: np.nan for k in ["top1", "top3", "top5", "rank_le_10", "rank_le_20", "rank_le_50", "median_rank", "mean_rank"]} | {"num_cases": 0}
    return {
        "num_cases": total,
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "rank_le_10": float(np.mean(arr <= 10)),
        "rank_le_20": float(np.mean(arr <= 20)),
        "rank_le_50": float(np.mean(arr <= 50)),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
    }


def rank_histogram(ranks: pd.Series) -> pd.DataFrame:
    bins = [
        ("1", lambda x: x == 1),
        ("2-3", lambda x: (x >= 2) & (x <= 3)),
        ("4-5", lambda x: (x >= 4) & (x <= 5)),
        ("6-10", lambda x: (x >= 6) & (x <= 10)),
        ("11-20", lambda x: (x >= 11) & (x <= 20)),
        ("21-50", lambda x: (x >= 21) & (x <= 50)),
        ("51-100", lambda x: (x >= 51) & (x <= 100)),
        ("101-500", lambda x: (x >= 101) & (x <= 500)),
        (">500", lambda x: x > 500),
    ]
    arr = pd.to_numeric(ranks, errors="coerce").dropna().astype(int)
    total = max(1, len(arr))
    return pd.DataFrame(
        [
            {"rank_bin": name, "count": int(mask(arr).sum()), "ratio": float(mask(arr).sum() / total)}
            for name, mask in bins
        ]
    )


def write_rank_decomposition(case_table: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    ranks = case_table["true_rank"]
    metrics = metric_from_ranks(ranks)
    arr = pd.to_numeric(ranks, errors="coerce").dropna().astype(int)
    total = int(len(arr))
    extra = {
        "gold_absent_from_top50_count": int((arr > 50).sum()),
        "gold_absent_from_top50_ratio": float((arr > 50).mean()),
        "gold_in_top50_but_rank_gt5_count": int(((arr > 5) & (arr <= 50)).sum()),
        "gold_in_top50_but_rank_gt5_ratio": float(((arr > 5) & (arr <= 50)).mean()),
        "gold_in_top5_but_not_top1_count": int(((arr > 1) & (arr <= 5)).sum()),
        "gold_in_top5_but_not_top1_ratio": float(((arr > 1) & (arr <= 5)).mean()),
    }
    summary = {**metrics, **extra}
    hist = rank_histogram(ranks)
    out = pd.concat(
        [
            pd.DataFrame([{"section": "summary", **summary}]),
            hist.assign(section="histogram"),
        ],
        ignore_index=True,
        sort=False,
    )
    out.to_csv(report_dir / "mimic_rank_decomposition.csv", index=False, encoding="utf-8-sig")

    absent = extra["gold_absent_from_top50_ratio"]
    rank_late = extra["gold_in_top50_but_rank_gt5_ratio"]
    main_reason = "gold 不在 top50 的 candidate recall 问题更大" if absent >= rank_late else "gold 在 top50 但排序靠后更大"
    lines = [
        "# mimic_test Rank Decomposition",
        "",
        "## 输入与口径",
        f"- exact details: `{DEFAULT_EXACT}`",
        "- rank 使用 HGNN full-pool exact evaluation 的 `true_rank`，不是后处理后的 `mainline_final_case_ranks.csv`。",
        "- top50 candidate recall 等价于 `true_rank <= 50`。",
        "",
        "## Summary",
        to_markdown(pd.DataFrame([summary])),
        "",
        "## Rank Histogram",
        to_markdown(hist),
        "",
        "## 结论",
        f"- mimic_test 低准确率的首要拆分结论：{main_reason}。",
        f"- `gold_absent_from_top50` 为 {extra['gold_absent_from_top50_count']}/{total} ({extra['gold_absent_from_top50_ratio']:.4f})。",
        f"- `gold_in_top50_but_rank_gt5` 为 {extra['gold_in_top50_but_rank_gt5_count']}/{total} ({extra['gold_in_top50_but_rank_gt5_ratio']:.4f})。",
        "- 对 `gold_absent_from_top50` 样本，单纯 reranker 或 hard negative 只能重排已有候选，理论上不能直接解决这部分样本；需要 candidate expansion、ontology-aware retrieval、similar-case retrieval 或 label/HPO 修复先把 gold 放进候选池。",
    ]
    write_md(report_dir / "mimic_rank_decomposition.md", lines)
    return summary


def hit_at_candidates(group: pd.DataFrame, labels: set[str], k: int) -> bool:
    top = group[group["original_rank"] <= k]
    return bool(set(top["candidate_id"].astype(str)) & labels)


def write_multilabel_audit(case_table: pd.DataFrame, candidates: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    cand_by_case = {case_id: group for case_id, group in candidates.groupby("case_id", sort=False)}
    rows: list[dict[str, Any]] = []

    def add_metric(scope: str, subset: pd.DataFrame, mode: str) -> None:
        if mode in {"exact", "original_exact"}:
            for k in [1, 3, 5, 50]:
                rows.append(
                    {
                        "row_type": "metric",
                        "scope": scope,
                        "mode": mode,
                        "k": k,
                        "num_cases": int(len(subset)),
                        "hit_rate": float((subset["true_rank"].astype(int) <= k).mean()) if len(subset) else np.nan,
                    }
                )
        else:
            for k in [1, 3, 5, 50]:
                hits = []
                for row in subset.itertuples(index=False):
                    group = cand_by_case.get(str(row.case_id), pd.DataFrame())
                    hits.append(hit_at_candidates(group, set(row.all_gold_labels), k))
                rows.append(
                    {
                        "row_type": "metric",
                        "scope": scope,
                        "mode": mode,
                        "k": k,
                        "num_cases": int(len(subset)),
                        "hit_rate": float(np.mean(hits)) if hits else np.nan,
                    }
                )

    single = case_table[case_table["gold_label_count"] == 1].copy()
    multi = case_table[case_table["gold_label_count"] > 1].copy()
    add_metric("all_cases", case_table, "original_exact")
    add_metric("all_cases", case_table, "any_label")
    add_metric("single_label_subset", single, "original_exact")
    add_metric("multi_label_subset", multi, "original_exact")
    add_metric("multi_label_subset", multi, "any_label")

    dist = case_table["gold_label_count"].value_counts().sort_index()
    for label_count, count in dist.items():
        rows.append(
            {
                "row_type": "label_count_distribution",
                "scope": "all_cases",
                "mode": "",
                "k": int(label_count),
                "num_cases": int(count),
                "hit_rate": float(count / len(case_table)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(report_dir / "mimic_multilabel_audit.csv", index=False, encoding="utf-8-sig")

    pivot = out[out["row_type"] == "metric"].pivot_table(
        index=["scope", "mode", "num_cases"], columns="k", values="hit_rate"
    ).reset_index()
    pivot.columns = [f"top{c}" if isinstance(c, int) else str(c) for c in pivot.columns]
    exact_top1 = float(pivot[(pivot["scope"] == "all_cases") & (pivot["mode"] == "original_exact")]["top1"].iloc[0])
    any_top1 = float(pivot[(pivot["scope"] == "all_cases") & (pivot["mode"] == "any_label")]["top1"].iloc[0])
    exact_top5 = float(pivot[(pivot["scope"] == "all_cases") & (pivot["mode"] == "original_exact")]["top5"].iloc[0])
    any_top5 = float(pivot[(pivot["scope"] == "all_cases") & (pivot["mode"] == "any_label")]["top5"].iloc[0])
    lines = [
        "# mimic_test Multi-label Audit",
        "",
        "## Case Label Counts",
        f"- single-label case 数量：{len(single)}",
        f"- multi-label case 数量：{len(multi)}",
        f"- multi-label case 占比：{len(multi) / len(case_table):.4f}",
        "",
        to_markdown(
            dist.rename_axis("label_count").reset_index(name="case_count").assign(
                ratio=lambda df: df["case_count"] / len(case_table)
            )
        ),
        "",
        "## Metrics",
        to_markdown(pivot),
        "",
        "## 结论",
        f"- 当前 evaluation 在 `load_test_cases` 中按 `case_id` 聚合后使用 `group_df[label_col].iloc[0]`，因此多标签病例会被单标签化。",
        f"- any-label top1 相比 original exact top1 提升 {any_top1 - exact_top1:.4f}；any-label top5 提升 {any_top5 - exact_top5:.4f}。",
        "- 这说明当前 mimic_test 准确率存在一定低估，但低估幅度不足以解释全部低分。",
        "- any-label evaluation 不应作为论文主表；建议标注为 supplementary / error analysis，用于说明多标签病例的潜在假阴性。",
    ]
    write_md(report_dir / "mimic_multilabel_audit.md", lines)
    return {
        "single_label_cases": int(len(single)),
        "multi_label_cases": int(len(multi)),
        "multi_label_ratio": float(len(multi) / len(case_table)),
        "any_top1_delta": float(any_top1 - exact_top1),
        "any_top5_delta": float(any_top5 - exact_top5),
    }


def load_resources() -> dict[str, Any]:
    disease_index = pd.read_excel(DISEASE_INDEX, dtype={"mondo_id": str})
    hpo_index = pd.read_excel(HPO_INDEX, dtype={"hpo_id": str})
    hyper = pd.read_csv(HYPEREDGE_CSV, dtype={"mondo_id": str, "hpo_id": str})
    hyper["mondo_id"] = hyper["mondo_id"].map(normalize_mondo)
    disease_hpos = {
        str(mondo): set(group["hpo_id"].dropna().astype(str).tolist())
        for mondo, group in hyper.groupby("mondo_id", sort=False)
    }
    hpo_disease_counts = hyper[["mondo_id", "hpo_id"]].drop_duplicates().groupby("hpo_id")["mondo_id"].nunique()
    n_disease = max(1, int(hyper["mondo_id"].nunique()))
    hpo_specificity = {
        str(hpo): float(math.log((1.0 + n_disease) / (1.0 + count)) + 1.0)
        for hpo, count in hpo_disease_counts.items()
    }
    npz = np.load(INCIDENCE_NPZ, allow_pickle=True)
    return {
        "disease_index": disease_index,
        "hpo_index": hpo_index,
        "disease_ids": set(disease_index["mondo_id"].astype(str)),
        "hpo_ids": set(hpo_index["hpo_id"].astype(str)),
        "hyperedge_ids": set(disease_hpos),
        "disease_hpos": disease_hpos,
        "hpo_specificity": hpo_specificity,
        "npz_shape": tuple(int(x) for x in npz["shape"].tolist()),
    }


def write_mapping_audit(case_table: pd.DataFrame, resources: dict[str, Any], mondo: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    labels = sorted(set(label for labels in case_table["all_gold_labels"] for label in labels))
    rows: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    for label in labels:
        normalized = normalize_mondo(label)
        in_index = normalized in resources["disease_ids"]
        in_hyper = normalized in resources["hyperedge_ids"]
        obsolete = normalized in mondo["deprecated"]
        replacement = mondo["replacements"].get(normalized, "")
        normalization_fix = normalized != label and normalized in resources["disease_ids"]
        row = {
            "mondo_label": label,
            "normalized_label": normalized,
            "in_disease_index": bool(in_index),
            "in_disease_hyperedge": bool(in_hyper),
            "is_obsolete_mondo": bool(obsolete),
            "replacement_mondo": replacement,
            "normalization_fix_possible": bool(normalization_fix),
            "needs_manual_review": bool((not in_index) or obsolete or (replacement and replacement != normalized)),
        }
        rows.append(row)
        if not in_index or not in_hyper or obsolete or normalization_fix:
            unmapped.append(row)
    label_df = pd.DataFrame(rows)
    summary = {
        "unique_mondo_labels": int(len(labels)),
        "labels_in_disease_index": int(label_df["in_disease_index"].sum()),
        "labels_not_in_disease_index": int((~label_df["in_disease_index"]).sum()),
        "labels_in_disease_hyperedge": int(label_df["in_disease_hyperedge"].sum()),
        "labels_without_disease_hyperedge": int((~label_df["in_disease_hyperedge"]).sum()),
        "obsolete_mondo_labels": int(label_df["is_obsolete_mondo"].sum()),
        "normalization_fix_possible": int(label_df["normalization_fix_possible"].sum()),
        "needs_manual_review": int(label_df["needs_manual_review"].sum()),
    }
    audit = pd.concat(
        [
            pd.DataFrame([{"row_type": "summary", **summary}]),
            label_df.assign(row_type="per_label"),
        ],
        ignore_index=True,
        sort=False,
    )
    audit.to_csv(report_dir / "mimic_label_mapping_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unmapped).to_csv(report_dir / "mimic_unmapped_labels.csv", index=False, encoding="utf-8-sig")

    case_unaligned = case_table[
        case_table["all_gold_labels"].map(lambda labels: any(normalize_mondo(label) not in resources["disease_ids"] for label in labels))
    ]
    lines = [
        "# mimic_test Label / MONDO Mapping Audit",
        "",
        "## Summary",
        to_markdown(pd.DataFrame([summary])),
        "",
        "## 结论",
        f"- mimic_test unique MONDO labels 为 {summary['unique_mondo_labels']}；不在 Disease_index 的 label 为 {summary['labels_not_in_disease_index']}。",
        f"- 没有对应 disease hyperedge 的 label 为 {summary['labels_without_disease_hyperedge']}。",
        f"- 可通过 `MONDO_` 到 `MONDO:` normalization 修复的 label 为 {summary['normalization_fix_possible']}。",
        f"- 受 label 不对齐直接影响的病例数为 {len(case_unaligned)}；当前低准确率主要不能归因于 Disease_index 缺失。",
        "- synonym / replacement / parent-child 可以解释部分 exact miss，但不能直接改写主 exact metric；建议作为 relaxed MONDO evaluation 的 supplementary error analysis。",
        "- 需要人工确认的 label 见 `mimic_unmapped_labels.csv`。",
    ]
    write_md(report_dir / "mimic_label_mapping_audit.md", lines)
    return summary


def overlap_metrics(
    case_hpos: set[str],
    disease_hpos: set[str],
    specificity: dict[str, float],
    semantic_matcher: HpoSemanticMatcher,
) -> dict[str, Any]:
    shared = case_hpos & disease_hpos
    case_ic_total = sum(float(specificity.get(hpo, 1.0)) for hpo in case_hpos)
    shared_ic = sum(float(specificity.get(hpo, 1.0)) for hpo in shared)
    semantic = semantic_matcher.score(
        case_hpos=case_hpos,
        disease_hpos=disease_hpos,
        hpo_specificity=specificity,
    )
    return {
        "disease_hpo_count": int(len(disease_hpos)),
        "exact_hpo_overlap_count": int(len(shared)),
        "exact_hpo_overlap_ratio": float(len(shared) / len(case_hpos)) if case_hpos else 0.0,
        "ic_weighted_overlap": float(shared_ic / case_ic_total) if case_ic_total > 0 else 0.0,
        "semantic_overlap": float(semantic.get("semantic_ic_overlap", 0.0)),
    }


def hpo_count_bucket(count: int) -> str:
    if count <= 3:
        return "1-3"
    if count <= 6:
        return "4-6"
    if count <= 10:
        return "7-10"
    return ">10"


def overlap_count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    return ">3"


def disease_hpo_count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    return ">20"


def write_hpo_overlap_audit(
    case_table: pd.DataFrame,
    candidates: pd.DataFrame,
    resources: dict[str, Any],
    report_dir: Path,
) -> dict[str, Any]:
    semantic_matcher, semantic_meta = HpoSemanticMatcher.from_project(PROJECT_ROOT)
    top1_by_case = candidates[candidates["original_rank"] == 1].set_index("case_id")
    rows: list[dict[str, Any]] = []
    for row in case_table.itertuples(index=False):
        case_hpos = set(row.case_hpos)
        gold = normalize_mondo(row.exact_gold_id)
        gold_hpos = resources["disease_hpos"].get(gold, set())
        gold_metrics = overlap_metrics(case_hpos, gold_hpos, resources["hpo_specificity"], semantic_matcher)
        top1_prediction = str(row.pred_top1)
        top1_hpos = resources["disease_hpos"].get(top1_prediction, set())
        top1_metrics = overlap_metrics(case_hpos, top1_hpos, resources["hpo_specificity"], semantic_matcher)
        rows.append(
            {
                "case_id": row.raw_case_id,
                "namespaced_case_id": row.case_id,
                "gold_mondo": gold,
                "all_gold_labels": ";".join(row.all_gold_labels),
                "case_hpo_count": int(row.case_hpo_count),
                "gold_disease_hpo_count": int(gold_metrics["disease_hpo_count"]),
                "exact_hpo_overlap_count": int(gold_metrics["exact_hpo_overlap_count"]),
                "exact_hpo_overlap_ratio": float(gold_metrics["exact_hpo_overlap_ratio"]),
                "ic_weighted_overlap": float(gold_metrics["ic_weighted_overlap"]),
                "semantic_overlap": float(gold_metrics["semantic_overlap"]),
                "overlap_zero": bool(gold_metrics["exact_hpo_overlap_count"] == 0),
                "overlap_le_1": bool(gold_metrics["exact_hpo_overlap_count"] <= 1),
                "gold_rank": int(row.true_rank),
                "gold_in_top50": bool(int(row.true_rank) <= 50),
                "top1_prediction": top1_prediction,
                "top1_case_hpo_overlap_count": int(top1_metrics["exact_hpo_overlap_count"]),
                "top1_case_hpo_overlap_ratio": float(top1_metrics["exact_hpo_overlap_ratio"]),
                "top1_ic_weighted_overlap": float(top1_metrics["ic_weighted_overlap"]),
                "top1_gold_disease_hpo_overlap_diff": int(top1_metrics["exact_hpo_overlap_count"] - gold_metrics["exact_hpo_overlap_count"]),
                "top1_gold_ic_overlap_diff": float(top1_metrics["ic_weighted_overlap"] - gold_metrics["ic_weighted_overlap"]),
                "case_hpo_count_bucket": hpo_count_bucket(int(row.case_hpo_count)),
                "exact_hpo_overlap_count_bucket": overlap_count_bucket(int(gold_metrics["exact_hpo_overlap_count"])),
                "gold_disease_hpo_count_bucket": disease_hpo_count_bucket(int(gold_metrics["disease_hpo_count"])),
                "gold_in_top50_bucket": "yes" if int(row.true_rank) <= 50 else "no",
            }
        )
    case_level = pd.DataFrame(rows)
    case_level.to_csv(report_dir / "mimic_hpo_hyperedge_overlap_case_level.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    for bucket_col in [
        "case_hpo_count_bucket",
        "exact_hpo_overlap_count_bucket",
        "gold_disease_hpo_count_bucket",
        "gold_in_top50_bucket",
    ]:
        for bucket, group in case_level.groupby(bucket_col, sort=False):
            summary_rows.append(
                {
                    "bucket_type": bucket_col,
                    "bucket": bucket,
                    "num_cases": int(len(group)),
                    "top1": float((group["gold_rank"] <= 1).mean()),
                    "top5": float((group["gold_rank"] <= 5).mean()),
                    "rank_le_50": float(group["gold_in_top50"].mean()),
                    "median_rank": float(group["gold_rank"].median()),
                    "mean_rank": float(group["gold_rank"].mean()),
                    "overlap_zero_ratio": float(group["overlap_zero"].mean()),
                    "overlap_le_1_ratio": float(group["overlap_le_1"].mean()),
                    "mean_exact_overlap_count": float(group["exact_hpo_overlap_count"].mean()),
                    "mean_ic_weighted_overlap": float(group["ic_weighted_overlap"].mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "mimic_hpo_hyperedge_overlap_summary.csv", index=False, encoding="utf-8-sig")

    zero_ratio = float(case_level["overlap_zero"].mean())
    le1_ratio = float(case_level["overlap_le_1"].mean())
    lines = [
        "# mimic_test HPO 与 Disease Hyperedge Overlap Audit",
        "",
        f"- semantic_overlap 来源：`HpoSemanticMatcher.from_project`；metadata: `{json.dumps(semantic_meta, ensure_ascii=False)}`",
        "",
        "## Overall",
        to_markdown(
            pd.DataFrame(
                [
                    {
                        "num_cases": len(case_level),
                        "overlap_zero_count": int(case_level["overlap_zero"].sum()),
                        "overlap_zero_ratio": zero_ratio,
                        "overlap_le_1_count": int(case_level["overlap_le_1"].sum()),
                        "overlap_le_1_ratio": le1_ratio,
                        "mean_exact_overlap_count": float(case_level["exact_hpo_overlap_count"].mean()),
                        "mean_ic_weighted_overlap": float(case_level["ic_weighted_overlap"].mean()),
                        "mean_semantic_overlap": float(case_level["semantic_overlap"].mean()),
                    }
                ]
            )
        ),
        "",
        "## Buckets",
        to_markdown(summary),
        "",
        "## 结论",
        f"- overlap=0 的样本占比为 {zero_ratio:.4f}；overlap<=1 的样本占比为 {le1_ratio:.4f}。",
        "- 如果 overlap=0 或 overlap<=1 比例较高，首先指向数据/HPO 抽取/知识库覆盖问题，模型排序只能在候选已有有效证据时发挥作用。",
        "- 多视图超边信息可以直接利用 exact/IC/semantic overlap、MONDO ontology、synonym/xref 做 candidate expansion 和 evidence rerank，因此更直接。",
        "- 图对比学习在 low-overlap 样本上容易把错误或弱证据 pair 当作正负对放大，除非先做 label 清洗、低置信样本过滤和 validation-selected 采样。",
    ]
    write_md(report_dir / "mimic_hpo_hyperedge_overlap.md", lines)
    return {
        "overlap_zero_ratio": zero_ratio,
        "overlap_le_1_ratio": le1_ratio,
        "mean_exact_overlap_count": float(case_level["exact_hpo_overlap_count"].mean()),
        "mean_ic_weighted_overlap": float(case_level["ic_weighted_overlap"].mean()),
    }


def write_candidate_recall_audit(
    case_table: pd.DataFrame,
    candidates: pd.DataFrame,
    resources: dict[str, Any],
    mondo: dict[str, Any],
    report_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    case_labels = case_table.set_index("case_id")["all_gold_labels"].to_dict()
    exact_gold = case_table.set_index("case_id")["exact_gold_id"].to_dict()
    rank_map = case_table.set_index("case_id")["true_rank"].astype(int).to_dict()
    augmented = candidates.copy()
    augmented["all_gold_labels"] = augmented["case_id"].map(lambda cid: ";".join(case_labels.get(cid, [])))
    augmented["exact_gold_id"] = augmented["case_id"].map(lambda cid: exact_gold.get(cid, ""))
    augmented["gold_in_top50"] = augmented["case_id"].map(lambda cid: int(rank_map.get(cid, 999999)) <= 50)
    augmented["candidate_is_exact_gold"] = augmented["candidate_id"] == augmented["exact_gold_id"]
    augmented["relation_to_exact_gold"] = [
        relation_to_gold(candidate, gold, mondo)
        for candidate, gold in zip(augmented["candidate_id"].astype(str), augmented["exact_gold_id"].astype(str), strict=True)
    ]

    def best_any_relation(row: pd.Series) -> str:
        labels = [label for label in str(row["all_gold_labels"]).split(";") if label]
        if not labels:
            return "unrelated_or_unknown"
        relations = [relation_to_gold(str(row["candidate_id"]), label, mondo) for label in labels]
        return sorted(relations, key=relation_priority)[0]

    augmented["best_relation_to_any_gold"] = augmented.apply(best_any_relation, axis=1)
    relation_case_flags: dict[str, dict[str, bool]] = {}
    for case_id, group in augmented.groupby("case_id", sort=False):
        relations = set(group["best_relation_to_any_gold"].astype(str))
        relation_case_flags[case_id] = {
            "top50_has_gold": "same_disease" in relations,
            "top50_has_synonym": "synonym_or_name_match" in relations,
            "top50_has_parent": "candidate_ancestor_of_gold" in relations,
            "top50_has_child": "candidate_descendant_of_gold" in relations,
            "top50_has_sibling": "same_parent" in relations,
            "top50_has_shared_ancestor": "shared_ancestor" in relations,
        }
    for col in [
        "top50_has_gold",
        "top50_has_synonym",
        "top50_has_parent",
        "top50_has_child",
        "top50_has_sibling",
        "top50_has_shared_ancestor",
    ]:
        augmented[col] = augmented["case_id"].map(lambda cid: relation_case_flags.get(cid, {}).get(col, False))

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_dir / "mimic_top50_candidates_with_evidence.csv", index=False, encoding="utf-8-sig")

    absent_case_ids = set(case_table.loc[case_table["true_rank"].astype(int) > 50, "case_id"].astype(str))
    absent_aug = augmented[augmented["case_id"].isin(absent_case_ids)]
    case_flags = pd.DataFrame.from_dict(relation_case_flags, orient="index").reset_index(names="case_id")
    absent_flags = case_flags[case_flags["case_id"].isin(absent_case_ids)]
    relation_counts = augmented["best_relation_to_any_gold"].value_counts().rename_axis("relation").reset_index(name="candidate_count")
    absent_case_summary = {
        "gold_absent_cases": int(len(absent_case_ids)),
        "absent_top50_has_synonym_cases": int(absent_flags["top50_has_synonym"].sum()) if not absent_flags.empty else 0,
        "absent_top50_has_parent_cases": int(absent_flags["top50_has_parent"].sum()) if not absent_flags.empty else 0,
        "absent_top50_has_child_cases": int(absent_flags["top50_has_child"].sum()) if not absent_flags.empty else 0,
        "absent_top50_has_sibling_cases": int(absent_flags["top50_has_sibling"].sum()) if not absent_flags.empty else 0,
        "absent_top50_has_shared_ancestor_cases": int(absent_flags["top50_has_shared_ancestor"].sum()) if not absent_flags.empty else 0,
        "absent_top50_mean_max_shared_hpo_count": float(absent_aug.groupby("case_id")["shared_hpo_count"].max().mean()) if not absent_aug.empty else np.nan,
        "absent_top50_mean_max_ic_overlap": float(absent_aug.groupby("case_id")["ic_weighted_overlap"].max().mean()) if not absent_aug.empty else np.nan,
    }
    lines = [
        "# mimic_test Candidate Recall Audit",
        "",
        f"- candidate file: `{DEFAULT_CANDIDATES}`",
        f"- output: `{output_dir / 'mimic_top50_candidates_with_evidence.csv'}`",
        "",
        "## Relation Counts in Top50 Rows",
        to_markdown(relation_counts),
        "",
        "## Gold Absent Case Summary",
        to_markdown(pd.DataFrame([absent_case_summary])),
        "",
        "## 结论",
        "- gold 不在 top50 的 case 中，如果 top50 经常出现 parent/child/sibling/shared_ancestor，说明需要 candidate expansion / ontology-aware retrieval。",
        "- gold 在 top50 但排得低的 case，更适合 reranker / hard negative。",
        "- 如果 top50 完全不相关且 HPO evidence coverage 很低，优先怀疑 HPO 抽取、label mapping 或 MIMIC domain shift。",
    ]
    write_md(report_dir / "mimic_candidate_recall_audit.md", lines)
    return absent_case_summary


def write_eval_path_audit(inputs: dict[str, Any], case_table: pd.DataFrame, resources: dict[str, Any], report_dir: Path) -> None:
    summary = inputs["exact_summary"]
    run_manifest = inputs["run_manifest"]
    candidate_meta = inputs["candidate_meta"]
    data_config = inputs["data_config"]
    per_dataset = pd.DataFrame(summary.get("per_dataset", []))
    mimic_row = per_dataset[per_dataset["dataset_name"] == "mimic_test_recleaned_mondo_hpo_rows"]
    disease_missing = int((~case_table["exact_gold_id"].isin(resources["disease_ids"])).sum())
    multi_count = int((case_table["gold_label_count"] > 1).sum())
    lines = [
        "# mimic_test Evaluation Path Audit",
        "",
        "## 当前评估链路",
        "- evaluation script: `python -m src.evaluation.evaluator`",
        f"- data config: `{summary.get('data_config_path', DATA_CONFIG)}`",
        f"- train config: `{summary.get('train_config_path', '')}`",
        f"- checkpoint: `{summary.get('checkpoint_path', '')}`",
        f"- checkpoint_epoch: `{summary.get('checkpoint_epoch', '')}`",
        f"- candidate/top50 file: `{candidate_meta.get('output_path', DEFAULT_CANDIDATES)}`",
        f"- exact output dir: `{DEFAULT_EXACT.parent}`",
        f"- final mainline output: `{run_manifest.get('final_outputs', {})}`",
        "",
        "## 输入与字段",
        f"- mimic_test input: `{DEFAULT_MIMIC}`",
        f"- label 字段: `{data_config.get('label_col', 'mondo_label')}`",
        f"- HPO 字段: `{data_config.get('hpo_col', 'hpo_id')}`",
        f"- case_id 字段: `{data_config.get('case_id_col', 'case_id')}`",
        f"- num_cases: {len(case_table)}",
        f"- MONDO ID 不在 Disease_index 的 current exact gold cases: {disease_missing}",
        f"- multi-label cases: {multi_count}",
        "",
        "## 静态资源",
        f"- Disease_index: `{DISEASE_INDEX}`",
        f"- HPO_index: `{HPO_INDEX}`",
        f"- disease incidence / hyperedge matrix: `{INCIDENCE_NPZ}`",
        f"- disease hyperedge CSV: `{HYPEREDGE_CSV}`",
        f"- HPO ontology used by candidate evidence export: `{candidate_meta.get('semantic', {}).get('ontology_path', '')}`",
        "",
        "## 当前 exact metric 计算方式",
        "- `src.evaluation.evaluator.load_test_cases` 先按 namespaced `case_id` 聚合，`mondo_label` 使用 `group_df[label_col].iloc[0]`，HPO 使用该 case 的去重 HPO 列表。",
        "- `evaluate` 对每个 case 计算全 disease pool 分数，`torch.argsort(scores, descending=True)` 后定位 gold disease index，得到 1-indexed `true_rank`。",
        "- `top1/top3/top5/rank_le_50` 由 `true_rank <= k` 计算，是 strict exact MONDO ID hit，不接受 synonym、ancestor、descendant 或任一 secondary label。",
        "- 因此存在多标签被单标签化的问题；本次报告不改变主 exact metric，只做 supplementary audit。",
        "",
        "## mimic_test exact per-dataset row",
        to_markdown(mimic_row) if not mimic_row.empty else "_未找到 mimic_test 行_",
    ]
    write_md(report_dir / "mimic_eval_path_audit.md", lines)


def write_method_comparison(
    rank_summary: dict[str, Any],
    multilabel_summary: dict[str, Any],
    mapping_summary: dict[str, Any],
    overlap_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    report_dir: Path,
) -> pd.DataFrame:
    recall_gap = float(rank_summary["gold_absent_from_top50_ratio"])
    late_gap = float(rank_summary["gold_in_top50_but_rank_gt5_ratio"])
    low_overlap = float(overlap_summary["overlap_le_1_ratio"])
    rows = [
        {
            "method": "方法A: 多视图、提取超边信息",
            "target_problem": "candidate expansion, ontology-aware retrieval, evidence rerank, multi-label/relaxed audit",
            "expected_help_for_mimic_top1": "中等：可通过 overlap/IC/semantic/similar-case rerank 改善已有候选排序",
            "expected_help_for_mimic_top5": "较高：能把近邻和证据强候选推入 top5",
            "expected_help_for_recall50": "较高：candidate expansion / ontology view 可处理 gold_absent_from_top50",
            "requires_training": "no-train 优先；后续可 light-train",
            "modifies_encoder": "否",
            "risk_of_overfitting": "低到中；必须 validation-selected fixed test",
            "risk_under_noisy_labels": "中；可用规则过滤和 error analysis 降低",
            "implementation_cost": "低到中",
            "recommended_or_not": "推荐作为第一优先级",
            "reason": f"gold_absent_from_top50={recall_gap:.4f}, top50内rank>5={late_gap:.4f}, overlap<=1={low_overlap:.4f}；该方法能同时覆盖召回、排序、mapping和证据解释，且不需要改encoder。",
        },
        {
            "method": "方法B: 图对比学习",
            "target_problem": "representation learning, supervised/cross-view contrast, hard negative contrast",
            "expected_help_for_mimic_top1": "不确定：依赖正负对质量，短期不如候选扩展直接",
            "expected_help_for_mimic_top5": "中等：可能改善相似疾病区分",
            "expected_help_for_recall50": "低到中：若仍使用同一候选导出路径，不能保证召回缺失样本进入top50",
            "requires_training": "是",
            "modifies_encoder": "通常需要；若只加投影头也要训练主表示",
            "risk_of_overfitting": "中到高",
            "risk_under_noisy_labels": "高；multi-label和low-overlap会制造错误正负对",
            "implementation_cost": "高",
            "recommended_or_not": "不推荐作为第一步；可作为P4后续实验",
            "reason": f"multi-label ratio={multilabel_summary['multi_label_ratio']:.4f}, label manual-review={mapping_summary['needs_manual_review']}, overlap<=1={low_overlap:.4f}；在清洗和候选召回未解决前容易放大噪声。",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "method_suitability_comparison.csv", index=False, encoding="utf-8-sig")
    return df


def write_final_recommendation(
    rank_summary: dict[str, Any],
    multilabel_summary: dict[str, Any],
    mapping_summary: dict[str, Any],
    overlap_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    method_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    lines = [
        "# Recommended mimic_test Next Step",
        "",
        "## 1. 低准确率第一原因",
        f"- candidate recall 低是第一原因：`gold_absent_from_top50` = {rank_summary['gold_absent_from_top50_count']} / {rank_summary['num_cases']} ({rank_summary['gold_absent_from_top50_ratio']:.4f})。",
        f"- top50 内排序差是第二层原因：`gold_in_top50_but_rank_gt5` = {rank_summary['gold_in_top50_but_rank_gt5_count']} / {rank_summary['num_cases']} ({rank_summary['gold_in_top50_but_rank_gt5_ratio']:.4f})。",
        f"- multi-label evaluation 有低估：multi-label 占比 {multilabel_summary['multi_label_ratio']:.4f}，any-label top1 delta={multilabel_summary['any_top1_delta']:.4f}，top5 delta={multilabel_summary['any_top5_delta']:.4f}，但不能解释全部低分。",
        f"- label mapping 不是主因：不在 Disease_index 的 label 数为 {mapping_summary['labels_not_in_disease_index']}，缺 hyperedge 的 label 数为 {mapping_summary['labels_without_disease_hyperedge']}。",
        f"- HPO / disease hyperedge overlap 是关键支撑问题：overlap=0 ratio={overlap_summary['overlap_zero_ratio']:.4f}，overlap<=1 ratio={overlap_summary['overlap_le_1_ratio']:.4f}。",
        "- MIMIC domain shift 很可能存在，表现为临床病例 HPO 与 rare disease hyperedge 对齐弱、候选召回不足和多标签诊断粒度混杂。",
        "",
        "## 2. 是否优先做多视图、提取超边信息",
        "- 推荐优先做。第一阶段应做 no-train，而不是直接训练。",
        "- 优先视图：patient-HPO view、disease-HPO hyperedge view、MONDO ontology view、HPO IC weighting view、disease synonym/xref view。",
        "- 最重要特征：exact overlap、IC-weighted overlap、semantic overlap、case/disease coverage、MONDO parent/child/sibling/synonym relation、similar-case candidate evidence。",
        "- 应先做 candidate expansion / ontology-aware retrieval，再做 validation-selected fixed-test rerank；否则 gold 不在 top50 的样本无法被 reranker 修复。",
        "",
        "## 3. 是否优先做图对比学习",
        "- 不推荐作为第一步。原因是当前主要瓶颈包含 candidate recall 缺失、低 overlap 和多标签/label noise；图对比学习需要训练且依赖高质量正负对，风险更高。",
        "- 若后续做，需要先满足：label 清洗完成、validation-selected protocol 固定、low-overlap 样本过滤或降权、positive pair 不从 test-side exploratory 结果构造。",
        "- 可用正负对：patient-disease exact gold positive、multi-label any-gold positive、MONDO sibling/same-parent hard negative、top50-above-gold hard negative、HPO-overlap hard negative。",
        "- 避免噪声的方法：过滤 overlap=0/<=1 低置信样本；对 ancestor/descendant 关系使用 soft negative 或低权重；正负对只从 train/validation 构造；test 仅 fixed evaluation。",
        "",
        "## 4. 下一步实验路线",
        "1. P0：不训练可修复的问题：label normalization、obsolete/replacement audit、multi-label supplementary metric、命令和路径 manifest 固化。",
        "2. P1：no-train / validation-selected rerank：使用 validation 选权重，fixed test 只跑一次；输出 exact 主表和 supplementary any-label/relaxed 表。",
        "3. P2：多视图超边特征增强：加入 MONDO ontology expansion、synonym/xref expansion、IC/semantic overlap evidence。",
        "4. P3：轻量训练 reranker：只训练 top50/topK reranker，不改 HGNN encoder，严格 train/validation/test 分离。",
        "5. P4：图对比学习或 hard negative training：在 P0-P3 后做，使用清洗后的正负对和低噪声采样。",
        "",
        "## 5. 最终推荐",
        "- 当前 mimic_test 提升的第一优先级：选择“多视图、提取超边信息”。",
        "- 图对比学习保留为后续 P4 实验，不作为当前第一步。",
        "",
        "## Method Comparison",
        to_markdown(method_df),
        "",
        "## Reproducibility",
        "- 本报告命令：`D:\\python\\python.exe tools\\analysis\\mimic_diagnosis.py`",
        "- 未训练新模型；未修改 HGNN encoder；未覆盖 baseline、exact evaluation 或 mainline 输出。",
    ]
    write_md(report_dir / "recommended_mimic_next_step.md", lines)


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    case_table = build_case_table(inputs["mimic"], inputs["exact"])
    resources = load_resources()
    mondo = load_mondo_graph(MONDO_JSON)

    write_eval_path_audit(inputs, case_table, resources, args.report_dir)
    rank_summary = write_rank_decomposition(case_table, args.report_dir)
    multilabel_summary = write_multilabel_audit(case_table, inputs["candidates"], args.report_dir)
    mapping_summary = write_mapping_audit(case_table, resources, mondo, args.report_dir)
    overlap_summary = write_hpo_overlap_audit(case_table, inputs["candidates"], resources, args.report_dir)
    candidate_summary = write_candidate_recall_audit(
        case_table,
        inputs["candidates"],
        resources,
        mondo,
        args.report_dir,
        args.output_dir,
    )
    method_df = write_method_comparison(
        rank_summary,
        multilabel_summary,
        mapping_summary,
        overlap_summary,
        candidate_summary,
        args.report_dir,
    )
    write_final_recommendation(
        rank_summary,
        multilabel_summary,
        mapping_summary,
        overlap_summary,
        candidate_summary,
        method_df,
        args.report_dir,
    )

    manifest = {
        "command": "D:\\python\\python.exe tools\\analysis\\mimic_diagnosis.py",
        "inputs": {
            "mimic_path": str(args.mimic_path.resolve()),
            "exact_details_path": str(args.exact_details_path.resolve()),
            "candidates_path": str(args.candidates_path.resolve()),
            "disease_index": str(DISEASE_INDEX.resolve()),
            "hyperedge_csv": str(HYPEREDGE_CSV.resolve()),
            "mondo_json": str(MONDO_JSON.resolve()),
        },
        "outputs": [
            str((args.report_dir / "mimic_eval_path_audit.md").resolve()),
            str((args.report_dir / "mimic_rank_decomposition.csv").resolve()),
            str((args.report_dir / "mimic_rank_decomposition.md").resolve()),
            str((args.report_dir / "mimic_multilabel_audit.csv").resolve()),
            str((args.report_dir / "mimic_multilabel_audit.md").resolve()),
            str((args.report_dir / "mimic_label_mapping_audit.csv").resolve()),
            str((args.report_dir / "mimic_unmapped_labels.csv").resolve()),
            str((args.report_dir / "mimic_label_mapping_audit.md").resolve()),
            str((args.report_dir / "mimic_hpo_hyperedge_overlap_case_level.csv").resolve()),
            str((args.report_dir / "mimic_hpo_hyperedge_overlap_summary.csv").resolve()),
            str((args.report_dir / "mimic_hpo_hyperedge_overlap.md").resolve()),
            str((args.output_dir / "mimic_top50_candidates_with_evidence.csv").resolve()),
            str((args.report_dir / "mimic_candidate_recall_audit.md").resolve()),
            str((args.report_dir / "method_suitability_comparison.csv").resolve()),
            str((args.report_dir / "recommended_mimic_next_step.md").resolve()),
        ],
    }
    (args.report_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
