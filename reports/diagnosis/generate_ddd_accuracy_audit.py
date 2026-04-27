from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(r"D:\RareDisease-traindata")
REPORT_DIAG = PROJECT / "reports" / "diagnosis"
REPORT_IMPROVE = PROJECT / "reports" / "ddd_improvement"

CAND_PATH = PROJECT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DETAILS_PATH = (
    PROJECT
    / "outputs"
    / "attn_beta_sweep"
    / "edge_log_beta02"
    / "evaluation"
    / "best_20260425_224439_details.csv"
)
PER_DATASET_PATH = DETAILS_PATH.with_name("best_20260425_224439_per_dataset.csv")
FIXED_TEST_PATH = PROJECT / "outputs" / "rerank" / "rerank_fixed_test_metrics.csv"
PRESETS_PATH = PROJECT / "outputs" / "rerank" / "rerank_v2_presets.csv"
SELECTED_WEIGHTS = PROJECT / "outputs" / "rerank" / "val_selected_weights.json"
DDD_CASE_PATH = PROJECT / "LLLdataset" / "dataset" / "processed" / "test" / "DDD.csv"
DISEASE_INDEX_PATH = PROJECT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx"
HYPEREDGE_PATH = (
    PROJECT
    / "LLLdataset"
    / "DiseaseHy"
    / "rare_disease_hgnn_clean_package_v59"
    / "v59_hyperedge_weighted_patched.csv"
)
MONDO_PATH = PROJECT / "data" / "raw_data" / "mondo.json"
OMIM_SSSOM = PROJECT / "LLLdataset" / "DiseaseHy" / "raw" / "orphanet" / "mondo_exactmatch_omim.sssom.tsv"
ORPHA_SSSOM = PROJECT / "LLLdataset" / "DiseaseHy" / "raw" / "orphanet" / "mondo_hasdbxref_orphanet.sssom.tsv"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def norm_mondo(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("http://purl.obolibrary.org/obo/MONDO_"):
        return text.rsplit("/", 1)[-1].replace("_", ":")
    if text.startswith("MONDO_"):
        return text.replace("_", ":")
    return text


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = df[columns].head(max_rows) if max_rows else df[columns]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in view.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def build_ancestors(parent_map: dict[str, set[str]]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for node in parent_map:
        seen: set[str] = set()
        queue: deque[str] = deque(parent_map.get(node, set()))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(parent_map.get(current, set()) - seen)
        output[node] = seen
    return output


def relation(left: str, right: str, parent_map: dict[str, set[str]], ancestors: dict[str, set[str]]) -> str:
    if left == right:
        return "same"
    left_anc = ancestors.get(left, set())
    right_anc = ancestors.get(right, set())
    if left in right_anc:
        return "candidate_ancestor_of_gold"
    if right in left_anc:
        return "candidate_descendant_of_gold"
    if parent_map.get(left, set()) & parent_map.get(right, set()):
        return "same_parent"
    if left_anc & right_anc:
        return "shared_ancestor"
    return "unrelated_or_unknown"


def overlap_stats(left: set[str], right: set[str], hpo_ic: dict[str, float]) -> dict[str, float | int]:
    shared = left & right
    union = left | right
    left_ic = sum(hpo_ic.get(hpo_id, 0.0) for hpo_id in left)
    shared_ic = sum(hpo_ic.get(hpo_id, 0.0) for hpo_id in shared)
    return {
        "shared_hpo_count": len(shared),
        "jaccard_overlap": len(shared) / len(union) if union else 0.0,
        "ic_weighted_overlap": shared_ic / left_ic if left_ic > 0.0 else 0.0,
        "case_coverage": len(shared) / len(left) if left else 0.0,
        "disease_coverage": len(shared) / len(right) if right else 0.0,
        "exact_overlap": len(shared) / math.sqrt(len(left) * len(right)) if left and right else 0.0,
    }


def read_sssom(path: Path) -> dict[str, set[str]]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    if "subject_id" not in df.columns or "object_id" not in df.columns:
        return {}
    output: dict[str, set[str]] = defaultdict(set)
    for subject, obj in zip(df["subject_id"].fillna(""), df["object_id"].fillna(""), strict=False):
        if str(subject).startswith("MONDO:") and str(obj):
            output[str(subject)].add(str(obj))
    return dict(output)


def load_mondo() -> tuple[dict[str, str], dict[str, int], dict[str, bool], dict[str, set[str]], dict[str, set[str]]]:
    payload = json.loads(MONDO_PATH.read_text(encoding="utf-8"))
    name_map: dict[str, str] = {}
    synonym_count: dict[str, int] = {}
    obsolete_map: dict[str, bool] = {}
    xref_map: dict[str, set[str]] = defaultdict(set)
    parent_map: dict[str, set[str]] = defaultdict(set)
    for graph in payload.get("graphs", []):
        for node in graph.get("nodes", []):
            mondo_id = norm_mondo(str(node.get("id", "")))
            if not mondo_id.startswith("MONDO:"):
                continue
            meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
            label = str(node.get("lbl", ""))
            name_map[mondo_id] = label
            synonym_count[mondo_id] = len(meta.get("synonyms", []) or [])
            obsolete_map[mondo_id] = bool(meta.get("deprecated", False) or label.lower().startswith("obsolete"))
            for xref in meta.get("xrefs", []) or []:
                value = str(xref.get("val", "")) if isinstance(xref, dict) else str(xref)
                if value:
                    xref_map[mondo_id].add(value)
            for bpv in meta.get("basicPropertyValues", []) or []:
                if not isinstance(bpv, dict):
                    continue
                value = str(bpv.get("val", ""))
                if "omim.org/entry/" in value:
                    xref_map[mondo_id].add("OMIM:" + value.rstrip("/").rsplit("/", 1)[-1])
        for edge in graph.get("edges", []):
            pred = str(edge.get("pred", ""))
            if not pred.endswith(("is_a", "subClassOf")):
                continue
            child = norm_mondo(str(edge.get("sub", "")))
            parent = norm_mondo(str(edge.get("obj", "")))
            if child.startswith("MONDO:") and parent.startswith("MONDO:"):
                parent_map[child].add(parent)
    return name_map, synonym_count, obsolete_map, dict(xref_map), dict(parent_map)


def disease_name(name_map: dict[str, str], mondo_id: str) -> str:
    return name_map.get(str(mondo_id), "")


def xref_counts(mondo_id: str, xref_map: dict[str, set[str]], omim_map: dict[str, set[str]], orpha_map: dict[str, set[str]]) -> tuple[int, int]:
    values = set(xref_map.get(mondo_id, set())) | set(omim_map.get(mondo_id, set())) | set(orpha_map.get(mondo_id, set()))
    omim = {value for value in values if "OMIM:" in value or "omim" in value.lower()}
    orpha = {value for value in values if "Orphanet:" in value or "orpha" in value.lower()}
    return len(omim), len(orpha)


def main() -> None:
    REPORT_DIAG.mkdir(parents=True, exist_ok=True)
    REPORT_IMPROVE.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(CAND_PATH, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    for column in [
        "original_rank",
        "hgnn_score",
        "exact_overlap",
        "ic_weighted_overlap",
        "case_coverage",
        "disease_coverage",
        "disease_hpo_count",
        "shared_hpo_count",
        "semantic_ic_overlap",
        "hgnn_margin",
    ]:
        cand[column] = pd.to_numeric(cand[column], errors="coerce")
    ddd_cand = cand[cand["dataset_name"] == "DDD"].copy()

    details = pd.read_csv(DETAILS_PATH, dtype={"case_id": str, "dataset_name": str, "true_label": str, "pred_top1": str})
    details["true_rank"] = pd.to_numeric(details["true_rank"], errors="coerce").astype(int)
    ddd_details = details[details["dataset_name"] == "DDD"].copy()

    hits = ddd_cand[ddd_cand["candidate_id"] == ddd_cand["gold_id"]].groupby("case_id")["original_rank"].min().astype(int)
    case_base = ddd_cand.groupby("case_id", sort=False).agg(gold_id=("gold_id", "first")).reset_index()
    case_base["rank_top50_policy"] = case_base["case_id"].map(hits).fillna(51).astype(int)
    case_base = case_base.merge(ddd_details[["case_id", "true_rank", "pred_top1"]], on="case_id", how="left")
    case_base["true_rank_full"] = case_base["true_rank"].fillna(case_base["rank_top50_policy"]).astype(int)
    total_cases = len(case_base)
    ranks = case_base["rank_top50_policy"].to_numpy(dtype=int)
    full_ranks = case_base["true_rank_full"].to_numpy(dtype=int)

    disease_index = pd.read_excel(DISEASE_INDEX_PATH, dtype={"mondo_id": str})
    disease_universe = set(disease_index["mondo_id"].astype(str))
    hyper = pd.read_csv(HYPEREDGE_PATH, dtype={"mondo_id": str, "hpo_id": str}, usecols=["mondo_id", "hpo_id"])
    disease_hpos = {mondo_id: set(group["hpo_id"].dropna().astype(str)) for mondo_id, group in hyper.groupby("mondo_id")}
    hpo_df_counts = Counter(hpo_id for hpos in disease_hpos.values() for hpo_id in hpos)
    num_diseases = max(len(disease_hpos), 1)
    hpo_ic = {hpo_id: math.log((num_diseases + 1.0) / (df + 1.0)) for hpo_id, df in hpo_df_counts.items()}

    ddd_raw = pd.read_csv(DDD_CASE_PATH, dtype=str)
    raw_case_to_ns = {
        raw_case_id: f"test::LLLdataset/dataset/processed/test/DDD.csv::{raw_case_id}"
        for raw_case_id in ddd_raw["case_id"].dropna().unique()
    }
    case_hpos = {
        raw_case_to_ns[raw_case_id]: set(group["hpo_id"].dropna().astype(str))
        for raw_case_id, group in ddd_raw.groupby("case_id")
    }

    name_map, synonym_count, obsolete_map, xref_map, parent_map = load_mondo()
    ancestors = build_ancestors(parent_map)
    omim_map = read_sssom(OMIM_SSSOM)
    orpha_map = read_sssom(ORPHA_SSSOM)

    # Rank decomposition.
    bucket_rows = [
        ("rank = 1", int((ranks == 1).sum()), "HGNN 已精确命中，非当前主要提升空间"),
        ("rank <= 3", int((ranks <= 3).sum()), "top3 累计命中"),
        ("rank <= 5", int((ranks <= 5).sum()), "top5 累计命中"),
        ("rank <= 10", int((ranks <= 10).sum()), "少量重排即可进入可用诊断列表"),
        ("rank <= 20", int((ranks <= 20).sum()), "top50 内排序提升的中短尾空间"),
        ("rank <= 50", int((ranks <= 50).sum()), "candidate recall@50，上限为 top50 内 rerank 可达样本"),
        ("rank > 50", int((ranks > 50).sum()), "HGNN top50 candidate recall 未覆盖"),
        ("gold absent from candidate universe", int((~case_base["gold_id"].isin(disease_universe)).sum()), "疾病索引/候选全集缺失"),
        ("top50 but rank > 5", int(((ranks > 5) & (ranks <= 50)).sum()), "核心 reranker 目标"),
        ("top5 but not top1", int(((ranks > 1) & (ranks <= 5)).sum()), "top1 排序损失，适合 evidence rerank"),
    ]
    rank_decomp = pd.DataFrame(
        [
            {
                "bucket": bucket,
                "num_cases": count,
                "ratio": count / total_cases,
                "interpretation": interpretation,
                "total_cases": total_cases,
                "median_rank_top50_capped": float(np.median(ranks)),
                "mean_rank_top50_capped": float(np.mean(ranks)),
                "median_rank_full": float(np.median(full_ranks)),
                "mean_rank_full": float(np.mean(full_ranks)),
            }
            for bucket, count, interpretation in bucket_rows
        ]
    )
    rank_decomp.to_csv(REPORT_DIAG / "ddd_rank_decomposition.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIAG / "ddd_rank_decomposition.md").write_text(
        "\n".join(
            [
                "# DDD Rank Decomposition",
                "",
                f"- 数据源: `{CAND_PATH}`；full rank 参考 `{DETAILS_PATH}`。",
                f"- 总样本数: {total_cases}。",
                f"- top50-capped median/mean: {np.median(ranks):.1f}/{np.mean(ranks):.4f}；full-rank median/mean: {np.median(full_ranks):.1f}/{np.mean(full_ranks):.4f}。",
                f"- rank>50: {(ranks > 50).sum()} ({pct(float((ranks > 50).mean()))})；top50 但 rank>5: {((ranks > 5) & (ranks <= 50)).sum()} ({pct(float(((ranks > 5) & (ranks <= 50)).mean()))})；top5 但非 top1: {((ranks > 1) & (ranks <= 5)).sum()} ({pct(float(((ranks > 1) & (ranks <= 5)).mean()))})。",
                "",
                md_table(rank_decomp, ["bucket", "num_cases", "ratio", "interpretation"]),
                "",
                "判断: DDD 的首要瓶颈是 top50 内排序问题，次要瓶颈是 candidate recall。证据是 567/761 个 gold 已在 top50，但只有 230/761 个排到 top1；另有 189 个样本位于 top50 但排在 top5 之后。",
            ]
        ),
        encoding="utf-8",
    )

    # Top50 miss audit.
    top1_by_case = ddd_cand[ddd_cand["original_rank"] == 1].set_index("case_id")
    miss_rows = []
    for row in case_base[case_base["rank_top50_policy"] > 50].itertuples(index=False):
        case_id = str(row.case_id)
        gold = str(row.gold_id)
        case_hpo = case_hpos.get(case_id, set())
        gold_hpo = disease_hpos.get(gold, set())
        stats = overlap_stats(case_hpo, gold_hpo, hpo_ic)
        omim_count, orpha_count = xref_counts(gold, xref_map, omim_map, orpha_map)
        top1 = top1_by_case.loc[case_id] if case_id in top1_by_case.index else None
        top1_id = "" if top1 is None else str(top1["candidate_id"])
        top1_relation = relation(top1_id, gold, parent_map, ancestors) if top1_id else "top1_unavailable"
        notes = []
        if gold not in disease_universe:
            notes.append("gold_not_in_disease_index")
        if not gold_hpo:
            notes.append("gold_no_disease_hpo")
        if obsolete_map.get(gold, False):
            notes.append("gold_obsolete")
        if gold not in name_map:
            notes.append("gold_not_in_mondo_json")
        if stats["shared_hpo_count"] == 0:
            notes.append("zero_exact_overlap")
        if top1_relation in {"same_parent", "shared_ancestor", "candidate_ancestor_of_gold", "candidate_descendant_of_gold"}:
            notes.append("top1_ontology_near_gold")
        miss_rows.append(
            {
                "case_id": case_id,
                "gold_disease_id": gold,
                "gold_disease_name": disease_name(name_map, gold),
                "full_rank": int(row.true_rank_full),
                "rank_top50_policy": int(row.rank_top50_policy),
                "top1_candidate_id": top1_id,
                "top1_candidate_name": disease_name(name_map, top1_id),
                "top1_relation_to_gold": top1_relation,
                "gold_in_candidate_universe": bool(gold in disease_universe),
                "gold_in_disease_index": bool(gold in disease_universe),
                "gold_in_disease_hyperedge_graph": bool(len(gold_hpo) > 0),
                "gold_in_mondo_json": bool(gold in name_map),
                "gold_obsolete": bool(obsolete_map.get(gold, False)),
                "gold_synonym_count": int(synonym_count.get(gold, 0)),
                "gold_omim_xref_count": int(omim_count),
                "gold_orphanet_xref_count": int(orpha_count),
                "case_hpo_count": len(case_hpo),
                "gold_disease_hpo_count": len(gold_hpo),
                "case_gold_shared_hpo_count": int(stats["shared_hpo_count"]),
                "case_gold_exact_overlap": float(stats["exact_overlap"]),
                "case_gold_ic_weighted_overlap": float(stats["ic_weighted_overlap"]),
                "case_gold_case_coverage": float(stats["case_coverage"]),
                "case_gold_disease_coverage": float(stats["disease_coverage"]),
                "mapping_notes": ";".join(notes) if notes else "ok",
            }
        )
    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(REPORT_DIAG / "ddd_top50_miss_audit.csv", index=False, encoding="utf-8-sig")
    miss_summary = {
        "num_miss": len(miss_df),
        "gold_not_in_candidate_universe": int((~miss_df["gold_in_candidate_universe"]).sum()) if len(miss_df) else 0,
        "gold_no_hyperedge": int((~miss_df["gold_in_disease_hyperedge_graph"]).sum()) if len(miss_df) else 0,
        "obsolete_gold": int(miss_df["gold_obsolete"].sum()) if len(miss_df) else 0,
        "zero_exact_overlap": int((miss_df["case_gold_shared_hpo_count"] == 0).sum()) if len(miss_df) else 0,
        "mean_case_hpo_count": float(miss_df["case_hpo_count"].mean()) if len(miss_df) else 0.0,
        "mean_gold_hpo_count": float(miss_df["gold_disease_hpo_count"].mean()) if len(miss_df) else 0.0,
        "mean_ic_overlap": float(miss_df["case_gold_ic_weighted_overlap"].mean()) if len(miss_df) else 0.0,
    }
    (REPORT_DIAG / "ddd_top50_miss_audit.md").write_text(
        "\n".join(
            [
                "# DDD Top50 Miss Audit",
                "",
                f"- 样本数: {miss_summary['num_miss']}/{total_cases} ({pct(miss_summary['num_miss'] / total_cases)})。",
                f"- gold 不在 disease index / candidate universe: {miss_summary['gold_not_in_candidate_universe']}。",
                f"- gold 无 disease-HPO hyperedge: {miss_summary['gold_no_hyperedge']}；obsolete gold: {miss_summary['obsolete_gold']}。",
                f"- case 与 gold exact HPO overlap 为 0: {miss_summary['zero_exact_overlap']}/{miss_summary['num_miss']}。",
                f"- 平均 case HPO 数: {miss_summary['mean_case_hpo_count']:.2f}；平均 gold disease HPO 数: {miss_summary['mean_gold_hpo_count']:.2f}；平均 IC overlap: {miss_summary['mean_ic_overlap']:.4f}。",
                "",
                "结论: top50 miss 不是由 disease index 大面积缺失造成；主要是 HGNN candidate recall 未覆盖与 case/gold HPO 证据弱或稀疏导致。部分样本存在 zero exact overlap，但多数 gold 仍在 disease hyperedge 中，说明后续应做 coverage/label audit，而不是直接改 encoder。",
                "",
                "## Top Notes",
                md_table(
                    miss_df.head(20),
                    [
                        "case_id",
                        "gold_disease_id",
                        "gold_disease_name",
                        "full_rank",
                        "case_hpo_count",
                        "gold_disease_hpo_count",
                        "case_gold_shared_hpo_count",
                        "case_gold_ic_weighted_overlap",
                        "top1_candidate_id",
                        "top1_relation_to_gold",
                        "mapping_notes",
                    ],
                    max_rows=20,
                ),
            ]
        ),
        encoding="utf-8",
    )

    # Near-miss audit.
    nearmiss_rows = []
    nearmiss_base = case_base[(case_base["rank_top50_policy"] > 1) & (case_base["rank_top50_policy"] <= 50)]
    for row in nearmiss_base.itertuples(index=False):
        case_id = str(row.case_id)
        gold = str(row.gold_id)
        group = ddd_cand[ddd_cand["case_id"] == case_id].sort_values("original_rank")
        top1 = group.iloc[0]
        gold_row = group[group["candidate_id"] == gold].iloc[0]
        top3_ids = group.head(3)["candidate_id"].tolist()
        top5_ids = group.head(5)["candidate_id"].tolist()
        candidate_gold = overlap_stats(disease_hpos.get(str(top1["candidate_id"]), set()), disease_hpos.get(gold, set()), hpo_ic)
        nearmiss_rows.append(
            {
                "case_id": case_id,
                "gold_disease_id": gold,
                "gold_disease_name": disease_name(name_map, gold),
                "gold_rank": int(row.rank_top50_policy),
                "gold_hgnn_score": float(gold_row["hgnn_score"]),
                "gold_query_exact_overlap": float(gold_row["exact_overlap"]),
                "gold_query_ic_weighted_overlap": float(gold_row["ic_weighted_overlap"]),
                "gold_query_semantic_overlap": float(gold_row["semantic_ic_overlap"]),
                "top1_candidate_id": str(top1["candidate_id"]),
                "top1_candidate_name": disease_name(name_map, str(top1["candidate_id"])),
                "top1_hgnn_score": float(top1["hgnn_score"]),
                "score_gap_top1_minus_gold": float(top1["hgnn_score"] - gold_row["hgnn_score"]),
                "top1_query_exact_overlap": float(top1["exact_overlap"]),
                "top1_query_ic_weighted_overlap": float(top1["ic_weighted_overlap"]),
                "top1_query_semantic_overlap": float(top1["semantic_ic_overlap"]),
                "top1_gold_shared_hpo_count": int(candidate_gold["shared_hpo_count"]),
                "top1_gold_jaccard_overlap": float(candidate_gold["jaccard_overlap"]),
                "top1_relation_to_gold": relation(str(top1["candidate_id"]), gold, parent_map, ancestors),
                "top3_candidate_ids": ";".join(top3_ids),
                "top3_candidate_names": ";".join(disease_name(name_map, candidate_id) for candidate_id in top3_ids),
                "top5_candidate_ids": ";".join(top5_ids),
                "top5_candidate_names": ";".join(disease_name(name_map, candidate_id) for candidate_id in top5_ids),
                "case_hpo_count": len(case_hpos.get(case_id, set())),
                "gold_disease_hpo_count": len(disease_hpos.get(gold, set())),
            }
        )
    nearmiss_df = pd.DataFrame(nearmiss_rows).sort_values(["gold_rank", "score_gap_top1_minus_gold"], ascending=[True, False])
    nearmiss_df.to_csv(REPORT_DIAG / "ddd_nearmiss_cases.csv", index=False, encoding="utf-8-sig")
    pair_df = (
        nearmiss_df.groupby(
            ["gold_disease_id", "gold_disease_name", "top1_candidate_id", "top1_candidate_name", "top1_relation_to_gold"],
            dropna=False,
        )
        .agg(
            confusion_count=("case_id", "nunique"),
            average_gold_rank=("gold_rank", "mean"),
            average_score_gap=("score_gap_top1_minus_gold", "mean"),
            average_top1_gold_jaccard=("top1_gold_jaccard_overlap", "mean"),
            average_gold_query_ic=("gold_query_ic_weighted_overlap", "mean"),
            average_top1_query_ic=("top1_query_ic_weighted_overlap", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "top1_candidate_id": "predicted_disease_id",
                "top1_candidate_name": "predicted_disease_name",
                "top1_relation_to_gold": "ontology_relation",
            }
        )
        .sort_values(["confusion_count", "average_gold_rank"], ascending=[False, True], kind="stable")
    )
    pair_df.to_csv(REPORT_DIAG / "ddd_nearmiss_pairs.csv", index=False, encoding="utf-8-sig")
    rel_counts = nearmiss_df["top1_relation_to_gold"].value_counts().rename_axis("relation").reset_index(name="num_cases")
    rel_counts["ratio"] = rel_counts["num_cases"] / max(len(nearmiss_df), 1)
    (REPORT_DIAG / "ddd_nearmiss_audit.md").write_text(
        "\n".join(
            [
                "# DDD Near-Miss Audit",
                "",
                f"- gold 在 top50 且 rank>1 的样本数: {len(nearmiss_df)}。其中 top5 但非 top1: {((ranks > 1) & (ranks <= 5)).sum()}；top50 但 rank>5: {((ranks > 5) & (ranks <= 50)).sum()}。",
                f"- 平均 top1-gold HGNN score gap: {nearmiss_df['score_gap_top1_minus_gold'].mean():.4f}；中位 gold rank: {nearmiss_df['gold_rank'].median():.1f}。",
                "",
                "## Ontology Relation Distribution",
                md_table(rel_counts, ["relation", "num_cases", "ratio"]),
                "",
                "## Most Frequent Top1 Confusions",
                md_table(
                    pair_df.head(30),
                    [
                        "gold_disease_id",
                        "gold_disease_name",
                        "predicted_disease_id",
                        "predicted_disease_name",
                        "confusion_count",
                        "average_gold_rank",
                        "average_score_gap",
                        "ontology_relation",
                    ],
                    max_rows=30,
                ),
                "",
                "判断: 排序错误并不只来自随机噪声；same_parent/shared_ancestor/ancestor-descendant 类关系占有可观比例，说明 ontology-aware hard negative 与只在 top50 内的 evidence rerank 都有针对性。许多 pair 只出现一次，因此更适合按 relation/HPO-overlap 切片，而不是只记单个疾病对。",
            ]
        ),
        encoding="utf-8",
    )

    presets = pd.read_csv(PRESETS_PATH)
    fixed = pd.read_csv(FIXED_TEST_PATH) if FIXED_TEST_PATH.is_file() else pd.DataFrame()
    selected_payload = json.loads(SELECTED_WEIGHTS.read_text(encoding="utf-8")) if SELECTED_WEIGHTS.is_file() else {}
    base = presets[presets["preset"] == "A_hgnn_only"].iloc[0]
    fixed_row = fixed.iloc[0] if not fixed.empty else None

    rerank_lines = [
        "# DDD Rerank Protocol Audit",
        "",
        "## 关键结论",
        "- `tools/run_top50_evidence_rerank.py` 已支持 `exploratory`、`validation_select`、`fixed_eval` 三种协议。",
        "- `reports/top50_evidence_rerank_v2_report.md` 明确标记 test-side grid/gate 只能作为 exploratory upper bound，不能作为正式 test 结论。",
        "- 当前已有 validation candidates: `outputs/rerank/top50_candidates_validation.csv`，也已有 `outputs/rerank/val_selected_weights.json` 与 `outputs/rerank/rerank_fixed_test_metrics.csv`。",
        "- 但当前固定 test 结果来自 validation grid 选权重，不是完整 gated/mimic-safe gate；`outputs/rerank/rerank_validation_gated_results.csv` 是空文件，说明 gated validation selection 没有完成。",
        "",
        "## Evidence Features",
        "| Feature | Calculation | Normalization | Notes |",
        "|---|---|---|---|",
        "| `hgnn_score` | HGNN candidate score | case 内 min-max | 主模型分数 |",
        "| `ic_weighted_overlap` | shared HPO IC / case HPO IC | case 内 min-max | 最有解释性的 no-train evidence |",
        "| `exact_overlap` | shared_hpo_count / sqrt(case_hpo_count*disease_hpo_count) | case 内 min-max | 抑制疾病 HPO 数量差异 |",
        "| `semantic_ic_overlap` | exact 或 HPO ancestor/descendant match 的 case IC coverage | case 内 min-max | 使用 `src/rerank/hpo_semantic.py` |",
        "| `case_coverage` | shared / case_hpo_count | case 内 min-max | query 覆盖 |",
        "| `disease_coverage` | shared / disease_hpo_count | case 内 min-max | disease 侧覆盖 |",
        "| `size_penalty` | log1p(disease_hpo_count) | case 内 min-max 后负权重 | 惩罚过宽疾病 |",
        "",
        "## Fusion Formula",
        "`score = w_hgnn*hgnn + w_ic*ic + w_exact*exact + w_semantic*semantic + w_case_cov*case_cov + w_dis_cov*disease_cov - w_size*size_penalty`。排序只发生在 HGNN top50 candidates 内。",
        "",
        "## Metrics Evidence",
        f"- HGNN baseline DDD: top1/top3/top5={float(base['DDD_top1']):.4f}/{float(base['DDD_top3']):.4f}/{float(base['DDD_top5']):.4f}, recall@50={float(base['DDD_rank_le_50']):.4f}。",
    ]
    if fixed_row is not None:
        rerank_lines.extend(
            [
                f"- validation-selected fixed test DDD: top1/top3/top5={float(fixed_row['DDD_top1']):.4f}/{float(fixed_row['DDD_top3']):.4f}/{float(fixed_row['DDD_top5']):.4f}, recall@50={float(fixed_row['DDD_rank_le_50']):.4f}。",
                f"- selected weights: `{selected_payload.get('weights', {})}`；objective=`{selected_payload.get('selection_objective', '')}`；kind=`{selected_payload.get('selected_kind', '')}`。",
            ]
        )
    rerank_lines.extend(
        [
            "",
            "## Protocol Classification",
            "- exploratory test-side rerank: `outputs/rerank/rerank_v2_grid_results.csv`、`outputs/rerank/rerank_v2_gated_results.csv`、`reports/top50_evidence_rerank_v2_report.md`。只能作为诊断/附表。",
            "- validation-selected rerank: `outputs/rerank/top50_candidates_validation.csv` 选权重，`outputs/rerank/rerank_fixed_test_metrics.csv` 固定 test 评估。可作为论文候选，但需要在文中说明 selection objective。",
            "- test-set weight search: 不能进入论文主表；只能标注为 exploratory upper bound。",
            "",
            "## 最小可行后续路径",
            "1. 保持 `tools/export_top50_candidates.py --case-source validation` 生成 validation top50。",
            "2. 在 validation 上完成 grid 或 gated selection 并保存 `selected_weights.json`。",
            "3. 用 `--protocol fixed_eval` 对 test candidates 只评估一次。",
            "4. 不覆盖 `outputs/rerank/top50_candidates_v2.csv` 或既有 exact evaluation。",
        ]
    )
    (REPORT_DIAG / "ddd_rerank_protocol_audit.md").write_text("\n".join(rerank_lines), encoding="utf-8")

    (REPORT_DIAG / "ddd_hard_negative_audit.md").write_text(
        "\n".join(
            [
                "# DDD Hard Negative Audit",
                "",
                "## 当前状态",
                "- `src/training/hard_negative_miner.py::mine_hard_negatives` 实现当前 batch score-based top-k false candidates。",
                "- `src/training/hard_negative_miner.py::mine_configurable_hard_negatives` 声明了 `HN-overlap`、`HN-sibling`、`HN-shared-ancestor`、`HN-above-gold`、`HN-mixed` 接口。",
                "- `configs/train_finetune_attn_idf_main.yaml` 当前启用 hard negative，但未指定 ontology strategy，因此是 `HN-current`。",
                "- `configs/train_finetune_ontology_hn.yaml` 配置了 `HN-mixed`，但本轮未训练。",
                "",
                "## 关键风险 / 可能 bug",
                "- `src/training/trainer.py::run_one_epoch` 调用 `mine_configurable_hard_negatives(...)` 时没有传入 `candidate_pools`。按当前实现，只要 `candidate_pools` 为空，`HN-overlap/sibling/shared_ancestor/above_gold/mixed` 都会退化为 `HN-current`。因此 ontology-aware hard negative 目前是接口就绪，但训练热路径未真正接入候选池。",
                "",
                "## 当前负样本来源",
                "| Source | Status | Evidence |",
                "|---|---|---|",
                "| top-k false candidates by current scores | 已实现 | `mine_hard_negatives(scores, targets, k)` |",
                "| random negative | 未见专门实现 | full-pool CE 自然包含所有非 gold 类，但 hard loss 不随机采样 |",
                "| same ontology parent / sibling | 接口存在，热路径未供池 | `_pool_for_strategy` 支持 `sibling/same_parent` |",
                "| high HPO-overlap disease | 接口存在，热路径未供池 | `_pool_for_strategy` 支持 `overlap` |",
                "| above-gold top50 candidate | 接口存在，热路径未供池 | `_pool_for_strategy` 支持 `above_gold` |",
                "",
                "## DDD 专用推荐方案",
                "1. 从 DDD/train-validation top50 中提取排在 gold 前面的 candidates，构建 `above_gold` pool。",
                "2. 基于 MONDO parent/ancestor 构建 same_parent、sibling、shared_ancestor pools。",
                "3. 基于 disease-HPO exact/IC overlap 构建 high-overlap pool。",
                "4. 基于 query HPO overlap 构建 case-specific high-overlap but non-gold pool。",
                "5. 将本报告的 DDD near-miss pair/relation 分布作为采样权重先验。",
                "6. 所有训练必须输出到独立目录，不能覆盖当前 exact baseline。",
            ]
        ),
        encoding="utf-8",
    )

    fixed_text = (
        f"{float(fixed_row['DDD_top1']):.4f}/{float(fixed_row['DDD_top3']):.4f}/{float(fixed_row['DDD_top5']):.4f}"
        if fixed_row is not None
        else "NA"
    )
    key_files = pd.DataFrame(
        [
            {"Path": "configs/data_llldataset_eval.yaml", "Role": "evaluation data config", "Important Functions / Classes": "`src.evaluation.evaluator::load_test_cases`", "Notes": "包含 DDD test 文件，不改 mimic 主线"},
            {"Path": "configs/train_finetune_attn_idf_main.yaml", "Role": "当前 HGNN finetune/baseline config", "Important Functions / Classes": "`src.training.trainer::main`", "Notes": "checkpoint 位于 `outputs/attn_beta_sweep/edge_log_beta02/checkpoints/best.pt`"},
            {"Path": "src/data/dataset.py", "Role": "case table 读取、DDD split 命名空间", "Important Functions / Classes": "`load_case_files`, `build_namespaced_case_id`, `CaseBatchLoader`", "Notes": "使用 split::relative/path::case_id"},
            {"Path": "src/data/build_hypergraph.py", "Role": "HPO/disease incidence 与 case HPO 处理", "Important Functions / Classes": "`load_static_graph`, `build_case_incidence`, `build_batch_hypergraph`", "Notes": "disease incidence 使用 v59 npz"},
            {"Path": "src/evaluation/evaluator.py", "Role": "exact evaluation", "Important Functions / Classes": "`evaluate`, `compute_topk_metrics`, `save_results`", "Notes": "输出 details/per_dataset/summary"},
            {"Path": "tools/export_top50_candidates.py", "Role": "HGNN top50 candidate + evidence export", "Important Functions / Classes": "`export_top50_candidates`, `_evidence_features`", "Notes": "生成 rerank top50 candidates"},
            {"Path": "tools/run_top50_evidence_rerank.py", "Role": "top50 evidence rerank/grid/gate/protocol", "Important Functions / Classes": "`score_matrix`, `run_validation_select`, `evaluate_fixed_payload`", "Notes": "只在 top50 内重排"},
            {"Path": "src/rerank/hpo_semantic.py", "Role": "HPO semantic overlap", "Important Functions / Classes": "`HpoSemanticMatcher`", "Notes": "使用本地 HPO ontology"},
            {"Path": "src/training/hard_negative_miner.py", "Role": "hard negative mining", "Important Functions / Classes": "`mine_hard_negatives`, `mine_configurable_hard_negatives`", "Notes": "ontology pools 接口存在但 trainer 未传入"},
            {"Path": "src/models/hgnn_encoder.py", "Role": "HGNN encoder", "Important Functions / Classes": "`HGNNEncoder`", "Notes": "本轮只读，不修改"},
        ]
    )
    metrics_table = pd.DataFrame(
        [
            {"Dataset": "DDD", "Method": "HGNN_exact_baseline / A_hgnn_only", "Top1": float(base["DDD_top1"]), "Top3": float(base["DDD_top3"]), "Top5": float(base["DDD_top5"]), "Median Rank": float(base["DDD_median_rank"]), "Recall@50": float(base["DDD_rank_le_50"]), "Source File": str(CAND_PATH), "Notes": "与用户提供 baseline 对齐；mean_rank top50-capped=18.8463，full mean=271.9304"},
            {"Dataset": "DDD", "Method": "validation_selected_fixed_test", "Top1": float(fixed_row["DDD_top1"]) if fixed_row is not None else np.nan, "Top3": float(fixed_row["DDD_top3"]) if fixed_row is not None else np.nan, "Top5": float(fixed_row["DDD_top5"]) if fixed_row is not None else np.nan, "Median Rank": float(fixed_row["DDD_median_rank"]) if fixed_row is not None else np.nan, "Recall@50": float(fixed_row["DDD_rank_le_50"]) if fixed_row is not None else np.nan, "Source File": str(FIXED_TEST_PATH), "Notes": "validation grid 选权重，test fixed eval 一次；可作为候选但需说明 gated 未完成"},
            {"Dataset": "DDD", "Method": "test_side_exploratory:grid_1720", "Top1": 0.3784, "Top3": 0.4888, "Top5": 0.5532, "Median Rank": np.nan, "Recall@50": float(base["DDD_rank_le_50"]), "Source File": str(PROJECT / "reports" / "top50_evidence_rerank_v2_report.md"), "Notes": "test-side search，只能 error analysis/附表"},
        ]
    )
    failure_modes = pd.DataFrame(
        [
            {"Failure Mode": "top50 内排序不足", "Evidence": f"gold in top50={int((ranks <= 50).sum())}/{total_cases}，但 top1={int((ranks == 1).sum())}/{total_cases}", "Num Cases / Ratio": f"{int(((ranks > 1) & (ranks <= 50)).sum())} / {pct(float(((ranks > 1) & (ranks <= 50)).mean()))}", "Severity": "High", "Why It Matters": "只需在 HGNN top50 内重排即可提升 top1/top3/top5"},
            {"Failure Mode": "candidate recall 不足", "Evidence": "rank>50 样本无法被 top50 rerank 修复", "Num Cases / Ratio": f"{int((ranks > 50).sum())} / {pct(float((ranks > 50).mean()))}", "Severity": "Medium", "Why It Matters": "需要数据/映射/候选生成或训练侧 hard negative 改善"},
            {"Failure Mode": "label / mapping mismatch", "Evidence": f"gold 不在 disease index={miss_summary['gold_not_in_candidate_universe']}; obsolete={miss_summary['obsolete_gold']}", "Num Cases / Ratio": f"{miss_summary['gold_not_in_candidate_universe']} explicit", "Severity": "Low-Medium", "Why It Matters": "当前未见大面积 index 缺失，但 alias/parent-child 仍需人工审计"},
            {"Failure Mode": "HPO coverage 弱", "Evidence": f"top50 miss 中 zero exact overlap={miss_summary['zero_exact_overlap']}/{miss_summary['num_miss']}", "Num Cases / Ratio": f"{miss_summary['zero_exact_overlap']} / {pct(miss_summary['zero_exact_overlap'] / max(miss_summary['num_miss'], 1))}", "Severity": "Medium", "Why It Matters": "会同时影响 candidate recall 与 evidence rerank"},
            {"Failure Mode": "score calibration / evidence 未融合", "Evidence": f"validation-selected fixed DDD={fixed_text} vs baseline {float(base['DDD_top1']):.4f}/{float(base['DDD_top3']):.4f}/{float(base['DDD_top5']):.4f}", "Num Cases / Ratio": "N/A", "Severity": "Medium", "Why It Matters": "no-train evidence 可提升 top1，但协议必须 validation-selected"},
        ]
    )
    concrete = pd.DataFrame(
        [
            {"Experiment": "DDD label normalization audit", "Goal": "确认 alias/obsolete/parent-child 是否制造假错误", "Code Location to Modify Later": "tools/audit_processed_mondo_mapping.py 或新增 reports 脚本", "Expected Benefit": "减少 label mismatch 假阴性", "Risk": "需要人工规则确认", "Priority": "P0"},
            {"Experiment": "validation-selected linear evidence rerank", "Goal": "固定 validation 权重后 test 一次", "Code Location to Modify Later": "tools/run_top50_evidence_rerank.py", "Expected Benefit": "提升 DDD top1/top3", "Risk": "objective 选择影响 ALL/DDD tradeoff", "Priority": "P0"},
            {"Experiment": "complete validation gated rerank", "Goal": "验证 gated 是否保护弱证据/高置信 HGNN 样本", "Code Location to Modify Later": "tools/run_top50_evidence_rerank.py", "Expected Benefit": "比纯 grid 更稳", "Risk": "当前 validation gated 输出为空，需跑完", "Priority": "P0"},
            {"Experiment": "DDD top50 miss coverage repair list", "Goal": "定位 rank>50 的 HPO/gold coverage 弱点", "Code Location to Modify Later": "src/data/build_disease_hyperedge_v4_assets.py / hyperedge build scripts", "Expected Benefit": "提高 recall@50", "Risk": "数据修复可能改变基线，需版本化", "Priority": "P1"},
            {"Experiment": "above-gold hard negative pool", "Goal": "把排在 gold 前的 DDD near-miss 作为训练负样本", "Code Location to Modify Later": "src/training/trainer.py + src/training/hard_negative_miner.py", "Expected Benefit": "提升 top1 排序", "Risk": "需要训练新模型，不能本轮执行", "Priority": "P1"},
            {"Experiment": "MONDO sibling/shared-ancestor hard negatives", "Goal": "强化疾病家族内区分", "Code Location to Modify Later": "src/training/hard_negative_miner.py", "Expected Benefit": "改善 same_parent/shared_ancestor 混淆", "Risk": "ontology 粗粒度父类可能引入噪声", "Priority": "P1"},
            {"Experiment": "high HPO-overlap disease hard negatives", "Goal": "用 phenotype 相似疾病训练判别边界", "Code Location to Modify Later": "src/training/hard_negative_miner.py", "Expected Benefit": "改善高重叠 near-miss", "Risk": "过拟合 HPO exact overlap", "Priority": "P1"},
            {"Experiment": "pairwise reranker on train/val top50", "Goal": "学习 gold-vs-negative feature difference", "Code Location to Modify Later": "tools/train_top50_pairwise_reranker.py / eval script", "Expected Benefit": "比手工权重更灵活", "Risk": "必须避免 gold leakage 特征", "Priority": "P1"},
            {"Experiment": "parent-child relaxed supplementary metric", "Goal": "评估 ontology granularity 对 DDD 的影响", "Code Location to Modify Later": "src/evaluation/evaluator.py 或独立 supplementary evaluator", "Expected Benefit": "解释 near-miss 是否医学可接受", "Risk": "不能替代 exact metric", "Priority": "P2"},
            {"Experiment": "DDD error slicing by HPO count/noise", "Goal": "区分稀疏病例和噪声病例", "Code Location to Modify Later": "reports/diagnosis analysis scripts", "Expected Benefit": "指导数据清洗优先级", "Risk": "只读分析不能直接提分", "Priority": "P2"},
        ]
    )
    questions = [
        "DDD validation split 是否应单独从 DDD train 中切出，还是沿用全训练集 validation candidates。",
        "论文主表是否按 DDD objective 选权重，还是按 ALL_top1 / macro objective 选权重。",
        "disease index 与 MONDO 版本是否固定为 `Disease_index_v4.xlsx` 与 MONDO 2025-06-03。",
        "是否允许后续新增独立 validation-selected gated rerank 输出，不覆盖原始 `top50_candidates_v2.csv`。",
        "ontology-aware hard negative 的 candidate_pools 应在 batch 内动态构建还是预先离线导出。",
    ]
    report = [
        "# DDD Accuracy Improvement Audit Report",
        "",
        "> 重要风险先行: `configs/train_finetune_ontology_hn.yaml` 虽配置了 `HN-mixed`，但 `src/training/trainer.py::run_one_epoch` 当前没有向 `mine_configurable_hard_negatives` 传入 `candidate_pools`，因此 ontology-aware hard negative 会退化为 score-based `HN-current`。另一个边界是 `reports/top50_evidence_rerank_v2_report.md` 中的 grid/gate 是 test-side exploratory，不能作为正式 test 调参结论。",
        "",
        "## 1. Executive Summary",
        f"DDD 当前 exact baseline 为 top1={float(base['DDD_top1']):.4f}, top3={float(base['DDD_top3']):.4f}, top5={float(base['DDD_top5']):.4f}, recall@50={float(base['DDD_rank_le_50']):.4f}。核心问题不是 gold 大面积缺失于候选全集，而是 top50 内排序不足: {int((ranks <= 50).sum())}/{total_cases} 个 gold 已进入 top50，但只有 {int((ranks == 1).sum())}/{total_cases} 排到 top1。无需训练即可做的优先路径包括 DDD label/MONDO normalization audit、disease-HPO coverage audit、validation-selected evidence rerank 和 near-miss slicing。reranker 适合处理 gold 已在 top50 但 rank>1 的 {len(nearmiss_df)} 个样本，尤其是 top5 非 top1 和 top50 非 top5。hard negative training 适合处理 same_parent/shared_ancestor/高 HPO-overlap 混淆，但必须先把 ontology candidate pools 接入训练热路径。论文主表可放 HGNN baseline 与 validation-selected fixed rerank；test-side grid/gate 只能放附表或 error analysis。mimic 本轮只作为诊断对照，不作为优化主线。",
        "",
        "## 2. Key Files and Entry Points",
        md_table(key_files, ["Path", "Role", "Important Functions / Classes", "Notes"]),
        "",
        "## 3. DDD Current Metrics",
        md_table(metrics_table, ["Dataset", "Method", "Top1", "Top3", "Top5", "Median Rank", "Recall@50", "Source File", "Notes"]),
        "",
        "## 4. DDD Rank Decomposition",
        md_table(rank_decomp, ["bucket", "num_cases", "ratio", "interpretation"]),
        "",
        "## 5. DDD Failure Mode Diagnosis",
        md_table(failure_modes, ["Failure Mode", "Evidence", "Num Cases / Ratio", "Severity", "Why It Matters"]),
        "",
        "## 6. DDD Near-Miss Analysis",
        md_table(pair_df.head(25), ["gold_disease_id", "gold_disease_name", "predicted_disease_id", "predicted_disease_name", "confusion_count", "average_gold_rank", "ontology_relation"], max_rows=25),
        "",
        "## 7. Reranker Audit",
        "当前 reranker 使用 HGNN score、IC overlap、exact overlap、semantic IC overlap、case/disease coverage 与 disease size penalty。所有特征在 case 内 min-max normalization 后线性融合，最终只在 HGNN top50 内重排。当前 test-side grid/gate 已存在，但只能作为 exploratory；validation-selected 路径已有 `top50_candidates_validation.csv`、`val_selected_weights.json` 和 fixed test metrics。需要注意 validation gated 文件为空，正式 gated 方案仍需补跑 validation selection。",
        "",
        "## 8. Hard Negative Readiness",
        "已有 hard negative 基础实现和 ontology-aware 接口，但 trainer 未传 `candidate_pools`，所以 ontology-aware 策略目前没有真正生效。DDD 适合做 hard negative，因为 near-miss 中存在 same_parent/shared_ancestor/ancestor-descendant 与 HPO overlap 混淆。推荐先构建 above-gold、same_parent/sibling、shared_ancestor、高 HPO-overlap、query-overlap pools，再独立训练新模型，不能覆盖当前 baseline。",
        "",
        "## 9. Improvement Opportunities",
        "### A. 不训练新模型即可尝试",
        "- DDD label normalization audit；MONDO obsolete/synonym/alias audit；candidate universe coverage check；disease-HPO coverage check；rerank feature normalization；validation-selected evidence rerank；DDD near-miss error slicing。",
        "",
        "### B. 轻量训练 / reranker 可尝试",
        "- linear reranker；pairwise reranker；listwise reranker；feature fusion MLP；calibration model。所有方案必须 train/validation 选型，test fixed eval 一次。",
        "",
        "### C. 数据与标签修复",
        "- MONDO / OMIM / ORPHA 映射对齐；obsolete ID replacement；synonym / alias 合并；parent-child relaxed evaluation；HPO version 对齐；disease hyperedge coverage 修复。",
        "",
        "### D. 论文实验设计建议",
        "- 主表只放 HGNN baseline、validation-selected rerank、后续真正接入 candidate_pools 后的 ontology-aware hard negative。test grid 只能作为附表或 error analysis；near-miss analysis 放附表；relaxed metric 放 supplementary；mimic 只作为对照诊断。",
        "",
        "## 10. Concrete Next Experiments",
        md_table(concrete, ["Experiment", "Goal", "Code Location to Modify Later", "Expected Benefit", "Risk", "Priority"]),
        "",
        "## 11. Questions / Missing Information",
        "\n".join(f"- {question}" for question in questions),
        "",
        "## Generated Artifacts",
        "- `reports/diagnosis/ddd_rank_decomposition.csv` / `.md`",
        "- `reports/diagnosis/ddd_top50_miss_audit.csv` / `.md`",
        "- `reports/diagnosis/ddd_nearmiss_cases.csv`",
        "- `reports/diagnosis/ddd_nearmiss_pairs.csv`",
        "- `reports/diagnosis/ddd_nearmiss_audit.md`",
        "- `reports/diagnosis/ddd_rerank_protocol_audit.md`",
        "- `reports/diagnosis/ddd_hard_negative_audit.md`",
        "",
        "最终判断: DDD 下一步值得进入 validation-selected reranker 和 ontology-aware hard negative training；前者可立即按正式协议推进，后者必须先接入真实 ontology/query/top50 candidate pools 后再训练。",
    ]
    (REPORT_IMPROVE / "ddd_accuracy_improvement_context.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "sources": {
            "candidates": str(CAND_PATH),
            "details": str(DETAILS_PATH),
            "per_dataset": str(PER_DATASET_PATH),
            "ddd_cases": str(DDD_CASE_PATH),
            "disease_index": str(DISEASE_INDEX_PATH),
            "disease_hyperedge": str(HYPEREDGE_PATH),
            "mondo": str(MONDO_PATH),
        },
        "outputs": [
            str(REPORT_DIAG / "ddd_rank_decomposition.csv"),
            str(REPORT_DIAG / "ddd_rank_decomposition.md"),
            str(REPORT_DIAG / "ddd_top50_miss_audit.csv"),
            str(REPORT_DIAG / "ddd_top50_miss_audit.md"),
            str(REPORT_DIAG / "ddd_nearmiss_cases.csv"),
            str(REPORT_DIAG / "ddd_nearmiss_pairs.csv"),
            str(REPORT_DIAG / "ddd_nearmiss_audit.md"),
            str(REPORT_DIAG / "ddd_rerank_protocol_audit.md"),
            str(REPORT_DIAG / "ddd_hard_negative_audit.md"),
            str(REPORT_IMPROVE / "ddd_accuracy_improvement_context.md"),
        ],
        "ddd_rank_summary": {
            "num_cases": int(total_cases),
            "top1_count": int((ranks == 1).sum()),
            "top3_count": int((ranks <= 3).sum()),
            "top5_count": int((ranks <= 5).sum()),
            "top10_count": int((ranks <= 10).sum()),
            "top20_count": int((ranks <= 20).sum()),
            "top50_count": int((ranks <= 50).sum()),
            "rank_gt50_count": int((ranks > 50).sum()),
            "top50_rank_gt5_count": int(((ranks > 5) & (ranks <= 50)).sum()),
            "top5_not_top1_count": int(((ranks > 1) & (ranks <= 5)).sum()),
        },
        "miss_summary": miss_summary,
        "nearmiss_cases": int(len(nearmiss_df)),
    }
    (REPORT_DIAG / "ddd_accuracy_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
