from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_case_files
from src.evaluation.evaluator import load_static_resources, load_test_cases, load_yaml_config
from src.training.trainer import resolve_train_files, split_train_val_by_case


DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_finetune_attn_idf_main.yaml"
DEFAULT_TEST_CANDIDATES = PROJECT_ROOT / "outputs" / "mimic_next" / "top50_candidates_recleaned_test.csv"
DEFAULT_VAL_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_validation.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "mimic_next"
DEFAULT_FROZEN_CONFIG = DEFAULT_REPORT_DIR / "frozen_similar_case_aug_config.json"
MIMIC_PREFIX = "mimic_test"
DEEPRARE_TOP5_TARGET = 0.39


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mainline MIMIC HGNN + SimilarCase-Aug evaluation.")
    parser.add_argument("--similar-case-topk", default="3,5,10,20")
    parser.add_argument("--similar-case-weight", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--score-type", default="raw_similarity,rank_decay")
    parser.add_argument("--disable-gated-rerank", action="store_true")
    parser.add_argument("--gated-sim-weight", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--gated-ic-weight", default="0.0,0.05,0.1")
    parser.add_argument("--gated-agree-boost", default="0.0,0.05,0.1")
    parser.add_argument("--gated-protect-bonus", default="0.0,0.05,0.1")
    parser.add_argument("--use-frozen-config", action="store_true")
    parser.add_argument("--frozen-config-path", type=Path, default=DEFAULT_FROZEN_CONFIG)
    parser.add_argument("--data-config-path", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--train-config-path", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--validation-candidates-path", type=Path, default=DEFAULT_VAL_CANDIDATES)
    parser.add_argument("--test-candidates-path", type=Path, default=DEFAULT_TEST_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--similarity-device", default="auto", help="auto, cpu, cuda, or cuda:N for SimilarCase TF-IDF similarity.")
    parser.add_argument("--similarity-batch-size", type=int, default=256)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_score_types(value: str) -> list[str]:
    allowed = {"raw_similarity", "rank_decay", "similarity_times_label_frequency_penalty"}
    scores = [part.strip() for part in str(value).split(",") if part.strip()]
    invalid = [score for score in scores if score not in allowed]
    if invalid:
        raise ValueError(f"Unsupported score_type: {invalid}")
    return scores


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无记录_"
    use_df = df.head(max_rows).copy() if max_rows else df.copy()
    use_df = use_df.fillna("").astype(str)
    cols = list(use_df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in use_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def hpo_doc(hpos: list[str] | set[str]) -> str:
    return " ".join(sorted({str(hpo).strip() for hpo in hpos if str(hpo).strip()}))


def metric_from_ranks(ranks: list[int] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(ranks, dtype=int)
    return {
        "total_cases": int(arr.size),
        "top1": float(np.mean(arr <= 1)),
        "top3": float(np.mean(arr <= 3)),
        "top5": float(np.mean(arr <= 5)),
        "rank_le_50": float(np.mean(arr <= 50)),
        "rank_gt_50_cases": int(np.sum(arr > 50)),
        "gold_in_top50_but_rank_gt5_cases": int(np.sum((arr > 5) & (arr <= 50))),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
    }


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    for col in ["original_rank", "hgnn_score", "ic_weighted_overlap", "exact_overlap", "shared_hpo_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_case_tables(data_config: dict[str, Any], data_config_path: Path, train_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))
    train_files = resolve_train_files(train_config["paths"])
    all_train = load_case_files(
        file_paths=[str(path) for path in train_files],
        case_id_col=case_id_col,
        label_col=label_col,
        disease_index_path=train_config["paths"]["disease_index_path"],
        split_namespace="train",
    )
    train_df, val_df = split_train_val_by_case(
        all_train,
        val_ratio=float(train_config["data"]["val_ratio"]),
        random_seed=int(train_config["data"]["random_seed"]),
        case_id_col=case_id_col,
    )
    test_bundle = load_test_cases(data_config, data_config_path)
    test_df = test_bundle["raw_df"].copy()
    test_df = test_df[test_df["_source_file"].astype(str).apply(lambda value: Path(value).stem.startswith(MIMIC_PREFIX))]

    def to_table(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for case_id, group in df.groupby(case_id_col, sort=False):
            labels = sorted(set(group[label_col].dropna().astype(str).tolist()))
            hpos = sorted(set(group[hpo_col].dropna().astype(str).tolist()))
            rows.append(
                {
                    "case_id": str(case_id),
                    "primary_label": str(group[label_col].iloc[0]),
                    "label_set": labels,
                    "hpo_ids": hpos,
                    "hpo_count": len(hpos),
                    "label_count": len(labels),
                    "source_file": Path(str(group["_source_file"].iloc[0])).name if "_source_file" in group else "",
                }
            )
        return pd.DataFrame(rows)

    return to_table(train_df), to_table(val_df), to_table(test_df)


def build_disease_overlap_resources(train_config: dict[str, Any]) -> dict[str, set[str]]:
    resources = load_static_resources(train_config)
    matrix = resources["H_disease"].tocsc()
    hpo_df = resources["hpo_index_df"]
    disease_df = resources["disease_index_df"]
    idx_to_hpo = dict(zip(hpo_df["hpo_idx"].astype(int), hpo_df["hpo_id"].astype(str)))
    idx_to_disease = dict(zip(disease_df["disease_idx"].astype(int), disease_df["mondo_id"].astype(str)))
    disease_hpos: dict[str, set[str]] = {}
    for col_idx in range(matrix.shape[1]):
        disease = idx_to_disease[col_idx]
        disease_hpos[disease] = {idx_to_hpo[int(idx)] for idx in matrix[:, col_idx].indices if int(idx) in idx_to_hpo}
    return disease_hpos


def build_overlap_metric_resources(train_config: dict[str, Any]) -> dict[str, Any]:
    disease_hpos = build_disease_overlap_resources(train_config)
    hpo_disease_counts: Counter[str] = Counter()
    for hpos in disease_hpos.values():
        for hpo in hpos:
            hpo_disease_counts[hpo] += 1
    n_disease = max(1, len(disease_hpos))
    hpo_ic = {
        hpo: float(math.log((n_disease + 1.0) / (count + 1.0)) + 1.0)
        for hpo, count in hpo_disease_counts.items()
    }
    return {"disease_hpos": disease_hpos, "hpo_ic": hpo_ic}


def overlap_metrics(case_hpos: set[str], disease_hpos: set[str], hpo_ic: dict[str, float]) -> dict[str, Any]:
    shared = case_hpos & disease_hpos
    case_ic = sum(float(hpo_ic.get(hpo, 1.0)) for hpo in case_hpos)
    shared_ic = sum(float(hpo_ic.get(hpo, 1.0)) for hpo in shared)
    return {
        "exact_overlap": float(len(shared) / len(case_hpos)) if case_hpos else 0.0,
        "ic_weighted_overlap": float(shared_ic / case_ic) if case_ic > 0 else 0.0,
        "shared_hpo_count": int(len(shared)),
    }


def resolve_similarity_device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("similarity-device requested CUDA, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def append_similar_rows(
    rows: list[dict[str, Any]],
    *,
    query: Any,
    indices: np.ndarray,
    scores: np.ndarray,
    train_ids: list[str],
    train_labels: list[str],
    train_label_counts: Counter,
) -> None:
    ranked = [(int(idx), float(score)) for idx, score in zip(indices, scores, strict=True) if float(score) > 0.0]
    for rank, (idx, score) in enumerate(ranked, start=1):
        label = train_labels[idx]
        rows.append(
            {
                "case_id": query.case_id,
                "query_primary_label": query.primary_label,
                "matched_case_id": train_ids[idx],
                "matched_label": label,
                "similar_rank": rank,
                "raw_similarity": score,
                "rank_decay": float(score / math.sqrt(rank)),
                "similarity_times_label_frequency_penalty": float(score / math.sqrt(train_label_counts[label])),
            }
        )


def compute_similar_matches_cpu(
    query_x: Any,
    lib_x: Any,
    query_table: pd.DataFrame,
    train_ids: list[str],
    train_labels: list[str],
    train_label_counts: Counter,
    max_topk: int,
) -> pd.DataFrame:
    sim = (query_x @ lib_x.T).tocsr()
    rows: list[dict[str, Any]] = []
    for row_idx, query in enumerate(query_table.itertuples(index=False)):
        scores = sim.getrow(row_idx).toarray().ravel()
        if scores.size == 0:
            continue
        top_idx = np.argpartition(-scores, min(max_topk, len(scores) - 1))[:max_topk]
        order = top_idx[np.argsort(-scores[top_idx])]
        append_similar_rows(
            rows,
            query=query,
            indices=order,
            scores=scores[order],
            train_ids=train_ids,
            train_labels=train_labels,
            train_label_counts=train_label_counts,
        )
    return pd.DataFrame(rows)


def compute_similar_matches_cuda(
    query_x: Any,
    lib_x: Any,
    query_table: pd.DataFrame,
    train_ids: list[str],
    train_labels: list[str],
    train_label_counts: Counter,
    max_topk: int,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    lib_coo = lib_x.tocoo()
    indices_np = np.vstack([lib_coo.row, lib_coo.col]).astype(np.int64, copy=False)
    indices = torch.from_numpy(indices_np).to(device=device)
    values = torch.from_numpy(lib_coo.data.astype(np.float32, copy=False)).to(device=device)
    lib_sparse = torch.sparse_coo_tensor(indices, values, lib_coo.shape, device=device).coalesce()

    rows: list[dict[str, Any]] = []
    queries = list(query_table.itertuples(index=False))
    k = min(int(max_topk), int(lib_x.shape[0]))
    print(
        f"[similarity] CUDA sparse mm: query_cases={len(queries)} train_cases={lib_x.shape[0]} vocab={lib_x.shape[1]} topk={k} batch_size={batch_size}",
        flush=True,
    )
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            query_dense_np = query_x[start:end].toarray().astype(np.float32, copy=False)
            query_dense = torch.from_numpy(query_dense_np).to(device=device)
            scores = torch.sparse.mm(lib_sparse, query_dense.T).T
            top_scores, top_indices = torch.topk(scores, k=k, dim=1)
            top_scores_np = top_scores.detach().cpu().numpy()
            top_indices_np = top_indices.detach().cpu().numpy()
            for local_idx, query in enumerate(queries[start:end]):
                append_similar_rows(
                    rows,
                    query=query,
                    indices=top_indices_np[local_idx],
                    scores=top_scores_np[local_idx],
                    train_ids=train_ids,
                    train_labels=train_labels,
                    train_label_counts=train_label_counts,
                )
            print(f"[similarity] processed {end}/{len(queries)} query cases on {device}", flush=True)
            del query_dense, scores, top_scores, top_indices
    return pd.DataFrame(rows)


def compute_similar_matches(
    query_table: pd.DataFrame,
    train_table: pd.DataFrame,
    max_topk: int,
    *,
    device_name: str = "auto",
    batch_size: int = 256,
    label: str = "query",
) -> pd.DataFrame:
    lib_docs = [hpo_doc(hpos) for hpos in train_table["hpo_ids"].tolist()]
    query_docs = [hpo_doc(hpos) for hpos in query_table["hpo_ids"].tolist()]
    device = resolve_similarity_device(device_name)
    print(
        f"[similarity:{label}] vectorizing query_cases={len(query_docs)} train_cases={len(lib_docs)} device={device}",
        flush=True,
    )
    vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+")
    matrix = vectorizer.fit_transform(lib_docs + query_docs).astype(np.float32)
    lib_x = matrix[: len(lib_docs)]
    query_x = matrix[len(lib_docs):]
    train_ids = train_table["case_id"].astype(str).tolist()
    train_labels = train_table["primary_label"].astype(str).tolist()
    train_label_counts = Counter(train_labels)

    if device.type == "cuda":
        try:
            return compute_similar_matches_cuda(
                query_x,
                lib_x,
                query_table,
                train_ids,
                train_labels,
                train_label_counts,
                max_topk,
                device,
                max(1, int(batch_size)),
            )
        except RuntimeError as exc:
            if str(device_name).strip().lower() != "auto":
                raise
            print(f"[WARN] CUDA SimilarCase failed, falling back to CPU: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[similarity:{label}] using CPU sparse similarity", flush=True)
    return compute_similar_matches_cpu(
        query_x,
        lib_x,
        query_table,
        train_ids,
        train_labels,
        train_label_counts,
        max_topk,
    )


def hgnn_source(candidates: pd.DataFrame, case_table: pd.DataFrame) -> pd.DataFrame:
    valid_cases = set(case_table["case_id"].astype(str))
    df = candidates[candidates["case_id"].isin(valid_cases)].copy()
    max_rank = df.groupby("case_id")["original_rank"].transform("max").clip(lower=1)
    df["hgnn_component"] = 1.0 - ((df["original_rank"] - 1) / max_rank)
    df["similar_component"] = 0.0
    return df[["case_id", "gold_id", "candidate_id", "hgnn_component", "similar_component", "hgnn_score", "ic_weighted_overlap", "exact_overlap"]]


def similar_source(similar_matches: pd.DataFrame, topk: int, score_type: str, case_table: pd.DataFrame) -> pd.DataFrame:
    labels = {row.case_id: row.primary_label for row in case_table.itertuples(index=False)}
    subset = similar_matches[similar_matches["similar_rank"] <= topk].copy()
    if subset.empty:
        return pd.DataFrame(columns=["case_id", "gold_id", "candidate_id", "similar_component", "matched_case_ids", "matched_labels", "similar_case_source_score"])
    rows = []
    for (case_id, label), group in subset.groupby(["case_id", "matched_label"], sort=False):
        idx = group[score_type].astype(float).idxmax()
        best = group.loc[idx]
        ordered = group.sort_values(score_type, ascending=False)
        rows.append(
            {
                "case_id": case_id,
                "gold_id": labels.get(case_id, ""),
                "candidate_id": label,
                "similar_component": float(best[score_type]),
                "matched_case_ids": "|".join(ordered["matched_case_id"].head(5).astype(str).tolist()),
                "matched_labels": "|".join(ordered["matched_label"].head(5).astype(str).tolist()),
                "similar_case_source_score": float(best[score_type]),
            }
        )
    out = pd.DataFrame(rows)
    max_score = out.groupby("case_id")["similar_component"].transform("max").replace(0, 1.0)
    out["similar_component"] = out["similar_component"] / max_score
    return out


def combine(
    hgnn: pd.DataFrame,
    similar: pd.DataFrame,
    *,
    similar_weight: float,
    use_similar_case: bool = True,
    keep_evidence: bool = True,
) -> pd.DataFrame:
    rows = hgnn.copy()
    if use_similar_case and not similar.empty:
        rows = pd.concat(
            [
                rows,
                similar.assign(hgnn_component=0.0, hgnn_score=0.0, ic_weighted_overlap=0.0, exact_overlap=0.0),
            ],
            ignore_index=True,
            sort=False,
        )
    rows["similar_component"] = pd.to_numeric(rows.get("similar_component", 0.0), errors="coerce").fillna(0.0)
    rows["hgnn_component"] = pd.to_numeric(rows.get("hgnn_component", 0.0), errors="coerce").fillna(0.0)
    for col in ["matched_case_ids", "matched_labels"]:
        if col not in rows.columns:
            rows[col] = ""
    if "similar_case_source_score" not in rows.columns:
        rows["similar_case_source_score"] = 0.0
    rows["similar_case_source_score"] = pd.to_numeric(rows["similar_case_source_score"], errors="coerce").fillna(0.0)
    rows["score"] = rows["hgnn_component"] + similar_weight * rows["similar_component"]
    agg_spec: dict[str, Any] = {
        "score": ("score", "sum"),
        "hgnn_component": ("hgnn_component", "max"),
        "similar_component": ("similar_component", "max"),
        "similar_case_source_score": ("similar_case_source_score", "max"),
    }
    if keep_evidence:
        agg_spec.update(
            {
                "matched_case_ids": ("matched_case_ids", lambda values: "|".join(sorted({v for v in values.dropna().astype(str) if v}))),
                "matched_labels": ("matched_labels", lambda values: "|".join(sorted({v for v in values.dropna().astype(str) if v}))),
            }
        )
    agg = rows.groupby(["case_id", "gold_id", "candidate_id"], sort=False).agg(**agg_spec).reset_index()
    if not keep_evidence:
        agg["matched_case_ids"] = ""
        agg["matched_labels"] = ""
    agg = agg.sort_values(["case_id", "score", "hgnn_component"], ascending=[True, False, False], kind="stable")
    agg["rank"] = agg.groupby("case_id").cumcount() + 1
    return agg


def evaluate(ranked: pd.DataFrame, case_table: pd.DataFrame, method: str) -> tuple[dict[str, Any], pd.DataFrame]:
    case_meta = case_table[["case_id", "primary_label", "label_set"]].copy()
    case_meta["case_id"] = case_meta["case_id"].astype(str)
    case_meta["primary_label"] = case_meta["primary_label"].astype(str)
    rank_view = ranked[["case_id", "candidate_id", "rank"]].copy()
    rank_view["case_id"] = rank_view["case_id"].astype(str)
    rank_view["candidate_id"] = rank_view["candidate_id"].astype(str)
    rank_view["rank"] = pd.to_numeric(rank_view["rank"], errors="coerce").fillna(9999).astype(int)

    exact = case_meta[["case_id", "primary_label"]].merge(
        rank_view,
        left_on=["case_id", "primary_label"],
        right_on=["case_id", "candidate_id"],
        how="left",
    )
    exact_rank = exact["rank"].fillna(9999).astype(int).to_numpy()

    label_rows = case_meta[["case_id", "label_set"]].explode("label_set").rename(columns={"label_set": "candidate_id"})
    label_rows["candidate_id"] = label_rows["candidate_id"].astype(str)
    any_hits = label_rows.merge(rank_view, on=["case_id", "candidate_id"], how="left")
    any_rank_by_case = any_hits.groupby("case_id", sort=False)["rank"].min()
    any_rank = case_meta["case_id"].map(any_rank_by_case).fillna(9999).astype(int).to_numpy()

    label_count = case_meta["label_set"].apply(len).astype(int).to_numpy()
    rows = pd.DataFrame(
        {
            "case_id": case_meta["case_id"].to_numpy(),
            "method": method,
            "exact_rank": exact_rank,
            "any_label_rank": any_rank,
            "label_count": label_count,
        }
    )
    metrics = {"method": method, **metric_from_ranks(exact_rank)}
    arr = np.asarray(any_rank)
    metrics.update(
        {
            "any_label_at_1": float(np.mean(arr <= 1)),
            "any_label_at_3": float(np.mean(arr <= 3)),
            "any_label_at_5": float(np.mean(arr <= 5)),
            "any_label_at_50": float(np.mean(arr <= 50)),
        }
    )
    return metrics, rows


def prepare_validation_rank_input(hgnn: pd.DataFrame, similar: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["case_id", "gold_id", "candidate_id"]
    hgnn_part = (
        hgnn[key_cols + ["hgnn_component"]]
        .copy()
        .groupby(key_cols, sort=False, as_index=False)["hgnn_component"]
        .max()
    )
    if similar.empty:
        hgnn_part["similar_component"] = 0.0
        return hgnn_part

    similar_part = (
        similar[key_cols + ["similar_component"]]
        .copy()
        .groupby(key_cols, sort=False, as_index=False)["similar_component"]
        .max()
    )
    prepared = hgnn_part.merge(similar_part, on=key_cols, how="outer")
    prepared["hgnn_component"] = pd.to_numeric(prepared["hgnn_component"], errors="coerce").fillna(0.0)
    prepared["similar_component"] = pd.to_numeric(prepared["similar_component"], errors="coerce").fillna(0.0)
    return prepared


def rank_prepared_validation(prepared: pd.DataFrame, similar_weight: float) -> pd.DataFrame:
    ranked = prepared[["case_id", "candidate_id", "hgnn_component", "similar_component"]].copy()
    ranked["score"] = ranked["hgnn_component"] + float(similar_weight) * ranked["similar_component"]
    ranked = ranked.sort_values(
        ["case_id", "score", "hgnn_component"],
        ascending=[True, False, False],
        kind="stable",
    )
    ranked["rank"] = ranked.groupby("case_id").cumcount() + 1
    return ranked[["case_id", "candidate_id", "rank"]]


def rank_scored_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    ranked = scored.sort_values(
        ["case_id", "score", "hgnn_component"],
        ascending=[True, False, False],
        kind="stable",
    ).copy()
    ranked["rank"] = ranked.groupby("case_id").cumcount() + 1
    return ranked


def add_overlap_features(df: pd.DataFrame, case_table: pd.DataFrame, resources: dict[str, Any]) -> pd.DataFrame:
    case_hpos = {row.case_id: set(row.hpo_ids) for row in case_table.itertuples(index=False)}
    disease_hpos = resources["disease_hpos"]
    hpo_ic = resources["hpo_ic"]
    rows = []
    for row in df.itertuples(index=False):
        rows.append(overlap_metrics(case_hpos.get(str(row.case_id), set()), disease_hpos.get(str(row.candidate_id), set()), hpo_ic))
    metrics = pd.DataFrame(rows)
    out = df.reset_index(drop=True).copy()
    for col in ["exact_overlap", "ic_weighted_overlap", "shared_hpo_count"]:
        fallback = metrics[col] if col in metrics else 0.0
        if col not in out.columns:
            out[col] = fallback
        else:
            current = pd.to_numeric(out[col], errors="coerce")
            out[col] = current.mask(current.fillna(0.0).le(0.0), fallback).fillna(0.0)
    return out


def prepare_gated_input(
    hgnn: pd.DataFrame,
    similar: pd.DataFrame,
    case_table: pd.DataFrame,
    resources: dict[str, Any],
) -> pd.DataFrame:
    rows = hgnn.copy()
    if not similar.empty:
        rows = pd.concat(
            [
                rows,
                similar.assign(hgnn_component=0.0, hgnn_score=0.0, ic_weighted_overlap=0.0, exact_overlap=0.0),
            ],
            ignore_index=True,
            sort=False,
        )
    for col in ["hgnn_component", "similar_component", "similar_case_source_score", "ic_weighted_overlap", "exact_overlap"]:
        if col not in rows.columns:
            rows[col] = 0.0
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
    for col in ["matched_case_ids", "matched_labels"]:
        if col not in rows.columns:
            rows[col] = ""
    if "similar_case_source_score" not in rows.columns:
        rows["similar_case_source_score"] = 0.0
    agg = rows.groupby(["case_id", "gold_id", "candidate_id"], sort=False).agg(
        hgnn_component=("hgnn_component", "max"),
        similar_component=("similar_component", "max"),
        similar_case_source_score=("similar_case_source_score", "max"),
        ic_weighted_overlap=("ic_weighted_overlap", "max"),
        exact_overlap=("exact_overlap", "max"),
        matched_case_ids=("matched_case_ids", lambda values: "|".join(sorted({v for v in values.dropna().astype(str) if v}))),
        matched_labels=("matched_labels", lambda values: "|".join(sorted({v for v in values.dropna().astype(str) if v}))),
    ).reset_index()
    return add_overlap_features(agg, case_table, resources)


def score_gated(
    prepared: pd.DataFrame,
    *,
    sim_weight: float,
    ic_weight: float,
    agree_boost: float,
    protect_bonus: float,
) -> pd.DataFrame:
    scored = prepared.copy()
    agreement = (scored["hgnn_component"] > 0) & (scored["similar_component"] > 0)
    low_overlap_sim_only = (scored["hgnn_component"] == 0) & (scored["ic_weighted_overlap"] <= 0.02)
    sim_eff = scored["similar_component"] * float(sim_weight)
    sim_eff = np.where(low_overlap_sim_only, sim_eff * 0.5, sim_eff)
    scored["score"] = (
        scored["hgnn_component"]
        + sim_eff
        + float(ic_weight) * scored["ic_weighted_overlap"]
        + np.where(agreement, float(agree_boost), 0.0)
    )
    top1_idx = scored.sort_values(["case_id", "hgnn_component"], ascending=[True, False]).groupby("case_id").head(1).index
    scored.loc[top1_idx, "score"] = scored.loc[top1_idx, "score"] + float(protect_bonus)
    return rank_scored_candidates(scored)


def top5_candidates_cell(ranked: pd.DataFrame, case_id: str) -> str:
    group = ranked[ranked["case_id"] == case_id].sort_values("rank").head(5)
    return "|".join(group["candidate_id"].astype(str).tolist())


def select_on_validation(
    val_hgnn: pd.DataFrame,
    val_sim: pd.DataFrame,
    val_table: pd.DataFrame,
    topks: list[int],
    weights: list[float],
    score_types: list[str],
    *,
    enable_gated_rerank: bool = True,
    gated_sim_weights: list[float] | None = None,
    gated_ic_weights: list[float] | None = None,
    gated_agree_boosts: list[float] | None = None,
    gated_protect_bonuses: list[float] | None = None,
    overlap_resources: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    best: dict[str, Any] | None = None
    print("[selection] evaluating HGNN-only validation baseline", flush=True)
    baseline_prepared = prepare_validation_rank_input(val_hgnn, pd.DataFrame())
    baseline_ranked = rank_prepared_validation(baseline_prepared, similar_weight=0.0)
    baseline_metrics, _ = evaluate(baseline_ranked, val_table, "HGNN only")
    rows.append(
        {
            "source_combination": "HGNN only",
            "status": "evaluated",
            "similar_case_topk": 0,
            "similar_case_weight": 0.0,
            "similar_case_score_type": "",
            **baseline_metrics,
        }
    )
    total = len(topks) * len(weights) * len(score_types)
    done = 0
    prepared_cache: dict[tuple[int, str], pd.DataFrame] = {}
    for topk in topks:
        for score_type in score_types:
            sim_source = similar_source(val_sim, topk, score_type, val_table)
            prepared_cache[(topk, score_type)] = prepare_validation_rank_input(val_hgnn, sim_source)
            print(
                f"[selection] cached rank input topk={topk} score_type={score_type} "
                f"similar_rows={len(sim_source)} rank_rows={len(prepared_cache[(topk, score_type)])}",
                flush=True,
            )
            for weight in weights:
                done += 1
                print(
                    f"[selection] evaluating {done}/{total}: topk={topk} weight={weight} score_type={score_type}",
                    flush=True,
                )
                ranked = rank_prepared_validation(prepared_cache[(topk, score_type)], similar_weight=weight)
                metrics, _ = evaluate(ranked, val_table, "HGNN + similar_case")
                row = {
                    "source_combination": "HGNN + similar_case",
                    "rerank_mode": "simple",
                    "status": "evaluated",
                    "similar_case_topk": topk,
                    "similar_case_weight": weight,
                    "similar_case_score_type": score_type,
                    "gated_sim_weight": np.nan,
                    "gated_ic_weight": np.nan,
                    "gated_agree_boost": np.nan,
                    "gated_protect_bonus": np.nan,
                    **metrics,
                }
                rows.append(row)
                key = (float(metrics["top5"]), float(metrics["rank_le_50"]), float(metrics["top1"]), -float(metrics["mean_rank"]))
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "row": row,
                        "simple_topk": topk,
                        "simple_score_type": score_type,
                    }
    if (
        enable_gated_rerank
        and best is not None
        and overlap_resources is not None
        and int(best["row"]["similar_case_topk"]) > 0
    ):
        base_topk = int(best["simple_topk"])
        base_score_type = str(best["simple_score_type"])
        base_sim_source = similar_source(val_sim, base_topk, base_score_type, val_table)
        gated_input = prepare_gated_input(val_hgnn, base_sim_source, val_table, overlap_resources)
        sim_grid = gated_sim_weights or [0.2, 0.3, 0.4, 0.5]
        ic_grid = gated_ic_weights or [0.0, 0.05, 0.1]
        agree_grid = gated_agree_boosts or [0.0, 0.05, 0.1]
        protect_grid = gated_protect_bonuses or [0.0, 0.05, 0.1]
        total_gated = len(sim_grid) * len(ic_grid) * len(agree_grid) * len(protect_grid)
        done_gated = 0
        for sim_weight in sim_grid:
            for ic_weight in ic_grid:
                for agree_boost in agree_grid:
                    for protect_bonus in protect_grid:
                        done_gated += 1
                        print(
                            f"[selection] evaluating gated {done_gated}/{total_gated}: "
                            f"base_topk={base_topk} score_type={base_score_type} "
                            f"sim_weight={sim_weight} ic_weight={ic_weight} "
                            f"agree_boost={agree_boost} protect_bonus={protect_bonus}",
                            flush=True,
                        )
                        ranked = score_gated(
                            gated_input,
                            sim_weight=float(sim_weight),
                            ic_weight=float(ic_weight),
                            agree_boost=float(agree_boost),
                            protect_bonus=float(protect_bonus),
                        )
                        metrics, _ = evaluate(ranked, val_table, "HGNN + gated_similar_case")
                        row = {
                            "source_combination": "HGNN + gated_similar_case",
                            "rerank_mode": "gated",
                            "status": "evaluated",
                            "similar_case_topk": base_topk,
                            "similar_case_weight": np.nan,
                            "similar_case_score_type": base_score_type,
                            "gated_sim_weight": float(sim_weight),
                            "gated_ic_weight": float(ic_weight),
                            "gated_agree_boost": float(agree_boost),
                            "gated_protect_bonus": float(protect_bonus),
                            **metrics,
                        }
                        rows.append(row)
                        key = (
                            float(metrics["top5"]),
                            float(metrics["rank_le_50"]),
                            float(metrics["top1"]),
                            -float(metrics["mean_rank"]),
                        )
                        if best is None or key > best["key"]:
                            best = {
                                "key": key,
                                "row": row,
                                "simple_topk": base_topk,
                                "simple_score_type": base_score_type,
                            }
    if best is None:
        raise RuntimeError("No SimilarCase validation config was evaluated.")
    return best["row"], pd.DataFrame(rows)


def frozen_selection(path: Path) -> dict[str, Any]:
    config = json.load(open(path, encoding="utf-8"))
    return {
        "source_combination": "HGNN + similar_case",
        "rerank_mode": "simple",
        "status": "loaded_frozen_config",
        "similar_case_topk": int(config["similar_case_topk"]),
        "similar_case_weight": float(config["similar_case_weight"]),
        "similar_case_score_type": str(config["score_type"]),
        "gated_sim_weight": np.nan,
        "gated_ic_weight": np.nan,
        "gated_agree_boost": np.nan,
        "gated_protect_bonus": np.nan,
    }


def write_case_analysis(
    output_dir: Path,
    baseline_case_ranks: pd.DataFrame,
    test_case_ranks: pd.DataFrame,
    test_ranked: pd.DataFrame,
    test_table: pd.DataFrame,
    disease_hpos: dict[str, set[str]],
) -> None:
    label_sets = {row.case_id: set(row.label_set) for row in test_table.itertuples(index=False)}
    labels_cell = {row.case_id: "|".join(row.label_set) for row in test_table.itertuples(index=False)}
    primary = {row.case_id: str(row.primary_label) for row in test_table.itertuples(index=False)}
    hpo_count = {row.case_id: int(row.hpo_count) for row in test_table.itertuples(index=False)}
    hpos_by_case = {row.case_id: set(row.hpo_ids) for row in test_table.itertuples(index=False)}

    merged = baseline_case_ranks[["case_id", "exact_rank"]].rename(columns={"exact_rank": "baseline_rank"}).merge(
        test_case_ranks[["case_id", "exact_rank"]].rename(columns={"exact_rank": "augmented_rank"}),
        on="case_id",
        how="left",
    )
    recovered = merged[(merged["baseline_rank"] > 50) & (merged["augmented_rank"] <= 50)].copy()
    recovered_rows = []
    for row in recovered.itertuples(index=False):
        case_id = row.case_id
        gold = primary[case_id]
        gold_row = test_ranked[(test_ranked["case_id"] == case_id) & (test_ranked["candidate_id"] == gold)].sort_values("rank").head(1)
        overlap = len(hpos_by_case[case_id] & set(disease_hpos.get(gold, set())))
        recovered_rows.append(
            {
                "case_id": case_id,
                "gold_labels": labels_cell[case_id],
                "baseline_rank": int(row.baseline_rank),
                "augmented_rank": int(row.augmented_rank),
                "baseline_top5_hit": int(row.baseline_rank <= 5),
                "augmented_top5_hit": int(row.augmented_rank <= 5),
                "source_that_recovered_gold": "similar_case" if not gold_row.empty and float(gold_row.iloc[0].get("similar_component", 0.0)) > 0 else "hgnn_or_score_tie",
                "similar_case_source_score": float(gold_row.iloc[0].get("similar_case_source_score", 0.0)) if not gold_row.empty else 0.0,
                "matched_similar_case_ids": str(gold_row.iloc[0].get("matched_case_ids", "")) if not gold_row.empty else "",
                "matched_similar_case_labels": str(gold_row.iloc[0].get("matched_labels", "")) if not gold_row.empty else "",
                "case_hpo_count": hpo_count[case_id],
                "gold_disease_hpo_overlap": overlap,
                "label_subset": "multi_label" if len(label_sets[case_id]) > 1 else "single_label",
            }
        )
    recovered_df = pd.DataFrame(recovered_rows)
    write_csv(recovered_df, output_dir / "recovered_rank_gt50_cases.csv")

    near = merged[(merged["baseline_rank"] > 5) & (merged["augmented_rank"].between(6, 20))].copy()
    near_rows = []
    for row in near.itertuples(index=False):
        case_id = row.case_id
        gold = primary[case_id]
        group = test_ranked[test_ranked["case_id"] == case_id].sort_values("rank")
        gold_group = group[group["candidate_id"] == gold].sort_values("rank")
        if gold_group.empty:
            continue
        gold_row = gold_group.iloc[0]
        top5_min = float(group.head(5)["score"].min()) if len(group) >= 5 else float(group["score"].min())
        near_rows.append(
            {
                "case_id": case_id,
                "gold_labels": labels_cell[case_id],
                "augmented_rank": int(gold_row["rank"]),
                "top5_candidates": top5_candidates_cell(test_ranked, case_id),
                "gold_candidate_score": float(gold_row["score"]),
                "top5_score_margin": float(top5_min - float(gold_row["score"])),
                "possible_fix": "increase_similar_case_weight" if float(gold_row.get("similar_component", 0.0)) > 0 else "not_similar_case_weight_only",
                "recommended_weight_change": "validation-only recheck only" if float(gold_row.get("similar_component", 0.0)) > 0 else "needs new source or reranker",
                "similar_case_source_score": float(gold_row.get("similar_case_source_score", 0.0)),
            }
        )
    near_df = pd.DataFrame(near_rows).sort_values(["top5_score_margin", "augmented_rank"], ascending=[True, True]) if near_rows else pd.DataFrame()
    write_csv(near_df, output_dir / "near_miss_top5_cases.csv")

    write_text(
        output_dir / "recovered_rank_gt50_cases.md",
        "\n".join(
            [
                "# recovered rank>50 cases",
                "",
                f"- recovered to top50: {len(recovered_df)}",
                f"- recovered to top5: {int(recovered_df['augmented_top5_hit'].sum()) if not recovered_df.empty else 0}",
                f"- recovered to top50 but not top5: {int((recovered_df['augmented_top5_hit'] == 0).sum()) if not recovered_df.empty else 0}",
            ]
        ),
    )
    write_text(
        output_dir / "near_miss_top5_cases.md",
        "\n".join(
            [
                "# near-miss top5 cases",
                "",
                f"- exact failed cases with baseline rank>5 and augmented rank 6-20: {len(near_df)}",
                f"- near-miss cases with similar_case evidence: {int((near_df.get('similar_case_source_score', pd.Series(dtype=float)).astype(float) > 0).sum()) if not near_df.empty else 0}",
                "",
                df_to_markdown(near_df.head(30), max_rows=30),
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_config = load_yaml_config(args.data_config_path)
    train_config = load_yaml_config(args.train_config_path)
    train_table, val_table, test_table = load_case_tables(data_config, args.data_config_path, train_config)

    val_candidates = load_candidates(args.validation_candidates_path)
    test_candidates = load_candidates(args.test_candidates_path)
    test_candidates = test_candidates[test_candidates["dataset_name"].astype(str).str.startswith(MIMIC_PREFIX)].copy()
    val_hgnn = hgnn_source(val_candidates, val_table)
    test_hgnn = hgnn_source(test_candidates, test_table)
    overlap_resources = build_overlap_metric_resources(train_config)

    if args.use_frozen_config:
        selected = frozen_selection(args.frozen_config_path)
        val_ablation = pd.DataFrame([selected])
    else:
        topks = parse_int_list(args.similar_case_topk)
        weights = parse_float_list(args.similar_case_weight)
        score_types = parse_score_types(args.score_type)
        val_sim = compute_similar_matches(
            val_table,
            train_table,
            max(topks),
            device_name=args.similarity_device,
            batch_size=args.similarity_batch_size,
            label="validation",
        )
        selected, val_ablation = select_on_validation(
            val_hgnn,
            val_sim,
            val_table,
            topks,
            weights,
            score_types,
            enable_gated_rerank=not args.disable_gated_rerank,
            gated_sim_weights=parse_float_list(args.gated_sim_weight),
            gated_ic_weights=parse_float_list(args.gated_ic_weight),
            gated_agree_boosts=parse_float_list(args.gated_agree_boost),
            gated_protect_bonuses=parse_float_list(args.gated_protect_bonus),
            overlap_resources=overlap_resources,
        )
    write_csv(val_ablation, args.output_dir / "similar_case_val_selection.csv")

    selected_topk = int(selected["similar_case_topk"])
    selected_mode = str(selected.get("rerank_mode", "simple"))
    selected_weight = float(selected["similar_case_weight"]) if selected_mode == "simple" else 0.0
    selected_score_type = str(selected["similar_case_score_type"])
    test_sim = compute_similar_matches(
        test_table,
        train_table,
        selected_topk,
        device_name=args.similarity_device,
        batch_size=args.similarity_batch_size,
        label="test",
    )
    test_sim_source = similar_source(test_sim, selected_topk, selected_score_type, test_table)
    if selected_mode == "gated":
        test_ranked = score_gated(
            prepare_gated_input(test_hgnn, test_sim_source, test_table, overlap_resources),
            sim_weight=float(selected["gated_sim_weight"]),
            ic_weight=float(selected["gated_ic_weight"]),
            agree_boost=float(selected["gated_agree_boost"]),
            protect_bonus=float(selected["gated_protect_bonus"]),
        )
    else:
        test_ranked = combine(test_hgnn, test_sim_source, similar_weight=selected_weight)
    baseline_ranked = combine(test_hgnn, pd.DataFrame(), similar_weight=0.0, use_similar_case=False)
    write_csv(test_ranked, args.output_dir / "similar_case_fixed_test_ranked_candidates.csv")

    test_metrics, test_case_ranks = evaluate(test_ranked, test_table, "validation_selected_fixed_weights")
    baseline_metrics, baseline_case_ranks = evaluate(baseline_ranked, test_table, "hgnn_top50_baseline")
    fixed_out = pd.DataFrame(
        [
            {
                **test_metrics,
                "selected_source_combination": str(selected.get("source_combination", "HGNN + similar_case")),
                "selected_rerank_mode": selected_mode,
                "selected_similar_case_topk": selected_topk,
                "selected_similar_case_weight": selected_weight,
                "selected_similar_case_score_type": selected_score_type,
                "selected_gated_sim_weight": selected.get("gated_sim_weight", np.nan),
                "selected_gated_ic_weight": selected.get("gated_ic_weight", np.nan),
                "selected_gated_agree_boost": selected.get("gated_agree_boost", np.nan),
                "selected_gated_protect_bonus": selected.get("gated_protect_bonus", np.nan),
            }
        ]
    )
    write_csv(fixed_out, args.output_dir / "similar_case_fixed_test.csv")
    write_text(
        args.output_dir / "similar_case_fixed_test.md",
        "\n".join(
            [
                "# similar_case fixed test",
                "",
                "- 协议: validation-selected fixed weights，用 test 只评估一次。",
                f"- selected source_combination: {selected.get('source_combination', 'HGNN + similar_case')}",
                f"- selected rerank_mode: {selected_mode}",
                f"- selected topk/weight/score_type: {selected_topk} / {selected_weight} / {selected_score_type}",
                "",
                df_to_markdown(fixed_out),
            ]
        ),
    )

    disease_hpos = overlap_resources["disease_hpos"]
    write_case_analysis(args.output_dir, baseline_case_ranks, test_case_ranks, test_ranked, test_table, disease_hpos)
    print(
        json.dumps(
            {
                "baseline": baseline_metrics,
                "fixed_test": test_metrics,
                "selected": {
                    "similar_case_topk": selected_topk,
                    "similar_case_weight": selected_weight,
                    "score_type": selected_score_type,
                    "rerank_mode": selected_mode,
                    "gated_sim_weight": selected.get("gated_sim_weight", None),
                    "gated_ic_weight": selected.get("gated_ic_weight", None),
                    "gated_agree_boost": selected.get("gated_agree_boost", None),
                    "gated_protect_bonus": selected.get("gated_protect_bonus", None),
                    "selected_by": "frozen_config" if args.use_frozen_config else "validation",
                },
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
