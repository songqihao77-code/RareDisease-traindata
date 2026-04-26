from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_case_files
from src.evaluation.evaluator import load_test_cases, load_yaml_config
from src.training.trainer import resolve_train_files, split_train_val_by_case

DEFAULT_CANDIDATES = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_finetune_attn_idf_main.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "augmentation"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "deeprare_parity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run protocol-separated HGNN_AUG candidate augmentation.")
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--data-config-path", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--train-config-path", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--source-weights-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--similar-top-n", type=int, default=10)
    parser.add_argument("--per-case-limit", type=int, default=100)
    return parser.parse_args()


def hpo_doc(hpo_ids: list[str]) -> str:
    return " ".join(sorted({str(hpo) for hpo in hpo_ids if str(hpo).strip()}))


def load_weights(path: Path | None) -> tuple[dict[str, float], str]:
    defaults = {"hgnn": 1.0, "static_hpo_soft": 0.20, "similar_case": 0.35, "mapping": 0.10}
    if path is None:
        return defaults, "exploratory_defaults_not_main_table"
    if not path.is_file():
        raise FileNotFoundError(f"Source weights not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = {**defaults, **{str(k): float(v) for k, v in payload.get("source_weights", payload).items()}}
    return weights, "validation_selected_fixed_eval"


def load_train_library(train_config: dict[str, Any], data_config: dict[str, Any]) -> pd.DataFrame:
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))
    train_files = resolve_train_files(train_config["paths"])
    df = load_case_files(
        file_paths=[str(path) for path in train_files],
        case_id_col=case_id_col,
        label_col=label_col,
        disease_index_path=train_config["paths"]["disease_index_path"],
        split_namespace="train",
    )
    train_df, _ = split_train_val_by_case(
        df,
        val_ratio=float(train_config["data"]["val_ratio"]),
        random_seed=int(train_config["data"]["random_seed"]),
        case_id_col=case_id_col,
    )
    records = []
    for case_id, group in train_df.groupby(case_id_col, sort=False):
        records.append(
            {
                "case_id": str(case_id),
                "gold_id": str(group[label_col].iloc[0]),
                "hpo_doc": hpo_doc([str(hpo) for hpo in group[hpo_col].dropna().tolist()]),
            }
        )
    return pd.DataFrame(records)


def similar_case_candidates(test_bundle: dict[str, Any], library: pd.DataFrame, top_n: int) -> pd.DataFrame:
    test_records = [
        {"case_id": str(row.case_id), "hpo_doc": hpo_doc(list(row.hpo_ids))}
        for row in test_bundle["case_table"].itertuples(index=False)
    ]
    test_df = pd.DataFrame(test_records)
    if test_df.empty or library.empty:
        return pd.DataFrame(columns=["case_id", "candidate_id", "source", "source_score"])
    vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+")
    matrix = vectorizer.fit_transform(pd.concat([library["hpo_doc"], test_df["hpo_doc"]], ignore_index=True))
    lib_x = matrix[: len(library)]
    test_x = matrix[len(library) :]
    sim = cosine_similarity(test_x, lib_x)
    rows = []
    for row_idx, case_id in enumerate(test_df["case_id"].tolist()):
        top_idx = np.argsort(-sim[row_idx])[:top_n]
        by_disease: dict[str, float] = {}
        for idx in top_idx:
            disease = str(library.iloc[int(idx)]["gold_id"])
            by_disease[disease] = max(by_disease.get(disease, 0.0), float(sim[row_idx, int(idx)]))
        for disease, score in by_disease.items():
            rows.append({"case_id": case_id, "candidate_id": disease, "source": "similar_case", "source_score": score})
    return pd.DataFrame(rows)


def normalize_hgnn_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hgnn_norm"] = out.groupby("case_id")["hgnn_score"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 1.0
    )
    out["static_hpo_soft_score"] = out.groupby("case_id")["ic_weighted_overlap"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0.0
    )
    return out


def build_augmented(candidates: pd.DataFrame, similar: pd.DataFrame, weights: dict[str, float], limit: int) -> pd.DataFrame:
    base = normalize_hgnn_candidates(candidates)
    rows = []
    for row in base.itertuples(index=False):
        score = float(weights["hgnn"]) * float(row.hgnn_norm) + float(weights["static_hpo_soft"]) * float(row.static_hpo_soft_score)
        rows.append(
            {
                "case_id": row.case_id,
                "dataset_name": row.dataset_name,
                "gold_id": row.gold_id,
                "candidate_id": row.candidate_id,
                "hgnn_original_rank": int(row.original_rank),
                "sources": "hgnn_top50;static_hpo_soft",
                "aug_score": score,
            }
        )
    meta = base.groupby("case_id", sort=False)[["dataset_name", "gold_id"]].first().to_dict(orient="index")
    for row in similar.itertuples(index=False):
        if row.case_id not in meta:
            continue
        rows.append(
            {
                "case_id": row.case_id,
                "dataset_name": meta[row.case_id]["dataset_name"],
                "gold_id": meta[row.case_id]["gold_id"],
                "candidate_id": row.candidate_id,
                "hgnn_original_rank": "",
                "sources": row.source,
                "aug_score": float(weights["similar_case"]) * float(row.source_score),
            }
        )
    aug = pd.DataFrame(rows)
    if aug.empty:
        return aug
    agg = (
        aug.groupby(["case_id", "dataset_name", "gold_id", "candidate_id"], sort=False)
        .agg({"aug_score": "sum", "sources": lambda s: ";".join(sorted(set(";".join(s).split(";")))), "hgnn_original_rank": "first"})
        .reset_index()
    )
    agg = agg.sort_values(["case_id", "aug_score"], ascending=[True, False], kind="stable")
    agg["aug_rank"] = agg.groupby("case_id").cumcount() + 1
    return agg.loc[agg["aug_rank"] <= int(limit)].copy()


def evaluate_exact(aug: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for dataset, dataset_df in aug.groupby("dataset_name", sort=True):
        ranks = []
        for _, group in dataset_df.groupby("case_id", sort=False):
            hits = group.loc[group["candidate_id"] == group["gold_id"], "aug_rank"].tolist()
            ranks.append(int(min(hits)) if hits else int(group["aug_rank"].max()) + 1)
        arr = np.asarray(ranks, dtype=int)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "num_cases": int(len(arr)),
                "top1": float((arr <= 1).mean()),
                "top3": float((arr <= 3).mean()),
                "top5": float((arr <= 5).mean()),
                "rank_le_50": float((arr <= 50).mean()),
                "median_rank": float(np.median(arr)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    weights, protocol_status = load_weights(args.source_weights_path)
    candidates = pd.read_csv(args.candidates_path, dtype={"case_id": str, "dataset_name": str, "gold_id": str, "candidate_id": str})
    data_config = load_yaml_config(args.data_config_path)
    train_config = load_yaml_config(args.train_config_path)
    test_bundle = load_test_cases(data_config, args.data_config_path)
    library = load_train_library(train_config, data_config)
    similar = similar_case_candidates(test_bundle, library, args.similar_top_n)
    aug = build_augmented(candidates, similar, weights, args.per_case_limit)
    aug_path = args.output_dir / "hgnn_aug_candidates_test.csv"
    aug.to_csv(aug_path, index=False, encoding="utf-8-sig")

    exact = evaluate_exact(aug, f"HGNN_AUG:{protocol_status}")
    exact.to_csv(args.report_dir / "hgnn_aug_exact_test.csv", index=False, encoding="utf-8-sig")
    relaxed = exact.copy()
    relaxed["method"] = relaxed["method"] + ":relaxed_not_implemented_exact_only"
    relaxed["note"] = "Relaxed/synonym/parent-child evaluation is supplementary only and requires mapping validation."
    relaxed.to_csv(args.report_dir / "hgnn_aug_relaxed_supplementary.csv", index=False, encoding="utf-8-sig")

    ablation_rows = [
        {"source": "hgnn_top50", "status": "available", "main_table_eligible": True},
        {"source": "static_hpo_soft", "status": "available_as_soft_score_only", "main_table_eligible": protocol_status == "validation_selected_fixed_eval"},
        {"source": "similar_case", "status": "available_train_library_tfidf", "main_table_eligible": protocol_status == "validation_selected_fixed_eval"},
        {"source": "PubCaseFinder", "status": "unavailable_api_not_called", "main_table_eligible": False},
        {"source": "PhenoBrain", "status": "unavailable_api_not_called", "main_table_eligible": False},
        {"source": "synonym_omim_orpha_mondo_mapping", "status": "interface_reserved_not_applied", "main_table_eligible": False},
    ]
    pd.DataFrame(ablation_rows).to_csv(args.report_dir / "hgnn_aug_source_ablation.csv", index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "candidates": str(aug_path.resolve()),
                "exact": str((args.report_dir / "hgnn_aug_exact_test.csv").resolve()),
                "protocol_status": protocol_status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
