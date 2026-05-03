from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_hypergraph import build_batch_hypergraph
from src.data.dataset import CaseBatchLoader, load_case_files
from src.evaluation.evaluator import (
    _check_case_id_namespace_overlap,
    _resolve_device,
    load_checkpoint_model,
    load_static_resources,
    load_test_cases,
    load_yaml_config,
)
from src.rerank.hpo_semantic import HpoSemanticMatcher
from src.training.trainer import resolve_train_files, split_train_val_by_case


DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_llldataset_eval.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_finetune_attn_idf_main.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "rerank" / "top50_candidates_v2.csv"
DEFAULT_CORE_HPO_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HGNN top50 candidates with no-train evidence features."
    )
    parser.add_argument("--data-config-path", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--train-config-path", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hpo-ontology-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--case-source",
        choices=["test", "train", "validation"],
        default="test",
        help="Export candidates for the configured test set, train split, or train-config validation split.",
    )
    return parser.parse_args()


def _load_train_or_validation_cases(
    *,
    train_config: dict[str, Any],
    train_config_path: Path,
    data_config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    case_id_col = str(data_config.get("case_id_col", "case_id"))
    label_col = str(data_config.get("label_col", "mondo_label"))
    hpo_col = str(data_config.get("hpo_col", "hpo_id"))
    batch_size = int(data_config.get("batch_size", train_config.get("data", {}).get("batch_size", 8)))

    train_files = resolve_train_files(train_config["paths"])
    all_df = load_case_files(
        file_paths=[str(path) for path in train_files],
        case_id_col=case_id_col,
        label_col=label_col,
        disease_index_path=train_config["paths"]["disease_index_path"],
        split_namespace="train",
    )
    train_df, val_df = split_train_val_by_case(
        all_df,
        val_ratio=float(train_config["data"]["val_ratio"]),
        random_seed=int(train_config["data"]["random_seed"]),
        case_id_col=case_id_col,
    )
    if split == "train":
        selected_df = train_df
    elif split == "validation":
        selected_df = val_df
    else:
        raise ValueError(f"Unsupported train split: {split!r}")
    if selected_df.empty:
        raise ValueError(f"{split} split is empty; cannot export candidates.")

    raw_df = selected_df[[case_id_col, label_col, hpo_col, "_source_file"]].copy()
    case_records: list[dict[str, Any]] = []
    for case_id, group_df in raw_df.groupby(case_id_col, sort=False):
        case_records.append(
            {
                "case_id": str(case_id),
                "mondo_label": str(group_df[label_col].iloc[0]),
                "hpo_ids": [str(hpo) for hpo in group_df[hpo_col].dropna().unique().tolist()],
                "source_file": Path(str(group_df["_source_file"].iloc[0])).name,
            }
        )
    return {
        "raw_df": raw_df,
        "case_table": pd.DataFrame(case_records),
        "batch_size": batch_size,
        "case_id_col": case_id_col,
        "label_col": label_col,
        "hpo_col": hpo_col,
        "test_files": [str(path) for path in train_files],
        "train_config_path": str(train_config_path.resolve()),
    }


def _nonzero_hpo_indices(matrix: Any, disease_idx: int) -> set[int]:
    col = matrix[:, int(disease_idx)]
    return {int(idx) for idx in col.nonzero()[0].tolist()}


def _weighted_hpo_indices(matrix: Any, disease_idx: int) -> list[tuple[int, float]]:
    col = matrix[:, int(disease_idx)].tocoo()
    pairs = [(int(row), float(value)) for row, value in zip(col.row, col.data, strict=True) if float(value) > 0.0]
    return sorted(pairs, key=lambda item: (-item[1], item[0]))


def _case_hpo_indices(case_hpo_ids: list[str], hpo_to_idx: dict[str, int]) -> set[int]:
    return {int(hpo_to_idx[hpo_id]) for hpo_id in case_hpo_ids if hpo_id in hpo_to_idx}


def _hpo_ids_from_indices(indices: set[int], idx_to_hpo: dict[int, str]) -> set[str]:
    return {idx_to_hpo[idx] for idx in indices if idx in idx_to_hpo}


def _evidence_features(
    *,
    case_hpo_idx: set[int],
    disease_hpo_idx: set[int],
    hpo_specificity: Any,
    semantic_matcher: HpoSemanticMatcher,
    idx_to_hpo: dict[int, str],
    hpo_specificity_by_id: dict[str, float],
) -> dict[str, float | int]:
    shared = case_hpo_idx & disease_hpo_idx
    case_count = len(case_hpo_idx)
    disease_count = len(disease_hpo_idx)
    union_count = len(case_hpo_idx | disease_hpo_idx)

    if case_count and disease_count:
        exact_overlap = len(shared) / math.sqrt(case_count * disease_count)
    else:
        exact_overlap = 0.0

    case_ic_total = float(sum(float(hpo_specificity[idx]) for idx in case_hpo_idx))
    shared_ic_total = float(sum(float(hpo_specificity[idx]) for idx in shared))

    semantic = semantic_matcher.score(
        case_hpos=_hpo_ids_from_indices(case_hpo_idx, idx_to_hpo),
        disease_hpos=_hpo_ids_from_indices(disease_hpo_idx, idx_to_hpo),
        hpo_specificity=hpo_specificity_by_id,
    )

    return {
        "exact_overlap": float(exact_overlap),
        "ic_weighted_overlap": float(shared_ic_total / case_ic_total) if case_ic_total > 0 else 0.0,
        "case_coverage": float(len(shared) / case_count) if case_count else 0.0,
        "disease_coverage": float(len(shared) / disease_count) if disease_count else 0.0,
        "disease_hpo_count": int(disease_count),
        "shared_hpo_count": int(len(shared)),
        "jaccard_overlap": float(len(shared) / union_count) if union_count else 0.0,
        **semantic,
    }


def _core_evidence_features(
    *,
    case_hpo_idx: set[int],
    disease_weighted_hpos: list[tuple[int, float]],
    semantic_matcher: HpoSemanticMatcher,
    idx_to_hpo: dict[int, str],
    core_top_k: int = DEFAULT_CORE_HPO_TOP_K,
) -> dict[str, float | int]:
    core = disease_weighted_hpos[: max(0, int(core_top_k))]
    if not core:
        return {
            "case_hpo_count": int(len(case_hpo_idx)),
            "disease_core_hpo_count_top5": 0,
            "core_exact_coverage_top5": 0.0,
            "core_semantic_coverage_top5": 0.0,
            "core_missing_exact_top5": 0.0,
            "core_missing_semantic_top5": 0.0,
        }

    denom = float(sum(weight for _, weight in core))
    if denom <= 0.0:
        denom = float(len(core))
        core = [(idx, 1.0) for idx, _ in core]

    exact_covered_weight = float(sum(weight for idx, weight in core if idx in case_hpo_idx))
    semantic_covered_weight = exact_covered_weight
    if semantic_matcher.available and case_hpo_idx:
        case_hpos = _hpo_ids_from_indices(case_hpo_idx, idx_to_hpo)
        for idx, weight in core:
            if idx in case_hpo_idx:
                continue
            core_hpo = idx_to_hpo.get(idx)
            if core_hpo is None:
                continue
            if any(semantic_matcher.related(case_hpo, core_hpo) for case_hpo in case_hpos):
                semantic_covered_weight += float(weight)

    exact_coverage = float(exact_covered_weight / denom)
    semantic_coverage = float(min(semantic_covered_weight / denom, 1.0))
    return {
        "case_hpo_count": int(len(case_hpo_idx)),
        "disease_core_hpo_count_top5": int(len(core)),
        "core_exact_coverage_top5": exact_coverage,
        "core_semantic_coverage_top5": semantic_coverage,
        "core_missing_exact_top5": float(1.0 - exact_coverage),
        "core_missing_semantic_top5": float(1.0 - semantic_coverage),
    }


def export_top50_candidates(
    *,
    data_config_path: Path,
    train_config_path: Path,
    checkpoint_path: Path | None,
    output_path: Path,
    top_k: int,
    hpo_ontology_path: Path | None = None,
    case_source: str = "test",
) -> Path:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")

    data_config = load_yaml_config(data_config_path)
    train_config = load_yaml_config(train_config_path)
    device = _resolve_device(train_config)

    resources = load_static_resources(train_config)
    semantic_matcher, semantic_metadata = HpoSemanticMatcher.from_project(
        PROJECT_ROOT,
        explicit_path=hpo_ontology_path,
    )
    if not semantic_metadata.get("available"):
        print(f"[WARN] {semantic_metadata.get('warning')}")
    else:
        print(f"[INFO] HPO ontology: {semantic_metadata.get('ontology_path')}")

    if case_source == "test":
        test_bundle = load_test_cases(data_config, data_config_path)
        overlap_summary = _check_case_id_namespace_overlap(
            train_config,
            train_config_path,
            case_id_col=test_bundle["case_id_col"],
            test_case_ids=set(test_bundle["case_table"]["case_id"].astype(str).tolist()),
        )
        if overlap_summary["overlap_count"] > 0:
            raise RuntimeError(f"train/test case_id overlap detected: {overlap_summary}")
    elif case_source in {"train", "validation"}:
        test_bundle = _load_train_or_validation_cases(
            train_config=train_config,
            train_config_path=train_config_path,
            data_config=data_config,
            split=case_source,
        )
    else:
        raise ValueError(f"Unsupported case_source: {case_source!r}")

    hpo_to_idx = resources["hpo_to_idx"]
    disease_to_idx = resources["disease_to_idx"]
    idx_to_hpo = {int(idx): str(hpo_id) for hpo_id, idx in hpo_to_idx.items()}
    hpo_specificity_by_id = {
        str(hpo_id): float(resources["hpo_specificity"][int(idx)])
        for hpo_id, idx in hpo_to_idx.items()
    }
    case_table = test_bundle["case_table"].copy()
    case_table["valid_hpo_ids"] = case_table["hpo_ids"].apply(
        lambda hpo_ids: [hpo_id for hpo_id in hpo_ids if hpo_id in hpo_to_idx]
    )
    case_table["valid_hpo_count"] = case_table["valid_hpo_ids"].apply(len)
    case_table["skip_reason"] = None
    case_table.loc[~case_table["mondo_label"].isin(disease_to_idx), "skip_reason"] = (
        "label_not_in_disease_index"
    )
    case_table.loc[
        case_table["skip_reason"].isna() & (case_table["valid_hpo_count"] == 0),
        "skip_reason",
    ] = "no_valid_hpo"
    evaluable_cases = case_table[case_table["skip_reason"].isna()].copy()
    if evaluable_cases.empty:
        raise ValueError("No evaluable test cases.")

    case_id_col = test_bundle["case_id_col"]
    label_col = test_bundle["label_col"]
    hpo_col = test_bundle["hpo_col"]
    eval_df = test_bundle["raw_df"].copy()
    eval_df = eval_df[eval_df[case_id_col].isin(set(evaluable_cases["case_id"]))].copy()
    eval_df = eval_df[eval_df[hpo_col].isin(set(hpo_to_idx))].copy()

    loader = CaseBatchLoader(
        df=eval_df,
        batch_size=int(test_bundle["batch_size"]),
        case_id_col=case_id_col,
    )
    model, resolved_checkpoint_path, checkpoint = load_checkpoint_model(
        train_config=train_config,
        checkpoint_path=checkpoint_path,
        num_hpo=resources["num_hpo"],
        device=device,
    )
    disease_side_cache = model.precompute_disease_side(resources["H_disease"])
    cached_node_repr = disease_side_cache["node_repr"]
    cached_disease_repr = disease_side_cache["disease_repr"]

    case_source_map = dict(zip(case_table["case_id"], case_table["source_file"], strict=True))
    case_hpo_idx_map = {
        str(row.case_id): _case_hpo_indices(list(row.valid_hpo_ids), hpo_to_idx)
        for row in evaluable_cases.itertuples(index=False)
    }
    disease_hpo_idx_map = {
        int(disease_idx): _nonzero_hpo_indices(resources["H_disease"], int(disease_idx))
        for disease_idx in range(int(resources["num_disease"]))
    }
    disease_weighted_hpo_map = {
        int(disease_idx): _weighted_hpo_indices(resources["H_disease"], int(disease_idx))
        for disease_idx in range(int(resources["num_disease"]))
    }

    rows: list[dict[str, Any]] = []
    processed_case_ids: set[str] = set()
    real_top_k = min(int(top_k), int(resources["num_disease"]))

    print(f"[INFO] checkpoint: {resolved_checkpoint_path}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] evaluable cases: {len(evaluable_cases)}")

    with torch.inference_mode():
        for batch_idx in range(len(loader)):
            batch_df = loader.get_batch(batch_idx)
            batch_graph = build_batch_hypergraph(
                case_df=batch_df,
                hpo_to_idx=hpo_to_idx,
                disease_to_idx=disease_to_idx,
                H_disease=resources["H_disease"],
                top_50_hpos=resources.get("top_50_hpos", []),
                case_id_col=case_id_col,
                label_col=label_col,
                hpo_col=hpo_col,
                case_noise_control=train_config.get("case_noise_control"),
                hpo_specificity=resources.get("hpo_specificity"),
                include_combined_h=False,
                verbose=False,
            )
            outputs = model(
                batch_graph,
                return_intermediate=False,
                node_repr_override=cached_node_repr,
                disease_repr_override=cached_disease_repr,
            )
            scores = outputs["scores"]
            topk = torch.topk(scores, k=real_top_k, dim=1)

            case_ids = batch_graph["case_ids"]
            case_labels = batch_graph["case_labels"]
            for row_idx, (case_id, gold_id) in enumerate(zip(case_ids, case_labels, strict=True)):
                case_id = str(case_id)
                source_file = str(case_source_map[case_id])
                dataset_name = Path(source_file).stem
                case_hpo_idx = case_hpo_idx_map[case_id]

                for rank_offset, (candidate_idx, score) in enumerate(
                    zip(
                        topk.indices[row_idx].tolist(),
                        topk.values[row_idx].tolist(),
                        strict=True,
                    ),
                    start=1,
                ):
                    candidate_idx = int(candidate_idx)
                    candidate_id = str(resources["disease_labels"][candidate_idx])
                    features = _evidence_features(
                        case_hpo_idx=case_hpo_idx,
                        disease_hpo_idx=disease_hpo_idx_map[candidate_idx],
                        hpo_specificity=resources["hpo_specificity"],
                        semantic_matcher=semantic_matcher,
                        idx_to_hpo=idx_to_hpo,
                        hpo_specificity_by_id=hpo_specificity_by_id,
                    )
                    core_features = _core_evidence_features(
                        case_hpo_idx=case_hpo_idx,
                        disease_weighted_hpos=disease_weighted_hpo_map[candidate_idx],
                        semantic_matcher=semantic_matcher,
                        idx_to_hpo=idx_to_hpo,
                    )
                    rows.append(
                        {
                            "case_id": case_id,
                            "dataset_name": dataset_name,
                            "gold_id": str(gold_id),
                            "candidate_id": candidate_id,
                            "original_rank": int(rank_offset),
                            "hgnn_score": float(score),
                            **features,
                            **core_features,
                        }
                    )
                processed_case_ids.add(case_id)

            if (batch_idx + 1) % 25 == 0 or batch_idx + 1 == len(loader):
                print(f"[INFO] processed batches: {batch_idx + 1}/{len(loader)}")

    expected_case_ids = set(evaluable_cases["case_id"].astype(str))
    missing_case_ids = sorted(expected_case_ids - processed_case_ids)
    if missing_case_ids:
        raise RuntimeError(f"Missing evaluated cases: {missing_case_ids[:10]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    score_by_case = out_df.groupby("case_id")["hgnn_score"]
    top1_scores = score_by_case.transform("max")
    top2_scores = out_df.groupby("case_id")["hgnn_score"].transform(
        lambda values: values.nlargest(2).iloc[-1] if len(values) >= 2 else values.max()
    )
    out_df["hgnn_margin"] = top1_scores - top2_scores
    out_df["max_exact_overlap_in_case"] = out_df.groupby("case_id")["shared_hpo_count"].transform("max")
    out_df["max_ic_overlap_in_case"] = out_df.groupby("case_id")["ic_weighted_overlap"].transform("max")
    out_df["evidence_rank_by_ic"] = (
        out_df.sort_values(
            ["case_id", "ic_weighted_overlap", "shared_hpo_count", "original_rank"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("case_id")
        .cumcount()
        + 1
    )
    out_df = out_df.sort_values(["case_id", "original_rank"], kind="stable").reset_index(drop=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_config_path": str(data_config_path.resolve()),
        "train_config_path": str(train_config_path.resolve()),
        "case_source": str(case_source),
        "checkpoint_path": str(resolved_checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "top_k": real_top_k,
        "num_cases": int(len(processed_case_ids)),
        "num_rows": int(len(out_df)),
        "output_path": str(output_path.resolve()),
        "semantic": semantic_metadata,
        "feature_definitions": {
            "exact_overlap": "shared_hpo_count / sqrt(case_hpo_count * disease_hpo_count)",
            "ic_weighted_overlap": "sum(case-side HPO specificity for shared HPO) / sum(case-side HPO specificity)",
            "case_coverage": "shared_hpo_count / case_hpo_count",
            "disease_coverage": "shared_hpo_count / disease_hpo_count",
            "hgnn_margin": "case-level top1_hgnn_score - top2_hgnn_score",
            "max_exact_overlap_in_case": "case-level max shared_hpo_count among top50 candidates",
            "max_ic_overlap_in_case": "case-level max ic_weighted_overlap among top50 candidates",
            "evidence_rank_by_ic": "candidate rank within a case by ic_weighted_overlap desc",
            "semantic_ic_overlap": "case IC coverage by exact or HPO ancestor/descendant matches",
            "semantic_coverage_score": "mean of case and disease coverage under exact or HPO ancestor/descendant matches",
            "core_missing_semantic_top5": "1 - weighted coverage of the candidate disease top5 weighted HPOs by exact or HPO ancestor/descendant case matches",
            "core_missing_exact_top5": "1 - weighted exact coverage of the candidate disease top5 weighted HPOs by case HPOs",
        },
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return output_path


def main() -> None:
    args = parse_args()
    export_top50_candidates(
        data_config_path=args.data_config_path,
        train_config_path=args.train_config_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        top_k=args.top_k,
        hpo_ontology_path=args.hpo_ontology_path,
        case_source=args.case_source,
    )


if __name__ == "__main__":
    main()
