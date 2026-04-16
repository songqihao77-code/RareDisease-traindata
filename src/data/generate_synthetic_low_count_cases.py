from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "total_data" / "mondo_label_case_counts_total_data.xlsx"
)
DEFAULT_TOTAL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "total_data"
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "raw_data" / "DiseaseHy"
DEFAULT_MONDO_JSON_PATH = PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json"
DEFAULT_HPO_JSON_PATH = PROJECT_ROOT / "data" / "raw_data" / "hp-base.json"
DEFAULT_HPO_INDEX_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "HPO_index_v4.xlsx"
DEFAULT_DISEASE_INDEX_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generation_data"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "synthetic_low_count_mondo_cases_knowledge_llm.xlsx"
CHECKPOINT_SUFFIX = ".checkpoint.pkl"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
API_KEY_PLACEHOLDER = "PASTE_YOUR_SILICONFLOW_API_KEY_HERE"
DEFAULT_API_KEY = "sk-qrpdbywtmhliiexfmnzytwemtpbymccxgwdjtvbobmminrkq"

REAL_DATASET_EXCLUDES = {
    "PubCaseFinder_DiseaseHyperedge.xlsx",
    "mondo_label_case_counts_total_data.xlsx",
    "mondo_overlap_analysis.xlsx",
}


@dataclass(frozen=True)
class GenerationConfig:
    support_threshold: int = 10
    seed: int = 20260327
    start_case_number: int = 1
    core_min_sources: int = 2
    hard_core_min_sources: int = 3
    target_hpo_slack: int = 2
    max_noise_hpo: int = 2
    noise_support_quantile: float = 0.75
    max_noise_candidates: int = 16
    llm_timeout_sec: int = 120
    llm_max_retries: int = 3
    llm_temperature: float = 0.2
    llm_max_tokens: int = 256
    max_generation_attempts: int = 8
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL


@dataclass(frozen=True)
class SelectorRequest:
    mondo_id: str
    disease_name: str
    hard_core_hpo_ids: tuple[str, ...]
    soft_core_hpo_ids: tuple[str, ...]
    optional_non_core_hpo_ids: tuple[str, ...]
    noise_candidate_ids: tuple[str, ...]
    target_hpo_count: int
    min_hpo_count: int
    max_hpo_count: int
    max_noise_hpo: int


@dataclass(frozen=True)
class SelectorDecision:
    selected_soft_core_hpo_ids: tuple[str, ...]
    selected_optional_hpo_ids: tuple[str, ...]
    selected_noise_hpo_ids: tuple[str, ...]
    raw_response: str
    selector_mode: str
    used_fallback: bool = False


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
    )


def _load_vocab_ids(path: Path, id_col: str) -> set[str]:
    df = pd.read_excel(path, usecols=[id_col], dtype=str)
    return set(_clean_text(df[id_col]).dropna().astype(str))


def _unique_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _weighted_sample_without_replacement(
    rng: np.random.Generator,
    values: np.ndarray,
    sample_size: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if sample_size <= 0 or len(values) == 0:
        return np.asarray([], dtype=values.dtype if len(values) else object)
    if sample_size >= len(values):
        return values.copy()

    if weights is None:
        probs = None
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, a_min=0.0, a_max=None)
        probs = None if float(weights.sum()) <= 0.0 else weights / weights.sum()

    chosen_idx = rng.choice(len(values), size=sample_size, replace=False, p=probs)
    return values[np.asarray(chosen_idx, dtype=int)]


def _format_hpo_list(hpo_ids: list[str] | tuple[str, ...], hpo_name_map: dict[str, str]) -> str:
    if not hpo_ids:
        return "(none)"
    return "\n".join(f"- {hpo_id}: {hpo_name_map.get(hpo_id, hpo_id)}" for hpo_id in hpo_ids)


def _serialize_hpo_ids(hpo_ids: list[str] | tuple[str, ...] | set[str]) -> str:
    return "|".join(sorted(map(str, hpo_ids)))


def _deserialize_hpo_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ()
    return tuple(part for part in text.split("|") if part)


def _normalize_error_reason(error_text: Any) -> str:
    text = str(error_text or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if "duplicate case signature" in lowered:
        return "duplicate_case_signature"
    if "generated case has too many hpos" in lowered:
        return "too_many_hpos"
    if "generated case has too few hpos" in lowered:
        return "too_few_hpos"
    if "invalid noise hpos" in lowered:
        return "invalid_noise_hpos"
    if "invalid optional hpos" in lowered:
        return "invalid_optional_hpos"
    if "invalid soft-core hpos" in lowered:
        return "invalid_soft_core_hpos"
    if "missing required hard-core hpos" in lowered:
        return "missing_required_hard_core_hpos"
    if "failed to extract json object from response" in lowered:
        return "llm_non_json_response"
    if lowered.startswith("http "):
        return "llm_http_error"
    if "does not have any knowledge hpo annotations" in lowered:
        return "no_knowledge_hpos"
    return text[:120]


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Failed to extract JSON object from response.")
    return text[start : end + 1]


def _normalize_obo_id(node_id: str, prefix: str) -> str | None:
    prefix_upper = prefix.upper()
    normalized = node_id.strip()
    if not normalized:
        return None
    if f"{prefix_upper}:" in normalized:
        suffix = normalized.split(f"{prefix_upper}:", maxsplit=1)[1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
    elif f"{prefix_upper}_" in normalized:
        suffix = normalized.split(f"{prefix_upper}_", maxsplit=1)[1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
    else:
        return None
    if not digits:
        return None
    return f"{prefix_upper}:{digits.zfill(7)}"


def default_checkpoint_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}{CHECKPOINT_SUFFIX}"


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint_path} is not a valid checkpoint payload.")
    return payload


def save_checkpoint(
    checkpoint_path: Path,
    *,
    config: GenerationConfig,
    next_case_number: int,
    generated_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    completed_mondo_ids: set[str],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "support_threshold": int(config.support_threshold),
        "seed": int(config.seed),
        "start_case_number": int(config.start_case_number),
        "base_url": str(config.base_url),
        "model": str(config.model),
        "core_min_sources": int(config.core_min_sources),
        "hard_core_min_sources": int(config.hard_core_min_sources),
        "target_hpo_slack": int(config.target_hpo_slack),
        "next_case_number": int(next_case_number),
        "generated_rows": generated_rows,
        "metadata_rows": metadata_rows,
        "summary_rows": summary_rows,
        "audit_rows": audit_rows,
        "completed_mondo_ids": sorted(map(str, completed_mondo_ids)),
    }
    temp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    with temp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(checkpoint_path)


def maybe_save_checkpoint(
    checkpoint_path: Path | None,
    *,
    config: GenerationConfig,
    next_case_number: int,
    generated_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    completed_mondo_ids: set[str],
    reason: str,
) -> None:
    if checkpoint_path is None:
        return
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        next_case_number=next_case_number,
        generated_rows=generated_rows,
        metadata_rows=metadata_rows,
        summary_rows=summary_rows,
        audit_rows=audit_rows,
        completed_mondo_ids=completed_mondo_ids,
    )
    print(
        "[INFO] checkpoint_saved "
        f"path={checkpoint_path} "
        f"reason={reason} "
        f"completed_mondo_count={len(completed_mondo_ids)} "
        f"generated_case_count={len(metadata_rows)}",
        flush=True,
    )


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    return int(float(text))


def summarize_existing_mondo_progress(
    mondo_id: str,
    metadata_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, int]:
    accepted_rows = [row for row in metadata_rows if str(row.get("mondo_label", "")) == mondo_id]
    audit_rows_for_mondo = [row for row in audit_rows if str(row.get("mondo_label", "")) == mondo_id]
    processed_generation_count = len(audit_rows_for_mondo)
    failed_case_count = sum(1 for row in audit_rows_for_mondo if str(row.get("status", "")) == "failed")
    return {
        "processed_generation_count": processed_generation_count,
        "generated_case_count": len(accepted_rows),
        "failed_case_count": failed_case_count,
        "generated_hpo_row_count": sum(_to_int(row.get("final_hpo_count")) for row in accepted_rows),
        "total_selected_soft_core_count": sum(_to_int(row.get("selected_soft_core_count")) for row in accepted_rows),
        "total_selected_optional_count": sum(_to_int(row.get("selected_optional_count")) for row in accepted_rows),
        "total_selected_noise_count": sum(_to_int(row.get("selected_noise_count")) for row in accepted_rows),
        "total_attempt_count": sum(_to_int(row.get("attempt_count")) for row in audit_rows_for_mondo),
    }


def load_target_mondo_counts(
    stats_path: Path,
    support_threshold: int,
    disease_vocab: set[str],
) -> pd.DataFrame:
    overall_df = pd.read_excel(stats_path, sheet_name="overall_counts", dtype=str)
    overall_df["mondo_label"] = _clean_text(overall_df["mondo_label"])
    overall_df["total_case_pair_count"] = pd.to_numeric(
        overall_df["total_case_pair_count"],
        errors="coerce",
    )
    overall_df = overall_df.dropna(subset=["mondo_label", "total_case_pair_count"]).copy()
    overall_df["total_case_pair_count"] = overall_df["total_case_pair_count"].astype(int)
    overall_df = overall_df[overall_df["mondo_label"].isin(disease_vocab)].copy()
    target_df = overall_df[overall_df["total_case_pair_count"] < support_threshold].copy()
    target_df["target_generated_case_count"] = support_threshold - target_df["total_case_pair_count"]
    return target_df.sort_values(["total_case_pair_count", "mondo_label"], ascending=[True, True])


def load_ontology_name_map(json_path: Path, prefix: str) -> dict[str, str]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"{json_path.name} is missing graphs[0].")

    mapping: dict[str, str] = {}
    for node in graphs[0].get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "CLASS":
            continue
        normalized_id = _normalize_obo_id(str(node.get("id", "")), prefix)
        if not normalized_id:
            continue
        label = str(node.get("lbl", "")).strip()
        if label:
            mapping[normalized_id] = label

    if not mapping:
        raise ValueError(f"Failed to parse any {prefix} labels from {json_path}.")
    return mapping


def load_real_signatures_and_support(
    total_data_dir: Path,
    hpo_vocab: set[str],
    disease_vocab: set[str],
) -> tuple[dict[str, set[frozenset[str]]], pd.Series, dict[str, dict[str, int]]]:
    frames: list[pd.DataFrame] = []
    for path in sorted(total_data_dir.glob("*.xlsx")):
        if path.name.startswith("~$") or path.name in REAL_DATASET_EXCLUDES:
            continue
        df = pd.read_excel(path, usecols=["case_id", "mondo_label", "hpo_id"], dtype=str)
        df["case_id"] = _clean_text(df["case_id"])
        df["mondo_label"] = _clean_text(df["mondo_label"])
        df["hpo_id"] = _clean_text(df["hpo_id"])
        df = df.dropna(subset=["case_id", "mondo_label", "hpo_id"]).copy()
        df = df[df["mondo_label"].isin(disease_vocab) & df["hpo_id"].isin(hpo_vocab)].copy()
        if df.empty:
            continue
        df["case_key"] = path.stem + "_" + df["case_id"].astype(str)
        frames.append(df[["case_key", "mondo_label", "hpo_id"]])

    if not frames:
        raise ValueError("Failed to load any real cases from total_data.")

    merged = pd.concat(frames, ignore_index=True)
    signatures_by_mondo: dict[str, set[frozenset[str]]] = {}
    hpo_count_stats_by_mondo: dict[str, dict[str, int]] = {}
    grouped_case_lengths: dict[str, list[int]] = {}
    for (_, mondo_label), group_df in merged.groupby(["case_key", "mondo_label"], sort=False):
        hpo_ids = frozenset(map(str, _unique_preserve_order(group_df["hpo_id"].tolist())))
        mondo_id = str(mondo_label)
        signatures_by_mondo.setdefault(mondo_id, set()).add(hpo_ids)
        grouped_case_lengths.setdefault(mondo_id, []).append(len(hpo_ids))

    for mondo_id, lengths in grouped_case_lengths.items():
        arr = np.asarray(lengths, dtype=int)
        hpo_count_stats_by_mondo[mondo_id] = {
            "min_hpo_count": int(arr.min()),
            "max_hpo_count": int(arr.max()),
            "median_hpo_count": int(np.median(arr)),
        }

    support_df = merged.drop_duplicates(subset=["case_key", "hpo_id"])
    global_hpo_support = support_df.groupby("hpo_id")["case_key"].nunique().astype(int).sort_values(
        ascending=False
    )
    return signatures_by_mondo, global_hpo_support, hpo_count_stats_by_mondo


def load_knowledge_annotations(
    knowledge_dir: Path,
    hpo_vocab: set[str],
    disease_vocab: set[str],
) -> pd.DataFrame:
    source_to_file = {
        "GARD": "GARD.xlsx",
        "HPOA": "HPOA.xlsx",
        "orphanet": "orphanet.xlsx",
    }
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}

    for source_name, file_name in source_to_file.items():
        path = knowledge_dir / file_name
        df = pd.read_excel(path, usecols=["mondo_id", "hpo_id", "weight"], dtype=str)
        df["mondo_id"] = _clean_text(df["mondo_id"])
        df["hpo_id"] = _clean_text(df["hpo_id"])
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df = df.dropna(subset=["mondo_id", "hpo_id", "weight"]).copy()
        df = df[df["mondo_id"].isin(disease_vocab) & df["hpo_id"].isin(hpo_vocab)].copy()

        for row in df.itertuples(index=False):
            key = (str(row.mondo_id), str(row.hpo_id))
            bucket = aggregate.setdefault(
                key,
                {"mondo_label": str(row.mondo_id), "hpo_id": str(row.hpo_id), "source_names": set(), "weights": []},
            )
            bucket["source_names"].add(source_name)
            bucket["weights"].append(float(row.weight))

    rows: list[dict[str, Any]] = []
    for bucket in aggregate.values():
        source_names = sorted(bucket["source_names"])
        weights = bucket["weights"]
        rows.append(
            {
                "mondo_label": bucket["mondo_label"],
                "hpo_id": bucket["hpo_id"],
                "source_count": int(len(source_names)),
                "source_names": "|".join(source_names),
                "max_weight": float(max(weights)),
                "mean_weight": float(sum(weights) / len(weights)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["mondo_label", "hpo_id", "source_count", "source_names", "max_weight", "mean_weight"]
        )
    return pd.DataFrame(rows).sort_values(
        ["mondo_label", "source_count", "max_weight", "hpo_id"],
        ascending=[True, False, False, True],
    )


def build_noise_pool(
    global_hpo_support: pd.Series,
    hpo_name_map: dict[str, str],
    noise_support_quantile: float,
) -> pd.DataFrame:
    if global_hpo_support.empty:
        raise ValueError("global_hpo_support is empty.")

    cutoff = float(global_hpo_support.quantile(noise_support_quantile))
    filtered = global_hpo_support[global_hpo_support >= cutoff].astype(int)
    if filtered.empty:
        filtered = global_hpo_support.astype(int)

    noise_df = filtered.reset_index()
    noise_df.columns = ["hpo_id", "case_support"]
    noise_df["hpo_name"] = noise_df["hpo_id"].map(lambda hpo_id: hpo_name_map.get(hpo_id, hpo_id))
    return noise_df.sort_values(["case_support", "hpo_id"], ascending=[False, True]).reset_index(drop=True)


def build_knowledge_profiles(
    knowledge_df: pd.DataFrame,
    core_min_sources: int,
    hard_core_min_sources: int,
) -> dict[str, dict[str, tuple[str, ...]]]:
    profiles: dict[str, dict[str, tuple[str, ...]]] = {}
    if knowledge_df.empty:
        return profiles

    for mondo_id, group_df in knowledge_df.groupby("mondo_label", sort=False):
        ordered_hpo_ids = _unique_preserve_order(group_df["hpo_id"].astype(str).tolist())
        hard_core_hpo_ids = _unique_preserve_order(
            group_df.loc[group_df["source_count"] >= hard_core_min_sources, "hpo_id"].astype(str).tolist()
        )
        soft_core_hpo_ids = _unique_preserve_order(
            group_df.loc[
                (group_df["source_count"] >= core_min_sources) & (group_df["source_count"] < hard_core_min_sources),
                "hpo_id",
            ].astype(str).tolist()
        )
        core_set = set(hard_core_hpo_ids) | set(soft_core_hpo_ids)
        optional_hpo_ids = [hpo_id for hpo_id in ordered_hpo_ids if hpo_id not in core_set]
        profiles[str(mondo_id)] = {
            "knowledge_hpo_ids": tuple(ordered_hpo_ids),
            "hard_core_hpo_ids": tuple(hard_core_hpo_ids),
            "soft_core_hpo_ids": tuple(soft_core_hpo_ids),
            "optional_non_core_hpo_ids": tuple(optional_hpo_ids),
        }
    return profiles


def resolve_hpo_count_range(
    mondo_id: str,
    hard_core_hpo_ids: tuple[str, ...],
    hpo_count_stats_by_mondo: dict[str, dict[str, int]],
    global_default_target_hpo_count: int,
    target_hpo_slack: int,
) -> tuple[int, int, int]:
    stats = hpo_count_stats_by_mondo.get(mondo_id, {})
    target_hpo_count = int(stats.get("median_hpo_count", global_default_target_hpo_count))
    min_hpo_count = max(int(len(hard_core_hpo_ids)), target_hpo_count - int(target_hpo_slack))
    max_hpo_count = max(min_hpo_count, target_hpo_count + int(target_hpo_slack))
    return target_hpo_count, min_hpo_count, max_hpo_count


def estimate_non_noise_case_capacity(
    selectable_hpo_count: int,
    min_extra_count: int,
    max_extra_count: int,
    limit: int,
) -> int:
    if limit <= 0:
        return 0
    if min_extra_count > max_extra_count:
        return 0
    total = 0
    upper = min(int(max_extra_count), int(selectable_hpo_count))
    lower = max(0, int(min_extra_count))
    for extra_count in range(lower, upper + 1):
        total += math.comb(int(selectable_hpo_count), extra_count)
        if total >= limit:
            return int(limit)
    return int(total)


def build_generation_plan(
    requested_case_count: int,
    hard_core_hpo_ids: tuple[str, ...],
    soft_core_hpo_ids: tuple[str, ...],
    optional_hpo_ids: tuple[str, ...],
    target_hpo_count: int,
    min_hpo_count: int,
    max_hpo_count: int,
    config: GenerationConfig,
) -> dict[str, Any]:
    hard_core_count = int(len(hard_core_hpo_ids))
    remaining_slot_budget = max(0, int(max_hpo_count) - hard_core_count)
    effective_target_hpo_count = min(max(int(target_hpo_count), hard_core_count), int(max_hpo_count))
    effective_max_noise_hpo = min(int(config.max_noise_hpo), remaining_slot_budget)
    if hard_core_count >= effective_target_hpo_count:
        effective_max_noise_hpo = 0

    effective_soft_core_hpo_ids = tuple(soft_core_hpo_ids) if remaining_slot_budget > 0 else ()
    effective_optional_hpo_ids = tuple(optional_hpo_ids) if remaining_slot_budget > 0 else ()
    selectable_hpo_count = int(len(effective_soft_core_hpo_ids) + len(effective_optional_hpo_ids))
    min_extra_count = max(0, int(min_hpo_count) - hard_core_count)
    max_extra_count = remaining_slot_budget

    if selectable_hpo_count == 0 and effective_max_noise_hpo == 0:
        estimated_case_capacity = 1
    elif effective_max_noise_hpo == 0:
        estimated_case_capacity = estimate_non_noise_case_capacity(
            selectable_hpo_count=selectable_hpo_count,
            min_extra_count=min_extra_count,
            max_extra_count=max_extra_count,
            limit=max(1, int(requested_case_count)),
        )
    else:
        estimated_case_capacity = int(requested_case_count)

    effective_requested_case_count = min(int(requested_case_count), int(estimated_case_capacity))
    deterministic_case_only = selectable_hpo_count == 0 and effective_max_noise_hpo == 0
    return {
        "effective_requested_case_count": int(effective_requested_case_count),
        "effective_target_hpo_count": int(effective_target_hpo_count),
        "effective_max_noise_hpo": int(effective_max_noise_hpo),
        "effective_soft_core_hpo_ids": effective_soft_core_hpo_ids,
        "effective_optional_hpo_ids": effective_optional_hpo_ids,
        "remaining_slot_budget": int(remaining_slot_budget),
        "estimated_case_capacity": int(estimated_case_capacity),
        "deterministic_case_only": bool(deterministic_case_only),
    }


def choose_noise_candidates(
    noise_pool_df: pd.DataFrame,
    excluded_hpo_ids: set[str],
    rng: np.random.Generator,
    config: GenerationConfig,
) -> list[str]:
    candidate_df = noise_pool_df.loc[~noise_pool_df["hpo_id"].isin(excluded_hpo_ids)].copy()
    if candidate_df.empty:
        return []
    sample_size = min(int(config.max_noise_candidates), len(candidate_df))
    chosen_hpo_ids = _weighted_sample_without_replacement(
        rng=rng,
        values=candidate_df["hpo_id"].astype(str).to_numpy(),
        sample_size=sample_size,
        weights=candidate_df["case_support"].to_numpy(dtype=float),
    )
    return list(map(str, chosen_hpo_ids.tolist()))


def build_selector_request(
    mondo_id: str,
    disease_name: str,
    hard_core_hpo_ids: list[str],
    soft_core_hpo_ids: list[str],
    optional_non_core_hpo_ids: list[str],
    noise_candidate_ids: list[str],
    target_hpo_count: int,
    min_hpo_count: int,
    max_hpo_count: int,
    config: GenerationConfig,
    max_noise_hpo: int | None = None,
) -> SelectorRequest:
    return SelectorRequest(
        mondo_id=mondo_id,
        disease_name=disease_name,
        hard_core_hpo_ids=tuple(hard_core_hpo_ids),
        soft_core_hpo_ids=tuple(soft_core_hpo_ids),
        optional_non_core_hpo_ids=tuple(optional_non_core_hpo_ids),
        noise_candidate_ids=tuple(noise_candidate_ids),
        target_hpo_count=int(target_hpo_count),
        min_hpo_count=int(min_hpo_count),
        max_hpo_count=int(max_hpo_count),
        max_noise_hpo=int(config.max_noise_hpo if max_noise_hpo is None else max_noise_hpo),
    )


def build_case_from_decision(
    request: SelectorRequest,
    selected_soft_core_hpo_ids: tuple[str, ...],
    selected_optional_hpo_ids: tuple[str, ...],
    selected_noise_hpo_ids: tuple[str, ...],
) -> list[str]:
    return _unique_preserve_order(
        list(request.hard_core_hpo_ids)
        + list(selected_soft_core_hpo_ids)
        + list(selected_optional_hpo_ids)
        + list(selected_noise_hpo_ids)
    )


def validate_selector_decision(
    request: SelectorRequest,
    decision: SelectorDecision,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    soft_core_set = set(request.soft_core_hpo_ids)
    optional_set = set(request.optional_non_core_hpo_ids)
    noise_set = set(request.noise_candidate_ids)
    selected_soft_core = tuple(_unique_preserve_order(list(decision.selected_soft_core_hpo_ids)))
    selected_optional = tuple(_unique_preserve_order(list(decision.selected_optional_hpo_ids)))
    selected_noise = tuple(_unique_preserve_order(list(decision.selected_noise_hpo_ids)))

    if any(hpo_id not in soft_core_set for hpo_id in selected_soft_core):
        raise ValueError(f"Decision contains invalid soft-core HPOs: {selected_soft_core}")
    if any(hpo_id not in optional_set for hpo_id in selected_optional):
        raise ValueError(f"Decision contains invalid optional HPOs: {selected_optional}")
    if any(hpo_id not in noise_set for hpo_id in selected_noise):
        raise ValueError(f"Decision contains invalid noise HPOs: {selected_noise}")
    if len(selected_noise) > request.max_noise_hpo:
        raise ValueError(f"Selected noise HPO count {len(selected_noise)} exceeds limit {request.max_noise_hpo}.")

    return selected_soft_core, selected_optional, selected_noise


def validate_generated_case(
    mondo_id: str,
    final_hpo_ids: list[str],
    required_hard_core_hpo_ids: tuple[str, ...],
    min_hpo_count: int,
    max_hpo_count: int,
    existing_signatures: set[frozenset[str]],
) -> frozenset[str]:
    signature = frozenset(map(str, final_hpo_ids))
    if not signature:
        raise ValueError(f"{mondo_id} generated empty HPO set.")
    if not set(required_hard_core_hpo_ids).issubset(signature):
        raise ValueError(f"{mondo_id} generated case is missing required hard-core HPOs.")
    if len(signature) < int(min_hpo_count):
        raise ValueError(f"{mondo_id} generated case has too few HPOs: {len(signature)} < {min_hpo_count}.")
    if len(signature) > int(max_hpo_count):
        raise ValueError(f"{mondo_id} generated case has too many HPOs: {len(signature)} > {max_hpo_count}.")
    if signature in existing_signatures:
        raise ValueError(f"{mondo_id} generated duplicate case signature.")
    return signature


class HeuristicSelector:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, request: SelectorRequest, hpo_name_map: dict[str, str]) -> SelectorDecision:
        del hpo_name_map
        soft_core_ids = np.asarray(request.soft_core_hpo_ids, dtype=object)
        optional_ids = np.asarray(request.optional_non_core_hpo_ids, dtype=object)
        noise_ids = np.asarray(request.noise_candidate_ids, dtype=object)

        hard_count = len(request.hard_core_hpo_ids)
        min_soft_plus_optional = max(0, request.min_hpo_count - hard_count)
        max_soft_plus_optional = max(0, request.max_hpo_count - hard_count)
        max_available_non_noise = len(soft_core_ids) + len(optional_ids)
        if max_available_non_noise <= 0:
            target_non_noise = 0
        else:
            lower = min(min_soft_plus_optional, max_available_non_noise)
            upper = min(max_soft_plus_optional, max_available_non_noise)
            if upper < lower:
                lower = upper
            preferred = min(max(request.target_hpo_count - hard_count, lower), upper)
            target_non_noise = preferred

        target_soft = min(len(soft_core_ids), target_non_noise)
        chosen_soft = _weighted_sample_without_replacement(
            rng=self.rng,
            values=soft_core_ids,
            sample_size=target_soft,
            weights=None,
        )
        selected_soft_core = tuple(map(str, chosen_soft.tolist()))

        remaining_non_noise = max(0, target_non_noise - len(selected_soft_core))
        chosen_optional = _weighted_sample_without_replacement(
            rng=self.rng,
            values=optional_ids,
            sample_size=min(len(optional_ids), remaining_non_noise),
            weights=None,
        )
        selected_optional = tuple(map(str, chosen_optional.tolist()))

        max_noise = min(request.max_noise_hpo, len(noise_ids), max(0, request.max_hpo_count - hard_count - len(selected_soft_core) - len(selected_optional)))
        target_noise = min(max(0, request.target_hpo_count - hard_count - len(selected_soft_core) - len(selected_optional)), max_noise)
        chosen_noise = _weighted_sample_without_replacement(
            rng=self.rng,
            values=noise_ids,
            sample_size=target_noise,
            weights=None,
        )
        selected_noise = tuple(map(str, chosen_noise.tolist()))
        raw_response = json.dumps(
            {
                "selected_soft_core_hpo_ids": list(selected_soft_core),
                "selected_optional_hpo_ids": list(selected_optional),
                "selected_noise_hpo_ids": list(selected_noise),
                "rationale": "heuristic_selector",
            },
            ensure_ascii=True,
        )
        return SelectorDecision(
            selected_soft_core_hpo_ids=selected_soft_core,
            selected_optional_hpo_ids=selected_optional,
            selected_noise_hpo_ids=selected_noise,
            raw_response=raw_response,
            selector_mode="heuristic",
            used_fallback=False,
        )


class SiliconFlowSelector:
    def __init__(
        self,
        api_key: str,
        config: GenerationConfig,
        fallback_selector: HeuristicSelector | None = None,
    ) -> None:
        self.api_key = api_key
        self.config = config
        self.fallback_selector = fallback_selector
        self.response_format_enabled = True
        self.enable_thinking_enabled = True

    def _build_endpoint(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return base_url + "/chat/completions"
        return base_url + "/chat/completions"

    def _build_messages(
        self,
        request: SelectorRequest,
        hpo_name_map: dict[str, str],
    ) -> list[dict[str, str]]:
        rules = [
            "You are generating a synthetic rare-disease HPO profile from curated disease knowledge only.",
            "Use only the MONDO ID, disease name, required hard-core HPOs, selectable soft-core HPOs, selectable optional non-core HPOs, and allowed noise HPOs provided below.",
            "Never use or infer any hidden patient-level dataset content.",
            "Keep every required hard-core HPO.",
            "Soft-core selections must come only from selectable_soft_core_hpo_ids.",
            "Optional selections must come only from optional_non_core_hpo_ids.",
            f"Final total HPO count must be between {request.min_hpo_count} and {request.max_hpo_count}, and target {request.target_hpo_count} when possible.",
            f"Select at most {request.max_noise_hpo} noise HPOs and only from allowed_noise_hpo_ids.",
            'Return JSON only with keys "selected_soft_core_hpo_ids", "selected_optional_hpo_ids", "selected_noise_hpo_ids", and optional "rationale".',
        ]
        user_prompt = "\n".join(
            [
                f"MONDO ID: {request.mondo_id}",
                f"Disease name: {request.disease_name}",
                f"target_hpo_count: {request.target_hpo_count}",
                f"min_hpo_count: {request.min_hpo_count}",
                f"max_hpo_count: {request.max_hpo_count}",
                "required_hard_core_hpo_ids:",
                _format_hpo_list(list(request.hard_core_hpo_ids), hpo_name_map),
                "selectable_soft_core_hpo_ids:",
                _format_hpo_list(list(request.soft_core_hpo_ids), hpo_name_map),
                "optional_non_core_hpo_ids:",
                _format_hpo_list(list(request.optional_non_core_hpo_ids), hpo_name_map),
                "allowed_noise_hpo_ids:",
                _format_hpo_list(list(request.noise_candidate_ids), hpo_name_map),
            ]
        )
        return [
            {"role": "system", "content": "\n".join(rules)},
            {"role": "user", "content": user_prompt},
        ]

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._build_endpoint(),
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.llm_timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_response(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("SiliconFlow response is missing choices[0].")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("SiliconFlow response is missing message.content.")
        return content

    def _select_with_api(
        self,
        request: SelectorRequest,
        hpo_name_map: dict[str, str],
    ) -> SelectorDecision:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._build_messages(request, hpo_name_map),
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
        }
        if self.enable_thinking_enabled:
            payload["enable_thinking"] = False
        if self.response_format_enabled:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for _ in range(self.config.llm_max_retries):
            try:
                response_payload = self._post_json(payload)
                raw_response = self._parse_response(response_payload)
                parsed = json.loads(_extract_json_object(raw_response))
                selected_soft_core = tuple(map(str, parsed.get("selected_soft_core_hpo_ids", []) or []))
                selected_optional = tuple(map(str, parsed.get("selected_optional_hpo_ids", []) or []))
                selected_noise = tuple(map(str, parsed.get("selected_noise_hpo_ids", []) or []))
                return SelectorDecision(
                    selected_soft_core_hpo_ids=selected_soft_core,
                    selected_optional_hpo_ids=selected_optional,
                    selected_noise_hpo_ids=selected_noise,
                    raw_response=raw_response,
                    selector_mode="siliconflow",
                    used_fallback=False,
                )
            except urllib.error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="ignore")
                error_summary = error_text.strip() or exc.reason or str(exc)
                last_error = RuntimeError(f"HTTP {exc.code}: {error_summary}")
                if exc.code in (401, 403):
                    raise last_error from exc
                if self.response_format_enabled and "response_format" in error_text.lower():
                    self.response_format_enabled = False
                elif self.enable_thinking_enabled and "enable_thinking" in error_text.lower():
                    self.enable_thinking_enabled = False
                else:
                    time.sleep(1.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(1.0)

        raise RuntimeError(f"SiliconFlow request failed after retries: {last_error}") from last_error

    def select(self, request: SelectorRequest, hpo_name_map: dict[str, str]) -> SelectorDecision:
        try:
            return self._select_with_api(request, hpo_name_map)
        except Exception:
            if self.fallback_selector is None:
                raise
            fallback = self.fallback_selector.select(request, hpo_name_map)
            return SelectorDecision(
                selected_soft_core_hpo_ids=fallback.selected_soft_core_hpo_ids,
                selected_optional_hpo_ids=fallback.selected_optional_hpo_ids,
                selected_noise_hpo_ids=fallback.selected_noise_hpo_ids,
                raw_response=fallback.raw_response,
                selector_mode="siliconflow",
                used_fallback=True,
            )


def seed_for_text(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def build_selector_for_mondo(
    base_selector: HeuristicSelector | SiliconFlowSelector,
    config: GenerationConfig,
    mondo_id: str,
) -> HeuristicSelector | SiliconFlowSelector:
    mondo_seed = seed_for_text(config.seed, mondo_id)
    if isinstance(base_selector, HeuristicSelector):
        return HeuristicSelector(seed=mondo_seed)
    fallback_selector = HeuristicSelector(seed=mondo_seed) if base_selector.fallback_selector else None
    return SiliconFlowSelector(
        api_key=base_selector.api_key,
        config=config,
        fallback_selector=fallback_selector,
    )


def upsert_summary_row(summary_rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    mondo_id = str(row.get("mondo_label", ""))
    for index, existing_row in enumerate(summary_rows):
        if str(existing_row.get("mondo_label", "")) == mondo_id:
            summary_rows[index] = row
            return
    summary_rows.append(row)


def generate_cases(
    target_df: pd.DataFrame,
    real_signatures_by_mondo: dict[str, set[frozenset[str]]],
    hpo_count_stats_by_mondo: dict[str, dict[str, int]],
    knowledge_df: pd.DataFrame,
    noise_pool_df: pd.DataFrame,
    mondo_name_map: dict[str, str],
    hpo_name_map: dict[str, str],
    selector: HeuristicSelector | SiliconFlowSelector,
    config: GenerationConfig,
    checkpoint_path: Path | None = None,
    checkpoint_every_mondos: int = 10,
    checkpoint_every_cases: int = 1,
    resume_payload: dict[str, Any] | None = None,
    max_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_workers = max(1, int(max_workers))
    return _generate_cases_parallel(
        target_df=target_df,
        real_signatures_by_mondo=real_signatures_by_mondo,
        hpo_count_stats_by_mondo=hpo_count_stats_by_mondo,
        knowledge_df=knowledge_df,
        noise_pool_df=noise_pool_df,
        mondo_name_map=mondo_name_map,
        hpo_name_map=hpo_name_map,
        selector=selector,
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_every_mondos=checkpoint_every_mondos,
        checkpoint_every_cases=checkpoint_every_cases,
        resume_payload=resume_payload,
        max_workers=max_workers,
    )
    rng = np.random.default_rng(config.seed)
    total_target_mondo_count = int(len(target_df))
    total_requested_generated_case_count = int(target_df["target_generated_case_count"].sum()) if not target_df.empty else 0
    print(
        "[INFO] generation_started "
        f"target_mondo_count={total_target_mondo_count} "
        f"requested_generated_case_count={total_requested_generated_case_count}",
        flush=True,
    )

    knowledge_profiles = build_knowledge_profiles(
        knowledge_df,
        config.core_min_sources,
        config.hard_core_min_sources,
    )
    existing_signatures_by_mondo = {
        mondo_id: set(signatures)
        for mondo_id, signatures in real_signatures_by_mondo.items()
    }
    global_default_target_hpo_count = int(
        np.median(
            [
                stats["median_hpo_count"]
                for stats in hpo_count_stats_by_mondo.values()
                if int(stats.get("median_hpo_count", 0)) > 0
            ]
            or [10]
        )
    )

    next_case_number = int(config.start_case_number)
    generated_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    completed_mondo_ids: set[str] = set()
    checkpoint_every_mondos = max(1, int(checkpoint_every_mondos))
    checkpoint_every_cases = max(0, int(checkpoint_every_cases))

    if resume_payload:
        for field_name, expected_value in (
            ("support_threshold", config.support_threshold),
            ("seed", config.seed),
            ("start_case_number", config.start_case_number),
            ("base_url", config.base_url),
            ("model", config.model),
            ("core_min_sources", config.core_min_sources),
            ("hard_core_min_sources", config.hard_core_min_sources),
            ("target_hpo_slack", config.target_hpo_slack),
        ):
            actual_value = resume_payload.get(field_name)
            if actual_value is not None and str(actual_value) != str(expected_value):
                raise ValueError(
                    f"Checkpoint config mismatch for {field_name}: expected={expected_value}, got={actual_value}"
                )

        next_case_number = int(resume_payload.get("next_case_number", next_case_number))
        generated_rows = list(resume_payload.get("generated_rows", []))
        metadata_rows = list(resume_payload.get("metadata_rows", []))
        summary_rows = list(resume_payload.get("summary_rows", []))
        audit_rows = list(resume_payload.get("audit_rows", []))
        completed_mondo_ids = set(map(str, resume_payload.get("completed_mondo_ids", [])))

        for row in metadata_rows:
            mondo_id = str(row.get("mondo_label", "")).strip()
            final_hpo_ids = frozenset(_deserialize_hpo_ids(row.get("final_hpo_ids", "")))
            if mondo_id and final_hpo_ids:
                existing_signatures_by_mondo.setdefault(mondo_id, set()).add(final_hpo_ids)

        print(
            "[INFO] checkpoint_loaded "
            f"path={checkpoint_path} "
            f"completed_mondo_count={len(completed_mondo_ids)} "
            f"generated_case_count={len(metadata_rows)} "
            f"next_case_number={next_case_number}",
            flush=True,
        )

    for mondo_index, target_row in enumerate(target_df.itertuples(index=False), start=1):
        mondo_id = str(target_row.mondo_label)
        if mondo_id in completed_mondo_ids:
            continue

        original_count = int(target_row.total_case_pair_count)
        requested_case_count = int(target_row.target_generated_case_count)
        disease_name = mondo_name_map.get(mondo_id, mondo_id)
        profile = knowledge_profiles.get(mondo_id)
        if profile is None:
            profile = {
                "knowledge_hpo_ids": (),
                "hard_core_hpo_ids": (),
                "soft_core_hpo_ids": (),
                "optional_non_core_hpo_ids": (),
            }

        knowledge_hpo_ids = tuple(profile["knowledge_hpo_ids"])
        hard_core_hpo_ids = tuple(profile["hard_core_hpo_ids"])
        soft_core_hpo_ids = tuple(profile["soft_core_hpo_ids"])
        optional_hpo_ids = tuple(profile["optional_non_core_hpo_ids"])
        target_hpo_count, min_hpo_count, max_hpo_count = resolve_hpo_count_range(
            mondo_id,
            hard_core_hpo_ids,
            hpo_count_stats_by_mondo,
            global_default_target_hpo_count,
            config.target_hpo_slack,
        )
        generation_plan = build_generation_plan(
            requested_case_count=requested_case_count,
            hard_core_hpo_ids=hard_core_hpo_ids,
            soft_core_hpo_ids=soft_core_hpo_ids,
            optional_hpo_ids=optional_hpo_ids,
            target_hpo_count=target_hpo_count,
            min_hpo_count=min_hpo_count,
            max_hpo_count=max_hpo_count,
            config=config,
        )
        effective_requested_case_count = int(generation_plan["effective_requested_case_count"])
        effective_target_hpo_count = int(generation_plan["effective_target_hpo_count"])
        effective_max_noise_hpo = int(generation_plan["effective_max_noise_hpo"])
        effective_soft_core_hpo_ids = tuple(generation_plan["effective_soft_core_hpo_ids"])
        effective_optional_hpo_ids = tuple(generation_plan["effective_optional_hpo_ids"])
        remaining_slot_budget = int(generation_plan["remaining_slot_budget"])
        estimated_case_capacity = int(generation_plan["estimated_case_capacity"])
        deterministic_case_only = bool(generation_plan["deterministic_case_only"])
        existing_signatures = existing_signatures_by_mondo.setdefault(
            mondo_id,
            set(real_signatures_by_mondo.get(mondo_id, set())),
        )

        print(
            "[INFO] mondo_started "
            f"index={mondo_index}/{total_target_mondo_count} "
            f"mondo_label={mondo_id} "
            f"original_count={original_count} "
            f"need_generate={requested_case_count} "
            f"knowledge_hpo_count={len(knowledge_hpo_ids)} "
            f"hard_core_hpo_count={len(hard_core_hpo_ids)} "
            f"soft_core_hpo_count={len(soft_core_hpo_ids)} "
            f"target_hpo_count={target_hpo_count}",
            flush=True,
        )

        progress = summarize_existing_mondo_progress(mondo_id, metadata_rows, audit_rows)
        generated_case_count = int(progress["generated_case_count"])
        failed_case_count = int(progress["failed_case_count"])
        generated_hpo_row_count = int(progress["generated_hpo_row_count"])
        total_selected_soft_core_count = int(progress["total_selected_soft_core_count"])
        total_selected_optional_count = int(progress["total_selected_optional_count"])
        total_selected_noise_count = int(progress["total_selected_noise_count"])
        total_attempt_count = int(progress["total_attempt_count"])
        start_generation_index = int(progress["processed_generation_count"])

        if not knowledge_hpo_ids:
            error_text = f"{mondo_id} does not have any knowledge HPO annotations."
            for generation_index in range(start_generation_index, requested_case_count):
                audit_rows.append(
                    {
                        "case_id": "",
                        "mondo_label": mondo_id,
                        "generation_index": generation_index + 1,
                        "attempt_count": 0,
                        "selector_mode": getattr(selector, "__class__", type(selector)).__name__,
                        "status": "failed",
                        "error": error_text,
                        "raw_response": "",
                        "selected_soft_core_hpo_ids": "",
                        "selected_optional_hpo_ids": "",
                        "selected_noise_hpo_ids": "",
                        "duplicate_retry_count": 0,
                        "used_fallback": False,
                    }
                )
                if checkpoint_every_cases > 0 and len(audit_rows) % checkpoint_every_cases == 0:
                    maybe_save_checkpoint(
                        checkpoint_path,
                        config=config,
                        next_case_number=next_case_number,
                        generated_rows=generated_rows,
                        metadata_rows=metadata_rows,
                        summary_rows=summary_rows,
                        audit_rows=audit_rows,
                        completed_mondo_ids=completed_mondo_ids,
                        reason=f"case_failed:{mondo_id}:{generation_index + 1}",
                    )
            failed_case_count = requested_case_count
        else:
            for generation_index in range(start_generation_index, requested_case_count):
                success = False
                last_error_text = ""
                last_raw_response = ""
                last_selector_mode = getattr(selector, "__class__", type(selector)).__name__
                last_used_fallback = False
                duplicate_retry_count = 0

                for attempt_index in range(1, config.max_generation_attempts + 1):
                    total_attempt_count += 1
                    noise_candidate_ids = choose_noise_candidates(
                        noise_pool_df=noise_pool_df,
                        excluded_hpo_ids=set(knowledge_hpo_ids),
                        rng=rng,
                        config=config,
                    )
                    request = build_selector_request(
                        mondo_id=mondo_id,
                        disease_name=disease_name,
                        hard_core_hpo_ids=list(hard_core_hpo_ids),
                        soft_core_hpo_ids=list(soft_core_hpo_ids),
                        optional_non_core_hpo_ids=list(optional_hpo_ids),
                        noise_candidate_ids=noise_candidate_ids,
                        target_hpo_count=target_hpo_count,
                        min_hpo_count=min_hpo_count,
                        max_hpo_count=max_hpo_count,
                        config=config,
                    )

                    try:
                        decision = selector.select(request, hpo_name_map)
                        last_raw_response = decision.raw_response
                        last_selector_mode = decision.selector_mode
                        last_used_fallback = bool(decision.used_fallback)
                        (
                            selected_soft_core_hpo_ids,
                            selected_optional_hpo_ids,
                            selected_noise_hpo_ids,
                        ) = validate_selector_decision(request, decision)
                        final_hpo_ids = build_case_from_decision(
                            request,
                            selected_soft_core_hpo_ids,
                            selected_optional_hpo_ids,
                            selected_noise_hpo_ids,
                        )
                        signature = validate_generated_case(
                            mondo_id=mondo_id,
                            final_hpo_ids=final_hpo_ids,
                            required_hard_core_hpo_ids=hard_core_hpo_ids,
                            min_hpo_count=min_hpo_count,
                            max_hpo_count=max_hpo_count,
                            existing_signatures=existing_signatures,
                        )
                    except Exception as exc:  # noqa: BLE001
                        last_error_text = str(exc)
                        if "duplicate case signature" in last_error_text:
                            duplicate_retry_count += 1
                        if attempt_index == max_attempts_for_case:
                            audit_rows.append(
                                {
                                    "case_id": "",
                                    "mondo_label": mondo_id,
                                    "generation_index": generation_index + 1,
                                    "attempt_count": attempt_index,
                                    "selector_mode": last_selector_mode,
                                    "status": "failed",
                                    "error": last_error_text,
                                    "raw_response": last_raw_response,
                                    "selected_soft_core_hpo_ids": "",
                                    "selected_optional_hpo_ids": "",
                                    "selected_noise_hpo_ids": "",
                                    "duplicate_retry_count": duplicate_retry_count,
                                    "used_fallback": last_used_fallback,
                                }
                            )
                            if checkpoint_every_cases > 0 and len(audit_rows) % checkpoint_every_cases == 0:
                                maybe_save_checkpoint(
                                    checkpoint_path,
                                    config=config,
                                    next_case_number=next_case_number,
                                    generated_rows=generated_rows,
                                    metadata_rows=metadata_rows,
                                    summary_rows=summary_rows,
                                    audit_rows=audit_rows,
                                    completed_mondo_ids=completed_mondo_ids,
                                    reason=f"case_failed:{mondo_id}:{generation_index + 1}",
                                )
                        continue

                    generated_case_id = f"synthetic_case_{next_case_number}"
                    next_case_number += 1
                    existing_signatures.add(signature)

                    for hpo_id in final_hpo_ids:
                        generated_rows.append(
                            {
                                "case_id": generated_case_id,
                                "mondo_label": mondo_id,
                                "hpo_id": hpo_id,
                                "case_source": "synthetic",
                            }
                        )

                    metadata_rows.append(
                        {
                            "case_id": generated_case_id,
                            "mondo_label": mondo_id,
                            "case_source": "synthetic",
                            "original_total_case_pair_count": original_count,
                            "requested_target_total_case_count": config.support_threshold,
                            "target_hpo_count": int(target_hpo_count),
                            "min_hpo_count": int(min_hpo_count),
                            "max_hpo_count": int(max_hpo_count),
                            "hard_core_count": int(len(hard_core_hpo_ids)),
                            "soft_core_count": int(len(soft_core_hpo_ids)),
                            "optional_non_core_count": int(len(optional_hpo_ids)),
                            "noise_candidate_count": int(len(noise_candidate_ids)),
                            "final_hpo_count": int(len(final_hpo_ids)),
                            "selected_soft_core_count": int(len(selected_soft_core_hpo_ids)),
                            "selected_optional_count": int(len(selected_optional_hpo_ids)),
                            "selected_noise_count": int(len(selected_noise_hpo_ids)),
                            "hard_core_hpo_ids": _serialize_hpo_ids(hard_core_hpo_ids),
                            "soft_core_hpo_ids": _serialize_hpo_ids(soft_core_hpo_ids),
                            "optional_non_core_hpo_ids": _serialize_hpo_ids(optional_hpo_ids),
                            "noise_candidate_hpo_ids": _serialize_hpo_ids(noise_candidate_ids),
                            "selected_soft_core_hpo_ids": _serialize_hpo_ids(selected_soft_core_hpo_ids),
                            "selected_optional_hpo_ids": _serialize_hpo_ids(selected_optional_hpo_ids),
                            "selected_noise_hpo_ids": _serialize_hpo_ids(selected_noise_hpo_ids),
                            "final_hpo_ids": _serialize_hpo_ids(final_hpo_ids),
                            "duplicate_retry_count": int(duplicate_retry_count),
                            "quality_filter_reason": "keep",
                        }
                    )
                    audit_rows.append(
                        {
                            "case_id": generated_case_id,
                            "mondo_label": mondo_id,
                            "generation_index": generation_index + 1,
                            "attempt_count": attempt_index,
                            "selector_mode": decision.selector_mode,
                            "status": "accepted",
                            "error": "",
                            "raw_response": decision.raw_response,
                            "selected_soft_core_hpo_ids": _serialize_hpo_ids(selected_soft_core_hpo_ids),
                            "selected_optional_hpo_ids": _serialize_hpo_ids(selected_optional_hpo_ids),
                            "selected_noise_hpo_ids": _serialize_hpo_ids(selected_noise_hpo_ids),
                            "duplicate_retry_count": int(duplicate_retry_count),
                            "used_fallback": bool(decision.used_fallback),
                        }
                    )

                    generated_case_count += 1
                    generated_hpo_row_count += len(final_hpo_ids)
                    total_selected_soft_core_count += len(selected_soft_core_hpo_ids)
                    total_selected_optional_count += len(selected_optional_hpo_ids)
                    total_selected_noise_count += len(selected_noise_hpo_ids)
                    if checkpoint_every_cases > 0 and len(audit_rows) % checkpoint_every_cases == 0:
                        maybe_save_checkpoint(
                            checkpoint_path,
                            config=config,
                            next_case_number=next_case_number,
                            generated_rows=generated_rows,
                            metadata_rows=metadata_rows,
                            summary_rows=summary_rows,
                            audit_rows=audit_rows,
                            completed_mondo_ids=completed_mondo_ids,
                            reason=f"case_accepted:{mondo_id}:{generation_index + 1}",
                        )
                    success = True
                    break

                if not success:
                    failed_case_count += 1
                    print(
                        f"[WARN] {mondo_id} failed to generate case {generation_index + 1}: {last_error_text}",
                        flush=True,
                    )

        summary_rows.append(
            {
                "mondo_label": mondo_id,
                "original_total_case_pair_count": original_count,
                "requested_generated_case_count": requested_case_count,
                "generated_case_count": generated_case_count,
                "failed_case_count": failed_case_count,
                "total_after_generation": original_count + generated_case_count,
                "knowledge_hpo_count": int(len(knowledge_hpo_ids)),
                "target_hpo_count": int(target_hpo_count),
                "min_hpo_count": int(min_hpo_count),
                "max_hpo_count": int(max_hpo_count),
                "hard_core_count": int(len(hard_core_hpo_ids)),
                "soft_core_count": int(len(soft_core_hpo_ids)),
                "optional_non_core_count": int(len(optional_hpo_ids)),
                "avg_selected_soft_core_count": (
                    total_selected_soft_core_count / generated_case_count if generated_case_count else 0.0
                ),
                "avg_selected_optional_count": (
                    total_selected_optional_count / generated_case_count if generated_case_count else 0.0
                ),
                "avg_selected_noise_count": (
                    total_selected_noise_count / generated_case_count if generated_case_count else 0.0
                ),
                "avg_final_hpo_count": (
                    generated_hpo_row_count / generated_case_count if generated_case_count else 0.0
                ),
                "avg_attempt_count": (
                    total_attempt_count / requested_case_count if requested_case_count else 0.0
                ),
            }
        )
        print(
            "[INFO] mondo_finished "
            f"index={mondo_index}/{total_target_mondo_count} "
            f"mondo_label={mondo_id} "
            f"generated_case_count={generated_case_count} "
            f"failed_case_count={failed_case_count} "
            f"total_after_generation={original_count + generated_case_count} "
            f"avg_attempt_count={(total_attempt_count / requested_case_count if requested_case_count else 0.0):.2f}",
            flush=True,
        )
        completed_mondo_ids.add(mondo_id)
        if checkpoint_path and (
            len(completed_mondo_ids) % checkpoint_every_mondos == 0
            or len(completed_mondo_ids) == total_target_mondo_count
        ):
            maybe_save_checkpoint(
                checkpoint_path,
                config=config,
                next_case_number=next_case_number,
                generated_rows=generated_rows,
                metadata_rows=metadata_rows,
                summary_rows=summary_rows,
                audit_rows=audit_rows,
                completed_mondo_ids=completed_mondo_ids,
                reason=f"mondo_completed:{mondo_id}",
            )

    generated_df = pd.DataFrame(generated_rows)
    metadata_df = pd.DataFrame(metadata_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["original_total_case_pair_count", "mondo_label"], ascending=[True, True]
    )
    audit_df = pd.DataFrame(audit_rows)
    params_df = pd.DataFrame(
        [
            {"parameter": "support_threshold", "value": config.support_threshold},
            {"parameter": "seed", "value": config.seed},
            {"parameter": "start_case_number", "value": config.start_case_number},
            {"parameter": "core_min_sources", "value": config.core_min_sources},
            {"parameter": "hard_core_min_sources", "value": config.hard_core_min_sources},
            {"parameter": "target_hpo_slack", "value": config.target_hpo_slack},
            {"parameter": "max_noise_hpo", "value": config.max_noise_hpo},
            {"parameter": "noise_support_quantile", "value": config.noise_support_quantile},
            {"parameter": "max_noise_candidates", "value": config.max_noise_candidates},
            {"parameter": "max_generation_attempts", "value": config.max_generation_attempts},
            {"parameter": "checkpoint_every_cases", "value": checkpoint_every_cases},
            {"parameter": "max_workers", "value": max_workers},
            {"parameter": "base_url", "value": config.base_url},
            {"parameter": "model", "value": config.model},
            {
                "parameter": "generation_rule",
                "value": (
                    "Knowledge-only generation; keep all core HPOs; optionally select non-core knowledge HPOs; "
                    "add at most 2 global noise HPOs; exact-deduplicate against total_data."
                ),
            },
        ]
    )
    return generated_df, metadata_df, summary_df, params_df, audit_df


def _generate_cases_parallel(
    target_df: pd.DataFrame,
    real_signatures_by_mondo: dict[str, set[frozenset[str]]],
    hpo_count_stats_by_mondo: dict[str, dict[str, int]],
    knowledge_df: pd.DataFrame,
    noise_pool_df: pd.DataFrame,
    mondo_name_map: dict[str, str],
    hpo_name_map: dict[str, str],
    selector: HeuristicSelector | SiliconFlowSelector,
    config: GenerationConfig,
    checkpoint_path: Path | None,
    checkpoint_every_mondos: int,
    checkpoint_every_cases: int,
    resume_payload: dict[str, Any] | None,
    max_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_target_mondo_count = int(len(target_df))
    total_requested_generated_case_count = int(target_df["target_generated_case_count"].sum()) if not target_df.empty else 0
    print(
        "[INFO] generation_started "
        f"target_mondo_count={total_target_mondo_count} "
        f"requested_generated_case_count={total_requested_generated_case_count}",
        flush=True,
    )

    knowledge_profiles = build_knowledge_profiles(
        knowledge_df,
        config.core_min_sources,
        config.hard_core_min_sources,
    )
    existing_signatures_by_mondo = {
        mondo_id: set(signatures)
        for mondo_id, signatures in real_signatures_by_mondo.items()
    }
    global_default_target_hpo_count = int(
        np.median(
            [
                stats["median_hpo_count"]
                for stats in hpo_count_stats_by_mondo.values()
                if int(stats.get("median_hpo_count", 0)) > 0
            ]
            or [10]
        )
    )

    next_case_number = int(config.start_case_number)
    generated_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    completed_mondo_ids: set[str] = set()
    checkpoint_every_mondos = max(1, int(checkpoint_every_mondos))
    checkpoint_every_cases = max(0, int(checkpoint_every_cases))
    max_workers = max(1, int(max_workers))

    if resume_payload:
        for field_name, expected_value in (
            ("support_threshold", config.support_threshold),
            ("seed", config.seed),
            ("start_case_number", config.start_case_number),
            ("base_url", config.base_url),
            ("model", config.model),
            ("core_min_sources", config.core_min_sources),
            ("hard_core_min_sources", config.hard_core_min_sources),
            ("target_hpo_slack", config.target_hpo_slack),
        ):
            actual_value = resume_payload.get(field_name)
            if actual_value is not None and str(actual_value) != str(expected_value):
                raise ValueError(
                    f"Checkpoint config mismatch for {field_name}: expected={expected_value}, got={actual_value}"
                )

        next_case_number = int(resume_payload.get("next_case_number", next_case_number))
        generated_rows = list(resume_payload.get("generated_rows", []))
        metadata_rows = list(resume_payload.get("metadata_rows", []))
        summary_rows = list(resume_payload.get("summary_rows", []))
        audit_rows = list(resume_payload.get("audit_rows", []))
        completed_mondo_ids = set(map(str, resume_payload.get("completed_mondo_ids", [])))

        for row in metadata_rows:
            mondo_id = str(row.get("mondo_label", "")).strip()
            final_hpo_ids = frozenset(_deserialize_hpo_ids(row.get("final_hpo_ids", "")))
            if mondo_id and final_hpo_ids:
                existing_signatures_by_mondo.setdefault(mondo_id, set()).add(final_hpo_ids)

        print(
            "[INFO] checkpoint_loaded "
            f"path={checkpoint_path} "
            f"completed_mondo_count={len(completed_mondo_ids)} "
            f"generated_case_count={len(metadata_rows)} "
            f"next_case_number={next_case_number}",
            flush=True,
        )

    pending_targets: list[tuple[int, Any]] = []
    for mondo_index, target_row in enumerate(target_df.itertuples(index=False), start=1):
        mondo_id = str(target_row.mondo_label)
        if mondo_id not in completed_mondo_ids:
            pending_targets.append((mondo_index, target_row))

    effective_workers = min(max_workers, len(pending_targets)) if pending_targets else 1
    print(
        "[INFO] parallel_config "
        f"requested_max_workers={max_workers} "
        f"effective_workers={effective_workers} "
        f"pending_mondo_count={len(pending_targets)}",
        flush=True,
    )

    state_lock = threading.Lock()

    def record_failed_case(
        *,
        mondo_id: str,
        generation_index: int,
        attempt_count: int,
        selector_mode: str,
        error_text: str,
        raw_response: str,
        duplicate_retry_count: int,
        used_fallback: bool,
        reason: str,
    ) -> None:
        nonlocal next_case_number
        with state_lock:
            audit_rows.append(
                {
                    "case_id": "",
                    "mondo_label": mondo_id,
                    "generation_index": generation_index,
                    "attempt_count": attempt_count,
                    "selector_mode": selector_mode,
                    "status": "failed",
                    "error": error_text,
                    "raw_response": raw_response,
                    "selected_soft_core_hpo_ids": "",
                    "selected_optional_hpo_ids": "",
                    "selected_noise_hpo_ids": "",
                    "duplicate_retry_count": duplicate_retry_count,
                    "used_fallback": used_fallback,
                }
            )
            if checkpoint_every_cases > 0 and len(audit_rows) % checkpoint_every_cases == 0:
                maybe_save_checkpoint(
                    checkpoint_path,
                    config=config,
                    next_case_number=next_case_number,
                    generated_rows=generated_rows,
                    metadata_rows=metadata_rows,
                    summary_rows=summary_rows,
                    audit_rows=audit_rows,
                    completed_mondo_ids=completed_mondo_ids,
                    reason=reason,
                )

    def record_accepted_case(
        *,
        mondo_id: str,
        generation_index: int,
        original_count: int,
        target_hpo_count: int,
        min_hpo_count: int,
        max_hpo_count: int,
        hard_core_hpo_ids: tuple[str, ...],
        soft_core_hpo_ids: tuple[str, ...],
        optional_hpo_ids: tuple[str, ...],
        noise_candidate_ids: list[str],
        final_hpo_ids: list[str],
        selected_soft_core_hpo_ids: tuple[str, ...],
        selected_optional_hpo_ids: tuple[str, ...],
        selected_noise_hpo_ids: tuple[str, ...],
        attempt_index: int,
        decision: SelectorDecision,
        duplicate_retry_count: int,
        signature: frozenset[str],
        existing_signatures: set[frozenset[str]],
    ) -> None:
        nonlocal next_case_number
        existing_signatures.add(signature)
        with state_lock:
            generated_case_id = f"synthetic_case_{next_case_number}"
            next_case_number += 1
            for hpo_id in final_hpo_ids:
                generated_rows.append(
                    {
                        "case_id": generated_case_id,
                        "mondo_label": mondo_id,
                        "hpo_id": hpo_id,
                        "case_source": "synthetic",
                    }
                )

            metadata_rows.append(
                {
                    "case_id": generated_case_id,
                    "mondo_label": mondo_id,
                    "case_source": "synthetic",
                    "original_total_case_pair_count": original_count,
                    "requested_target_total_case_count": config.support_threshold,
                    "target_hpo_count": int(target_hpo_count),
                    "min_hpo_count": int(min_hpo_count),
                    "max_hpo_count": int(max_hpo_count),
                    "hard_core_count": int(len(hard_core_hpo_ids)),
                    "soft_core_count": int(len(soft_core_hpo_ids)),
                    "optional_non_core_count": int(len(optional_hpo_ids)),
                    "noise_candidate_count": int(len(noise_candidate_ids)),
                    "final_hpo_count": int(len(final_hpo_ids)),
                    "selected_soft_core_count": int(len(selected_soft_core_hpo_ids)),
                    "selected_optional_count": int(len(selected_optional_hpo_ids)),
                    "selected_noise_count": int(len(selected_noise_hpo_ids)),
                    "hard_core_hpo_ids": _serialize_hpo_ids(hard_core_hpo_ids),
                    "soft_core_hpo_ids": _serialize_hpo_ids(soft_core_hpo_ids),
                    "optional_non_core_hpo_ids": _serialize_hpo_ids(optional_hpo_ids),
                    "noise_candidate_hpo_ids": _serialize_hpo_ids(noise_candidate_ids),
                    "selected_soft_core_hpo_ids": _serialize_hpo_ids(selected_soft_core_hpo_ids),
                    "selected_optional_hpo_ids": _serialize_hpo_ids(selected_optional_hpo_ids),
                    "selected_noise_hpo_ids": _serialize_hpo_ids(selected_noise_hpo_ids),
                    "final_hpo_ids": _serialize_hpo_ids(final_hpo_ids),
                    "duplicate_retry_count": int(duplicate_retry_count),
                    "quality_filter_reason": "keep",
                }
            )
            audit_rows.append(
                {
                    "case_id": generated_case_id,
                    "mondo_label": mondo_id,
                    "generation_index": generation_index,
                    "attempt_count": attempt_index,
                    "selector_mode": decision.selector_mode,
                    "status": "accepted",
                    "error": "",
                    "raw_response": decision.raw_response,
                    "selected_soft_core_hpo_ids": _serialize_hpo_ids(selected_soft_core_hpo_ids),
                    "selected_optional_hpo_ids": _serialize_hpo_ids(selected_optional_hpo_ids),
                    "selected_noise_hpo_ids": _serialize_hpo_ids(selected_noise_hpo_ids),
                    "duplicate_retry_count": int(duplicate_retry_count),
                    "used_fallback": bool(decision.used_fallback),
                }
            )
            if checkpoint_every_cases > 0 and len(audit_rows) % checkpoint_every_cases == 0:
                maybe_save_checkpoint(
                    checkpoint_path,
                    config=config,
                    next_case_number=next_case_number,
                    generated_rows=generated_rows,
                    metadata_rows=metadata_rows,
                    summary_rows=summary_rows,
                    audit_rows=audit_rows,
                    completed_mondo_ids=completed_mondo_ids,
                    reason=f"case_accepted:{mondo_id}:{generation_index}",
                )

    def complete_mondo(mondo_id: str, summary_row: dict[str, Any]) -> None:
        nonlocal next_case_number
        with state_lock:
            upsert_summary_row(summary_rows, summary_row)
            completed_mondo_ids.add(mondo_id)
            if checkpoint_path and (
                len(completed_mondo_ids) % checkpoint_every_mondos == 0
                or len(completed_mondo_ids) == total_target_mondo_count
            ):
                maybe_save_checkpoint(
                    checkpoint_path,
                    config=config,
                    next_case_number=next_case_number,
                    generated_rows=generated_rows,
                    metadata_rows=metadata_rows,
                    summary_rows=summary_rows,
                    audit_rows=audit_rows,
                    completed_mondo_ids=completed_mondo_ids,
                    reason=f"mondo_completed:{mondo_id}",
                )

    def process_single_mondo(mondo_index: int, target_row: Any) -> None:
        mondo_id = str(target_row.mondo_label)
        mondo_selector = build_selector_for_mondo(selector, config, mondo_id)
        rng = np.random.default_rng(seed_for_text(config.seed, f"noise:{mondo_id}"))
        original_count = int(target_row.total_case_pair_count)
        requested_case_count = int(target_row.target_generated_case_count)
        disease_name = mondo_name_map.get(mondo_id, mondo_id)
        profile = knowledge_profiles.get(
            mondo_id,
            {
                "knowledge_hpo_ids": (),
                "hard_core_hpo_ids": (),
                "soft_core_hpo_ids": (),
                "optional_non_core_hpo_ids": (),
            },
        )

        knowledge_hpo_ids = tuple(profile["knowledge_hpo_ids"])
        hard_core_hpo_ids = tuple(profile["hard_core_hpo_ids"])
        soft_core_hpo_ids = tuple(profile["soft_core_hpo_ids"])
        optional_hpo_ids = tuple(profile["optional_non_core_hpo_ids"])
        target_hpo_count, min_hpo_count, max_hpo_count = resolve_hpo_count_range(
            mondo_id,
            hard_core_hpo_ids,
            hpo_count_stats_by_mondo,
            global_default_target_hpo_count,
            config.target_hpo_slack,
        )
        generation_plan = build_generation_plan(
            requested_case_count=requested_case_count,
            hard_core_hpo_ids=hard_core_hpo_ids,
            soft_core_hpo_ids=soft_core_hpo_ids,
            optional_hpo_ids=optional_hpo_ids,
            target_hpo_count=target_hpo_count,
            min_hpo_count=min_hpo_count,
            max_hpo_count=max_hpo_count,
            config=config,
        )
        effective_requested_case_count = int(generation_plan["effective_requested_case_count"])
        effective_target_hpo_count = int(generation_plan["effective_target_hpo_count"])
        effective_max_noise_hpo = int(generation_plan["effective_max_noise_hpo"])
        effective_soft_core_hpo_ids = tuple(generation_plan["effective_soft_core_hpo_ids"])
        effective_optional_hpo_ids = tuple(generation_plan["effective_optional_hpo_ids"])
        remaining_slot_budget = int(generation_plan["remaining_slot_budget"])
        estimated_case_capacity = int(generation_plan["estimated_case_capacity"])
        deterministic_case_only = bool(generation_plan["deterministic_case_only"])
        existing_signatures = existing_signatures_by_mondo.setdefault(
            mondo_id,
            set(real_signatures_by_mondo.get(mondo_id, set())),
        )
        progress = summarize_existing_mondo_progress(mondo_id, metadata_rows, audit_rows)
        generated_case_count = int(progress["generated_case_count"])
        failed_case_count = int(progress["failed_case_count"])
        generated_hpo_row_count = int(progress["generated_hpo_row_count"])
        total_selected_soft_core_count = int(progress["total_selected_soft_core_count"])
        total_selected_optional_count = int(progress["total_selected_optional_count"])
        total_selected_noise_count = int(progress["total_selected_noise_count"])
        total_attempt_count = int(progress["total_attempt_count"])
        start_generation_index = int(progress["processed_generation_count"])

        print(
            "[INFO] mondo_started "
            f"index={mondo_index}/{total_target_mondo_count} "
            f"mondo_label={mondo_id} "
            f"original_count={original_count} "
            f"need_generate={requested_case_count} "
            f"effective_need_generate={effective_requested_case_count} "
            f"knowledge_hpo_count={len(knowledge_hpo_ids)} "
            f"hard_core_hpo_count={len(hard_core_hpo_ids)} "
            f"soft_core_hpo_count={len(soft_core_hpo_ids)} "
            f"target_hpo_count={effective_target_hpo_count} "
            f"remaining_slot_budget={remaining_slot_budget} "
            f"effective_max_noise_hpo={effective_max_noise_hpo}",
            flush=True,
        )

        if not knowledge_hpo_ids:
            error_text = f"{mondo_id} does not have any knowledge HPO annotations."
            for generation_index in range(start_generation_index, effective_requested_case_count):
                record_failed_case(
                    mondo_id=mondo_id,
                    generation_index=generation_index + 1,
                    attempt_count=0,
                    selector_mode=getattr(mondo_selector, "__class__", type(mondo_selector)).__name__,
                    error_text=error_text,
                    raw_response="",
                    duplicate_retry_count=0,
                    used_fallback=False,
                    reason=f"case_failed:{mondo_id}:{generation_index + 1}",
                )
                failed_case_count += 1
        else:
            for generation_index in range(start_generation_index, effective_requested_case_count):
                success = False
                last_error_text = ""
                last_raw_response = ""
                last_selector_mode = "deterministic" if deterministic_case_only else getattr(
                    mondo_selector, "__class__", type(mondo_selector)
                ).__name__
                last_used_fallback = False
                duplicate_retry_count = 0

                max_attempts_for_case = 1 if deterministic_case_only else config.max_generation_attempts
                for attempt_index in range(1, max_attempts_for_case + 1):
                    total_attempt_count += 1
                    if deterministic_case_only:
                        noise_candidate_ids = []
                    else:
                        noise_candidate_ids = choose_noise_candidates(
                            noise_pool_df=noise_pool_df,
                            excluded_hpo_ids=set(knowledge_hpo_ids),
                            rng=rng,
                            config=config,
                        )
                    request = build_selector_request(
                        mondo_id=mondo_id,
                        disease_name=disease_name,
                        hard_core_hpo_ids=list(hard_core_hpo_ids),
                        soft_core_hpo_ids=list(effective_soft_core_hpo_ids),
                        optional_non_core_hpo_ids=list(effective_optional_hpo_ids),
                        noise_candidate_ids=noise_candidate_ids,
                        target_hpo_count=effective_target_hpo_count,
                        min_hpo_count=min_hpo_count,
                        max_hpo_count=max_hpo_count,
                        config=config,
                        max_noise_hpo=effective_max_noise_hpo,
                    )
                    try:
                        if deterministic_case_only:
                            decision = SelectorDecision(
                                selected_soft_core_hpo_ids=(),
                                selected_optional_hpo_ids=(),
                                selected_noise_hpo_ids=(),
                                raw_response='{"selected_soft_core_hpo_ids":[],"selected_optional_hpo_ids":[],"selected_noise_hpo_ids":[],"rationale":"deterministic_case_only"}',
                                selector_mode="deterministic",
                                used_fallback=False,
                            )
                        else:
                            decision = mondo_selector.select(request, hpo_name_map)
                        last_raw_response = decision.raw_response
                        last_selector_mode = decision.selector_mode
                        last_used_fallback = bool(decision.used_fallback)
                        (
                            selected_soft_core_hpo_ids,
                            selected_optional_hpo_ids,
                            selected_noise_hpo_ids,
                        ) = validate_selector_decision(request, decision)
                        final_hpo_ids = build_case_from_decision(
                            request,
                            selected_soft_core_hpo_ids,
                            selected_optional_hpo_ids,
                            selected_noise_hpo_ids,
                        )
                        signature = validate_generated_case(
                            mondo_id=mondo_id,
                            final_hpo_ids=final_hpo_ids,
                            required_hard_core_hpo_ids=hard_core_hpo_ids,
                            min_hpo_count=min_hpo_count,
                            max_hpo_count=max_hpo_count,
                            existing_signatures=existing_signatures,
                        )
                    except Exception as exc:  # noqa: BLE001
                        last_error_text = str(exc)
                        if "duplicate case signature" in last_error_text:
                            duplicate_retry_count += 1
                        if attempt_index == config.max_generation_attempts:
                            record_failed_case(
                                mondo_id=mondo_id,
                                generation_index=generation_index + 1,
                                attempt_count=attempt_index,
                                selector_mode=last_selector_mode,
                                error_text=last_error_text,
                                raw_response=last_raw_response,
                                duplicate_retry_count=duplicate_retry_count,
                                used_fallback=last_used_fallback,
                                reason=f"case_failed:{mondo_id}:{generation_index + 1}",
                            )
                        continue

                    record_accepted_case(
                        mondo_id=mondo_id,
                        generation_index=generation_index + 1,
                        original_count=original_count,
                        target_hpo_count=effective_target_hpo_count,
                        min_hpo_count=min_hpo_count,
                        max_hpo_count=max_hpo_count,
                        hard_core_hpo_ids=hard_core_hpo_ids,
                        soft_core_hpo_ids=effective_soft_core_hpo_ids,
                        optional_hpo_ids=effective_optional_hpo_ids,
                        noise_candidate_ids=noise_candidate_ids,
                        final_hpo_ids=final_hpo_ids,
                        selected_soft_core_hpo_ids=selected_soft_core_hpo_ids,
                        selected_optional_hpo_ids=selected_optional_hpo_ids,
                        selected_noise_hpo_ids=selected_noise_hpo_ids,
                        attempt_index=attempt_index,
                        decision=decision,
                        duplicate_retry_count=duplicate_retry_count,
                        signature=signature,
                        existing_signatures=existing_signatures,
                    )
                    generated_case_count += 1
                    generated_hpo_row_count += len(final_hpo_ids)
                    total_selected_soft_core_count += len(selected_soft_core_hpo_ids)
                    total_selected_optional_count += len(selected_optional_hpo_ids)
                    total_selected_noise_count += len(selected_noise_hpo_ids)
                    success = True
                    break

                if not success:
                    failed_case_count += 1
                    print(
                        f"[WARN] {mondo_id} failed to generate case {generation_index + 1}: {last_error_text}",
                        flush=True,
                    )

        summary_row = {
            "mondo_label": mondo_id,
            "original_total_case_pair_count": original_count,
            "requested_generated_case_count": requested_case_count,
            "effective_requested_generated_case_count": effective_requested_case_count,
            "generated_case_count": generated_case_count,
            "failed_case_count": failed_case_count,
            "total_after_generation": original_count + generated_case_count,
            "knowledge_hpo_count": int(len(knowledge_hpo_ids)),
            "target_hpo_count": int(effective_target_hpo_count),
            "min_hpo_count": int(min_hpo_count),
            "max_hpo_count": int(max_hpo_count),
            "hard_core_count": int(len(hard_core_hpo_ids)),
            "soft_core_count": int(len(soft_core_hpo_ids)),
            "optional_non_core_count": int(len(optional_hpo_ids)),
            "remaining_slot_budget": int(remaining_slot_budget),
            "effective_max_noise_hpo": int(effective_max_noise_hpo),
            "estimated_case_capacity": int(estimated_case_capacity),
            "deterministic_case_only": bool(deterministic_case_only),
            "avg_selected_soft_core_count": (
                total_selected_soft_core_count / generated_case_count if generated_case_count else 0.0
            ),
            "avg_selected_optional_count": (
                total_selected_optional_count / generated_case_count if generated_case_count else 0.0
            ),
            "avg_selected_noise_count": (
                total_selected_noise_count / generated_case_count if generated_case_count else 0.0
            ),
            "avg_final_hpo_count": (
                generated_hpo_row_count / generated_case_count if generated_case_count else 0.0
            ),
            "avg_attempt_count": (
                total_attempt_count / effective_requested_case_count if effective_requested_case_count else 0.0
            ),
        }
        complete_mondo(mondo_id, summary_row)
        print(
            "[INFO] mondo_finished "
            f"index={mondo_index}/{total_target_mondo_count} "
            f"mondo_label={mondo_id} "
            f"generated_case_count={generated_case_count} "
            f"failed_case_count={failed_case_count} "
            f"total_after_generation={original_count + generated_case_count} "
            f"avg_attempt_count={(total_attempt_count / effective_requested_case_count if effective_requested_case_count else 0.0):.2f}",
            flush=True,
        )

    if effective_workers <= 1:
        for mondo_index, target_row in pending_targets:
            process_single_mondo(mondo_index, target_row)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(process_single_mondo, mondo_index, target_row)
                for mondo_index, target_row in pending_targets
            ]
            for future in as_completed(futures):
                future.result()

    generated_df = pd.DataFrame(generated_rows)
    metadata_df = pd.DataFrame(metadata_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["original_total_case_pair_count", "mondo_label"], ascending=[True, True]
    )
    audit_df = pd.DataFrame(audit_rows)
    params_df = pd.DataFrame(
        [
            {"parameter": "support_threshold", "value": config.support_threshold},
            {"parameter": "seed", "value": config.seed},
            {"parameter": "start_case_number", "value": config.start_case_number},
            {"parameter": "core_min_sources", "value": config.core_min_sources},
            {"parameter": "hard_core_min_sources", "value": config.hard_core_min_sources},
            {"parameter": "target_hpo_slack", "value": config.target_hpo_slack},
            {"parameter": "max_noise_hpo", "value": config.max_noise_hpo},
            {"parameter": "noise_support_quantile", "value": config.noise_support_quantile},
            {"parameter": "max_noise_candidates", "value": config.max_noise_candidates},
            {"parameter": "max_generation_attempts", "value": config.max_generation_attempts},
            {"parameter": "checkpoint_every_cases", "value": checkpoint_every_cases},
            {"parameter": "max_workers", "value": max_workers},
            {"parameter": "base_url", "value": config.base_url},
            {"parameter": "model", "value": config.model},
            {
                "parameter": "generation_rule",
                "value": (
                    "Knowledge-only generation; keep all core HPOs; optionally select non-core knowledge HPOs; "
                    "add at most 2 global noise HPOs; exact-deduplicate against total_data."
                ),
            },
        ]
    )
    return generated_df, metadata_df, summary_df, params_df, audit_df


def build_underfilled_mondo_report(summary_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "mondo_label",
                "requested_generated_case_count",
                "effective_requested_generated_case_count",
                "generated_case_count",
                "failed_case_count",
                "missing_vs_requested_count",
                "missing_vs_effective_count",
                "capacity_capped",
                "deterministic_case_only",
                "estimated_case_capacity",
                "remaining_slot_budget",
                "effective_max_noise_hpo",
                "top_failed_reason",
                "failed_reason_summary",
                "underfilled_reason_summary",
            ]
        )

    failed_reason_map: dict[str, dict[str, int]] = {}
    if not audit_df.empty and {"mondo_label", "status", "error"}.issubset(audit_df.columns):
        failed_df = audit_df.loc[audit_df["status"].astype(str) == "failed", ["mondo_label", "error"]].copy()
        failed_df["reason_key"] = failed_df["error"].map(_normalize_error_reason)
        for mondo_id, group_df in failed_df.groupby("mondo_label", sort=False):
            reason_counts = (
                group_df["reason_key"]
                .fillna("")
                .astype(str)
                .loc[lambda s: s != ""]
                .value_counts()
                .to_dict()
            )
            failed_reason_map[str(mondo_id)] = {str(key): int(value) for key, value in reason_counts.items()}

    report_rows: list[dict[str, Any]] = []
    for row in summary_df.itertuples(index=False):
        mondo_id = str(getattr(row, "mondo_label"))
        requested_generated_case_count = int(getattr(row, "requested_generated_case_count", 0))
        effective_requested_generated_case_count = int(
            getattr(row, "effective_requested_generated_case_count", requested_generated_case_count)
        )
        generated_case_count = int(getattr(row, "generated_case_count", 0))
        failed_case_count = int(getattr(row, "failed_case_count", 0))
        missing_vs_requested_count = max(0, requested_generated_case_count - generated_case_count)
        missing_vs_effective_count = max(0, effective_requested_generated_case_count - generated_case_count)
        capacity_capped = effective_requested_generated_case_count < requested_generated_case_count
        if missing_vs_requested_count <= 0 and missing_vs_effective_count <= 0 and not capacity_capped:
            continue

        reason_counts = failed_reason_map.get(mondo_id, {})
        sorted_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
        failed_reason_summary = "; ".join(f"{reason}:{count}" for reason, count in sorted_reasons)
        top_failed_reason = sorted_reasons[0][0] if sorted_reasons else ""

        underfilled_reasons: list[str] = []
        if capacity_capped:
            underfilled_reasons.append("adaptive_capacity_cap")
        if failed_case_count > 0:
            underfilled_reasons.append("generation_failures")
        if not underfilled_reasons:
            underfilled_reasons.append("unknown")

        report_rows.append(
            {
                "mondo_label": mondo_id,
                "requested_generated_case_count": requested_generated_case_count,
                "effective_requested_generated_case_count": effective_requested_generated_case_count,
                "generated_case_count": generated_case_count,
                "failed_case_count": failed_case_count,
                "missing_vs_requested_count": missing_vs_requested_count,
                "missing_vs_effective_count": missing_vs_effective_count,
                "capacity_capped": bool(capacity_capped),
                "deterministic_case_only": bool(getattr(row, "deterministic_case_only", False)),
                "estimated_case_capacity": int(getattr(row, "estimated_case_capacity", 0)),
                "remaining_slot_budget": int(getattr(row, "remaining_slot_budget", 0)),
                "effective_max_noise_hpo": int(getattr(row, "effective_max_noise_hpo", 0)),
                "top_failed_reason": top_failed_reason,
                "failed_reason_summary": failed_reason_summary,
                "underfilled_reason_summary": "|".join(underfilled_reasons),
            }
        )

    report_df = pd.DataFrame(report_rows)
    if report_df.empty:
        return pd.DataFrame(
            columns=[
                "mondo_label",
                "requested_generated_case_count",
                "effective_requested_generated_case_count",
                "generated_case_count",
                "failed_case_count",
                "missing_vs_requested_count",
                "missing_vs_effective_count",
                "capacity_capped",
                "deterministic_case_only",
                "estimated_case_capacity",
                "remaining_slot_budget",
                "effective_max_noise_hpo",
                "top_failed_reason",
                "failed_reason_summary",
                "underfilled_reason_summary",
            ]
        )
    return report_df.sort_values(
        ["missing_vs_requested_count", "missing_vs_effective_count", "mondo_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def save_outputs(
    output_path: Path,
    generated_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    params_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    unmet_df = build_underfilled_mondo_report(summary_df, audit_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        generated_df.to_excel(writer, sheet_name="generated_cases", index=False)
        metadata_df.to_excel(writer, sheet_name="case_metadata", index=False)
        summary_df.to_excel(writer, sheet_name="generation_summary", index=False)
        params_df.to_excel(writer, sheet_name="run_params", index=False)
        audit_df.to_excel(writer, sheet_name="llm_audit", index=False)
        unmet_df.to_excel(writer, sheet_name="underfilled_mondo_summary", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate low-count MONDO synthetic cases from curated disease knowledge only."
    )
    parser.add_argument("--stats_path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--total_data_dir", type=Path, default=DEFAULT_TOTAL_DATA_DIR)
    parser.add_argument("--knowledge_dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--mondo_json_path", type=Path, default=DEFAULT_MONDO_JSON_PATH)
    parser.add_argument("--hpo_json_path", type=Path, default=DEFAULT_HPO_JSON_PATH)
    parser.add_argument("--hpo_index_path", type=Path, default=DEFAULT_HPO_INDEX_PATH)
    parser.add_argument("--disease_index_path", type=Path, default=DEFAULT_DISEASE_INDEX_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--checkpoint_every_mondos", type=int, default=10)
    parser.add_argument("--checkpoint_every_cases", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--reset_checkpoint", action="store_true")
    parser.add_argument("--keep_checkpoint", action="store_true")
    parser.add_argument("--max_target_mondos", type=int, default=0)
    parser.add_argument("--selector_mode", choices=("siliconflow", "heuristic"), default="siliconflow")
    parser.add_argument("--allow_heuristic_fallback", action="store_true")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY)
    parser.add_argument("--api_key_env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--support_threshold", type=int, default=GenerationConfig.support_threshold)
    parser.add_argument("--seed", type=int, default=GenerationConfig.seed)
    parser.add_argument("--start_case_number", type=int, default=GenerationConfig.start_case_number)
    parser.add_argument("--core_min_sources", type=int, default=GenerationConfig.core_min_sources)
    parser.add_argument("--hard_core_min_sources", type=int, default=GenerationConfig.hard_core_min_sources)
    parser.add_argument("--target_hpo_slack", type=int, default=GenerationConfig.target_hpo_slack)
    parser.add_argument("--max_noise_hpo", type=int, default=GenerationConfig.max_noise_hpo)
    parser.add_argument("--noise_support_quantile", type=float, default=GenerationConfig.noise_support_quantile)
    parser.add_argument("--max_noise_candidates", type=int, default=GenerationConfig.max_noise_candidates)
    parser.add_argument("--max_generation_attempts", type=int, default=GenerationConfig.max_generation_attempts)
    parser.add_argument("--llm_timeout_sec", type=int, default=GenerationConfig.llm_timeout_sec)
    parser.add_argument("--llm_max_retries", type=int, default=GenerationConfig.llm_max_retries)
    parser.add_argument("--llm_temperature", type=float, default=GenerationConfig.llm_temperature)
    parser.add_argument("--llm_max_tokens", type=int, default=GenerationConfig.llm_max_tokens)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        support_threshold=int(args.support_threshold),
        seed=int(args.seed),
        start_case_number=int(args.start_case_number),
        core_min_sources=int(args.core_min_sources),
        hard_core_min_sources=int(args.hard_core_min_sources),
        target_hpo_slack=int(args.target_hpo_slack),
        max_noise_hpo=int(args.max_noise_hpo),
        noise_support_quantile=float(args.noise_support_quantile),
        max_noise_candidates=int(args.max_noise_candidates),
        llm_timeout_sec=int(args.llm_timeout_sec),
        llm_max_retries=int(args.llm_max_retries),
        llm_temperature=float(args.llm_temperature),
        llm_max_tokens=int(args.llm_max_tokens),
        max_generation_attempts=int(args.max_generation_attempts),
        base_url=str(args.base_url),
        model=str(args.model),
    )

    hpo_vocab = _load_vocab_ids(args.hpo_index_path, id_col="hpo_id")
    disease_vocab = _load_vocab_ids(args.disease_index_path, id_col="mondo_id")
    target_df = load_target_mondo_counts(args.stats_path, config.support_threshold, disease_vocab)
    if int(args.max_target_mondos) > 0:
        target_df = target_df.head(int(args.max_target_mondos)).copy()
    mondo_name_map = load_ontology_name_map(args.mondo_json_path, prefix="MONDO")
    hpo_name_map = load_ontology_name_map(args.hpo_json_path, prefix="HP")
    real_signatures_by_mondo, global_hpo_support, hpo_count_stats_by_mondo = load_real_signatures_and_support(
        args.total_data_dir,
        hpo_vocab,
        disease_vocab,
    )
    knowledge_df = load_knowledge_annotations(args.knowledge_dir, hpo_vocab, disease_vocab)
    noise_pool_df = build_noise_pool(global_hpo_support, hpo_name_map, config.noise_support_quantile)
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else default_checkpoint_path(args.output_path)
    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[INFO] checkpoint_removed path={checkpoint_path}", flush=True)
    resume_payload = None if args.reset_checkpoint else load_checkpoint(checkpoint_path)
    print(
        "[INFO] data_loaded "
        f"target_mondo_count={len(target_df)} "
        f"knowledge_row_count={len(knowledge_df)} "
        f"real_signature_mondo_count={len(real_signatures_by_mondo)} "
        f"hpo_count_stats_mondo_count={len(hpo_count_stats_by_mondo)} "
        f"noise_pool_size={len(noise_pool_df)} "
        f"checkpoint_path={checkpoint_path}",
        flush=True,
    )

    heuristic_selector = HeuristicSelector(seed=config.seed)
    if args.selector_mode == "heuristic":
        selector: HeuristicSelector | SiliconFlowSelector = heuristic_selector
    else:
        api_key = str(args.api_key).strip()
        if not api_key or api_key == API_KEY_PLACEHOLDER:
            api_key = os.getenv(args.api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                "Missing SiliconFlow API key. "
                f"Please fill DEFAULT_API_KEY in this file or set {args.api_key_env} before running."
            )
        selector = SiliconFlowSelector(
            api_key=api_key,
            config=config,
            fallback_selector=heuristic_selector if args.allow_heuristic_fallback else None,
        )

    generated_df, metadata_df, summary_df, params_df, audit_df = generate_cases(
        target_df=target_df,
        real_signatures_by_mondo=real_signatures_by_mondo,
        hpo_count_stats_by_mondo=hpo_count_stats_by_mondo,
        knowledge_df=knowledge_df,
        noise_pool_df=noise_pool_df,
        mondo_name_map=mondo_name_map,
        hpo_name_map=hpo_name_map,
        selector=selector,
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_every_mondos=int(args.checkpoint_every_mondos),
        checkpoint_every_cases=int(args.checkpoint_every_cases),
        resume_payload=resume_payload,
        max_workers=int(args.max_workers),
    )
    save_outputs(
        output_path=args.output_path,
        generated_df=generated_df,
        metadata_df=metadata_df,
        summary_df=summary_df,
        params_df=params_df,
        audit_df=audit_df,
    )
    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink()
        print(f"[INFO] checkpoint_removed path={checkpoint_path}", flush=True)

    print(f"output_path={args.output_path}", flush=True)
    print(f"target_mondo_count={len(target_df)}", flush=True)
    print(f"generated_case_count={metadata_df['case_id'].nunique() if not metadata_df.empty else 0}", flush=True)
    print(f"generated_row_count={len(generated_df)}", flush=True)
    print(summary_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
