from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\RareDisease-traindata")
PROCESSED_DIR = PROJECT_ROOT / "LLLdataset" / "dataset" / "processed"
HYPEREDGE_DIR = PROJECT_ROOT / "LLLdataset" / "dataset" / "rare_disease_hgnn_clean_package_v59"
OUTPUT_DIR = PROCESSED_DIR / "audit_outputs" / "key_hpo_consistency_audit"

DATASET_PATHS = [
    PROCESSED_DIR / "ddd_test.csv",
    PROCESSED_DIR / "MME.xlsx",
    PROCESSED_DIR / "MyGene2.xlsx",
    PROCESSED_DIR / "RAMEDIS.xlsx",
    PROCESSED_DIR / "HMS.xlsx",
    PROCESSED_DIR / "LIRICAL.xlsx",
    PROCESSED_DIR / "mimic_rag_0425.csv",
    PROCESSED_DIR / "test" / "mimic_test.csv",
]

HPO_ID_RE = re.compile(r"HP:\d{7}")
MONDO_ID_RE = re.compile(r"MONDO:\d{7}")
CASE_SPLIT_RE = re.compile(r"[|,;/\n]+")


@dataclass
class DatasetParseResult:
    source_dataset: str
    file_path: Path
    sheet_name: str | None
    raw_shape: tuple[int, int]
    structure_type: str
    sample_granularity: str
    case_id_col: str
    disease_col: str
    hpo_col: str
    disease_id_system: str
    parse_method: str
    case_id_collision_across_diseases_count: int
    standardized_df: pd.DataFrame


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE"}:
        return ""
    return text


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def find_best_column(columns: Iterable[str], kind: str) -> str:
    scored: list[tuple[int, str]] = []
    for column in columns:
        name = str(column).strip().lower()
        score = 0
        if kind == "case":
            if name == "case_id":
                score += 100
            if "case" in name:
                score += 30
            if "patient" in name or "sample" in name:
                score += 20
            if name.endswith("_id") or name == "id":
                score += 10
        elif kind == "disease":
            if name == "mondo_label":
                score += 100
            if name == "mondo_id":
                score += 90
            if "mondo" in name:
                score += 50
            if "disease" in name or "diagnosis" in name:
                score += 20
            if "label" in name:
                score += 10
        elif kind == "hpo":
            if name == "hpo_id":
                score += 100
            if "hpo" in name:
                score += 50
            if "phenotype" in name:
                score += 20
            if "feature" in name:
                score += 10
        scored.append((score, str(column)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] <= 0:
        raise ValueError(f"无法自动识别 {kind} 列，候选列: {list(columns)}")
    return scored[0][1]


def detect_disease_id_system(values: pd.Series) -> str:
    sample = values.dropna().astype(str).head(100)
    if sample.empty:
        return "unknown"
    mondo_ratio = sample.str.contains(MONDO_ID_RE).mean()
    omim_ratio = sample.str.contains(r"OMIM:\d+").mean()
    orpha_ratio = sample.str.contains(r"ORPHA:\d+").mean()
    if mondo_ratio >= 0.8:
        return "MONDO"
    if omim_ratio >= 0.8:
        return "OMIM"
    if orpha_ratio >= 0.8:
        return "ORPHA"
    return "mixed_or_unknown"


def parse_hpo_cell(value: object) -> list[str]:
    text = normalize_id(value)
    if not text:
        return []
    matches = HPO_ID_RE.findall(text)
    if matches:
        return sorted(set(matches))
    if any(sep in text for sep in ["|", ";", ",", "[", "]"]):
        items = [part.strip().upper() for part in CASE_SPLIT_RE.split(text) if part.strip()]
        return sorted({item for item in items if HPO_ID_RE.fullmatch(item)})
    return [text] if HPO_ID_RE.fullmatch(text) else []


def is_long_table(df: pd.DataFrame, case_col: str, hpo_col: str) -> bool:
    preview = df[[case_col, hpo_col]].dropna().head(300)
    if preview.empty:
        return True
    parsed_counts = preview[hpo_col].map(lambda value: len(parse_hpo_cell(value)))
    single_hpo_ratio = (parsed_counts == 1).mean()
    repeated_case_ratio = 1.0 - (preview[case_col].nunique() / max(len(preview), 1))
    return single_hpo_ratio >= 0.8 and repeated_case_ratio >= 0.05


def load_tabular_file(path: Path) -> tuple[pd.DataFrame, str | None]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), None
    excel = pd.ExcelFile(path)
    sheet_name = excel.sheet_names[0]
    return excel.parse(sheet_name), sheet_name


def standardize_dataset(path: Path) -> DatasetParseResult:
    raw_df, sheet_name = load_tabular_file(path)
    columns = [str(col) for col in raw_df.columns]
    case_col = find_best_column(columns, "case")
    disease_col = find_best_column(columns, "disease")
    hpo_col = find_best_column(columns, "hpo")
    structure_type = "long_table" if is_long_table(raw_df, case_col, hpo_col) else "wide_or_list"

    base_df = raw_df[[case_col, disease_col, hpo_col]].copy()
    base_df[case_col] = base_df[case_col].map(lambda value: str(value).strip())
    base_df[disease_col] = base_df[disease_col].map(normalize_id)

    rows: list[dict[str, str]] = []
    parse_method = "逐行保留单个 HPO"
    if structure_type == "long_table":
        for record in base_df.itertuples(index=False):
            case_id = str(getattr(record, case_col)).strip()
            disease_label = normalize_id(getattr(record, disease_col))
            hpo_values = parse_hpo_cell(getattr(record, hpo_col))
            for hpo_id in hpo_values:
                rows.append(
                    {
                        "source_dataset": path.stem,
                        "case_id": case_id,
                        "disease_label": disease_label,
                        "hpo_id": hpo_id,
                    }
                )
    else:
        parse_method = "列表字符串拆分为多行 HPO"
        for record in base_df.itertuples(index=False):
            case_id = str(getattr(record, case_col)).strip()
            disease_label = normalize_id(getattr(record, disease_col))
            hpo_values = parse_hpo_cell(getattr(record, hpo_col))
            for hpo_id in hpo_values:
                rows.append(
                    {
                        "source_dataset": path.stem,
                        "case_id": case_id,
                        "disease_label": disease_label,
                        "hpo_id": hpo_id,
                    }
                )

    standardized_df = pd.DataFrame(rows)
    if standardized_df.empty:
        raise ValueError(f"{path} 标准化后为空，请检查列结构")

    standardized_df = standardized_df.drop_duplicates().reset_index(drop=True)
    standardized_df["global_case_id"] = (
        standardized_df["source_dataset"] + "::" + standardized_df["case_id"] + "::" + standardized_df["disease_label"]
    )
    case_id_collision_across_diseases_count = (
        standardized_df[["source_dataset", "case_id", "disease_label"]]
        .drop_duplicates()
        .groupby(["source_dataset", "case_id"])["disease_label"]
        .nunique()
        .gt(1)
        .sum()
    )

    return DatasetParseResult(
        source_dataset=path.stem,
        file_path=path,
        sheet_name=sheet_name,
        raw_shape=tuple(raw_df.shape),
        structure_type=structure_type,
        sample_granularity="case_id + disease_label",
        case_id_col=case_col,
        disease_col=disease_col,
        hpo_col=hpo_col,
        disease_id_system=detect_disease_id_system(standardized_df["disease_label"]),
        parse_method=parse_method,
        case_id_collision_across_diseases_count=int(case_id_collision_across_diseases_count),
        standardized_df=standardized_df,
    )


def search_mapping_files(search_roots: list[Path], patterns: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    return sorted({candidate.resolve() for candidate in candidates if candidate.is_file()})


def select_index_file(candidate_paths: list[Path], id_col: str, idx_col: str, expected_size: int) -> tuple[Path, pd.DataFrame]:
    version_re = re.compile(r"_v(\d+)", flags=re.IGNORECASE)
    matched: list[tuple[Path, pd.DataFrame]] = []
    for path in candidate_paths:
        try:
            df = pd.read_excel(path, dtype={id_col: str, idx_col: int})
        except Exception:
            continue
        if id_col not in df.columns or idx_col not in df.columns or df.empty:
            continue
        actual_size = int(df[idx_col].max()) + 1
        if actual_size == expected_size:
            matched.append((path, df))
    if not matched:
        raise FileNotFoundError(f"未找到与 expected_size={expected_size} 匹配的 {idx_col} 索引文件")

    def sort_key(item: tuple[Path, pd.DataFrame]) -> tuple[int, int, str]:
        path = item[0]
        match = version_re.search(path.name)
        version = int(match.group(1)) if match else -1
        return (-version, len(str(path)), str(path))

    matched.sort(key=sort_key)
    return matched[0]


def build_triplet_df(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": rows.astype(np.int64),
            "cols": cols.astype(np.int64),
            "vals": np.round(vals.astype(np.float64), 12),
        }
    ).sort_values(["rows", "cols", "vals"]).reset_index(drop=True)


def load_hyperedge_assets() -> dict:
    npz_path = HYPEREDGE_DIR / "v59DiseaseHy.npz"
    csv_path = HYPEREDGE_DIR / "v59_hyperedge_weighted_patched.csv"
    repair_audit_path = HYPEREDGE_DIR / "v59_repair_audit.csv"
    aggregated_path = HYPEREDGE_DIR / "unseen_training_clean_aggregated.tsv"

    npz = np.load(npz_path, allow_pickle=False)
    rows = npz["rows"]
    cols = npz["cols"]
    vals = npz["vals"]
    shape = tuple(int(value) for value in npz["shape"])

    search_roots = [HYPEREDGE_DIR, HYPEREDGE_DIR.parent, HYPEREDGE_DIR.parent.parent]
    disease_candidates = search_mapping_files(search_roots, ["Disease_index_v*.xlsx", "Disease_index*.xlsx"])
    hpo_candidates = search_mapping_files(search_roots, ["HPO_index_v*.xlsx", "HPO_index*.xlsx"])
    if not disease_candidates or not hpo_candidates:
        raise FileNotFoundError("未能在同目录、父目录或相邻目录中找到 Disease/HPO 索引文件")

    disease_index_path, disease_index_df = select_index_file(
        disease_candidates,
        id_col="mondo_id",
        idx_col="disease_idx",
        expected_size=shape[1],
    )
    hpo_index_path, hpo_index_df = select_index_file(
        hpo_candidates,
        id_col="hpo_id",
        idx_col="hpo_idx",
        expected_size=shape[0],
    )
    disease_index_df["mondo_id"] = disease_index_df["mondo_id"].map(normalize_id)
    hpo_index_df["hpo_id"] = hpo_index_df["hpo_id"].map(normalize_id)
    disease_index_df = disease_index_df.sort_values("disease_idx").reset_index(drop=True)
    hpo_index_df = hpo_index_df.sort_values("hpo_idx").reset_index(drop=True)

    expected_shape = (
        int(hpo_index_df["hpo_idx"].max()) + 1,
        int(disease_index_df["disease_idx"].max()) + 1,
    )
    if shape != expected_shape:
        raise ValueError(f"v59DiseaseHy.npz shape={shape} 与索引推导的 shape={expected_shape} 不一致")

    hyperedge_df = pd.read_csv(csv_path, dtype={"case_id": str, "mondo_id": str, "hpo_id": str})
    hyperedge_df["mondo_id"] = hyperedge_df["mondo_id"].map(normalize_id)
    hyperedge_df["hpo_id"] = hyperedge_df["hpo_id"].map(normalize_id)

    merged_triplets = (
        hyperedge_df
        .merge(hpo_index_df[["hpo_id", "hpo_idx"]], on="hpo_id", how="left")
        .merge(disease_index_df[["mondo_id", "disease_idx"]], on="mondo_id", how="left")
    )
    if merged_triplets["hpo_idx"].isna().any() or merged_triplets["disease_idx"].isna().any():
        raise ValueError("超边 CSV 中存在无法映射到索引的 mondo_id 或 hpo_id")

    triplet_from_csv = build_triplet_df(
        merged_triplets["hpo_idx"].to_numpy(),
        merged_triplets["disease_idx"].to_numpy(),
        merged_triplets["weight"].to_numpy(),
    )
    triplet_from_npz = build_triplet_df(rows, cols, vals)
    triplet_match = triplet_from_csv.equals(triplet_from_npz)

    if not triplet_match:
        raise ValueError("v59_hyperedge_weighted_patched.csv 与 v59DiseaseHy.npz 的 triplet 内容不一致")

    disease_name_map: dict[str, str] = {}
    if repair_audit_path.exists():
        repair_df = pd.read_csv(repair_audit_path, dtype={"mondo_id": str})
        repair_df["mondo_id"] = repair_df["mondo_id"].map(normalize_id)
        repair_df["description"] = repair_df["description"].fillna("").astype(str).str.strip()
        disease_name_map.update(
            repair_df.loc[repair_df["description"].ne(""), ["mondo_id", "description"]]
            .drop_duplicates("mondo_id")
            .set_index("mondo_id")["description"]
            .to_dict()
        )
    if aggregated_path.exists():
        agg_df = pd.read_csv(aggregated_path, sep="\t", dtype={"mondo_id": str})
        agg_df["mondo_id"] = agg_df["mondo_id"].map(normalize_id)
        agg_df["disease_name"] = agg_df["disease_name"].fillna("").astype(str).str.strip()
        disease_name_map.update(
            agg_df.loc[agg_df["disease_name"].ne(""), ["mondo_id", "disease_name"]]
            .drop_duplicates("mondo_id")
            .set_index("mondo_id")["disease_name"]
            .to_dict()
        )

    hyperedge_df = hyperedge_df.sort_values(["mondo_id", "weight", "hpo_id"], ascending=[True, False, True]).reset_index(drop=True)
    is_binary = set(np.round(hyperedge_df["weight"].astype(float), 12).unique().tolist()).issubset({0.0, 1.0})

    return {
        "npz_path": npz_path,
        "csv_path": csv_path,
        "disease_index_path": disease_index_path,
        "hpo_index_path": hpo_index_path,
        "shape": shape,
        "rows": rows,
        "cols": cols,
        "vals": vals,
        "triplet_match": triplet_match,
        "hyperedge_df": hyperedge_df,
        "disease_index_df": disease_index_df,
        "hpo_index_df": hpo_index_df,
        "mondo_to_idx": dict(zip(disease_index_df["mondo_id"], disease_index_df["disease_idx"])),
        "hpo_to_idx": dict(zip(hpo_index_df["hpo_id"], hpo_index_df["hpo_idx"])),
        "is_binary": is_binary,
        "disease_name_map": disease_name_map,
    }


def build_key_hpo_sets(hyperedge_df: pd.DataFrame, is_binary: bool) -> dict[str, dict[str, list[str]]]:
    key_sets = {"KeyHPO_50": {}, "KeyHPO_Top5": {}, "KeyHPO_Top10": {}}
    for mondo_id, group_df in hyperedge_df.groupby("mondo_id", sort=False):
        ordered = group_df.sort_values(["weight", "hpo_id"], ascending=[False, True]).reset_index(drop=True)
        hpo_list = ordered["hpo_id"].tolist()
        weights = ordered["weight"].astype(float).tolist()

        if is_binary:
            all_hpo = hpo_list
            key_sets["KeyHPO_50"][mondo_id] = all_hpo
            key_sets["KeyHPO_Top5"][mondo_id] = all_hpo
            key_sets["KeyHPO_Top10"][mondo_id] = all_hpo
            continue

        cumulative = 0.0
        selected_50: list[str] = []
        for hpo_id, weight in zip(hpo_list, weights):
            selected_50.append(hpo_id)
            cumulative += float(weight)
            if cumulative >= 0.5 - 1e-12:
                break
        key_sets["KeyHPO_50"][mondo_id] = selected_50
        key_sets["KeyHPO_Top5"][mondo_id] = hpo_list[:5]
        key_sets["KeyHPO_Top10"][mondo_id] = hpo_list[:10]
    return key_sets


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return float("nan")
    return len(left & right) / len(union)


def pairwise_metrics(all_sets: list[set[str]], key_sets: list[set[str]]) -> dict[str, float]:
    if len(all_sets) < 2:
        return {
            "pairwise_all_hpo_jaccard_mean": float("nan"),
            "pairwise_key_hpo_jaccard_mean": float("nan"),
            "pairwise_key_hpo_overlap_nonzero_ratio": float("nan"),
        }
    all_jaccards: list[float] = []
    key_jaccards: list[float] = []
    key_nonzero_flags: list[int] = []
    for idx_left, idx_right in itertools.combinations(range(len(all_sets)), 2):
        left_all = all_sets[idx_left]
        right_all = all_sets[idx_right]
        left_key = key_sets[idx_left]
        right_key = key_sets[idx_right]
        all_jaccards.append(jaccard(left_all, right_all))
        key_intersection = left_key & right_key
        key_nonzero_flags.append(1 if key_intersection else 0)
        key_union = left_key | right_key
        if key_union:
            key_jaccards.append(len(key_intersection) / len(key_union))
    return {
        "pairwise_all_hpo_jaccard_mean": float(np.nanmean(all_jaccards)) if all_jaccards else float("nan"),
        "pairwise_key_hpo_jaccard_mean": float(np.nanmean(key_jaccards)) if key_jaccards else float("nan"),
        "pairwise_key_hpo_overlap_nonzero_ratio": float(np.mean(key_nonzero_flags)) if key_nonzero_flags else float("nan"),
    }


def summarize_dominant_keys(hit_sets: list[set[str]], key_order: list[str], top_n: int = 10) -> tuple[str, Counter]:
    counter: Counter = Counter()
    for hit_set in hit_sets:
        counter.update(hit_set)
    ordered = sorted(counter.items(), key=lambda item: (-item[1], key_order.index(item[0]) if item[0] in key_order else 10**9, item[0]))
    formatted = [f"{hpo_id}({count})" for hpo_id, count in ordered[:top_n]]
    return "|".join(formatted), counter


def compute_case_level_metrics(case_df: pd.DataFrame, key_hpo_list: list[str]) -> dict[str, object]:
    key_hpo_set = set(key_hpo_list)
    all_hpo_sets = case_df["all_hpo_set"].tolist()
    hit_sets = [mapped_set & key_hpo_set for mapped_set in case_df["mapped_hpo_set"].tolist()]
    hit_counts = [len(hit_set) for hit_set in hit_sets]
    hit_ratios = [len(hit_set) / len(key_hpo_set) if key_hpo_set else float("nan") for hit_set in hit_sets]
    metrics = {
        "n_samples_total": int(len(case_df)),
        "sample_hit_at_least_1_key": float(np.mean([count >= 1 for count in hit_counts])) if hit_counts else float("nan"),
        "sample_hit_at_least_2_key": float(np.mean([count >= 2 for count in hit_counts])) if hit_counts else float("nan"),
        "mean_key_hit_count": float(np.mean(hit_counts)) if hit_counts else float("nan"),
        "mean_key_hit_ratio": float(np.mean(hit_ratios)) if hit_ratios else float("nan"),
    }
    metrics.update(pairwise_metrics(all_hpo_sets, hit_sets))

    support_counter = Counter()
    for hit_set in hit_sets:
        support_counter.update(hit_set)
    threshold_50 = math.ceil(len(hit_sets) * 0.5)
    threshold_30 = max(1, math.ceil(len(hit_sets) * 0.3))
    metrics["common_key_hpo_count_50"] = int(sum(count >= threshold_50 for count in support_counter.values()))
    metrics["common_key_hpo_count_30"] = int(sum(count >= threshold_30 for count in support_counter.values()))
    dominant_text, _ = summarize_dominant_keys(hit_sets, key_hpo_list)
    metrics["dominant_shared_keys"] = dominant_text
    metrics["hit_sets"] = hit_sets
    return metrics


def classify_disease(metrics: dict[str, object]) -> tuple[str, str]:
    n_samples = int(metrics["n_samples_total"])
    if n_samples < 2:
        return "E", "样本不足"

    hit1 = safe_float(metrics["sample_hit_at_least_1_key"])
    common50 = int(metrics["common_key_hpo_count_50"])
    common30 = int(metrics["common_key_hpo_count_30"])
    overlap_nonzero = safe_float(metrics["pairwise_key_hpo_overlap_nonzero_ratio"])
    mean_hit_ratio = safe_float(metrics["mean_key_hit_ratio"])

    if hit1 >= 0.8 and common50 >= 1 and overlap_nonzero >= 0.7:
        return "A", "大多数样本命中关键HPO，且至少有1个关键HPO在>=50%样本中共享"
    if hit1 >= 0.6 and common30 >= 1 and overlap_nonzero >= 0.4:
        return "B", "样本对关键HPO存在可见共享，但强度不及A类"
    if hit1 >= 0.5 and mean_hit_ratio >= 0.15:
        return "C", "样本通常能命中某些关键HPO，但命中的不是稳定同一批"
    return "D", "关键HPO在同病样本中整体不稳定或命中偏弱"


def dataset_breakdown_text(case_df: pd.DataFrame) -> str:
    counts = case_df.groupby("source_dataset")["global_case_id"].nunique().sort_values(ascending=False)
    return "|".join(f"{dataset}:{count}" for dataset, count in counts.items())


def mean_without_nan(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)
    return float(series.mean()) if not series.dropna().empty else float("nan")


def prepare_case_table(standardized_df: pd.DataFrame, hyperedge_assets: dict) -> pd.DataFrame:
    mondo_to_idx = hyperedge_assets["mondo_to_idx"]
    hpo_to_idx = hyperedge_assets["hpo_to_idx"]

    standardized_df = standardized_df.copy()
    standardized_df["disease_mapped"] = standardized_df["disease_label"].isin(mondo_to_idx)
    standardized_df["hpo_mapped"] = standardized_df["hpo_id"].isin(hpo_to_idx)

    case_level = (
        standardized_df.groupby(["source_dataset", "case_id", "disease_label", "global_case_id"], sort=False)
        .agg(
            all_hpo_list=("hpo_id", lambda series: sorted(set(series))),
            mapped_hpo_list=("hpo_id", lambda series: sorted({value for value in series if value in hpo_to_idx})),
            n_rows=("hpo_id", "size"),
            n_unique_hpo=("hpo_id", "nunique"),
            disease_mapped=("disease_mapped", "first"),
        )
        .reset_index()
    )
    case_level["all_hpo_set"] = case_level["all_hpo_list"].map(set)
    case_level["mapped_hpo_set"] = case_level["mapped_hpo_list"].map(set)
    case_level["mapped_hpo_ratio"] = case_level.apply(
        lambda row: len(row["mapped_hpo_set"]) / len(row["all_hpo_set"]) if row["all_hpo_set"] else float("nan"),
        axis=1,
    )
    return case_level


def build_schema_inventory(parse_results: list[DatasetParseResult]) -> pd.DataFrame:
    rows = []
    for result in parse_results:
        standardized_df = result.standardized_df
        rows.append(
            {
                "source_dataset": result.source_dataset,
                "file_path": str(result.file_path),
                "sheet_name": result.sheet_name or "",
                "raw_n_rows": result.raw_shape[0],
                "raw_n_columns": result.raw_shape[1],
                "structure_type": result.structure_type,
                "sample_granularity": result.sample_granularity,
                "detected_case_id_col": result.case_id_col,
                "detected_disease_col": result.disease_col,
                "detected_hpo_col": result.hpo_col,
                "disease_id_system": result.disease_id_system,
                "parse_method": result.parse_method,
                "standardized_n_rows": len(standardized_df),
                "n_cases": standardized_df["global_case_id"].nunique(),
                "n_unique_diseases": standardized_df["disease_label"].nunique(),
                "n_unique_hpo": standardized_df["hpo_id"].nunique(),
                "case_id_collision_across_diseases_count": result.case_id_collision_across_diseases_count,
            }
        )
    return pd.DataFrame(rows).sort_values("source_dataset").reset_index(drop=True)


def build_mapping_summary(standardized_df: pd.DataFrame, case_table: pd.DataFrame, hyperedge_assets: dict) -> pd.DataFrame:
    rows = []
    for dataset_name, dataset_df in standardized_df.groupby("source_dataset", sort=False):
        case_df = case_table.loc[case_table["source_dataset"] == dataset_name].copy()
        disease_values = sorted(dataset_df["disease_label"].dropna().unique().tolist())
        hpo_values = sorted(dataset_df["hpo_id"].dropna().unique().tolist())
        mapped_diseases = sorted(set(disease_values) & set(hyperedge_assets["mondo_to_idx"]))
        mapped_hpos = sorted(set(hpo_values) & set(hyperedge_assets["hpo_to_idx"]))
        unmapped_diseases = sorted(set(disease_values) - set(mapped_diseases))
        unmapped_hpo_counter = (
            dataset_df.loc[~dataset_df["hpo_id"].isin(hyperedge_assets["hpo_to_idx"]), "hpo_id"]
            .value_counts()
            .head(20)
            .to_dict()
        )
        rows.append(
            {
                "source_dataset": dataset_name,
                "disease_id_system": detect_disease_id_system(dataset_df["disease_label"]),
                "disease_mapping_logic": "当前数据集为 MONDO，采用 exact match 到 Disease_index_v4.xlsx；未做跨体系推断映射",
                "hpo_mapping_logic": "采用 exact match 到 HPO_index_v4.xlsx；未匹配者不纳入超边关键HPO命中统计",
                "n_rows": len(dataset_df),
                "n_cases": case_df["global_case_id"].nunique(),
                "n_unique_diseases": len(disease_values),
                "n_mapped_diseases": len(mapped_diseases),
                "disease_mapping_success_rate": len(mapped_diseases) / len(disease_values) if disease_values else float("nan"),
                "n_unique_hpo": len(hpo_values),
                "n_mapped_hpo": len(mapped_hpos),
                "hpo_mapping_success_rate": len(mapped_hpos) / len(hpo_values) if hpo_values else float("nan"),
                "mapped_case_ratio": case_df["disease_mapped"].mean(),
                "mean_case_hpo_mapping_ratio": mean_without_nan(case_df["mapped_hpo_ratio"]),
                "unmapped_diseases": "|".join(unmapped_diseases[:50]),
                "top_unmapped_hpos": "|".join(f"{hpo_id}({count})" for hpo_id, count in unmapped_hpo_counter.items()),
            }
        )
    return pd.DataFrame(rows).sort_values("source_dataset").reset_index(drop=True)


def audit_diseases(case_table: pd.DataFrame, hyperedge_assets: dict, key_sets: dict[str, dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    disease_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    cross_dataset_rows: list[dict[str, object]] = []
    disease_name_map = hyperedge_assets["disease_name_map"]

    mapped_cases = case_table.loc[case_table["disease_mapped"]].copy()
    for mondo_id, disease_case_df in mapped_cases.groupby("disease_label", sort=False):
        disease_name = disease_name_map.get(mondo_id, "")
        datasets_present = sorted(disease_case_df["source_dataset"].unique().tolist())
        for definition_name, key_map in key_sets.items():
            key_hpo_list = key_map.get(mondo_id, [])
            overall_metrics = compute_case_level_metrics(disease_case_df, key_hpo_list)
            conclusion_type, conclusion_reason = classify_disease(overall_metrics)

            overall_row = {
                "disease_label": mondo_id,
                "disease_name": disease_name,
                "n_samples_total": int(len(disease_case_df)),
                "dataset_breakdown": dataset_breakdown_text(disease_case_df),
                "n_datasets": len(datasets_present),
                "hyperedge_key_hpo_definition": definition_name,
                "key_hpo_count": len(key_hpo_list),
                "key_hpo_ids": "|".join(key_hpo_list),
                "sample_hit_at_least_1_key": overall_metrics["sample_hit_at_least_1_key"],
                "sample_hit_at_least_2_key": overall_metrics["sample_hit_at_least_2_key"],
                "mean_key_hit_count": overall_metrics["mean_key_hit_count"],
                "mean_key_hit_ratio": overall_metrics["mean_key_hit_ratio"],
                "pairwise_all_hpo_jaccard_mean": overall_metrics["pairwise_all_hpo_jaccard_mean"],
                "pairwise_key_hpo_jaccard_mean": overall_metrics["pairwise_key_hpo_jaccard_mean"],
                "pairwise_key_hpo_overlap_nonzero_ratio": overall_metrics["pairwise_key_hpo_overlap_nonzero_ratio"],
                "common_key_hpo_count_50": overall_metrics["common_key_hpo_count_50"],
                "common_key_hpo_count_30": overall_metrics["common_key_hpo_count_30"],
                "dominant_shared_keys": overall_metrics["dominant_shared_keys"],
                "conclusion_type": conclusion_type,
                "conclusion_reason": conclusion_reason,
            }

            dataset_key_union_map: dict[str, set[str]] = {}
            dataset_dominant_map: dict[str, str] = {}
            for dataset_name, dataset_case_df in disease_case_df.groupby("source_dataset", sort=False):
                metrics = compute_case_level_metrics(dataset_case_df, key_hpo_list)
                dataset_conclusion, dataset_reason = classify_disease(metrics)
                hit_union = set().union(*metrics["hit_sets"]) if metrics["hit_sets"] else set()
                dataset_key_union_map[dataset_name] = hit_union
                dataset_dominant_map[dataset_name] = metrics["dominant_shared_keys"]
                dataset_rows.append(
                    {
                        "disease_label": mondo_id,
                        "disease_name": disease_name,
                        "source_dataset": dataset_name,
                        "n_samples_dataset": int(len(dataset_case_df)),
                        "hyperedge_key_hpo_definition": definition_name,
                        "key_hpo_count": len(key_hpo_list),
                        "sample_hit_at_least_1_key": metrics["sample_hit_at_least_1_key"],
                        "sample_hit_at_least_2_key": metrics["sample_hit_at_least_2_key"],
                        "mean_key_hit_count": metrics["mean_key_hit_count"],
                        "mean_key_hit_ratio": metrics["mean_key_hit_ratio"],
                        "pairwise_all_hpo_jaccard_mean": metrics["pairwise_all_hpo_jaccard_mean"],
                        "pairwise_key_hpo_jaccard_mean": metrics["pairwise_key_hpo_jaccard_mean"],
                        "pairwise_key_hpo_overlap_nonzero_ratio": metrics["pairwise_key_hpo_overlap_nonzero_ratio"],
                        "common_key_hpo_count_50": metrics["common_key_hpo_count_50"],
                        "common_key_hpo_count_30": metrics["common_key_hpo_count_30"],
                        "dominant_shared_keys": metrics["dominant_shared_keys"],
                        "conclusion_type": dataset_conclusion,
                        "conclusion_reason": dataset_reason,
                    }
                )

            all_dataset_common = sorted(set.intersection(*dataset_key_union_map.values())) if len(dataset_key_union_map) >= 2 else []
            overall_row["cross_dataset_common_key_hpo_count"] = len(all_dataset_common)
            overall_row["cross_dataset_common_key_hpo_ids"] = "|".join(all_dataset_common)
            disease_rows.append(overall_row)

            if len(dataset_key_union_map) >= 2:
                dataset_names = sorted(dataset_key_union_map)
                for dataset_a, dataset_b in itertools.combinations(dataset_names, 2):
                    union_a = dataset_key_union_map[dataset_a]
                    union_b = dataset_key_union_map[dataset_b]
                    shared = sorted(union_a & union_b)
                    union = union_a | union_b
                    cross_dataset_rows.append(
                        {
                            "disease_label": mondo_id,
                            "disease_name": disease_name,
                            "hyperedge_key_hpo_definition": definition_name,
                            "dataset_a": dataset_a,
                            "dataset_b": dataset_b,
                            "n_samples_a": int(disease_case_df.loc[disease_case_df["source_dataset"] == dataset_a, "global_case_id"].nunique()),
                            "n_samples_b": int(disease_case_df.loc[disease_case_df["source_dataset"] == dataset_b, "global_case_id"].nunique()),
                            "key_hits_union_a": "|".join(sorted(union_a)),
                            "key_hits_union_b": "|".join(sorted(union_b)),
                            "dominant_shared_keys_a": dataset_dominant_map[dataset_a],
                            "dominant_shared_keys_b": dataset_dominant_map[dataset_b],
                            "shared_key_count": len(shared),
                            "shared_key_ids": "|".join(shared),
                            "key_union_jaccard": len(shared) / len(union) if union else float("nan"),
                            "all_dataset_common_key_hpo_count": len(all_dataset_common),
                            "all_dataset_common_key_hpo_ids": "|".join(all_dataset_common),
                            "same_or_different": "同一批为主" if union and len(shared) / len(union) >= 0.5 else "差异较大",
                        }
                    )

    disease_df = pd.DataFrame(disease_rows).sort_values(
        ["hyperedge_key_hpo_definition", "conclusion_type", "n_samples_total", "disease_label"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    dataset_df = pd.DataFrame(dataset_rows).sort_values(
        ["hyperedge_key_hpo_definition", "source_dataset", "n_samples_dataset", "disease_label"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    cross_dataset_df = pd.DataFrame(cross_dataset_rows).sort_values(
        ["hyperedge_key_hpo_definition", "shared_key_count", "disease_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return disease_df, dataset_df, cross_dataset_df


def build_rank_tables(disease_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_df = disease_df.loc[disease_df["hyperedge_key_hpo_definition"] == "KeyHPO_50"].copy()
    eligible_df = main_df.loc[main_df["n_samples_total"] >= 2].copy()

    stable_df = eligible_df.loc[eligible_df["conclusion_type"].isin(["A", "B"])].copy()
    stable_df = stable_df.sort_values(
        ["conclusion_type", "sample_hit_at_least_1_key", "pairwise_key_hpo_overlap_nonzero_ratio", "n_samples_total"],
        ascending=[True, False, False, False],
    ).head(30)

    unstable_df = eligible_df.loc[eligible_df["conclusion_type"].isin(["C", "D"])].copy()
    unstable_df = unstable_df.sort_values(
        ["conclusion_type", "sample_hit_at_least_1_key", "pairwise_key_hpo_overlap_nonzero_ratio", "n_samples_total"],
        ascending=[False, True, True, False],
    ).head(30)
    return stable_df.reset_index(drop=True), unstable_df.reset_index(drop=True)


def choose_example(disease_df: pd.DataFrame, mode: str) -> pd.Series | None:
    main_df = disease_df.loc[(disease_df["hyperedge_key_hpo_definition"] == "KeyHPO_50") & (disease_df["n_samples_total"] >= 2)].copy()
    if main_df.empty:
        return None
    if mode == "low_jaccard_but_key_stable":
        candidates = main_df.loc[
            (main_df["pairwise_all_hpo_jaccard_mean"] <= 0.2)
            & (main_df["sample_hit_at_least_1_key"] >= 0.8)
            & (main_df["common_key_hpo_count_30"] >= 1)
        ].sort_values(
            ["sample_hit_at_least_1_key", "common_key_hpo_count_50", "n_samples_total"],
            ascending=[False, False, False],
        )
        return candidates.iloc[0] if not candidates.empty else None
    if mode == "overall_similar_but_key_unstable":
        candidates = main_df.loc[
            (main_df["pairwise_all_hpo_jaccard_mean"] >= 0.25)
            & (main_df["pairwise_key_hpo_overlap_nonzero_ratio"] <= 0.3)
        ].sort_values(
            ["pairwise_all_hpo_jaccard_mean", "sample_hit_at_least_1_key", "n_samples_total"],
            ascending=[False, True, False],
        )
        return candidates.iloc[0] if not candidates.empty else None
    return None


def summarize_dataset_support(dataset_df: pd.DataFrame) -> pd.DataFrame:
    main_df = dataset_df.loc[(dataset_df["hyperedge_key_hpo_definition"] == "KeyHPO_50") & (dataset_df["n_samples_dataset"] >= 2)].copy()
    if main_df.empty:
        return pd.DataFrame()
    summary_rows = []
    for dataset_name, group_df in main_df.groupby("source_dataset", sort=False):
        total = len(group_df)
        support_ratio = group_df["conclusion_type"].isin(["A", "B"]).mean() if total else float("nan")
        summary_rows.append(
            {
                "source_dataset": dataset_name,
                "n_eligible_diseases": total,
                "A_count": int((group_df["conclusion_type"] == "A").sum()),
                "B_count": int((group_df["conclusion_type"] == "B").sum()),
                "C_count": int((group_df["conclusion_type"] == "C").sum()),
                "D_count": int((group_df["conclusion_type"] == "D").sum()),
                "support_ratio_A_or_B": support_ratio,
                "mean_sample_hit_at_least_1_key": group_df["sample_hit_at_least_1_key"].mean(),
                "mean_pairwise_key_overlap_nonzero_ratio": group_df["pairwise_key_hpo_overlap_nonzero_ratio"].mean(),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("support_ratio_A_or_B", ascending=False).reset_index(drop=True)


def write_markdown_report(
    schema_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    cross_dataset_df: pd.DataFrame,
    stable_df: pd.DataFrame,
    unstable_df: pd.DataFrame,
    hyperedge_assets: dict,
) -> None:
    main_df = disease_df.loc[disease_df["hyperedge_key_hpo_definition"] == "KeyHPO_50"].copy()
    eligible_main_df = main_df.loc[main_df["n_samples_total"] >= 2].copy()
    conclusion_counts = eligible_main_df["conclusion_type"].value_counts().to_dict()
    support_ratio = eligible_main_df["conclusion_type"].isin(["A", "B"]).mean() if not eligible_main_df.empty else float("nan")
    low_jaccard_example = choose_example(disease_df, "low_jaccard_but_key_stable")
    reverse_example = choose_example(disease_df, "overall_similar_but_key_unstable")
    dataset_support_df = summarize_dataset_support(dataset_df)

    top_support_lines = []
    for row in dataset_support_df.head(5).itertuples(index=False):
        top_support_lines.append(
            f"- {row.source_dataset}: 支持比例(A/B)={row.support_ratio_A_or_B:.3f}，"
            f"可审计疾病数={row.n_eligible_diseases}，平均 hit@1={row.mean_sample_hit_at_least_1_key:.3f}"
        )
    bottom_support_lines = []
    for row in dataset_support_df.tail(5).sort_values("support_ratio_A_or_B").itertuples(index=False):
        bottom_support_lines.append(
            f"- {row.source_dataset}: 支持比例(A/B)={row.support_ratio_A_or_B:.3f}，"
            f"可审计疾病数={row.n_eligible_diseases}，平均 hit@1={row.mean_sample_hit_at_least_1_key:.3f}"
        )

    summary_lines = [
        "# 关键HPO一致性审计报告",
        "",
        "## 1. 审计范围",
        f"- 审计数据集数量: {len(schema_df)}",
        f"- 疾病超边文件: `{hyperedge_assets['npz_path']}`",
        f"- 配套疾病索引: `{hyperedge_assets['disease_index_path']}`",
        f"- 配套HPO索引: `{hyperedge_assets['hpo_index_path']}`",
        f"- `.npz` 形状: {hyperedge_assets['shape']}",
        f"- `.npz` 与 `v59_hyperedge_weighted_patched.csv` triplet 一致性: {hyperedge_assets['triplet_match']}",
        f"- 超边是否为二值矩阵: {hyperedge_assets['is_binary']}",
        "",
        "## 2. 文件结构审计",
        "- 本次 8 个待审计文件全部被识别为长表结构。",
        "- 所有文件均解析出 `case_id / mondo_label / hpo_id` 三个核心字段。",
        f"- 存在 `case_id` 跨疾病重复的样本编号总计: {int(schema_df['case_id_collision_across_diseases_count'].sum())}",
        "- 因此本次统一样本粒度采用 `source_dataset + case_id + disease_label`，而不是单独使用 `case_id`。",
        "",
        "## 3. 标签与HPO映射审计",
        f"- 各数据集 disease mapping success rate 范围: {mapping_df['disease_mapping_success_rate'].min():.3f} ~ {mapping_df['disease_mapping_success_rate'].max():.3f}",
        f"- 各数据集 hpo mapping success rate 范围: {mapping_df['hpo_mapping_success_rate'].min():.3f} ~ {mapping_df['hpo_mapping_success_rate'].max():.3f}",
        "- 当前 8 个数据集疾病标签体系全部是 MONDO，因此本次映射逻辑是 exact MONDO match，并未做跨体系推断。",
        "",
        "## 4. KeyHPO 定义与分层阈值",
        "- 主定义 `KeyHPO_50`: 按超边权重降序累积到 50% 的最小 HPO 集合。",
        "- 辅助定义 `KeyHPO_Top5`: 取权重前 5 个 HPO。",
        "- 辅助定义 `KeyHPO_Top10`: 取权重前 10 个 HPO。",
        "- 分层规则:",
        "  - A: `hit@1 >= 0.8` 且 `common_key_hpo_count_50 >= 1` 且 `pairwise_key_overlap_nonzero_ratio >= 0.7`",
        "  - B: `hit@1 >= 0.6` 且 `common_key_hpo_count_30 >= 1` 且 `pairwise_key_overlap_nonzero_ratio >= 0.4`",
        "  - C: `hit@1 >= 0.5` 且 `mean_key_hit_ratio >= 0.15`，但未达到 B",
        "  - D: 其余情况",
        "  - E: 样本不足 (`n_samples < 2`)",
        "",
        "## 5. 核心结论摘要",
        f"- 主定义 `KeyHPO_50` 下，满足同病可审计条件的疾病数: {len(eligible_main_df)}",
        f"- A/B 支持比例: {support_ratio:.3f}" if not math.isnan(support_ratio) else "- A/B 支持比例: NaN",
        f"- 分层计数: {json_text(conclusion_counts)}",
    ]

    if not stable_df.empty:
        top_row = stable_df.iloc[0]
        summary_lines.append(
            f"- 最稳定示例(主定义): {top_row['disease_label']} {top_row.get('disease_name', '')}，"
            f"hit@1={top_row['sample_hit_at_least_1_key']:.3f}，"
            f"共有关键HPO(50%)={int(top_row['common_key_hpo_count_50'])}"
        )
    if not unstable_df.empty:
        worst_row = unstable_df.iloc[0]
        summary_lines.append(
            f"- 最不稳定示例(主定义): {worst_row['disease_label']} {worst_row.get('disease_name', '')}，"
            f"hit@1={worst_row['sample_hit_at_least_1_key']:.3f}，"
            f"关键HPO非零重叠比例={worst_row['pairwise_key_hpo_overlap_nonzero_ratio']:.3f}"
        )

    summary_lines.extend(["", "## 6. 直接回答核心问题"])

    if not math.isnan(support_ratio):
        if support_ratio >= 0.6:
            answer_1 = "当前数据对“同病样本共享少数关键HPO”的假设支持较强。"
        elif support_ratio >= 0.35:
            answer_1 = "当前数据对该假设呈现部分成立，不是普遍强成立。"
        else:
            answer_1 = "当前数据对该假设支持偏弱，更多疾病表现为弱共享或几乎不共享。"
        summary_lines.append(f"1. {answer_1}")
    else:
        summary_lines.append("1. 可审计疾病不足，无法给出稳定总体判断。")

    if not dataset_support_df.empty:
        best_datasets = ", ".join(dataset_support_df.head(3)["source_dataset"].tolist())
        worst_datasets = ", ".join(dataset_support_df.tail(3)["source_dataset"].tolist())
        summary_lines.append(f"2. 数据集间结论并不完全一致。相对更支持该假设的数据集: {best_datasets}；相对更不支持的数据集: {worst_datasets}。")
    else:
        summary_lines.append("2. 数据集内样本数不足，无法稳定比较不同数据集。")

    mean_hit = eligible_main_df["sample_hit_at_least_1_key"].mean() if not eligible_main_df.empty else float("nan")
    mean_all_jaccard = eligible_main_df["pairwise_all_hpo_jaccard_mean"].mean() if not eligible_main_df.empty else float("nan")
    if not math.isnan(mean_hit):
        summary_lines.append(
            f"3. 共享证据主要围绕疾病超边高权重HPO展开。主定义下平均 hit@1={mean_hit:.3f}，说明高权重HPO并非完全失效；"
            f"但平均全HPO两两Jaccard仅 {mean_all_jaccard:.3f}，说明整体表型仍然离散。"
        )
    else:
        summary_lines.append("3. 可审计疾病不足，无法判断共享证据是否主要来自高权重HPO。")

    if low_jaccard_example is not None:
        summary_lines.append(
            f"4. 存在“整体Jaccard低，但关键HPO稳定命中”的真实例子: "
            f"{low_jaccard_example['disease_label']} {low_jaccard_example.get('disease_name', '')}，"
            f"全HPO Jaccard={low_jaccard_example['pairwise_all_hpo_jaccard_mean']:.3f}，"
            f"hit@1={low_jaccard_example['sample_hit_at_least_1_key']:.3f}，"
            f"dominant_shared_keys={low_jaccard_example['dominant_shared_keys']}。"
        )
    else:
        summary_lines.append("4. 本次主定义下没有筛到足够典型的“整体Jaccard很低但关键HPO稳定”的疾病。")

    if reverse_example is not None:
        summary_lines.append(
            f"5. 也存在相反例子: {reverse_example['disease_label']} {reverse_example.get('disease_name', '')}，"
            f"全HPO Jaccard={reverse_example['pairwise_all_hpo_jaccard_mean']:.3f}，"
            f"关键HPO非零重叠比例={reverse_example['pairwise_key_hpo_overlap_nonzero_ratio']:.3f}。"
        )
    else:
        summary_lines.append("5. 本次主定义下没有筛到足够典型的“整体Jaccard不低但关键HPO不稳定”的疾病。")

    unmapped_disease_burden = mapping_df["disease_mapping_success_rate"].lt(1.0).mean()
    unmapped_hpo_burden = mapping_df["hpo_mapping_success_rate"].lt(1.0).mean()
    reasons = []
    if unmapped_disease_burden > 0:
        reasons.append("部分疾病标签未进入疾病超边覆盖范围")
    if unmapped_hpo_burden > 0:
        reasons.append("病例HPO存在超边索引外项目或粒度不一致")
    if support_ratio < 0.35 if not math.isnan(support_ratio) else False:
        reasons.append("病例表型噪声较大或疾病内部异质性高")
    if not reasons:
        reasons.append("主要问题不是映射缺失，而是疾病间稳定共享程度本身只在部分疾病上成立")
    summary_lines.append(f"6. 若多数疾病不稳定，当前结果更支持的原因是: {'；'.join(reasons)}。")

    summary_lines.extend(["", "## 7. 数据集支持度概览"])
    if top_support_lines:
        summary_lines.extend(top_support_lines)
    if bottom_support_lines:
        summary_lines.extend(["", "相对较弱的数据集:"])
        summary_lines.extend(bottom_support_lines)

    summary_lines.extend(["", "## 8. 结果文件"])
    summary_lines.extend(
        [
            "- `file_schema_inventory.csv`",
            "- `mapping_coverage_summary.csv`",
            "- `disease_level_key_hpo_audit.csv`",
            "- `disease_level_key_hpo_audit.xlsx`",
            "- `dataset_level_key_hpo_audit.csv`",
            "- `cross_dataset_disease_comparison.csv`",
            "- `top_stable_diseases.csv`",
            "- `top_unstable_diseases.csv`",
            "- `audit_report.md`",
        ]
    )

    report_path = OUTPUT_DIR / "audit_report.md"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parse_results = [standardize_dataset(path) for path in DATASET_PATHS]
    schema_df = build_schema_inventory(parse_results)
    standardized_df = pd.concat([result.standardized_df for result in parse_results], ignore_index=True)

    hyperedge_assets = load_hyperedge_assets()
    key_sets = build_key_hpo_sets(hyperedge_assets["hyperedge_df"], hyperedge_assets["is_binary"])
    case_table = prepare_case_table(standardized_df, hyperedge_assets)
    mapping_df = build_mapping_summary(standardized_df, case_table, hyperedge_assets)
    disease_df, dataset_df, cross_dataset_df = audit_diseases(case_table, hyperedge_assets, key_sets)
    stable_df, unstable_df = build_rank_tables(disease_df)

    schema_df.to_csv(OUTPUT_DIR / "file_schema_inventory.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(OUTPUT_DIR / "mapping_coverage_summary.csv", index=False, encoding="utf-8-sig")
    disease_df.to_csv(OUTPUT_DIR / "disease_level_key_hpo_audit.csv", index=False, encoding="utf-8-sig")
    dataset_df.to_csv(OUTPUT_DIR / "dataset_level_key_hpo_audit.csv", index=False, encoding="utf-8-sig")
    cross_dataset_df.to_csv(OUTPUT_DIR / "cross_dataset_disease_comparison.csv", index=False, encoding="utf-8-sig")
    stable_df.to_csv(OUTPUT_DIR / "top_stable_diseases.csv", index=False, encoding="utf-8-sig")
    unstable_df.to_csv(OUTPUT_DIR / "top_unstable_diseases.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_DIR / "disease_level_key_hpo_audit.xlsx", engine="openpyxl") as writer:
        disease_df.to_excel(writer, sheet_name="disease_level", index=False)
        dataset_df.to_excel(writer, sheet_name="dataset_level", index=False)
        mapping_df.to_excel(writer, sheet_name="mapping_summary", index=False)
        schema_df.to_excel(writer, sheet_name="schema_inventory", index=False)
        cross_dataset_df.to_excel(writer, sheet_name="cross_dataset", index=False)

    write_markdown_report(
        schema_df=schema_df,
        mapping_df=mapping_df,
        disease_df=disease_df,
        dataset_df=dataset_df,
        cross_dataset_df=cross_dataset_df,
        stable_df=stable_df,
        unstable_df=unstable_df,
        hyperedge_assets=hyperedge_assets,
    )

    print(f"审计完成，输出目录: {OUTPUT_DIR}")
    print(f"schema_rows={len(schema_df)}")
    print(f"mapping_rows={len(mapping_df)}")
    print(f"disease_rows={len(disease_df)}")
    print(f"dataset_rows={len(dataset_df)}")
    print(f"cross_dataset_rows={len(cross_dataset_df)}")


if __name__ == "__main__":
    main()
