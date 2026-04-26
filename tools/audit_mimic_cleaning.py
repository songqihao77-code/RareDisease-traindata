from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = PROJECT_ROOT / "LLLdataset" / "dataset" / "mimic_test.csv"
DEFAULT_CLEANED = PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "test" / "mimic_test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "mimic_cleaning_audit"

ORPHA_TO_MONDO_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "raw" / "orphanet" / "mondo_hasdbxref_orphanet.sssom.tsv"
OMIM_TO_MONDO_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "raw" / "orphanet" / "mondo_exactmatch_omim.sssom.tsv"
ORPHA_TO_OMIM_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "raw" / "orphanet" / "orpha2omim.json"
ORPHA_TO_NAME_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "raw" / "orphanet" / "orpha2name.json"
DISEASE_INDEX_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "Disease_index_v4.xlsx"
HPO_INDEX_PATH = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "HPO_index_v4.xlsx"
HYPEREDGE_PATHS = [
    PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "rare_disease_hgnn_clean_package_v59" / "v59_hyperedge_weighted_patched.csv",
    PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "DiseaseHyperedge_data_v5.xlsx",
    PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed" / "DiseaseHyperedge_data_v4.xlsx",
]
MONDO_JSON_PATHS = [
    PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json",
    PROJECT_ROOT / "data" / "raw_data" / "mondo.json",
    PROJECT_ROOT / "LLLdataset" / "knowledge" / "mondo-rare.json",
]
HPO_OBO_PATHS = [
    PROJECT_ROOT / "data" / "raw_data" / "hp-base.obo",
]

HPO_RE = re.compile(r"HP[:_]\d{7}", flags=re.IGNORECASE)
MONDO_RE = re.compile(r"MONDO[:_]\d{1,7}", flags=re.IGNORECASE)
ORPHA_RE = re.compile(r"(?:ORPHA|ORPHANET)[:_]\d+", flags=re.IGNORECASE)
OMIM_RE = re.compile(r"OMIM[:_]\d+", flags=re.IGNORECASE)
ICD_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\d+)?\b|\b\d{3}(?:\.\d+)?\b", flags=re.IGNORECASE)


@dataclass
class CaseRecord:
    dataset: str
    case_key: str
    raw_index: int | None
    ids: dict[str, str]
    disease_names: list[str]
    disease_ids: list[str]
    mondo_labels: list[str]
    hpo_ids: list[str]
    text_hash: str
    text_preview: str
    row_count: int
    source_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for original vs cleaned mimic_test.")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无记录_"
    text_df = df.fillna("").astype(str)
    columns = list(text_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in text_df.iterrows():
        cells = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError(f"Unsupported table format: {path}")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def ordered_unique(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^(obsolete|non rare in europe)\s*:\s*", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_loose(text: str) -> list[str]:
    if not text:
        return []
    if any(sep in text for sep in [";", "|", "\n", "\t"]):
        parts = re.split(r"[;|\n\t]+", text)
    else:
        parts = re.split(r",\s*", text)
    return [part.strip().strip("'\"") for part in parts if part.strip().strip("'\"")]


def parse_listish(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        text = parsed
    if parsed is None:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return split_loose(text)


def normalize_hpo(value: Any) -> str | None:
    if is_missing(value):
        return None
    match = HPO_RE.search(str(value))
    if not match:
        bare = re.fullmatch(r"\d{7}", str(value).strip())
        if bare:
            return f"HP:{bare.group(0)}"
        return None
    return match.group(0).upper().replace("_", ":")


def normalize_mondo(value: Any) -> str | None:
    if is_missing(value):
        return None
    match = MONDO_RE.search(str(value))
    if not match:
        return None
    digits = re.search(r"\d+", match.group(0))
    if not digits:
        return None
    return f"MONDO:{digits.group(0).zfill(7)}"


def normalize_orpha(value: Any) -> str | None:
    if is_missing(value):
        return None
    match = ORPHA_RE.search(str(value))
    if not match:
        return None
    digits = re.search(r"\d+", match.group(0))
    if not digits:
        return None
    return f"ORPHA:{digits.group(0)}"


def normalize_omim(value: Any) -> str | None:
    if is_missing(value):
        return None
    match = OMIM_RE.search(str(value))
    if not match:
        return None
    digits = re.search(r"\d+", match.group(0))
    if not digits:
        return None
    return f"OMIM:{digits.group(0)}"


def parse_hpo_values(value: Any) -> list[str]:
    candidates = parse_listish(value)
    if not candidates:
        candidates = HPO_RE.findall("" if is_missing(value) else str(value))
    hpos = [normalized for item in candidates if (normalized := normalize_hpo(item))]
    return sorted(set(hpos))


def parse_prefixed_ids(value: Any, regex: re.Pattern[str], normalizer: Any) -> list[str]:
    candidates = parse_listish(value)
    if not candidates:
        candidates = regex.findall("" if is_missing(value) else str(value))
    ids = [normalized for item in candidates if (normalized := normalizer(item))]
    return sorted(set(ids))


def sha1_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()


def preview(value: Any, limit: int = 160) -> str:
    if is_missing(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def list_to_cell(values: list[str] | set[str] | tuple[str, ...]) -> str:
    return "|".join(sorted(str(value) for value in values if str(value).strip()))


def nonempty_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def pipe_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.contains("|", regex=False).sum())


def infer_column_role(column: str, sample_values: list[str]) -> str:
    name = column.lower()
    values_text = " ".join(sample_values[:20])
    if name in {"case_id"}:
        return "case_id 字段"
    if name in {"patient_id", "subject_id", "hadm_id", "note_id"}:
        return f"{column} ID 字段"
    if name in {"mondo_label", "mondo_id"}:
        return "MONDO disease ID / gold label 字段"
    if "hpo" in name or HPO_RE.search(values_text):
        return "HPO phenotype 字段"
    if "orpha" in name or ORPHA_RE.search(values_text):
        return "ORPHA disease ID/name 字段"
    if "omim" in name or OMIM_RE.search(values_text):
        return "OMIM disease ID 字段"
    if "icd" in name or ICD_RE.search(values_text):
        return "ICD code 字段"
    if name in {"diagnosis", "disease", "rare", "rare_disease", "label"} or "diagnosis" in name:
        return "disease label/name 字段"
    if name in {"text", "note", "clinical_text"} or "text" in name:
        return "free-text 临床文本字段"
    if "embedding" in name:
        return "embedding / 向量字段"
    if "split" in name:
        return "split 字段"
    return "无法确认"


def infer_cell_format(series: pd.Series) -> str:
    non_null = [str(value).strip() for value in series.dropna().head(50).tolist() if str(value).strip()]
    if not non_null:
        return "empty/unknown"
    counts = Counter()
    for text in non_null:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                counts["JSON list"] += 1
                continue
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                counts["Python literal list"] += 1
                continue
        except Exception:
            pass
        if ";" in text:
            counts["semicolon separated string"] += 1
        elif "|" in text:
            counts["pipe separated string"] += 1
        elif "," in text:
            counts["comma separated string"] += 1
        else:
            counts["single string"] += 1
    return counts.most_common(1)[0][0] if counts else "unknown"


def load_simple_mapping(path: Path, object_normalizer: Any) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    if not path.is_file():
        return mapping
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"subject_id", "object_id"}.issubset(df.columns):
        return mapping
    for row in df.itertuples(index=False):
        mondo_id = normalize_mondo(getattr(row, "subject_id", ""))
        object_id = object_normalizer(getattr(row, "object_id", ""))
        if mondo_id and object_id:
            mapping[object_id].add(mondo_id)
    return mapping


def load_json_dict(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): str(value) for key, value in data.items()}


def load_orpha_to_omim(path: Path) -> dict[str, str]:
    raw = load_json_dict(path)
    result: dict[str, str] = {}
    for key, value in raw.items():
        orpha = normalize_orpha(key)
        omim = normalize_omim(value) or (f"OMIM:{value.strip()}" if str(value).strip().isdigit() else None)
        if orpha and omim:
            result[orpha] = omim
    return result


def load_orpha_to_name(path: Path) -> dict[str, str]:
    raw = load_json_dict(path)
    result: dict[str, str] = {}
    for key, value in raw.items():
        orpha = normalize_orpha(key)
        if orpha:
            result[orpha] = str(value).strip()
    return result


def load_index_set(path: Path, column: str) -> set[str]:
    if not path.is_file():
        return set()
    df = pd.read_excel(path, dtype=str, usecols=[column])
    return {str(value).strip() for value in df[column].dropna().tolist() if str(value).strip()}


def load_mondo_resource() -> dict[str, Any]:
    resource = {
        "path": "",
        "labels": {},
        "name_to_ids": defaultdict(set),
        "parents": defaultdict(set),
        "children": defaultdict(set),
        "obsolete": set(),
        "alt_to_id": {},
        "replaced_by": defaultdict(set),
        "available": False,
        "error": "",
    }
    path = next((candidate for candidate in MONDO_JSON_PATHS if candidate.is_file()), None)
    if path is None:
        resource["error"] = "无法确认，需要额外文件：MONDO JSON/OBO"
        return resource
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        graphs = data.get("graphs", [])
        graph = graphs[0] if graphs else {}
        for node in graph.get("nodes", []):
            mondo_id = normalize_mondo(node.get("id"))
            if not mondo_id:
                continue
            meta = node.get("meta") or {}
            label = str(node.get("lbl") or "").strip()
            if label:
                resource["labels"][mondo_id] = label
                norm = normalize_text(label)
                if norm:
                    resource["name_to_ids"][norm].add(mondo_id)
            if meta.get("deprecated") is True or label.lower().startswith("obsolete "):
                resource["obsolete"].add(mondo_id)
            for basic_property in meta.get("basicPropertyValues", []) or []:
                pred = str(basic_property.get("pred") or "")
                val = str(basic_property.get("val") or "")
                if "IAO_0100001" in pred or "replaced_by" in pred:
                    replaced = normalize_mondo(val)
                    if replaced:
                        resource["replaced_by"][mondo_id].add(replaced)
            for xref in meta.get("xrefs", []) or []:
                val = xref.get("val") if isinstance(xref, dict) else xref
                alt = normalize_mondo(val)
                if alt and alt != mondo_id:
                    resource["alt_to_id"][alt] = mondo_id
            for synonym in meta.get("synonyms", []) or []:
                if not isinstance(synonym, dict):
                    continue
                value = synonym.get("val")
                norm = normalize_text(value)
                if norm:
                    resource["name_to_ids"][norm].add(mondo_id)
        for edge in graph.get("edges", []):
            sub = normalize_mondo(edge.get("sub"))
            obj = normalize_mondo(edge.get("obj"))
            pred = str(edge.get("pred") or "")
            if sub and obj and ("subClassOf" in pred or pred.endswith("/is_a") or pred == "is_a"):
                resource["parents"][sub].add(obj)
                resource["children"][obj].add(sub)
        resource["path"] = str(path)
        resource["available"] = True
        resource["name_to_ids"] = {key: set(values) for key, values in resource["name_to_ids"].items()}
        return resource
    except Exception as exc:
        resource["error"] = f"MONDO 资源加载失败：{exc}"
        return resource


def load_hpo_resource() -> dict[str, Any]:
    resource = {
        "path": "",
        "valid": load_index_set(HPO_INDEX_PATH, "hpo_id"),
        "obsolete": set(),
        "alt_to_id": {},
        "parents": defaultdict(set),
        "ancestors": {},
        "available": False,
        "error": "",
    }
    path = next((candidate for candidate in HPO_OBO_PATHS if candidate.is_file()), None)
    if path is None:
        resource["error"] = "无法确认，需要额外文件：HPO OBO"
        return resource
    try:
        current_id: str | None = None
        current_obsolete = False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line == "[Term]":
                    current_id = None
                    current_obsolete = False
                    continue
                if line.startswith("id: "):
                    current_id = normalize_hpo(line[4:])
                    if current_id:
                        resource["valid"].add(current_id)
                    continue
                if current_id and line.startswith("alt_id: "):
                    alt = normalize_hpo(line[8:])
                    if alt:
                        resource["alt_to_id"][alt] = current_id
                    continue
                if current_id and line.startswith("is_a: "):
                    parent = normalize_hpo(line[6:])
                    if parent:
                        resource["parents"][current_id].add(parent)
                    continue
                if current_id and line == "is_obsolete: true":
                    current_obsolete = True
                    resource["obsolete"].add(current_id)
        resource["path"] = str(path)
        resource["available"] = True
        return resource
    except Exception as exc:
        resource["error"] = f"HPO 资源加载失败：{exc}"
        return resource


def hpo_ancestors(hpo_id: str, hpo_resource: dict[str, Any]) -> set[str]:
    cache = hpo_resource.setdefault("ancestors", {})
    if hpo_id in cache:
        return cache[hpo_id]
    parents = hpo_resource.get("parents", {})
    result: set[str] = set()
    queue: deque[str] = deque(parents.get(hpo_id, set()))
    while queue:
        parent = queue.popleft()
        if parent in result:
            continue
        result.add(parent)
        queue.extend(parents.get(parent, set()))
    cache[hpo_id] = result
    return result


def load_hyperedge() -> tuple[dict[str, set[str]], dict[tuple[str, str], float], str, str]:
    path = next((candidate for candidate in HYPEREDGE_PATHS if candidate.is_file()), None)
    if path is None:
        return {}, {}, "", "无法确认，需要额外文件：disease-HPO mapping / hyperedge"
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype=str)
        else:
            df = pd.read_excel(path, dtype=str)
        lower_to_col = {col.lower(): col for col in df.columns}
        disease_col = lower_to_col.get("mondo_id") or lower_to_col.get("mondo_label") or lower_to_col.get("disease_id")
        hpo_col = lower_to_col.get("hpo_id") or lower_to_col.get("hpo")
        weight_col = lower_to_col.get("weight") or lower_to_col.get("raw_weight")
        if not disease_col or not hpo_col:
            return {}, {}, str(path), "hyperedge 文件缺少 mondo_id/hpo_id 列"
        disease_hpos: dict[str, set[str]] = defaultdict(set)
        weights: dict[tuple[str, str], float] = {}
        for row in df.itertuples(index=False):
            mondo = normalize_mondo(getattr(row, disease_col))
            hpo = normalize_hpo(getattr(row, hpo_col))
            if not mondo or not hpo:
                continue
            disease_hpos[mondo].add(hpo)
            raw_weight = getattr(row, weight_col) if weight_col else None
            try:
                weights[(mondo, hpo)] = float(raw_weight)
            except Exception:
                weights[(mondo, hpo)] = 1.0
        return dict(disease_hpos), weights, str(path), ""
    except Exception as exc:
        return {}, {}, str(path), f"hyperedge 加载失败：{exc}"


def resolve_raw_mondo_labels(
    orpha_ids: list[str],
    disease_names: list[str],
    *,
    orpha_to_mondo: dict[str, set[str]],
    orpha_to_omim: dict[str, str],
    omim_to_mondo: dict[str, set[str]],
    orpha_to_name: dict[str, str],
    mondo_resource: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    resolved: list[str] = []
    routes: list[str] = []
    unresolved: list[str] = []
    manual_overrides = {
        "ORPHA:2284": "MONDO:0003780",
        "ORPHA:3274": "MONDO:0008523",
        "ORPHA:238691": "MONDO:0002404",
        "ORPHA:564127": "MONDO:0005377",
        "ORPHA:567560": "MONDO:0018882",
        "ORPHA:686462": "MONDO:0017853",
        "ORPHA:689001": "MONDO:0006061",
        "ORPHA:95501": "MONDO:0015790",
        "ORPHA:97569": "MONDO:0002462",
    }
    for idx, orpha_id in enumerate(orpha_ids):
        hits = sorted(orpha_to_mondo.get(orpha_id, set()))
        if hits:
            resolved.extend(hits)
            routes.append(f"{orpha_id}:orpha_exact")
            continue
        omim = orpha_to_omim.get(orpha_id)
        if omim and omim_to_mondo.get(omim):
            resolved.extend(sorted(omim_to_mondo[omim]))
            routes.append(f"{orpha_id}:orpha_via_omim")
            continue
        if orpha_id in manual_overrides:
            resolved.append(manual_overrides[orpha_id])
            routes.append(f"{orpha_id}:manual_orpha_to_mondo")
            continue
        name_candidates = []
        if orpha_to_name.get(orpha_id):
            name_candidates.append(orpha_to_name[orpha_id])
        if idx < len(disease_names):
            name_candidates.append(disease_names[idx])
        name_hit = False
        name_index = mondo_resource.get("name_to_ids", {}) if mondo_resource.get("available") else {}
        for candidate_name in name_candidates:
            normalized = normalize_text(candidate_name)
            ids = sorted(name_index.get(normalized, set())) if normalized else []
            if len(ids) == 1:
                resolved.extend(ids)
                routes.append(f"{orpha_id}:name_unique")
                name_hit = True
                break
        if not name_hit:
            unresolved.append(orpha_id)
    return sorted(set(resolved)), routes, unresolved


def build_original_cases(
    original_df: pd.DataFrame,
    orpha_to_mondo: dict[str, set[str]],
    orpha_to_omim: dict[str, str],
    omim_to_mondo: dict[str, set[str]],
    orpha_to_name: dict[str, str],
    mondo_resource: dict[str, Any],
) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for idx, row in original_df.reset_index(drop=True).iterrows():
        note_id = str(row.get("note_id", "")).strip()
        orpha_ids = parse_prefixed_ids(row.get("orpha"), ORPHA_RE, normalize_orpha)
        disease_names = parse_listish(row.get("orpha_names"))
        if not disease_names and not is_missing(row.get("diagnosis")):
            disease_names = [str(row.get("diagnosis")).strip()]
        mondo_labels, routes, unresolved = resolve_raw_mondo_labels(
            orpha_ids,
            disease_names,
            orpha_to_mondo=orpha_to_mondo,
            orpha_to_omim=orpha_to_omim,
            omim_to_mondo=omim_to_mondo,
            orpha_to_name=orpha_to_name,
            mondo_resource=mondo_resource,
        )
        ids = {
            "note_id": note_id,
            "fallback_case_id": f"case_{idx + 1}",
            "icd_code": str(row.get("icd_code", "")).strip(),
            "orpha": list_to_cell(orpha_ids),
            "resolve_routes": list_to_cell(routes),
            "unresolved_orpha": list_to_cell(unresolved),
        }
        disease_ids = sorted(set(orpha_ids + parse_prefixed_ids(row.get("icd_code"), ICD_RE, lambda value: str(value).strip())))
        cases.append(
            CaseRecord(
                dataset="original",
                case_key=note_id or f"original_row_{idx + 1}",
                raw_index=idx + 1,
                ids=ids,
                disease_names=disease_names,
                disease_ids=disease_ids,
                mondo_labels=mondo_labels,
                hpo_ids=parse_hpo_values(row.get("HPO")),
                text_hash=sha1_text(row.get("text")),
                text_preview=preview(row.get("text")),
                row_count=1,
                source_note="; ".join(
                    part for part in [
                        f"diagnosis={preview(row.get('diagnosis'), 80)}",
                        f"rare={preview(row.get('rare'), 80)}",
                    ] if part
                ),
            )
        )
    return cases


def build_cleaned_cases(cleaned_df: pd.DataFrame) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for case_id, group in cleaned_df.groupby("case_id", sort=False):
        labels = sorted(set(label for value in group["mondo_label"].tolist() if (label := normalize_mondo(value))))
        hpos = sorted(set(hpo for value in group["hpo_id"].tolist() if (hpo := normalize_hpo(value))))
        cases.append(
            CaseRecord(
                dataset="cleaned",
                case_key=str(case_id).strip(),
                raw_index=None,
                ids={"case_id": str(case_id).strip()},
                disease_names=[],
                disease_ids=labels,
                mondo_labels=labels,
                hpo_ids=hpos,
                text_hash="",
                text_preview="",
                row_count=int(len(group)),
                source_note="long format grouped by case_id",
            )
        )
    return cases


def case_signature(case: CaseRecord) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(sorted(case.mondo_labels)), tuple(sorted(case.hpo_ids))


def build_case_matches(original_cases: list[CaseRecord], cleaned_cases: list[CaseRecord]) -> tuple[pd.DataFrame, dict[str, str]]:
    cleaned_by_case_id = {case.case_key: case for case in cleaned_cases}
    cleaned_signature_map: dict[tuple[tuple[str, ...], tuple[str, ...]], list[CaseRecord]] = defaultdict(list)
    for case in cleaned_cases:
        cleaned_signature_map[case_signature(case)].append(case)

    used_cleaned: set[str] = set()
    rows: list[dict[str, Any]] = []
    original_to_cleaned: dict[str, str] = {}

    def add_row(issue_type: str, original: CaseRecord | None, cleaned: CaseRecord | None, method: str, reason: str) -> None:
        original_hpos = set(original.hpo_ids) if original else set()
        cleaned_hpos = set(cleaned.hpo_ids) if cleaned else set()
        row = {
            "issue_type": issue_type,
            "match_method": method,
            "original_id": original.case_key if original else "",
            "cleaned_id": cleaned.case_key if cleaned else "",
            "original_note_id": original.ids.get("note_id", "") if original else "",
            "cleaned_case_id": cleaned.ids.get("case_id", "") if cleaned else "",
            "original_label": list_to_cell(original.mondo_labels or original.disease_names) if original else "",
            "cleaned_label": list_to_cell(cleaned.mondo_labels) if cleaned else "",
            "original_hpo_count": len(original_hpos),
            "cleaned_hpo_count": len(cleaned_hpos),
            "hpo_jaccard": jaccard(original_hpos, cleaned_hpos),
            "original_ids": json.dumps(original.ids, ensure_ascii=False) if original else "",
            "cleaned_ids": json.dumps(cleaned.ids, ensure_ascii=False) if cleaned else "",
            "reason": reason,
        }
        rows.append(row)
        if original and cleaned:
            original_to_cleaned[original.case_key] = cleaned.case_key
            used_cleaned.add(cleaned.case_key)

    for original in original_cases:
        direct_candidates = [
            original.case_key,
            original.ids.get("note_id", ""),
            original.ids.get("fallback_case_id", ""),
        ]
        matched: CaseRecord | None = None
        method = ""
        reason = ""
        for candidate in direct_candidates[:2]:
            if candidate and candidate in cleaned_by_case_id and candidate not in used_cleaned:
                matched = cleaned_by_case_id[candidate]
                method = "direct_id"
                reason = "case_id/note_id 完全相同"
                break
        if matched is None:
            signature = case_signature(original)
            candidates = [case for case in cleaned_signature_map.get(signature, []) if case.case_key not in used_cleaned]
            if len(candidates) == 1:
                matched = candidates[0]
                method = "label_hpo_signature"
                reason = "MONDO label set + HPO set 完全一致"
            elif len(candidates) > 1:
                matched = candidates[0]
                method = "ambiguous_label_hpo_signature"
                reason = f"MONDO label set + HPO set 有 {len(candidates)} 个候选，按未使用候选顺序匹配"
        if matched is None:
            fallback_id = original.ids.get("fallback_case_id", "")
            fallback = cleaned_by_case_id.get(fallback_id)
            if fallback and fallback.case_key not in used_cleaned:
                overlap = jaccard(set(original.hpo_ids), set(fallback.hpo_ids))
                same_label = set(original.mondo_labels) == set(fallback.mondo_labels)
                if same_label and overlap >= 0.8:
                    matched = fallback
                    method = "row_number_fallback"
                    reason = "仅因 case_数字 与原始行号一致且 label/HPO 高度一致而接受；不是原始显式 ID"
        if matched is None:
            add_row("missing_in_cleaned", original, None, "unmatched", "cleaned 中找不到同 ID 或同 label+HPO signature 病例")
        else:
            issue = "matched"
            if set(original.mondo_labels) != set(matched.mondo_labels):
                issue = "label_mismatch"
            elif set(original.hpo_ids) != set(matched.hpo_ids):
                issue = "hpo_mismatch"
            add_row(issue, original, matched, method, reason)

    for cleaned in cleaned_cases:
        if cleaned.case_key not in used_cleaned:
            add_row("extra_in_cleaned", None, cleaned, "unmatched", "original 中找不到同 ID 或同 label+HPO signature 病例")

    return pd.DataFrame(rows), original_to_cleaned


def audit_schema(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for dataset_name, df in [("original", original_df), ("cleaned", cleaned_df)]:
        id_candidates = [col for col in df.columns if col.lower() in {"case_id", "patient_id", "subject_id", "hadm_id", "note_id"}]
        case_col = "case_id" if "case_id" in df.columns else "note_id" if "note_id" in df.columns else (id_candidates[0] if id_candidates else "")
        for column in df.columns:
            sample_values = [preview(value, 100) for value in df[column].dropna().head(5).tolist()]
            rows.append(
                {
                    "dataset": dataset_name,
                    "row_count": int(len(df)),
                    "unique_cases": int(df[case_col].nunique(dropna=True)) if case_col else "",
                    "case_id_basis": case_col or "无法确认",
                    "column": column,
                    "inferred_meaning": infer_column_role(column, sample_values),
                    "null_ratio": float(df[column].isna().mean()),
                    "unique_values": int(df[column].nunique(dropna=True)),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "format_guess": infer_cell_format(df[column]),
                    "sample_values": json.dumps(sample_values, ensure_ascii=False),
                    "has_split_field": "split" in {col.lower() for col in df.columns},
                }
            )
    schema_df = pd.DataFrame(rows)
    write_csv(schema_df, output_dir / "mimic_schema_audit.csv")

    lines = ["# mimic_test schema audit", ""]
    for dataset_name, df in [("original", original_df), ("cleaned", cleaned_df)]:
        case_col = "case_id" if "case_id" in df.columns else "note_id" if "note_id" in df.columns else ""
        lines.extend(
            [
                f"## {dataset_name}",
                f"- 行数: {len(df)}",
                f"- 唯一病例数: {df[case_col].nunique(dropna=True) if case_col else '无法确认'}",
                f"- 病例 ID 依据: `{case_col}`" if case_col else "- 病例 ID 依据: 无法确认",
                f"- 列名: {', '.join(f'`{col}`' for col in df.columns)}",
                f"- split 字段: {'有' if 'split' in {col.lower() for col in df.columns} else '无'}",
                f"- 重复行数量: {int(df.duplicated().sum())}",
                "",
                "| column | inferred meaning | null_ratio | format_guess |",
                "|---|---:|---:|---|",
            ]
        )
        for _, row in schema_df.loc[schema_df["dataset"] == dataset_name].iterrows():
            lines.append(f"| `{row['column']}` | {row['inferred_meaning']} | {row['null_ratio']:.4f} | {row['format_guess']} |")
        lines.append("")
    write_text(output_dir / "mimic_schema_audit.md", "\n".join(lines))


def build_duplicate_report(cases: list[CaseRecord], dataset_name: str) -> pd.DataFrame:
    sig_counter = Counter(case_signature(case) for case in cases)
    rows = []
    for case in cases:
        sig = case_signature(case)
        if sig_counter[sig] > 1:
            rows.append(
                {
                    "dataset": dataset_name,
                    "issue_type": "duplicate_label_hpo_signature",
                    "case_id": case.case_key,
                    "note_id": case.ids.get("note_id", ""),
                    "label": list_to_cell(case.mondo_labels),
                    "hpo_count": len(case.hpo_ids),
                    "duplicate_signature_count": sig_counter[sig],
                    "reason": "同一 label set + HPO set 出现多次；需人工判断是否真实重复病例或同病同表型复现",
                }
            )
    return pd.DataFrame(rows)


def audit_case_granularity(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    original_cases: list[CaseRecord],
    cleaned_cases: list[CaseRecord],
    matching_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    missing_df = matching_df.loc[matching_df["issue_type"] == "missing_in_cleaned"].copy()
    duplicates = pd.concat(
        [build_duplicate_report(original_cases, "original"), build_duplicate_report(cleaned_cases, "cleaned")],
        ignore_index=True,
    )
    if duplicates.empty:
        duplicates = pd.DataFrame(
            columns=[
                "dataset",
                "issue_type",
                "case_id",
                "note_id",
                "label",
                "hpo_count",
                "duplicate_signature_count",
                "reason",
            ]
        )
    write_csv(matching_df, output_dir / "mimic_case_matching.csv")
    write_csv(missing_df, output_dir / "mimic_missing_cases.csv")
    write_csv(duplicates, output_dir / "mimic_duplicate_cases.csv")

    original_note_unique = "note_id" in original_df.columns and original_df["note_id"].nunique(dropna=True) == len(original_df)
    cleaned_case_count = cleaned_df["case_id"].nunique(dropna=True) if "case_id" in cleaned_df.columns else 0
    original_rows = len(original_df)
    cleaned_rows = len(cleaned_df)
    lines = [
        "# mimic_test case granularity audit",
        "",
        f"- original rows / unique cases: {original_rows} / {len(original_cases)}",
        f"- cleaned rows / unique cases: {cleaned_rows} / {cleaned_case_count}",
        f"- original `note_id` 是否唯一: {'是' if original_note_unique else '否或无法确认'}",
        f"- cleaned 病例粒度: `case_id` 聚合后的病例级长表；每个病例可有多个 `mondo_label` 和多个 `hpo_id`。",
        f"- original 病例粒度判断: {'note-level' if original_note_unique else 'unclear'}。没有 `subject_id` / `hadm_id` / `patient_id`，无法确认 patient-level 或 admission-level。",
        f"- cleaned 病例粒度判断: case-level long format；来源是否严格 note-level 无法仅凭 cleaned 确认。",
        f"- matched cases: {int((matching_df['issue_type'] == 'matched').sum())}",
        f"- label mismatch cases: {int((matching_df['issue_type'] == 'label_mismatch').sum())}",
        f"- HPO mismatch cases: {int((matching_df['issue_type'] == 'hpo_mismatch').sum())}",
        f"- missing in cleaned: {len(missing_df)}",
        f"- extra in cleaned: {int((matching_df['issue_type'] == 'extra_in_cleaned').sum())}",
        "",
        "## 关键判断",
        "",
        "- original 是 1875 行且 1875 个唯一 `note_id`。",
        f"- cleaned 是 {cleaned_rows} 行长表，按 `case_id` 聚合后是 {cleaned_case_count} 个病例。",
        "- cleaned 之所以是 1873，不是因为长表行数，而是因为 `case_1` 到 `case_1873` 只有 1873 个唯一病例。",
        "- 是否存在一个 patient 多个 admission、一个 admission 多个 note：无法确认，需要包含 `subject_id` / `hadm_id` / patient-level ID 的源文件。",
        "- 是否存在一个 note 多个 disease label：original 可由 `orpha` 列判断，cleaned 可由同一 `case_id` 下多个 `mondo_label` 判断。",
        "- 是否存在原始多个病例被合并或一个病例被拆分：本脚本用 `label set + HPO set` 签名和重复签名表辅助定位，最终仍需结合原始 MIMIC subject/hadm/note 映射人工确认。",
        "",
    ]
    if not missing_df.empty:
        lines.append("## cleaned 缺失病例")
        lines.append("")
        for _, row in missing_df.head(20).iterrows():
            lines.append(
                f"- original_id={row['original_id']}, note_id={row['original_note_id']}, "
                f"label={row['original_label']}, hpo_count={row['original_hpo_count']}, reason={row['reason']}"
            )
        lines.append("")
    write_text(output_dir / "mimic_case_granularity.md", "\n".join(lines))


def mondo_parent_child(left: set[str], right: set[str], mondo_resource: dict[str, Any]) -> bool:
    parents = mondo_resource.get("parents", {})
    children = mondo_resource.get("children", {})
    for a in left:
        if parents.get(a, set()) & right or children.get(a, set()) & right:
            return True
    return False


def classify_label_match(original: CaseRecord | None, cleaned: CaseRecord | None, mondo_resource: dict[str, Any]) -> tuple[str, str, str]:
    if original is None:
        return "extra_in_cleaned", "cleaned 中多出病例/标签", "critical"
    if cleaned is None:
        return "missing_in_cleaned", "cleaned 缺失 original 病例/标签", "critical"
    left = set(original.mondo_labels)
    right = set(cleaned.mondo_labels)
    if left == right and left:
        return "exact_same", "", "low"
    if not left:
        return "uncertain", "original ORPHA/名称未能映射到 MONDO，无法确认 label 是否一致", "high"
    if not right:
        return "missing_in_cleaned", "cleaned 缺少 disease label", "critical"
    if mondo_resource.get("available") and mondo_parent_child(left, right, mondo_resource):
        return "parent_child", "original 与 cleaned 标签存在 MONDO 直接父子关系，exact evaluation 会视为不同", "high"
    return "id_mismatch", "original 解析出的 MONDO set 与 cleaned MONDO set 不一致", "critical"


def audit_labels(
    original_cases: list[CaseRecord],
    cleaned_cases: list[CaseRecord],
    matching_df: pd.DataFrame,
    output_dir: Path,
    mondo_resource: dict[str, Any],
) -> None:
    original_map = {case.case_key: case for case in original_cases}
    cleaned_map = {case.case_key: case for case in cleaned_cases}
    rows: list[dict[str, Any]] = []
    for _, match in matching_df.iterrows():
        original = original_map.get(str(match["original_id"]))
        cleaned = cleaned_map.get(str(match["cleaned_id"]))
        match_type, problem, severity = classify_label_match(original, cleaned, mondo_resource)
        if cleaned:
            obsolete_labels = sorted(set(cleaned.mondo_labels) & set(mondo_resource.get("obsolete", set())))
            if obsolete_labels:
                match_type = "id_mismatch"
                problem = f"cleaned 使用 obsolete MONDO: {list_to_cell(obsolete_labels)}"
                severity = "critical"
        rows.append(
            {
                "case_key": original.case_key if original else cleaned.case_key if cleaned else "",
                "cleaned_case_id": cleaned.case_key if cleaned else "",
                "original_note_id": original.ids.get("note_id", "") if original else "",
                "original_disease": list_to_cell(original.disease_names) if original else "",
                "original_ids": json.dumps(original.ids, ensure_ascii=False) if original else "",
                "original_mondo_labels": list_to_cell(original.mondo_labels) if original else "",
                "cleaned_disease": list_to_cell(cleaned.mondo_labels) if cleaned else "",
                "cleaned_ids": list_to_cell(cleaned.mondo_labels) if cleaned else "",
                "match_type": match_type,
                "problem": problem,
                "severity": severity,
            }
        )
    label_df = pd.DataFrame(rows)
    write_csv(label_df, output_dir / "mimic_label_diff.csv")

    summary = label_df.groupby(["match_type", "severity"], dropna=False).size().reset_index(name="case_count")
    lines = [
        "# mimic_test label / disease ID audit",
        "",
        f"- MONDO 资源: {mondo_resource.get('path') or mondo_resource.get('error') or '无法确认，需要额外文件'}",
        f"- exact_same: {int((label_df['match_type'] == 'exact_same').sum())}",
        f"- id_mismatch: {int((label_df['match_type'] == 'id_mismatch').sum())}",
        f"- missing_in_cleaned: {int((label_df['match_type'] == 'missing_in_cleaned').sum())}",
        f"- extra_in_cleaned: {int((label_df['match_type'] == 'extra_in_cleaned').sum())}",
        f"- uncertain: {int((label_df['match_type'] == 'uncertain').sum())}",
        "",
        "## 分布",
        "",
        df_to_markdown(summary),
        "",
        "## 审计说明",
        "",
        "- original 的 disease ID 主要来自 `orpha`，本脚本使用本地 `mondo_hasdbxref_orphanet.sssom.tsv`、`orpha2omim.json`、`mondo_exactmatch_omim.sssom.tsv` 和少量项目内既有 manual override 映射到 MONDO。",
        "- ICD 粗粒度字段存在于 original `icd_code`，但 cleaned gold label 使用 `mondo_label`；本脚本没有把 ICD 直接当作 exact gold。",
        "- OMIM gene ID 与 phenotype ID、ORPHA group ID 与 disorder ID 是否混淆：仅凭当前 CSV 无法完全确认，需要原始 Orphanet/OMIM 语义层级文件逐项人工核对。",
        "- synonym、obsolete、parent-child 只在本地 MONDO JSON 可加载时判断；无法确认的项目会标为 `uncertain`。",
    ]
    write_text(output_dir / "mimic_label_summary.md", "\n".join(lines))


def audit_multilabel(
    original_cases: list[CaseRecord],
    cleaned_cases: list[CaseRecord],
    matching_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    original_map = {case.case_key: case for case in original_cases}
    cleaned_map = {case.case_key: case for case in cleaned_cases}
    rows: list[dict[str, Any]] = []
    for _, match in matching_df.iterrows():
        original = original_map.get(str(match["original_id"]))
        cleaned = cleaned_map.get(str(match["cleaned_id"]))
        original_labels = set(original.mondo_labels) if original else set()
        cleaned_labels = set(cleaned.mondo_labels) if cleaned else set()
        lost = original_labels - cleaned_labels
        added = cleaned_labels - original_labels
        impact = "none"
        if lost:
            impact = "会低估 exact evaluation；gold label 被 cleaned 丢失"
        elif added:
            impact = "可能改变 any-label 或 relaxed evaluation 口径"
        elif len(original_labels) > 1:
            impact = "exact 只取一个 gold 时可能低估；any-label@k 可作为 supplementary"
        rows.append(
            {
                "case_key": original.case_key if original else cleaned.case_key if cleaned else "",
                "cleaned_case_id": cleaned.case_key if cleaned else "",
                "original_note_id": original.ids.get("note_id", "") if original else "",
                "original_labels": list_to_cell(original_labels),
                "cleaned_labels": list_to_cell(cleaned_labels),
                "lost_labels": list_to_cell(lost),
                "added_labels": list_to_cell(added),
                "impact_on_exact_eval": impact,
            }
        )
    diff_df = pd.DataFrame(rows)
    write_csv(diff_df, output_dir / "mimic_multilabel_diff.csv")

    original_multi = sum(len(case.mondo_labels) > 1 for case in original_cases)
    cleaned_multi = sum(len(case.mondo_labels) > 1 for case in cleaned_cases)
    compressed = int(((diff_df["original_labels"].str.contains(r"\|", regex=True)) & ~diff_df["cleaned_labels"].str.contains(r"\|", regex=True, na=False)).sum())
    lines = [
        "# mimic_test multilabel audit",
        "",
        f"- original single-label cases: {sum(len(case.mondo_labels) == 1 for case in original_cases)}",
        f"- original multi-label cases: {original_multi}",
        f"- original unresolved-label cases: {sum(len(case.mondo_labels) == 0 for case in original_cases)}",
        f"- cleaned single-label cases: {sum(len(case.mondo_labels) == 1 for case in cleaned_cases)}",
        f"- cleaned multi-label cases: {cleaned_multi}",
        f"- original multi-label 但 cleaned 变 single-label 的病例数: {compressed}",
        f"- original 多标签中被 cleaned 丢掉 gold labels 的病例数: {nonempty_count(diff_df['lost_labels'])}",
        f"- cleaned 中新增 labels 的病例数: {nonempty_count(diff_df['added_labels'])}",
        "",
        "## 判断",
        "",
        "- 当前 exact top1/top3/top5 可能被多标签处理低估，前提是 evaluator 只取每个 `case_id` 的第一个 `mondo_label` 作为唯一 gold；这与已有 any-label@5 明显高于 multi-label exact top5 的现象一致。",
        "- any-label@k 应只作为 supplementary，不能替代正式 exact evaluation。",
        "- 建议同时保留 original exact、cleaned exact、any-label 三种口径，并在报告中明确 gold 选择规则。",
    ]
    write_text(output_dir / "mimic_multilabel_summary.md", "\n".join(lines))


def audit_hpo(
    original_cases: list[CaseRecord],
    cleaned_cases: list[CaseRecord],
    matching_df: pd.DataFrame,
    output_dir: Path,
    hpo_resource: dict[str, Any],
) -> pd.DataFrame:
    original_map = {case.case_key: case for case in original_cases}
    cleaned_map = {case.case_key: case for case in cleaned_cases}
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    valid_hpos = set(hpo_resource.get("valid", set()))
    obsolete_hpos = set(hpo_resource.get("obsolete", set()))
    for _, match in matching_df.iterrows():
        original = original_map.get(str(match["original_id"]))
        cleaned = cleaned_map.get(str(match["cleaned_id"]))
        original_hpos = set(original.hpo_ids) if original else set()
        cleaned_hpos = set(cleaned.hpo_ids) if cleaned else set()
        invalid = sorted(hpo for hpo in (original_hpos | cleaned_hpos) if not HPO_RE.fullmatch(hpo) or (valid_hpos and hpo not in valid_hpos))
        obsolete = sorted((original_hpos | cleaned_hpos) & obsolete_hpos)
        issue_parts = []
        if invalid:
            issue_parts.append("invalid_hpo")
        if obsolete:
            issue_parts.append("obsolete_hpo")
        if original and cleaned and len(cleaned_hpos) < len(original_hpos) * 0.7:
            issue_parts.append("large_hpo_loss")
        if cleaned and len(cleaned_hpos) == 0:
            issue_parts.append("zero_hpo")
        if cleaned and len(cleaned_hpos) >= 30:
            issue_parts.append("very_many_hpo")
        rows.append(
            {
                "case_key": original.case_key if original else cleaned.case_key if cleaned else "",
                "cleaned_case_id": cleaned.case_key if cleaned else "",
                "original_note_id": original.ids.get("note_id", "") if original else "",
                "original_hpo_count": len(original_hpos),
                "cleaned_hpo_count": len(cleaned_hpos),
                "jaccard": jaccard(original_hpos, cleaned_hpos),
                "lost_hpo": list_to_cell(original_hpos - cleaned_hpos),
                "added_hpo": list_to_cell(cleaned_hpos - original_hpos),
                "invalid_hpo": list_to_cell(invalid),
                "obsolete_hpo": list_to_cell(obsolete),
                "suspicious_issue": "|".join(issue_parts),
            }
        )
        for hpo in invalid:
            invalid_rows.append(
                {
                    "case_key": original.case_key if original else cleaned.case_key if cleaned else "",
                    "cleaned_case_id": cleaned.case_key if cleaned else "",
                    "hpo_id": hpo,
                    "issue": "invalid_format_or_not_in_HPO_index",
                }
            )
        for hpo in obsolete:
            invalid_rows.append(
                {
                    "case_key": original.case_key if original else cleaned.case_key if cleaned else "",
                    "cleaned_case_id": cleaned.case_key if cleaned else "",
                    "hpo_id": hpo,
                    "issue": "obsolete_hpo",
                }
            )
    hpo_df = pd.DataFrame(rows)
    write_csv(hpo_df, output_dir / "mimic_hpo_diff.csv")
    write_csv(pd.DataFrame(invalid_rows), output_dir / "mimic_invalid_hpo.csv")

    cleaned_counts = [len(case.hpo_ids) for case in cleaned_cases]
    original_counts = [len(case.hpo_ids) for case in original_cases]
    jaccards = hpo_df["jaccard"].dropna().astype(float).tolist()
    cleaned_hpo_freq = Counter(hpo for case in cleaned_cases for hpo in case.hpo_ids)
    lost_freq = Counter(hpo for value in hpo_df["lost_hpo"].astype(str) for hpo in value.split("|") if hpo)
    added_freq = Counter(hpo for value in hpo_df["added_hpo"].astype(str) for hpo in value.split("|") if hpo)
    lines = [
        "# mimic_test HPO audit",
        "",
        f"- HPO 资源: {hpo_resource.get('path') or hpo_resource.get('error') or 'HPO_index_v4.xlsx only'}",
        f"- original 平均 HPO 数: {np.mean(original_counts):.2f}",
        f"- original 中位 HPO 数: {median(original_counts):.2f}",
        f"- cleaned 平均 HPO 数: {np.mean(cleaned_counts):.2f}",
        f"- cleaned 中位 HPO 数: {median(cleaned_counts):.2f}",
        f"- cleaned HPO 数为 0 的病例数: {sum(count == 0 for count in cleaned_counts)}",
        f"- cleaned HPO 数 <= 1 的病例数: {sum(count <= 1 for count in cleaned_counts)}",
        f"- cleaned HPO 数 >= 30 的病例数: {sum(count >= 30 for count in cleaned_counts)}",
        f"- Jaccard min/median/mean/max: {min(jaccards):.4f} / {median(jaccards):.4f} / {np.mean(jaccards):.4f} / {max(jaccards):.4f}",
        f"- invalid/obsolete HPO rows: {len(invalid_rows)}",
        "",
        "## 高频 HPO top50",
        "",
        df_to_markdown(pd.DataFrame(cleaned_hpo_freq.most_common(50), columns=["hpo_id", "count"])),
        "",
        "## 被删除最多 HPO top50",
        "",
        df_to_markdown(pd.DataFrame(lost_freq.most_common(50), columns=["hpo_id", "count"])),
        "",
        "## 新增最多 HPO top50",
        "",
        df_to_markdown(pd.DataFrame(added_freq.most_common(50), columns=["hpo_id", "count"])),
        "",
        "## 无法确认项",
        "",
        "- 是否误删高信息量 HPO、加入过泛 HPO、误加入否定症状、家族史、检查项目或治疗反应：仅凭 `HPO` ID 列无法完全判断，需要 HPO 术语名称、原始抽取证据 spans 或人工标注说明。",
    ]
    write_text(output_dir / "mimic_hpo_summary.md", "\n".join(lines))
    return hpo_df


def semantic_overlap(case_hpos: set[str], disease_hpos: set[str], hpo_resource: dict[str, Any]) -> float | str:
    if not hpo_resource.get("available"):
        return "无法确认，需要额外文件"
    if not case_hpos:
        return 0.0
    hits = 0
    for hpo in case_hpos:
        if hpo in disease_hpos:
            hits += 1
            continue
        ancestors = hpo_ancestors(hpo, hpo_resource)
        if ancestors & disease_hpos:
            hits += 1
    return hits / len(case_hpos)


def audit_coverage(
    cleaned_cases: list[CaseRecord],
    output_dir: Path,
    disease_hpos: dict[str, set[str]],
    disease_weights: dict[tuple[str, str], float],
    hyperedge_path: str,
    hyperedge_error: str,
    hpo_resource: dict[str, Any],
    mondo_resource: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    disease_index = load_index_set(DISEASE_INDEX_PATH, "mondo_id")
    obsolete = set(mondo_resource.get("obsolete", set()))
    for case in cleaned_cases:
        case_hpos = set(case.hpo_ids)
        for label in case.mondo_labels or [""]:
            gold_hpos = set(disease_hpos.get(label, set()))
            shared = case_hpos & gold_hpos
            denom = math.sqrt(len(case_hpos) * len(gold_hpos)) if case_hpos and gold_hpos else 0.0
            total_weight = sum(float(disease_weights.get((label, hpo), 1.0)) for hpo in gold_hpos)
            shared_weight = sum(float(disease_weights.get((label, hpo), 1.0)) for hpo in shared)
            rows.append(
                {
                    "case_key": case.case_key,
                    "gold_label": label,
                    "gold_label_count": len(case.mondo_labels),
                    "case_hpo_count": len(case_hpos),
                    "gold_disease_hpo_count": len(gold_hpos),
                    "exact_overlap": (len(shared) / denom) if denom else 0.0,
                    "shared_hpo_count": len(shared),
                    "shared_hpo": list_to_cell(shared),
                    "overlap_zero": int(len(shared) == 0),
                    "overlap_le_1": int(len(shared) <= 1),
                    "ic_weighted_overlap": (shared_weight / total_weight) if total_weight else "无法确认，需要额外文件",
                    "semantic_overlap": semantic_overlap(case_hpos, gold_hpos, hpo_resource),
                    "seen_label": int(label in disease_index or label in disease_hpos),
                    "unmapped_label": int(label not in disease_index and label not in disease_hpos),
                    "obsolete_label": int(label in obsolete),
                }
            )
    coverage_df = pd.DataFrame(rows)
    write_csv(coverage_df, output_dir / "mimic_gold_hpo_coverage.csv")
    zero_df = coverage_df.loc[coverage_df["overlap_zero"] == 1].copy()
    write_csv(zero_df, output_dir / "mimic_overlap_zero_cases.csv")

    case_level_zero = coverage_df.groupby("case_key")["overlap_zero"].min()
    mapped = coverage_df.loc[coverage_df["unmapped_label"] == 0].copy()
    lines = [
        "# mimic_test gold disease - HPO coverage audit",
        "",
        f"- disease-HPO resource: {hyperedge_path or hyperedge_error or '无法确认，需要额外文件'}",
        f"- cleaned case count: {len(cleaned_cases)}",
        f"- case-label rows: {len(coverage_df)}",
        f"- unmapped label rows: {int(coverage_df['unmapped_label'].sum())}",
        f"- obsolete label rows: {int(coverage_df['obsolete_label'].sum())}",
        f"- overlap_zero case-label rows: {int(coverage_df['overlap_zero'].sum())}",
        f"- overlap_zero case rate（任一 gold label 有重叠即不算 zero）: {float(case_level_zero.mean()):.4f}",
        f"- mean exact_overlap(mapped rows): {float(mapped['exact_overlap'].mean()) if not mapped.empty else float('nan'):.4f}",
        "",
        "## 判断",
        "",
        "- 若 original 与 cleaned 的 HPO Jaccard 接近 1 且 lost_hpo 很少，则 overlap_zero 高不主要由 cleaned HPO 丢失导致。",
        "- 若 `unmapped_label` 或 `obsolete_label` 很少，则 overlap_zero 不主要由 cleaned disease label 映射失败导致。",
        "- 若 gold disease hyperedge 本身 HPO 数很少或为 0，则更像 disease-HPO knowledge base 覆盖不足。",
        "- 对仍然 overlap_zero 的病例，需要回看原始 `text` 和 HPO 抽取证据，判断是否临床文本本身缺少典型表型或 HPO 抽取噪声。",
    ]
    write_text(output_dir / "mimic_gold_hpo_coverage_summary.md", "\n".join(lines))
    return coverage_df


def discover_split_files() -> list[tuple[str, Path]]:
    roots = [
        ("train", PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "train"),
        ("val", PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "val"),
        ("validation", PROJECT_ROOT / "LLLdataset" / "dataset" / "processed" / "validation"),
        ("aux_processed", PROJECT_ROOT / "LLLdataset" / "dataset" / "processed"),
    ]
    files: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for split, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*")):
            if path.name.startswith("~$") or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue
            if split == "aux_processed" and path.stem.lower() in {"mimic_test", "ddd_test", "total"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append((split, path))
    return files


def table_to_case_signatures(path: Path) -> pd.DataFrame:
    df = read_table(path)
    lower_cols = {col.lower(): col for col in df.columns}
    case_col = lower_cols.get("case_id")
    label_col = lower_cols.get("mondo_label") or lower_cols.get("mondo_id")
    hpo_col = lower_cols.get("hpo_id") or lower_cols.get("hpo")
    if not case_col or not label_col or not hpo_col:
        return pd.DataFrame()
    rows = []
    for case_id, group in df.groupby(case_col, sort=False):
        labels = sorted(set(label for value in group[label_col].tolist() if (label := normalize_mondo(value))))
        hpos = sorted(set(hpo for value in group[hpo_col].tolist() if (hpo := normalize_hpo(value))))
        rows.append(
            {
                "case_id": str(case_id),
                "label_signature": list_to_cell(labels),
                "hpo_signature": list_to_cell(hpos),
                "label_hpo_signature": json.dumps([labels, hpos], ensure_ascii=False),
                "hpo_count": len(hpos),
                "label_count": len(labels),
            }
        )
    return pd.DataFrame(rows)


def audit_split_leakage(cleaned_path: Path, output_dir: Path) -> pd.DataFrame:
    split_files = discover_split_files()
    target = table_to_case_signatures(cleaned_path)
    target["target_case_id"] = target["case_id"]
    target_signatures = set(target["label_hpo_signature"])
    target_case_ids = set(target["case_id"])
    target_hpo_label = set(zip(target["label_signature"], target["hpo_signature"]))
    rows: list[dict[str, Any]] = []
    for split, path in split_files:
        sig_df = table_to_case_signatures(path)
        if sig_df.empty:
            continue
        is_self = path.resolve() == cleaned_path.resolve()
        for row in sig_df.itertuples(index=False):
            issue_types = []
            same_dataset_stem = path.stem.lower() == cleaned_path.stem.lower()
            if not is_self and same_dataset_stem and str(row.case_id) in target_case_ids:
                issue_types.append("raw_case_id_overlap")
            if not is_self and row.label_hpo_signature in target_signatures:
                issue_types.append("same_label_hpo_signature")
            if not is_self and (row.label_signature, row.hpo_signature) in target_hpo_label:
                issue_types.append("same_hpo_set_same_label")
            if is_self:
                issue_types.append("self_test_file")
            if issue_types and not is_self:
                rows.append(
                    {
                        "issue_type": "|".join(sorted(set(issue_types))),
                        "split": split,
                        "file": str(path),
                        "case_id": row.case_id,
                        "label_signature": row.label_signature,
                        "hpo_count": row.hpo_count,
                        "reason": "与 cleaned mimic_test 存在同数据集 raw case_id 或 label+HPO signature 重叠；需确认是否真实泄漏或合成/相似病例库重复。不同数据集的 case_数字碰撞不计为 leakage。",
                    }
                )
    leakage_df = pd.DataFrame(rows)
    if leakage_df.empty:
        leakage_df = pd.DataFrame(
            columns=["issue_type", "split", "file", "case_id", "label_signature", "hpo_count", "reason"]
        )
    write_csv(leakage_df, output_dir / "mimic_split_leakage.csv")
    lines = [
        "# mimic_test split / leakage audit",
        "",
        f"- 搜索到 split/case 文件数: {len(split_files)}",
        f"- leakage/collision rows: {len(leakage_df)}",
        f"- cleaned mimic_test 是否混入 train/val: {'发现潜在重叠，见 CSV' if len(leakage_df) else '未发现基于 case_id 或 label+HPO signature 的直接重叠'}",
        "",
        "## 无法确认项",
        "",
        "- 同一个 patient 是否跨 split、同一个 `subject_id` / `hadm_id` / `note_id` 是否跨 split：processed train/test 表缺少这些字段，无法确认，需要包含 MIMIC patient/admission/note 映射的源文件。",
        "- similar-case library 是否包含 mimic_test 的重复 case：本脚本检查了 processed 根目录下的 `mimic_rag_0425.csv` 等表的 label+HPO signature 重叠；是否文本级重复仍需要原始 clinical text 或 text hash 库。",
        "",
    ]
    if not leakage_df.empty:
        lines.extend(["## 潜在重叠摘要", "", df_to_markdown(leakage_df.groupby(["split", "issue_type"]).size().reset_index(name="count")), ""])
    write_text(output_dir / "mimic_split_leakage.md", "\n".join(lines))
    return leakage_df


def build_accuracy_impact(
    label_df: pd.DataFrame,
    multilabel_df: pd.DataFrame,
    hpo_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    critical_label = int(label_df["severity"].eq("critical").sum())
    lost_label_cases = nonempty_count(multilabel_df["lost_labels"])
    hpo_loss_cases = int(hpo_df["suspicious_issue"].astype(str).str.contains("large_hpo_loss", na=False).sum())
    invalid_hpo_cases = nonempty_count(hpo_df["invalid_hpo"]) + nonempty_count(hpo_df["obsolete_hpo"])
    overlap_zero_cases = int(coverage_df.groupby("case_key")["overlap_zero"].min().sum()) if not coverage_df.empty else 0
    unmapped_rows = int(coverage_df["unmapped_label"].sum()) if not coverage_df.empty else 0
    obsolete_label_rows = int(coverage_df["obsolete_label"].sum()) if not coverage_df.empty else 0
    leakage_rows = len(leakage_df)
    rows.extend(
        [
            {
                "issue": "label ID 错误 / missing case / cleaned 多出病例",
                "affected_cases": critical_label,
                "expected_accuracy_impact": "critical: 会直接改变 exact gold，可能造成 top1/top3/top5 被错误计算",
                "fix_priority": "A",
                "recommended_fix": "先修复映射和病例对应关系；保留原始 exact 口径，不删除困难样本",
            },
            {
                "issue": "多标签 gold 在 exact evaluator 中只取首个标签",
                "affected_cases": pipe_count(multilabel_df["original_labels"]),
                "expected_accuracy_impact": "high: exact top-k 可能低估，any-label@k 明显更高时尤其需要报告 supplementary",
                "fix_priority": "A/C",
                "recommended_fix": "保留 cleaned 多标签；同时报告 original exact、cleaned exact、any-label，不把 any-label 当正式 exact",
            },
            {
                "issue": "cleaned 丢失 original gold labels",
                "affected_cases": lost_label_cases,
                "expected_accuracy_impact": "critical: gold label 丢失会让 exact evaluation 不可信",
                "fix_priority": "A",
                "recommended_fix": "恢复丢失 label 或明确剔除原因；不可为提高准确率删除测试样本",
            },
            {
                "issue": "HPO 大量丢失或 invalid/obsolete HPO",
                "affected_cases": hpo_loss_cases + invalid_hpo_cases,
                "expected_accuracy_impact": "medium/high: 会降低 gold disease-HPO overlap 和检索证据质量",
                "fix_priority": "B",
                "recommended_fix": "统一 HP:0000000 格式，恢复误删 HPO，必要时用 HPO OBO 处理 obsolete/alt_id",
            },
            {
                "issue": "gold disease-HPO overlap_zero",
                "affected_cases": overlap_zero_cases,
                "expected_accuracy_impact": "high for rank>50: 模型缺少 gold hyperedge 证据时难以召回",
                "fix_priority": "B/D",
                "recommended_fix": "区分 cleaned HPO 丢失、label 映射错误、KB 覆盖不足和原文信息不足；不要在 test set 调参",
            },
            {
                "issue": "unmapped/obsolete label in hyperedge/index",
                "affected_cases": unmapped_rows + obsolete_label_rows,
                "expected_accuracy_impact": "critical/high: gold 不在疾病池或 KB 中时 exact retrieval 不可比",
                "fix_priority": "A/B",
                "recommended_fix": "修复 MONDO obsolete/alt_id/replaced_by 映射，并记录版本",
            },
            {
                "issue": "train/test leakage 或 similar-case 重复",
                "affected_cases": leakage_rows,
                "expected_accuracy_impact": "方向不固定: 泄漏通常虚高，split 不一致会让测试口径不可信",
                "fix_priority": "A",
                "recommended_fix": "用 subject_id/hadm_id/note_id/text hash 做最终确认；不要覆盖已有 exact evaluation",
            },
            {
                "issue": "gold 已进入 top50 但 rank>5",
                "affected_cases": "由现有 rank 明细确认",
                "expected_accuracy_impact": "更像模型排序能力或候选重排问题，不一定是清洗错误",
                "fix_priority": "D",
                "recommended_fix": "作为模型问题单独分析；不要用 test set 调参",
            },
        ]
    )
    impact_df = pd.DataFrame(rows)
    write_csv(impact_df, output_dir / "mimic_accuracy_impact.csv")
    return impact_df


def build_final_report(
    output_dir: Path,
    original_path: Path,
    cleaned_path: Path,
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    matching_df: pd.DataFrame,
    label_df: pd.DataFrame,
    multilabel_df: pd.DataFrame,
    hpo_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    resources: dict[str, Any],
) -> None:
    cleaned_case_count = cleaned_df["case_id"].nunique(dropna=True)
    missing_count = int((matching_df["issue_type"] == "missing_in_cleaned").sum())
    critical_label = int(label_df["severity"].eq("critical").sum())
    exact_same = int((label_df["match_type"] == "exact_same").sum())
    hpo_jaccard_median = float(hpo_df["jaccard"].median()) if not hpo_df.empty else float("nan")
    overlap_zero_rate = float(coverage_df.groupby("case_key")["overlap_zero"].min().mean()) if not coverage_df.empty else float("nan")
    multilabel_cases = int((cleaned_df.groupby("case_id")["mondo_label"].nunique() > 1).sum())

    if missing_count == 0 and critical_label == 0 and hpo_jaccard_median >= 0.99:
        judgment = "cleaned mimic_test 基本可信；当前低准确率更可能来自多标签 exact 口径、gold-HPO 覆盖和模型排序能力。"
    else:
        judgment = "cleaned mimic_test 可用于审计但不应直接视为完全可信；需要先核对缺失病例和 critical label/mapping 问题。"

    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    lines = [
        "# mimic_test Cleaning Audit Report",
        "",
        "## 1. Overall Judgment",
        judgment,
        "",
        "## 2. Dataset Schema Comparison",
        f"- original: {len(original_df)} rows，列为 {', '.join(f'`{col}`' for col in original_df.columns)}。",
        f"- cleaned: {len(cleaned_df)} rows，按 `case_id` 聚合为 {cleaned_case_count} cases，列为 {', '.join(f'`{col}`' for col in cleaned_df.columns)}。",
        "- original 是宽表，包含临床 `text`、`note_id`、`orpha`、`orpha_names`、`HPO`；cleaned 是病例-HPO/label 长表。",
        f"- 本次报告实际读取的 cleaned 文件: `{cleaned_path}`。",
        "",
        "## 3. Case Count and Granularity",
        f"- original 有 {len(original_df)} rows / {original_df['note_id'].nunique() if 'note_id' in original_df.columns else '无法确认'} unique `note_id`。",
        f"- cleaned 有 {len(cleaned_df)} rows / {cleaned_case_count} unique `case_id`。",
        f"- cleaned 为 1873 的直接原因是唯一 `case_id` 从 `case_1` 到 `case_1873`，而不是 21749 行长表。",
        f"- missing_in_cleaned: {missing_count}。如果 original 是 1875，则缺失病例见 `mimic_missing_cases.csv`。",
        "- original 粒度判断为 note-level；cleaned 是 case-level long format。patient/admission-level 无法确认，需要 `subject_id` / `hadm_id`。",
        "",
        "## 4. Case-Level Matching",
        f"- matched: {int((matching_df['issue_type'] == 'matched').sum())}",
        f"- label_mismatch: {int((matching_df['issue_type'] == 'label_mismatch').sum())}",
        f"- hpo_mismatch: {int((matching_df['issue_type'] == 'hpo_mismatch').sum())}",
        f"- missing_in_cleaned: {missing_count}",
        f"- extra_in_cleaned: {int((matching_df['issue_type'] == 'extra_in_cleaned').sum())}",
        "- 匹配优先级按显式 ID、`MONDO label set + HPO set`、最后行号 fallback；行号 fallback 会在 CSV 中标出。",
        "",
        "## 5. Label / Disease ID Audit",
        f"- exact_same: {exact_same}",
        f"- critical severity rows: {critical_label}",
        f"- uncertain rows: {int((label_df['match_type'] == 'uncertain').sum())}",
        "- ICD 没有被当作 exact gold；ORPHA 到 MONDO 的映射使用本地 SSSOM/JSON 资源。",
        "- OMIM gene/phenotype、ORPHA group/disorder、parent/subtype 的细粒度混淆无法仅凭当前 CSV 完全确认，需要额外 ontology/manual curation 文件。",
        "",
        "## 6. Multi-label Audit",
        f"- cleaned multi-label cases: {multilabel_cases}",
        f"- original multi-label cases（按解析出的 MONDO set）: {pipe_count(multilabel_df['original_labels'])}",
        "- 当前 exact top1/top3/top5 可能被多标签处理低估；any-label@k 应只作为 supplementary。",
        "- 建议保留 original exact、cleaned exact、any-label 三种口径。",
        "",
        "## 7. HPO Audit",
        f"- HPO Jaccard median: {hpo_jaccard_median:.4f}",
        f"- invalid/obsolete HPO cases: {nonempty_count(hpo_df['invalid_hpo']) + nonempty_count(hpo_df['obsolete_hpo'])}",
        f"- large_hpo_loss cases: {int(hpo_df['suspicious_issue'].astype(str).str.contains('large_hpo_loss', na=False).sum())}",
        "- 是否误删高信息量 HPO、误加入否定症状、家族史或治疗反应：无法确认，需要 HPO 抽取证据 spans 或人工标注说明。",
        "",
        "## 8. Gold Disease-HPO Coverage",
        f"- disease-HPO resource: {resources.get('hyperedge_path') or resources.get('hyperedge_error')}",
        f"- overlap_zero_rate（case-level，任一 gold label 有重叠即不算 zero）: {overlap_zero_rate:.4f}",
        f"- unmapped label rows: {int(coverage_df['unmapped_label'].sum()) if not coverage_df.empty else '无法确认'}",
        "- overlap_zero 高若同时伴随 HPO Jaccard 高、unmapped label 低，则更可能来自 disease-HPO KB 覆盖不足或原始文本信息不足，而不是 cleaned HPO 大量丢失。",
        "",
        "## 9. Split / Leakage Audit",
        f"- leakage/collision rows: {len(leakage_df)}",
        "- patient/subject/hadm/note 跨 split 无法确认，因为 processed train/test 缺少这些 ID。",
        "",
        "## 10. Accuracy Impact Analysis",
        "当前指标 top1=0.1917、top3=0.2995、top5=0.3540、rank<=50=0.6151。影响判断如下：",
        "",
        df_to_markdown(impact_df),
        "",
        "## 11. Fix Plan",
        "1. 优先人工核对 `mimic_missing_cases.csv` 和 `mimic_label_diff.csv` 中 severity=critical 的病例，修复映射，不删除测试样本。",
        "2. 明确 evaluator 对 multi-label 的 gold 选择规则，保留 exact 正式口径，并把 any-label@k 作为 supplementary。",
        "3. 对 overlap_zero 病例抽样回看原始 clinical text、HPO 抽取证据和 v59 disease-HPO hyperedge，区分文本不足、HPO 清洗、label 映射和 KB 缺口。",
        "4. 补充 subject_id/hadm_id/note_id 映射后重新做 split leakage 审计。",
        "",
        "## 12. Reproducible Commands",
        "```powershell",
        r"D:\python\python.exe -m py_compile tools\audit_mimic_cleaning.py",
        f"D:\\python\\python.exe tools\\audit_mimic_cleaning.py --original {original_path} --cleaned {cleaned_path} --output-dir {output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir}",
        "```",
        "",
        "生成文件:",
        "",
    ]
    if cleaned_path.resolve() != DEFAULT_CLEANED.resolve():
        lines.insert(
            11,
            f"- 注意: 用户指定的 `{DEFAULT_CLEANED}` 在最终重跑时不可读；本报告使用当前可读的 `{cleaned_path}` 生成。若恢复指定文件，请按第 12 节命令重跑。",
        )
    lines.extend(f"- `{name}`" for name in files)
    write_text(output_dir / "final_mimic_cleaning_audit.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_output_dir(output_dir)

    original_df = read_table(args.original)
    cleaned_df = read_table(args.cleaned)

    audit_schema(original_df, cleaned_df, output_dir)

    orpha_to_mondo = load_simple_mapping(ORPHA_TO_MONDO_PATH, normalize_orpha)
    omim_to_mondo = load_simple_mapping(OMIM_TO_MONDO_PATH, normalize_omim)
    orpha_to_omim = load_orpha_to_omim(ORPHA_TO_OMIM_PATH)
    orpha_to_name = load_orpha_to_name(ORPHA_TO_NAME_PATH)
    mondo_resource = load_mondo_resource()
    hpo_resource = load_hpo_resource()

    original_cases = build_original_cases(
        original_df,
        orpha_to_mondo=orpha_to_mondo,
        orpha_to_omim=orpha_to_omim,
        omim_to_mondo=omim_to_mondo,
        orpha_to_name=orpha_to_name,
        mondo_resource=mondo_resource,
    )
    cleaned_cases = build_cleaned_cases(cleaned_df)

    matching_df, _ = build_case_matches(original_cases, cleaned_cases)
    audit_case_granularity(original_df, cleaned_df, original_cases, cleaned_cases, matching_df, output_dir)

    audit_labels(original_cases, cleaned_cases, matching_df, output_dir, mondo_resource)
    label_df = pd.read_csv(output_dir / "mimic_label_diff.csv", dtype=str)

    audit_multilabel(original_cases, cleaned_cases, matching_df, output_dir)
    multilabel_df = pd.read_csv(output_dir / "mimic_multilabel_diff.csv", dtype=str)

    hpo_df = audit_hpo(original_cases, cleaned_cases, matching_df, output_dir, hpo_resource)

    disease_hpos, disease_weights, hyperedge_path, hyperedge_error = load_hyperedge()
    coverage_df = audit_coverage(
        cleaned_cases,
        output_dir,
        disease_hpos,
        disease_weights,
        hyperedge_path,
        hyperedge_error,
        hpo_resource,
        mondo_resource,
    )

    leakage_df = audit_split_leakage(args.cleaned, output_dir)

    impact_df = build_accuracy_impact(label_df, multilabel_df, hpo_df, coverage_df, leakage_df, output_dir)
    build_final_report(
        output_dir,
        args.original,
        args.cleaned,
        original_df,
        cleaned_df,
        matching_df,
        label_df,
        multilabel_df,
        hpo_df,
        coverage_df,
        leakage_df,
        impact_df,
        {
            "hyperedge_path": hyperedge_path,
            "hyperedge_error": hyperedge_error,
            "mondo_path": mondo_resource.get("path"),
            "hpo_path": hpo_resource.get("path"),
        },
    )

    manifest = {
        "original": str(args.original),
        "cleaned": str(args.cleaned),
        "output_dir": str(output_dir),
        "original_rows": int(len(original_df)),
        "cleaned_rows": int(len(cleaned_df)),
        "cleaned_cases": int(cleaned_df["case_id"].nunique(dropna=True)) if "case_id" in cleaned_df.columns else None,
        "generated_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
