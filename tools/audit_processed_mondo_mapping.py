from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(r"D:\RareDisease-traindata")
RAW_DIR = PROJECT_ROOT / "LLLdataset" / "dataset" / "多标签" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "LLLdataset" / "dataset" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

DB_DIR = Path(r"D:\DeepRare-main\database")
OMIM_SSSOM_PATH = DB_DIR / "mondo_exactmatch_omim.sssom.tsv"
ORPHA_SSSOM_PATH = DB_DIR / "mondo_hasdbxref_orphanet.sssom.tsv"
ORPHA_TO_OMIM_PATH = DB_DIR / "orpha2omim.json"
ORPHA_TO_NAME_PATH = DB_DIR / "orpha2name.json"
MONDO_JSON_CANDIDATES = [
    Path(r"D:\Deep\new_data\mondo-base_v20260303.json"),
    DB_DIR / "mondo-rare.json",
]

HPO_RE = re.compile(r"HP:\d{7}")
NAME_PREFIXES_TO_STRIP = ("obsolete:", "non rare in europe:")

MANUAL_ORPHA_TO_MONDO_OVERRIDES = {
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


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_path: Path
    processed_path: Path
    raw_format: str


@dataclass(frozen=True)
class RawCase:
    dataset: str
    source_case_id: str
    hpo_ids: tuple[str, ...]
    disease_tokens: tuple[str, ...]
    disease_name: str
    disease_name_tokens: tuple[str, ...]


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    for prefix in NAME_PREFIXES_TO_STRIP:
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_string_list(raw_value: object) -> list[str]:
    if raw_value is None or pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [match.group(0) for match in re.finditer(r"[A-Za-z]+[:_]\d+", text)]


def parse_hpo_values(raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, (list, tuple, set)):
        candidates = [str(item).strip().upper() for item in raw_value]
    else:
        candidates = [match.group(0).upper() for match in HPO_RE.finditer(str(raw_value))]
    return tuple(sorted(set(candidate for candidate in candidates if HPO_RE.fullmatch(candidate))))


def normalize_prefixed_id(raw_value: object, prefix: str, digits: int = 0) -> str | None:
    if raw_value is None:
        return None
    match = re.search(rf"{re.escape(prefix)}[:_]?(\d+)", str(raw_value), flags=re.IGNORECASE)
    if not match:
        return None
    digits_text = match.group(1).zfill(digits) if digits > 0 else match.group(1)
    return f"{prefix}:{digits_text}"


def normalize_mondo(raw_value: object) -> str | None:
    return normalize_prefixed_id(raw_value, "MONDO", digits=7)


def normalize_omim(raw_value: object) -> str | None:
    return normalize_prefixed_id(raw_value, "OMIM", digits=0)


def normalize_orpha(raw_value: object) -> str | None:
    normalized = normalize_prefixed_id(raw_value, "ORPHA", digits=0)
    if normalized is not None:
        return normalized
    normalized = normalize_prefixed_id(raw_value, "Orphanet", digits=0)
    if normalized is None:
        return None
    return f"ORPHA:{normalized.split(':', 1)[1]}"


def parse_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def choose_mondo_json_path() -> Path:
    for path in MONDO_JSON_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find MONDO JSON from: {MONDO_JSON_CANDIDATES}")


def build_dataset_configs() -> list[DatasetConfig]:
    return [
        DatasetConfig("ddd_test", RAW_DIR / "ddd_test.csv", PROCESSED_DIR / "ddd_test.csv", "csv_ddd"),
        DatasetConfig("mimic_test", RAW_DIR / "mimic_test.csv", PROCESSED_DIR / "test" / "mimic_test.csv", "csv_mimic"),
        DatasetConfig("mimic_rag_0425", RAW_DIR / "mimic_rag_0425.csv", PROCESSED_DIR / "mimic_rag_0425.csv", "csv_mimic"),
        DatasetConfig("HMS", RAW_DIR / "HMS.jsonl", PROCESSED_DIR / "HMS.xlsx", "jsonl"),
        DatasetConfig("LIRICAL", RAW_DIR / "LIRICAL.jsonl", PROCESSED_DIR / "LIRICAL.xlsx", "jsonl"),
        DatasetConfig("MME", RAW_DIR / "MME.jsonl", PROCESSED_DIR / "MME.xlsx", "jsonl"),
        DatasetConfig("RAMEDIS", RAW_DIR / "RAMEDIS.jsonl", PROCESSED_DIR / "RAMEDIS.xlsx", "jsonl"),
        DatasetConfig("MyGene2", RAW_DIR / "mygene2_5.7.22.txt", PROCESSED_DIR / "MyGene2.xlsx", "mygene2"),
    ]


def load_raw_cases(config: DatasetConfig) -> list[RawCase]:
    if config.raw_format == "csv_ddd":
        df = pd.read_csv(config.raw_path, dtype=str)
        return [
            RawCase(
                dataset=config.name,
                source_case_id=str(row["id"]).strip(),
                hpo_ids=parse_hpo_values(row.get("phenotype")),
                disease_tokens=tuple(parse_string_list(row.get("rare_disease"))),
                disease_name="",
                disease_name_tokens=(),
            )
            for _, row in df.iterrows()
        ]

    if config.raw_format == "csv_mimic":
        df = pd.read_csv(config.raw_path, dtype=str)
        return [
            RawCase(
                dataset=config.name,
                source_case_id=str(row["note_id"]).strip(),
                hpo_ids=parse_hpo_values(row.get("HPO")),
                disease_tokens=tuple(parse_string_list(row.get("orpha"))),
                disease_name="; ".join(parse_string_list(row.get("orpha_names"))),
                disease_name_tokens=tuple(parse_string_list(row.get("orpha_names"))),
            )
            for _, row in df.iterrows()
        ]

    if config.raw_format == "jsonl":
        rows = parse_jsonl(config.raw_path)
        return [
            RawCase(
                dataset=config.name,
                source_case_id=f"line_{idx}",
                hpo_ids=parse_hpo_values(row.get("Phenotype", [])),
                disease_tokens=tuple(str(item).strip() for item in row.get("RareDisease", []) if str(item).strip()),
                disease_name="",
                disease_name_tokens=(),
            )
            for idx, row in enumerate(rows, start=1)
        ]

    if config.raw_format == "mygene2":
        rows = parse_jsonl(config.raw_path)
        cases: list[RawCase] = []
        for row in rows:
            tokens: list[str] = []
            for disease_id in row.get("true_diseases", []) or []:
                try:
                    tokens.append(f"MONDO:{int(disease_id):07d}")
                except Exception:
                    pass
            omim_value = row.get("omim")
            if omim_value:
                tokens.append(f"OMIM:{str(omim_value).strip()}")
            for orpha_value in row.get("orpha_id", []) or []:
                tokens.append(f"ORPHA:{str(orpha_value).strip()}")
            cases.append(
                RawCase(
                    dataset=config.name,
                    source_case_id=str(row.get("id", "")).strip(),
                    hpo_ids=parse_hpo_values(row.get("positive_phenotypes", [])),
                    disease_tokens=tuple(tokens),
                    disease_name=str(row.get("disease_name", "")).strip(),
                    disease_name_tokens=(str(row.get("disease_name", "")).strip(),),
                )
            )
        return cases

    raise ValueError(f"Unsupported raw format: {config.raw_format}")


def load_omim_to_mondo(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        mondo_id = normalize_mondo(getattr(row, "subject_id", None))
        omim_id = normalize_omim(getattr(row, "object_id", None))
        if mondo_id and omim_id:
            mapping[omim_id].add(mondo_id)
    return mapping


def load_orpha_to_mondo(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        mondo_id = normalize_mondo(getattr(row, "subject_id", None))
        orpha_id = normalize_orpha(getattr(row, "object_id", None))
        if mondo_id and orpha_id:
            mapping[orpha_id].add(mondo_id)
    return mapping


def load_orpha_simple_json(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    result: dict[str, str] = {}
    for key, value in data.items():
        orpha_id = normalize_orpha(key)
        if orpha_id is not None:
            result[orpha_id] = str(value).strip()
    return result


def load_mondo_name_index(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    graphs = data.get("graphs", [])
    if not graphs:
        raise ValueError(f"No graphs in MONDO file: {path}")

    name_to_ids: dict[str, set[str]] = defaultdict(set)
    for node in graphs[0].get("nodes", []):
        mondo_id = normalize_mondo(node.get("id"))
        if mondo_id is None:
            continue
        meta = node.get("meta") or {}
        if meta.get("deprecated") is True:
            continue

        labels: list[str] = []
        primary_label = node.get("lbl")
        if isinstance(primary_label, str) and primary_label.strip():
            if not primary_label.lower().startswith("obsolete "):
                labels.append(primary_label.strip())

        for synonym in meta.get("synonyms", []):
            if not isinstance(synonym, dict):
                continue
            value = synonym.get("val")
            if isinstance(value, str) and value.strip():
                labels.append(value.strip())

        for label in ordered_unique(labels):
            normalized = normalize_text(label)
            if normalized:
                name_to_ids[normalized].add(mondo_id)

    return {key: sorted(values) for key, values in name_to_ids.items()}


def resolve_by_name(name: str, mondo_name_index: dict[str, list[str]]) -> tuple[list[str], str | None]:
    normalized = normalize_text(name)
    if not normalized:
        return [], None
    hits = mondo_name_index.get(normalized, [])
    if len(hits) == 1:
        return hits, "name_unique"
    return [], None


def resolve_case(
    raw_case: RawCase,
    omim_to_mondo: dict[str, set[str]],
    orpha_to_mondo: dict[str, set[str]],
    orpha_to_omim: dict[str, str],
    orpha_to_name: dict[str, str],
    mondo_name_index: dict[str, list[str]],
) -> dict[str, object]:
    resolved: list[str] = []
    route_counter: Counter[str] = Counter()
    unresolved_tokens: list[str] = []

    for idx, token in enumerate(raw_case.disease_tokens):
        if mondo_id := normalize_mondo(token):
            resolved.append(mondo_id)
            route_counter["input_mondo"] += 1
            continue

        if omim_id := normalize_omim(token):
            hits = sorted(omim_to_mondo.get(omim_id, set()))
            if hits:
                resolved.extend(hits)
                route_counter["omim_exact"] += 1
            else:
                unresolved_tokens.append(token)
            continue

        if orpha_id := normalize_orpha(token):
            hits = sorted(orpha_to_mondo.get(orpha_id, set()))
            if hits:
                resolved.extend(hits)
                route_counter["orpha_exact"] += 1
                continue

            via_omim = orpha_to_omim.get(orpha_id)
            if via_omim:
                via_hits = sorted(omim_to_mondo.get(via_omim, set()))
                if via_hits:
                    resolved.extend(via_hits)
                    route_counter["orpha_via_omim"] += 1
                    continue

            manual_hit = MANUAL_ORPHA_TO_MONDO_OVERRIDES.get(orpha_id)
            if manual_hit:
                resolved.append(manual_hit)
                route_counter["manual_orpha_to_mondo"] += 1
                continue

            candidate_name = orpha_to_name.get(orpha_id, "")
            name_hits, route = resolve_by_name(candidate_name, mondo_name_index)
            if name_hits and route:
                resolved.extend(name_hits)
                route_counter["orpha_name_unique"] += 1
                continue

            if idx < len(raw_case.disease_name_tokens):
                aligned_name = raw_case.disease_name_tokens[idx]
                name_hits, route = resolve_by_name(aligned_name, mondo_name_index)
                if name_hits and route:
                    resolved.extend(name_hits)
                    route_counter["aligned_name_unique"] += 1
                    continue

            unresolved_tokens.append(token)
            continue

        unresolved_tokens.append(token)

    if not resolved and raw_case.disease_name:
        hits, route = resolve_by_name(raw_case.disease_name, mondo_name_index)
        if hits and route:
            resolved.extend(hits)
            route_counter["disease_name_unique"] += 1

    return {
        "source_case_id": raw_case.source_case_id,
        "hpo_ids": raw_case.hpo_ids,
        "mondo_labels": tuple(sorted(ordered_unique(resolved))),
        "routes": dict(route_counter),
        "unresolved_tokens": unresolved_tokens,
    }


def load_processed_cases(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)
    df = df.fillna("")
    cases: list[dict[str, object]] = []
    for case_id, group in df.groupby("case_id", sort=False):
        mondo_labels = tuple(sorted({str(value).strip() for value in group["mondo_label"] if str(value).strip()}))
        hpo_ids = tuple(sorted({str(value).strip() for value in group["hpo_id"] if str(value).strip()}))
        cases.append({"case_id": str(case_id).strip(), "mondo_labels": mondo_labels, "hpo_ids": hpo_ids})
    return cases


def signature_counter(
    cases: list[dict[str, object]],
    include_empty_mondo: bool,
) -> Counter[tuple[tuple[str, ...], tuple[str, ...]]]:
    counter: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    for case in cases:
        mondo_labels = tuple(case["mondo_labels"])
        hpo_ids = tuple(case["hpo_ids"])
        if not hpo_ids:
            continue
        if not mondo_labels and not include_empty_mondo:
            continue
        counter[(mondo_labels, hpo_ids)] += 1
    return counter


def hpo_collision_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    hpo_to_mondos: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for case in cases:
        hpo_ids = tuple(case["hpo_ids"])
        mondo_labels = tuple(case["mondo_labels"])
        if hpo_ids:
            hpo_to_mondos[hpo_ids].append(mondo_labels)

    collision_rows: list[dict[str, object]] = []
    for hpo_ids, mondo_lists in hpo_to_mondos.items():
        flat_unique = sorted({mondo for mondo_list in mondo_lists for mondo in mondo_list if mondo})
        unique_sets = sorted({tuple(mondo_list) for mondo_list in mondo_lists})
        if len(flat_unique) <= 1:
            continue
        collision_rows.append(
            {
                "hpo_ids": hpo_ids,
                "n_cases": len(mondo_lists),
                "n_unique_mondo_ids": len(flat_unique),
                "n_unique_mondo_sets": len(unique_sets),
                "mondo_sets": [list(mondo_set) for mondo_set in unique_sets[:5]],
            }
        )

    collision_rows.sort(
        key=lambda item: (-int(item["n_unique_mondo_ids"]), -int(item["n_cases"]), item["hpo_ids"])
    )
    return {
        "collision_hpo_signatures": len(collision_rows),
        "top_examples": collision_rows[:10],
    }


def diff_counter(
    left: Counter[tuple[tuple[str, ...], tuple[str, ...]]],
    right: Counter[tuple[tuple[str, ...], tuple[str, ...]]],
    max_items: int = 10,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for signature, count in (left - right).most_common(max_items):
        mondo_labels, hpo_ids = signature
        rows.append(
            {
                "count": count,
                "mondo_labels": list(mondo_labels),
                "hpo_count": len(hpo_ids),
                "hpo_ids_preview": list(hpo_ids[:8]),
            }
        )
    return rows


def sorted_case_key(case: dict[str, object]) -> tuple[int, str]:
    case_id = str(case.get("case_id", ""))
    match = re.search(r"(\d+)$", case_id)
    if match:
        return int(match.group(1)), case_id
    return 10**12, case_id


def match_processed_to_raw_candidates(
    raw_cases: list[dict[str, object]],
    processed_cases: list[dict[str, object]],
) -> dict[str, object]:
    raw_by_hpo: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for raw_case in raw_cases:
        hpo_ids = tuple(raw_case["hpo_ids"])
        if hpo_ids:
            raw_by_hpo[hpo_ids].append(raw_case)

    processed_by_hpo: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for processed_case in processed_cases:
        hpo_ids = tuple(processed_case["hpo_ids"])
        if hpo_ids:
            processed_by_hpo[hpo_ids].append(processed_case)

    supported_count = 0
    exact_equal_count = 0
    subset_only_count = 0
    unsupported_examples: list[dict[str, object]] = []

    for hpo_ids, processed_group in processed_by_hpo.items():
        raw_group = raw_by_hpo.get(hpo_ids, [])
        raw_group_sorted = sorted(
            raw_group,
            key=lambda item: (len(tuple(item["mondo_labels"])), str(item["source_case_id"])),
        )
        used = [False] * len(raw_group_sorted)

        processed_group_sorted = sorted(
            processed_group,
            key=lambda item: (-len(tuple(item["mondo_labels"])), sorted_case_key(item)),
        )

        for processed_case in processed_group_sorted:
            processed_set = set(processed_case["mondo_labels"])
            matched_index: int | None = None
            matched_exact = False

            for idx, raw_case in enumerate(raw_group_sorted):
                if used[idx]:
                    continue
                raw_set = set(raw_case["mondo_labels"])
                if processed_set.issubset(raw_set):
                    matched_index = idx
                    matched_exact = processed_set == raw_set
                    break

            if matched_index is None:
                unsupported_examples.append(
                    {
                        "case_id": processed_case["case_id"],
                        "mondo_labels": list(processed_case["mondo_labels"]),
                        "hpo_count": len(hpo_ids),
                        "hpo_ids_preview": list(hpo_ids[:8]),
                        "raw_candidate_sets_preview": [
                            list(raw_case["mondo_labels"]) for raw_case in raw_group_sorted[:5]
                        ],
                    }
                )
                continue

            used[matched_index] = True
            supported_count += 1
            if matched_exact:
                exact_equal_count += 1
            else:
                subset_only_count += 1

    return {
        "supported_count": supported_count,
        "exact_equal_count": exact_equal_count,
        "subset_only_count": subset_only_count,
        "unsupported_count": len(processed_cases) - supported_count,
        "unsupported_examples": unsupported_examples[:10],
    }


def build_report(summary: dict[str, object], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Processed MONDO Mapping Audit")
    lines.append("")
    lines.append("## 总体结论")
    lines.append("")
    global_summary = summary["global"]
    lines.append(
        f"- 数据集数: {global_summary['dataset_count']}，processed 病例被原始标注支持: "
        f"{global_summary['supported_processed_cases']} / {global_summary['processed_cases']}。"
    )
    lines.append(
        f"- 原始样本中存在同病例多 MONDO 的情况: {global_summary['raw_multi_label_cases']} 例；"
        f"映射后保留为多标签的病例: {global_summary['processed_multi_label_cases']} 例。"
    )
    lines.append(
        f"- 映射后按 `HPO 集合` 聚合后，出现“同一套 HPO 对应多个 MONDO”的签名数: "
        f"{global_summary['processed_hpo_collision_signatures']}。"
    )
    lines.append("")
    lines.append("## 分数据集结果")
    lines.append("")

    for dataset in summary["datasets"]:
        lines.append(f"### {dataset['name']}")
        lines.append("")
        lines.append(
            f"- raw_cases={dataset['raw_cases']}，processed_cases={dataset['processed_cases']}，"
            f"supported_processed_cases={dataset['supported_processed_cases']}，"
            f"unsupported_processed_cases={dataset['unsupported_processed_cases']}"
        )
        lines.append(
            f"- exact_equal_processed_cases={dataset['exact_equal_processed_cases']}，"
            f"subset_only_processed_cases={dataset['subset_only_processed_cases']}，"
            f"expected_retained_cases={dataset['expected_retained_cases']}"
        )
        lines.append(
            f"- raw_multi_label_cases={dataset['raw_multi_label_cases']}，"
            f"processed_multi_label_cases={dataset['processed_multi_label_cases']}，"
            f"processed_blank_mondo_cases={dataset['processed_blank_mondo_cases']}"
        )
        lines.append(
            f"- unresolved_raw_cases={dataset['unresolved_raw_cases']}，"
            f"route_counts={json.dumps(dataset['route_counts'], ensure_ascii=False, sort_keys=True)}"
        )
        lines.append(
            f"- raw_hpo_collision_signatures={dataset['raw_hpo_collision_signatures']}，"
            f"processed_hpo_collision_signatures={dataset['processed_hpo_collision_signatures']}"
        )
        if dataset["unsupported_examples"]:
            lines.append("- 无法被原始标注支持的 processed 样例:")
            for item in dataset["unsupported_examples"]:
                lines.append(
                    f"  - case_id={item['case_id']}, mondo={item['mondo_labels']}, "
                    f"hpo_count={item['hpo_count']}, hpo_preview={item['hpo_ids_preview']}, "
                    f"raw_candidates={item['raw_candidate_sets_preview']}"
                )
        if dataset["processed_hpo_collision_examples"]:
            lines.append("- 同一 HPO 对应多个 MONDO 样例:")
            for item in dataset["processed_hpo_collision_examples"][:3]:
                lines.append(
                    f"  - n_cases={item['n_cases']}, n_unique_mondo_ids={item['n_unique_mondo_ids']}, "
                    f"hpo_count={len(item['hpo_ids'])}, mondo_sets={item['mondo_sets']}"
                )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    mondo_json_path = choose_mondo_json_path()
    omim_to_mondo = load_omim_to_mondo(OMIM_SSSOM_PATH)
    orpha_to_mondo = load_orpha_to_mondo(ORPHA_SSSOM_PATH)
    orpha_to_omim = load_orpha_simple_json(ORPHA_TO_OMIM_PATH)
    orpha_to_name = load_orpha_simple_json(ORPHA_TO_NAME_PATH)
    mondo_name_index = load_mondo_name_index(mondo_json_path)

    dataset_summaries: list[dict[str, object]] = []
    global_raw_multi = 0
    global_processed_multi = 0
    global_processed_hpo_collisions = 0
    global_processed_cases = 0
    global_supported_cases = 0

    for config in build_dataset_configs():
        raw_cases = load_raw_cases(config)
        resolved_cases = [
            resolve_case(
                raw_case=raw_case,
                omim_to_mondo=omim_to_mondo,
                orpha_to_mondo=orpha_to_mondo,
                orpha_to_omim=orpha_to_omim,
                orpha_to_name=orpha_to_name,
                mondo_name_index=mondo_name_index,
            )
            for raw_case in raw_cases
        ]
        processed_cases = load_processed_cases(config.processed_path)

        processed_blank_mondo_cases = sum(1 for case in processed_cases if not case["mondo_labels"])
        include_empty_mondo = processed_blank_mondo_cases > 0

        expected_cases = [
            case
            for case in resolved_cases
            if case["hpo_ids"] and (case["mondo_labels"] or include_empty_mondo)
        ]

        raw_multi_label_cases = sum(1 for case in expected_cases if len(case["mondo_labels"]) > 1)
        processed_multi_label_cases = sum(1 for case in processed_cases if len(case["mondo_labels"]) > 1)
        unresolved_raw_cases = sum(1 for case in resolved_cases if case["hpo_ids"] and not case["mondo_labels"])
        route_counts: Counter[str] = Counter()
        for case in resolved_cases:
            route_counts.update(case["routes"])

        expected_counter = signature_counter(expected_cases, include_empty_mondo=include_empty_mondo)
        processed_counter = signature_counter(processed_cases, include_empty_mondo=include_empty_mondo)

        raw_hpo_collision = hpo_collision_summary(expected_cases)
        processed_hpo_collision = hpo_collision_summary(processed_cases)
        support_stats = match_processed_to_raw_candidates(expected_cases, processed_cases)

        global_raw_multi += raw_multi_label_cases
        global_processed_multi += processed_multi_label_cases
        global_processed_hpo_collisions += int(processed_hpo_collision["collision_hpo_signatures"])
        global_processed_cases += len(processed_cases)
        global_supported_cases += int(support_stats["supported_count"])

        dataset_summaries.append(
            {
                "name": config.name,
                "raw_cases": len(raw_cases),
                "processed_cases": len(processed_cases),
                "expected_retained_cases": len(expected_cases),
                "signature_exact_match": expected_counter == processed_counter,
                "supported_processed_cases": int(support_stats["supported_count"]),
                "unsupported_processed_cases": int(support_stats["unsupported_count"]),
                "exact_equal_processed_cases": int(support_stats["exact_equal_count"]),
                "subset_only_processed_cases": int(support_stats["subset_only_count"]),
                "raw_multi_label_cases": raw_multi_label_cases,
                "processed_multi_label_cases": processed_multi_label_cases,
                "processed_blank_mondo_cases": processed_blank_mondo_cases,
                "unresolved_raw_cases": unresolved_raw_cases,
                "route_counts": dict(sorted(route_counts.items())),
                "raw_hpo_collision_signatures": int(raw_hpo_collision["collision_hpo_signatures"]),
                "processed_hpo_collision_signatures": int(processed_hpo_collision["collision_hpo_signatures"]),
                "missing_signature_examples": diff_counter(expected_counter, processed_counter),
                "extra_signature_examples": diff_counter(processed_counter, expected_counter),
                "unsupported_examples": support_stats["unsupported_examples"],
                "processed_hpo_collision_examples": processed_hpo_collision["top_examples"],
            }
        )

    summary = {
        "global": {
            "dataset_count": len(dataset_summaries),
            "exact_match_dataset_count": sum(1 for item in dataset_summaries if item["signature_exact_match"]),
            "processed_cases": global_processed_cases,
            "supported_processed_cases": global_supported_cases,
            "raw_multi_label_cases": global_raw_multi,
            "processed_multi_label_cases": global_processed_multi,
            "processed_hpo_collision_signatures": global_processed_hpo_collisions,
        },
        "datasets": dataset_summaries,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "processed_mondo_audit_20260421.json"
    md_path = REPORT_DIR / "processed_mondo_audit_20260421.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(summary, md_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
