from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from openpyxl import Workbook


GENERIC_DISEASE_TOKENS = {
    "and",
    "by",
    "cause",
    "caused",
    "condition",
    "deficiency",
    "disease",
    "disorder",
    "form",
    "in",
    "of",
    "or",
    "related",
    "syndrome",
    "the",
    "to",
    "type",
    "with",
    "without",
}
FUZZY_MIN_SCORE = 0.90
FUZZY_MIN_MARGIN = 0.03
OUTPUT_HEADERS = ["case_id", "mondo_label", "hpo_id"]
DEEPRARE_MANUAL_NAME_TO_MONDO = {
    # These mappings were verified against local MONDO / DeepRare resources.
    "COL1A1-related osteogenesis imperfecta spectrum": "MONDO:0019019",
    "PIK3CA-related overgrowth spectrum disorder with or without megalencephaly, capillary malformation, polymicrogyria and lipomatous overgrowth": "MONDO:1040002",
    "UNC80-related persistent hypotonia, encephalopathy, growth retardation, and severe intellectual disability": "MONDO:0014777",
    "TSEN15-related pontocerebellar hypoplasia and progressive microcephaly": "MONDO:0014874",
    "UBA5-related severe infantile-onset encephalopathy": "MONDO:0014933",
    "PRKD1-related syndromic congenital heart defects": "MONDO:0100614",
    "PPA2-related sudden arrhythmic cardiac death after infectious or alcohol trigger": "MONDO:0014973",
    "USP18-related severe pseudo-TORCH syndrome": "MONDO:0018828",
    "NANS-related infantile-onset severe developmental delay and skeletal dysplasia": "MONDO:0012495",
    "KMT2B-related complex early-onset dystonia": "MONDO:0015004",
    "MBOAT7-related intellectual disability accompanied by epilepsy and autistic features": "MONDO:0014962",
    "MORC2-related axonal neuropathy and neurodevelopmental disorder": "MONDO:0014736",
    "MECR-related childhood-onset dystonia and optic atrophy": "MONDO:0015003",
    "PLPBP-related vitamin-B6-dependent epilepsy": "MONDO:0009945",
    "MDH2-related early-onset severe encephalopathy": "MONDO:0015025",
    "MYPN-related childhood-onset, slowly progressive nemaline myopathy": "MONDO:0015023",
    "PRUNE1-related PEHO like condition": "MONDO:0020495",
    "ZNF462-related craniofacial anomalies, corpus callosum dysgenesis, ptosis, and developmental delay": "MONDO:0032836",
    "PLAA-related lethal infantile epileptic encephalopathy": "MONDO:0060502",
    "NACC1-related infantile epilepsy, cataracts, and profound developmental delay": "MONDO:0800475",
    "NADK2-related dienoyl-CoA reductase deficiency with hyperlysinemia": "MONDO:0014464",
    "OTUD7A-related 15q13.3 deletions phenocopy": "MONDO:0012774",
    "MYF5-related external ophthalmoplegia, rib, and vertebral anomalies": "MONDO:0032565",
    "UFC1-related severe early-onset encephalopathy with progressive microcephaly": "MONDO:0010397",
    "CRKL-related bladder exstrophy plus": "MONDO:0010805",
    "MAB21L1-related cerebello-oculo-facio-genital syndrome": "MONDO:0032774",
    "NAXE-related lethal neurometabolic disorder of early childhood": "MONDO:0020781",
    "NAXD-related neurodegenerative disorder exacerbated by febrile illnesses": "MONDO:0034121",
    "FARS2-related neurometabolic disorder": "MONDO:0013986",
    "PMPCB-related neurodegeneration in early childhood": "MONDO:0054785",
    "VPS53-related progressive cerebella-cerebral atrophy": "MONDO:0014370",
    "KIF14-related severe microcephaly and short stature": "MONDO:0100346",
    "PLS3-related osteoporosis with fractures": "MONDO:0018315",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed DDD.xlsx from DDG2P CSV with MONDO rescue rules."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(r"D:\RareDisease-traindata\LLLdataset\dataset\DDG2P_2025-05-28.csv"),
        help="DDG2P CSV input path.",
    )
    parser.add_argument(
        "--omim-sssom",
        type=Path,
        default=Path(r"D:\DeepRare-data\mondo_exactmatch_omim.sssom.tsv"),
        help="MONDO exact OMIM SSSOM path.",
    )
    parser.add_argument(
        "--mondo-json",
        type=Path,
        default=Path(r"D:\Deep\new_data\mondo-base_v20260303.json"),
        help="MONDO JSON path used for disease-name mapping.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path(r"D:\RareDisease-traindata\LLLdataset\dataset\processed\DDD.xlsx"),
        help="Output XLSX path.",
    )
    parser.add_argument(
        "--sheet-name",
        default="Sheet1",
        help="Worksheet name. Default matches existing processed files.",
    )
    return parser.parse_args()


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKD", str(text))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def strip_gene_related_prefix(text: str | None) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKD", str(text))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.replace("\u2013", "-").replace("\u2014", "-").strip()
    value = re.sub(
        r"^(?:[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*)(?:\s+and\s+(?:[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*))*-related\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value


def simplify_disease_text(text: str | None) -> str:
    value = normalize_text(strip_gene_related_prefix(text))
    value = re.sub(
        r"\b(and|by|cause|caused|condition|deficiency|disease|disorder|form|in|of|or|syndrome|the|to|type|with|without)\b",
        " ",
        value,
    )
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_numeric_tokens(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"\d+[a-z]*", text))))


def informative_tokens(text: str | None) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if token not in GENERIC_DISEASE_TOKENS and len(token) > 2
    }


def common_prefix_length(a: str, b: str) -> int:
    size = min(len(a), len(b))
    for idx in range(size):
        if a[idx] != b[idx]:
            return idx
    return size


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_mondo_id(raw_id: str | None) -> str | None:
    if raw_id is None:
        return None
    value = str(raw_id).strip()
    if not value:
        return None
    if value.startswith("MONDO:"):
        return value
    match = re.search(r"MONDO[_:](\d+)", value)
    if match:
        return f"MONDO:{match.group(1)}"
    return None


def add_label(mapping: dict[str, set[str]], key: str, mondo_id: str) -> None:
    if key:
        mapping[key].add(mondo_id)


def load_omim_to_mondo(sssom_path: Path) -> dict[str, set[str]]:
    omim_to_mondo: dict[str, set[str]] = defaultdict(set)
    with sssom_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mondo_id = extract_mondo_id(row.get("subject_id"))
            omim_id = (row.get("object_id") or "").strip()
            if mondo_id is None or not omim_id.startswith("OMIM:"):
                continue
            omim_to_mondo[omim_id].add(mondo_id)
    return omim_to_mondo


def load_mondo_name_resources(mondo_json_path: Path) -> dict[str, object]:
    with mondo_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    graphs = data.get("graphs", [])
    if not graphs:
        raise ValueError(f"No graphs found in {mondo_json_path}")

    mondo_exact: dict[str, set[str]] = defaultdict(set)
    mondo_simplified: dict[str, set[str]] = defaultdict(set)
    fuzzy_label_to_mondo: dict[str, set[str]] = defaultdict(set)

    for node in graphs[0].get("nodes", []):
        mondo_id = extract_mondo_id(node.get("id"))
        if mondo_id is None:
            continue

        meta = node.get("meta") or {}
        if meta.get("deprecated") is True:
            continue

        labels: list[str] = []
        primary_label = node.get("lbl")
        if isinstance(primary_label, str) and primary_label.strip():
            if primary_label.lower().startswith("obsolete "):
                continue
            labels.append(primary_label.strip())

        for synonym in meta.get("synonyms", []):
            if not isinstance(synonym, dict):
                continue
            value = synonym.get("val")
            if isinstance(value, str) and value.strip():
                labels.append(value.strip())

        for label in ordered_unique(labels):
            normalized = normalize_text(label)
            simplified = simplify_disease_text(label)
            add_label(mondo_exact, normalized, mondo_id)
            add_label(mondo_simplified, simplified, mondo_id)
            add_label(fuzzy_label_to_mondo, normalized, mondo_id)
            add_label(fuzzy_label_to_mondo, simplified, mondo_id)

    fuzzy_prefix_index: dict[str, set[str]] = defaultdict(set)
    fuzzy_label_tokens: dict[str, set[str]] = {}
    fuzzy_label_numeric_tokens: dict[str, tuple[str, ...]] = {}
    for label in fuzzy_label_to_mondo:
        tokens = informative_tokens(label)
        if not tokens:
            continue
        fuzzy_label_tokens[label] = tokens
        fuzzy_label_numeric_tokens[label] = extract_numeric_tokens(label)
        for token in tokens:
            prefix = token[:5] if len(token) >= 5 else token
            fuzzy_prefix_index[prefix].add(label)

    return {
        "mondo_exact": mondo_exact,
        "mondo_simplified": mondo_simplified,
        "fuzzy_label_to_mondo": fuzzy_label_to_mondo,
        "fuzzy_prefix_index": fuzzy_prefix_index,
        "fuzzy_label_tokens": fuzzy_label_tokens,
        "fuzzy_label_numeric_tokens": fuzzy_label_numeric_tokens,
    }


def unique_hit(mapping: dict[str, set[str]], key: str) -> str | None:
    if not key:
        return None
    hits = mapping.get(key, set())
    if len(hits) == 1:
        return next(iter(hits))
    return None


def query_variants(disease_name: str | None) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    raw_text = str(disease_name or "").strip()
    stripped_text = strip_gene_related_prefix(raw_text).strip()
    normalized_raw = normalize_text(raw_text)
    normalized_stripped = normalize_text(stripped_text)
    simplified_raw = simplify_disease_text(raw_text)
    simplified_stripped = simplify_disease_text(stripped_text)

    for route, value in [
        ("name_exact", normalized_raw),
        ("name_exact_stripped", normalized_stripped),
        ("name_simplified", simplified_raw),
        ("name_simplified_stripped", simplified_stripped),
    ]:
        if value and (route, value) not in variants:
            variants.append((route, value))
    return variants


def fuzzy_match_name(disease_name: str | None, resources: dict[str, object]) -> tuple[str | None, str | None]:
    for _, query in query_variants(disease_name):
        tokens = informative_tokens(query)
        if not tokens:
            continue

        candidate_labels: set[str] = set()
        for token in tokens:
            prefix = token[:5] if len(token) >= 5 else token
            candidate_labels.update(resources["fuzzy_prefix_index"].get(prefix, set()))
        if not candidate_labels:
            continue

        numeric_tokens = extract_numeric_tokens(query)
        best_label: str | None = None
        best_score = 0.0
        second_score = 0.0

        for label in candidate_labels:
            label_numeric_tokens = resources["fuzzy_label_numeric_tokens"].get(label, ())
            if (numeric_tokens or label_numeric_tokens) and numeric_tokens != label_numeric_tokens:
                continue

            label_tokens = resources["fuzzy_label_tokens"].get(label, set())
            if not (tokens & label_tokens):
                prefix_overlap = any(
                    common_prefix_length(a, b) >= 6
                    for a in tokens
                    for b in label_tokens
                )
                if not prefix_overlap:
                    continue

            score = SequenceMatcher(None, query, label).ratio()
            if score > best_score:
                second_score = best_score
                best_score = score
                best_label = label
            elif score > second_score:
                second_score = score

        if best_label is None:
            continue

        mondo_ids = resources["fuzzy_label_to_mondo"].get(best_label, set())
        if len(mondo_ids) != 1:
            continue
        if best_score < FUZZY_MIN_SCORE or best_score - second_score < FUZZY_MIN_MARGIN:
            continue
        return next(iter(mondo_ids)), f"name_fuzzy:{best_score:.3f}"

    return None, None


def resolve_mondo(
    existing_mondo: str,
    disease_mim: str,
    disease_name: str,
    omim_to_mondo: dict[str, set[str]],
    resources: dict[str, object],
) -> tuple[str | None, str]:
    existing = (existing_mondo or "").strip()
    if existing:
        return existing, "existing"

    omim_raw = (disease_mim or "").strip()
    if omim_raw:
        omim_id = f"OMIM:{omim_raw}" if not omim_raw.startswith("OMIM:") else omim_raw
        omim_hits = omim_to_mondo.get(omim_id, set())
        if len(omim_hits) == 1:
            return next(iter(omim_hits)), "omim_exact"

    for route, query in query_variants(disease_name):
        mondo_id = unique_hit(resources["mondo_exact"], query)
        if mondo_id is not None:
            return mondo_id, route
        mondo_id = unique_hit(resources["mondo_simplified"], query)
        if mondo_id is not None:
            return mondo_id, route

    mondo_id, fuzzy_route = fuzzy_match_name(disease_name, resources)
    if mondo_id is not None and fuzzy_route is not None:
        return mondo_id, fuzzy_route

    return None, "unresolved"


def extract_hpo_ids(phenotypes: str | None) -> list[str]:
    if phenotypes is None:
        return []
    values = re.findall(r"HP:\d{7}", str(phenotypes))
    return ordered_unique(values)


def resolve_manual_mondo_override(disease_name: str | None) -> tuple[str | None, str | None]:
    normalized_name = str(disease_name or "").strip()
    mondo_id = DEEPRARE_MANUAL_NAME_TO_MONDO.get(normalized_name)
    if mondo_id is None:
        return None, None
    return mondo_id, "manual_override"


def write_output(rows: list[dict[str, str]], output_path: Path, sheet_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(OUTPUT_HEADERS)
    for row in rows:
        ws.append([row["case_id"], row["mondo_label"], row["hpo_id"]])
    wb.save(output_path)


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    omim_sssom = args.omim_sssom.resolve()
    mondo_json = args.mondo_json.resolve()
    output_xlsx = args.output_xlsx.resolve()

    omim_to_mondo = load_omim_to_mondo(omim_sssom)
    mondo_resources = load_mondo_name_resources(mondo_json)

    stats = Counter()
    route_counter = Counter()
    exported_rows: list[dict[str, str]] = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for source_index, row in enumerate(reader, start=1):
            stats["total_source_rows"] += 1

            hpo_ids = extract_hpo_ids(row.get("phenotypes"))
            if not hpo_ids:
                stats["discarded_missing_or_invalid_phenotypes"] += 1
                continue

            stats["rows_after_phenotype_filter"] += 1
            stats["exported_case_count"] += 1
            mondo_id, route = resolve_mondo(
                existing_mondo=row.get("disease MONDO", ""),
                disease_mim=row.get("disease mim", ""),
                disease_name=row.get("disease name", ""),
                omim_to_mondo=omim_to_mondo,
                resources=mondo_resources,
            )
            if mondo_id is None:
                manual_mondo_id, manual_route = resolve_manual_mondo_override(row.get("disease name", ""))
                if manual_mondo_id is not None and manual_route is not None:
                    mondo_id, route = manual_mondo_id, manual_route
                else:
                    route = "blank"
            route_counter[route] += 1

            if mondo_id is None:
                stats["rows_unmapped_mondo"] += 1
                mondo_id = ""
            else:
                stats["rows_mapped_mondo"] += 1

            case_id = f"case_{stats['exported_case_count']}"
            for hpo_id in hpo_ids:
                exported_rows.append(
                    {
                        "case_id": case_id,
                        "mondo_label": mondo_id,
                        "hpo_id": hpo_id,
                    }
                )

    write_output(exported_rows, output_xlsx, args.sheet_name)

    stats["exported_triplets"] = len(exported_rows)

    print(f"Output file: {output_xlsx}")
    print(f"Sheet name: {args.sheet_name}")
    print(f"Total source rows: {stats['total_source_rows']}")
    print(f"Discarded missing/invalid phenotypes: {stats['discarded_missing_or_invalid_phenotypes']}")
    print(f"Rows after phenotype filter: {stats['rows_after_phenotype_filter']}")
    print(f"Rows mapped to MONDO: {stats['rows_mapped_mondo']}")
    print(f"Rows still unmapped to MONDO: {stats['rows_unmapped_mondo']}")
    print(f"Exported case count: {stats['exported_case_count']}")
    print(f"Exported triplet row count: {stats['exported_triplets']}")
    print("Mapping route counts:")
    for route, count in sorted(route_counter.items()):
        print(f"  {route}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
