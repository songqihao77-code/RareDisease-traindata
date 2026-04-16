from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_DIR = Path(r"D:\RareDisease-traindata\LLLdataset\dataset")
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "processed"
DEFAULT_ORPHA_TO_MONDO_PATH = Path(
    r"D:\DeepRare-main\database\mondo_hasdbxref_orphanet.sssom.tsv"
)
DEFAULT_OMIM_TO_MONDO_PATH = Path(r"D:\DeepRare-main\database\mondo_exactmatch_omim.sssom.tsv")
DEFAULT_ORPHA_TO_OMIM_PATH = Path(r"D:\DeepRare-main\database\orpha2omim.json")
DEFAULT_ORPHA_TO_NAME_PATH = Path(r"D:\DeepRare-main\database\orpha2name.json")
DEFAULT_MONDO_RARE_PATH = Path(r"D:\DeepRare-main\database\mondo-rare.json")

NAME_PREFIXES_TO_STRIP = ("obsolete:", "non rare in europe:")
OUTPUT_COLUMNS = ["case_id", "mondo_label", "hpo_id"]

# 这批 ORPHA 在本地 MONDO 资源里没有可直接命中的 xref，
# 但疾病名称能稳定落到更宽但语义明确的 MONDO 词条。
MANUAL_ORPHA_TO_MONDO_OVERRIDES = {
    "ORPHA:2284": "MONDO:0003780",   # Primary T cell immunodeficiency -> T-cell immunodeficiency
    "ORPHA:3274": "MONDO:0008523",   # Granulomatous arthritis of childhood -> Blau syndrome / early-onset sarcoidosis
    "ORPHA:238691": "MONDO:0002404",  # Congenital liver hemangioma -> liver hemangioma
    "ORPHA:564127": "MONDO:0005377",  # Genetic nephrotic syndrome -> nephrotic syndrome
    "ORPHA:567560": "MONDO:0018882",  # Systemic vasculitis associated with glomerulopathy -> vasculitis
    "ORPHA:686462": "MONDO:0017853",  # Non-fibrotic hypersensitivity pneumonitis -> hypersensitivity pneumonitis
    "ORPHA:689001": "MONDO:0006061",  # Isolated spontaneous vertebral artery dissection -> cervical artery dissection
    "ORPHA:95501": "MONDO:0015790",   # Congenital central diabetes insipidus -> central diabetes insipidus
    "ORPHA:97569": "MONDO:0002462",   # Unclassified glomerulonephritis -> glomerulonephritis
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_path: Path
    output_path: Path
    case_id_col: str
    disease_col: str
    hpo_col: str
    mode: str


@dataclass(frozen=True)
class MappingBundle:
    orpha_to_mondo: dict[str, str]
    omim_to_mondo: dict[str, str]
    orpha_to_omim: dict[str, str]
    orpha_to_name: dict[str, str]
    disease_name_to_mondo: dict[str, list[str]]
    manual_orpha_to_mondo: dict[str, str]


@dataclass(frozen=True)
class ResolutionResult:
    mondo_labels: list[str]
    missing_orpha_ids: set[str]
    methods: set[str]
    has_orpha: bool
    has_omim: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 LLLdataset 的 mimic / ddd 原始 CSV 处理成 MONDO-HPO 长表 CSV。"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="原始数据目录，默认指向 LLLdataset/dataset。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="处理后输出目录，默认指向 LLLdataset/dataset/processed。",
    )
    parser.add_argument(
        "--orpha-to-mondo",
        type=Path,
        default=DEFAULT_ORPHA_TO_MONDO_PATH,
        help="Orphanet -> MONDO 映射表路径（SSSOM TSV）。",
    )
    parser.add_argument(
        "--omim-to-mondo",
        type=Path,
        default=DEFAULT_OMIM_TO_MONDO_PATH,
        help="OMIM -> MONDO 映射表路径（SSSOM TSV）。",
    )
    parser.add_argument(
        "--orpha-to-omim",
        type=Path,
        default=DEFAULT_ORPHA_TO_OMIM_PATH,
        help="Orphanet -> OMIM 映射表路径（JSON）。",
    )
    parser.add_argument(
        "--orpha-to-name",
        type=Path,
        default=DEFAULT_ORPHA_TO_NAME_PATH,
        help="Orphanet -> 疾病名映射表路径（JSON）。",
    )
    parser.add_argument(
        "--mondo-rare",
        type=Path,
        default=DEFAULT_MONDO_RARE_PATH,
        help="MONDO 稀有病本体文件路径（JSON）。",
    )
    return parser.parse_args()


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_string_list(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return re.findall(r"[A-Za-z]+:\d+", text)


def normalize_prefixed_id(raw_value: str | None, prefix: str, digits: int = 7) -> str | None:
    if raw_value is None:
        return None

    match = re.search(rf"{re.escape(prefix)}[:_]?(\d+)", str(raw_value), flags=re.IGNORECASE)
    if not match:
        return None
    if digits == 0:
        return f"{prefix}:{match.group(1)}"
    return f"{prefix}:{match.group(1).zfill(digits)}"


def normalize_orpha_key(raw_value: str | None) -> str | None:
    normalized = normalize_prefixed_id(raw_value, "ORPHA", digits=0)
    if normalized is None:
        normalized = normalize_prefixed_id(raw_value, "Orphanet", digits=0)
    if normalized is None:
        return None
    return normalized.split(":", 1)[1]


def make_orpha_id(orpha_key: str) -> str:
    return f"ORPHA:{orpha_key}"


def extract_hpo_ids(raw_value: object) -> list[str]:
    hpo_ids: list[str] = []
    for token in parse_string_list(raw_value):
        normalized = normalize_prefixed_id(token, "HP")
        if normalized is not None:
            hpo_ids.append(normalized)
    return ordered_unique(hpo_ids)


def normalize_disease_name(raw_value: str | None) -> str:
    if raw_value is None:
        return ""

    text = str(raw_value).strip().lower()
    for prefix in NAME_PREFIXES_TO_STRIP:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_specific_lookup_key(lookup_key: str) -> bool:
    parts = lookup_key.split()
    return len(parts) >= 2 or any(char.isdigit() for char in lookup_key)


def iter_disease_name_lookup_keys(raw_name: str | None) -> list[str]:
    if raw_name is None:
        return []

    raw_text = str(raw_name).strip()
    if not raw_text:
        return []

    candidates = [raw_text]
    for splitter in ("/", ";", "|"):
        expanded: list[str] = []
        for candidate in candidates:
            expanded.extend(part.strip() for part in candidate.split(splitter) if part.strip())
        candidates.extend(expanded)

    keys: list[str] = []
    for candidate in candidates:
        lookup_key = normalize_disease_name(candidate)
        if lookup_key and is_specific_lookup_key(lookup_key):
            keys.append(lookup_key)
    return ordered_unique(keys)


def add_name_to_mondo_index(
    index: defaultdict[str, set[str]],
    raw_name: str | None,
    mondo_id: str | None,
) -> None:
    if mondo_id is None:
        return
    for lookup_key in iter_disease_name_lookup_keys(raw_name):
        index[lookup_key].add(mondo_id)


def load_json(raw_path: Path) -> object:
    with raw_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_orpha_to_mondo_map(mapping_path: Path) -> dict[str, str]:
    df = pd.read_csv(mapping_path, sep="\t", dtype=str)
    required_columns = {"subject_id", "object_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{mapping_path.name} 缺少必要列: {missing_text}")

    mapping: dict[str, str] = {}
    for row in df.itertuples(index=False):
        orpha_key = normalize_orpha_key(getattr(row, "object_id", None))
        mondo_id = normalize_prefixed_id(getattr(row, "subject_id", None), "MONDO")
        if orpha_key is None or mondo_id is None:
            continue
        mapping[orpha_key] = mondo_id

    if not mapping:
        raise ValueError(f"{mapping_path} 未解析出任何有效的 ORPHA -> MONDO 映射。")
    return mapping


def load_omim_to_mondo_map(mapping_path: Path) -> dict[str, str]:
    df = pd.read_csv(mapping_path, sep="\t", dtype=str)
    required_columns = {"subject_id", "object_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{mapping_path.name} 缺少必要列: {missing_text}")

    mapping: dict[str, str] = {}
    for row in df.itertuples(index=False):
        omim_id = normalize_prefixed_id(getattr(row, "object_id", None), "OMIM", digits=0)
        mondo_id = normalize_prefixed_id(getattr(row, "subject_id", None), "MONDO")
        if omim_id is None or mondo_id is None:
            continue
        mapping[omim_id] = mondo_id

    if not mapping:
        raise ValueError(f"{mapping_path} 未解析出任何有效的 OMIM -> MONDO 映射。")
    return mapping


def load_orpha_to_omim_map(mapping_path: Path) -> dict[str, str]:
    raw_mapping = load_json(mapping_path)
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{mapping_path} 不是预期的 JSON 对象。")

    mapping: dict[str, str] = {}
    for raw_orpha_id, raw_omim_id in raw_mapping.items():
        orpha_key = normalize_orpha_key(str(raw_orpha_id))
        omim_id = normalize_prefixed_id(str(raw_omim_id), "OMIM", digits=0)
        if orpha_key is None or omim_id is None:
            continue
        mapping[make_orpha_id(orpha_key)] = omim_id
    return mapping


def load_orpha_to_name_map(orpha_name_path: Path, orpha_mapping_path: Path) -> dict[str, str]:
    raw_mapping = load_json(orpha_name_path)
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{orpha_name_path} 不是预期的 JSON 对象。")

    mapping: dict[str, str] = {}
    for raw_orpha_id, raw_name in raw_mapping.items():
        orpha_key = normalize_orpha_key(str(raw_orpha_id))
        if orpha_key is None:
            continue
        name = str(raw_name).strip()
        if name:
            mapping[make_orpha_id(orpha_key)] = name

    sssom_df = pd.read_csv(orpha_mapping_path, sep="\t", dtype=str)
    for row in sssom_df.itertuples(index=False):
        orpha_key = normalize_orpha_key(getattr(row, "object_id", None))
        object_label = getattr(row, "object_label", None)
        if orpha_key is None or not isinstance(object_label, str) or not object_label.strip():
            continue
        mapping.setdefault(make_orpha_id(orpha_key), object_label.strip())

    if not mapping:
        raise ValueError(f"{orpha_name_path} 与 {orpha_mapping_path} 未解析出任何有效的 ORPHA 名称。")
    return mapping


def load_disease_name_to_mondo_map(
    mondo_rare_path: Path,
    orpha_mapping_path: Path,
    omim_mapping_path: Path,
) -> dict[str, list[str]]:
    index: defaultdict[str, set[str]] = defaultdict(set)

    mondo_rare = load_json(mondo_rare_path)
    graphs = mondo_rare.get("graphs", []) if isinstance(mondo_rare, dict) else []
    nodes = graphs[0].get("nodes", []) if graphs else []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        mondo_id = normalize_prefixed_id(node.get("id"), "MONDO")
        if mondo_id is None:
            continue

        add_name_to_mondo_index(index, node.get("lbl"), mondo_id)
        for synonym in node.get("meta", {}).get("synonyms", []):
            if isinstance(synonym, dict):
                add_name_to_mondo_index(index, synonym.get("val"), mondo_id)

    for mapping_path in (orpha_mapping_path, omim_mapping_path):
        df = pd.read_csv(mapping_path, sep="\t", dtype=str)
        for row in df.itertuples(index=False):
            mondo_id = normalize_prefixed_id(getattr(row, "subject_id", None), "MONDO")
            add_name_to_mondo_index(index, getattr(row, "subject_label", None), mondo_id)
            add_name_to_mondo_index(index, getattr(row, "object_label", None), mondo_id)

    if not index:
        raise ValueError("疾病名称索引为空，无法执行名称到 MONDO 的补映射。")
    return {key: sorted(value) for key, value in index.items()}


def build_mapping_bundle(args: argparse.Namespace) -> MappingBundle:
    orpha_to_mondo_path = args.orpha_to_mondo.resolve()
    omim_to_mondo_path = args.omim_to_mondo.resolve()
    orpha_to_omim_path = args.orpha_to_omim.resolve()
    orpha_to_name_path = args.orpha_to_name.resolve()
    mondo_rare_path = args.mondo_rare.resolve()

    return MappingBundle(
        orpha_to_mondo=load_orpha_to_mondo_map(orpha_to_mondo_path),
        omim_to_mondo=load_omim_to_mondo_map(omim_to_mondo_path),
        orpha_to_omim=load_orpha_to_omim_map(orpha_to_omim_path),
        orpha_to_name=load_orpha_to_name_map(orpha_to_name_path, orpha_to_mondo_path),
        disease_name_to_mondo=load_disease_name_to_mondo_map(
            mondo_rare_path=mondo_rare_path,
            orpha_mapping_path=orpha_to_mondo_path,
            omim_mapping_path=omim_to_mondo_path,
        ),
        manual_orpha_to_mondo=MANUAL_ORPHA_TO_MONDO_OVERRIDES.copy(),
    )


def lookup_mondo_ids_by_name(
    raw_name: str | None,
    disease_name_to_mondo: dict[str, list[str]],
) -> list[str]:
    mondo_ids: list[str] = []
    for lookup_key in iter_disease_name_lookup_keys(raw_name):
        mondo_ids.extend(disease_name_to_mondo.get(lookup_key, []))
    return ordered_unique(mondo_ids)


def resolve_orpha_to_mondo(
    orpha_key: str,
    mappings: MappingBundle,
) -> tuple[str | None, str | None]:
    if mapped := mappings.orpha_to_mondo.get(orpha_key):
        return mapped, "orpha_to_mondo"

    orpha_id = make_orpha_id(orpha_key)
    omim_id = mappings.orpha_to_omim.get(orpha_id)
    if omim_id is not None and (mapped := mappings.omim_to_mondo.get(omim_id)) is not None:
        return mapped, "orpha_via_omim_to_mondo"

    if mapped := mappings.manual_orpha_to_mondo.get(orpha_id):
        return mapped, "manual_orpha_to_mondo"

    candidate_name = mappings.orpha_to_name.get(orpha_id)
    mondo_ids = lookup_mondo_ids_by_name(candidate_name, mappings.disease_name_to_mondo)
    if len(mondo_ids) == 1:
        return mondo_ids[0], "orpha_name_to_mondo"
    if len(mondo_ids) > 1:
        return None, "orpha_name_ambiguous"

    return None, None


def resolve_case_mondo_labels(raw_value: object, mappings: MappingBundle) -> ResolutionResult:
    raw_tokens = parse_string_list(raw_value)
    has_orpha = any(normalize_orpha_key(token) is not None for token in raw_tokens)
    has_omim = any(
        normalize_prefixed_id(token, "OMIM", digits=0) is not None for token in raw_tokens
    )

    mondo_labels = ordered_unique(
        [
            normalized
            for token in raw_tokens
            if (normalized := normalize_prefixed_id(token, "MONDO")) is not None
        ]
    )
    if mondo_labels:
        return ResolutionResult(
            mondo_labels=mondo_labels,
            missing_orpha_ids=set(),
            methods={"input_mondo"},
            has_orpha=has_orpha,
            has_omim=has_omim,
        )

    resolved_mondo_labels: list[str] = []
    methods: set[str] = set()
    missing_orpha_ids: set[str] = set()

    for token in raw_tokens:
        omim_id = normalize_prefixed_id(token, "OMIM", digits=0)
        if omim_id is None:
            continue
        mapped = mappings.omim_to_mondo.get(omim_id)
        if mapped is None:
            continue
        resolved_mondo_labels.append(mapped)
        methods.add("omim_to_mondo")

    for token in raw_tokens:
        orpha_key = normalize_orpha_key(token)
        if orpha_key is None:
            continue
        mapped, method = resolve_orpha_to_mondo(orpha_key, mappings)
        if method is not None:
            methods.add(method)
        if mapped is None:
            missing_orpha_ids.add(orpha_key)
            continue
        resolved_mondo_labels.append(mapped)

    return ResolutionResult(
        mondo_labels=ordered_unique(resolved_mondo_labels),
        missing_orpha_ids=missing_orpha_ids,
        methods=methods,
        has_orpha=has_orpha,
        has_omim=has_omim,
    )


def build_dataset_specs(dataset_dir: Path, output_dir: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="mimic_rag_0425",
            input_path=dataset_dir / "mimic_rag_0425.csv",
            output_path=output_dir / "mimic_rag_0425.csv",
            case_id_col="note_id",
            disease_col="orpha",
            hpo_col="HPO",
            mode="mimic",
        ),
        DatasetSpec(
            name="mimic_test",
            input_path=dataset_dir / "mimic_test.csv",
            output_path=output_dir / "mimic_test.csv",
            case_id_col="note_id",
            disease_col="orpha",
            hpo_col="HPO",
            mode="mimic",
        ),
        DatasetSpec(
            name="ddd_test",
            input_path=dataset_dir / "ddd_test.csv",
            output_path=output_dir / "ddd_test.csv",
            case_id_col="id",
            disease_col="rare_disease",
            hpo_col="phenotype",
            mode="ddd",
        ),
    ]


def process_dataset(spec: DatasetSpec, mappings: MappingBundle) -> dict[str, object]:
    df = pd.read_csv(spec.input_path, dtype=str)
    required_columns = {spec.case_id_col, spec.disease_col, spec.hpo_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{spec.input_path.name} 缺少必要列: {missing_text}")

    stats = Counter()
    missing_orpha_ids: set[str] = set()
    case_id_map: dict[str, str] = {}
    output_rows: list[dict[str, str]] = []

    for row in df.itertuples(index=False):
        stats["source_rows"] += 1
        source_case_id = str(getattr(row, spec.case_id_col, "")).strip()
        if not source_case_id:
            stats["dropped_missing_case_id"] += 1
            continue

        hpo_ids = extract_hpo_ids(getattr(row, spec.hpo_col, None))
        if not hpo_ids:
            stats["dropped_no_hpo"] += 1
            continue

        resolution = resolve_case_mondo_labels(getattr(row, spec.disease_col, None), mappings)
        if resolution.has_orpha:
            stats["rows_with_orpha_input"] += 1
        if resolution.has_omim:
            stats["rows_with_omim_input"] += 1
        for method in resolution.methods:
            stats[f"rows_with_{method}"] += 1
        if resolution.missing_orpha_ids:
            missing_orpha_ids.update(resolution.missing_orpha_ids)
            stats["rows_with_unmapped_orpha"] += 1

        if not resolution.mondo_labels:
            stats["dropped_no_mondo"] += 1
            if spec.mode == "ddd":
                if resolution.has_omim and not resolution.has_orpha:
                    stats["ddd_dropped_omim_only_or_non_mondo"] += 1
                elif resolution.has_orpha:
                    stats["ddd_dropped_unmapped_orpha_only"] += 1
            continue

        normalized_case_id = case_id_map.setdefault(
            source_case_id,
            f"case_{len(case_id_map) + 1}",
        )

        stats["retained_cases"] += 1
        if len(resolution.mondo_labels) > 1:
            stats["multi_label_cases"] += 1

        for mondo_label in resolution.mondo_labels:
            for hpo_id in hpo_ids:
                output_rows.append(
                    {
                        "case_id": normalized_case_id,
                        "mondo_label": mondo_label,
                        "hpo_id": hpo_id,
                    }
                )
                stats["exported_triplets"] += 1

    output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(spec.output_path, index=False, encoding="utf-8-sig")

    stats["unique_case_count"] = int(output_df["case_id"].nunique()) if not output_df.empty else 0
    stats["unique_mondo_count"] = (
        int(output_df["mondo_label"].nunique()) if not output_df.empty else 0
    )
    stats["unique_hpo_count"] = int(output_df["hpo_id"].nunique()) if not output_df.empty else 0

    return {
        "spec": spec,
        "stats": dict(stats),
        "missing_orpha_ids": sorted(missing_orpha_ids, key=lambda value: int(value)),
        "output_rows": len(output_df),
    }


def print_summary(result: dict[str, object]) -> None:
    spec: DatasetSpec = result["spec"]  # type: ignore[assignment]
    stats: dict[str, int] = result["stats"]  # type: ignore[assignment]
    missing_orpha_ids: list[str] = result["missing_orpha_ids"]  # type: ignore[assignment]

    print(f"[{spec.name}] 输出文件: {spec.output_path}")
    print(f"  source_rows={stats.get('source_rows', 0)}")
    print(f"  retained_cases={stats.get('retained_cases', 0)}")
    print(f"  dropped_missing_case_id={stats.get('dropped_missing_case_id', 0)}")
    print(f"  dropped_no_hpo={stats.get('dropped_no_hpo', 0)}")
    print(f"  dropped_no_mondo={stats.get('dropped_no_mondo', 0)}")
    print(f"  multi_label_cases={stats.get('multi_label_cases', 0)}")
    print(f"  exported_triplets={stats.get('exported_triplets', 0)}")
    print(f"  unique_case_count={stats.get('unique_case_count', 0)}")
    print(f"  unique_mondo_count={stats.get('unique_mondo_count', 0)}")
    print(f"  unique_hpo_count={stats.get('unique_hpo_count', 0)}")
    print(f"  rows_with_orpha_input={stats.get('rows_with_orpha_input', 0)}")
    print(f"  rows_with_omim_input={stats.get('rows_with_omim_input', 0)}")
    print(f"  rows_with_input_mondo={stats.get('rows_with_input_mondo', 0)}")
    print(f"  rows_with_omim_to_mondo={stats.get('rows_with_omim_to_mondo', 0)}")
    print(f"  rows_with_orpha_to_mondo={stats.get('rows_with_orpha_to_mondo', 0)}")
    print(
        "  rows_with_orpha_via_omim_to_mondo="
        f"{stats.get('rows_with_orpha_via_omim_to_mondo', 0)}"
    )
    print(f"  rows_with_manual_orpha_to_mondo={stats.get('rows_with_manual_orpha_to_mondo', 0)}")
    print(f"  rows_with_orpha_name_to_mondo={stats.get('rows_with_orpha_name_to_mondo', 0)}")
    print(f"  rows_with_orpha_name_ambiguous={stats.get('rows_with_orpha_name_ambiguous', 0)}")
    print(f"  rows_with_unmapped_orpha={stats.get('rows_with_unmapped_orpha', 0)}")
    if spec.mode == "ddd":
        print(
            "  ddd_dropped_omim_only_or_non_mondo="
            f"{stats.get('ddd_dropped_omim_only_or_non_mondo', 0)}"
        )
        print(
            "  ddd_dropped_unmapped_orpha_only="
            f"{stats.get('ddd_dropped_unmapped_orpha_only', 0)}"
        )
    if missing_orpha_ids:
        preview = ", ".join(f"ORPHA:{value}" for value in missing_orpha_ids[:20])
        print(f"  unmapped_orpha_unique={len(missing_orpha_ids)}")
        print(f"  unmapped_orpha_preview={preview}")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    mappings = build_mapping_bundle(args)
    dataset_specs = build_dataset_specs(dataset_dir, output_dir)

    for spec in dataset_specs:
        result = process_dataset(spec, mappings)
        print_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
