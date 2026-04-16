from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.incidence_builder import build_sparse_triplets, write_outputs

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "LLLdataset" / "DiseaseHy" / "processed"
DEFAULT_V4_DATA_FILE = DEFAULT_PROCESSED_DIR / "DiseaseHyperedge_data_v4.xlsx"
DEFAULT_DISEASE_INDEX_V3 = DEFAULT_PROCESSED_DIR / "Disease_index_v3.xlsx"
DEFAULT_HPO_INDEX_V3 = DEFAULT_PROCESSED_DIR / "HPO_index_v3.xlsx"
DEFAULT_DISEASE_INDEX_V4 = DEFAULT_PROCESSED_DIR / "Disease_index_v4.xlsx"
DEFAULT_HPO_INDEX_V4 = DEFAULT_PROCESSED_DIR / "HPO_index_v4.xlsx"
DEFAULT_TRIPLETS_V4_NPZ = DEFAULT_PROCESSED_DIR / "DiseaseHyperedge_sparse_triplets_v4.npz"
DEFAULT_TRIPLETS_V4_XLSX = DEFAULT_PROCESSED_DIR / "DiseaseHyperedge_sparse_triplets_v4.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 DiseaseHyperedge v4 的索引文件和 sparse triplets。")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_V4_DATA_FILE)
    parser.add_argument("--disease-index-v3", type=Path, default=DEFAULT_DISEASE_INDEX_V3)
    parser.add_argument("--hpo-index-v3", type=Path, default=DEFAULT_HPO_INDEX_V3)
    parser.add_argument("--disease-index-v4", type=Path, default=DEFAULT_DISEASE_INDEX_V4)
    parser.add_argument("--hpo-index-v4", type=Path, default=DEFAULT_HPO_INDEX_V4)
    parser.add_argument("--triplets-v4-npz", type=Path, default=DEFAULT_TRIPLETS_V4_NPZ)
    parser.add_argument("--triplets-v4-xlsx", type=Path, default=DEFAULT_TRIPLETS_V4_XLSX)
    return parser.parse_args()


def _sort_mondo_ids(mondo_ids: list[str]) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        try:
            return int(value.split(":", 1)[1]), value
        except Exception:
            return 10**18, value

    return sorted(mondo_ids, key=key)


def _build_disease_index_v4(
    data_file: Path,
    disease_index_v3: Path,
    output_file: Path,
) -> tuple[pd.DataFrame, list[str]]:
    data_df = pd.read_excel(data_file, dtype={"mondo_id": str})
    v3_df = pd.read_excel(disease_index_v3, dtype={"mondo_id": str})

    required_cols = {"mondo_id", "disease_idx"}
    missing_cols = required_cols - set(v3_df.columns)
    if missing_cols:
        raise ValueError(f"{disease_index_v3.name} 缺少必要列: {', '.join(sorted(missing_cols))}")

    v3_df = v3_df[["mondo_id", "disease_idx"]].copy()
    v3_df["disease_idx"] = v3_df["disease_idx"].astype(int)
    v3_df = v3_df.sort_values("disease_idx").reset_index(drop=True)

    expected = list(range(len(v3_df)))
    actual = v3_df["disease_idx"].tolist()
    if actual != expected:
        raise ValueError(f"{disease_index_v3.name} 的 disease_idx 必须从 0 开始连续编号。")

    existing_mondo_ids = set(v3_df["mondo_id"].astype(str))
    v4_mondo_ids = set(data_df["mondo_id"].dropna().astype(str).str.strip())
    missing_mondo_ids = _sort_mondo_ids(list(v4_mondo_ids - existing_mondo_ids))

    next_idx = int(v3_df["disease_idx"].max()) + 1 if not v3_df.empty else 0
    append_rows = [
        {"mondo_id": mondo_id, "disease_idx": next_idx + offset}
        for offset, mondo_id in enumerate(missing_mondo_ids)
    ]

    v4_df = pd.concat([v3_df, pd.DataFrame(append_rows)], ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    v4_df.to_excel(output_file, index=False)
    return v4_df, missing_mondo_ids


def _build_hpo_index_v4(
    data_file: Path,
    hpo_index_v3: Path,
    output_file: Path,
) -> tuple[pd.DataFrame, list[str]]:
    data_df = pd.read_excel(data_file, dtype={"hpo_id": str})
    v3_df = pd.read_excel(hpo_index_v3, dtype={"hpo_id": str})

    required_cols = {"hpo_id", "hpo_idx"}
    missing_cols = required_cols - set(v3_df.columns)
    if missing_cols:
        raise ValueError(f"{hpo_index_v3.name} 缺少必要列: {', '.join(sorted(missing_cols))}")

    v3_df = v3_df[["hpo_id", "hpo_idx"]].copy()
    v3_df["hpo_idx"] = v3_df["hpo_idx"].astype(int)
    v3_df = v3_df.sort_values("hpo_idx").reset_index(drop=True)

    expected = list(range(len(v3_df)))
    actual = v3_df["hpo_idx"].tolist()
    if actual != expected:
        raise ValueError(f"{hpo_index_v3.name} 的 hpo_idx 必须从 0 开始连续编号。")

    existing_hpo_ids = set(v3_df["hpo_id"].astype(str))
    v4_hpo_ids = set(data_df["hpo_id"].dropna().astype(str).str.strip())
    missing_hpo_ids = sorted(v4_hpo_ids - existing_hpo_ids)
    if missing_hpo_ids:
        next_idx = int(v3_df["hpo_idx"].max()) + 1 if not v3_df.empty else 0
        append_rows = [
            {"hpo_id": hpo_id, "hpo_idx": next_idx + offset}
            for offset, hpo_id in enumerate(missing_hpo_ids)
        ]
        v4_df = pd.concat([v3_df, pd.DataFrame(append_rows)], ignore_index=True)
    else:
        v4_df = v3_df.copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    v4_df.to_excel(output_file, index=False)
    return v4_df, missing_hpo_ids


def main() -> int:
    args = parse_args()
    data_file = args.data_file.resolve()
    disease_index_v3 = args.disease_index_v3.resolve()
    hpo_index_v3 = args.hpo_index_v3.resolve()
    disease_index_v4 = args.disease_index_v4.resolve()
    hpo_index_v4 = args.hpo_index_v4.resolve()
    triplets_v4_npz = args.triplets_v4_npz.resolve()
    triplets_v4_xlsx = args.triplets_v4_xlsx.resolve()

    disease_index_df, appended_mondo_ids = _build_disease_index_v4(
        data_file=data_file,
        disease_index_v3=disease_index_v3,
        output_file=disease_index_v4,
    )
    hpo_index_df, appended_hpo_ids = _build_hpo_index_v4(
        data_file=data_file,
        hpo_index_v3=hpo_index_v3,
        output_file=hpo_index_v4,
    )

    rows, cols, vals, shape, triplets_df, merged_df = build_sparse_triplets(
        data_file=data_file,
        disease_index_file=disease_index_v4,
        hpo_index_file=hpo_index_v4,
    )
    write_outputs(
        output_npz=triplets_v4_npz,
        output_xlsx=triplets_v4_xlsx,
        rows=rows,
        cols=cols,
        vals=vals,
        shape=shape,
        triplets_df=triplets_df,
        merged_df=merged_df,
    )

    print(f"已生成 {disease_index_v4}")
    print(f"已生成 {hpo_index_v4}")
    print(f"已生成 {triplets_v4_npz}")
    print(f"已生成 {triplets_v4_xlsx}")
    print(f"disease_index_v4_rows={len(disease_index_df)} appended_mondo={len(appended_mondo_ids)}")
    print(f"hpo_index_v4_rows={len(hpo_index_df)} appended_hpo={len(appended_hpo_ids)}")
    print(f"triplet_count={len(rows)}")
    print(f"shape=({int(shape[0])}, {int(shape[1])})")

    if appended_mondo_ids:
        print(f"新增 mondo_id 示例: {appended_mondo_ids[:10]}")
    if appended_hpo_ids:
        print(f"新增 hpo_id 示例: {appended_hpo_ids[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
