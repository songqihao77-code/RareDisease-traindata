"""Dataset loading and case-based batching helpers."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"
_DISEASE_INDEX_PATH = Path(
    r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx"
)


def load_config(config_path: Path = _CONFIG_PATH) -> dict:
    """Load the shared data config."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _natural_key(value: str) -> list[int | str]:
    """Sort strings by natural order, e.g. case_2 before case_10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def load_disease_index_map(index_path: str | Path) -> dict[str, int]:
    """Load the MONDO to disease index mapping."""
    index_path = Path(index_path)
    df = pd.read_excel(index_path, dtype={"mondo_id": str})

    required_cols = {"mondo_id", "disease_idx"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_text = ", ".join(sorted(missing_cols))
        raise ValueError(f"{index_path.name} 缺少必要列: {missing_text}")

    df = df[["mondo_id", "disease_idx"]].copy()
    df["disease_idx"] = df["disease_idx"].astype(int)
    return dict(zip(df["mondo_id"], df["disease_idx"]))


def read_case_table_file(path: str | Path) -> pd.DataFrame:
    """Read a case table from either Excel or CSV."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError(f"不支持的病例文件格式: {path}")


def load_case_files(
    file_paths: list,
    case_id_col: str = "case_id",
    label_col: str = "mondo_label",
    disease_index_path: str | Path = _DISEASE_INDEX_PATH,
) -> pd.DataFrame:
    """
    Load and merge case files.

    Each case id is prefixed with the source filename stem so that ids from
    different datasets do not collide after concatenation.
    """
    dfs = []
    for path in file_paths:
        df = read_case_table_file(path)
        if label_col not in df.columns and "mondo_id" in df.columns:
            df = df.rename(columns={"mondo_id": label_col})

        missing_cols = [col for col in (case_id_col, label_col) if col not in df.columns]
        if missing_cols:
            missing_text = ", ".join(missing_cols)
            raise ValueError(f"{Path(path).name} 缺少必要列: {missing_text}")

        stem = Path(path).stem
        df[case_id_col] = stem + "_" + df[case_id_col]
        dfs.append(df)
        print(f"  已加载: {Path(path).name}, 行数: {len(df)}")

    merged = pd.concat(dfs, ignore_index=True)

    if label_col not in merged.columns:
        raise ValueError(f"合并后的病例表缺少必要列: {label_col}")

    disease2idx = load_disease_index_map(disease_index_path)
    merged["gold_disease_idx"] = merged[label_col].map(disease2idx)

    missing_mask = merged["gold_disease_idx"].isna()
    if missing_mask.any():
        dropped_rows = int(missing_mask.sum())
        dropped_cases = merged.loc[missing_mask, case_id_col].dropna().astype(str).nunique()
        missing_mondo = merged.loc[missing_mask, label_col].drop_duplicates().head(10).tolist()
        missing_mondo = ["<缺失值>" if pd.isna(x) else str(x) for x in missing_mondo]
        print(
            f"[WARN] 跳过不在疾病索引中的标签: rows={dropped_rows}, "
            f"cases={dropped_cases}, sample_labels={missing_mondo}"
        )
        merged = merged.loc[~missing_mask].copy()

    if merged.empty:
        raise ValueError("过滤索引外疾病标签后，训练数据为空。请检查 disease_index 与训练数据。")

    merged["gold_disease_idx"] = merged["gold_disease_idx"].astype(int)
    print(f"合并后总行数: {len(merged)}, 唯一 case_id: {merged[case_id_col].nunique()}")
    print(f"已生成 gold_disease_idx，唯一疾病索引数: {merged['gold_disease_idx'].nunique()}")
    return merged


class CaseBatchLoader:
    """Split a dataframe into batches while keeping each case together."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int | None = None,
        case_id_col: str = "case_id",
        config_path: Path = _CONFIG_PATH,
    ):
        if batch_size is None:
            batch_size = load_config(config_path)["batch_size"]

        self.df = df
        self.batch_size = batch_size
        self.case_id_col = case_id_col

        all_ids = df[case_id_col].dropna().unique().tolist()
        self.base_case_ids: list[str] = sorted(all_ids, key=_natural_key)
        self._active_case_ids: list[str] = list(self.base_case_ids)

        # 预先缓存每个 case 对应的行号，避免每个 step 都做 pandas.isin。
        self._case_row_indices: dict[str, np.ndarray] = {
            str(case_id): np.asarray(row_idx, dtype=np.int64)
            for case_id, row_idx in df.groupby(case_id_col, sort=False).indices.items()
        }
        self._active_batch_row_indices: list[np.ndarray] = []
        self._rebuild_active_batches()
        print(
            f"CaseBatchLoader: 共 {len(self.base_case_ids)} 个 case_id, "
            f"batch_size={batch_size}, 总 batch 数={len(self)}"
        )

    @property
    def case_ids(self) -> list[str]:
        """Return the case order for the current epoch."""
        return self._active_case_ids

    def _rebuild_active_batches(self) -> None:
        """按当前 epoch 的 case 顺序预构建每个 batch 的行号列表。

        这里显式按原始行号排序，保持与旧实现
        `df[df[self.case_id_col].isin(batch_ids)]` 一样的行顺序，
        从而不改变 batch 内病例顺序语义。
        """
        batch_rows: list[np.ndarray] = []
        for start in range(0, len(self._active_case_ids), self.batch_size):
            batch_ids = self._active_case_ids[start : start + self.batch_size]
            row_parts = [self._case_row_indices[str(case_id)] for case_id in batch_ids]
            if not row_parts:
                batch_rows.append(np.empty((0,), dtype=np.int64))
                continue
            row_idx = np.concatenate(row_parts)
            row_idx.sort()
            batch_rows.append(row_idx)
        self._active_batch_row_indices = batch_rows

    def set_epoch(self, epoch: int, shuffle: bool, random_seed: int) -> None:
        """Update case order for the current epoch."""
        if shuffle:
            rng = random.Random(random_seed + epoch)
            ids = list(self.base_case_ids)
            rng.shuffle(ids)
            self._active_case_ids = ids
        else:
            self._active_case_ids = list(self.base_case_ids)
        self._rebuild_active_batches()

    def __len__(self) -> int:
        """Return the total number of batches."""
        return len(self._active_batch_row_indices)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Yield one dataframe per case batch."""
        for row_idx in self._active_batch_row_indices:
            yield self.df.iloc[row_idx].copy()

    def get_batch(self, batch_idx: int) -> pd.DataFrame:
        """Return a single batch by 0-based index."""
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError(f"batch_idx {batch_idx} 超出范围 [0, {len(self) - 1}]")
        return self.df.iloc[self._active_batch_row_indices[batch_idx]].copy()
