"""Dataset loading and case-based batching helpers."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"
_DISEASE_INDEX_PATH = Path(
    r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx"
)

KNOWN_SOURCE_NAMES = {
    "ddd": "DDD",
    "hms": "HMS",
    "lirical": "LIRICAL",
    "mme": "MME",
    "mygene2": "MyGene2",
    "ramedis": "RAMEDIS",
    "mimic_rag_0425": "mimic_rag_0425",
    "fakedisease": "FakeDisease",
    "mimic_test": "mimic_test",
}

REAL_SOURCE_NAMES = ("DDD", "HMS", "LIRICAL", "MME", "MyGene2", "RAMEDIS")
SYNTHETIC_SOURCE_NAMES = ("FakeDisease", "mimic_rag_0425")


def load_config(config_path: Path = _CONFIG_PATH) -> dict:
    """Load the shared data config."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _natural_key(value: str) -> list[int | str]:
    """Sort strings by natural order, e.g. case_2 before case_10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def normalize_source_name(source_path: str | Path) -> str:
    """把源文件路径稳定归一化成可统计的数据源名称。"""
    stem = Path(source_path).stem.strip()
    normalized_key = stem.lower()
    return KNOWN_SOURCE_NAMES.get(normalized_key, stem)


def is_real_source(source_name: str, real_sources: tuple[str, ...] | list[str] | None = None) -> bool:
    """判断当前数据源是否属于真实数据源集合。"""
    real_source_set = set(real_sources or REAL_SOURCE_NAMES)
    return str(source_name) in real_source_set


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


def build_case_namespace(source_path: str | Path, split_namespace: str) -> str:
    """构造带 split 与相对路径信息的 case 命名空间。"""
    if not split_namespace:
        raise ValueError("split_namespace 不能为空。")

    path = Path(source_path).resolve()
    try:
        source_part = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_part = path.as_posix()
    return f"{split_namespace}::{source_part}"


def build_namespaced_case_id(
    raw_case_id: str | int,
    source_path: str | Path,
    split_namespace: str,
) -> str:
    """构造不会在 train/test 之间碰撞的 case_id。"""
    raw_case_id_text = str(raw_case_id)
    return f"{build_case_namespace(source_path, split_namespace)}::{raw_case_id_text}"


def load_namespaced_case_ids(
    file_paths: list[str | Path],
    *,
    case_id_col: str = "case_id",
    split_namespace: str,
) -> set[str]:
    """只读取 case_id，并按统一命名空间规则返回集合。"""
    case_ids: set[str] = set()
    for path in file_paths:
        df = read_case_table_file(path)
        if case_id_col not in df.columns:
            raise ValueError(f"{Path(path).name} 缺少必要列: {case_id_col}")
        for raw_case_id in df[case_id_col].dropna().astype(str).unique().tolist():
            case_ids.add(build_namespaced_case_id(raw_case_id, path, split_namespace))
    return case_ids


def load_case_files(
    file_paths: list,
    case_id_col: str = "case_id",
    label_col: str = "mondo_label",
    disease_index_path: str | Path = _DISEASE_INDEX_PATH,
    split_namespace: str = "train",
) -> pd.DataFrame:
    """
    Load and merge case files.

    每个 case_id 都会被改写成带 split 与相对路径命名空间的形式，例如：
    `train::LLLdataset/dataset/processed/train/DDD.csv::case_1`
    这样不会再因为 train/test 同 stem 文件而发生命名碰撞。
    """
    dfs = []
    for path in file_paths:
        resolved_path = Path(path).resolve()
        source_name = normalize_source_name(resolved_path)
        df = read_case_table_file(path)
        if label_col not in df.columns and "mondo_id" in df.columns:
            df = df.rename(columns={"mondo_id": label_col})

        missing_cols = [col for col in (case_id_col, label_col) if col not in df.columns]
        if missing_cols:
            missing_text = ", ".join(missing_cols)
            raise ValueError(f"{Path(path).name} 缺少必要列: {missing_text}")

        df[case_id_col] = df[case_id_col].astype(str).apply(
            lambda raw_case_id: build_namespaced_case_id(raw_case_id, path, split_namespace)
        )
        df["_case_namespace"] = build_case_namespace(path, split_namespace)
        df["_source_file"] = str(resolved_path)
        df["_source_name"] = source_name
        df["_is_real_source"] = bool(is_real_source(source_name))
        dfs.append(df)
        print(f"  已加载: {Path(path).name}, 行数: {len(df)}")

    merged = pd.concat(dfs, ignore_index=True)

    if label_col not in merged.columns:
        raise ValueError(f"合并后的病例表缺少必要列: {label_col}")

    disease2idx = load_disease_index_map(disease_index_path)
    # 这里的 gold_disease_idx 仅表示“纯 disease index 空间中的标签索引”。
    # 它不是 combined H 列号，也不是 scorer 空间中的局部索引。
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
        sampler_mode: str = "natural",
        source_balanced_target_cases: int | None = None,
        config_path: Path = _CONFIG_PATH,
    ):
        if batch_size is None:
            batch_size = load_config(config_path)["batch_size"]

        self.df = df
        self.batch_size = int(batch_size)
        self.case_id_col = case_id_col
        self.sampler_mode = str(sampler_mode)
        if self.sampler_mode not in {"natural", "source_balanced"}:
            raise ValueError(
                f"sampler_mode 只支持 'natural' 或 'source_balanced'，当前为 {self.sampler_mode!r}"
            )
        self.source_balanced_target_cases = (
            None if source_balanced_target_cases is None else int(source_balanced_target_cases)
        )
        if self.source_balanced_target_cases is not None and self.source_balanced_target_cases <= 0:
            raise ValueError("source_balanced_target_cases 必须大于 0。")

        all_ids = df[case_id_col].dropna().unique().tolist()
        self.base_case_ids: list[str] = sorted(all_ids, key=_natural_key)
        self._active_case_ids: list[str] = list(self.base_case_ids)
        self._case_source_map = self._build_case_source_map()
        self._source_case_ids = self._build_source_case_ids()
        if self.sampler_mode == "source_balanced":
            self._active_case_ids = self._build_source_balanced_case_ids(
                epoch=0,
                shuffle=False,
                random_seed=0,
            )

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

    def _build_case_source_map(self) -> dict[str, str]:
        """构建 case_id 到 source_name 的稳定映射。"""
        if "_source_name" in self.df.columns:
            source_series = self.df.groupby(self.case_id_col, sort=False)["_source_name"].first()
            return {str(case_id): str(source_name) for case_id, source_name in source_series.items()}

        if "_source_file" in self.df.columns:
            source_series = self.df.groupby(self.case_id_col, sort=False)["_source_file"].first()
            return {
                str(case_id): normalize_source_name(source_file)
                for case_id, source_file in source_series.items()
            }

        return {str(case_id): "unknown_source" for case_id in self.base_case_ids}

    def _build_source_case_ids(self) -> dict[str, list[str]]:
        """按 source_name 把 case_id 分组，供 source-balanced 采样使用。"""
        source_case_ids: dict[str, list[str]] = {}
        for case_id in self.base_case_ids:
            source_name = self._case_source_map.get(str(case_id), "unknown_source")
            source_case_ids.setdefault(source_name, []).append(str(case_id))
        return source_case_ids

    def _resolve_source_balanced_target_cases(self) -> int:
        """确定每个 source 在一个 epoch 中的目标 case 暴露量。"""
        if self.source_balanced_target_cases is not None:
            return self.source_balanced_target_cases
        if not self._source_case_ids:
            return 0
        return max(1, int(np.ceil(len(self.base_case_ids) / len(self._source_case_ids))))

    def _sample_source_case_ids(
        self,
        case_ids: list[str],
        target_count: int,
        rng: random.Random,
        shuffle: bool,
    ) -> list[str]:
        """对单个 source 做下采样或循环补样，保证目标暴露量一致。"""
        if target_count <= 0 or not case_ids:
            return []

        working_ids = list(case_ids)
        if shuffle and len(working_ids) > 1:
            rng.shuffle(working_ids)

        sampled_ids: list[str] = []
        while len(sampled_ids) < target_count:
            cycle_ids = list(working_ids)
            # 小 source 样本不足时按轮次重复采样，并在每轮重新打乱，避免固定重复顺序。
            if shuffle and len(cycle_ids) > 1:
                rng.shuffle(cycle_ids)
            remaining = target_count - len(sampled_ids)
            sampled_ids.extend(cycle_ids[:remaining])
        return sampled_ids

    def _build_source_balanced_case_ids(self, epoch: int, shuffle: bool, random_seed: int) -> list[str]:
        """按 source 做近似轮转采样，降低大 source 对 epoch 的天然主导。"""
        if not self._source_case_ids:
            return []

        rng = random.Random(random_seed + epoch)
        source_names = sorted(self._source_case_ids)
        if shuffle and len(source_names) > 1:
            rng.shuffle(source_names)

        target_count = self._resolve_source_balanced_target_cases()
        sampled_by_source = {
            source_name: self._sample_source_case_ids(
                self._source_case_ids[source_name],
                target_count=target_count,
                rng=rng,
                shuffle=shuffle,
            )
            for source_name in source_names
        }

        balanced_case_ids: list[str] = []
        for offset in range(target_count):
            for source_name in source_names:
                balanced_case_ids.append(sampled_by_source[source_name][offset])
        return balanced_case_ids

    def get_active_source_counts(self) -> dict[str, int]:
        """返回当前 epoch 顺序下各 source 的 case 数。"""
        source_counts: dict[str, int] = {}
        for case_id in self._active_case_ids:
            source_name = self._case_source_map.get(str(case_id), "unknown_source")
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
        return source_counts

    def get_sampling_summary(self) -> dict[str, Any]:
        """返回当前采样模式和 source 分布摘要，便于训练日志审查。"""
        return {
            "sampler_mode": self.sampler_mode,
            "num_cases": int(len(self._active_case_ids)),
            "source_balanced_target_cases": (
                None
                if self.sampler_mode != "source_balanced"
                else int(self._resolve_source_balanced_target_cases())
            ),
            "source_counts": self.get_active_source_counts(),
        }

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
        if self.sampler_mode == "source_balanced":
            self._active_case_ids = self._build_source_balanced_case_ids(
                epoch=epoch,
                shuffle=shuffle,
                random_seed=random_seed,
            )
        elif shuffle:
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
