"""
数据集加载与 batch 切分程序
按 case_id 自然排序后切分为固定大小的 batch，供训练循环调用。
batch_size 统一从 configs/data.yaml 读取，整个项目只需改配置文件。
"""

import re
import yaml
import pandas as pd
from pathlib import Path
from typing import Iterator

# 配置文件路径（相对于本文件向上三级到项目根目录）
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"


# ── 1. 加载配置文件 ───────────────────────────────────────────────────────────
def load_config(config_path: Path = _CONFIG_PATH) -> dict:
    """读取 YAML 配置，返回配置 dict。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 2. 自然排序键（case_1 < case_2 < ... < case_10，而非字典序）────────────
def _natural_key(s: str):
    """将字符串中的数字段按数值排序，其余段按字符串排序。"""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]


# ── 3. 加载并合并多个病例文件 ────────────────────────────────────────────────
def load_case_files(
    file_paths: list,
    case_id_col: str = "case_id",
) -> pd.DataFrame:
    """
    读取多个 xlsx 病例文件并合并为一张表。
    自动为 case_id 添加文件名前缀（如 DDD_case_1），避免不同文件间 case_id 冲突。
    """
    dfs = []
    for path in file_paths:
        df = pd.read_excel(path, dtype=str)
        stem = Path(path).stem                        # 文件名（不含扩展名）
        df[case_id_col] = stem + "_" + df[case_id_col]
        dfs.append(df)
        print(f"  已加载：{Path(path).name}，行数：{len(df)}")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数：{len(merged)}，唯一 case_id：{merged[case_id_col].nunique()}")
    return merged


# ── 4. 按 case_id 切分 batch 的迭代器 ───────────────────────────────────────
class CaseBatchLoader:
    """
    按 case_id 自然顺序切分 batch，每次迭代返回一个 batch 的 DataFrame 子表。
    batch_size 优先使用传参值，否则从 configs/data.yaml 读取。

    用法：
        loader = CaseBatchLoader(df)
        for batch_df in loader:
            result = build_batch_hypergraph(batch_df, ...)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = None,
        case_id_col: str = "case_id",
        config_path: Path = _CONFIG_PATH,
    ):
        if batch_size is None:
            batch_size = load_config(config_path)["batch_size"]

        self.df          = df
        self.batch_size  = batch_size
        self.case_id_col = case_id_col

        # 按自然顺序排列所有唯一 case_id
        all_ids = df[case_id_col].dropna().unique().tolist()
        self.case_ids = sorted(all_ids, key=_natural_key)
        print(f"CaseBatchLoader：共 {len(self.case_ids)} 个 case_id，"
              f"batch_size={batch_size}，总 batch 数={len(self)}")

    def __len__(self) -> int:
        """总 batch 数（向上取整）。"""
        return (len(self.case_ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """每次迭代返回一个 batch 的 DataFrame 子表（包含该批 case_id 的所有行）。"""
        for i in range(0, len(self.case_ids), self.batch_size):
            batch_ids = self.case_ids[i : i + self.batch_size]
            yield self.df[self.df[self.case_id_col].isin(batch_ids)].copy()

    def get_batch(self, batch_idx: int) -> pd.DataFrame:
        """按索引取某个 batch 的 DataFrame（0-based）。"""
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError(f"batch_idx {batch_idx} 超出范围 [0, {len(self)-1}]")
        start = batch_idx * self.batch_size
        batch_ids = self.case_ids[start : start + self.batch_size]
        return self.df[self.df[self.case_id_col].isin(batch_ids)].copy()
