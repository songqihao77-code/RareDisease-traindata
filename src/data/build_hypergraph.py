"""统一超图构建。

负责：
- 加载静态 disease incidence
- 从病例表构建 `H_case`
- 按需构建 `H = [H_case | H_disease]`
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz

_HPO_INDEX_PATH = r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\HPO_index_v4.xlsx"
_DISEASE_INDEX_PATH = r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx"
_DISEASE_INC_PATH = r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\v60\v60DiseaseHy.npz"


def load_index_file(path: str, id_col: str, idx_col: str) -> dict:
    """读取索引文件。"""
    df = pd.read_excel(path, dtype={id_col: str, idx_col: int})
    return dict(zip(df[id_col], df[idx_col]))


def load_disease_incidence(path: str) -> csr_matrix:
    """读取疾病超边稀疏矩阵。"""
    try:
        return load_npz(path)
    except Exception:
        pass

    npz = np.load(path, allow_pickle=False)
    keys = set(npz.keys())
    row_key = next((k for k in ("rows", "row") if k in keys), None)
    col_key = next((k for k in ("cols", "col") if k in keys), None)
    value_key = next((k for k in ("vals", "data") if k in keys), None)
    if None in (row_key, col_key, value_key):
        raise ValueError(f"无法识别 npz 键名: {keys}")

    shape = tuple(npz["shape"])
    return csr_matrix((npz[value_key], (npz[row_key], npz[col_key])), shape=shape)


def build_case_incidence(
    case_df: pd.DataFrame,
    hpo_to_idx: dict,
    disease_to_idx: dict,
    case_id_col: str = "case_id",
    label_col: str = "mondo_label",
    hpo_col: str = "hpo_id",
    hpo_dropout_prob: float = 0.0,
    hpo_corruption_prob: float = 0.0,
    top_50_hpos: list[str] | None = None,
    rng: random.Random | None = None,
    verbose: bool = False,
) -> dict:
    """从病例表构建 `H_case`。"""
    if not 0.0 <= hpo_dropout_prob <= 1.0:
        raise ValueError("hpo_dropout_prob 必须位于 [0, 1]。")
    if not 0.0 <= hpo_corruption_prob <= 1.0:
        raise ValueError("hpo_corruption_prob 必须位于 [0, 1]。")

    all_hpos_pool = list(hpo_to_idx.keys()) if hpo_corruption_prob > 0.0 else []
    num_hpo = len(hpo_to_idx)
    df = case_df.dropna(subset=[case_id_col, label_col, hpo_col])

    rows: list[int] = []
    cols: list[int] = []
    case_ids: list[str] = []
    case_labels: list[str] = []
    gold_cols: list[int] = []
    skip_disease = 0
    skip_hpo = 0

    for case_id, group_df in df.groupby(case_id_col, sort=False):
        mondo_id = group_df[label_col].iloc[0]
        if mondo_id not in disease_to_idx:
            skip_disease += 1
            continue

        valid_hpos = [hpo_id for hpo_id in group_df[hpo_col].unique() if hpo_id in hpo_to_idx]
        if not valid_hpos:
            skip_hpo += 1
            continue

        # 保持原始训练语义：随机丢弃部分 HPO，但至少保留一个。
        if hpo_dropout_prob > 0.0:
            dropout_rng = rng if rng is not None else random
            kept_hpos = [hpo_id for hpo_id in valid_hpos if dropout_rng.random() > hpo_dropout_prob]
            if kept_hpos:
                valid_hpos = kept_hpos
            else:
                valid_hpos = [valid_hpos[dropout_rng.randrange(len(valid_hpos))]]

        # 保持原始训练语义：按配置注入少量噪声 HPO。
        if hpo_corruption_prob > 0.0:
            corruption_rng = rng if rng is not None else random
            if corruption_rng.random() < hpo_corruption_prob:
                num_noise = corruption_rng.randint(1, 3)
                if corruption_rng.random() < 0.5 and top_50_hpos:
                    pool_to_use = top_50_hpos
                else:
                    pool_to_use = all_hpos_pool

                noise_samples = corruption_rng.sample(pool_to_use, k=min(num_noise, len(pool_to_use)))
                for nh in noise_samples:
                    if nh not in valid_hpos:
                        valid_hpos.append(nh)

        case_col = len(case_ids)
        for hpo_id in valid_hpos:
            rows.append(hpo_to_idx[hpo_id])
            cols.append(case_col)

        case_ids.append(case_id)
        case_labels.append(mondo_id)
        gold_cols.append(disease_to_idx[mondo_id])

    num_case = len(case_ids)
    if verbose:
        print(
            f"病例：保留 {num_case}，"
            f"丢弃（疾病不在索引）{skip_disease}，"
            f"丢弃（无有效 HPO）{skip_hpo}"
        )

    h_case = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_hpo, num_case),
    )
    return {
        "H_case": h_case,
        "case_ids": case_ids,
        "case_labels": case_labels,
        "gold_disease_cols": gold_cols,
    }


def load_static_graph(
    hpo_index_path: str = _HPO_INDEX_PATH,
    disease_index_path: str = _DISEASE_INDEX_PATH,
    disease_incidence_path: str = _DISEASE_INC_PATH,
    hpo_id_col: str = "hpo_id",
    hpo_idx_col: str = "hpo_idx",
    disease_id_col: str = "mondo_id",
    disease_idx_col: str = "disease_idx",
) -> dict:
    """训练开始前加载一次静态图。"""
    hpo_to_idx = load_index_file(hpo_index_path, hpo_id_col, hpo_idx_col)
    disease_to_idx = load_index_file(disease_index_path, disease_id_col, disease_idx_col)
    num_hpo = len(hpo_to_idx)
    num_disease = len(disease_to_idx)
    print(f"静态图加载完成：HPO {num_hpo}，疾病 {num_disease}")

    h_disease = load_disease_incidence(disease_incidence_path)
    if h_disease.shape != (num_hpo, num_disease):
        raise ValueError(
            f"H_disease shape 异常: {h_disease.shape}，期望 ({num_hpo}, {num_disease})"
        )

    hpo_degrees = np.array(h_disease.sum(axis=1)).flatten()
    top_50_indices = hpo_degrees.argsort()[-50:][::-1]
    idx_to_hpo = {v: k for k, v in hpo_to_idx.items()}
    top_50_hpos = [idx_to_hpo[i] for i in top_50_indices]

    return {
        "hpo_to_idx": hpo_to_idx,
        "disease_to_idx": disease_to_idx,
        "H_disease": h_disease,
        "num_hpo": num_hpo,
        "num_disease": num_disease,
        "top_50_hpos": top_50_hpos,
    }


def build_batch_hypergraph(
    case_df: pd.DataFrame,
    hpo_to_idx: dict,
    disease_to_idx: dict,
    H_disease: csr_matrix,
    top_50_hpos: list[str] | None = None,
    case_id_col: str = "case_id",
    label_col: str = "mondo_label",
    hpo_col: str = "hpo_id",
    hpo_dropout_prob: float = 0.0,
    hpo_corruption_prob: float = 0.0,
    rng: random.Random | None = None,
    verbose: bool = False,
    include_combined_h: bool = True,
) -> dict:
    """为当前 batch 构建超图。

    默认仍返回 `H = [H_case | H_disease]` 以兼容测试和调试。
    若热路径不需要 `H`，可传 `include_combined_h=False`，跳过昂贵的 `hstack`。
    """
    num_disease = H_disease.shape[1]
    batch_before = case_df[case_id_col].nunique() if case_id_col in case_df.columns else len(case_df)
    if verbose:
        print(f"[batch] 输入病例数：{batch_before}")

    result = build_case_incidence(
        case_df=case_df,
        hpo_to_idx=hpo_to_idx,
        disease_to_idx=disease_to_idx,
        case_id_col=case_id_col,
        label_col=label_col,
        hpo_col=hpo_col,
        hpo_dropout_prob=hpo_dropout_prob,
        hpo_corruption_prob=hpo_corruption_prob,
        top_50_hpos=top_50_hpos,
        rng=rng,
        verbose=verbose,
    )
    h_case = result["H_case"]
    num_case = h_case.shape[1]

    if num_case == 0:
        raise ValueError("当前 batch 过滤后没有可用病例，请检查疾病索引或 HPO 映射。")

    h_all = None
    if include_combined_h:
        h_all = hstack([h_case, H_disease], format="csr")
        if verbose:
            print(f"[batch] H_case: {h_case.shape}  H_disease: {H_disease.shape}  H: {h_all.shape}")
    elif verbose:
        print(f"[batch] H_case: {h_case.shape}  H_disease: {H_disease.shape}  H: <skipped>")

    case_cols_global = list(range(num_case))
    disease_cols_global = list(range(num_case, num_case + num_disease))
    gold_disease_cols_global = [num_case + col for col in result["gold_disease_cols"]]

    batch_graph = {
        "H_case": h_case,
        "H_disease": H_disease,
        "case_ids": result["case_ids"],
        "case_labels": result["case_labels"],
        "case_cols_global": case_cols_global,
        "disease_cols_global": disease_cols_global,
        "gold_disease_cols_global": gold_disease_cols_global,
    }
    if h_all is not None:
        batch_graph["H"] = h_all
    return batch_graph
