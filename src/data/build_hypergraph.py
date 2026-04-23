"""超图构建。

负责：
- 加载静态 disease incidence
- 从病例表构建 `H_case`
- 按需构建兼容字段 `H = [H_case | H_disease]`

说明：
- 当前训练/评估热路径默认是 disease-only encoder，不再依赖 `H=[H_case|H_disease]`
- 这里保留 combined H 仅用于兼容旧调试接口
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz

_HPO_INDEX_PATH = r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\HPO_index_v4.xlsx"
_DISEASE_INDEX_PATH = r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx"
# Default to the current safe disease hyperedge base.
# v60 mixed dataset-case statistics into disease hyperedges and must not be used
# as the default train/eval incidence source.
_DISEASE_INC_PATH = r"D:\RareDisease-traindata\LLLdataset\DiseaseHy\rare_disease_hgnn_clean_package_v59\v59DiseaseHy.npz"

_DEFAULT_CASE_NOISE_CONTROL = {
    "enabled": False,
    "mode": "weight_only",
    "weighting": "sqrt_idf",
    "alpha": 0.5,
    "normalize_weights": True,
    "log_stats": False,
}


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


def resolve_case_noise_control(case_noise_control: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve and validate the case-side noise-control config."""
    cfg = dict(_DEFAULT_CASE_NOISE_CONTROL)
    if case_noise_control is None:
        return cfg
    if not isinstance(case_noise_control, Mapping):
        raise TypeError("case_noise_control 必须是 Mapping/dict。")

    cfg.update(dict(case_noise_control))
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["mode"] = str(cfg["mode"])
    cfg["weighting"] = str(cfg["weighting"])
    cfg["alpha"] = float(cfg["alpha"])
    cfg["normalize_weights"] = bool(cfg["normalize_weights"])
    cfg["log_stats"] = bool(cfg["log_stats"])

    if cfg["mode"] != "weight_only":
        raise ValueError("当前仅保留 case_noise_control.mode='weight_only'。")
    if cfg["weighting"] not in {"binary", "idf", "sqrt_idf", "power_idf"}:
        raise ValueError(
            "case_noise_control.weighting 只支持 'binary'、'idf'、'sqrt_idf'、'power_idf'。"
        )
    if cfg["alpha"] <= 0.0:
        raise ValueError("case_noise_control.alpha 必须是正数。")
    return cfg


def _compute_hpo_specificity(H_disease: csr_matrix) -> np.ndarray:
    """Precompute disease-side HPO specificity for case-side weighting."""
    h_disease_csr = H_disease.tocsr()
    num_disease = int(h_disease_csr.shape[1])
    hpo_df = np.diff(h_disease_csr.indptr).astype(np.float32, copy=False)
    if num_disease <= 0:
        return np.zeros_like(hpo_df, dtype=np.float32)
    return np.log((float(num_disease) + 1.0) / (hpo_df + 1.0)).astype(np.float32, copy=False)


def _build_case_hpo_weights(
    hpo_indices: list[int],
    *,
    hpo_specificity: np.ndarray,
    weighting: str,
    alpha: float,
    normalize_weights: bool,
) -> np.ndarray:
    if not hpo_indices:
        return np.zeros((0,), dtype=np.float32)

    if weighting == "binary":
        weights = np.ones((len(hpo_indices),), dtype=np.float32)
    else:
        specificity = np.clip(
            hpo_specificity[np.asarray(hpo_indices, dtype=np.int64)],
            a_min=0.0,
            a_max=None,
        ).astype(np.float32, copy=False)
        if weighting == "idf":
            weights = specificity
        elif weighting == "sqrt_idf":
            weights = np.sqrt(specificity).astype(np.float32, copy=False)
        elif weighting == "power_idf":
            weights = np.power(specificity, float(alpha)).astype(np.float32, copy=False)
        else:
            raise ValueError(f"未知 weighting 模式: {weighting}")

    if normalize_weights and weights.size > 0:
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.full((weights.size,), 1.0 / float(weights.size), dtype=np.float32)
        else:
            weights = weights / weight_sum
    return weights.astype(np.float32, copy=False)


def _compute_case_weight_entropy(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        probs = np.full((weights.size,), 1.0 / float(weights.size), dtype=np.float32)
    else:
        probs = weights / weight_sum
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _apply_case_noise_control(
    valid_hpos: list[str],
    *,
    hpo_to_idx: Mapping[str, int],
    hpo_specificity: np.ndarray | None,
    case_noise_control: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cfg = resolve_case_noise_control(case_noise_control)
    raw_hpo_count = len(valid_hpos)
    if raw_hpo_count == 0:
        return {
            "hpo_ids": [],
            "weights": np.zeros((0,), dtype=np.float32),
            "stats": {
                "enabled": bool(cfg["enabled"]),
                "mode": cfg["mode"],
                "weighting": cfg["weighting"],
                "alpha": float(cfg["alpha"]),
                "normalize_weights": bool(cfg["normalize_weights"]),
                "log_stats": bool(cfg["log_stats"]),
                "raw_hpo_count": 0,
                "kept_hpo_count": 0,
                "dropped_hpo_count": 0,
                "drop_ratio": 0.0,
                "weight_entropy": 0.0,
            },
        }

    retained_hpos = list(valid_hpos)
    if cfg["enabled"]:
        if hpo_specificity is None:
            raise ValueError("启用 case_noise_control 时必须提供 hpo_specificity。")
        retained_indices = [int(hpo_to_idx[hpo_id]) for hpo_id in retained_hpos]
        weights = _build_case_hpo_weights(
            retained_indices,
            hpo_specificity=(
                hpo_specificity if hpo_specificity is not None else np.zeros((0,), dtype=np.float32)
            ),
            weighting=str(cfg["weighting"]),
            alpha=float(cfg["alpha"]),
            normalize_weights=bool(cfg["normalize_weights"]),
        )
    else:
        weights = np.ones((len(retained_hpos),), dtype=np.float32)

    kept_hpo_count = len(retained_hpos)
    dropped_hpo_count = raw_hpo_count - kept_hpo_count
    return {
        "hpo_ids": retained_hpos,
        "weights": weights,
        "stats": {
            "enabled": bool(cfg["enabled"]),
            "mode": cfg["mode"],
            "weighting": cfg["weighting"],
            "alpha": float(cfg["alpha"]),
            "normalize_weights": bool(cfg["normalize_weights"]),
            "log_stats": bool(cfg["log_stats"]),
            "raw_hpo_count": int(raw_hpo_count),
            "kept_hpo_count": int(kept_hpo_count),
            "dropped_hpo_count": int(dropped_hpo_count),
            "drop_ratio": float(dropped_hpo_count / float(raw_hpo_count)) if raw_hpo_count else 0.0,
            "weight_entropy": _compute_case_weight_entropy(weights),
        },
    }


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
    case_noise_control: Mapping[str, Any] | None = None,
    hpo_specificity: np.ndarray | None = None,
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
    vals: list[float] = []
    case_ids: list[str] = []
    case_labels: list[str] = []
    gold_disease_idx: list[int] = []
    skip_disease = 0
    skip_hpo = 0
    noise_cfg = resolve_case_noise_control(case_noise_control)
    raw_hpo_total = 0
    kept_hpo_total = 0
    dropped_hpo_total = 0
    weight_entropy_total = 0.0
    pruned_case_count = 0

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

        controlled_case = _apply_case_noise_control(
            valid_hpos,
            hpo_to_idx=hpo_to_idx,
            hpo_specificity=hpo_specificity,
            case_noise_control=noise_cfg,
        )
        valid_hpos = controlled_case["hpo_ids"]
        valid_weights = controlled_case["weights"]
        case_noise_stats = controlled_case["stats"]
        raw_hpo_total += int(case_noise_stats["raw_hpo_count"])
        kept_hpo_total += int(case_noise_stats["kept_hpo_count"])
        dropped_hpo_total += int(case_noise_stats["dropped_hpo_count"])
        weight_entropy_total += float(case_noise_stats["weight_entropy"])
        if int(case_noise_stats["dropped_hpo_count"]) > 0:
            pruned_case_count += 1

        if not valid_hpos:
            skip_hpo += 1
            continue

        case_col = len(case_ids)
        for hpo_id, edge_weight in zip(valid_hpos, valid_weights):
            rows.append(hpo_to_idx[hpo_id])
            cols.append(case_col)
            vals.append(float(edge_weight))

        case_ids.append(case_id)
        case_labels.append(mondo_id)
        gold_disease_idx.append(disease_to_idx[mondo_id])

    num_case = len(case_ids)
    if verbose:
        print(
            f"病例：保留 {num_case}，"
            f"丢弃（疾病不在索引）{skip_disease}，"
            f"丢弃（无有效 HPO）{skip_hpo}"
        )

    h_case = csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=(num_hpo, num_case),
    )
    aggregated_case_noise_stats = {
        "enabled": bool(noise_cfg["enabled"]),
        "mode": str(noise_cfg["mode"]),
        "weighting": str(noise_cfg["weighting"]),
        "alpha": float(noise_cfg["alpha"]),
        "normalize_weights": bool(noise_cfg["normalize_weights"]),
        "log_stats": bool(noise_cfg["log_stats"]),
        "num_cases": int(num_case),
        "pruned_case_count": int(pruned_case_count),
        "raw_hpo_total": int(raw_hpo_total),
        "kept_hpo_total": int(kept_hpo_total),
        "dropped_hpo_total": int(dropped_hpo_total),
        "drop_ratio": float(dropped_hpo_total / float(raw_hpo_total)) if raw_hpo_total else 0.0,
        "mean_raw_hpo_per_case": float(raw_hpo_total / float(num_case)) if num_case else 0.0,
        "mean_kept_hpo_per_case": float(kept_hpo_total / float(num_case)) if num_case else 0.0,
        "mean_case_weight_entropy": float(weight_entropy_total / float(num_case)) if num_case else 0.0,
    }
    if verbose and aggregated_case_noise_stats["enabled"] and aggregated_case_noise_stats["log_stats"]:
        print(
            "[batch] CaseNoise "
            f"raw_hpo={aggregated_case_noise_stats['raw_hpo_total']} "
            f"kept_hpo={aggregated_case_noise_stats['kept_hpo_total']} "
            f"drop_ratio={aggregated_case_noise_stats['drop_ratio']:.4f} "
            f"mean_kept_hpo={aggregated_case_noise_stats['mean_kept_hpo_per_case']:.2f} "
            f"mean_weight_entropy={aggregated_case_noise_stats['mean_case_weight_entropy']:.4f}"
        )
    return {
        "H_case": h_case,
        "case_ids": case_ids,
        "case_labels": case_labels,
        # 这里的 gold_disease_idx 是纯 disease index 空间，不带 num_case 偏移。
        "gold_disease_idx": gold_disease_idx,
        "case_noise_stats": aggregated_case_noise_stats,
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

    hpo_specificity = _compute_hpo_specificity(h_disease)
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
        "hpo_specificity": hpo_specificity,
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
    case_noise_control: Mapping[str, Any] | None = None,
    hpo_specificity: np.ndarray | None = None,
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
        case_noise_control=case_noise_control,
        hpo_specificity=hpo_specificity,
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

    gold_disease_idx = [int(idx) for idx in result["gold_disease_idx"]]
    if len(gold_disease_idx) != num_case:
        raise ValueError("gold_disease_idx 长度必须与 num_case 一致。")
    if any(idx < 0 or idx >= num_disease for idx in gold_disease_idx):
        raise ValueError("gold_disease_idx 中存在越界 disease index。")

    # 历史兼容说明：
    # - gold_disease_idx: 疾病索引空间中的纯 disease_idx
    # - gold_disease_col_in_combined_h: 若构造 H=[H_case|H_disease]，真实疾病所在列号
    # - gold_disease_cols_global: 兼容旧字段名，语义等价于 gold_disease_col_in_combined_h
    gold_disease_col_in_combined_h = [num_case + idx for idx in gold_disease_idx]

    batch_graph = {
        "H_case": h_case,
        "H_disease": H_disease,
        "case_ids": result["case_ids"],
        "case_labels": result["case_labels"],
        "case_cols_global": case_cols_global,
        "disease_cols_global": disease_cols_global,
        "gold_disease_idx": gold_disease_idx,
        "gold_disease_col_in_combined_h": gold_disease_col_in_combined_h,
        "gold_disease_cols_global": gold_disease_col_in_combined_h,
        "case_noise_stats": result["case_noise_stats"],
    }
    if h_all is not None:
        batch_graph["H"] = h_all
    return batch_graph
