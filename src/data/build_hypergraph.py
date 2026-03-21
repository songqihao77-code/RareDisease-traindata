"""
统一超图构建程序
只负责构建 H = [H_case | H_disease]，不涉及 HGNN 传播 / 训练 / loss。
病例文件为长格式：每行 = 一个病例的一个 HPO 术语。

对外接口：
  load_static_graph(...)      → 训练开始时调用一次，缓存静态图
  build_batch_hypergraph(...) → 训练每个 batch 时调用，动态构建 H_batch
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz

# ── 默认路径（知识库文件固定不变）────────────────────────────────────────────
_HPO_INDEX_PATH      = r"D:\RareDisease\data\processed\knowledge\HPO_index_v2.xlsx"
_DISEASE_INDEX_PATH  = r"D:\RareDisease\data\processed\knowledge\Disease_index_v2.xlsx"
_DISEASE_INC_PATH    = r"D:\RareDisease\data\processed\knowledge\DiseaseHyperedge_sparse_triplets_v2.npz"


# ── 1. 读取索引文件 ───────────────────────────────────────────────────────────
def load_index_file(path: str, id_col: str, idx_col: str) -> dict:
    """读取 Excel 索引，返回 {id字符串 -> 整数索引} 字典。"""
    df = pd.read_excel(path, dtype={id_col: str, idx_col: int})
    return dict(zip(df[id_col], df[idx_col]))


# ── 2. 加载疾病超边稀疏矩阵 ──────────────────────────────────────────────────
def load_disease_incidence(path: str) -> csr_matrix:
    """
    兼容两种 npz 格式：
      ① 标准 scipy sparse npz → 直接 load_npz
      ② triplets npz → 含 rows/cols/vals/shape（或 row/col/data）字段
    """
    try:
        return load_npz(path)
    except Exception:
        pass  # 不是标准格式，尝试 triplets

    npz  = np.load(path, allow_pickle=False)
    keys = set(npz.keys())
    r = next((k for k in ("rows", "row") if k in keys), None)
    c = next((k for k in ("cols", "col") if k in keys), None)
    v = next((k for k in ("vals", "data")  if k in keys), None)
    if None in (r, c, v):
        raise ValueError(f"无法识别 npz 键名 {keys}，期望 rows/cols/vals 或 row/col/data")
    shape = tuple(npz["shape"])
    return csr_matrix((npz[v], (npz[r], npz[c])), shape=shape)


# ── 3. 从 DataFrame 构建 H_case（供内部复用）─────────────────────────────────
def build_case_incidence(
    case_df: pd.DataFrame,           # 长格式病例表（每行 = 一个 HPO）
    hpo_to_idx: dict,
    disease_to_idx: dict,
    case_id_col: str = "case_id",
    label_col: str   = "mondo_label",
    hpo_col: str     = "hpo_id",
) -> dict:
    """
    从长格式病例 DataFrame 构建二值病例超边矩阵 H_case。
    丢弃：疾病不在索引中 / 映射后无有效 HPO 的病例。
    """
    num_hpo = len(hpo_to_idx)
    df = case_df.dropna(subset=[case_id_col, label_col, hpo_col])

    rows, cols = [], []
    case_ids, case_labels, gold_cols = [], [], []
    skip_disease = skip_hpo = 0

    for cid, grp in df.groupby(case_id_col, sort=False):
        mondo = grp[label_col].iloc[0]
        if mondo not in disease_to_idx:          # 疾病不在索引中
            skip_disease += 1
            continue

        valid = [h for h in grp[hpo_col].unique() if h in hpo_to_idx]
        if not valid:                             # 无有效 HPO
            skip_hpo += 1
            continue

        case_col = len(case_ids)                 # 当前病例在 H_case 中的列索引
        for h in valid:
            rows.append(hpo_to_idx[h])
            cols.append(case_col)

        case_ids.append(cid)
        case_labels.append(mondo)
        gold_cols.append(disease_to_idx[mondo])  # H_disease 列空间下的索引

    num_case = len(case_ids)
    print(f"  病例：保留 {num_case}，"
          f"丢弃（疾病不在索引）{skip_disease}，"
          f"丢弃（无有效HPO）{skip_hpo}")

    H_case = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_hpo, num_case),
    )
    return {
        "H_case":      H_case,
        "case_ids":    case_ids,
        "case_labels": case_labels,
        "gold_disease_cols": gold_cols,   # 后续加 offset 转为全局列索引
    }


# ── 4. 加载静态知识图（训练开始时调用一次）──────────────────────────────────
def load_static_graph(
    hpo_index_path: str         = _HPO_INDEX_PATH,
    disease_index_path: str     = _DISEASE_INDEX_PATH,
    disease_incidence_path: str = _DISEASE_INC_PATH,
    hpo_id_col: str             = "hpo_id",
    hpo_idx_col: str            = "hpo_idx",
    disease_id_col: str         = "mondo_id",
    disease_idx_col: str        = "disease_idx",
) -> dict:
    """
    加载静态知识图，训练开始时调用一次并缓存结果。
    返回 hpo_to_idx、disease_to_idx、H_disease（以及 num_hpo、num_disease）。
    """
    hpo_to_idx     = load_index_file(hpo_index_path,     hpo_id_col,     hpo_idx_col)
    disease_to_idx = load_index_file(disease_index_path, disease_id_col, disease_idx_col)
    num_hpo, num_disease = len(hpo_to_idx), len(disease_to_idx)
    print(f"静态图加载完成：HPO {num_hpo}，疾病 {num_disease}")

    H_disease = load_disease_incidence(disease_incidence_path)
    assert H_disease.shape == (num_hpo, num_disease), \
        f"H_disease shape 异常：{H_disease.shape}，期望 ({num_hpo}, {num_disease})"

    return {
        "hpo_to_idx":    hpo_to_idx,
        "disease_to_idx": disease_to_idx,
        "H_disease":     H_disease,
        "num_hpo":       num_hpo,
        "num_disease":   num_disease,
    }


# ── 5. 训练时 batch 构图（传入 DataFrame 子集）───────────────────────────────
def build_batch_hypergraph(
    case_df: pd.DataFrame,           # 当前 batch 的病例 DataFrame（长格式子表）
    hpo_to_idx: dict,
    disease_to_idx: dict,
    H_disease: csr_matrix,
    case_id_col: str = "case_id",
    label_col: str   = "mondo_label",
    hpo_col: str     = "hpo_id",
) -> dict:
    """
    训练时每个 batch 调用：给定病例 DataFrame 子表，构建 H_batch = [H_case | H_disease]。
    hpo_to_idx / disease_to_idx / H_disease 由 load_static_graph() 提前缓存后传入。
    """
    num_disease  = H_disease.shape[1]
    batch_before = case_df[case_id_col].nunique() if case_id_col in case_df.columns else len(case_df)
    print(f"[batch] 输入病例数：{batch_before}")

    # 复用核心构图逻辑
    res = build_case_incidence(
        case_df, hpo_to_idx, disease_to_idx,
        case_id_col=case_id_col, label_col=label_col, hpo_col=hpo_col,
    )
    H_case   = res["H_case"]
    num_case = H_case.shape[1]

    if num_case == 0:
        raise ValueError("当前 batch 过滤后没有可用病例，请检查疾病索引或 HPO 映射。")

    # 拼接：病例列在前，疾病列在后
    H = hstack([H_case, H_disease], format="csr")
    print(f"[batch] H_case: {H_case.shape}  H_disease: {H_disease.shape}  H: {H.shape}")

    case_cols_global    = list(range(num_case))
    disease_cols_global = list(range(num_case, num_case + num_disease))
    gold_disease_cols_global = [num_case + c for c in res["gold_disease_cols"]]

    return {
        "H":                        H,
        "H_case":                   H_case,
        "H_disease":                H_disease,
        "case_ids":                 res["case_ids"],
        "case_labels":              res["case_labels"],
        "case_cols_global":         case_cols_global,
        "disease_cols_global":      disease_cols_global,
        "gold_disease_cols_global": gold_disease_cols_global,
    }
