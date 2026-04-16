from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "mondo_overlap"

MICRO_TRAIN_FILES = {
    "HMS": PROJECT_ROOT / "data" / "processed" / "train" / "HMS.xlsx",
    "LIRICA": PROJECT_ROOT / "data" / "processed" / "train" / "LIRICA.xlsx",
    "MIMIC-Rare": PROJECT_ROOT / "data" / "processed" / "train" / "MIMIC-Rare.xlsx",
    "MME": PROJECT_ROOT / "data" / "processed" / "train" / "MME.xlsx",
    "Mygene2": PROJECT_ROOT / "data" / "processed" / "train" / "Mygene2.xlsx",
    "RAMEDIS": PROJECT_ROOT / "data" / "processed" / "train" / "RAMEDIS.xlsx",
    "DDD": PROJECT_ROOT / "data" / "processed" / "train" / "DDD.xlsx",
}

PRETRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train" / "PubCaseFinder_DiseaseHyperedge.xlsx"

CASE_ID_CANDIDATES = [
    "case_id",
    "caseid",
    "patient_id",
    "patientid",
    "record_id",
    "subject_id",
    "sample_id",
    "participant_id",
]

MONDO_CANDIDATES = [
    "mondo_id",
    "mondo_label",
    "label_mondo",
    "disease_mondo",
    "disease_id",
    "disease_label",
    "mondo",
]

BUCKET_ORDER = ["=1", "2-3", "4-9", ">=10"]


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def find_column(columns: list[str], candidates: list[str], label: str, required: bool) -> str | None:
    normalized_map = {normalize_column_name(col): col for col in columns}

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized_map:
            return normalized_map[key]

    heuristic_matches = []
    for col in columns:
        norm = normalize_column_name(col)
        if label == "case_id" and (
            ("case" in norm and "id" in norm)
            or ("patient" in norm and "id" in norm)
            or ("subject" in norm and "id" in norm)
            or ("sample" in norm and "id" in norm)
        ):
            heuristic_matches.append(col)
        if label == "mondo" and ("mondo" in norm or ("disease" in norm and ("id" in norm or "label" in norm))):
            heuristic_matches.append(col)

    heuristic_matches = list(dict.fromkeys(heuristic_matches))
    if len(heuristic_matches) == 1:
        return heuristic_matches[0]

    if not required:
        return None

    extra = f"候选命中={heuristic_matches}" if heuristic_matches else "未命中任何候选列"
    raise ValueError(f"无法识别 {label} 列。{extra}。当前列名：{columns}")


def normalize_mondo(value: object) -> tuple[str | None, str]:
    if pd.isna(value):
        return None, "empty"

    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None, "empty"

    mondo_match = re.fullmatch(r"MONDO[:_]?(\d+)", text)
    if mondo_match:
        return f"MONDO:{mondo_match.group(1).zfill(7)}", "valid"

    digits_match = re.fullmatch(r"(\d+)", text)
    if digits_match:
        return f"MONDO:{digits_match.group(1).zfill(7)}", "valid"

    return None, "invalid"


def normalize_case_id(value: object, row_index: int) -> tuple[str, bool]:
    if pd.isna(value):
        return f"__MISSING_CASE_ID_ROW_{row_index}", True

    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NULL"}:
        return f"__MISSING_CASE_ID_ROW_{row_index}", True
    return text, False


def support_bucket(support: int) -> str:
    if support == 1:
        return "=1"
    if 2 <= support <= 3:
        return "2-3"
    if 4 <= support <= 9:
        return "4-9"
    return ">=10"


def ratio_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{numerator / denominator:.2%}"


def load_dataset_cases(dataset_name: str, file_path: Path, require_case_id: bool = True) -> dict:
    df = pd.read_excel(file_path, sheet_name=0, dtype=object)
    columns = [str(col) for col in df.columns]

    case_col = find_column(columns, CASE_ID_CANDIDATES, "case_id", required=require_case_id)
    mondo_col = find_column(columns, MONDO_CANDIDATES, "mondo", required=True)

    work = pd.DataFrame({"raw_mondo": df[mondo_col]})
    raw_rows = len(work)

    normalized_mondos = [normalize_mondo(value) for value in work["raw_mondo"]]
    work["mondo_norm"] = [item[0] for item in normalized_mondos]
    work["mondo_status"] = [item[1] for item in normalized_mondos]

    if case_col is None:
        unique_mondos = sorted(set(work.loc[work["mondo_status"] == "valid", "mondo_norm"]))
        print(
            f"[识别] {dataset_name}: case列=无(按 disease-level 处理), "
            f"MONDO列={mondo_col}, 原始行数={raw_rows}, unique MONDO数={len(unique_mondos)}"
        )
        return {
            "dataset": dataset_name,
            "file_path": str(file_path),
            "case_col": None,
            "mondo_col": mondo_col,
            "raw_rows": raw_rows,
            "case_count": 0,
            "missing_case_id_row_count": 0,
            "row_empty_mondo_count": int((work["mondo_status"] == "empty").sum()),
            "row_invalid_mondo_count": int((work["mondo_status"] == "invalid").sum()),
            "row_valid_mondo_count": int((work["mondo_status"] == "valid").sum()),
            "case_details": pd.DataFrame(),
            "internal_support": {},
            "unique_mondos": set(unique_mondos),
        }

    normalized_cases = [normalize_case_id(value, idx) for idx, value in enumerate(df[case_col], start=2)]
    work["case_id"] = [item[0] for item in normalized_cases]
    missing_case_id_row_count = sum(item[1] for item in normalized_cases)

    case_rows = []
    mondo_to_case_keys = defaultdict(set)
    grouped = work.groupby("case_id", sort=False, dropna=False)
    for case_id, group in grouped:
        valid_mondos = sorted(set(group.loc[group["mondo_status"] == "valid", "mondo_norm"]))
        case_key = f"{dataset_name}::{case_id}"
        for mondo in valid_mondos:
            mondo_to_case_keys[mondo].add(case_key)

        case_rows.append(
            {
                "case_id": case_id,
                "case_key": case_key,
                "valid_mondo_count": len(valid_mondos),
                "valid_mondos": "|".join(valid_mondos),
                "has_empty_mondo_row": bool((group["mondo_status"] == "empty").any()),
                "has_invalid_mondo_row": bool((group["mondo_status"] == "invalid").any()),
                "row_count": int(len(group)),
            }
        )

    case_details = pd.DataFrame(case_rows)
    internal_support = {mondo: len(case_keys) for mondo, case_keys in mondo_to_case_keys.items()}
    unique_mondos = set(internal_support)

    print(
        f"[识别] {dataset_name}: case列={case_col}, MONDO列={mondo_col}, "
        f"原始行数={raw_rows}, 去重后病例数={len(case_details)}"
    )

    return {
        "dataset": dataset_name,
        "file_path": str(file_path),
        "case_col": case_col,
        "mondo_col": mondo_col,
        "raw_rows": raw_rows,
        "case_count": int(len(case_details)),
        "missing_case_id_row_count": int(missing_case_id_row_count),
        "row_empty_mondo_count": int((work["mondo_status"] == "empty").sum()),
        "row_invalid_mondo_count": int((work["mondo_status"] == "invalid").sum()),
        "row_valid_mondo_count": int((work["mondo_status"] == "valid").sum()),
        "case_details": case_details,
        "internal_support": internal_support,
        "unique_mondos": unique_mondos,
    }


def build_global_support(dataset_results: dict[str, dict]) -> tuple[dict[str, int], dict[str, set[str]], pd.DataFrame]:
    global_support = defaultdict(int)
    dataset_presence = defaultdict(set)

    for dataset, result in dataset_results.items():
        for mondo, support in result["internal_support"].items():
            global_support[mondo] += support
            dataset_presence[mondo].add(dataset)

    rows = []
    for mondo in sorted(global_support):
        row = {
            "mondo": mondo,
            "global_case_support": int(global_support[mondo]),
            "support_bucket": support_bucket(int(global_support[mondo])),
            "dataset_occurrence_count": int(len(dataset_presence[mondo])),
            "datasets": "|".join(sorted(dataset_presence[mondo])),
        }
        for dataset in MICRO_TRAIN_FILES:
            row[f"support_{dataset}"] = int(dataset_results[dataset]["internal_support"].get(mondo, 0))
        rows.append(row)

    return dict(global_support), dict(dataset_presence), pd.DataFrame(rows)


def compute_internal_stats(dataset_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for dataset in MICRO_TRAIN_FILES:
        result = dataset_results[dataset]
        unique_mondo_count = len(result["unique_mondos"])
        total_case_support = sum(result["internal_support"].values())
        singleton_count = sum(1 for support in result["internal_support"].values() if support == 1)
        multi_label_case_count = int((result["case_details"]["valid_mondo_count"] > 1).sum())
        empty_or_invalid_case_count = int((result["case_details"]["valid_mondo_count"] == 0).sum())

        rows.append(
            {
                "dataset": dataset,
                "file_path": result["file_path"],
                "case_column": result["case_col"],
                "mondo_column": result["mondo_col"],
                "raw_rows": int(result["raw_rows"]),
                "case_count": int(result["case_count"]),
                "unique_mondo_count": int(unique_mondo_count),
                "avg_cases_per_mondo": round(total_case_support / unique_mondo_count, 4) if unique_mondo_count else 0.0,
                "singleton_mondo_count": int(singleton_count),
                "singleton_mondo_ratio": round(singleton_count / unique_mondo_count, 4) if unique_mondo_count else 0.0,
                "multi_label_case_count": multi_label_case_count,
                "empty_or_invalid_mondo_case_count": empty_or_invalid_case_count,
                "row_empty_mondo_count": int(result["row_empty_mondo_count"]),
                "row_invalid_mondo_count": int(result["row_invalid_mondo_count"]),
                "missing_case_id_row_count": int(result["missing_case_id_row_count"]),
            }
        )

    return pd.DataFrame(rows)


def compute_dataset_vs_other_overlap(dataset_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    datasets = list(MICRO_TRAIN_FILES)
    for dataset in datasets:
        current_mondos = dataset_results[dataset]["unique_mondos"]
        other_mondos = set().union(*(dataset_results[other]["unique_mondos"] for other in datasets if other != dataset))
        overlap_count = len(current_mondos & other_mondos)
        only_count = len(current_mondos - other_mondos)
        total = len(current_mondos)

        rows.append(
            {
                "dataset": dataset,
                "unique_mondo_count": int(total),
                "overlap_with_other_count": int(overlap_count),
                "only_in_this_dataset_count": int(only_count),
                "overlap_ratio": round(overlap_count / total, 4) if total else 0.0,
                "only_in_this_dataset_ratio": round(only_count / total, 4) if total else 0.0,
            }
        )

    return pd.DataFrame(rows)


def compute_dataset_vs_pretrain_overlap(dataset_results: dict[str, dict], pretrain_mondos: set[str]) -> pd.DataFrame:
    rows = []
    for dataset in MICRO_TRAIN_FILES:
        current_mondos = dataset_results[dataset]["unique_mondos"]
        overlap_count = len(current_mondos & pretrain_mondos)
        only_count = len(current_mondos - pretrain_mondos)
        total = len(current_mondos)

        rows.append(
            {
                "dataset": dataset,
                "unique_mondo_count": int(total),
                "overlap_with_pretrain_count": int(overlap_count),
                "not_in_pretrain_count": int(only_count),
                "overlap_ratio": round(overlap_count / total, 4) if total else 0.0,
                "not_in_pretrain_ratio": round(only_count / total, 4) if total else 0.0,
            }
        )

    return pd.DataFrame(rows)


def compute_singleton_resolution(dataset_results: dict[str, dict], pretrain_mondos: set[str]) -> pd.DataFrame:
    rows = []
    datasets = list(MICRO_TRAIN_FILES)
    for dataset in datasets:
        current_support = dataset_results[dataset]["internal_support"]
        singleton_mondos = {mondo for mondo, support in current_support.items() if support == 1}
        other_mondos = set().union(*(dataset_results[other]["unique_mondos"] for other in datasets if other != dataset))

        seen_in_other = singleton_mondos & other_mondos
        not_in_other = singleton_mondos - other_mondos
        pretrain_only = {mondo for mondo in not_in_other if mondo in pretrain_mondos}
        unseen_anywhere = not_in_other - pretrain_mondos
        total = len(singleton_mondos)

        rows.append(
            {
                "dataset": dataset,
                "internal_singleton_mondo_count": int(total),
                "singleton_seen_in_other_micro_count": int(len(seen_in_other)),
                "singleton_seen_in_other_micro_ratio": round(len(seen_in_other) / total, 4) if total else 0.0,
                "singleton_not_in_other_but_in_pretrain_count": int(len(pretrain_only)),
                "singleton_not_in_other_but_in_pretrain_ratio": round(len(pretrain_only) / total, 4) if total else 0.0,
                "singleton_in_neither_micro_nor_pretrain_count": int(len(unseen_anywhere)),
                "singleton_in_neither_micro_nor_pretrain_ratio": round(len(unseen_anywhere) / total, 4) if total else 0.0,
            }
        )

    return pd.DataFrame(rows)


def compute_dataset_global_support_buckets(
    dataset_results: dict[str, dict],
    global_support: dict[str, int],
    pretrain_mondos: set[str],
) -> pd.DataFrame:
    rows = []
    for dataset in MICRO_TRAIN_FILES:
        current_mondos = dataset_results[dataset]["unique_mondos"]
        total = len(current_mondos)

        for bucket in BUCKET_ORDER:
            bucket_mondos = {mondo for mondo in current_mondos if support_bucket(global_support[mondo]) == bucket}
            pretrain_overlap = bucket_mondos & pretrain_mondos
            bucket_total = len(bucket_mondos)
            rows.append(
                {
                    "dataset": dataset,
                    "support_bucket": bucket,
                    "mondo_count": int(bucket_total),
                    "mondo_ratio_within_dataset": round(bucket_total / total, 4) if total else 0.0,
                    "pretrain_overlap_count": int(len(pretrain_overlap)),
                    "pretrain_overlap_ratio_within_bucket": round(len(pretrain_overlap) / bucket_total, 4)
                    if bucket_total
                    else 0.0,
                }
            )

    return pd.DataFrame(rows)


def compute_pairwise_overlap_matrix(dataset_results: dict[str, dict]) -> pd.DataFrame:
    datasets = list(MICRO_TRAIN_FILES)
    rows = []
    for left in datasets:
        row = {"dataset": left}
        left_mondos = dataset_results[left]["unique_mondos"]
        for right in datasets:
            right_mondos = dataset_results[right]["unique_mondos"]
            row[right] = int(len(left_mondos & right_mondos))
        rows.append(row)
    return pd.DataFrame(rows)


def build_readme_summary(
    internal_stats: pd.DataFrame,
    global_support_df: pd.DataFrame,
    overlap_other_df: pd.DataFrame,
    overlap_pretrain_df: pd.DataFrame,
    singleton_resolution_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    pretrain_mondos: set[str],
) -> str:
    global_bucket_counts = (
        global_support_df["support_bucket"]
        .value_counts()
        .reindex(BUCKET_ORDER, fill_value=0)
        .to_dict()
    )
    global_unique_mondo_count = int(len(global_support_df))
    global_singleton_count = int(global_bucket_counts["=1"])

    highest_singleton_row = internal_stats.sort_values("singleton_mondo_ratio", ascending=False).iloc[0]
    highest_only_row = overlap_other_df.sort_values("only_in_this_dataset_ratio", ascending=False).iloc[0]
    lowest_pretrain_row = overlap_pretrain_df.sort_values("overlap_ratio", ascending=True).iloc[0]
    weakest_repeat_row = (
        bucket_df[bucket_df["support_bucket"] == "=1"]
        .sort_values("mondo_ratio_within_dataset", ascending=False)
        .iloc[0]
    )

    singleton_rows = singleton_resolution_df.set_index("dataset")
    overlap_other_rows = overlap_other_df.set_index("dataset")
    overlap_pretrain_rows = overlap_pretrain_df.set_index("dataset")
    bucket_rows = bucket_df.pivot(index="dataset", columns="support_bucket", values="mondo_ratio_within_dataset").fillna(0.0)

    q3_coverage = []
    q3_generalization = []
    q3_mixed = []
    for dataset in MICRO_TRAIN_FILES:
        pretrain_overlap_ratio = float(overlap_pretrain_rows.loc[dataset, "overlap_ratio"])
        unseen_ratio = float(singleton_rows.loc[dataset, "singleton_in_neither_micro_nor_pretrain_ratio"])

        if pretrain_overlap_ratio < 0.6 or unseen_ratio >= 0.25:
            q3_coverage.append(dataset)
        elif pretrain_overlap_ratio >= 0.9 and unseen_ratio <= 0.15:
            q3_generalization.append(dataset)
        else:
            q3_mixed.append(dataset)

    lines = [
        "MONDO overlap 分析摘要",
        "======================",
        "",
        "统计口径：",
        "- 微调训练集按病例级去重，同一 case_id 的多行 HPO 先聚合后再统计 MONDO support。",
        "- MONDO 统一规范化为 MONDO:XXXXXXX；空值和非法字符串单独计数。",
        f"- 预训练集文件：{PRETRAIN_FILE}",
        f"- 预训练集 unique MONDO 数：{len(pretrain_mondos)}",
        "",
        "全局微调 MONDO support 分布：",
        f"- unique MONDO 总数：{global_unique_mondo_count}",
        f"- global singleton MONDO：{global_singleton_count} ({ratio_text(global_singleton_count, global_unique_mondo_count)})",
        f"- support = 1：{global_bucket_counts['=1']}",
        f"- support = 2-3：{global_bucket_counts['2-3']}",
        f"- support = 4-9：{global_bucket_counts['4-9']}",
        f"- support >= 10：{global_bucket_counts['>=10']}",
        "",
        "关键结论：",
        (
            f"- 内部 singleton 占比最高的是 {highest_singleton_row['dataset']}，"
            f"占比 {highest_singleton_row['singleton_mondo_ratio']:.2%}。"
        ),
        (
            f"- only-in-this-dataset 占比最高的是 {highest_only_row['dataset']}，"
            f"占比 {highest_only_row['only_in_this_dataset_ratio']:.2%}。"
        ),
        (
            f"- 与预训练集 overlap 最低的是 {lowest_pretrain_row['dataset']}，"
            f"overlap 仅 {lowest_pretrain_row['overlap_ratio']:.2%}。"
        ),
        (
            f"- 在全局微调中最缺少重复支持的是 {weakest_repeat_row['dataset']}，"
            f"其 unique MONDO 中有 {weakest_repeat_row['mondo_ratio_within_dataset']:.2%} 的全局 support 仍然只有 1。"
        ),
        "",
        "问题 1：DDD、LIRICA 的内部 singleton MONDO，有多少在其他微调训练集里也出现过？",
        (
            f"- DDD：{int(singleton_rows.loc['DDD', 'singleton_seen_in_other_micro_count'])} / "
            f"{int(singleton_rows.loc['DDD', 'internal_singleton_mondo_count'])} "
            f"({singleton_rows.loc['DDD', 'singleton_seen_in_other_micro_ratio']:.2%})。"
        ),
        (
            f"- LIRICA：{int(singleton_rows.loc['LIRICA', 'singleton_seen_in_other_micro_count'])} / "
            f"{int(singleton_rows.loc['LIRICA', 'internal_singleton_mondo_count'])} "
            f"({singleton_rows.loc['LIRICA', 'singleton_seen_in_other_micro_ratio']:.2%})。"
        ),
        "",
        "问题 2：DDD、LIRICA 的内部 singleton MONDO，有多少虽然不在其他微调训练集里，但在 PubCaseFinder 预训练集里出现过？",
        (
            f"- DDD：{int(singleton_rows.loc['DDD', 'singleton_not_in_other_but_in_pretrain_count'])} / "
            f"{int(singleton_rows.loc['DDD', 'internal_singleton_mondo_count'])} "
            f"({singleton_rows.loc['DDD', 'singleton_not_in_other_but_in_pretrain_ratio']:.2%})。"
        ),
        (
            f"- LIRICA：{int(singleton_rows.loc['LIRICA', 'singleton_not_in_other_but_in_pretrain_count'])} / "
            f"{int(singleton_rows.loc['LIRICA', 'internal_singleton_mondo_count'])} "
            f"({singleton_rows.loc['LIRICA', 'singleton_not_in_other_but_in_pretrain_ratio']:.2%})。"
        ),
        "",
        "问题 3：哪些数据集更像是全局 coverage 不足，哪些更像是跨数据集泛化不足？",
        "- 更像 coverage 不足："
        + ("、".join(q3_coverage) if q3_coverage else "无明显数据集")
        + "。判断依据是预训练 overlap 偏低，或内部 singleton 中有较高比例在其他微调和预训练里都没出现。",
        "- 更像泛化不足："
        + ("、".join(q3_generalization) if q3_generalization else "无明显数据集")
        + "。这类数据集的 MONDO 在全局微调和预训练中大多已经见过，如果效果仍差，更可能是表型表达差异、特征聚合或 scorer/readout 泛化不够。",
        "- 两者都有、需要结合实验验证："
        + ("、".join(q3_mixed) if q3_mixed else "无")
        + "。",
        "",
        "问题 4：下一步更应该优先做分层采样，还是优先做更强的 readout/scorer？",
        "- 更建议先做分层采样与长尾感知训练。原因是当前全局 support=1 的 MONDO 仍然很多，且 DDD、LIRICA 这类数据集内部 singleton 非常集中；如果 batch 继续被头部标签主导，更强的 scorer 也无法补足尾部标签的监督信号。",
        "- readout/scorer 仍然值得做，但更适合作为第二优先级，重点服务于那些 overlap 已经较高、却仍可能存在跨来源表型分布差异的数据集。",
        "",
        "补充提示：",
        "- 详细表格请看 outputs/analysis/mondo_overlap 目录下的 1-7 号 CSV。",
        "- 如果需要进一步定位具体标签，可以直接查看 2_global_mondo_support.csv 里每个 MONDO 的 support 与出现数据集。",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_results = {}
    for dataset, file_path in MICRO_TRAIN_FILES.items():
        dataset_results[dataset] = load_dataset_cases(dataset, file_path, require_case_id=True)

    pretrain_result = load_dataset_cases("PubCaseFinder", PRETRAIN_FILE, require_case_id=False)
    pretrain_mondos = set(pretrain_result["unique_mondos"])

    internal_stats = compute_internal_stats(dataset_results)
    global_support, dataset_presence, global_support_df = build_global_support(dataset_results)
    global_support_df["in_pretrain"] = global_support_df["mondo"].isin(pretrain_mondos)

    overlap_other_df = compute_dataset_vs_other_overlap(dataset_results)
    overlap_pretrain_df = compute_dataset_vs_pretrain_overlap(dataset_results, pretrain_mondos)
    singleton_resolution_df = compute_singleton_resolution(dataset_results, pretrain_mondos)
    bucket_df = compute_dataset_global_support_buckets(dataset_results, global_support, pretrain_mondos)
    pairwise_matrix_df = compute_pairwise_overlap_matrix(dataset_results)

    internal_stats.to_csv(OUTPUT_DIR / "1_dataset_internal_stats.csv", index=False, encoding="utf-8-sig")
    global_support_df.to_csv(OUTPUT_DIR / "2_global_mondo_support.csv", index=False, encoding="utf-8-sig")
    overlap_other_df.to_csv(OUTPUT_DIR / "3_dataset_vs_other_overlap.csv", index=False, encoding="utf-8-sig")
    overlap_pretrain_df.to_csv(OUTPUT_DIR / "4_dataset_vs_pretrain_overlap.csv", index=False, encoding="utf-8-sig")
    singleton_resolution_df.to_csv(OUTPUT_DIR / "5_dataset_singleton_resolution.csv", index=False, encoding="utf-8-sig")
    bucket_df.to_csv(OUTPUT_DIR / "6_dataset_global_support_buckets.csv", index=False, encoding="utf-8-sig")
    pairwise_matrix_df.to_csv(OUTPUT_DIR / "7_pairwise_dataset_overlap_matrix.csv", index=False, encoding="utf-8-sig")

    readme_text = build_readme_summary(
        internal_stats=internal_stats,
        global_support_df=global_support_df,
        overlap_other_df=overlap_other_df,
        overlap_pretrain_df=overlap_pretrain_df,
        singleton_resolution_df=singleton_resolution_df,
        bucket_df=bucket_df,
        pretrain_mondos=pretrain_mondos,
    )
    (OUTPUT_DIR / "8_readme_summary.txt").write_text(readme_text, encoding="utf-8-sig")

    highest_singleton_row = internal_stats.sort_values("singleton_mondo_ratio", ascending=False).iloc[0]
    highest_only_row = overlap_other_df.sort_values("only_in_this_dataset_ratio", ascending=False).iloc[0]
    lowest_pretrain_row = overlap_pretrain_df.sort_values("overlap_ratio", ascending=True).iloc[0]
    weakest_repeat_row = (
        bucket_df[bucket_df["support_bucket"] == "=1"]
        .sort_values("mondo_ratio_within_dataset", ascending=False)
        .iloc[0]
    )

    print("")
    print("[输出完成] 已生成 8 个分析文件：")
    for file_name in [
        "1_dataset_internal_stats.csv",
        "2_global_mondo_support.csv",
        "3_dataset_vs_other_overlap.csv",
        "4_dataset_vs_pretrain_overlap.csv",
        "5_dataset_singleton_resolution.csv",
        "6_dataset_global_support_buckets.csv",
        "7_pairwise_dataset_overlap_matrix.csv",
        "8_readme_summary.txt",
    ]:
        print(f"- {OUTPUT_DIR / file_name}")

    print("")
    print("[关键摘要]")
    print(
        f"- 内部 singleton 占比最高：{highest_singleton_row['dataset']} "
        f"({highest_singleton_row['singleton_mondo_ratio']:.2%})"
    )
    print(
        f"- only-in-this-dataset 占比最高：{highest_only_row['dataset']} "
        f"({highest_only_row['only_in_this_dataset_ratio']:.2%})"
    )
    print(
        f"- 与预训练 overlap 最低：{lowest_pretrain_row['dataset']} "
        f"({lowest_pretrain_row['overlap_ratio']:.2%})"
    )
    print(
        f"- 在全局微调中最缺少重复支持：{weakest_repeat_row['dataset']} "
        f"({weakest_repeat_row['mondo_ratio_within_dataset']:.2%} 的 MONDO 全局 support=1)"
    )


if __name__ == "__main__":
    main()
