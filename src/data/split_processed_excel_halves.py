from pathlib import Path

import pandas as pd
import random


RANDOM_SEED = 42
TRAIN_RATIO = 2 / 3
FILES_TO_SPLIT = [
    "HMS.xlsx",
    "LIRICAL.xlsx",
    "mimic-rare(law).xlsx",
    "MME.xlsx",
    "MyGene2.xlsx",
    "RAMEDIS.xlsx",
]
MONDO_COLUMN = "mondo_label"
CASE_COLUMN = "case_id"


def split_cases_within_mondo(
    mondo_df: pd.DataFrame, rng: random.Random
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_groups = list(mondo_df.groupby(CASE_COLUMN, sort=False))

    if len(case_groups) == 1:
        return mondo_df.reset_index(drop=True), mondo_df.iloc[0:0].copy()

    rng.shuffle(case_groups)
    target_rows = len(mondo_df) * TRAIN_RATIO
    cumulative_rows = 0
    best_cut_index = 1
    best_distance = None

    for cut_index, (_, case_df) in enumerate(case_groups[:-1], start=1):
        cumulative_rows += len(case_df)
        distance = abs(target_rows - cumulative_rows)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_cut_index = cut_index

    train_case_ids = {case_id for case_id, _ in case_groups[:best_cut_index]}
    train_df = mondo_df[mondo_df[CASE_COLUMN].isin(train_case_ids)].reset_index(drop=True)
    test_df = mondo_df[~mondo_df[CASE_COLUMN].isin(train_case_ids)].reset_index(drop=True)

    return train_df, test_df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "LLLdataset" / "dataset" / "processed"
    train_dir = processed_dir / "train"
    test_dir = processed_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for file_name in FILES_TO_SPLIT:
        source_path = processed_dir / file_name
        df = pd.read_excel(source_path, sheet_name="Sheet1")
        rng = random.Random(RANDOM_SEED)
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []

        for _, mondo_df in df.groupby(MONDO_COLUMN, sort=False):
            train_part, test_part = split_cases_within_mondo(mondo_df, rng)
            if not train_part.empty:
                train_parts.append(train_part)
            if not test_part.empty:
                test_parts.append(test_part)

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)

        train_path = train_dir / file_name
        test_path = test_dir / file_name

        train_df.to_excel(train_path, index=False)
        test_df.to_excel(test_path, index=False)

        print(
            f"{file_name}: total={len(df)}, train={len(train_df)}, "
            f"test={len(test_df)}, train_mondos={train_df[MONDO_COLUMN].nunique()}, "
            f"test_mondos={test_df[MONDO_COLUMN].nunique()}, "
            f"shared_mondos={len(set(train_df[MONDO_COLUMN]) & set(test_df[MONDO_COLUMN]))}"
        )


if __name__ == "__main__":
    main()
