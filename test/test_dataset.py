try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project(allow_omp_duplicate=True)

import pandas as pd

from src.data.dataset import CaseBatchLoader, load_case_files


def _make_df(num_cases: int = 10, rows_per_case: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(1, num_cases + 1):
        for j in range(rows_per_case):
            rows.append({"case_id": f"case_{i}", "hpo": f"HP:{1000 + j}"})
    return pd.DataFrame(rows)


def test_default_order_stable() -> None:
    df = _make_df(10)
    loader = CaseBatchLoader(df, batch_size=3)
    assert loader.case_ids == loader.base_case_ids


def test_shuffle_different_epochs() -> None:
    df = _make_df(20)
    loader = CaseBatchLoader(df, batch_size=4)

    loader.set_epoch(epoch=1, shuffle=True, random_seed=42)
    order_epoch1 = list(loader.case_ids)

    loader.set_epoch(epoch=2, shuffle=True, random_seed=42)
    order_epoch2 = list(loader.case_ids)

    assert order_epoch1 != order_epoch2


def test_shuffle_same_epoch_reproducible() -> None:
    df = _make_df(20)
    loader = CaseBatchLoader(df, batch_size=4)

    loader.set_epoch(epoch=5, shuffle=True, random_seed=99)
    order_a = list(loader.case_ids)

    loader.set_epoch(epoch=5, shuffle=True, random_seed=99)
    order_b = list(loader.case_ids)

    assert order_a == order_b


def test_shuffle_keeps_case_together() -> None:
    df = _make_df(12, rows_per_case=4)
    loader = CaseBatchLoader(df, batch_size=3)

    loader.set_epoch(epoch=1, shuffle=True, random_seed=42)
    for batch_idx in range(len(loader)):
        batch_df = loader.get_batch(batch_idx)
        for case_id in batch_df["case_id"].unique():
            expected_rows = len(df[df["case_id"] == case_id])
            actual_rows = len(batch_df[batch_df["case_id"] == case_id])
            assert actual_rows == expected_rows


def test_no_case_lost_or_duplicated() -> None:
    df = _make_df(15)
    loader = CaseBatchLoader(df, batch_size=4)

    loader.set_epoch(epoch=3, shuffle=True, random_seed=7)

    all_case_ids = []
    for batch_idx in range(len(loader)):
        batch_df = loader.get_batch(batch_idx)
        all_case_ids.extend(batch_df["case_id"].unique().tolist())

    assert sorted(all_case_ids) == sorted(loader.base_case_ids)
    assert len(all_case_ids) == len(set(all_case_ids))


def test_load_case_files_supports_csv(tmp_path) -> None:
    disease_index_path = tmp_path / "Disease_index_v4.xlsx"
    pd.DataFrame(
        [
            {"mondo_id": "MONDO:0001", "disease_idx": 0},
            {"mondo_id": "MONDO:0002", "disease_idx": 1},
        ]
    ).to_excel(disease_index_path, index=False)

    csv_path = tmp_path / "DDD.csv"
    pd.DataFrame(
        [
            {"case_id": "case_1", "mondo_label": "MONDO:0001", "hpo_id": "HP:0001"},
            {"case_id": "case_1", "mondo_label": "MONDO:0001", "hpo_id": "HP:0002"},
        ]
    ).to_csv(csv_path, index=False)

    loaded = load_case_files([csv_path], disease_index_path=disease_index_path)

    assert len(loaded) == 2
    assert loaded["case_id"].tolist() == ["DDD_case_1", "DDD_case_1"]
    assert loaded["gold_disease_idx"].tolist() == [0, 0]


def test_load_case_files_accepts_mondo_id_alias(tmp_path) -> None:
    disease_index_path = tmp_path / "Disease_index_v4.xlsx"
    pd.DataFrame(
        [
            {"mondo_id": "MONDO:0001", "disease_idx": 0},
        ]
    ).to_excel(disease_index_path, index=False)

    xlsx_path = tmp_path / "FakeDisease.xlsx"
    pd.DataFrame(
        [
            {"case_id": "fake_1", "mondo_id": "MONDO:0001", "hpo_id": "HP:0001"},
            {"case_id": "fake_1", "mondo_id": "MONDO:0001", "hpo_id": "HP:0002"},
        ]
    ).to_excel(xlsx_path, index=False)

    loaded = load_case_files([xlsx_path], disease_index_path=disease_index_path)

    assert "mondo_label" in loaded.columns
    assert loaded["mondo_label"].tolist() == ["MONDO:0001", "MONDO:0001"]
    assert loaded["gold_disease_idx"].tolist() == [0, 0]
