try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project(allow_omp_duplicate=True)

import random

import pandas as pd
from scipy.sparse import csr_matrix

from src.data.build_hypergraph import build_batch_hypergraph, build_case_incidence, load_static_graph
from src.data.dataset import CaseBatchLoader, load_case_files, load_config


def main() -> list[dict]:
    cfg = load_config()
    static = load_static_graph()
    df = load_case_files(cfg["train_files"])
    loader = CaseBatchLoader(df, batch_size=cfg["batch_size"])

    print(f"总 batch 数：{len(loader)}")

    reports: list[dict] = []
    for i, batch_df in enumerate(loader):
        result = build_batch_hypergraph(
            batch_df,
            static["hpo_to_idx"],
            static["disease_to_idx"],
            static["H_disease"],
        )
        print(f"batch {i}: H={result['H'].shape}, cases={result['case_ids']}")
        reports.append(
            {
                "batch_index": i,
                "shape": tuple(result["H"].shape),
                "case_ids": list(result["case_ids"]),
            }
        )
        if i == 2:
            break

    return reports


def test_build_hypergraph_smoke() -> None:
    reports = main()
    assert len(reports) == 3
    assert all(report["shape"][0] > 0 and report["shape"][1] > 0 for report in reports)
    assert all(report["case_ids"] for report in reports)


def test_build_case_incidence_without_dropout_keeps_all_hpo() -> None:
    case_df = pd.DataFrame(
        {
            "case_id": ["case_1", "case_1", "case_2", "case_2"],
            "mondo_label": ["MONDO:1", "MONDO:1", "MONDO:2", "MONDO:2"],
            "hpo_id": ["HP:1", "HP:2", "HP:2", "HP:3"],
        }
    )
    hpo_to_idx = {"HP:1": 0, "HP:2": 1, "HP:3": 2}
    disease_to_idx = {"MONDO:1": 0, "MONDO:2": 1}

    result = build_case_incidence(
        case_df=case_df,
        hpo_to_idx=hpo_to_idx,
        disease_to_idx=disease_to_idx,
        hpo_dropout_prob=0.0,
    )

    assert result["H_case"].shape == (3, 2)
    assert result["H_case"].nnz == 4


def test_build_case_incidence_dropout_keeps_each_case_non_empty() -> None:
    case_df = pd.DataFrame(
        {
            "case_id": ["case_1", "case_1", "case_1", "case_2", "case_2", "case_2"],
            "mondo_label": ["MONDO:1", "MONDO:1", "MONDO:1", "MONDO:2", "MONDO:2", "MONDO:2"],
            "hpo_id": ["HP:1", "HP:2", "HP:3", "HP:2", "HP:3", "HP:4"],
        }
    )
    hpo_to_idx = {"HP:1": 0, "HP:2": 1, "HP:3": 2, "HP:4": 3}
    disease_to_idx = {"MONDO:1": 0, "MONDO:2": 1}

    result = build_case_incidence(
        case_df=case_df,
        hpo_to_idx=hpo_to_idx,
        disease_to_idx=disease_to_idx,
        hpo_dropout_prob=1.0,
        rng=random.Random(0),
    )

    assert result["H_case"].shape == (4, 2)
    assert result["H_case"].getnnz(axis=0).tolist() == [1, 1]
    assert result["H_case"].nnz == 2


def test_build_batch_hypergraph_dropout_only_changes_h_case() -> None:
    case_df = pd.DataFrame(
        {
            "case_id": ["case_1", "case_1", "case_1", "case_2", "case_2", "case_2"],
            "mondo_label": ["MONDO:1", "MONDO:1", "MONDO:1", "MONDO:2", "MONDO:2", "MONDO:2"],
            "hpo_id": ["HP:1", "HP:2", "HP:3", "HP:2", "HP:3", "HP:4"],
        }
    )
    hpo_to_idx = {"HP:1": 0, "HP:2": 1, "HP:3": 2, "HP:4": 3}
    disease_to_idx = {"MONDO:1": 0, "MONDO:2": 1}
    h_disease = csr_matrix(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    original = build_batch_hypergraph(
        case_df,
        hpo_to_idx,
        disease_to_idx,
        h_disease,
    )
    dropped = build_batch_hypergraph(
        case_df,
        hpo_to_idx,
        disease_to_idx,
        h_disease,
        hpo_dropout_prob=1.0,
        rng=random.Random(0),
    )

    assert original["H_disease"].nnz == dropped["H_disease"].nnz
    assert dropped["H_case"].nnz < original["H_case"].nnz
    assert dropped["H_case"].getnnz(axis=0).tolist() == [1, 1]


if __name__ == "__main__":
    main()
