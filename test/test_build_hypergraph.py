try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project(allow_omp_duplicate=True)

from src.data.build_hypergraph import build_batch_hypergraph, load_static_graph
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


if __name__ == "__main__":
    main()
