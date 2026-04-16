try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project()

import pandas as pd

from src.data.build_llldataset_mondo_case_tables import (
    DatasetSpec,
    MappingBundle,
    process_dataset,
    resolve_case_mondo_labels,
)


def make_mapping_bundle() -> MappingBundle:
    return MappingBundle(
        orpha_to_mondo={
            "111": "MONDO:0000001",
        },
        omim_to_mondo={
            "OMIM:100100": "MONDO:0000002",
            "OMIM:200200": "MONDO:0000003",
        },
        orpha_to_omim={
            "ORPHA:222": "OMIM:200200",
        },
        orpha_to_name={
            "ORPHA:333": "Turnpenny-Fry syndrome",
            "ORPHA:444": "Ambiguous syndrome",
        },
        disease_name_to_mondo={
            "turnpenny fry syndrome": ["MONDO:0000004"],
            "ambiguous syndrome": ["MONDO:0000005", "MONDO:0000006"],
        },
        manual_orpha_to_mondo={
            "ORPHA:555": "MONDO:0000007",
        },
    )


def test_resolve_case_mondo_labels_supports_new_fallbacks() -> None:
    mappings = make_mapping_bundle()

    result = resolve_case_mondo_labels(
        "['OMIM:100100', 'ORPHA:111', 'ORPHA:222', 'ORPHA:333', 'ORPHA:444', 'ORPHA:555']",
        mappings,
    )

    assert result.mondo_labels == [
        "MONDO:0000002",
        "MONDO:0000001",
        "MONDO:0000003",
        "MONDO:0000004",
        "MONDO:0000007",
    ]
    assert result.missing_orpha_ids == {"444"}
    assert result.methods == {
        "omim_to_mondo",
        "orpha_to_mondo",
        "orpha_via_omim_to_mondo",
        "manual_orpha_to_mondo",
        "orpha_name_to_mondo",
        "orpha_name_ambiguous",
    }
    assert result.has_orpha is True
    assert result.has_omim is True


def test_resolve_case_mondo_labels_keeps_existing_input_mondo_priority() -> None:
    mappings = make_mapping_bundle()

    result = resolve_case_mondo_labels("['MONDO:0099999', 'OMIM:100100']", mappings)

    assert result.mondo_labels == ["MONDO:0099999"]
    assert result.methods == {"input_mondo"}
    assert result.has_omim is True


def test_process_dataset_renumbers_case_id_like_hms(tmp_path) -> None:
    input_path = tmp_path / "ddd_test.csv"
    output_path = tmp_path / "processed.csv"
    pd.DataFrame(
        [
            {
                "id": "raw_case_a",
                "phenotype": "['HP:0000001', 'HP:0000002']",
                "rare_disease": "['ORPHA:111']",
            },
            {
                "id": "raw_case_drop",
                "phenotype": "['HP:0000003']",
                "rare_disease": "['ORPHA:999999']",
            },
            {
                "id": "raw_case_b",
                "phenotype": "['HP:0000004']",
                "rare_disease": "['OMIM:100100']",
            },
        ]
    ).to_csv(input_path, index=False)

    spec = DatasetSpec(
        name="ddd_test",
        input_path=input_path,
        output_path=output_path,
        case_id_col="id",
        disease_col="rare_disease",
        hpo_col="phenotype",
        mode="ddd",
    )

    result = process_dataset(spec, make_mapping_bundle())
    output_df = pd.read_csv(output_path, dtype=str)

    assert output_df["case_id"].drop_duplicates().tolist() == ["case_1", "case_2"]
    assert output_df.to_dict(orient="records") == [
        {
            "case_id": "case_1",
            "mondo_label": "MONDO:0000001",
            "hpo_id": "HP:0000001",
        },
        {
            "case_id": "case_1",
            "mondo_label": "MONDO:0000001",
            "hpo_id": "HP:0000002",
        },
        {
            "case_id": "case_2",
            "mondo_label": "MONDO:0000002",
            "hpo_id": "HP:0000004",
        },
    ]
    assert result["stats"]["dropped_no_mondo"] == 1
    assert result["stats"]["unique_case_count"] == 2
