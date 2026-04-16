from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from test._script_bootstrap import bootstrap_project


PROJECT_ROOT = bootstrap_project()
SCRIPT_PATH = PROJECT_ROOT / "src" / "data" / "generate_synthetic_low_count_cases.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_low_count_cases", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_excel(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_excel(path, index=False)


def test_load_ontology_name_map_parses_mondo_and_hpo_ids(tmp_path: Path) -> None:
    mondo_path = tmp_path / "mondo.json"
    mondo_path.write_text(
        json.dumps(
            {
                "graphs": [
                    {
                        "nodes": [
                            {"id": "http://purl.obolibrary.org/obo/MONDO_0001234", "type": "CLASS", "lbl": "d1"},
                            {"id": "x", "type": "INDIVIDUAL", "lbl": "skip"},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    hpo_path = tmp_path / "hp.json"
    hpo_path.write_text(
        json.dumps(
            {
                "graphs": [
                    {
                        "nodes": [
                            {"id": "http://purl.obolibrary.org/obo/HP_0004321", "type": "CLASS", "lbl": "p1"},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    mondo_map = MODULE.load_ontology_name_map(mondo_path, "MONDO")
    hpo_map = MODULE.load_ontology_name_map(hpo_path, "HP")

    assert mondo_map["MONDO:0001234"] == "d1"
    assert hpo_map["HP:0004321"] == "p1"


def test_load_knowledge_annotations_builds_source_counts(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "DiseaseHy"
    knowledge_dir.mkdir()
    _write_excel(
        knowledge_dir / "GARD.xlsx",
        [
            {"mondo_id": "MONDO:0000001", "hpo_id": "HP:0000001", "weight": 0.9},
            {"mondo_id": "MONDO:0000001", "hpo_id": "HP:0000002", "weight": 0.5},
        ],
    )
    _write_excel(
        knowledge_dir / "HPOA.xlsx",
        [
            {"mondo_id": "MONDO:0000001", "hpo_id": "HP:0000001", "weight": 0.8},
        ],
    )
    _write_excel(
        knowledge_dir / "orphanet.xlsx",
        [
            {"mondo_id": "MONDO:0000001", "hpo_id": "HP:0000003", "weight": 0.7},
        ],
    )

    knowledge_df = MODULE.load_knowledge_annotations(
        knowledge_dir=knowledge_dir,
        hpo_vocab={"HP:0000001", "HP:0000002", "HP:0000003"},
        disease_vocab={"MONDO:0000001"},
    )
    profiles = MODULE.build_knowledge_profiles(knowledge_df, core_min_sources=2, hard_core_min_sources=3)

    assert set(knowledge_df["hpo_id"]) == {"HP:0000001", "HP:0000002", "HP:0000003"}
    assert profiles["MONDO:0000001"]["hard_core_hpo_ids"] == ()
    assert profiles["MONDO:0000001"]["soft_core_hpo_ids"] == ("HP:0000001",)
    assert set(profiles["MONDO:0000001"]["optional_non_core_hpo_ids"]) == {"HP:0000002", "HP:0000003"}


def test_validate_selector_decision_rejects_invalid_ids() -> None:
    request = MODULE.SelectorRequest(
        mondo_id="MONDO:0000001",
        disease_name="d1",
        hard_core_hpo_ids=("HP:0000001",),
        soft_core_hpo_ids=("HP:0000003",),
        optional_non_core_hpo_ids=("HP:0000002",),
        noise_candidate_ids=("HP:0000009",),
        target_hpo_count=2,
        min_hpo_count=1,
        max_hpo_count=3,
        max_noise_hpo=2,
    )
    decision = MODULE.SelectorDecision(
        selected_soft_core_hpo_ids=(),
        selected_optional_hpo_ids=("HP:9999999",),
        selected_noise_hpo_ids=(),
        raw_response="{}",
        selector_mode="heuristic",
    )

    with pytest.raises(ValueError, match="invalid optional HPOs"):
        MODULE.validate_selector_decision(request, decision)


def test_validate_generated_case_rejects_exact_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate case signature"):
        MODULE.validate_generated_case(
            mondo_id="MONDO:0000001",
            final_hpo_ids=["HP:0000001", "HP:0000002"],
            required_hard_core_hpo_ids=("HP:0000001",),
            min_hpo_count=1,
            max_hpo_count=3,
            existing_signatures={frozenset({"HP:0000001", "HP:0000002"})},
        )


def test_build_underfilled_mondo_report_summarizes_capacity_and_failures() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "mondo_label": "MONDO:0000001",
                "requested_generated_case_count": 9,
                "effective_requested_generated_case_count": 1,
                "generated_case_count": 1,
                "failed_case_count": 0,
                "estimated_case_capacity": 1,
                "remaining_slot_budget": 0,
                "effective_max_noise_hpo": 0,
                "deterministic_case_only": True,
            },
            {
                "mondo_label": "MONDO:0000002",
                "requested_generated_case_count": 5,
                "effective_requested_generated_case_count": 5,
                "generated_case_count": 3,
                "failed_case_count": 2,
                "estimated_case_capacity": 5,
                "remaining_slot_budget": 2,
                "effective_max_noise_hpo": 1,
                "deterministic_case_only": False,
            },
        ]
    )
    audit_df = pd.DataFrame(
        [
            {"mondo_label": "MONDO:0000002", "status": "failed", "error": "MONDO:0000002 generated duplicate case signature."},
            {"mondo_label": "MONDO:0000002", "status": "failed", "error": "MONDO:0000002 generated duplicate case signature."},
            {"mondo_label": "MONDO:0000002", "status": "accepted", "error": ""},
        ]
    )

    report_df = MODULE.build_underfilled_mondo_report(summary_df, audit_df)

    assert list(report_df["mondo_label"]) == ["MONDO:0000001", "MONDO:0000002"]
    assert bool(report_df.loc[0, "capacity_capped"]) is True
    assert report_df.loc[0, "underfilled_reason_summary"] == "adaptive_capacity_cap"
    assert report_df.loc[1, "top_failed_reason"] == "duplicate_case_signature"
    assert "generation_failures" in report_df.loc[1, "underfilled_reason_summary"]


def test_build_generation_plan_caps_hard_core_overflow_cases() -> None:
    plan = MODULE.build_generation_plan(
        requested_case_count=9,
        hard_core_hpo_ids=tuple(f"HP:{index:07d}" for index in range(1, 54)),
        soft_core_hpo_ids=("HP:0001001", "HP:0001002"),
        optional_hpo_ids=("HP:0002001",),
        target_hpo_count=13,
        min_hpo_count=53,
        max_hpo_count=53,
        config=MODULE.GenerationConfig(max_noise_hpo=2),
    )

    assert plan["effective_requested_case_count"] == 1
    assert plan["effective_target_hpo_count"] == 53
    assert plan["effective_max_noise_hpo"] == 0
    assert plan["effective_soft_core_hpo_ids"] == ()
    assert plan["effective_optional_hpo_ids"] == ()
    assert plan["deterministic_case_only"] is True


def test_siliconflow_prompt_does_not_include_seed_fields() -> None:
    selector = MODULE.SiliconFlowSelector(
        api_key="dummy",
        config=MODULE.GenerationConfig(),
    )
    request = MODULE.SelectorRequest(
        mondo_id="MONDO:0000001",
        disease_name="d1",
        hard_core_hpo_ids=("HP:0000001",),
        soft_core_hpo_ids=("HP:0000002",),
        optional_non_core_hpo_ids=("HP:0000002",),
        noise_candidate_ids=("HP:0000009",),
        target_hpo_count=3,
        min_hpo_count=2,
        max_hpo_count=4,
        max_noise_hpo=2,
    )

    messages = selector._build_messages(request, {"HP:0000001": "p1", "HP:0000002": "p2", "HP:0000009": "n1"})
    content = "\n".join(message["content"] for message in messages)

    assert "seed_case_id" not in content
    assert "seed_case_hpo_ids" not in content
    assert "MONDO ID: MONDO:0000001" in content


def test_generate_cases_heuristic_preserves_core_and_limits_noise() -> None:
    target_df = pd.DataFrame(
        [
            {
                "mondo_label": "MONDO:0000001",
                "total_case_pair_count": 8,
                "target_generated_case_count": 2,
            }
        ]
    )
    knowledge_df = pd.DataFrame(
        [
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000001", "source_count": 3, "source_names": "GARD|HPOA|orphanet", "max_weight": 1.0, "mean_weight": 1.0},
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000002", "source_count": 2, "source_names": "GARD|HPOA", "max_weight": 0.9, "mean_weight": 0.85},
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000003", "source_count": 1, "source_names": "GARD", "max_weight": 0.7, "mean_weight": 0.7},
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000004", "source_count": 1, "source_names": "orphanet", "max_weight": 0.6, "mean_weight": 0.6},
        ]
    )
    noise_pool_df = pd.DataFrame(
        [
            {"hpo_id": "HP:0000010", "case_support": 20, "hpo_name": "n1"},
            {"hpo_id": "HP:0000011", "case_support": 18, "hpo_name": "n2"},
            {"hpo_id": "HP:0000012", "case_support": 16, "hpo_name": "n3"},
            {"hpo_id": "HP:0000013", "case_support": 14, "hpo_name": "n4"},
        ]
    )

    generated_df, metadata_df, summary_df, _, audit_df = MODULE.generate_cases(
        target_df=target_df,
        real_signatures_by_mondo={},
        hpo_count_stats_by_mondo={"MONDO:0000001": {"median_hpo_count": 3, "min_hpo_count": 2, "max_hpo_count": 4}},
        knowledge_df=knowledge_df,
        noise_pool_df=noise_pool_df,
        mondo_name_map={"MONDO:0000001": "d1"},
        hpo_name_map={
            "HP:0000001": "p1",
            "HP:0000002": "p2",
            "HP:0000003": "p3",
            "HP:0000004": "p4",
            "HP:0000010": "n1",
            "HP:0000011": "n2",
            "HP:0000012": "n3",
            "HP:0000013": "n4",
        },
        selector=MODULE.HeuristicSelector(seed=7),
        config=MODULE.GenerationConfig(seed=7, max_generation_attempts=12, max_noise_hpo=2),
    )

    assert metadata_df["case_id"].nunique() == 2
    assert list(generated_df.columns) == ["case_id", "mondo_label", "hpo_id", "case_source"]
    assert "seed_case_id" not in audit_df.columns
    assert summary_df.loc[0, "generated_case_count"] == 2

    for row in metadata_df.itertuples(index=False):
        final_hpos = set(str(row.final_hpo_ids).split("|"))
        selected_noise = [item for item in str(row.selected_noise_hpo_ids).split("|") if item]
        assert "HP:0000001" in final_hpos
        assert row.final_hpo_count >= row.min_hpo_count
        assert row.final_hpo_count <= row.max_hpo_count
        assert row.selected_soft_core_count >= 0
        assert len(selected_noise) <= 2


def test_generate_cases_resume_mid_mondo_does_not_duplicate_completed_cases() -> None:
    target_df = pd.DataFrame(
        [
            {
                "mondo_label": "MONDO:0000001",
                "total_case_pair_count": 8,
                "target_generated_case_count": 2,
            }
        ]
    )
    knowledge_df = pd.DataFrame(
        [
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000001", "source_count": 1, "source_names": "GARD", "max_weight": 1.0, "mean_weight": 1.0},
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000002", "source_count": 1, "source_names": "HPOA", "max_weight": 0.8, "mean_weight": 0.8},
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000003", "source_count": 1, "source_names": "orphanet", "max_weight": 0.7, "mean_weight": 0.7},
        ]
    )
    noise_pool_df = pd.DataFrame(
        [
            {"hpo_id": "HP:0000010", "case_support": 20, "hpo_name": "n1"},
            {"hpo_id": "HP:0000011", "case_support": 18, "hpo_name": "n2"},
        ]
    )
    resume_payload = {
        "support_threshold": 10,
        "seed": 7,
        "start_case_number": 1,
        "base_url": MODULE.DEFAULT_BASE_URL,
        "model": MODULE.DEFAULT_MODEL,
        "core_min_sources": 2,
        "hard_core_min_sources": 3,
        "target_hpo_slack": 2,
        "next_case_number": 2,
        "generated_rows": [
            {"case_id": "synthetic_case_1", "mondo_label": "MONDO:0000001", "hpo_id": "HP:0000001", "case_source": "synthetic"},
            {"case_id": "synthetic_case_1", "mondo_label": "MONDO:0000001", "hpo_id": "HP:0000002", "case_source": "synthetic"},
        ],
        "metadata_rows": [
            {
                "case_id": "synthetic_case_1",
                "mondo_label": "MONDO:0000001",
                "case_source": "synthetic",
                "original_total_case_pair_count": 8,
                    "requested_target_total_case_count": 10,
                    "target_hpo_count": 2,
                    "min_hpo_count": 1,
                    "max_hpo_count": 4,
                    "hard_core_count": 0,
                    "soft_core_count": 0,
                    "optional_non_core_count": 3,
                    "noise_candidate_count": 2,
                    "final_hpo_count": 2,
                    "selected_soft_core_count": 0,
                    "selected_optional_count": 2,
                    "selected_noise_count": 0,
                    "hard_core_hpo_ids": "",
                    "soft_core_hpo_ids": "",
                    "optional_non_core_hpo_ids": "HP:0000001|HP:0000002|HP:0000003",
                    "noise_candidate_hpo_ids": "HP:0000010|HP:0000011",
                    "selected_soft_core_hpo_ids": "",
                    "selected_optional_hpo_ids": "HP:0000001|HP:0000002",
                    "selected_noise_hpo_ids": "",
                    "final_hpo_ids": "HP:0000001|HP:0000002",
                "duplicate_retry_count": 0,
                "quality_filter_reason": "keep",
            }
        ],
        "summary_rows": [],
        "audit_rows": [
            {
                "case_id": "synthetic_case_1",
                "mondo_label": "MONDO:0000001",
                "generation_index": 1,
                "attempt_count": 1,
                "selector_mode": "heuristic",
                "status": "accepted",
                "error": "",
                "raw_response": "{}",
                "selected_soft_core_hpo_ids": "",
                "selected_optional_hpo_ids": "HP:0000001|HP:0000002",
                "selected_noise_hpo_ids": "",
                "duplicate_retry_count": 0,
                "used_fallback": False,
            }
        ],
        "completed_mondo_ids": [],
    }

    _, metadata_df, summary_df, _, audit_df = MODULE.generate_cases(
        target_df=target_df,
        real_signatures_by_mondo={},
            hpo_count_stats_by_mondo={"MONDO:0000001": {"median_hpo_count": 2, "min_hpo_count": 1, "max_hpo_count": 3}},
            knowledge_df=knowledge_df,
            noise_pool_df=noise_pool_df,
            mondo_name_map={"MONDO:0000001": "d1"},
            hpo_name_map={"HP:0000001": "p1", "HP:0000002": "p2", "HP:0000003": "p3", "HP:0000010": "n1", "HP:0000011": "n2"},
        selector=MODULE.HeuristicSelector(seed=7),
        config=MODULE.GenerationConfig(seed=7, max_generation_attempts=8),
        resume_payload=resume_payload,
    )

    assert metadata_df["case_id"].nunique() == 2
    assert summary_df.loc[0, "generated_case_count"] == 2
    assert len(audit_df) == 2


def test_generate_cases_parallel_runs_multiple_mondos() -> None:
    target_df = pd.DataFrame(
        [
            {
                "mondo_label": "MONDO:0000001",
                "total_case_pair_count": 9,
                "target_generated_case_count": 1,
            },
            {
                "mondo_label": "MONDO:0000002",
                "total_case_pair_count": 8,
                "target_generated_case_count": 1,
            },
        ]
    )
    knowledge_df = pd.DataFrame(
        [
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000001", "source_count": 3, "source_names": "GARD|HPOA|orphanet", "max_weight": 1.0, "mean_weight": 1.0},
            {"mondo_label": "MONDO:0000001", "hpo_id": "HP:0000002", "source_count": 1, "source_names": "GARD", "max_weight": 0.8, "mean_weight": 0.8},
            {"mondo_label": "MONDO:0000002", "hpo_id": "HP:0000003", "source_count": 3, "source_names": "GARD|HPOA|orphanet", "max_weight": 1.0, "mean_weight": 1.0},
            {"mondo_label": "MONDO:0000002", "hpo_id": "HP:0000004", "source_count": 1, "source_names": "HPOA", "max_weight": 0.7, "mean_weight": 0.7},
        ]
    )
    noise_pool_df = pd.DataFrame(
        [
            {"hpo_id": "HP:0000010", "case_support": 20, "hpo_name": "n1"},
            {"hpo_id": "HP:0000011", "case_support": 18, "hpo_name": "n2"},
        ]
    )

    generated_df, metadata_df, summary_df, params_df, audit_df = MODULE.generate_cases(
        target_df=target_df,
        real_signatures_by_mondo={},
        hpo_count_stats_by_mondo={
            "MONDO:0000001": {"median_hpo_count": 2, "min_hpo_count": 1, "max_hpo_count": 3},
            "MONDO:0000002": {"median_hpo_count": 2, "min_hpo_count": 1, "max_hpo_count": 3},
        },
        knowledge_df=knowledge_df,
        noise_pool_df=noise_pool_df,
        mondo_name_map={"MONDO:0000001": "d1", "MONDO:0000002": "d2"},
        hpo_name_map={
            "HP:0000001": "p1",
            "HP:0000002": "p2",
            "HP:0000003": "p3",
            "HP:0000004": "p4",
            "HP:0000010": "n1",
            "HP:0000011": "n2",
        },
        selector=MODULE.HeuristicSelector(seed=11),
        config=MODULE.GenerationConfig(seed=11, max_generation_attempts=8, max_noise_hpo=1),
        max_workers=2,
    )

    assert metadata_df["case_id"].nunique() == 2
    assert set(summary_df["mondo_label"]) == {"MONDO:0000001", "MONDO:0000002"}
    assert set(audit_df["status"]) == {"accepted"}
    assert "seed_case_id" not in audit_df.columns
    assert int(params_df.loc[params_df["parameter"] == "max_workers", "value"].iloc[0]) == 2
    assert set(generated_df["mondo_label"]) == {"MONDO:0000001", "MONDO:0000002"}


def test_generate_cases_caps_requested_count_for_deterministic_hard_core_cases() -> None:
    target_df = pd.DataFrame(
        [
            {
                "mondo_label": "MONDO:0000009",
                "total_case_pair_count": 1,
                "target_generated_case_count": 5,
            }
        ]
    )
    knowledge_df = pd.DataFrame(
        [
            {"mondo_label": "MONDO:0000009", "hpo_id": "HP:0000001", "source_count": 3, "source_names": "GARD|HPOA|orphanet", "max_weight": 1.0, "mean_weight": 1.0},
            {"mondo_label": "MONDO:0000009", "hpo_id": "HP:0000002", "source_count": 3, "source_names": "GARD|HPOA|orphanet", "max_weight": 0.9, "mean_weight": 0.9},
        ]
    )
    noise_pool_df = pd.DataFrame(
        [
            {"hpo_id": "HP:0000010", "case_support": 20, "hpo_name": "n1"},
            {"hpo_id": "HP:0000011", "case_support": 18, "hpo_name": "n2"},
        ]
    )

    _, metadata_df, summary_df, _, audit_df = MODULE.generate_cases(
        target_df=target_df,
        real_signatures_by_mondo={},
        hpo_count_stats_by_mondo={"MONDO:0000009": {"median_hpo_count": 1, "min_hpo_count": 1, "max_hpo_count": 2}},
        knowledge_df=knowledge_df,
        noise_pool_df=noise_pool_df,
        mondo_name_map={"MONDO:0000009": "d9"},
        hpo_name_map={"HP:0000001": "p1", "HP:0000002": "p2", "HP:0000010": "n1", "HP:0000011": "n2"},
        selector=MODULE.HeuristicSelector(seed=5),
        config=MODULE.GenerationConfig(seed=5, max_generation_attempts=8, max_noise_hpo=2),
        max_workers=1,
    )

    assert metadata_df["case_id"].nunique() == 1
    assert summary_df.loc[0, "requested_generated_case_count"] == 5
    assert summary_df.loc[0, "effective_requested_generated_case_count"] == 1
    assert summary_df.loc[0, "generated_case_count"] == 1
    assert summary_df.loc[0, "failed_case_count"] == 0
    assert bool(summary_df.loc[0, "deterministic_case_only"]) is True
    assert set(audit_df["selector_mode"]) == {"deterministic"}
