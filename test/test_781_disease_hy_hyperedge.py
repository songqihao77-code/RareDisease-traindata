from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from test._script_bootstrap import bootstrap_project


PROJECT_ROOT = bootstrap_project()
SCRIPT_PATH = PROJECT_ROOT / "LLLdataset" / "dataset" / "781DiseaseHyHyperedge.py"
SPEC = importlib.util.spec_from_file_location("disease_hy_781", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeLLM:
    def __init__(self, phenotypes=None, selections=None):
        self.phenotypes = phenotypes or []
        self.selections = selections or {}

    def extract_phenotypes_from_page(self, *, disease_name, aliases, page_content):
        del disease_name, aliases, page_content
        return True, "returned", self.phenotypes

    def select_hpo_candidates(self, *, disease_name, mondo_id, candidate_items):
        del disease_name, mondo_id, candidate_items
        return self.selections


def _build_matcher(tmp_path: Path) -> MODULE.Definition2IdMatcher:
    definition_path = tmp_path / "definition2id.json"
    definition_path.write_text(
        json.dumps(
            {
                "Headache": "HP:0002315",
                "Severe headache": "HP:0002315",
                "Chest pain": "HP:0100749",
                "Weight loss": "HP:0001824",
            }
        ),
        encoding="utf-8",
    )
    return MODULE.Definition2IdMatcher(
        definition_path,
        {
            "HP:0002315": "Headache",
            "HP:0100749": "Chest pain",
            "HP:0001824": "Weight loss",
        },
    )


def test_frequency_text_to_weight_maps_expected_buckets() -> None:
    assert MODULE.frequency_text_to_weight("Always present") == 1.0
    assert MODULE.frequency_text_to_weight("Very frequent (80-99%)") == 0.9
    assert MODULE.frequency_text_to_weight("Frequent") == 0.6
    assert MODULE.frequency_text_to_weight("Occasional") == 0.3
    assert MODULE.frequency_text_to_weight("Uncommon") == 0.1
    assert MODULE.frequency_text_to_weight("") == 0.35


def test_definition_matcher_supports_exact_and_normalized_lookup(tmp_path: Path) -> None:
    matcher = _build_matcher(tmp_path)

    assert matcher.exact_match("Headache") == ("HP:0002315", "definition2id_exact")
    assert matcher.exact_match("  severe-headache  ") == ("HP:0002315", "definition2id_normalized")


def test_aggregate_mapped_rows_merges_sources_and_normalizes_weight() -> None:
    rows = [
        MODULE.MappedPhenotype(
            case_id="case_1",
            case_order=1,
            mondo_id="MONDO:1",
            disease_name="d1",
            source_tier="tier_1",
            phenotype_text="Headache",
            frequency_text="Frequent",
            evidence_span="e1",
            source_title="t1",
            source_url="u1",
            hpo_id="HP:0002315",
            hpo_name="Headache",
            raw_weight=0.6,
            mapping_method="definition2id_exact",
        ),
        MODULE.MappedPhenotype(
            case_id="case_1",
            case_order=1,
            mondo_id="MONDO:1",
            disease_name="d1",
            source_tier="tier_1",
            phenotype_text="Severe headache",
            frequency_text="Very frequent",
            evidence_span="e2",
            source_title="t2",
            source_url="u2",
            hpo_id="HP:0002315",
            hpo_name="Headache",
            raw_weight=0.9,
            mapping_method="llm_candidate_select",
        ),
        MODULE.MappedPhenotype(
            case_id="case_1",
            case_order=1,
            mondo_id="MONDO:1",
            disease_name="d1",
            source_tier="tier_1",
            phenotype_text="Chest pain",
            frequency_text="Occasional",
            evidence_span="e3",
            source_title="t3",
            source_url="u3",
            hpo_id="HP:0100749",
            hpo_name="Chest pain",
            raw_weight=0.3,
            mapping_method="definition2id_exact",
        ),
    ]

    aggregated = MODULE.aggregate_mapped_rows(rows)

    assert len(aggregated) == 2
    headache = next(row for row in aggregated if row["hpo_id"] == "HP:0002315")
    chest_pain = next(row for row in aggregated if row["hpo_id"] == "HP:0100749")
    assert headache["raw_weight"] == 0.9
    assert headache["source_count"] == 2
    assert "definition2id_exact" in headache["mapping_method"]
    assert "llm_candidate_select" in headache["mapping_method"]
    assert round(headache["weight"] + chest_pain["weight"], 6) == 1.0


def test_process_disease_stops_after_first_successful_tier(monkeypatch, tmp_path: Path) -> None:
    matcher = _build_matcher(tmp_path)
    record = MODULE.DiseaseRecord(case_id="case_1", case_order=1, mondo_id="MONDO:1", disease_name="d1")
    page = MODULE.PageContent(
        query="q1",
        search_engine="duckduckgo",
        tier_name="tier_1",
        domain="cancer.gov",
        url="https://www.cancer.gov/types/test",
        final_url="https://www.cancer.gov/types/test",
        title="d1 page",
        text="d1 page text",
        fetch_method="requests",
        status="ok",
        error_reason="",
    )
    phenotypes = [
        MODULE.ExtractedPhenotype(
            phenotype_text="Headache",
            frequency_text="Frequent",
            evidence_span="span",
            negated=False,
            source_title="title",
            source_url="https://www.cancer.gov/types/test",
        )
    ]
    llm = _FakeLLM(phenotypes=phenotypes)
    searched_tiers: list[str] = []

    monkeypatch.setattr(MODULE, "build_requests_session", lambda: object())
    monkeypatch.setattr(MODULE, "build_queries_for_tier", lambda disease_name, aliases, domains: [("q1", domains[0])])
    monkeypatch.setattr(
        MODULE,
        "search_official_pages",
        lambda **kwargs: searched_tiers.append(kwargs["tier_name"]) or [
            MODULE.SearchCandidate(
                query=kwargs["query"],
                search_engine="duckduckgo",
                tier_name=kwargs["tier_name"],
                domain=kwargs["domain"],
                url="https://www.cancer.gov/types/test",
                title="candidate",
                snippet="snippet",
                rank=1,
            )
        ],
    )
    monkeypatch.setattr(MODULE, "fetch_candidate_page", lambda **kwargs: page)
    monkeypatch.setattr(MODULE, "page_matches_disease", lambda page_content, aliases: True)

    result = MODULE.process_disease(
        record=record,
        aliases=["d1"],
        matcher=matcher,
        llm=llm,
        chromedriver_dir=Path("."),
        request_timeout_sec=1,
        request_delay_sec=0.0,
    )

    assert result.summary_row["status"] == "success"
    assert result.summary_row["matched_tier"] == "tier_1"
    assert searched_tiers == ["tier_1"]


def test_process_disease_records_failure_when_all_tiers_fail(monkeypatch, tmp_path: Path) -> None:
    matcher = _build_matcher(tmp_path)
    record = MODULE.DiseaseRecord(case_id="case_1", case_order=1, mondo_id="MONDO:1", disease_name="d1")
    llm = _FakeLLM()

    monkeypatch.setattr(MODULE, "build_requests_session", lambda: object())
    monkeypatch.setattr(MODULE, "build_queries_for_tier", lambda disease_name, aliases, domains: [("q1", domains[0])])
    monkeypatch.setattr(MODULE, "search_official_pages", lambda **kwargs: [])

    result = MODULE.process_disease(
        record=record,
        aliases=["d1"],
        matcher=matcher,
        llm=llm,
        chromedriver_dir=Path("."),
        request_timeout_sec=1,
        request_delay_sec=0.0,
    )

    assert result.final_rows == []
    assert result.summary_row["status"] == "failed"
    assert result.summary_row["failure_reason"] == "no_official_source_found"
    assert result.source_audit_rows


def test_resolve_api_key_supports_default_env_and_misconfigured_api_key_env(monkeypatch) -> None:
    assert MODULE.DEFAULT_API_KEY_ENV == "SILICONFLOW_API_KEY"
    parser = argparse.Namespace(api_key="", api_key_env="SILICONFLOW_API_KEY")

    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-env-test")
    assert MODULE.resolve_api_key(parser) == "sk-env-test"

    parser.api_key_env = "sk-inline-test"
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    assert MODULE.resolve_api_key(parser) == "sk-inline-test"
