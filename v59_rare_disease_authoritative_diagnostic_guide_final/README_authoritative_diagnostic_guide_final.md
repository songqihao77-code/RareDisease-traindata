# v59 Rare Disease Authoritative Diagnostic Guide for HGNN-LLM RAG Reranking

## Project goal

This is a structured, auditable rare-disease evidence library for HGNN top-30 candidate reranking with LLM + RAG. It is not a direct clinical diagnostic guideline for patient care.

## Why “authoritative diagnostic guide / RAG evidence library”

The library is “authoritative” because every external medical fact is tied to cited sources already integrated in v2, and unsupported fields remain `NOT_FOUND` or empty. It is a diagnostic guide for model reranking, not a full clinical practice guideline.

## Inputs

- `v59_rare_disease_rag_evidence_library_v2_final.zip`
- `v59_hyperedge_weighted_patched.xlsx`
- `mondo_name_mapping.csv/xlsx`: not provided in this run
- top-30 candidate files: not provided in this run

## Outputs

See the final zip for machine-readable disease cards, RAG chunks, quality reports, obsolete mapping reports, and rerank prompt template.

## Data scale

- v59 unique MONDO diseases: 10772
- v59 MONDO-HPO edges: 227907
- v59 unique HPO terms: 10792
- final disease cards: 10772
- final RAG chunks: 53858

## Evidence coverage

- authoritative phenotype source: 10097 / 10772
- confirmatory tests: 0 / 10772
- authoritative differential diagnosis: 0 / 10772
- candidate mimics for rerank: 10770 / 10772
- age of onset: 4726 / 10772
- clinical course: 504 / 10772
- genes: 7313 / 10772
- inheritance: 7795 / 10772

## Field principles

- `NOT_FOUND` means not captured in this evidence library, not evidence of absence.
- v59 `weight` is HGNN hyperedge weight, not clinical frequency.
- `candidate_mimics_for_rerank` are model-derived HPO-overlap mimics with `clinical_authority=false`.
- `authoritative_differentials` must come from authoritative clinical or expert sources; none were added automatically in this run.
- `confirmatory_tests` are empty unless a cited source explicitly supports a test.

## Quality flags

- `low_confidence_card=true` means use with uncertainty.
- `manual_review_required` means the card should not be used as strong primary explanation without curation.
- Obsolete MONDO/HPO are retained; replacement/consider mappings only augment RAG or relaxed evaluation.

## Final card status

{'ready_with_limitations': 7778, 'manual_review_required': 2994}

## Limitations

- No new disease-name mapping file was provided, so `provided_name` remains `NOT_FOUND`.
- No top-30 candidate frequency file was provided, so candidate mimics are based on v59/HPO-overlap, not HGNN co-occurrence.
- Confirmatory tests and authoritative clinical differential diagnosis were not mechanically inferred.
- This resource is for model reranking and error analysis, not direct clinical decision-making.

## Recommended manual review

Prioritize:
1. `manual_review_required`
2. `low_confidence_card=true`
3. obsolete MONDO without replacement
4. obsolete HPO without replacement/consider
5. no authoritative phenotype source
6. high-weight v59 HPO with external validation conflict
