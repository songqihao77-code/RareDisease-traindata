# Final quality control report — authoritative diagnostic guide

## Coverage checks

| Check | Result |
|---|---:|
| v59 unique MONDO | 10772 |
| final disease cards | 10772 |
| MONDO set coverage | PASS |
| v59 MONDO-HPO edges | 227907 |
| final card HPO edges | 227907 |
| HPO count mismatch by disease | 0 |
| raw_weight / weight mismatch | 0 |
| JSONL valid | True |

## Source citation and fact policy

- External medical facts are inherited from v2 source citations.
- Candidate mimics are relabeled as `candidate_mimic` / `model_derived`.
- `authoritative_differentials` are empty unless an authoritative source exists.
- `confirmatory_tests` are empty unless a cited test source exists.

## Name mapping

- provided_name filled: 0 / 10772
- mapping file provided: False
- action: keep `provided_name=NOT_FOUND`; no disease name was fabricated.

## Evidence coverage

| Field | Count |
|---|---:|
| authoritative phenotype source | 10097 |
| confirmatory tests | 0 |
| authoritative differential diagnosis | 0 |
| candidate mimics for rerank | 10770 |
| age_of_onset | 4726 |
| clinical_course | 504 |
| genes | 7313 |
| inheritance | 7795 |

## Differential diagnosis layering

- authoritative differential diagnosis total rows: 0
- candidate mimics total rows: 32309
- candidate mimics are flagged `clinical_authority=false`.

## Obsolete ontology handling

| Item | Count |
|---|---:|
| obsolete MONDO rows | 211 |
| obsolete MONDO with replacement | 43 |
| obsolete MONDO without replacement | 168 |
| obsolete HPO rows | 962 |
| obsolete HPO with mapping | 957 |
| obsolete HPO without mapping | 5 |

## Final card status

| Status | Count |
|---|---:|
| ready_with_limitations | 7778 |
| manual_review_required | 2994 |


## Review priority

| Priority | Count |
|---|---:|
| medium | 5873 |
| high | 4899 |


## Low confidence and manual review

- low_confidence_card: 2816
- manual_review_required / needs_manual_review: 2994
- usable for rerank with limitations: 7778
- requires manual review before strong explanation: 2994

## Most missing fields

- provided_name: 10772
- confirmatory_tests: 10772
- authoritative_differential_diagnosis: 10772
- clinical_course: 10268
- age_of_onset: 6046
- genes: 3459
- inheritance: 2977
- authoritative_phenotypes: 675

## Manual review recommendations

1. Review missing authoritative phenotype source disease cards.
2. Review obsolete MONDO without replacement.
3. Review obsolete HPO without replacement/consider.
4. Curate confirmatory tests from GeneReviews/OMIM/Orphanet only when cited.
5. Curate authoritative clinical differentials separately from HPO-overlap candidate mimics.

## Automated final QC scan

- card_lines: 10772
- edge_count_v59: 227907
- json_valid_lines: 10772
- mondo_set_equal_v59: 1
- total_hpo_edges_cards: 227907
- unique_hpo_v59: 10792
- unique_mondo_cards: 10772
- unique_mondo_v59: 10772

No blocking QC examples were detected.
