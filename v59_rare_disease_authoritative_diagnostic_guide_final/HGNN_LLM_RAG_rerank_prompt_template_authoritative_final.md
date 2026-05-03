# HGNN-LLM RAG rerank prompt template — authoritative final

## Purpose

Use this prompt to rerank HGNN top-30 candidate diseases with the fixed evidence library `v59 Rare Disease Authoritative Diagnostic Guide for HGNN-LLM RAG Reranking`.

The LLM must only rerank the supplied HGNN top-30 candidates. It must not introduce diseases outside the candidate list and must not use medical knowledge outside the provided disease cards or RAG chunks.

## Required input format

```json
{
  "case_id": "case identifier",
  "case_hpo_terms": [
    {"hpo_id": "HP:xxxxxxx", "label": "HPO label", "observed": true}
  ],
  "hgnn_top30_candidates": [
    {
      "rank": 1,
      "mondo_id": "MONDO:xxxxxxx",
      "hgnn_score": 0.0,
      "disease_name": "candidate disease name"
    }
  ],
  "rag_evidence": [
    {
      "mondo_id": "MONDO:xxxxxxx",
      "chunk_type": "phenotype",
      "retrieval_text": "text from rag_chunks_authoritative_final.jsonl",
      "metadata": {}
    }
  ]
}
```

## Hard rules

1. Only reorder the supplied HGNN top-30 candidates.
2. Do not add a new disease.
3. Do not use facts outside the provided disease card/RAG chunks.
4. `NOT_FOUND` means the evidence library does not contain that field; it does not mean the feature is absent in the disease.
5. v59 `weight` is HGNN hyperedge weight, not clinical frequency.
6. HPOA/Orphanet/GeneReviews/OMIM frequencies, when present, are the only clinical frequency evidence.
7. `candidate_mimics_for_rerank` are model/data-derived HPO-overlap candidates, not authoritative clinical differential diagnosis.
8. `authoritative_differentials` may be used as clinical differential evidence only when cited.
9. Low-confidence or manual-review cards must be penalized or explicitly marked uncertain.
10. Do not convert gene association into confirmatory testing unless a cited confirmatory test is present.

## Suggested scoring rubric

Score each candidate from 0 to 100 using only provided evidence.

- `phenotype_match_score` (0-30): exact and semantically close matches between case HPO and candidate evidence.
- `high_value_hpo_match_score` (0-15): match to cardinal/high-value authoritative phenotypes or top v59 HPO. Remember v59 weights are not clinical frequencies.
- `v59_weighted_match_score` (0-10): weighted support from matched v59 HPO terms. Treat as model-derived support only.
- `authoritative_phenotype_match_score` (0-20): support from HPOA/Orphanet/GeneReviews/OMIM phenotype evidence.
- `gene_inheritance_support_score` (0-10): only if case contains gene or inheritance evidence and the card cites matching gene/inheritance.
- `onset_support_score` (0-5): only if case onset is provided and matches cited onset.
- `differential_support_score` (0-5): use authoritative differentials when available; use candidate mimics only as a caution for tie-breaking.
- `uncertainty_penalty` (-0 to -25): penalize low-confidence, manual-review, obsolete ID without mapping, missing authoritative phenotype source, or evidence conflicts.

## Handling obsolete terms

- Obsolete MONDO: keep original ID for exact evaluation; use replacement only for RAG expansion or relaxed evaluation.
- Obsolete HPO: compute strict match using original HPO and optional mapped match using `mapped_hpo_for_rag`; report both if used.

## Output format

Return strict JSON only:

```json
{
  "case_id": "case identifier",
  "reranked_candidates": [
    {
      "rank": 1,
      "mondo_id": "MONDO:xxxxxxx",
      "disease_name": "candidate disease name",
      "score": 0.0,
      "score_components": {
        "phenotype_match_score": 0.0,
        "high_value_hpo_match_score": 0.0,
        "v59_weighted_match_score": 0.0,
        "authoritative_phenotype_match_score": 0.0,
        "gene_inheritance_support_score": 0.0,
        "onset_support_score": 0.0,
        "differential_support_score": 0.0,
        "uncertainty_penalty": 0.0
      },
      "supporting_evidence": [
        "Cite only evidence present in the supplied RAG chunks."
      ],
      "contradicting_evidence": [
        "Only include if the card explicitly states a conflict or exclusion."
      ],
      "uncertainty": [
        "low_confidence_card=true",
        "manual_review_required",
        "confirmatory_tests NOT_FOUND"
      ],
      "notes": "Do not treat NOT_FOUND as negative evidence."
    }
  ]
}
```

## Experiment reproducibility

Use the same prompt, same evidence library version, same chunk retrieval settings, and fixed HGNN top-30 candidates for all test runs. Tune any scoring weights on validation only; do not tune on test.
