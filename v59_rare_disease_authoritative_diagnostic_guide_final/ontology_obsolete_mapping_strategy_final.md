# Ontology obsolete mapping strategy — authoritative final

## Scope

This strategy applies to `v59_rare_disease_authoritative_diagnostic_guide_final`. It preserves the original v59 disease space and original v59 MONDO-HPO edges.

## Obsolete MONDO policy

- Original v59 MONDO IDs are never deleted from disease cards.
- Exact evaluation keeps the original v59 MONDO ID.
- If a MONDO term has `replaced_by`, RAG retrieval may expand to the replacement term, but the card keeps the original ID.
- Replacement hits must be reported only in a separate relaxed evaluation, not as primary exact-match performance.
- Obsolete MONDO without clear replacement is `manual_review_required`.

Summary:
- obsolete MONDO rows: 211
- with replacement: 43
- without replacement / manual review: 168

## Obsolete HPO policy

- Original v59 HPO IDs, `raw_weight`, and `weight` are never deleted or modified.
- If `replaced_by` or `consider` exists, RAG indexing may include mapped HPO IDs while retaining the old HPO ID.
- Rerank scoring should report `strict_hpo_match_score` and `mapped_hpo_match_score` separately.
- Obsolete HPO without replacement/consider must be manually reviewed.

Summary:
- obsolete HPO rows: 962
- with replacement/consider mapping: 957
- without mapping / manual review: 5

## Recommended scoring

1. Use original v59 HPO for strict matching.
2. Use mapped HPO only as an auxiliary feature.
3. Never convert an obsolete HPO edge into a new edge by deleting the original.
4. In rerank explanations, explicitly state when mapped HPO evidence contributed.
