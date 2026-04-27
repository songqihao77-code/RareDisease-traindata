# mimic_test Candidate Recall Audit

- candidate file: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage4_candidates\top50_candidates_test.csv`
- output: `D:\RareDisease-traindata\outputs\mimic_diagnosis\mimic_top50_candidates_with_evidence.csv`

## Relation Counts in Top50 Rows
| relation | candidate_count |
| --- | --- |
| shared_ancestor | 79929 |
| unrelated_or_unknown | 5296 |
| same_parent | 3225 |
| candidate_descendant_of_gold | 2609 |
| same_disease | 1311 |
| candidate_ancestor_of_gold | 1158 |
| synonym_or_name_match | 122 |

## Gold Absent Case Summary
| gold_absent_cases | absent_top50_has_synonym_cases | absent_top50_has_parent_cases | absent_top50_has_child_cases | absent_top50_has_sibling_cases | absent_top50_has_shared_ancestor_cases | absent_top50_mean_max_shared_hpo_count | absent_top50_mean_max_ic_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 702 | 16 | 99 | 81 | 253 | 677 | 3.1481 | 0.3418 |

## 结论
- gold 不在 top50 的 case 中，如果 top50 经常出现 parent/child/sibling/shared_ancestor，说明需要 candidate expansion / ontology-aware retrieval。
- gold 在 top50 但排得低的 case，更适合 reranker / hard negative。
- 如果 top50 完全不相关且 HPO evidence coverage 很低，优先怀疑 HPO 抽取、label mapping 或 MIMIC domain shift。
