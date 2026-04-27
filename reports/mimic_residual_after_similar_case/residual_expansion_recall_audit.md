# Residual expansion recall audit

## Summary
| final_rank_gt50_cases | mondo_relation_recovered_gold_cases | hpo_hyperedge_recovered_gold_cases_top100 | hpo_hyperedge_recovered_gold_cases_top200 | similar_case_top30_recovered_gold_cases_analysis_only | similar_case_top50_recovered_gold_cases_analysis_only | validation_similar_top30_gold_cases | validation_similar_top50_gold_cases | test_analysis_only_similar_top30_gold_cases | test_analysis_only_similar_top50_gold_cases | still_unrecovered_by_any_analysis_expansion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 645 | 75 | 95 | 120 | 0 | 0 | 1809 | 1840 | 0 | 0 | 472 |

## 结论
- test expansion 文件只作为 analysis-only，不用于选择 topk 或权重。
- MONDO relation expansion 若只能找回少量 gold，说明 strict exact miss 不是简单 parent/child/sibling 扩展即可解决。
- HPO hyperedge expansion 找回的 gold 如果主要在 top100/top200，而不是 top5，说明下一步需要 light reranker，而不是只做 candidate generation。
- SimilarCase topk 30/50 只有 validation 有稳定证据时才建议扩大；不能用 test analysis-only 反向选择。
