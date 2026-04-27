# Failure type summary after SimilarCase-Aug

| failure_type | num_cases | case_ratio | mean_final_rank | median_final_rank | mean_baseline_rank | similar_gold_evidence_ratio | mean_gold_similarity | multilabel_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_similar_case_no_or_weak_evidence | 645 | 0.3444 | 9274.4155 | 9999.0000 | 1115.8403 | 0.0000 | 0.0000 | 0.1674 |
| B_evidence_but_weight_insufficient | 194 | 0.1036 | 26.9021 | 30.0000 | 268.6031 | 1.0000 | 0.3104 | 0.1186 |
| C_wrong_evidence_interference | 120 | 0.0641 | 4.5583 | 4.0000 | 2.1333 | 0.3667 | 0.1207 | 0.0917 |
| D_candidate_recall_missing | 598 | 0.3193 | 9999.0000 | 9999.0000 | 1199.9766 | 0.0000 | 0.0000 | 0.1689 |
| E_label_ontology_relaxed_hit | 1078 | 0.5755 | 5281.0139 | 9999.0000 | 675.1985 | 0.1790 | 0.0555 | 0.1475 |

## 类型解释
- A：final rank>50 且 gold 的 SimilarCase evidence 很弱或没有，优先看候选扩展或上游数据。
- B：gold 有 evidence 但仍在 6-50，适合 gated rerank / stronger light reranker。
- C：SimilarCase 把错误疾病推高造成误伤，适合 confidence gate 和 HGNN top1 protection。
- D：final top50 不含 gold，candidate recall 缺失，rerank/hard negative 不能单独解决。
- E：Top candidates 与 gold 有 ontology 近邻关系，只能作为 relaxed supplementary。
