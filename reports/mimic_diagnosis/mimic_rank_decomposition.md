# mimic_test Rank Decomposition

## 输入与口径
- exact details: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage3_exact_eval\exact_details.csv`
- rank 使用 HGNN full-pool exact evaluation 的 `true_rank`，不是后处理后的 `mainline_final_case_ranks.csv`。
- top50 candidate recall 等价于 `true_rank <= 50`。

## Summary
| num_cases | top1 | top3 | top5 | rank_le_10 | rank_le_20 | rank_le_50 | median_rank | mean_rank | gold_absent_from_top50_count | gold_absent_from_top50_ratio | gold_in_top50_but_rank_gt5_count | gold_in_top50_but_rank_gt5_ratio | gold_in_top5_but_not_top1_count | gold_in_top5_but_not_top1_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1873 | 0.1778 | 0.2883 | 0.3412 | 0.4218 | 0.4891 | 0.6252 | 22.0000 | 416.3604 | 702 | 0.3748 | 532 | 0.2840 | 306 | 0.1634 |

## Rank Histogram
| rank_bin | count | ratio |
| --- | --- | --- |
| 1 | 333 | 0.1778 |
| 2-3 | 207 | 0.1105 |
| 4-5 | 99 | 0.0529 |
| 6-10 | 151 | 0.0806 |
| 11-20 | 126 | 0.0673 |
| 21-50 | 255 | 0.1361 |
| 51-100 | 193 | 0.1030 |
| 101-500 | 308 | 0.1644 |
| >500 | 201 | 0.1073 |

## 结论
- mimic_test 低准确率的首要拆分结论：gold 不在 top50 的 candidate recall 问题更大。
- `gold_absent_from_top50` 为 702/1873 (0.3748)。
- `gold_in_top50_but_rank_gt5` 为 532/1873 (0.2840)。
- 对 `gold_absent_from_top50` 样本，单纯 reranker 或 hard negative 只能重排已有候选，理论上不能直接解决这部分样本；需要 candidate expansion、ontology-aware retrieval、similar-case retrieval 或 label/HPO 修复先把 gold 放进候选池。
