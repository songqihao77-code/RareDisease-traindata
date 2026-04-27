# Multiview Hyperedge Feature Probe Summary

## Scope

- 只读取已有 `top50_candidates_validation.csv` 和 `top50_candidates_test.csv`。
- 不训练模型，不覆盖主线 checkpoint、exact evaluation 或 `mainline_final_metrics.csv`。
- `test` 结果只用于固定分析；任何权重、阈值或融合策略仍必须在 `validation` 上选择。

## Dataset-Level Signal

| split | dataset | cases | top1 | top5 | gold_in_top50 | gold_mean_ic | non_gold_mean_ic | ic_gap | wrong_top1_high_overlap_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test | DDD | 761 | 0.3062 | 0.4901 | 0.7438 | 0.5012 | 0.1669 | 0.3343 | 0.5015 |
| test | HMS | 25 | 0.3200 | 0.4800 | 0.8800 | 0.2140 | 0.0852 | 0.1288 | 0.5714 |
| test | LIRICAL | 59 | 0.5085 | 0.6780 | 0.8475 | 0.5407 | 0.1793 | 0.3615 | 0.5000 |
| test | MME | 10 | 0.9000 | 0.9000 | 0.9000 | 0.5242 | 0.0634 | 0.4608 | nan |
| test | MyGene2 | 33 | 0.8788 | 0.8788 | 0.9697 | 0.4614 | 0.1177 | 0.3437 | 0.6667 |
| test | RAMEDIS | 217 | 0.7880 | 0.9309 | 0.9862 | 0.1831 | 0.0739 | 0.1093 | 0.6744 |
| test | mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.1778 | 0.3412 | 0.6252 | 0.1376 | 0.0779 | 0.0597 | 0.7088 |
| validation | DDD | 164 | 0.3537 | 0.5061 | 0.7134 | 0.5275 | 0.1443 | 0.3832 | 0.4915 |
| validation | FakeDisease | 1163 | 0.7549 | 0.8667 | 0.9415 | 0.6483 | 0.1449 | 0.5034 | 0.6267 |
| validation | HMS | 3 | 0.3333 | 1.0000 | 1.0000 | 0.1950 | 0.1123 | 0.0828 | 0.5000 |
| validation | LIRICAL | 33 | 0.1515 | 0.4242 | 0.7879 | 0.4718 | 0.1821 | 0.2897 | 0.4286 |
| validation | MME | 3 | 1.0000 | 1.0000 | 1.0000 | 0.3665 | 0.0654 | 0.3012 | nan |
| validation | MyGene2 | 11 | 0.3636 | 0.5455 | 0.5455 | 0.4363 | 0.2551 | 0.1812 | 0.5000 |
| validation | RAMEDIS | 40 | 0.7000 | 0.8500 | 0.9000 | 0.1728 | 0.0700 | 0.1028 | 0.7500 |
| validation | mimic_rag_0425 | 729 | 0.2510 | 0.4005 | 0.6818 | 0.1425 | 0.0766 | 0.0658 | 0.7452 |

## Required Questions

1. gold candidate 的 overlap 特征整体高于 non-gold。以 DDD test 为例，gold `ic_weighted_overlap` 均值为 0.5012，non-gold 为 0.1669，差值 0.3343。
2. DDD 中 gold 已在 top50 但未 top1 的样本是主要可作用区间：这类样本 333 个。hyperedge overlap 可用于 top50 内重排，但不能召回 top50 外 gold。
3. mimic_test 中 gold 不在 top50 的样本占比仍高，test `gold_in_top50`=0.6252，miss rate=0.3748；top50 内 rerank 对这些样本无能为力。
4. top1 错误样本中，错误 top1 经常拥有不低于 gold 的 exact overlap：DDD test 该比例为 0.5015。
5. 因此单纯按 overlap 降序重排有风险；应保留 HGNN score，并只在 validation 上选择融合权重。
6. DDD gold overlap 明显更高，支持尝试 no-train multiview reranker，优先作为 top50 内排序模块。
7. 当前证据更支持 DDD top50 内排序收益，不支持声称 mimic_test 会因 reranker 获得全局明显提升。
8. `ic_weighted_overlap` 与 `is_gold` 存在正相关，但相关性不是替代模型的充分证据；建议作为 reranker 特征而不是 encoder 改造依据。

## Outputs

- `outputs/multiview_hyperedge_probe/candidate_multiview_features.csv`
- `outputs/multiview_hyperedge_probe/dataset_feature_summary.csv`
- `outputs/multiview_hyperedge_probe/feature_gold_correlation.csv`
