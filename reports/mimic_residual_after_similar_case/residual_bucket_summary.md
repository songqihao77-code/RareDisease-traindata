# Residual bucket summary after SimilarCase-Aug

| bucket | num_cases | case_ratio | final_top1 | final_top3 | final_top5 | final_rank_le_50 | final_median_rank | final_mean_rank | mean_baseline_rank | mean_rank_delta_baseline_minus_final | similar_gold_evidence_ratio | obsolete_label_ratio | multilabel_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_rank>50 | 645 | 0.3444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9999.0000 | 9274.4155 | 1115.8403 | -8158.5752 | 0.0000 | 0.0481 | 0.1674 |
| final_rank 21-50 | 259 | 0.1383 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 33.0000 | 33.9035 | 209.4479 | 175.5444 | 0.5328 | 0.0116 | 0.1236 |
| final_rank 6-20 | 215 | 0.1148 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10.0000 | 10.9116 | 13.3814 | 2.4698 | 0.2605 | 0.0326 | 0.1163 |
| final_rank 2-5 | 362 | 0.1933 | 0.0000 | 0.6878 | 1.0000 | 1.0000 | 3.0000 | 3.0304 | 6.1685 | 3.1381 | 0.8398 | 0.0166 | 0.1050 |
| final_rank=1 | 392 | 0.2093 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.9617 | 0.9617 | 1.0000 | 0.0102 | 0.0612 |
| baseline_rank>50 -> final_rank<=50 | 104 | 0.0555 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 35.0000 | 35.7212 | 472.8750 | 437.1538 | 1.0000 | 0.0096 | 0.1250 |
| baseline_rank 6-50 -> final_rank<=5 | 168 | 0.0897 | 0.1548 | 0.6488 | 1.0000 | 1.0000 | 3.0000 | 2.9464 | 11.4702 | 8.5238 | 1.0000 | 0.0060 | 0.0655 |
| baseline_rank<=5 -> final_rank>5 harmed | 53 | 0.0283 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 7.0000 | 7.1509 | 3.5660 | -3.5849 | 0.0000 | 0.0377 | 0.1321 |
| baseline_rank=1 -> final_rank>1 harmed | 69 | 0.0368 | 0.0000 | 0.8551 | 0.9710 | 1.0000 | 2.0000 | 2.6087 | 1.0000 | -1.6087 | 0.6377 | 0.0435 | 0.0580 |

## 结论
- `final_rank>50` 是 SimilarCase-Aug 后仍无法由 rerank 直接解决的主要 residual recall 问题。
- `final_rank 6-20` 和 `21-50` 是 stronger rerank / gated multiview evidence 的主要目标。
- harmed bucket 说明 SimilarCase-Aug 存在误伤，需要 confidence gate 或 HGNN top1 protection。
