# 下一步 Codex 执行建议

## 直接回答
1. 已经达到全部 DeepRare 标准的数据集：RareBench-MME, RareBench-RAMEDIS, MyGene。
2. 只达到 Top5、但 Top1/Top3 未达标的数据集：MIMIC-IV-Rare。
3. 距离较远的数据集：RareBench-HMS, DDD。near-miss/high_variance：RareBench-LIRICAL。unavailable：Xinhua Hosp.。
4. 每个数据集还差多少 case 见下表。
5. 第一优先级：MIMIC-IV-Rare 做 validation-selected light reranker 的 top5/top10 局部排序；DDD 同时做 reranking 与 candidate recall audit。Top5 未达标的数据集才继续 candidate expansion。
6. 需要 hard negative training，尤其是 DDD，因为 Rank<=50 明显高于 Top5，说明很多 gold 可恢复但排序不够靠前。
7. 图对比学习继续后置，等 label/mapping 与 candidate recall audit 稳定后再做。
8. 最适合写入论文主表的结果：outputs/mainline_full_pipeline/mainline_final_metrics_with_sources.csv 中的 current mainline final strict exact fixed-test metrics。
9. 只能做 supplementary：mimic any-label 指标、relaxed MONDO/ancestor/sibling/synonym/replacement 匹配、小样本 high_variance 分析，以及 unavailable/not-comparable mapping。

## Case Gaps
| deeprare_dataset_name | project_dataset                     | num_cases | top1_case_gap | top3_case_gap | top5_case_gap | target_status     | priority_level | notes                                                   |
| --------------------- | ----------------------------------- | --------- | ------------- | ------------- | ------------- | ----------------- | -------------- | ------------------------------------------------------- |
| RareBench-MME         | MME                                 | 10        | 0             | 0             | 0             | all_reached       | P3             | high_variance                                           |
| RareBench-HMS         | HMS                                 | 25        | 7             | 6             | 6             | far_from_target   | P2             | high_variance                                           |
| RareBench-LIRICAL     | LIRICAL                             | 59        | 4             | 1             | 1             | far_from_target   | P2             | high_variance; strict target 未全达，但 Top3/Top5 只差约 1 case |
| RareBench-RAMEDIS     | RAMEDIS                             | 217       | 0             | 0             | 0             | all_reached       | P3             |                                                         |
| MIMIC-IV-Rare         | mimic_test_recleaned_mondo_hpo_rows | 1873      | 152           | 53            | 0             | top5_reached_only | P0             | Top5 已达标；Top1/Top3 仍低于 target                           |
| MyGene                | MyGene2                             | 33        | 0             | 0             | 0             | all_reached       | P3             | high_variance; 从 MyGene2 近似映射                           |
| DDD                   | DDD                                 | 761       | 80            | 79            | 69            | far_from_target   | P0             |                                                         |
| Xinhua Hosp.          |                                     |           |               |               |               | unavailable       | P3             | unavailable / not comparable                            |

## 推荐下一步任务
实现一个统一的 validation-selected light reranker 实验：
- Input：stage4 当前 top50 candidate tables 加 current final ranks。
- Scope：MIMIC-IV-Rare top5/top10 局部排序，以及 DDD top50 reranking。
- Features：HGNN score/rank/margin、SimilarCase score、exact HPO overlap、IC overlap、semantic HPO coverage、MONDO relation features、source-count features。
- Selection：validation-only objective，优先 Recall@1 和 Recall@3，并拒绝导致 Recall@5 下降的配置。
- Output：fixed-test strict exact Recall@1/@3/@5；supplementary relaxed analyses 单独成表。
