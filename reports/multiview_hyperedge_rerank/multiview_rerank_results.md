# Multiview Hyperedge Rerank Results

## 1. 实验目的

实现独立 no-train linear fusion reranker，在 HGNN top50 内融合多视图/超边证据，重点验证是否改善 DDD top50 内排序。

## 2. 输入文件

- 候选特征：`outputs/multiview_hyperedge_probe/candidate_multiview_features.csv`
- 输出目录：`outputs/multiview_hyperedge_rerank`

## 3. 约束说明

- 没有训练新模型。
- 没有修改 `src/models/hgnn_encoder.py`、`src/models/model_pipeline.py` 或训练代码。
- 没有覆盖 checkpoint、exact evaluation 或 `outputs/mainline_full_pipeline/mainline_final_metrics.csv`。
- 权重只在 validation split 上选择；test split 只加载固定权重评估一次。

## 4. 使用特征

| feature | source column | status |
|---|---|---|
| hgnn | `candidate_score` | used |
| ic | `ic_weighted_overlap` | used |
| exact | `exact_hpo_overlap` | used |
| jaccard | `jaccard_hpo_overlap` | used |
| case_ratio | `overlap_ratio_case` | used |
| candidate_ratio | `overlap_ratio_candidate` | used |
| ancestor | `ancestor_expanded_overlap` | used |
| semantic | `ancestor_expanded_coverage` | used |

## 5. 跳过特征

无。

## 6. Validation 权重搜索空间

- objective: `ddd_only`
- 实际搜索组合数: `13824`
- `w_hgnn`: [1.0]
- `w_ic`: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
- `w_exact`: [0.0, 0.05, 0.1, 0.2]
- `w_jaccard`: [0.0, 0.05, 0.1, 0.2]
- `w_case_ratio`: [0.0, 0.05, 0.1]
- `w_candidate_ratio`: [0.0, 0.05, 0.1]
- `w_ancestor`: [0.0, 0.05, 0.1, 0.2]
- `w_semantic`: [0.0, 0.05, 0.1, 0.2]

## 7. Selected Weights

- `w_hgnn`=1, `w_ic`=0.5, `w_exact`=0.2, `w_jaccard`=0.2, `w_case_ratio`=0.1, `w_candidate_ratio`=0.1, `w_ancestor`=0.2, `w_semantic`=0

## 8. Validation 最优结果

- `DDD_mean_rank`: 17.573171
- `DDD_rank_le_50`: 0.713415
- `DDD_top1`: 0.457317
- `DDD_top3`: 0.542683
- `DDD_top5`: 0.585366
- `macro_top1`: 0.436536
- `macro_top5`: 0.662311
- `overall_mean_rank`: 12.977633

## 9. Fixed Test 结果

| dataset | top1 | top3 | top5 | top10 | rank_le_50 | mean_rank | top1_delta_vs_hgnn | top5_delta_vs_hgnn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 0.233042 | 0.364674 | 0.428475 | 0.510410 | 0.693083 | 21.6780 | -0.039960 | -0.009402 |
| DDD | 0.391590 | 0.512484 | 0.554534 | 0.633377 | 0.743758 | 16.9987 | 0.085414 | 0.064389 |
| HMS | 0.280000 | 0.440000 | 0.560000 | 0.680000 | 0.880000 | 11.9200 | -0.040000 | 0.080000 |
| LIRICAL | 0.593220 | 0.644068 | 0.694915 | 0.728814 | 0.847458 | 11.5593 | 0.084746 | 0.016949 |
| MME | 0.800000 | 0.900000 | 0.900000 | 0.900000 | 0.900000 | 6.1000 | -0.100000 | 0.000000 |
| MyGene2 | 0.575758 | 0.818182 | 0.818182 | 0.909091 | 0.969697 | 4.9091 | -0.303030 | -0.060606 |
| RAMEDIS | 0.294931 | 0.705069 | 0.834101 | 0.935484 | 0.986175 | 4.4608 | -0.493088 | -0.096774 |
| mimic_test_recleaned_mondo_hpo_rows | 0.140416 | 0.244527 | 0.310731 | 0.392952 | 0.625200 | 26.4015 | -0.037373 | -0.030432 |

## 10. DDD 专项结果

- HGNN baseline DDD top1/top3/top5: `0.306176/0.438896/0.490145`。
- Multiview DDD top1/top3/top5: `0.391590/0.512484/0.554534`。
- DDD recovered cases: `87`；harmed cases: `22`。

## 11. mimic_test 结果

- `mimic_test_recleaned_mondo_hpo_rows` HGNN baseline top1/top5/gold_in_top50: `0.177790/0.341164/0.625200`。
- `mimic_test_recleaned_mondo_hpo_rows` multiview top1/top5/gold_in_top50: `0.140416/0.310731/0.625200`。
- gold 不在 top50 的比例约 `0.374800`；top50 reranker 无法修复这些样本。

## 12. 与 HGNN baseline 对比

| dataset | hgnn_top1 | multiview_top1 | delta_top1 | hgnn_top5 | multiview_top5 | delta_top5 | gold_in_top50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 0.273002 | 0.233042 | -0.039960 | 0.437878 | 0.428475 | -0.009402 | 0.693083 |
| DDD | 0.306176 | 0.391590 | 0.085414 | 0.490145 | 0.554534 | 0.064389 | 0.743758 |
| HMS | 0.320000 | 0.280000 | -0.040000 | 0.480000 | 0.560000 | 0.080000 | 0.880000 |
| LIRICAL | 0.508475 | 0.593220 | 0.084746 | 0.677966 | 0.694915 | 0.016949 | 0.847458 |
| MME | 0.900000 | 0.800000 | -0.100000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 |
| MyGene2 | 0.878788 | 0.575758 | -0.303030 | 0.878788 | 0.818182 | -0.060606 | 0.969697 |
| RAMEDIS | 0.788018 | 0.294931 | -0.493088 | 0.930876 | 0.834101 | -0.096774 | 0.986175 |
| mimic_test_recleaned_mondo_hpo_rows | 0.177790 | 0.140416 | -0.037373 | 0.341164 | 0.310731 | -0.030432 | 0.625200 |

## 13. 与现有 DDD evidence rerank 对比

| method | top1 | top3 | top5 | rank_le_50 | mean_rank |
|---|---:|---:|---:|---:|---:|
| hgnn_top50_baseline | 0.306176 | 0.438896 | 0.490145 | 0.743758 | 18.7372 |
| existing_ddd_evidence_rerank | 0.375821 | 0.496715 | 0.540079 | 0.743758 | 17.2050 |
| new_multiview_hyperedge_rerank | 0.391590 | 0.512484 | 0.554534 | 0.743758 | 16.9987 |
- 新 multiview rerank 的 DDD top1 超过现有 DDD evidence rerank。

## 14. recovered/harmed case 数量

| transition | num_cases | case_fraction |
|---|---:|---:|
| harmed | 22 | 0.028909 |
| recovered | 87 | 0.114323 |
| unchanged_correct | 211 | 0.277267 |
| unchanged_wrong | 441 | 0.579501 |

## 15. 是否建议进入主线

建议作为 DDD 后处理主线候选，但仍需保留现有 DDD evidence rerank 对照。

## 16. 是否建议进入论文主表

可作为主表候选，前提是明确 validation-selected、fixed-test 协议。

## 17. 下一步建议

- 如果后续复现稳定，可作为主表候选；下一步比较 no-train reranker 与 lightweight reranker。
- 不建议下一步直接改 encoder；hard negative mining 只适合作为独立训练 ablation。
