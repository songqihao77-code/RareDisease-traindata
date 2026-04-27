# mimic_test Mainline Final Reanalysis

## 口径修正
- 本报告使用 `outputs/mainline_full_pipeline/mainline_final_case_ranks.csv` 的 `final_rank`，不是只看 `stage3_exact_eval/exact_details.csv` 的 baseline `true_rank`。
- `mimic_test_recleaned_mondo_hpo_rows` 在最终汇总中应用的模块是 `similar_case_fixed_test`。
- 没有重跑训练；没有覆盖 mainline 输出。

## 当前 mainline final 指标
| dataset | cases | top1 | top3 | top5 | rank_le_50 |
| --- | --- | --- | --- | --- | --- |
| mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.2093 | 0.3422 | 0.4026 | 0.6556 |

## 指标来源
| dataset | cases | top1 | top3 | top5 | rank_le_50 | module_applied | source_result_path | source_dataset_name | checkpoint_path | data_config_path | train_config_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.2093 | 0.3422 | 0.4026 | 0.6556 | similar_case_fixed_test | D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage6_mimic_similar_case\similar_case_fixed_test.csv | mimic_test_recleaned_mondo_hpo_rows | D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline\configs\stage2_finetune.yaml |

## Baseline vs Final
| rank_source | num_cases | top1 | top3 | top5 | rank_le_10 | rank_le_20 | rank_le_50 | rank_gt_50_count | rank_gt_50_ratio | gold_in_top50_but_rank_gt5_count | gold_in_top50_but_rank_gt5_ratio | gold_in_top5_but_not_top1_count | gold_in_top5_but_not_top1_ratio | median_rank | mean_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HGNN exact baseline | 1873 | 0.1778 | 0.2883 | 0.3412 | 0.4218 | 0.4891 | 0.6252 | 702 | 0.3748 | 532 | 0.2840 | 306 | 0.1634 | 22.0000 | 416.3604 |
| mainline final SimilarCase-Aug | 1873 | 0.2093 | 0.3422 | 0.4026 | 0.4634 | 0.5174 | 0.6556 | 645 | 0.3444 | 474 | 0.2531 | 362 | 0.1933 | 16.0000 | 3200.5414 |

## Delta
| top1_delta | top3_delta | top5_delta | rank_le_50_delta | rank_gt_50_count_delta | top50_late_count_delta |
| --- | --- | --- | --- | --- | --- |
| 0.0315 | 0.0539 | 0.0614 | 0.0304 | -57 | -58 |

## Rank Transition
| baseline_gt50_to_final_le50 | baseline_le50_to_final_gt50 | baseline_gt50_to_final_le5 | baseline_6_50_to_final_le5 | baseline_gt5_to_final_le5 | baseline_le5_to_final_gt5 | baseline_not_top1_to_final_top1 | baseline_top1_to_final_not_top1 | improved_rank | worsened_rank | unchanged_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 104 | 47 | 0 | 168 | 168 | 53 | 128 | 69 | 549 | 992 | 332 |

## 文档 frozen config vs 当前 mainline 输出
| source | topk | weight | score_type | top1 | top3 | top5 | rank_le_50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| docx / reports/mimic_next frozen config | 10 | 0.4000 | raw_similarity | 0.2093 | 0.3300 | 0.3940 | 0.6498 |
| current outputs/mainline_full_pipeline | 20 | 0.5000 | raw_similarity | 0.2093 | 0.3422 | 0.4026 | 0.6556 |

## Overlap Bucket Delta
| bucket_type | bucket | num_cases | baseline_top5 | final_top5 | top5_delta | baseline_rank_le_50 | final_rank_le_50 | rank_le_50_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_hpo_overlap_count_bucket | 2-3 | 440 | 0.4795 | 0.5386 | 0.0591 | 0.7500 | 0.7682 | 0.0182 |
| exact_hpo_overlap_count_bucket | 1 | 618 | 0.3673 | 0.4288 | 0.0615 | 0.6521 | 0.6909 | 0.0388 |
| exact_hpo_overlap_count_bucket | 0 | 687 | 0.1951 | 0.2693 | 0.0742 | 0.4891 | 0.5328 | 0.0437 |
| exact_hpo_overlap_count_bucket | >3 | 128 | 0.5234 | 0.5234 | 0.0000 | 0.7969 | 0.7578 | -0.0391 |
| case_hpo_count_bucket | >10 | 782 | 0.3632 | 0.4079 | 0.0448 | 0.6624 | 0.6777 | 0.0153 |
| case_hpo_count_bucket | 7-10 | 615 | 0.3545 | 0.4065 | 0.0520 | 0.6211 | 0.6602 | 0.0390 |
| case_hpo_count_bucket | 4-6 | 372 | 0.3118 | 0.4167 | 0.1048 | 0.6075 | 0.6425 | 0.0349 |
| case_hpo_count_bucket | 1-3 | 104 | 0.2019 | 0.2885 | 0.0865 | 0.4327 | 0.5096 | 0.0769 |
| gold_in_top50_bucket | yes | 1171 | 0.5457 | 0.6439 | 0.0982 | 1.0000 | 0.9599 | -0.0401 |
| gold_in_top50_bucket | no | 702 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1481 | 0.1481 |

## 机制判断
- 当前 mainline final 的 `rank<=50` 从 0.6252 提升到 0.6556，净提升 0.0304。
- baseline rank>50 被拉回 final rank<=50 的样本为 104；但 baseline rank>50 直接进入 final top5 的样本为 0。
- baseline rank 6-50 被推入 final top5 的样本为 168，说明 Top5 提升主要来自 top50 内局部重排，而不是真正解决大规模候选召回。
- 当前仍有 final rank>50 的病例 645 个，占比 0.3444；这部分仍不是 reranker 或 hard negative 单独能解决的问题。
- `stage6` recovered case 文件行数为 104，near-miss 文件行数为 162。
- 因此，SimilarCase-Aug 已经是有效的 no-train 多视图/病例库证据增强模块；后续第一优先级仍应沿着 candidate expansion + evidence rerank 做，而不是直接转图对比学习。

## 复现命令
- 已有 checkpoint 评估链路：`D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode eval_only`
- 完整训练链路：`D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode full`
- 本复核脚本：`D:\python\python.exe tools\analysis\mimic_mainline_final_reanalysis.py`

## Checkpoint 一致性
- run_manifest finetune_checkpoint: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt`
- test candidate checkpoint: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt`
