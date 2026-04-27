# Current mimic mainline audit

## 当前主线结果
| dataset | cases | top1 | top3 | top5 | rank_le_50 |
| --- | --- | --- | --- | --- | --- |
| mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.2093 | 0.3422 | 0.4026 | 0.6556 |

## 来源
| dataset | cases | top1 | top3 | top5 | rank_le_50 | module_applied | source_result_path | source_dataset_name | checkpoint_path | data_config_path | train_config_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.2093 | 0.3422 | 0.4026 | 0.6556 | similar_case_fixed_test | D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage6_mimic_similar_case\similar_case_fixed_test.csv | mimic_test_recleaned_mondo_hpo_rows | D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline\configs\stage2_finetune.yaml |

## docx frozen config vs current mainline
| config | topk | weight | score_type | top1 | top3 | top5 | rank_le_50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| docx frozen config | 10 | 0.4000 | raw_similarity | 0.2093 | 0.3300 | 0.3940 | 0.6498 |
| current mainline | 20 | 0.5000 | raw_similarity | 0.2093 | 0.3422 | 0.4026 | 0.6556 |

## validation selection check
- current mainline selected config: topk=20, weight=0.5, score_type=raw_similarity
- validation best config by script key `(top5, rank_le_50, top1, -mean_rank)`: topk=20, weight=0.5, score_type=raw_similarity
- current mainline topk=20, weight=0.5 是否来自 validation-selected fixed test：是

## 结论
- 如果采用当前 `outputs/mainline_full_pipeline` 作为正式主线，则主表建议采用 current mainline：Top1=0.2093, Top3=0.3422, Top5=0.4026, Rank<=50=0.6556。
- docx frozen config Top5=0.3940 是较早 frozen 配置；它与当前输出配置不同，不应和 current mainline 混写为同一个实验。
- 当前检查可确认 topk=20, weight=0.5 是现有 `similar_case_val_selection.csv` 按脚本选择规则得到的 validation-selected fixed-test 配置。

## 可复现命令与路径
- full: `D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode full`
- eval_only: `D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode eval_only`
- final metrics: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\mainline_final_metrics.csv`
- final case ranks: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\mainline_final_case_ranks.csv`
- SimilarCase fixed test: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage6_mimic_similar_case\similar_case_fixed_test.csv`
- run_manifest checkpoint: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt`
