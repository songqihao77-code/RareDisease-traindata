# mimic_test Evaluation Path Audit

## 当前评估链路
- evaluation script: `python -m src.evaluation.evaluator`
- data config: `D:\RareDisease-traindata\configs\data_llldataset_eval.yaml`
- train config: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\configs\stage3_exact_eval_train.yaml`
- checkpoint: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt`
- checkpoint_epoch: `10`
- candidate/top50 file: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage4_candidates\top50_candidates_test.csv`
- exact output dir: `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage3_exact_eval`
- final mainline output: `{'metrics': 'D:\\RareDisease-traindata\\outputs\\mainline_full_pipeline\\mainline_final_metrics.csv', 'metrics_with_sources': 'D:\\RareDisease-traindata\\outputs\\mainline_full_pipeline\\mainline_final_metrics_with_sources.csv', 'case_ranks': 'D:\\RareDisease-traindata\\outputs\\mainline_full_pipeline\\mainline_final_case_ranks.csv'}`

## 输入与字段
- mimic_test input: `D:\RareDisease-traindata\LLLdataset\dataset\processed\test\mimic_test_recleaned_mondo_hpo_rows.csv`
- label 字段: `mondo_label`
- HPO 字段: `hpo_id`
- case_id 字段: `case_id`
- num_cases: 1873
- MONDO ID 不在 Disease_index 的 current exact gold cases: 0
- multi-label cases: 227

## 静态资源
- Disease_index: `D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx`
- HPO_index: `D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\HPO_index_v4.xlsx`
- disease incidence / hyperedge matrix: `D:\RareDisease-traindata\LLLdataset\DiseaseHy\rare_disease_hgnn_clean_package_v59\v59DiseaseHy.npz`
- disease hyperedge CSV: `D:\RareDisease-traindata\LLLdataset\DiseaseHy\rare_disease_hgnn_clean_package_v59\v59_hyperedge_weighted_patched.csv`
- HPO ontology used by candidate evidence export: `D:\RareDisease-traindata\raw_data\hp.json`

## 当前 exact metric 计算方式
- `src.evaluation.evaluator.load_test_cases` 先按 namespaced `case_id` 聚合，`mondo_label` 使用 `group_df[label_col].iloc[0]`，HPO 使用该 case 的去重 HPO 列表。
- `evaluate` 对每个 case 计算全 disease pool 分数，`torch.argsort(scores, descending=True)` 后定位 gold disease index，得到 1-indexed `true_rank`。
- `top1/top3/top5/rank_le_50` 由 `true_rank <= k` 计算，是 strict exact MONDO ID hit，不接受 synonym、ancestor、descendant 或任一 secondary label。
- 因此存在多标签被单标签化的问题；本次报告不改变主 exact metric，只做 supplementary audit。

## mimic_test exact per-dataset row
| dataset_name | source_file | num_cases | num_evaluable | num_skipped | num_skipped_missing_label | num_skipped_no_valid_hpo | top1 | top3 | top5 | mean_rank | median_rank | rank_le_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_test_recleaned_mondo_hpo_rows | mimic_test_recleaned_mondo_hpo_rows.csv | 1873 | 1873 | 0 | 0 | 0 | 0.1778 | 0.2883 | 0.3412 | 416.3604 | 22.0000 | 0.6252 |
