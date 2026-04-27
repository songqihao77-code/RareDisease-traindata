# HGNN + Validation-selected DDD Grid Rerank + Mimic Mainline

## 1. Protocol
- no training
- no HGNN encoder change
- no test-side tuning
- no exact evaluation overwrite
- DDD uses fixed validation-selected grid weights (`DDD_top1`)
- mimic_test uses existing `HGNN_SimilarCase_Aug` fixed output
- all other datasets use HGNN baseline ranks

## 2. Inputs
- config: `D:\RareDisease-traindata\configs\mainline_hgnn_val_grid_rerank.yaml`
- final case ranks: `D:\RareDisease-traindata\outputs\mainline_hgnn_val_grid_rerank_all\mainline_final_case_ranks.csv`
- manifest: `D:\RareDisease-traindata\outputs\mainline_hgnn_val_grid_rerank_all\mainline_run_manifest.json`

## 3. HGNN Baseline Metrics
| method | dataset_name | num_cases | top1 | top3 | top5 | median_rank | mean_rank | rank_le_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HGNN_exact_baseline | DDD | 761 | 0.31011826544021026 | 0.4296977660972405 | 0.47174770039421815 | 7.0 | 18.88699080157687 | 0.7529566360052562 |
| HGNN_exact_baseline | HMS | 25 | 0.4 | 0.44 | 0.6 | 4.0 | 14.4 | 0.84 |
| HGNN_exact_baseline | LIRICAL | 59 | 0.4576271186440678 | 0.5254237288135594 | 0.5254237288135594 | 2.0 | 14.067796610169491 | 0.847457627118644 |
| HGNN_exact_baseline | MME | 10 | 0.8 | 0.8 | 0.8 | 1.0 | 3.8 | 1.0 |
| HGNN_exact_baseline | MyGene2 | 33 | 0.8484848484848485 | 0.8787878787878788 | 0.9090909090909091 | 1.0 | 5.666666666666667 | 0.9090909090909091 |
| HGNN_exact_baseline | RAMEDIS | 217 | 0.7695852534562212 | 0.8387096774193549 | 0.8847926267281107 | 1.0 | 4.28110599078341 | 0.9585253456221198 |
| HGNN_exact_baseline | mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.1575013347570742 | 0.24345969033635878 | 0.2984516817939135 | 32.0 | 28.83876134543513 | 0.5686065136145222 |
| HGNN_exact_baseline | ALL | 2978 | 0.25889858965748824 | 0.35057085292142376 | 0.40094022834116855 | 14.0 | 23.751511081262592 | 0.6571524513096038 |

## 4. Final Mainline Metrics
| method | dataset_name | num_cases | top1 | top3 | top5 | median_rank | mean_rank | rank_le_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mainline_hgnn_val_grid_rerank | DDD | 761 | 0.3745072273324573 | 0.48620236530880423 | 0.5269382391590013 | 4.0 | 17.0946123521682 | 0.7529566360052562 |
| mainline_hgnn_val_grid_rerank | HMS | 25 | 0.4 | 0.44 | 0.6 | 4.0 | 14.4 | 0.84 |
| mainline_hgnn_val_grid_rerank | LIRICAL | 59 | 0.4576271186440678 | 0.5254237288135594 | 0.5254237288135594 | 2.0 | 14.067796610169491 | 0.847457627118644 |
| mainline_hgnn_val_grid_rerank | MME | 10 | 0.8 | 0.8 | 0.8 | 1.0 | 3.8 | 1.0 |
| mainline_hgnn_val_grid_rerank | MyGene2 | 33 | 0.8484848484848485 | 0.8787878787878788 | 0.9090909090909091 | 1.0 | 5.666666666666667 | 0.9090909090909091 |
| mainline_hgnn_val_grid_rerank | RAMEDIS | 217 | 0.7695852534562212 | 0.8387096774193549 | 0.8847926267281107 | 1.0 | 4.28110599078341 | 0.9585253456221198 |
| mainline_hgnn_val_grid_rerank | mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.1575013347570742 | 0.24345969033635878 | 0.2984516817939135 | 32.0 | 28.83876134543513 | 0.5686065136145222 |
| mainline_hgnn_val_grid_rerank | ALL | 2978 | 0.27535258562793824 | 0.36501007387508394 | 0.4150436534586971 | 12.0 | 23.293485560779047 | 0.6571524513096038 |

## 5. Delta vs Historical Baseline
| dataset_name | num_cases | baseline_top1 | baseline_top3 | baseline_top5 | baseline_median_rank | baseline_mean_rank | baseline_rank_le_50 | final_top1 | final_top3 | final_top5 | final_median_rank | final_mean_rank | final_rank_le_50 | delta_top1 | delta_top3 | delta_top5 | delta_rank_le_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 2978 | 0.25889858965748824 | 0.35057085292142376 | 0.40094022834116855 | 14.0 | 23.751511081262592 | 0.6571524513096038 | 0.27535258562793824 | 0.36501007387508394 | 0.4150436534586971 | 12.0 | 23.293485560779047 | 0.6571524513096038 | 0.016453995970449997 | 0.014439220953660181 | 0.014103425117528545 | 0.0 |
| DDD | 761 | 0.31011826544021026 | 0.4296977660972405 | 0.47174770039421815 | 7.0 | 18.88699080157687 | 0.7529566360052562 | 0.3745072273324573 | 0.48620236530880423 | 0.5269382391590013 | 4.0 | 17.0946123521682 | 0.7529566360052562 | 0.06438896189224702 | 0.056504599211563755 | 0.05519053876478314 | 0.0 |
| HMS | 25 | 0.4 | 0.44 | 0.6 | 4.0 | 14.4 | 0.84 | 0.4 | 0.44 | 0.6 | 4.0 | 14.4 | 0.84 | 0.0 | 0.0 | 0.0 | 0.0 |
| LIRICAL | 59 | 0.4576271186440678 | 0.5254237288135594 | 0.5254237288135594 | 2.0 | 14.067796610169491 | 0.847457627118644 | 0.4576271186440678 | 0.5254237288135594 | 0.5254237288135594 | 2.0 | 14.067796610169491 | 0.847457627118644 | 0.0 | 0.0 | 0.0 | 0.0 |
| MME | 10 | 0.8 | 0.8 | 0.8 | 1.0 | 3.8 | 1.0 | 0.8 | 0.8 | 0.8 | 1.0 | 3.8 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MyGene2 | 33 | 0.8484848484848485 | 0.8787878787878788 | 0.9090909090909091 | 1.0 | 5.666666666666667 | 0.9090909090909091 | 0.8484848484848485 | 0.8787878787878788 | 0.9090909090909091 | 1.0 | 5.666666666666667 | 0.9090909090909091 | 0.0 | 0.0 | 0.0 | 0.0 |
| RAMEDIS | 217 | 0.7695852534562212 | 0.8387096774193549 | 0.8847926267281107 | 1.0 | 4.28110599078341 | 0.9585253456221198 | 0.7695852534562212 | 0.8387096774193549 | 0.8847926267281107 | 1.0 | 4.28110599078341 | 0.9585253456221198 | 0.0 | 0.0 | 0.0 | 0.0 |
| mimic_test_recleaned_mondo_hpo_rows | 1873 | 0.1575013347570742 | 0.24345969033635878 | 0.2984516817939135 | 32.0 | 28.83876134543513 | 0.5686065136145222 | 0.1575013347570742 | 0.24345969033635878 | 0.2984516817939135 | 32.0 | 28.83876134543513 | 0.5686065136145222 | 0.0 | 0.0 | 0.0 | 0.0 |

## 6. Mainline Judgment
- DDD final mainline is `validation-selected grid rerank (DDD_top1)`.
- mimic mainline is enabled only for `mimic_test`.
- gated rerank, ontology-aware HN, and test-side exploratory grid/gate are not used.