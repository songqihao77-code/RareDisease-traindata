# DDD Validation-selected Grid Rerank Report

- protocol: validation candidates select grid weights, test candidates fixed evaluation once.
- DDD final mainline uses `validation_grid_DDD_top1`.
- gated rerank and test-side exploratory search are not mainline methods.
- validation candidates: `D:\RareDisease-traindata\outputs\rerank\top50_candidates_validation.csv`
- test candidates: `D:\RareDisease-traindata\outputs\rerank\top50_candidates_v2.csv`

## Fixed Test Metrics
|Protocol|Selection Objective|DDD Top1|DDD Top3|DDD Top5|Recall@50|Top1 Delta|Top3 Delta|Top5 Delta|Paper Usability|
|---|---|---|---|---|---|---|---|---|---|
|HGNN baseline||0.3101|0.4297|0.4717|0.7530|0.0000|0.0000|0.0000|主表 baseline|
|validation_grid_DDD_top1|DDD_top1|0.3745|0.4862|0.5269|0.7530|0.0644|0.0565|0.0552|validation-selected grid fixed test，可作为论文主线候选|
|validation_grid_ALL_top1|ALL_top1|0.3719|0.4849|0.5269|0.7530|0.0618|0.0552|0.0552|validation-selected grid fixed test，可作为论文主线候选|

## Selected Weights
### Grid
- `validation_grid_DDD_top1` objective=`DDD_top1`
  - weights: `{'w_hgnn': 0.7, 'w_ic': 0.2, 'w_exact': 0.1, 'w_semantic': 0.15, 'w_case_cov': 0.05, 'w_dis_cov': 0.0, 'w_size': 0.01}`
  - validation metrics: `{'DDD_top1': 0.4634146341463415, 'DDD_top3': 0.5609756097560976, 'DDD_top5': 0.6097560975609756, 'DDD_rank_le_50': 0.75, 'mimic_test_top1': nan, 'mimic_test_top3': nan, 'mimic_test_top5': nan, 'mimic_test_rank_le_50': nan, 'HMS_top1': 0.3333333333333333, 'HMS_top3': 0.6666666666666666, 'HMS_top5': 0.6666666666666666, 'HMS_rank_le_50': 1.0, 'LIRICAL_top1': 0.2424242424242424, 'LIRICAL_top3': 0.3939393939393939, 'LIRICAL_top5': 0.4242424242424242, 'LIRICAL_rank_le_50': 0.6363636363636364, 'ALL_top1': 0.5200372786579683, 'ALL_top3': 0.6057781919850885, 'ALL_top5': 0.6351351351351351, 'ALL_rank_le_50': 0.7931034482758621}`
- `validation_grid_ALL_top1` objective=`ALL_top1`
  - weights: `{'w_hgnn': 0.8, 'w_ic': 0.2, 'w_exact': 0.1, 'w_semantic': 0.15, 'w_case_cov': 0.05, 'w_dis_cov': 0.05, 'w_size': 0.02}`
  - validation metrics: `{'DDD_top1': 0.4634146341463415, 'DDD_top3': 0.5548780487804879, 'DDD_top5': 0.6036585365853658, 'DDD_rank_le_50': 0.75, 'mimic_test_top1': nan, 'mimic_test_top3': nan, 'mimic_test_top5': nan, 'mimic_test_rank_le_50': nan, 'HMS_top1': 0.3333333333333333, 'HMS_top3': 0.6666666666666666, 'HMS_top5': 0.6666666666666666, 'HMS_rank_le_50': 1.0, 'LIRICAL_top1': 0.1818181818181818, 'LIRICAL_top3': 0.3939393939393939, 'LIRICAL_top5': 0.4242424242424242, 'LIRICAL_rank_le_50': 0.6363636363636364, 'ALL_top1': 0.5232991612301957, 'ALL_top3': 0.6067101584342963, 'ALL_top5': 0.6369990680335508, 'ALL_rank_le_50': 0.7931034482758621}`

## Paper Boundary
- HGNN exact baseline can enter the main table.
- validation-selected grid fixed-test rerank is the final DDD mainline.
- gated rerank, HN dry-run, and test-side grid/gate are not mainline results.