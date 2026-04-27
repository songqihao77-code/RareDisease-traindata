# DDD Ontology Hard Negative Fixed Test Results

该模型是 DDD-specific enhancement，只对 DDD 应用；MIMIC / MME / RAMEDIS / MyGene2 保持 current mainline，不混写为 ALL general model。

## DDD Current vs Hard-Negative Model vs DeepRare Target
| dataset | model_scope  | num_cases | current_top1 | new_top1 | target_top1 | delta_top1 | current_top3 | new_top3 | target_top3 | delta_top3 | current_top5 | new_top5 | target_top5 | delta_top5 | current_rank_le_50 | new_rank_le_50 | delta_rank_le_50 | case_gap_before_top1 | case_gap_after_top1 | case_gap_before_top3 | case_gap_after_top3 | case_gap_before_top5 | case_gap_after_top5 | deeprare_ddd_target_reached |
| ------- | ------------ | --------- | ------------ | -------- | ----------- | ---------- | ------------ | -------- | ----------- | ---------- | ------------ | -------- | ----------- | ---------- | ------------------ | -------------- | ---------------- | -------------------- | ------------------- | -------------------- | ------------------- | -------------------- | ------------------- | --------------------------- |
| DDD     | DDD-specific | 761       | 0.3758       | 0.3653   | 0.4800      | -0.0105    | 0.4967       | 0.4809   | 0.6000      | -0.0158    | 0.5401       | 0.5204   | 0.6300      | -0.0197    | 0.7438             | 0.7438         | 0.0000           | 80                   | 88                  | 79                   | 91                  | 69                   | 84                  | 0                           |

## 结论
- Top1: 0.3758 -> 0.3653, target 0.48, delta -0.0105.
- Top3: 0.4967 -> 0.4809, target 0.60, delta -0.0158.
- Top5: 0.5401 -> 0.5204, target 0.63, delta -0.0197.
- Rank<=50: 0.7438 -> 0.7438.
- 是否达到 DeepRare DDD target: 否。
