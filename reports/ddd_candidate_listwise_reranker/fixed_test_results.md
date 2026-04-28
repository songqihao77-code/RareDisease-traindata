# DDD Candidate Listwise Fixed Test Results

| dataset | num_cases | current_top1 | failed_pairwise_top1 | listwise_top1 | target_top1 | delta_top1_vs_current | current_top3 | failed_pairwise_top3 | listwise_top3 | target_top3 | delta_top3_vs_current | current_top5 | failed_pairwise_top5 | listwise_top5 | target_top5 | delta_top5_vs_current | current_rank_le_50 | failed_pairwise_rank_le_50 | listwise_rank_le_50 | delta_rank_le_50_vs_current | case_gap_before_top1 | case_gap_after_top1 | case_gap_before_top3 | case_gap_after_top3 | case_gap_before_top5 | case_gap_after_top5 | deeprare_ddd_target_reached |
| ------- | --------- | ------------ | -------------------- | ------------- | ----------- | --------------------- | ------------ | -------------------- | ------------- | ----------- | --------------------- | ------------ | -------------------- | ------------- | ----------- | --------------------- | ------------------ | -------------------------- | ------------------- | --------------------------- | -------------------- | ------------------- | -------------------- | ------------------- | -------------------- | ------------------- | --------------------------- |
| DDD     | 761       | 0.3758       | 0.3653               | 0.3666        | 0.4800      | -0.0092               | 0.4967       | 0.4809               | 0.4796        | 0.6000      | -0.0171               | 0.5401       | 0.5204               | 0.5230        | 0.6300      | -0.0171               | 0.7438             | 0.7438                     | 0.7727              | 0.0289                      | 80                   | 87                  | 79                   | 92                  | 69                   | 82                  | 0                           |

## 误伤统计
- current top1 -> new not top1: 17
- current top3 -> new not top3: 17
- current top5 -> new not top5: 19

## 结论
- 是否达到 DeepRare DDD target: 否。
