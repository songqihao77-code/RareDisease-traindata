# DDD Case-Level Delta Summary

| dataset | num_cases | top1_gained_cases | top1_lost_cases | top3_gained_cases | top3_lost_cases | top5_gained_cases | top5_lost_cases | rank_improved | rank_worsened | rank_unchanged | final_rank_2_5_to_rank1 | final_rank_4_5_to_top3 | final_rank_6_50_to_top5 | rank_gt50_recalled_to_top50 |
| ------- | --------- | ----------------- | --------------- | ----------------- | --------------- | ----------------- | --------------- | ------------- | ------------- | -------------- | ----------------------- | ---------------------- | ----------------------- | --------------------------- |
| DDD     | 761       | 14                | 22              | 8                 | 20              | 8                 | 23              | 62            | 336           | 363            | 13                      | 5                      | 8                       | 0                           |

## Hard Negative 类型对改进的贡献
这里统计的是 improved cases 的 test top50 中出现过哪些 hard-negative-like 类型；不是因果归因。
| negative_type       | improved_case_count_with_type_present |
| ------------------- | ------------------------------------- |
| high_hpo_overlap    | 62                                    |
| hyperedge_similar   | 62                                    |
| random              | 62                                    |
| same_parent_sibling | 58                                    |
| similar_case_false  | 62                                    |
| top50_above_gold    | 52                                    |
