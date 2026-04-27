# Hard Negative Summary

## Overview
| split      | num_cases | hard_negative_rows | avg_hard_negatives_per_case | gold_in_top50_case_coverage | rank_2_5_cases | rank_6_20_cases | rank_21_50_cases | rank_gt_50_cases |
| ---------- | --------- | ------------------ | --------------------------- | --------------------------- | -------------- | --------------- | ---------------- | ---------------- |
| train      | 1358      | 31003              | 22.8299                     | 0.9698                      | 264            | 108             | 34               | 41               |
| validation | 164       | 2834               | 17.2805                     | 0.7134                      | 26             | 18              | 7                | 47               |

## Negative Type Counts
| split      | negative_type       | count |
| ---------- | ------------------- | ----- |
| train      | high_hpo_overlap    | 6585  |
| train      | hyperedge_similar   | 6585  |
| train      | random              | 6585  |
| train      | same_parent_sibling | 4680  |
| train      | similar_case_false  | 5410  |
| train      | top50_above_gold    | 1158  |
| validation | high_hpo_overlap    | 585   |
| validation | hyperedge_similar   | 585   |
| validation | random              | 585   |
| validation | same_parent_sibling | 406   |
| validation | similar_case_false  | 493   |
| validation | top50_above_gold    | 180   |
