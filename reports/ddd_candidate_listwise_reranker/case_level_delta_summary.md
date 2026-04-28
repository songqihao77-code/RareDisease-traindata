# DDD Listwise Case-Level Delta Summary

| dataset | num_cases | top1_gained_cases | top1_lost_cases | top3_gained_cases | top3_lost_cases | top5_gained_cases | top5_lost_cases | rank_improved | rank_worsened | rank_unchanged | current_rank_2_5_to_rank1 | current_rank_4_5_to_top3 | current_rank_6_50_to_top5 | current_rank_gt50_to_top50 | harmed_mean_current_rank |
| ------- | --------- | ----------------- | --------------- | ----------------- | --------------- | ----------------- | --------------- | ------------- | ------------- | -------------- | ------------------------- | ------------------------ | ------------------------- | -------------------------- | ------------------------ |
| DDD     | 761       | 10                | 17              | 4                 | 17              | 6                 | 19              | 80            | 295           | 386            | 10                        | 2                        | 6                         | 22                         | 33.5492                  |

## Expansion source 对改善病例的贡献
| expansion_source                 | improved_cases_with_source |
| -------------------------------- | -------------------------- |
| candidate_source_mondo_expansion | 80                         |
| candidate_source_hpo_expansion   | 80                         |
| candidate_source_similar_case    | 79                         |

## Harmed cases 共同特征
- harmed cases count: 295
- harmed mean current rank: 33.55
