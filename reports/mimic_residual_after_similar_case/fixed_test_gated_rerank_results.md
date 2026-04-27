# Fixed test gated rerank results

- 该结果只因为 validation top5 提升才执行 fixed test 一次。
- 与 current mainline `similar_case_fixed_test` 对比：Top1 0.2093 -> 0.2168（+0.0075），Top3 0.3422 -> 0.3460（+0.0038），Top5 0.4026 -> 0.4036（+0.0010），Rank<=50 0.6556 -> 0.6530（-0.0026）。
- 结论：gated rerank 是可进入对比表的 validation-selected fixed-test 候选，但 Top5 增益很小且 Rank<=50 有轻微下降，不建议未经更多验证就直接替换 current mainline。
| sim_weight | ic_weight | agree_boost | protect_bonus | val_num_cases | val_top1 | val_top3 | val_top5 | val_rank_le_50 | val_median_rank | val_mean_rank | baseline_current_val_top5 | top5_delta_vs_current | top1_delta_vs_current | test_num_cases | test_top1 | test_top3 | test_top5 | test_rank_le_50 | test_median_rank | test_mean_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.1000 | 0.1000 | 0.0500 | 2146 | 0.6393 | 0.7144 | 0.7437 | 0.9021 | 1.0000 | 905.2432 | 0.7363 | 0.0075 | 0.0028 | 1873 | 0.2168 | 0.3460 | 0.4036 | 0.6530 | 16.0000 | 3200.6983 |
