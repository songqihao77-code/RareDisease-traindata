# DDD Candidate Listwise Bootstrap 95% CI

| dataset | metric                        | mean    | ci95_low | ci95_high | stable_positive |
| ------- | ----------------------------- | ------- | -------- | --------- | --------------- |
| DDD     | top1                          | 0.3677  | 0.3351   | 0.4021    | 0               |
| DDD     | delta_top1_vs_current         | -0.0095 | -0.0237  | 0.0026    | 0               |
| DDD     | delta_top1_vs_failed_pairwise | 0.0012  | -0.0066  | 0.0092    | 0               |
| DDD     | top3                          | 0.4802  | 0.4455   | 0.5151    | 0               |
| DDD     | delta_top3_vs_current         | -0.0173 | -0.0289  | -0.0066   | 0               |
| DDD     | delta_top3_vs_failed_pairwise | -0.0017 | -0.0079  | 0.0053    | 0               |
| DDD     | top5                          | 0.5226  | 0.4823   | 0.5572    | 0               |
| DDD     | delta_top5_vs_current         | -0.0170 | -0.0302  | -0.0053   | 0               |
| DDD     | delta_top5_vs_failed_pairwise | 0.0027  | -0.0053  | 0.0118    | 0               |

## 结论
- 是否稳定优于 current: False
- 是否稳定优于 failed pairwise: False
- 是否接近或达到 DeepRare target：见 fixed_test_results.md。
