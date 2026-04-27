# DDD Bootstrap 95% CI

| dataset | metric     | mean    | ci95_low | ci95_high | num_cases | stable_above_current |
| ------- | ---------- | ------- | -------- | --------- | --------- | -------------------- |
| DDD     | top1       | 0.3665  | 0.3338   | 0.4015    | 761       | 0                    |
| DDD     | delta_top1 | -0.0107 | -0.0263  | 0.0039    | 761       | 0                    |
| DDD     | top3       | 0.4820  | 0.4468   | 0.5158    | 761       | 0                    |
| DDD     | delta_top3 | -0.0155 | -0.0289  | -0.0026   | 761       | 0                    |
| DDD     | top5       | 0.5199  | 0.4809   | 0.5539    | 761       | 0                    |
| DDD     | delta_top5 | -0.0197 | -0.0349  | -0.0053   | 761       | 0                    |

## 稳定性判断
- fixed test Top1/Top3/Top5: 0.3653/0.4809/0.5204.
- DeepRare target: 0.48/0.60/0.63.
- delta 是否稳定超过 current: False.
