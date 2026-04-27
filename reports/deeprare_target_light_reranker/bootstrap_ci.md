# Bootstrap 95% CI

CI 基于 selected light reranker 的 fixed test case-level ranks。HMS、LIRICAL、MME、MyGene2 因样本量小需按 high_variance 解释。

| dataset                             | metric     | mean    | ci95_low | ci95_high | num_cases | high_variance |
| ----------------------------------- | ---------- | ------- | -------- | --------- | --------- | ------------- |
| DDD                                 | top1       | 0.3594  | 0.3265   | 0.3942    | 761       | 0             |
| DDD                                 | delta_top1 | -0.0168 | -0.0355  | 0.0000    | 761       | 0             |
| DDD                                 | top3       | 0.4813  | 0.4513   | 0.5145    | 761       | 0             |
| DDD                                 | delta_top3 | -0.0161 | -0.0289  | -0.0026   | 761       | 0             |
| DDD                                 | top5       | 0.5205  | 0.4895   | 0.5552    | 761       | 0             |
| DDD                                 | delta_top5 | -0.0199 | -0.0355  | -0.0053   | 761       | 0             |
| HMS                                 | top1       | 0.3301  | 0.1390   | 0.5200    | 25        | 1             |
| HMS                                 | delta_top1 | 0.0000  | 0.0000   | 0.0000    | 25        | 1             |
| HMS                                 | top3       | 0.4084  | 0.2400   | 0.6000    | 25        | 1             |
| HMS                                 | delta_top3 | -0.0382 | -0.1200  | 0.0000    | 25        | 1             |
| HMS                                 | top5       | 0.4859  | 0.2800   | 0.6800    | 25        | 1             |
| HMS                                 | delta_top5 | 0.0000  | 0.0000   | 0.0000    | 25        | 1             |
| LIRICAL                             | top1       | 0.5083  | 0.3809   | 0.6271    | 59        | 1             |
| LIRICAL                             | delta_top1 | 0.0000  | 0.0000   | 0.0000    | 59        | 1             |
| LIRICAL                             | top3       | 0.6447  | 0.5254   | 0.7458    | 59        | 1             |
| LIRICAL                             | delta_top3 | 0.0000  | 0.0000   | 0.0000    | 59        | 1             |
| LIRICAL                             | top5       | 0.6800  | 0.5593   | 0.7966    | 59        | 1             |
| LIRICAL                             | delta_top5 | 0.0000  | 0.0000   | 0.0000    | 59        | 1             |
| MME                                 | top1       | 0.8990  | 0.7000   | 1.0000    | 10        | 1             |
| MME                                 | delta_top1 | 0.0000  | 0.0000   | 0.0000    | 10        | 1             |
| MME                                 | top3       | 0.8990  | 0.7000   | 1.0000    | 10        | 1             |
| MME                                 | delta_top3 | 0.0000  | 0.0000   | 0.0000    | 10        | 1             |
| MME                                 | top5       | 0.8990  | 0.7000   | 1.0000    | 10        | 1             |
| MME                                 | delta_top5 | 0.0000  | 0.0000   | 0.0000    | 10        | 1             |
| MyGene2                             | top1       | 0.8788  | 0.7576   | 0.9697    | 33        | 1             |
| MyGene2                             | delta_top1 | 0.0000  | 0.0000   | 0.0000    | 33        | 1             |
| MyGene2                             | top3       | 0.8788  | 0.7576   | 0.9697    | 33        | 1             |
| MyGene2                             | delta_top3 | 0.0000  | 0.0000   | 0.0000    | 33        | 1             |
| MyGene2                             | top5       | 0.8788  | 0.7576   | 0.9697    | 33        | 1             |
| MyGene2                             | delta_top5 | 0.0000  | 0.0000   | 0.0000    | 33        | 1             |
| RAMEDIS                             | top1       | 0.7889  | 0.7327   | 0.8387    | 217       | 0             |
| RAMEDIS                             | delta_top1 | 0.0000  | 0.0000   | 0.0000    | 217       | 0             |
| RAMEDIS                             | top3       | 0.8762  | 0.8295   | 0.9171    | 217       | 0             |
| RAMEDIS                             | delta_top3 | 0.0000  | 0.0000   | 0.0000    | 217       | 0             |
| RAMEDIS                             | top5       | 0.9307  | 0.8940   | 0.9585    | 217       | 0             |
| RAMEDIS                             | delta_top5 | 0.0000  | 0.0000   | 0.0000    | 217       | 0             |
| mimic_test_recleaned_mondo_hpo_rows | top1       | 0.2138  | 0.1951   | 0.2333    | 1873      | 0             |
| mimic_test_recleaned_mondo_hpo_rows | delta_top1 | 0.0050  | 0.0011   | 0.0101    | 1873      | 0             |
| mimic_test_recleaned_mondo_hpo_rows | top3       | 0.3416  | 0.3179   | 0.3628    | 1873      | 0             |
| mimic_test_recleaned_mondo_hpo_rows | delta_top3 | -0.0002 | -0.0043  | 0.0040    | 1873      | 0             |
| mimic_test_recleaned_mondo_hpo_rows | top5       | 0.3986  | 0.3764   | 0.4205    | 1873      | 0             |
| mimic_test_recleaned_mondo_hpo_rows | delta_top5 | -0.0034 | -0.0083  | 0.0016    | 1873      | 0             |

## 稳定性判断
- mimic_test_recleaned_mondo_hpo_rows: 95% CI 下界大于 0 的 delta 指标：delta_top1。
- DDD: 95% CI 下界大于 0 的 delta 指标：none。
