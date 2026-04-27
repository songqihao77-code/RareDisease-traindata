# mimic_test Failure Bucket Summary

## Multi-label Summary
| dataset | subset | num_cases | mean_gold_label_count | median_gold_label_count | exact_top1 | exact_top3 | exact_top5 | exact_top50 | any_label_hit_at_1 | any_label_hit_at_3 | any_label_hit_at_5 | any_label_hit_at_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_test | multi_label | 227 | 2.0837 | 2.0000 | 0.1278 | 0.2643 | 0.3084 | 0.4978 | 0.2687 | 0.4493 | 0.5110 | 0.7841 |
| mimic_test | single_label | 1646 | 1.0000 | 1.0000 | 0.2005 | 0.3044 | 0.3603 | 0.6312 | 0.2005 | 0.3044 | 0.3603 | 0.6312 |
| mimic_test | ALL | 1873 | 1.1313 | 1.0000 | 0.1917 | 0.2995 | 0.3540 | 0.6151 | 0.2088 | 0.3219 | 0.3785 | 0.6498 |

## Rank > 50 Buckets
- rank>50 cases: 721
- HPO generic threshold: mean_case_hpo_ic <= 5.3097
| bucket | num_cases | ratio_among_rank_gt50 |
| --- | --- | --- |
| overlap=0 | 351 | 0.4868 |
| other | 196 | 0.2718 |
| HPO_too_generic | 184 | 0.2552 |
| unseen_disease | 110 | 0.1526 |
| multi-label_misevaluation | 65 | 0.0902 |
| HPO_too_few | 61 | 0.0846 |
| obsolete_disease | 28 | 0.0388 |

## 结论
- `any-label@k` 已单独输出，未覆盖原始 exact metric。
- multi-label 病例的 any-label 命中明显高于 exact，说明 mimic_test 存在 formal exact 低估风险，但只能作为 supplementary。
- rank>50 中 `overlap=0` 和低信息量 HPO 是主要候选召回瓶颈，单纯 top50 rerank 无法解决这部分样本。
