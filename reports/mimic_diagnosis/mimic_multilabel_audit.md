# mimic_test Multi-label Audit

## Case Label Counts
- single-label case 数量：1646
- multi-label case 数量：227
- multi-label case 占比：0.1212

| label_count | case_count | ratio |
| --- | --- | --- |
| 1 | 1646 | 0.8788 |
| 2 | 208 | 0.1111 |
| 3 | 19 | 0.0101 |

## Metrics
| scope | mode | num_cases | top1 | top3 | top5 | top50 |
| --- | --- | --- | --- | --- | --- | --- |
| all_cases | any_label | 1873 | 0.1954 | 0.3102 | 0.3641 | 0.6631 |
| all_cases | original_exact | 1873 | 0.1778 | 0.2883 | 0.3412 | 0.6252 |
| multi_label_subset | any_label | 227 | 0.2247 | 0.3744 | 0.4449 | 0.8106 |
| multi_label_subset | original_exact | 227 | 0.0793 | 0.1938 | 0.2555 | 0.4978 |
| single_label_subset | original_exact | 1646 | 0.1914 | 0.3013 | 0.3530 | 0.6428 |

## 结论
- 当前 evaluation 在 `load_test_cases` 中按 `case_id` 聚合后使用 `group_df[label_col].iloc[0]`，因此多标签病例会被单标签化。
- any-label top1 相比 original exact top1 提升 0.0176；any-label top5 提升 0.0230。
- 这说明当前 mimic_test 准确率存在一定低估，但低估幅度不足以解释全部低分。
- any-label evaluation 不应作为论文主表；建议标注为 supplementary / error analysis，用于说明多标签病例的潜在假阴性。
