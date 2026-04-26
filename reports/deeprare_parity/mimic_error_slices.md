# mimic_test Error Slices

## Label Subsets

| dataset | subset | num_cases | mean_gold_label_count | median_gold_label_count | exact_top1 | exact_top3 | exact_top5 | exact_rank_le_50 | any_label_hit_at_1 | any_label_hit_at_3 | any_label_hit_at_5 | any_label_hit_at_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_test | multi_label | 227 | 2.0837 | 2.0000 | 0.1278 | 0.2643 | 0.3084 | 0.4978 | 0.2687 | 0.4493 | 0.5110 | 0.7841 |
| mimic_test | single_label | 1646 | 1.0000 | 1.0000 | 0.2005 | 0.3044 | 0.3603 | 0.6312 | 0.2005 | 0.3044 | 0.3603 | 0.6312 |
| mimic_test | ALL | 1873 | 1.1313 | 1.0000 | 0.1917 | 0.2995 | 0.3540 | 0.6151 | 0.2088 | 0.3219 | 0.3785 | 0.6498 |

## Top50 Miss Summary

- rank>50 cases: 721
- overlap_zero_rate: 0.4938
- seen_label_rate: 0.8918
- mean_ic_weighted_overlap: 0.0743
- mean_semantic_overlap: 0.1643

## Interpretation

- single-label exact 可以作为主表切片或附表；multi-label any-label 只能用于 supplementary。
- overlap_zero 在 rank>50 样本中偏高，说明 mimic_test 的主要瓶颈包含 candidate recall / phenotype coverage，而不只是 top50 内排序。
- candidate augmentation 应优先评估 rank>50 样本是否能被 similar-case 或 synonym/OMIM/ORPHA source 找回。
