# mimic_test HPO 与 Disease Hyperedge Overlap Audit

- semantic_overlap 来源：`HpoSemanticMatcher.from_project`；metadata: `{"available": true, "ontology_path": "D:\\RareDisease-traindata\\raw_data\\hp.json", "num_terms": 19388, "num_terms_with_ancestors": 19388}`

## Overall
| num_cases | overlap_zero_count | overlap_zero_ratio | overlap_le_1_count | overlap_le_1_ratio | mean_exact_overlap_count | mean_ic_weighted_overlap | mean_semantic_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1873 | 687 | 0.3668 | 1305 | 0.6967 | 1.1949 | 0.1184 | 0.2038 |

## Buckets
| bucket_type | bucket | num_cases | top1 | top5 | rank_le_50 | median_rank | mean_rank | overlap_zero_ratio | overlap_le_1_ratio | mean_exact_overlap_count | mean_ic_weighted_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_hpo_count_bucket | >10 | 782 | 0.2059 | 0.3632 | 0.6624 | 18.0000 | 449.4425 | 0.2813 | 0.5921 | 1.5639 | 0.1015 |
| case_hpo_count_bucket | 7-10 | 615 | 0.1642 | 0.3545 | 0.6211 | 21.0000 | 321.8764 | 0.3366 | 0.7057 | 1.1545 | 0.1276 |
| case_hpo_count_bucket | 4-6 | 372 | 0.1640 | 0.3118 | 0.6075 | 23.0000 | 393.2366 | 0.4946 | 0.8280 | 0.7339 | 0.1377 |
| case_hpo_count_bucket | 1-3 | 104 | 0.0962 | 0.2019 | 0.4327 | 101.5000 | 809.0481 | 0.7308 | 0.9615 | 0.3077 | 0.1220 |
| exact_hpo_overlap_count_bucket | 2-3 | 440 | 0.2727 | 0.4795 | 0.7500 | 6.0000 | 129.9250 | 0.0000 | 0.0000 | 2.3477 | 0.2267 |
| exact_hpo_overlap_count_bucket | 1 | 618 | 0.2201 | 0.3673 | 0.6521 | 18.0000 | 334.1230 | 0.0000 | 1.0000 | 1.0000 | 0.1237 |
| exact_hpo_overlap_count_bucket | 0 | 687 | 0.0684 | 0.1951 | 0.4891 | 55.0000 | 730.8006 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| exact_hpo_overlap_count_bucket | >3 | 128 | 0.2344 | 0.5234 | 0.7969 | 5.0000 | 110.3750 | 0.0000 | 0.0000 | 4.5859 | 0.3557 |
| gold_disease_hpo_count_bucket | >20 | 632 | 0.2215 | 0.3782 | 0.6361 | 23.0000 | 417.8354 | 0.1646 | 0.4620 | 1.9921 | 0.1845 |
| gold_disease_hpo_count_bucket | 6-20 | 751 | 0.1198 | 0.2903 | 0.5885 | 27.0000 | 304.1119 | 0.3941 | 0.7457 | 0.9720 | 0.0995 |
| gold_disease_hpo_count_bucket | 1-5 | 489 | 0.2106 | 0.3722 | 0.6687 | 14.0000 | 573.3395 | 0.5849 | 0.9243 | 0.5092 | 0.0622 |
| gold_disease_hpo_count_bucket | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 7020.0000 | 7020.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| gold_in_top50_bucket | yes | 1171 | 0.2844 | 0.5457 | 1.0000 | 4.0000 | 11.1682 | 0.2869 | 0.6311 | 1.4056 | 0.1434 |
| gold_in_top50_bucket | no | 702 | 0.0000 | 0.0000 | 0.0000 | 195.5000 | 1092.2578 | 0.5000 | 0.8063 | 0.8433 | 0.0766 |

## 结论
- overlap=0 的样本占比为 0.3668；overlap<=1 的样本占比为 0.6967。
- 如果 overlap=0 或 overlap<=1 比例较高，首先指向数据/HPO 抽取/知识库覆盖问题，模型排序只能在候选已有有效证据时发挥作用。
- 多视图超边信息可以直接利用 exact/IC/semantic overlap、MONDO ontology、synonym/xref 做 candidate expansion 和 evidence rerank，因此更直接。
- 图对比学习在 low-overlap 样本上容易把错误或弱证据 pair 当作正负对放大，除非先做 label 清洗、低置信样本过滤和 validation-selected 采样。
