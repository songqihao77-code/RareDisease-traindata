# Processed MONDO Mapping Audit

## 总体结论

- 数据集数: 8，processed 病例被原始标注支持: 12734 / 12735。
- 原始样本中存在同病例多 MONDO 的情况: 1594 例；映射后保留为多标签的病例: 1304 例。
- 映射后按 `HPO 集合` 聚合后，出现“同一套 HPO 对应多个 MONDO”的签名数: 1379。

## 分数据集结果

### ddd_test

- raw_cases=2283，processed_cases=2283，supported_processed_cases=2283，unsupported_processed_cases=0
- exact_equal_processed_cases=2142，subset_only_processed_cases=141，expected_retained_cases=2283
- raw_multi_label_cases=187，processed_multi_label_cases=46，processed_blank_mondo_cases=0
- unresolved_raw_cases=0，route_counts={"input_mondo": 634, "omim_exact": 801, "orpha_exact": 1471, "orpha_via_omim": 11}
- raw_hpo_collision_signatures=246，processed_hpo_collision_signatures=118
- 同一 HPO 对应多个 MONDO 样例:
  - n_cases=17, n_unique_mondo_ids=18, hpo_count=2, mondo_sets=[['MONDO:0004580'], ['MONDO:0009319', 'MONDO:0016305'], ['MONDO:0010806'], ['MONDO:0011075'], ['MONDO:0011974']]
  - n_cases=8, n_unique_mondo_ids=8, hpo_count=1, mondo_sets=[['MONDO:0007634'], ['MONDO:0010451'], ['MONDO:0010501'], ['MONDO:0010508'], ['MONDO:0014764']]
  - n_cases=8, n_unique_mondo_ids=8, hpo_count=2, mondo_sets=[['MONDO:0010250'], ['MONDO:0014995'], ['MONDO:0020631'], ['MONDO:0030909'], ['MONDO:0030921']]

### mimic_test

- raw_cases=1875，processed_cases=1873，supported_processed_cases=1873，unsupported_processed_cases=0
- exact_equal_processed_cases=1873，subset_only_processed_cases=0，expected_retained_cases=1874
- raw_multi_label_cases=227，processed_multi_label_cases=227，processed_blank_mondo_cases=0
- unresolved_raw_cases=1，route_counts={"manual_orpha_to_mondo": 6, "orpha_exact": 2109, "orpha_name_unique": 5}
- raw_hpo_collision_signatures=227，processed_hpo_collision_signatures=227
- 同一 HPO 对应多个 MONDO 样例:
  - n_cases=1, n_unique_mondo_ids=3, hpo_count=10, mondo_sets=[['MONDO:0007108', 'MONDO:0009637', 'MONDO:0019194']]
  - n_cases=1, n_unique_mondo_ids=3, hpo_count=10, mondo_sets=[['MONDO:0015128', 'MONDO:0016383', 'MONDO:0018907']]
  - n_cases=1, n_unique_mondo_ids=3, hpo_count=16, mondo_sets=[['MONDO:0011996', 'MONDO:0020547', 'MONDO:0034150']]

### mimic_rag_0425

- raw_cases=7313，processed_cases=7311，supported_processed_cases=7310，unsupported_processed_cases=1
- exact_equal_processed_cases=7308，subset_only_processed_cases=2，expected_retained_cases=7311
- raw_multi_label_cases=1032，processed_multi_label_cases=1031，processed_blank_mondo_cases=0
- unresolved_raw_cases=2，route_counts={"manual_orpha_to_mondo": 20, "orpha_exact": 8457, "orpha_name_unique": 11}
- raw_hpo_collision_signatures=1032，processed_hpo_collision_signatures=1031
- 无法被原始标注支持的 processed 样例:
  - case_id=case_7206, mondo=['MONDO:0020311', 'MONDO:0020520'], hpo_count=11, hpo_preview=['HP:0000969', 'HP:0002092', 'HP:0002094', 'HP:0002097', 'HP:0005185', 'HP:0012325', 'HP:0012398', 'HP:0012418'], raw_candidates=[['MONDO:0020311']]
- 同一 HPO 对应多个 MONDO 样例:
  - n_cases=1, n_unique_mondo_ids=6, hpo_count=19, mondo_sets=[['MONDO:0007194', 'MONDO:0007345', 'MONDO:0015924', 'MONDO:0016343', 'MONDO:0019499', 'MONDO:0020295']]
  - n_cases=1, n_unique_mondo_ids=5, hpo_count=8, mondo_sets=[['MONDO:0015128', 'MONDO:0015977', 'MONDO:0018015', 'MONDO:0018438', 'MONDO:0034150']]
  - n_cases=1, n_unique_mondo_ids=4, hpo_count=24, mondo_sets=[['MONDO:0007915', 'MONDO:0018896', 'MONDO:0019532', 'MONDO:8000010']]

### HMS

- raw_cases=88，processed_cases=88，supported_processed_cases=88，unsupported_processed_cases=0
- exact_equal_processed_cases=52，subset_only_processed_cases=36，expected_retained_cases=88
- raw_multi_label_cases=36，processed_multi_label_cases=0，processed_blank_mondo_cases=0
- unresolved_raw_cases=0，route_counts={"omim_exact": 131, "orpha_exact": 109}
- raw_hpo_collision_signatures=36，processed_hpo_collision_signatures=0

### LIRICAL

- raw_cases=370，processed_cases=370，supported_processed_cases=370，unsupported_processed_cases=0
- exact_equal_processed_cases=348，subset_only_processed_cases=22，expected_retained_cases=370
- raw_multi_label_cases=22，processed_multi_label_cases=0，processed_blank_mondo_cases=0
- unresolved_raw_cases=0，route_counts={"omim_exact": 370, "orpha_exact": 227}
- raw_hpo_collision_signatures=22，processed_hpo_collision_signatures=0

### MME

- raw_cases=40，processed_cases=40，supported_processed_cases=40，unsupported_processed_cases=0
- exact_equal_processed_cases=39，subset_only_processed_cases=1，expected_retained_cases=40
- raw_multi_label_cases=1，processed_multi_label_cases=0，processed_blank_mondo_cases=0
- unresolved_raw_cases=0，route_counts={"omim_exact": 40, "orpha_exact": 28}
- raw_hpo_collision_signatures=1，processed_hpo_collision_signatures=0

### RAMEDIS

- raw_cases=624，processed_cases=624，supported_processed_cases=624，unsupported_processed_cases=0
- exact_equal_processed_cases=583，subset_only_processed_cases=41，expected_retained_cases=624
- raw_multi_label_cases=41，processed_multi_label_cases=0，processed_blank_mondo_cases=0
- unresolved_raw_cases=0，route_counts={"omim_exact": 624, "orpha_exact": 591}
- raw_hpo_collision_signatures=42，processed_hpo_collision_signatures=2
- 同一 HPO 对应多个 MONDO 样例:
  - n_cases=11, n_unique_mondo_ids=2, hpo_count=6, mondo_sets=[['MONDO:0009861'], ['MONDO:0010162']]
  - n_cases=7, n_unique_mondo_ids=2, hpo_count=4, mondo_sets=[['MONDO:0008721'], ['MONDO:0008723']]

### MyGene2

- raw_cases=146，processed_cases=146，supported_processed_cases=146，unsupported_processed_cases=0
- exact_equal_processed_cases=98，subset_only_processed_cases=48，expected_retained_cases=146
- raw_multi_label_cases=48，processed_multi_label_cases=0，processed_blank_mondo_cases=0
- unresolved_raw_cases=0，route_counts={"input_mondo": 176, "omim_exact": 146, "orpha_exact": 184}
- raw_hpo_collision_signatures=47，processed_hpo_collision_signatures=1
- 同一 HPO 对应多个 MONDO 样例:
  - n_cases=2, n_unique_mondo_ids=2, hpo_count=1, mondo_sets=[['MONDO:0011512'], ['MONDO:0014029']]
