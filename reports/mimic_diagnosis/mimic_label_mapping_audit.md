# mimic_test Label / MONDO Mapping Audit

## Summary
| unique_mondo_labels | labels_in_disease_index | labels_not_in_disease_index | labels_in_disease_hyperedge | labels_without_disease_hyperedge | obsolete_mondo_labels | normalization_fix_possible | needs_manual_review |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 353 | 353 | 0 | 352 | 1 | 18 | 0 | 18 |

## 结论
- mimic_test unique MONDO labels 为 353；不在 Disease_index 的 label 为 0。
- 没有对应 disease hyperedge 的 label 为 1。
- 可通过 `MONDO_` 到 `MONDO:` normalization 修复的 label 为 0。
- 受 label 不对齐直接影响的病例数为 0；当前低准确率主要不能归因于 Disease_index 缺失。
- synonym / replacement / parent-child 可以解释部分 exact miss，但不能直接改写主 exact metric；建议作为 relaxed MONDO evaluation 的 supplementary error analysis。
- 需要人工确认的 label 见 `mimic_unmapped_labels.csv`。
