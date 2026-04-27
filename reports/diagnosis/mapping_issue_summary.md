# Mapping Issue Summary

## Issue Counts
| issue_type | count |
| --- | --- |
| obsolete_HPO | 248 |
| obsolete_MONDO | 62 |
| npz_disease_dimension_check | 1 |
| npz_hpo_dimension_check | 1 |
| xref_one_to_many_mapping_count | 1 |

## 关键判断
- `Disease_index_v4.xlsx` 行数与 `v59DiseaseHy.npz` disease 维度一致: True
- `HPO_index_v4.xlsx` 行数与 `v59DiseaseHy.npz` HPO 维度一致: True
- processed dataset 的 disease/HPO 映射问题见 `mondo_hpo_mapping_audit.csv`。
- MONDO xref 存在 one-to-many 风险，OMIM/ORPHA/ICD 到 MONDO 的自动映射不应无人工规则直接用于 exact gold 改写。
- synonym 或 parent-child 命中可以解释 near miss，但不能混入正式 exact evaluation。
