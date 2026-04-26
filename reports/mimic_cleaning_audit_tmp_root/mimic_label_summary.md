# mimic_test label / disease ID audit

- MONDO 资源: D:\RareDisease-traindata\data\raw_data\mondo-base.json
- exact_same: 1810
- id_mismatch: 63
- missing_in_cleaned: 2
- extra_in_cleaned: 0
- uncertain: 0

## 分布

| match_type | severity | case_count |
| --- | --- | --- |
| exact_same | low | 1810 |
| id_mismatch | critical | 63 |
| missing_in_cleaned | critical | 2 |

## 审计说明

- original 的 disease ID 主要来自 `orpha`，本脚本使用本地 `mondo_hasdbxref_orphanet.sssom.tsv`、`orpha2omim.json`、`mondo_exactmatch_omim.sssom.tsv` 和少量项目内既有 manual override 映射到 MONDO。
- ICD 粗粒度字段存在于 original `icd_code`，但 cleaned gold label 使用 `mondo_label`；本脚本没有把 ICD 直接当作 exact gold。
- OMIM gene ID 与 phenotype ID、ORPHA group ID 与 disorder ID 是否混淆：仅凭当前 CSV 无法完全确认，需要原始 Orphanet/OMIM 语义层级文件逐项人工核对。
- synonym、obsolete、parent-child 只在本地 MONDO JSON 可加载时判断；无法确认的项目会标为 `uncertain`。