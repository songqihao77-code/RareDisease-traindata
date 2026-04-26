# mimic_test Cleaning Audit Report

## 1. Overall Judgment
cleaned mimic_test 可用于审计但不应直接视为完全可信；需要先核对缺失病例和 critical label/mapping 问题。

## 2. Dataset Schema Comparison
- original: 1875 rows，列为 `text`, `icd_code`, `diagnosis`, `phenotype`, `rare`, `embedding`, `note_id`, `orpha`, `orpha_names`, `HPO`。
- cleaned: 21749 rows，按 `case_id` 聚合为 1873 cases，列为 `case_id`, `mondo_label`, `hpo_id`。
- original 是宽表，包含临床 `text`、`note_id`、`orpha`、`orpha_names`、`HPO`；cleaned 是病例-HPO/label 长表。

## 3. Case Count and Granularity
- original 有 1875 rows / 1875 unique `note_id`。
- cleaned 有 21749 rows / 1873 unique `case_id`。
- cleaned 为 1873 的直接原因是唯一 `case_id` 从 `case_1` 到 `case_1873`，而不是 21749 行长表。
- missing_in_cleaned: 2。如果 original 是 1875，则缺失病例见 `mimic_missing_cases.csv`。
- original 粒度判断为 note-level；cleaned 是 case-level long format。patient/admission-level 无法确认，需要 `subject_id` / `hadm_id`。

## 4. Case-Level Matching
- matched: 1873
- label_mismatch: 0
- hpo_mismatch: 0
- missing_in_cleaned: 2
- extra_in_cleaned: 0
- 匹配优先级按显式 ID、`MONDO label set + HPO set`、最后行号 fallback；行号 fallback 会在 CSV 中标出。

## 5. Label / Disease ID Audit
- exact_same: 1810
- critical severity rows: 65
- uncertain rows: 0
- ICD 没有被当作 exact gold；ORPHA 到 MONDO 的映射使用本地 SSSOM/JSON 资源。
- OMIM gene/phenotype、ORPHA group/disorder、parent/subtype 的细粒度混淆无法仅凭当前 CSV 完全确认，需要额外 ontology/manual curation 文件。

## 6. Multi-label Audit
- cleaned multi-label cases: 227
- original multi-label cases（按解析出的 MONDO set）: 227
- 当前 exact top1/top3/top5 可能被多标签处理低估；any-label@k 应只作为 supplementary。
- 建议保留 original exact、cleaned exact、any-label 三种口径。

## 7. HPO Audit
- HPO Jaccard median: 1.0000
- invalid/obsolete HPO cases: 0
- large_hpo_loss cases: 0
- 是否误删高信息量 HPO、误加入否定症状、家族史或治疗反应：无法确认，需要 HPO 抽取证据 spans 或人工标注说明。

## 8. Gold Disease-HPO Coverage
- disease-HPO resource: D:\RareDisease-traindata\LLLdataset\DiseaseHy\rare_disease_hgnn_clean_package_v59\v59_hyperedge_weighted_patched.csv
- overlap_zero_rate（case-level，任一 gold label 有重叠即不算 zero）: 0.3449
- unmapped label rows: 0
- overlap_zero 高若同时伴随 HPO Jaccard 高、unmapped label 低，则更可能来自 disease-HPO KB 覆盖不足或原始文本信息不足，而不是 cleaned HPO 大量丢失。

## 9. Split / Leakage Audit
- leakage/collision rows: 1872
- patient/subject/hadm/note 跨 split 无法确认，因为 processed train/test 缺少这些 ID。

## 10. Accuracy Impact Analysis
当前指标 top1=0.1917、top3=0.2995、top5=0.3540、rank<=50=0.6151。影响判断如下：

| issue | affected_cases | expected_accuracy_impact | fix_priority | recommended_fix |
| --- | --- | --- | --- | --- |
| label ID 错误 / missing case / cleaned 多出病例 | 65 | critical: 会直接改变 exact gold，可能造成 top1/top3/top5 被错误计算 | A | 先修复映射和病例对应关系；保留原始 exact 口径，不删除困难样本 |
| 多标签 gold 在 exact evaluator 中只取首个标签 | 227 | high: exact top-k 可能低估，any-label@k 明显更高时尤其需要报告 supplementary | A/C | 保留 cleaned 多标签；同时报告 original exact、cleaned exact、any-label，不把 any-label 当正式 exact |
| cleaned 丢失 original gold labels | 1 | critical: gold label 丢失会让 exact evaluation 不可信 | A | 恢复丢失 label 或明确剔除原因；不可为提高准确率删除测试样本 |
| HPO 大量丢失或 invalid/obsolete HPO | 0 | medium/high: 会降低 gold disease-HPO overlap 和检索证据质量 | B | 统一 HP:0000000 格式，恢复误删 HPO，必要时用 HPO OBO 处理 obsolete/alt_id |
| gold disease-HPO overlap_zero | 646 | high for rank>50: 模型缺少 gold hyperedge 证据时难以召回 | B/D | 区分 cleaned HPO 丢失、label 映射错误、KB 覆盖不足和原文信息不足；不要在 test set 调参 |
| unmapped/obsolete label in hyperedge/index | 0 | critical/high: gold 不在疾病池或 KB 中时 exact retrieval 不可比 | A/B | 修复 MONDO obsolete/alt_id/replaced_by 映射，并记录版本 |
| train/test leakage 或 similar-case 重复 | 1872 | 方向不固定: 泄漏通常虚高，split 不一致会让测试口径不可信 | A | 用 subject_id/hadm_id/note_id/text hash 做最终确认；不要覆盖已有 exact evaluation |
| gold 已进入 top50 但 rank>5 | 由现有 rank 明细确认 | 更像模型排序能力或候选重排问题，不一定是清洗错误 | D | 作为模型问题单独分析；不要用 test set 调参 |

## 11. Fix Plan
1. 优先人工核对 `mimic_missing_cases.csv` 和 `mimic_label_diff.csv` 中 severity=critical 的病例，修复映射，不删除测试样本。
2. 明确 evaluator 对 multi-label 的 gold 选择规则，保留 exact 正式口径，并把 any-label@k 作为 supplementary。
3. 对 overlap_zero 病例抽样回看原始 clinical text、HPO 抽取证据和 v59 disease-HPO hyperedge，区分文本不足、HPO 清洗、label 映射和 KB 缺口。
4. 补充 subject_id/hadm_id/note_id 映射后重新做 split leakage 审计。

## 12. Reproducible Commands
```powershell
D:\python\python.exe -m py_compile tools\audit_mimic_cleaning.py
D:\python\python.exe tools\audit_mimic_cleaning.py --original D:\RareDisease-traindata\LLLdataset\dataset\mimic_test.csv --cleaned D:\RareDisease-traindata\LLLdataset\dataset\processed\test\mimic_test.csv --output-dir reports\mimic_cleaning_audit
```

生成文件:

- `mimic_accuracy_impact.csv`
- `mimic_case_granularity.md`
- `mimic_case_matching.csv`
- `mimic_duplicate_cases.csv`
- `mimic_gold_hpo_coverage.csv`
- `mimic_gold_hpo_coverage_summary.md`
- `mimic_hpo_diff.csv`
- `mimic_hpo_summary.md`
- `mimic_invalid_hpo.csv`
- `mimic_label_diff.csv`
- `mimic_label_summary.md`
- `mimic_missing_cases.csv`
- `mimic_multilabel_diff.csv`
- `mimic_multilabel_summary.md`
- `mimic_overlap_zero_cases.csv`
- `mimic_schema_audit.csv`
- `mimic_schema_audit.md`
- `mimic_split_leakage.csv`
- `mimic_split_leakage.md`