# mimic_test case granularity audit

- original rows / unique cases: 1875 / 1875
- cleaned rows / unique cases: 21749 / 1873
- original `note_id` 是否唯一: 是
- cleaned 病例粒度: `case_id` 聚合后的病例级长表；每个病例可有多个 `mondo_label` 和多个 `hpo_id`。
- original 病例粒度判断: note-level。没有 `subject_id` / `hadm_id` / `patient_id`，无法确认 patient-level 或 admission-level。
- cleaned 病例粒度判断: case-level long format；来源是否严格 note-level 无法仅凭 cleaned 确认。
- matched cases: 1873
- label mismatch cases: 0
- HPO mismatch cases: 0
- missing in cleaned: 2
- extra in cleaned: 0

## 关键判断

- original 是 1875 行且 1875 个唯一 `note_id`。
- cleaned 是 21749 行长表，按 `case_id` 聚合后是 1873 个病例。
- cleaned 之所以是 1873，不是因为长表行数，而是因为 `case_1` 到 `case_1873` 只有 1873 个唯一病例。
- 是否存在一个 patient 多个 admission、一个 admission 多个 note：无法确认，需要包含 `subject_id` / `hadm_id` / patient-level ID 的源文件。
- 是否存在一个 note 多个 disease label：original 可由 `orpha` 列判断，cleaned 可由同一 `case_id` 下多个 `mondo_label` 判断。
- 是否存在原始多个病例被合并或一个病例被拆分：本脚本用 `label set + HPO set` 签名和重复签名表辅助定位，最终仍需结合原始 MIMIC subject/hadm/note 映射人工确认。

## cleaned 缺失病例

- original_id=18174227-DS-8, note_id=18174227-DS-8, label=MONDO:0043224, hpo_count=3, reason=cleaned 中找不到同 ID 或同 label+HPO signature 病例
- original_id=19928034-DS-13, note_id=19928034-DS-13, label=OBSOLETE: Lymphomatous meningitis, hpo_count=9, reason=cleaned 中找不到同 ID 或同 label+HPO signature 病例
