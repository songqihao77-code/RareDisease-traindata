# mimic_test split / leakage audit

- 搜索到 split/case 文件数: 18
- leakage/collision rows: 0
- cleaned mimic_test 是否混入 train/val: 未发现基于 case_id 或 label+HPO signature 的直接重叠

## 无法确认项

- 同一个 patient 是否跨 split、同一个 `subject_id` / `hadm_id` / `note_id` 是否跨 split：processed train/test 表缺少这些字段，无法确认，需要包含 MIMIC patient/admission/note 映射的源文件。
- similar-case library 是否包含 mimic_test 的重复 case：本脚本检查了 processed 根目录下的 `mimic_rag_0425.csv` 等表的 label+HPO signature 重叠；是否文本级重复仍需要原始 clinical text 或 text hash 库。
