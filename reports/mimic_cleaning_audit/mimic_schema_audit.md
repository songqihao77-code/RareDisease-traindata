# mimic_test schema audit

## original
- 行数: 1875
- 唯一病例数: 1875
- 病例 ID 依据: `note_id`
- 列名: `text`, `icd_code`, `diagnosis`, `phenotype`, `rare`, `embedding`, `note_id`, `orpha`, `orpha_names`, `HPO`
- split 字段: 无
- 重复行数量: 0

| column | inferred meaning | null_ratio | format_guess |
|---|---:|---:|---|
| `text` | free-text 临床文本字段 | 0.0000 | semicolon separated string |
| `icd_code` | ICD code 字段 | 0.0000 | Python literal list |
| `diagnosis` | disease label/name 字段 | 0.0000 | single string |
| `phenotype` | 无法确认 | 0.0000 | comma separated string |
| `rare` | disease label/name 字段 | 0.0000 | Python literal list |
| `embedding` | embedding / 向量字段 | 0.0000 | single string |
| `note_id` | note_id ID 字段 | 0.0000 | single string |
| `orpha` | ORPHA disease ID/name 字段 | 0.0000 | Python literal list |
| `orpha_names` | ORPHA disease ID/name 字段 | 0.0000 | Python literal list |
| `HPO` | HPO phenotype 字段 | 0.0000 | Python literal list |

## cleaned
- 行数: 21749
- 唯一病例数: 1873
- 病例 ID 依据: `case_id`
- 列名: `case_id`, `mondo_label`, `hpo_id`
- split 字段: 无
- 重复行数量: 0

| column | inferred meaning | null_ratio | format_guess |
|---|---:|---:|---|
| `case_id` | case_id 字段 | 0.0000 | single string |
| `mondo_label` | MONDO disease ID / gold label 字段 | 0.0000 | single string |
| `hpo_id` | HPO phenotype 字段 | 0.0000 | single string |
