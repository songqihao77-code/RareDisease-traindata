# Similar-case leakage audit

- matched pair count: 18408
- matched case 全部来自 train/library namespace: 是
- full case_id 与 test 重复数: 0
- 去 namespace 后的 local `case_N` 后缀重复数: 4946；这些是文件内局部行号，不是原始 note/patient/admission ID，不能单独作为 leakage 证据。
- same label + identical HPO set 数: 0
- patient/admission/same note: 无法确认 patient/admission-level leakage，需要原始 ID 映射。

## risk summary
| risk_level | count |
| --- | --- |
| low | 18408 |
