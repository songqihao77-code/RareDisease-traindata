# mimic_test multilabel audit

- original single-label cases: 1647
- original multi-label cases: 227
- original unresolved-label cases: 1
- cleaned single-label cases: 1646
- cleaned multi-label cases: 227
- original multi-label 但 cleaned 变 single-label 的病例数: 0
- original 多标签中被 cleaned 丢掉 gold labels 的病例数: 1
- cleaned 中新增 labels 的病例数: 0

## 判断

- 当前 exact top1/top3/top5 可能被多标签处理低估，前提是 evaluator 只取每个 `case_id` 的第一个 `mondo_label` 作为唯一 gold；这与已有 any-label@5 明显高于 multi-label exact top5 的现象一致。
- any-label@k 应只作为 supplementary，不能替代正式 exact evaluation。
- 建议同时保留 original exact、cleaned exact、any-label 三种口径，并在报告中明确 gold 选择规则。