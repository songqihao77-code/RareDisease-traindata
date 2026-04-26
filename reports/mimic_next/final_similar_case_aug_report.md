# Final SimilarCase-Aug Report

## 1. SimilarCase-Aug 是否有效？
有效。Top5 从 0.3566 提升到 0.3940，rank<=50 从 0.6188 提升到 0.6498。

## 2. 是否达到 DeepRare mimic Top5 target？
达到。DeepRare Top5 target=0.39，当前 fixed test Top5=0.3940。

## 3. Top1/Top3 是否达到 DeepRare？
仍未达到预期：Top1=0.2093，Top3=0.3300。当前模块主要改善 Top5/候选排序，不足以解决 Top1/Top3。

## 4. 是否存在 leakage 风险？
未发现 critical full-ID test leakage；critical=0，medium same-label-identical-HPO=0。去 namespace 后的 local `case_N` 后缀有重复，但这是文件内局部行号，不能作为真实 note/patient/admission 泄漏证据；缺少 subject_id/hadm_id/note_id 映射，patient/admission-level leakage 仍无法确认。

## 5. 当前结果能否进入主表？
可以作为 validation-selected fixed-test 的增强模块结果进入主表或附表；必须同时保留 HGNN exact baseline，any-label 只能 supplementary。

## 6. 是否建议继续训练 pairwise reranker？
当前不建议。Top5 已达标；bootstrap 显示 source 有效但 frozen 权重不是最稳定胜出的参数，现阶段应先补齐 leakage 审计和 validation protocol，而不是训练新 reranker。

## 7. 下一步最小实验是什么？
补齐 MIMIC 原始 note_id/subject_id/hadm_id 映射，做 patient/admission-level leakage audit；同时在 validation bootstrap 或不同 split 上复验 frozen config 稳定性。

## Stability note
- frozen config bootstrap selected count: 0
- frozen config bootstrap top5 improvement CI 为正，说明固定配置本身仍有稳定正增益。
- 但 bootstrap selection 更偏向 `similar_case_weight=0.5`，因此 frozen 权重选择存在轻度稳定性风险。
- 不根据 bootstrap 结果重跑 test，不做 test 调参。
