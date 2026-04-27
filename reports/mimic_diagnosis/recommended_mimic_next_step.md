# Recommended mimic_test Next Step

## 1. 低准确率第一原因
- candidate recall 低是第一原因：`gold_absent_from_top50` = 702 / 1873 (0.3748)。
- top50 内排序差是第二层原因：`gold_in_top50_but_rank_gt5` = 532 / 1873 (0.2840)。
- multi-label evaluation 有低估：multi-label 占比 0.1212，any-label top1 delta=0.0176，top5 delta=0.0230，但不能解释全部低分。
- label mapping 不是主因：不在 Disease_index 的 label 数为 0，缺 hyperedge 的 label 数为 1。
- HPO / disease hyperedge overlap 是关键支撑问题：overlap=0 ratio=0.3668，overlap<=1 ratio=0.6967。
- MIMIC domain shift 很可能存在，表现为临床病例 HPO 与 rare disease hyperedge 对齐弱、候选召回不足和多标签诊断粒度混杂。

## 2. 是否优先做多视图、提取超边信息
- 推荐优先做。第一阶段应做 no-train，而不是直接训练。
- 优先视图：patient-HPO view、disease-HPO hyperedge view、MONDO ontology view、HPO IC weighting view、disease synonym/xref view。
- 最重要特征：exact overlap、IC-weighted overlap、semantic overlap、case/disease coverage、MONDO parent/child/sibling/synonym relation、similar-case candidate evidence。
- 应先做 candidate expansion / ontology-aware retrieval，再做 validation-selected fixed-test rerank；否则 gold 不在 top50 的样本无法被 reranker 修复。

## 3. 是否优先做图对比学习
- 不推荐作为第一步。原因是当前主要瓶颈包含 candidate recall 缺失、低 overlap 和多标签/label noise；图对比学习需要训练且依赖高质量正负对，风险更高。
- 若后续做，需要先满足：label 清洗完成、validation-selected protocol 固定、low-overlap 样本过滤或降权、positive pair 不从 test-side exploratory 结果构造。
- 可用正负对：patient-disease exact gold positive、multi-label any-gold positive、MONDO sibling/same-parent hard negative、top50-above-gold hard negative、HPO-overlap hard negative。
- 避免噪声的方法：过滤 overlap=0/<=1 低置信样本；对 ancestor/descendant 关系使用 soft negative 或低权重；正负对只从 train/validation 构造；test 仅 fixed evaluation。

## 4. 下一步实验路线
1. P0：不训练可修复的问题：label normalization、obsolete/replacement audit、multi-label supplementary metric、命令和路径 manifest 固化。
2. P1：no-train / validation-selected rerank：使用 validation 选权重，fixed test 只跑一次；输出 exact 主表和 supplementary any-label/relaxed 表。
3. P2：多视图超边特征增强：加入 MONDO ontology expansion、synonym/xref expansion、IC/semantic overlap evidence。
4. P3：轻量训练 reranker：只训练 top50/topK reranker，不改 HGNN encoder，严格 train/validation/test 分离。
5. P4：图对比学习或 hard negative training：在 P0-P3 后做，使用清洗后的正负对和低噪声采样。

## 5. 最终推荐
- 当前 mimic_test 提升的第一优先级：选择“多视图、提取超边信息”。
- 图对比学习保留为后续 P4 实验，不作为当前第一步。

## Method Comparison
| method | target_problem | expected_help_for_mimic_top1 | expected_help_for_mimic_top5 | expected_help_for_recall50 | requires_training | modifies_encoder | risk_of_overfitting | risk_under_noisy_labels | implementation_cost | recommended_or_not | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 方法A: 多视图、提取超边信息 | candidate expansion, ontology-aware retrieval, evidence rerank, multi-label/relaxed audit | 中等：可通过 overlap/IC/semantic/similar-case rerank 改善已有候选排序 | 较高：能把近邻和证据强候选推入 top5 | 较高：candidate expansion / ontology view 可处理 gold_absent_from_top50 | no-train 优先；后续可 light-train | 否 | 低到中；必须 validation-selected fixed test | 中；可用规则过滤和 error analysis 降低 | 低到中 | 推荐作为第一优先级 | gold_absent_from_top50=0.3748, top50内rank>5=0.2840, overlap<=1=0.6967；该方法能同时覆盖召回、排序、mapping和证据解释，且不需要改encoder。 |
| 方法B: 图对比学习 | representation learning, supervised/cross-view contrast, hard negative contrast | 不确定：依赖正负对质量，短期不如候选扩展直接 | 中等：可能改善相似疾病区分 | 低到中：若仍使用同一候选导出路径，不能保证召回缺失样本进入top50 | 是 | 通常需要；若只加投影头也要训练主表示 | 中到高 | 高；multi-label和low-overlap会制造错误正负对 | 高 | 不推荐作为第一步；可作为P4后续实验 | multi-label ratio=0.1212, label manual-review=18, overlap<=1=0.6967；在清洗和候选召回未解决前容易放大噪声。 |

## Reproducibility
- 本报告命令：`D:\python\python.exe tools\analysis\mimic_diagnosis.py`
- 未训练新模型；未修改 HGNN encoder；未覆盖 baseline、exact evaluation 或 mainline 输出。
