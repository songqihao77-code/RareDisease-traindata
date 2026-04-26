# Recommended Next Experiments

## 可以进入论文主表的结果

- 原始 exact HGNN baseline：来自 `src.evaluation.evaluator` 的 top1/top3/top5/rank<=50，可进入主表。
- validation 选定、test 只评估一次的 top50 reranker：只有在使用 validation candidates 保存固定权重后，才能进入主表。

## 只能作为附表或 error analysis 的结果

- 当前 test-side rerank grid/gate 结果：只能作为 exploratory upper bound 或消融分析。
- `mimic_test` any-label hit：不覆盖原始 exact metric，只用于解释多标签病例的潜在低估。
- synonym / parent-child / obsolete relaxed evaluation：只能作为附加口径，不能替代 exact evaluation。

## 不需要训练即可优先修复

1. 补充 MONDO/HPO obsolete、alt_id、synonym、subclass 审计。
2. 对 `mimic_test` 多标签病例输出 exact 与 any-label 双口径。
3. 对 rank>50 样本补充 gold hyperedge coverage，并修复明显 unmapped/obsolete 标签。
4. 核对 HMS 25-case test split 与论文口径是否一致。

## 需要 reranker 的问题

1. gold 已在 top50 但 rank>5 的样本，尤其 DDD near-miss。
2. evidence 特征能解释的 top50 内排序错误，例如 exact/IC/semantic overlap 明显支持 gold。
3. validation-selected linear/listwise reranker，用固定权重或轻量模型只在 HGNN top50 内重排。

## 需要 hard negative training 的问题

1. DDD 中同父类、同祖先或 HPO 高重叠疾病反复混淆。
2. 当前在线 hard negatives 覆盖不到的 ontology sibling / disease family negatives。
3. top50 内候选分数差距小但语义上高度相近的病例。

## 暂不建议作为主线的方向

- 把 static HPO retrieval 作为硬候选池；现有诊断显示 static@50 整体弱且对 `mimic_test` 不稳。
- 在 test candidates 上直接选择 rerank 权重；必须改为 validation select -> fixed test。
