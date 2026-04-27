# DDD Next Experiment Recommendations After HN

## P0: Freeze Current DDD Result

- 固定 `validation-selected grid rerank (DDD_top1)` 作为 DDD 主结果。
- 暂停继续 test evaluation；当前 HN test evaluation 已经用过一次，不应重复。
- 将 ontology-aware HN exact 作为 negative ablation / supplementary result。
- 论文主表优先报告 HGNN baseline 与 HGNN + validation-selected rerank。

## P1: Validation-only Lighter HN Ablation

只在 validation 上做，不跑 test：

- 降低 HN loss weight，例如从 `2.0` 降到更轻的权重。
- 减少或关闭 `shared_ancestor`，避免语义过宽负样本主导训练。
- 增加 `above_gold` 覆盖，先解决 batch 内候选不足和 fallback 过高问题。
- 分离 sibling-only、overlap-only、above-gold-only，避免 mixed 策略掩盖单一来源效果。
- 记录 validation curve，不根据 test 选择 epoch、权重或策略。

## P2: Pairwise Top50 Reranker

优先级高于继续直接训练 HGNN：

- 输入 evidence features，如 `hgnn_score`、`ic_weighted_overlap`、`exact_overlap`、`semantic_ic_overlap`、coverage、disease size 等。
- 训练目标为 gold candidate 排在 top50 negatives 之前。
- 使用 validation candidates 选择模型、正则强度和 feature set。
- test 只在最终固定配置下运行一次。
- 与现有 no-train rerank 对齐，重点提升 top50 内排序，不改变 HGNN encoder。

## P3: Label / Ontology Relaxed Analysis

作为分析，不覆盖 exact metric：

- parent-child relaxed hit。
- synonym / obsolete correction。
- top1 near-miss 是否为 sibling、parent、child 或 shared ancestor。
- 多标签或等价疾病导致的潜在假错误审计。
- 所有 relaxed analysis 都应作为 supplementary，不替代 exact evaluation。

## Stop Conditions

- 不再使用 test set 调 HN、rerank gate 或 pairwise reranker。
- 不再新增 test-side grid/gate/search。
- 不修改 mimic 主线。
- 不修改 HGNN encoder。
- 不覆盖原始 exact evaluation。
