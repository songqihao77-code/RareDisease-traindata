# DeepRare Target 统一实验路线

## P0：锁定指标口径
- 确认所有 current mainline 数字均来自 validation-selected fixed test 或无 test-side tuning 的 fixed mainline test。
- 表格中统一把 Recall@1/@3/@5 与 Top1/Top3/Top5 作为 strict exact 同一指标族。
- 不混写 baseline exact、frozen config experiments、current mainline final results。

## P1：target-aware no-train rerank
- 对 Top5 未达标的数据集，先做 candidate recall audit 和 candidate expansion。
- 对 Top5 已达标但 Top1/Top3 未达标的数据集，做 top5/top10 局部重排。
- 所有权重只在 validation 上选择；test 只做 fixed evaluation。

## P2：light-train reranker
- 构造统一候选表。
- 特征包括 HGNN score、SimilarCase score、HPO overlap、IC overlap、MONDO relation、source count、rank/margin features，以及经过 validation 证明有效的数据集指示特征。
- 首选 linear / GBDT / pairwise reranker；目标函数优先优化 Recall@1 和 Recall@3，同时约束 Recall@5 不下降。

## P3：ontology-aware hard negative training
- 重点用于 DDD 以及其他 Rank<=50 高但 Top1/Top3 低的数据集。
- 负样本包括 same-parent、sibling、高 HPO-overlap、top50-above-gold。
- validation selection 与 fixed test reporting 必须分开。

## P4：图对比学习
- 仅在 label 清洗稳定、candidate recall 足够、正负对可靠后再做。
- 不作为当前缺口的第一优先级。
