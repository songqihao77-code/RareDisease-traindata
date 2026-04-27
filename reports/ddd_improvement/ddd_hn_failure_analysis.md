# DDD Ontology-aware HN Failure Analysis

## 1. Summary

HN exact training 不是当前最优主线：它只小幅提升 DDD top1，但没有改善 top3/top5/recall@50，整体不如 validation-selected rerank 稳定。

## 2. Metric Comparison

| Method | DDD Top1 | DDD Top3 | DDD Top5 | DDD Recall@50 | Overall Top1 | mimic Top1 | Role |
|---|---:|---:|---:|---:|---:|---:|---|
| HGNN baseline | 0.3022 | 0.4442 | 0.4967 | 0.7451 | 0.2794 | 0.1917 | baseline |
| validation-selected grid rerank (DDD_top1) | 0.3693 | 0.4875 | 0.5506 | 0.7451 | 0.2616 | 0.1639 | recommended DDD mainline |
| validation-selected gated rerank (DDD_top1) | 0.3693 | 0.4836 | 0.5453 | 0.7451 | 0.2693 | 0.1666 | candidate / supplementary |
| ontology-aware HN exact | 0.3154 | 0.4389 | 0.4954 | 0.7411 | 0.2307 | 0.1148 | negative trained ablation |

相对 HGNN baseline，HN exact 的 DDD top1 为 `+0.0132`，但 DDD top3 为 `-0.0053`，DDD top5 为 `-0.0013`，DDD recall@50 为 `-0.0040`。这说明 HN 把少量 case 推到 rank 1，但没有提升候选列表整体排序。

## 3. Training Log Diagnosis

正式训练输出目录为 `outputs/ddd_ontology_hn/`，训练日志为 `outputs/ddd_ontology_hn/logs/history_20260427_091719.csv/json`。训练配置监控 `val_DDD_top5`，最终 `best_epoch=2`，`best_val_DDD_top5=0.5548780487804879`。

best epoch 明显偏早。第 2 轮的 validation DDD 指标为 top1/top3/top5 `0.3659/0.4817/0.5549`，而最后第 15 轮为 `0.2866/0.4634/0.4939`。训练集 top1 从第 2 轮的 `0.6093` 持续升到第 15 轮的 `0.7937`，但 DDD validation top5 在第 2 轮后整体走弱，说明模型在 HN 约束下继续拟合训练分布，并没有转化为稳定的 DDD validation 排序收益。

validation 与 test 也存在落差。best epoch 的 `val_DDD_top5=0.5549`，但 fixed test DDD top5 为 `0.4954`，相差约 `0.0595`。validation DDD case 数较小，且上一轮报告中 validation DDD 只有 `164` 个 case，选择稳定性不足。第 4 轮曾达到最高 validation DDD top1 `0.3780`，但该轮 top5 `0.5488` 低于第 2 轮；这也说明不同目标之间存在摇摆。

## 4. Candidate Pool Diagnosis

dry-run 显示 `HN-mixed` 已经使用 candidate pools，而不是完全退化为 `HN-current`。其 composition 为 `current=0.494; above_gold=0.150; sibling=0.181; shared_ancestor=0.100; overlap=0.075`，fallback_rate 为 `0.49375`。这表示大约一半负样本仍来自 current score-based miner，ontology-aware pools 的实际覆盖有限。

`above_gold` 覆盖不足较明显。dry-run 平均 pool width 中 `above_gold=2.6875`，低于 `sibling/same_parent=9.125` 和 `shared_ancestor/overlap=20.0`；`HN-above-gold` 的 fallback_rate 达到 `0.725`。正式训练时也出现 warning：`above_gold requested but no current batch case has above-gold candidates`。这与 dry-run 风险一致，说明 above-gold pool 无法稳定覆盖每个 batch。

same_parent/shared_ancestor 负样本可能过宽。`shared_ancestor` 覆盖充分，但父/祖先关系粒度可能较粗，可能把弱相关疾病也作为 hard negative；overlap 负样本则可能过强，把表型相似、诊断上合理接近的疾病过度推远，损害泛化排序。

## 5. Ranking Behavior Diagnosis

HN 的主要变化集中在 rank 1。DDD details 中，HN exact 的 DDD rank 分布为：rank=1 有 `240/761`，rank<=3 有 `334/761`，rank<=5 有 `377/761`，rank<=50 有 `564/761`。baseline 对应为 rank=1 `230/761`，rank<=3 `338/761`，rank<=5 `378/761`，rank<=50 `567/761`。

这解释了指标模式：HN 多得到 10 个 top1 case，但损失了部分 top3/top5/top50 case。也就是说，HN 没有系统性地把 gold 往前移动，而是在少量 case 上更激进地提升 rank 1，同时对中高位排序和候选覆盖造成轻微伤害。

rerank 只在已有 top50 内重排，因此 recall@50 固定为 `0.7451`，更适合当前 DDD 问题。HN 训练会改变全局 embedding 和 candidate score 分布，因此可能改变 gold 是否进入 top50，带来 recall@50 下降。当前 DDD 低准确率主要来自 top50 内排序，直接 fine-tune HGNN 的收益反而不如 top50 evidence rerank 稳定。

## 6. Likely Causes

1. HN loss 权重偏大。配置中 `hard_negative.weight=2.0`，可能让模型过度响应 hard negatives，损伤全局排序结构。
2. candidate pool 覆盖不均。`HN-mixed` fallback_rate `0.49375`，`above_gold` 平均宽度仅 `2.6875`，导致训练信号不稳定。
3. ontology relation 粒度过宽。same_parent/shared_ancestor 可能提供语义过宽负样本，overlap 可能提供过强近邻负样本，二者都可能把合理近似疾病推远。
4. validation 选择不稳定。best epoch 在第 2 轮，validation DDD 样本较少，validation top5 与 test top5 有明显落差。
5. 训练目标与问题形态不匹配。DDD 当前主要需要 top50 内排序校正，而 HN fine-tuning 改变全局 candidate distribution，比 evidence rerank 更不稳定。

## 7. Recommendation

不建议继续以当前配置推进 HN exact training，也不建议再做 test-side HN 调参。HN 当前应作为 supplementary negative result / trained ablation，说明 direct ontology-aware hard negative fine-tuning 没有稳定改善整体 ranking。

后续如果继续 HN，只应在 validation 上做 lighter HN ablation：降低 HN loss weight，减少 shared_ancestor，提升 above_gold 覆盖，拆分 sibling-only、overlap-only、above-gold-only。更优先的方向是 pairwise/listwise top50 reranker：输入 evidence features，目标为 gold 排在 top50 negatives 之前，validation 选模型，test 只在最终固定配置下运行一次。
