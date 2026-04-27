# Hard Negative Code Audit

## 当前实现
- `src/training/hard_negative_miner.py::mine_hard_negatives` 是 score-based current model top-k negative。
- `mine_configurable_hard_negatives` 已有 `HN-overlap`、`HN-sibling`、`HN-above-gold`、`HN-mixed` 的接口和 fallback 逻辑。
- 但 `src/training/trainer.py` 调用时没有传入 `candidate_pools`，因此除 `HN-current` 外的策略目前会退化为 current top-k negative。
- 当前训练没有真正使用 ontology-aware negative、HPO-overlap negative、MONDO sibling/same-parent negative、top50-above-gold negative。

## False Negative 风险
- sibling/same-parent/ancestor-descendant 疾病可能是合理鉴别诊断或标注粒度差异，强负例可能损伤泛化。
- 建议先以较小 margin/weight 使用 ontology negative，并输出 relation bucket 指标。

## 最小改动方案
1. 在数据预处理阶段生成 disease-level pools：`overlap_pool`、`sibling_pool`、`same_parent_pool`。
2. 从 train/validation top50 candidates 生成 `above_gold_pool`，只用于训练/validation，不从 test 构造训练池。
3. 在 `build_batch_hypergraph` 或 trainer batch 侧按 gold disease idx 取 pool，传给 `mine_configurable_hard_negatives(candidate_pools=...)`。
4. 对 relation-aware negatives 使用较低权重或 soft margin，保留 `HN-current` 对照。

## 建议实验组
- `HGNN baseline`: 当前 exact baseline。
- `HN-current`: current top-k negative 对照。
- `HN-overlap`: 高 HPO-overlap negative。
- `HN-sibling`: MONDO sibling/same-parent negative。
- `HN-top50-above-gold`: validation/train top50 中排在 gold 前面的 negative。
- `HN-mixed`: current + overlap + sibling + shared_ancestor + above_gold。
- `HN-mixed + val-selected rerank`: 训练后固定 validation rerank 权重再 test 一次。

## 已生成配置
- `configs/experiments/hgnn_hn_current.yaml`
- `configs/experiments/hgnn_hn_overlap.yaml`
- `configs/experiments/hgnn_hn_sibling.yaml`
- `configs/experiments/hgnn_hn_top50_above_gold.yaml`
- `configs/experiments/hgnn_hn_mixed.yaml`

注意：这些配置先作为独立实验占位；在 trainer 传入 candidate pools 之前，ontology-aware 组不能解释为真正的 ontology-aware hard negative。
