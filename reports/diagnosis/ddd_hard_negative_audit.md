# DDD Hard Negative Audit

## 当前状态
- `src/training/hard_negative_miner.py::mine_hard_negatives` 实现当前 batch score-based top-k false candidates。
- `src/training/hard_negative_miner.py::mine_configurable_hard_negatives` 声明了 `HN-overlap`、`HN-sibling`、`HN-shared-ancestor`、`HN-above-gold`、`HN-mixed` 接口。
- `configs/train_finetune_attn_idf_main.yaml` 当前启用 hard negative，但未指定 ontology strategy，因此是 `HN-current`。
- `configs/train_finetune_ontology_hn.yaml` 配置了 `HN-mixed`，但本轮未训练。

## 关键风险 / 可能 bug
- `src/training/trainer.py::run_one_epoch` 调用 `mine_configurable_hard_negatives(...)` 时没有传入 `candidate_pools`。按当前实现，只要 `candidate_pools` 为空，`HN-overlap/sibling/shared_ancestor/above_gold/mixed` 都会退化为 `HN-current`。因此 ontology-aware hard negative 目前是接口就绪，但训练热路径未真正接入候选池。

## 当前负样本来源
| Source | Status | Evidence |
|---|---|---|
| top-k false candidates by current scores | 已实现 | `mine_hard_negatives(scores, targets, k)` |
| random negative | 未见专门实现 | full-pool CE 自然包含所有非 gold 类，但 hard loss 不随机采样 |
| same ontology parent / sibling | 接口存在，热路径未供池 | `_pool_for_strategy` 支持 `sibling/same_parent` |
| high HPO-overlap disease | 接口存在，热路径未供池 | `_pool_for_strategy` 支持 `overlap` |
| above-gold top50 candidate | 接口存在，热路径未供池 | `_pool_for_strategy` 支持 `above_gold` |

## DDD 专用推荐方案
1. 从 DDD/train-validation top50 中提取排在 gold 前面的 candidates，构建 `above_gold` pool。
2. 基于 MONDO parent/ancestor 构建 same_parent、sibling、shared_ancestor pools。
3. 基于 disease-HPO exact/IC overlap 构建 high-overlap pool。
4. 基于 query HPO overlap 构建 case-specific high-overlap but non-gold pool。
5. 将本报告的 DDD near-miss pair/relation 分布作为采样权重先验。
6. 所有训练必须输出到独立目录，不能覆盖当前 exact baseline。