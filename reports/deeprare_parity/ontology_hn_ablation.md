# Ontology Hard Negative Ablation

本轮只实现了可配置入口，没有启动 HGNN 训练，因此没有可报告的 test 指标。

已新增的 negative 类型：

| Type | Status | Notes |
| --- | --- | --- |
| HN-current | implemented | 保留原始 score-based hard negative。 |
| HN-overlap | interface-ready | 需要训练批次提供 overlap candidate pool。 |
| HN-sibling | interface-ready | 需要 MONDO same parent / sibling pool。 |
| HN-shared-ancestor | interface-ready | 需要 shared ancestor pool。 |
| HN-above-gold | interface-ready | 需要 train/validation top50 中排在 gold 前面的候选。 |
| HN-mixed | configured | `configs/train_finetune_ontology_hn.yaml` 已配置 current/overlap/sibling/shared_ancestor/above_gold 比例。 |

协议说明：该实验会训练新的 HGNN finetune checkpoint，必须输出到独立目录 `outputs/ontology_hn/mixed`，不能覆盖当前 exact baseline 或当前最好模型。
