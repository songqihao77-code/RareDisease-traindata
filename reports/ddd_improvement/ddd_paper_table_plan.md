# DDD Paper Table Plan

## 1. Main Table Recommendation

主表推荐只放以下方法：

| Method | Role | DDD Top1 | DDD Top3 | DDD Top5 | DDD Recall@50 | Notes |
|---|---|---:|---:|---:|---:|---|
| HGNN baseline | baseline | 0.3022 | 0.4442 | 0.4967 | 0.7451 | 原始 exact HGNN baseline。 |
| HGNN + validation-selected grid rerank | primary DDD result | 0.3693 | 0.4875 | 0.5506 | 0.7451 | 当前最强且协议合规的 DDD no-train 主结果。 |
| ontology-aware HN exact | optional trained ablation | 0.3154 | 0.4389 | 0.4954 | 0.7411 | 只能标注为 trained ablation，不作为最佳主线。 |

若主表空间有限，建议不放 HN exact，只在 supplementary 中报告。若主表放 HN，必须明确它不是最佳方法，也不能声称全面提升。

## 2. Methods Not Allowed In Main Table

以下结果不应进入主表：

- test-side exploratory grid/gate。
- HN dry-run。
- 任何使用 test set 选权重、调 gate 或调 HN 配置的结果。
- 未经 validation 选择的 exploratory rerank。

`grid_1720` 可作为 exploratory upper bound，但必须明确它是 test-side upper bound，不能作为正式方法。

## 3. Supplementary Table Suggestions

附表建议包含：

- HN failure analysis：说明 ontology-aware HN exact 只提升 top1，未改善 top3/top5/recall@50。
- HN candidate pool dry-run：展示 `HN-mixed` composition 和 fallback_rate。
- DDD near-miss ontology relation distribution：分析 top1 错误是否集中在 parent/sibling/ancestor relation。
- top50 rank decomposition：展示 DDD 主要提升空间来自 top50 内排序。
- validation-selected weights：记录 grid/gated 权重、selection objective、validation 指标和 fixed test 指标。
- exploratory upper bound：报告 `grid_1720`，但只标注为 upper bound / error analysis。

## 4. Recommended Narrative

DDD 低准确率主要来自 top50 内排序，而不是单纯 candidate recall。HGNN baseline 的 DDD recall@50 为 `0.7451`，说明相当多 gold 已在 top50 内；validation-selected evidence rerank 在不改变 recall@50 的情况下，将 DDD top1 从 `0.3022` 提升到 `0.3693`，top5 从 `0.4967` 提升到 `0.5506`。

no-train evidence rerank 的收益更稳定。它使用固定 validation 选择协议，不修改 HGNN encoder，也不改变 exact evaluation，只对 top50 内候选进行可解释重排。

direct ontology-aware HN fine-tuning 没有稳定改善整体 ranking。HN exact 只将 DDD top1 提到 `0.3154`，但 top3/top5/recall@50 均略低于 baseline，因此更适合作为 negative ablation，而不是主线方法。

后续更适合轻量 pairwise/listwise reranker，而不是直接改 HGNN encoder。pairwise/listwise reranker 可以把问题限定在 top50 内排序，并复用 evidence features 与 validation selection protocol。
