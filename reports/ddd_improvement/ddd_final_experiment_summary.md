# DDD Final Experiment Summary

## 1. Executive Summary

最优方法是 `validation-selected grid rerank (DDD_top1)`。它将 DDD top1/top3/top5 从 HGNN baseline 的 `0.3022/0.4442/0.4967` 提升到 `0.3693/0.4875/0.5506`，且 DDD recall@50 保持 `0.7451`。

ontology-aware HN exact training 没有成为主线。HN 只将 DDD top1 小幅提升到 `0.3154`，但 DDD top3/top5/recall@50 降为 `0.4389/0.4954/0.7411`。这说明 HN 没有稳定改善整体 ranking。

不建议继续用 test 调参，也不应重复 test evaluation。下一步应转向论文整理，或只做 validation-only pairwise/listwise reranker 与 lighter HN ablation。

## 2. Final Metrics

| Method | Training? | Selection Protocol | DDD Top1 | DDD Top3 | DDD Top5 | DDD Recall@50 | Overall Top1 | mimic Top1 | Paper Role |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| HGNN baseline | No | Exact baseline | 0.3022 | 0.4442 | 0.4967 | 0.7451 | 0.2794 | 0.1917 | baseline |
| validation-selected grid rerank (DDD_top1) | No | Validation selected; fixed test once | 0.3693 | 0.4875 | 0.5506 | 0.7451 | 0.2616 | 0.1639 | primary DDD result |
| validation-selected gated rerank (DDD_top1) | No | Bounded validation gate; fixed test once | 0.3693 | 0.4836 | 0.5453 | 0.7451 | 0.2693 | 0.1666 | candidate / supplementary |
| ontology-aware HN exact | Yes | Fixed config; one fixed test evaluation | 0.3154 | 0.4389 | 0.4954 | 0.7411 | 0.2307 | 0.1148 | negative trained ablation |
| test-side exploratory upper bound (`grid_1720`) | No | Test-side exploratory only | 0.3784 | 0.4888 | 0.5532 | 0.7451 | 0.2639 | 0.1644 | upper bound only |

## 3. Main Conclusion

DDD 的主要提升空间在 top50 内排序。HGNN baseline 已有 DDD recall@50 `0.7451`，而 validation-selected rerank 在不改变候选集合的情况下显著提升 top1/top3/top5。

no-train evidence rerank 是当前最稳妥的 DDD 主线。它不修改 HGNN encoder，不覆盖 exact evaluation，并且 selection protocol 可解释。

ontology-aware HN exact training 的结果不足以支持“全面提升”。它只带来 top1 小幅增益，同时损伤 top3/top5/recall@50，说明直接 fine-tuning HGNN 会改变全局候选分布，风险高于 top50 内重排。

## 4. What Goes Into Main Paper

主表建议放：

- HGNN baseline：DDD `0.3022/0.4442/0.4967/0.7451`。
- HGNN + validation-selected grid rerank：DDD `0.3693/0.4875/0.5506/0.7451`。
- 可选 ontology-aware HN exact：仅标注 trained ablation，DDD `0.3154/0.4389/0.4954/0.7411`。

## 5. What Goes Into Supplementary

附表建议放：

- HN failure analysis。
- HN candidate pool dry-run。
- DDD near-miss ontology relation distribution。
- top50 rank decomposition。
- validation-selected weights 和 gated config。
- test-side exploratory upper bound，只能标注 upper bound。

## 6. What Should Not Be Claimed

不要声称 HN 全面提升 DDD。HN 只提升 top1，top3/top5/recall@50 均未改善。

不要声称 mimic 被提升。HN exact 的 mimic top1 为 `0.1148`，低于 baseline 和 rerank。

不要把 test-side grid/gate 作为正式结果。`grid_1720` 只能作为 exploratory upper bound。

不要声称 HN + rerank 已证明有效，除非后续存在固定配置、一次性 test evaluation 的结果。

不要用 relaxed ontology analysis 覆盖 exact metric。

## 7. Recommended Next Step

当前停止 DDD test-side 实验，整理主表与附表。若继续实验，只做 validation-only pairwise/listwise reranker 或 lighter HN ablation；最终 test 只能在新方案完全固定后运行一次。

明确判断：DDD 最终主线应选择 validation-selected rerank，而不是 ontology-aware HN exact training。
