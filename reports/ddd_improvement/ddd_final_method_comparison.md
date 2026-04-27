# DDD Final Method Comparison

## 1. Method Conclusion

当前最优 DDD 方法是 `validation-selected grid rerank (DDD_top1)`。它不训练 HGNN，只在 validation candidates 上选择 evidence rerank 权重，再对 test candidates 做一次 fixed evaluation；DDD top1/top3/top5 达到 `0.3693/0.4875/0.5506`，明显优于 HGNN baseline 和 ontology-aware HN exact training。

ontology-aware HN exact training 没有成为成功主线。它将 DDD top1 从 `0.3022` 提到 `0.3154`，但 DDD top3、top5 和 recall@50 分别降到 `0.4389/0.4954/0.7411`，整体排序质量没有改善。

HN 不建议进入论文主表作为最佳方法；可以作为 trained ablation 或 supplementary negative result。后续不建议继续用 test 调参，也不应重复 test evaluation。若继续探索，应先暂停 HN 正式训练，转向 validation-only lighter HN ablation 或 top50 pairwise/listwise reranker。

最终推荐把 `validation-selected grid rerank (DDD_top1)` 作为 DDD 最终主线。

## 2. Final Comparison Table

| Method | Training? | Selection Protocol | DDD Top1 | DDD Top3 | DDD Top5 | DDD Recall@50 | Overall Top1 | mimic Top1 | Paper Role | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| HGNN baseline | No | Exact HGNN baseline / `A_hgnn_only` | 0.3022 | 0.4442 | 0.4967 | 0.7451 | 0.2794 | 0.1917 | Main table baseline | Original HGNN top50 exact baseline. |
| validation-selected grid rerank (DDD_top1) | No | Validation candidates select weights; test fixed once | 0.3693 | 0.4875 | 0.5506 | 0.7451 | 0.2616 | 0.1639 | Recommended main DDD result | Current strongest paper-ready DDD no-train result. |
| validation-selected gated rerank (DDD_top1) | No | Validation candidates select bounded gate; test fixed once | 0.3693 | 0.4836 | 0.5453 | 0.7451 | 0.2693 | 0.1666 | Candidate / supplementary main-table variant | Same DDD top1 as grid, slightly lower DDD top3/top5. |
| ontology-aware HN exact | Yes | Fixed train config; one fixed test evaluation after training | 0.3154 | 0.4389 | 0.4954 | 0.7411 | 0.2307 | 0.1148 | Trained ablation / supplementary negative result | Small DDD top1 gain, but ranking quality declines. |
| test-side exploratory upper bound (`grid_1720`) | No | Test-side exploratory grid search only | 0.3784 | 0.4888 | 0.5532 | 0.7451 | 0.2639 | 0.1644 | Supplementary upper bound only | Cannot enter main table because it used test-side selection. |

## 3. Input Files Read

- `reports/ddd_improvement/ddd_next_stage_execution_report.md`
- `reports/ddd_improvement/ddd_validation_selected_rerank_report.md`
- `outputs/rerank/ddd_rerank_fixed_test_metrics.csv`
- `outputs/rerank/ddd_rerank_fixed_test_by_dataset.csv`
- `reports/ddd_improvement/ddd_hn_candidate_pool_audit.md`
- `reports/ddd_improvement/ddd_hn_candidate_pool_stats.csv`
- `outputs/ddd_ontology_hn/logs/history_20260427_091719.csv`
- `outputs/ddd_ontology_hn/logs/history_20260427_091719.json`
- `outputs/ddd_ontology_hn/evaluation/best_20260427_093516_per_dataset.csv`
- `outputs/ddd_ontology_hn/evaluation/best_20260427_093516_details.csv`
- `outputs/ddd_ontology_hn/evaluation/best_20260427_093516_metrics.csv`: 未找到

## 4. Direct Answers

1. 当前最优 DDD 方法是 `validation-selected grid rerank (DDD_top1)`。
2. ontology-aware HN exact training 没有成功成为 DDD 主线。
3. HN 不值得作为最佳方法进入论文主表。
4. HN 适合作为 supplementary negative result / trained ablation。
5. 后续应暂停 HN test-side 实验；如果继续，只做 validation-only lighter HN ablation。
6. 推荐把 validation-selected rerank 作为最终 DDD 主线。
