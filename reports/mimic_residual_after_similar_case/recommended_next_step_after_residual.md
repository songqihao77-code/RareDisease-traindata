# Recommended next step after SimilarCase-Aug residual diagnosis

## 1. 当前 mimic 正式主线结果采用哪个配置
- 当前 `outputs/mainline_full_pipeline` 中应采用 current mainline：`topk=20`, `weight=0.5`, `score_type=raw_similarity`, Top5=0.4026。
- docx frozen config `topk=10`, `weight=0.4`, Top5=0.3940 是较早配置；如果论文以当前 mainline 为准，应在方法和表格中同步更新配置。

## 2. SimilarCase-Aug 已经解决了什么
- Top1/Top3/Top5/Rank<=50 都高于 HGNN exact baseline。
- 它主要把 baseline rank 6-50 的病例推入 Top5，并把一部分 rank>50 拉回 top50。

## 3. SimilarCase-Aug 没解决什么
- final rank>50 仍有 645/1873，说明 candidate recall residual 仍明显存在。
- final rank 6-50 仍有 474/1873，说明 top50 内排序 residual 仍存在。

## 4. 是否存在 SimilarCase 误伤
- baseline rank<=5 -> final rank>5 的病例数为 53。
- baseline rank=1 -> final rank>1 的病例数为 69。
- 因此存在误伤，需要 gate / HGNN top1 protection，而不是继续无门控增加 SimilarCase 权重。

## 5. residual candidate expansion 是否有新增价值
| final_rank_gt50_cases | mondo_relation_recovered_gold_cases | hpo_hyperedge_recovered_gold_cases_top100 | hpo_hyperedge_recovered_gold_cases_top200 | similar_case_top30_recovered_gold_cases_analysis_only | similar_case_top50_recovered_gold_cases_analysis_only | validation_similar_top30_gold_cases | validation_similar_top50_gold_cases | test_analysis_only_similar_top30_gold_cases | test_analysis_only_similar_top50_gold_cases | still_unrecovered_by_any_analysis_expansion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 645 | 75 | 95 | 120 | 0 | 0 | 1809 | 1840 | 0 | 0 | 472 |
- 如果扩展只能把 gold 拉到 top100/top200，下一步需要 light reranker 才可能转化为 Top5。

## 6. gated rerank 是否值得进入 fixed test
- validation gated top5 delta vs current: 0.0075
- 是否执行 fixed test：是。
- fixed test 结果：Top1=0.2168、Top3=0.3460、Top5=0.4036、Rank<=50=0.6530。
- 相比 current mainline：Top1 +0.0075、Top3 +0.0038、Top5 +0.0010，但 Rank<=50 -0.0026。
- 因此 gated rerank 值得作为 validation-selected fixed-test 候选进入对比表；但 Top5 增益很小且召回略降，不建议未经更多 validation/bootstrap 稳定性验证就替换 current mainline。

## 7. 图对比学习是否仍不作为第一优先级
- 仍不作为第一优先级。当前 residual 同时包含 candidate recall 缺失、top50 内排序、SimilarCase 误伤、low-overlap/label noise。
- 只有在 residual candidates 能稳定召回 gold、正负对可从 train/validation 构建、low-overlap 样本可过滤或降权后，才建议图对比学习。

## 8. 下一步最推荐实验
1. P0：固定 current mainline 配置与文档口径，避免 0.3940/0.4026 混写。
2. P1：保留 validation-selected gated SimilarCase + multiview evidence rerank 作为候选，对 Top1/Top3 有小幅收益；下一步先做稳定性验证，不直接替换主线。
3. P2：对 final rank>50 做 residual-targeted candidate expansion，重点看 MONDO/HPO expansion 能否提高 recall@100/200。
4. P3：如果 expansion 提升 recall@100/200 但不能进 Top5，再做 light-train reranker。
5. P4：最后再考虑图对比学习或 hard negative training。

## 论文主表与 supplementary
- strict exact current mainline 可进入主表，前提是说明配置来自 validation-selected fixed test。
- any-label、relaxed MONDO、ancestor/sibling/synonym/replacement 命中只能 supplementary。
- test analysis-only expansion 不能作为主表提升，只能说明潜在上限和下一步方向。
