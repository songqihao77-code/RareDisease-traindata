# 各数据集提升策略

所有建议均以 strict exact Recall@1/@3/@5 作为主结果。any-label、relaxed MONDO、ancestor/sibling/synonym/replacement 分析只能作为 supplementary。

## MIMIC-IV-Rare

- Current vs target: Top1 0.2093/0.2900 (case gap 152), Top3 0.3422/0.3700 (case gap 53), Top5 0.4026/0.3900 (case gap 0), Rank<=50 0.6556
- Rank bucket: rank=1 392, 2-3 249, 4-5 113, 6-20 215, 21-50 259, >50 645.
- 策略：Top5 已达到 DeepRare target，重点提升 Top1/Top3。优先做 light-train reranker、pairwise/listwise reranker、HGNN top1 protection、gated SimilarCase。
- 不优先继续无门控扩大 SimilarCase topk；图对比学习继续后置。

## DDD

- Current vs target: Top1 0.3758/0.4800 (case gap 80), Top3 0.4967/0.6000 (case gap 79), Top5 0.5401/0.6300 (case gap 69), Rank<=50 0.7438
- Rank bucket: rank=1 286, 2-3 92, 4-5 33, 6-20 112, 21-50 43, >50 195.
- 策略：DDD 三个 target 均未达到，但 Rank<=50 明显高于 Top5，说明很多 gold 已进入 top50 但排序不够靠前。
- 优先 ontology-aware hard negative training 与 pairwise/listwise reranking；负样本包括 same-parent、sibling、高 HPO-overlap、top50-above-gold。

## RareBench-HMS

- Current vs target: Top1 0.3200/0.5700 (case gap 7), Top3 0.4400/0.6500 (case gap 6), Top5 0.4800/0.7100 (case gap 6), Rank<=50 0.8800
- Rank bucket: rank=1 8, 2-3 3, 4-5 1, 6-20 8, 21-50 2, >50 3.
- 策略：项目测试集只有 25 例，标注 high_variance，不建议作为唯一主结论。
- 如果要提升，优先查 label/mapping 和 top50 miss；如增加 relaxed 分析，必须与 exact 主结果分开报告。

## RareBench-LIRICAL

- Current vs target: Top1 0.5085/0.5600 (case gap 4), Top3 0.6441/0.6500 (case gap 1), Top5 0.6780/0.6800 (case gap 1), Rank<=50 0.8475
- Rank bucket: rank=1 30, 2-3 8, 4-5 2, 6-20 6, 21-50 4, >50 9.
- 策略：Top3/Top5 只差约 1 case，属于 near-miss/high_variance，不应按大缺口处理。
- 优先 outlier audit、mapping audit、局部 rerank；不要盲目训练。

## RareBench-RAMEDIS

- Current vs target: Top1 0.7880/0.7300 (case gap 0), Top3 0.8756/0.8300 (case gap 0), Top5 0.9309/0.8500 (case gap 0), Rank<=50 0.9862
- Rank bucket: rank=1 171, 2-3 19, 4-5 12, 6-20 9, 21-50 3, >50 3.
- 策略：current mainline 已达到或超过 DeepRare targets，保持主线。
- 若后续追求 Top1 polish，可做 top5-to-top1 reranker；若未来 Top5 下降，先做 candidate recall audit。

## MyGene

- Current vs target: Top1 0.8788/0.7600 (case gap 0), Top3 0.8788/0.8000 (case gap 0), Top5 0.8788/0.8100 (case gap 0), Rank<=50 0.9697
- Rank bucket: rank=1 29, 2-3 0, 4-5 0, 6-20 2, 21-50 1, >50 1.
- 策略：current mainline 已达到或超过 DeepRare targets，保持主线。
- 若后续追求 Top1 polish，可做 top5-to-top1 reranker；若未来 Top5 下降，先做 candidate recall audit。

## RareBench-MME

- Current vs target: Top1 0.9000/0.7800 (case gap 0), Top3 0.9000/0.8500 (case gap 0), Top5 0.9000/0.9000 (case gap 0), Rank<=50 0.9000
- Rank bucket: rank=1 9, 2-3 0, 4-5 0, 6-20 0, 21-50 0, >50 1.
- 策略：当前 MME 已达到 DeepRare targets，但只有 10 例，必须标注 high_variance。
- 不要根据该数据集单独决定整体训练策略。

## Xinhua Hosp.

- Current vs target: unavailable / not comparable
- 策略：当前项目中 unavailable / not comparable。
- 除非新增并记录明确对应的数据集，否则不要把其他项目数据集硬映射成 Xinhua Hosp.。
