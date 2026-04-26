# Final Recommendation

## 1. 当前是否已经达到 DeepRare target

- 已全部达到的 dataset: RAMEDIS, MME, MyGene2
- 接近的 dataset: 无
- 明显不足的 dataset: DDD, mimic_test, LIRICAL, HMS

## 2. 数据集分层结论

- `MME`、`MyGene2`、`RAMEDIS` 当前 exact baseline 已达到或超过 DeepRare target，但除 mimic_test 外均存在 paper case 数与当前 split 不一致的问题，主表必须注明协议。
- `LIRICAL` 的 top5 已非常接近 0.68，但当前只有 59 cases，论文为 370 cases；先处理 split parity，再判断模型差距。
- `DDD` top1/top3/top5 均低于 DeepRare target，但 rank<=50=0.7451，说明 top50 reranker 和 hard negatives 仍有空间。
- `mimic_test` top50 upper bound 为 0.6151，且 rank>50 样本 overlap_zero 偏高；candidate augmentation 比单纯 top50 rerank 更关键。
- `HMS` 当前只有 25 cases，论文为 88 cases；当前结果不适合直接和 DeepRare 主表硬对齐。

## 3. 每个不足数据集的最有效路径

- `DDD`: validation-selected rerank 可小幅提升；下一步应做 pairwise reranker 和 ontology-aware hard negative training。
- `mimic_test`: 优先做 multi-label/any-label supplementary、synonym/obsolete mapping 检查、similar-case candidate recovery。
- `HMS`: 先补齐或确认 evaluation split，再做 reranker。
- `LIRICAL`: 当前 top5 接近目标，重点排查 outlier 和 split case 数。

## 4. 可进入论文主表

- 原始 exact HGNN baseline。
- validation-selected fixed test rerank；本轮可报告的是 linear grid selected 权重，完整 gated/mimic-safe gate 未完成。
- `HGNN_AUG` 的 exact metric 只有在 source weights 由 validation 选择后才可进入主表；本轮默认权重结果只能作为 exploratory。

## 5. 只能放附表或 error analysis

- test-side exploratory rerank/grid/gate upper bound。
- mimic_test any-label hit、multi-label audit。
- synonym/parent-child/obsolete relaxed evaluation。
- API/LLM/agentic source 的 availability 和 relaxed 命中。

## 6. 是否建议继续 hard negative training

建议继续，但必须作为独立训练目录。DDD near-miss 中 same-parent/shared-ancestor 候选很多，ontology-aware hard negatives 可能提升 top1/top3；不要覆盖当前最好模型。

## 7. 是否建议引入 agentic candidate augmentation

建议作为单独方法 `HGNN_AUG`。mimic_test 的瓶颈包含 candidate recall 和 phenotype coverage，单纯 top50 rerank 上限有限；augmentation 必须 validation 选择 source weights，test fixed eval。

## 8. 下一轮最小实验集合

1. 重新确认 HMS/LIRICAL/RAMEDIS/MME/MyGene2 是否能按论文 case 数重建 evaluation split。
2. 完成 gated/mimic-safe validation selection，保存固定权重后只跑一次 test。
3. 导出 train top50 candidates，训练轻量 pairwise reranker，并用 validation 选择超参。
4. 对 DDD 运行 ontology-aware hard negative 独立训练，观察 Top1/Top3/Top5。
5. 对 mimic_test rank>50 样本运行 validation-selected `HGNN_AUG` candidate recovery。
