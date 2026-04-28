# Recommended Paper Table

1. listwise reranker 是否能作为 DDD 新主线：否。
2. 是否达到 DeepRare DDD target：否。
3. 是否可以进入论文主表：不建议。
4. failed pairwise hard-negative head 是否作为负结果附表：是。
5. 是否仍需要 encoder-level hard negative fine-tuning：是，如果 listwise 未达标或 validation 未通过。
6. 是否需要 label/mapping/outlier audit：是。
7. 图对比学习是否仍后置：是。
8. MIMIC 是否仍需要单独 top1-oriented listwise reranker：是。
