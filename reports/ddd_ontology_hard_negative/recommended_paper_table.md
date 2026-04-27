# Recommended Paper Table

1. hard negative model 是否能作为 DDD 新主线：暂不建议作为 DDD 新主线。
2. 是否达到 DeepRare DDD target：否。
3. 是否牺牲 MIMIC / MME / RAMEDIS / MyGene2：否；本模型是 DDD-specific，只对 DDD 应用，其他数据集保持 current mainline。
4. 如果是 DDD-specific model，是否只能作为 dataset-specific enhancement：是，不能混写为 ALL general model。
5. light reranker 是否仍作为负结果附表：是。
6. 图对比学习是否仍后置：是。
7. 下一步是否需要 MIMIC top1-oriented listwise reranker：是，MIMIC 仍需要单独路线。
