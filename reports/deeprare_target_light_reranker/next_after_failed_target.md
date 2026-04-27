# Next After Failed Target

- MIMIC：如果 Top1 < 0.29 或 Top3 < 0.37，需要更强 top1-oriented pairwise/listwise reranker，并保留 Top5 >= 0.39 约束。
- DDD：如果 Top1/Top3/Top5 仍未达到 0.48/0.60/0.63，应进入 ontology-aware hard negative training。
- LIRICAL：优先做 outlier/mapping 修复，不建议盲目训练。
- HMS：只做 high-variance 附表观察。
- 图对比学习仍为 P4，继续后置。
