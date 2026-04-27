# Next After Failed DDD Target

- candidate expansion：需要。当前 rank>50 仍有大量 case，单靠 top50 内排序无法补齐所有 Top5 gap。
- listwise reranker：需要。当前 pairwise hard-negative head 仍可能牺牲部分已靠前病例，应做 listwise/top-k constrained objective。
- encoder-level hard negative fine-tuning：需要作为下一阶段，但必须保持 encoder 架构不变，只新增可开关 sampler/loss，并输出到独立 checkpoint。
- label/mapping/outlier audit：需要，尤其检查 DDD sibling/parent-child 近邻是否存在标注粒度问题。
- 图对比学习：仍后置，等 candidate expansion 和 encoder-level HN fine-tuning 后再评估。
