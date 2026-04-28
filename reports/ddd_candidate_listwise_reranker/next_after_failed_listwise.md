# Next After Failed Listwise

A. encoder-level hard negative fine-tuning：需要。保持 encoder 架构不变，新增可开关 hard-negative sampler 和 margin/listwise/supervised contrastive loss，独立 checkpoint，validation selected，test fixed once。
B. DDD label/mapping/outlier audit：需要。检查 sibling/parent-child exact miss、MONDO 粒度、gold 是否过细或过粗、HPO overlap 异常。
C. candidate generation 上游改造：如果 expanded recall 仍不能覆盖 rank>50，则需要继续。
D. 图对比学习：仍作为 P4，不作为当前优先项。
