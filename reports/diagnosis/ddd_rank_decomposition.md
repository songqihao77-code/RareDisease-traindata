# DDD Rank Decomposition

- 数据源: `D:\RareDisease-traindata\outputs\rerank\top50_candidates_v2.csv`；full rank 参考 `D:\RareDisease-traindata\outputs\attn_beta_sweep\edge_log_beta02\evaluation\best_20260425_224439_details.csv`。
- 总样本数: 761。
- top50-capped median/mean: 6.0/18.8463；full-rank median/mean: 6.0/271.9304。
- rank>50: 194 (25.49%)；top50 但 rank>5: 189 (24.84%)；top5 但非 top1: 148 (19.45%)。

|bucket|num_cases|ratio|interpretation|
|---|---|---|---|
|rank = 1|230|0.3022|HGNN 已精确命中，非当前主要提升空间|
|rank <= 3|338|0.4442|top3 累计命中|
|rank <= 5|378|0.4967|top5 累计命中|
|rank <= 10|438|0.5756|少量重排即可进入可用诊断列表|
|rank <= 20|496|0.6518|top50 内排序提升的中短尾空间|
|rank <= 50|567|0.7451|candidate recall@50，上限为 top50 内 rerank 可达样本|
|rank > 50|194|0.2549|HGNN top50 candidate recall 未覆盖|
|gold absent from candidate universe|0|0.0000|疾病索引/候选全集缺失|
|top50 but rank > 5|189|0.2484|核心 reranker 目标|
|top5 but not top1|148|0.1945|top1 排序损失，适合 evidence rerank|

判断: DDD 的首要瓶颈是 top50 内排序问题，次要瓶颈是 candidate recall。证据是 567/761 个 gold 已在 top50，但只有 230/761 个排到 top1；另有 189 个样本位于 top50 但排在 top5 之后。