# Candidate Recall Boost Summary

- generated_at: `2026-04-27T16:35:00`
- protocol: validation-selected weights, fixed test evaluation once.
- role: `candidate_recall_boost` / `evidence_fusion`; does not replace baseline exact evaluation.

## 指标对比
| dataset | metric_source | num_cases | top1 | top5 | recall_at_50 | median_rank | mrr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DDD | original_hgnn_topk | 761 | 0.31011826544021026 | 0.47174770039421815 | 0.7529566360052562 | 5.0 | 0.39578642888136306 |
| mimic_test_recleaned_mondo_hpo_rows | original_hgnn_topk | 1873 | 0.1575013347570742 | 0.2984516817939135 | 0.5686065136145222 | 21.0 | 0.2261455322595684 |
| ALL | original_hgnn_topk | 2978 | 0.25889858965748824 | 0.40094022834116855 | 0.6571524513096038 | 9.0 | 0.32949727413399393 |
| DDD | candidate_recall_boost | 761 | 0.36530880420499345 | 0.5321944809461235 | 0.7897503285151117 | 3.0 | 0.4504771067528193 |
| mimic_test_recleaned_mondo_hpo_rows | candidate_recall_boost | 1873 | 0.1388147357180993 | 0.2813667912439936 | 0.5670048051254671 | 20.0 | 0.2109666929021012 |
| ALL | candidate_recall_boost | 2978 | 0.24680993955674949 | 0.40530557421087976 | 0.6662189388851578 | 8.0 | 0.3250325977229261 |

## 关键问题回答
1. candidate recall boost 是否提高 mimic Recall@50：没有。mimic original=0.5686065136145222, boosted=0.5670048051254671, delta=-0.001602；newly_recalled=41，但也有部分原 top50 case 被挤出。
2. 是否提高 DDD Recall@50：提高。DDD original=0.7529566360052562, boosted=0.7897503285151117, outside_to_top50=33。
3. 是否牺牲 top1/top5：整体 top1 从 0.25889858965748824 降到 0.24680993955674949，整体 top5 从 0.40094022834116855 升到 0.40530557421087976；mimic top1/top5 下降，DDD top1/top5 上升。
4. 哪个 feature 最有效：validation 选择的非 HGNN 最大权重是 `w_exact`；最佳融合权重为 `{'w_hgnn': 1.0, 'w_exact': 0.2, 'w_ic': 0.1, 'w_semantic': 0.0, 'w_pretrain': 0.0, 'w_prior': 0.0}`。
5. mimic 的主要收益来源：从权重看是 exact overlap 与 IC overlap；semantic 和 pretrain-known 在 validation 选择中权重为 0。mimic test 上总体 Recall@50 未提升，说明该 evidence fusion 对 mimic 的召回收益不足。
6. DDD 的收益主要来自 recall 还是 rank：两者都有。outside_to_top50=33 表示 recall 提升；inside_top50_rank_up=220、rank_down=78 表示 top50 内排序也被改变。
7. 这个模块能否加入主线 pipeline：可以作为可选 candidate recall stage 加入，但不应替代 baseline exact，也不应和 top50 reranker 合并。
8. 如果加入，位置应为：raw real dataset -> alignment -> HGNN scoring -> candidate recall boost over top500/top1000 -> export boosted top50 -> validation-selected top50 reranker -> final evaluation。
9. 哪些结果可以进论文主表：只有 protocol 固定、validation-selected、fixed-test 的 strict exact 指标可以作为候选主表结果；本轮若以 ALL 为目标，需说明它提升 DDD/ALL Recall@50 但损害 mimic top1/top5。
10. 哪些只能作为 supplementary / error analysis：any-label、多标签命中、newly_recalled_cases、still_missed_cases、DDD rank movement、feature audit 都只能作为 supplementary 或 error analysis。

## Feature / Weight Analysis
| feature | w_hgnn | w_exact | w_ic | w_semantic | w_pretrain | w_prior | dataset | metric_source | num_cases | top1 | top3 | top5 | recall_at_50 | median_rank | mean_rank | mrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected_fusion | 1.0 | 0.2 | 0.1 | 0.0 | 0.0 | 0.0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| hgnn_only | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ALL | candidate_recall_boost | 983.0 | 0.2268565615462869 | 0.335707019328586 | 0.3916581892166836 | 0.6439471007121058 | nan | 55.39653179190751 | 0.3058106004390341 |
| exact_only | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | ALL | candidate_recall_boost | 983.0 | 0.1312309257375381 | 0.1871820956256358 | 0.2115971515768057 | 0.4435401831129196 | nan | 96.04508670520232 | 0.1807151200162332 |
| ic_only | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | ALL | candidate_recall_boost | 983.0 | 0.1078331637843336 | 0.1495422177009155 | 0.17293997965412 | 0.3967446592065107 | nan | 107.57919075144508 | 0.1490933018981117 |
| semantic_only | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | ALL | candidate_recall_boost | 983.0 | 0.0111902339776195 | 0.0335707019328586 | 0.0579857578840284 | 0.3041709053916582 | nan | 129.66473988439307 | 0.0449585385999981 |
| pretrain_known_only | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | ALL | candidate_recall_boost | 983.0 | 0.1332655137334689 | 0.1871820956256358 | 0.2146490335707019 | 0.4425228891149542 | nan | 95.93179190751444 | 0.1820711665459682 |