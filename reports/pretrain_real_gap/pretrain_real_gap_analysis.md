# Pretrain vs Real Dataset Gap Analysis

- generated_at: `2026-04-27T15:41:07`
- original exact 是主结果；aligned exact 是诊断 / 消融 / 数据标准化结果；relaxed / any-label 是 supplementary，不能替代 baseline。

## 1. 预训练数据和真实评估数据最大的差距是什么？
最大差距不是 model HPO index 覆盖不足，而是真实集的 HPO 分布、case 级 HPO 稀疏/噪声、多标签口径和 candidate recall/ranking 分解不同。所有真实集的 `real_hpo_in_model_index_ratio` 基本为 1.0，但 `real_hpo_in_pretrain_ratio` 在 mimic、DDD、RAMEDIS 明显偏低。

| dataset | real_hpo_in_pretrain_ratio | real_hpo_not_in_pretrain_count | real_hpo_in_model_index_ratio | low_information_hpo_ratio_depth_le_2 |
| --- | --- | --- | --- | --- |
| mimic_test_recleaned_mondo_hpo_rows | 0.6053333333333333 | 592 | 1.0 | 0.0053333333333333 |
| DDD | 0.6886920980926431 | 914 | 1.0 | 0.00783378746594 |
| RAMEDIS | 0.6952662721893491 | 103 | 1.0 | 0.0088757396449704 |
| HMS | 0.7681159420289855 | 64 | 1.0 | 0.0 |
| LIRICAL | 0.8547169811320755 | 77 | 1.0 | 0.0075471698113207 |
| MME | 0.8571428571428571 | 11 | 1.0 | 0.0 |
| MyGene2 | 0.8677685950413223 | 16 | 1.0 | 0.0165289256198347 |

## 2. 准确率下降主要来自什么？
按失败样本 bucket 看，mimic 主要是 candidate recall gap、multi-label evaluation gap 和 granularity/noise；DDD 同时有 top50 内 ranking gap 与 candidate recall gap；HMS/LIRICAL 小样本下 ranking gap 明显。MONDO label mismatch 存在但不是总体主因。

| dataset | Bucket 4: HPO noise gap | Bucket 6: multi-label evaluation gap | Bucket 7: candidate recall gap | Bucket 8: ranking gap |
| --- | --- | --- | --- | --- |
| DDD | 0.1020408163265306 | 0.0519480519480519 | 0.3580705009276438 | 0.4025974025974026 |
| HMS | 0.3333333333333333 | 0.0 | 0.2666666666666666 | 0.4 |
| LIRICAL | 0.0 | 0.0 | 0.28125 | 0.59375 |
| MME | 0.0 | 0.0 | 0.0 | 1.0 |
| MyGene2 | 0.2 | 0.0 | 0.6 | 0.0 |
| RAMEDIS | 0.08 | 0.0 | 0.18 | 0.32 |
| mimic_test_recleaned_mondo_hpo_rows | 0.0451002227171492 | 0.2332962138084632 | 0.5233853006681515 | 0.3151447661469933 |

## 3. 疾病标签 / MONDO 差异
DDD、LIRICAL、RAMEDIS、MyGene2 的 disease index 覆盖较好；HMS 有 1 个 unmapped disease；mimic 有少量 unmapped disease 和较多 obsolete MONDO mentions。seen/unseen 分层显示 unseen label 通常更难。

| dataset | real_disease_in_pretrain_ratio | real_disease_in_disease_index_ratio | unmapped_disease_count | seen_label_top1 | unseen_label_top1 |
| --- | --- | --- | --- | --- | --- |
| DDD | 0.984126984126984 | 1.0 | 0 | 0.3117408906882591 | 0.25 |
| HMS | 0.3684210526315789 | 0.9473684210526316 | 1 | 0.5 | 0.3076923076923077 |
| LIRICAL | 1.0 | 1.0 | 0 | 0.4576271186440678 | nan |
| MME | 0.75 | 1.0 | 0 | 0.7142857142857143 | 1.0 |
| MyGene2 | 0.9230769230769232 | 1.0 | 0 | 0.9032258064516128 | 0.0 |
| RAMEDIS | 0.9375 | 1.0 | 0 | 0.7788461538461539 | 0.5555555555555556 |
| mimic_test_recleaned_mondo_hpo_rows | 0.9716713881019832 | 0.9886685552407932 | 4 | 0.1579514824797843 | 0.1111111111111111 |

## 4. alignment 后准确率有没有提升？
保守 alignment 后整体只小幅变化：ALL top1 约从 0.2589 到 0.2594，top5 约从 0.4009 到 0.4014。提升主要来自 DDD/HMS/mimic 的少量 HPO canonicalization/ancestor 对齐样本；RAMEDIS top1 略降。说明默认保守 alignment 不是主要提升杠杆，但能稳定产出诊断口径。

| dataset_base | cases_aligned_normalized_label_exact | cases_original_single_label_exact | rank_le_50_aligned_normalized_label_exact | rank_le_50_original_single_label_exact | top1_aligned_normalized_label_exact | top1_original_single_label_exact | top5_aligned_normalized_label_exact | top5_original_single_label_exact | top1_delta_aligned_minus_original | top5_delta_aligned_minus_original |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 2972 | 2978 | 0.6584791386271871 | 0.6571524513096038 | 0.25942126514131897 | 0.25889858965748824 | 0.40141318977119783 | 0.40094022834116855 | 0.0005226754838307257 | 0.0004729614300292839 |
| DDD | 761 | 761 | 0.7555847568988173 | 0.7529566360052562 | 0.3114323258869908 | 0.31011826544021026 | 0.4704336399474376 | 0.47174770039421815 | 0.0013140604467805628 | -0.0013140604467805628 |
| HMS | 24 | 25 | 0.8333333333333334 | 0.84 | 0.4166666666666667 | 0.4 | 0.625 | 0.6 | 0.016666666666666663 | 0.025000000000000022 |
| LIRICAL | 59 | 59 | 0.847457627118644 | 0.847457627118644 | 0.4576271186440678 | 0.4576271186440678 | 0.5254237288135594 | 0.5254237288135594 | 0.0 | 0.0 |
| MME | 10 | 10 | 1.0 | 1.0 | 0.8 | 0.8 | 0.8 | 0.8 | 0.0 | 0.0 |
| MyGene2 | 33 | 33 | 0.9090909090909091 | 0.9090909090909091 | 0.8484848484848485 | 0.8484848484848485 | 0.9090909090909091 | 0.9090909090909091 | 0.0 | 0.0 |
| RAMEDIS | 217 | 217 | 0.9585253456221198 | 0.9585253456221198 | 0.7649769585253456 | 0.7695852534562212 | 0.8847926267281107 | 0.8847926267281107 | -0.004608294930875556 | 0.0 |
| mimic_test_recleaned_mondo_hpo_rows | 1868 | 1873 | 0.569593147751606 | 0.5686065136145222 | 0.15792291220556745 | 0.1575013347570742 | 0.29925053533190576 | 0.2984516817939135 | 0.00042157744849324885 | 0.0007988535379922501 |

## 5. 哪些 dataset 适合通过 alignment 提升？
mimic 和 DDD 最适合继续做 alignment 诊断，因为它们 HPO-in-pretrain 比例低、failure bucket 中 candidate recall / granularity / multi-label 占比高。HMS 可用于 label mapping 和小样本口径核对。LIRICAL 当前更像少数 outlier 与 top50 内排序问题。

## 6. 哪些 dataset 不适合只靠 alignment？
DDD、LIRICAL、MME 中 gold 已进入 top50 但未进 top1/top5 的样本比例较高，适合 reranker / hard negative / family-aware ranking。mimic 的 recall@50 仍低，top50 外样本不能靠 reranker 解决，需要先改善候选召回和多标签 evaluation 口径。

## 7. 如果没显著提升，真实瓶颈是什么？
真实瓶颈主要是 candidate recall 不足、top50 内排序不足、多标签病例被 exact single-label 压缩，以及病例 HPO 与 disease-HPO hyperedge 的 exact overlap 低。HPO vocabulary gap 存在，但不是因为 model index 缺失，而是 pretrain 分布和真实病例表达方式不一致。

## 8. alignment 模块是否可以加入主线 pipeline？
可以加入，但只能作为可配置 evaluation 前处理模块，默认不覆盖原始数据，不替代 original exact。建议默认启用 ID canonicalization / alt_id / replaced_by / 保守 ancestor map；强过滤、depth filter、频率 filter 默认关闭。

推荐位置：`raw real dataset -> HPO / MONDO canonicalization -> real_to_pretrain alignment -> HGNN input building -> exact evaluation -> top50 export -> rerank / post-processing`。

## 9. 论文结果使用边界
original exact 可进入论文主表。aligned exact 如果固定配置、一次性运行，可作为数据标准化消融或附表。any-label / relaxed 指标只能作为 supplementary 或 error analysis，不能替代 strict exact baseline。