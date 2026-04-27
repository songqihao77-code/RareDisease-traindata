# DDD Hard Negative Recommendation

- top1 错误但 gold 在 top50 的 near-miss cases: 337
- top50 中排在 gold 前面的 negative 平均数: 11.51

## Top1 Wrong Relation Distribution
| relation | num_cases |
| --- | --- |
| shared_ancestor | 183 |
| same_parent | 101 |
| candidate_descendant_of_gold | 21 |
| candidate_ancestor_of_gold | 16 |
| unrelated_or_unknown | 12 |
| synonym_or_name_match | 4 |

## 推荐 negative 类型
- `top50-above-gold negative`: 优先级最高，直接对应当前 DDD gold 被压在 top50 内的问题。
- `high HPO-overlap negative`: 适合提升 top1/top3 排序，但需要避免把语义等价或合理鉴别诊断当成强负例。
- `MONDO sibling/same-parent negative`: DDD 中常见同父类混淆，适合做 margin 较小的 hard negative。
- `current top-k negative`: 保留为基础组，作为 HN-current 对照。
- `mixed hard negative`: 论文实验最完整，但必须先在训练 loop 中真正构造并传入 ontology/overlap/top50 candidate pools。

## False Negative 风险
same-parent、sibling、ancestor-descendant 候选可能是临床上合理的近似诊断；建议降低 margin 或使用 soft label/低权重，不建议无差别强惩罚。
