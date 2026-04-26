# Top5 gain source analysis

- rank>50 recovered to top50 = 79。
- recovered to top5 = 0。
- 因此 Top5 提升主要不是来自 rank>50 病例直接进入 top5，而是原本 gold 已在候选集内但 rank>5 的病例被 `similar_case` 重排进 top5。
- near-miss top5 cases: 180。
- 其中带 similar_case evidence 的 rank 6-20 cases: 31。
- 其余 149 个 rank 6-20 case 更可能需要 reranker 或新 source，而不是单纯调 similar_case 权重。