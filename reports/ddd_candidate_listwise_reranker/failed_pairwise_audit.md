# Failed Pairwise Head Audit

- top1 gained/lost: 14/22
- top3 gained/lost: 8/20
- top5 gained/lost: 8/23
- baseline/current rank 1-5 被推坏病例数: 68
- rank 6-20 被推好病例数: 23
- rank 21-50 被推好病例数: 16

## 误伤类型
| hard_negative_like_type | worsened_case_count |
| ----------------------- | ------------------- |
| high_hpo_overlap        | 141                 |
| hyperedge_similar       | 141                 |
| random                  | 141                 |
| similar_case_false      | 141                 |
| top50_above_gold        | 135                 |
| same_parent_sibling     | 123                 |

## 结论
- pairwise head 能改善少量病例，但 lost cases 多于 gained cases，说明普通 pairwise objective 会误伤已靠前病例。
- sibling / same-parent 近邻可能存在 label granularity 问题，不能无差别强惩罚。
- frozen features 对 DDD 仍不足，尤其无法稳定保护 current top-k。
- 下一步应使用 listwise/top-k-aware objective，并加入 current-score anchoring 与 top-k protection。
