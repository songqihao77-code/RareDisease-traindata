# Similar-case validation stability

- bootstrap 次数: 100
- 选择标准: validation top5，其次 rank<=50，再其次 top1。
- frozen config `(topk=10, weight=0.4, raw_similarity)` 被选中次数: 0
- frozen config top5 improvement mean/std/95%CI: 0.0648 / 0.0061 / [0.0552, 0.0769]
- bootstrap-selected top5 improvement mean/std/95%CI: 0.0700 / 0.0059 / [0.0585, 0.0811]
- 解释: `similar_case` 的 Top5 增益在 bootstrap 中为正且 CI 不跨 0，说明 source 有效；但 frozen config 不是 bootstrap 最常胜出的参数，存在权重选择稳定性风险。
- 处理: 不根据 bootstrap 结果重跑 test，也不改 fixed-test 口径；后续如需更新权重，只能在新的 validation protocol 上重新冻结。

## selection counts
| similar_case_topk | similar_case_weight | score_type | selected_count | validation_top5 | validation_rank_le_50 | is_frozen_config |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.5 | raw_similarity | 69 | 0.7409133271202236 | 0.8946877912395154 | 0 |
| 20 | 0.5 | raw_similarity | 20 | 0.739049394221808 | 0.9012115563839702 | 0 |
| 5 | 0.5 | raw_similarity | 10 | 0.739049394221808 | 0.8886300093196645 | 0 |
| 10 | 0.5 | rank_decay | 1 | 0.7362534948741846 | 0.8942218080149115 | 0 |
| 10 | 0.4 | raw_similarity | 0 | 0.7362534948741846 | 0.8946877912395154 | 1 |
| 3 | 0.5 | raw_similarity | 0 | 0.7343895619757689 | 0.8802423112767941 | 0 |
| 5 | 0.5 | rank_decay | 0 | 0.7343895619757689 | 0.8886300093196645 | 0 |
| 20 | 0.5 | rank_decay | 0 | 0.7343895619757689 | 0.8956197576887233 | 0 |
| 5 | 0.4 | raw_similarity | 0 | 0.733923578751165 | 0.8886300093196645 | 0 |
| 20 | 0.4 | raw_similarity | 0 | 0.7329916123019571 | 0.9016775396085741 | 0 |
| 10 | 0.4 | rank_decay | 0 | 0.7315936626281454 | 0.8928238583410997 | 0 |
| 20 | 0.4 | rank_decay | 0 | 0.7311276794035415 | 0.8937558247903076 | 0 |
| 3 | 0.5 | rank_decay | 0 | 0.7301957129543336 | 0.8802423112767941 | 0 |
| 5 | 0.4 | rank_decay | 0 | 0.7301957129543336 | 0.8886300093196645 | 0 |
| 3 | 0.4 | raw_similarity | 0 | 0.7297297297297297 | 0.8802423112767941 | 0 |
| 3 | 0.4 | rank_decay | 0 | 0.7260018639328985 | 0.8802423112767941 | 0 |
| 5 | 0.3 | raw_similarity | 0 | 0.7241379310344828 | 0.8886300093196645 | 0 |
| 10 | 0.3 | raw_similarity | 0 | 0.7236719478098789 | 0.8946877912395154 | 0 |
| 10 | 0.3 | rank_decay | 0 | 0.7222739981360671 | 0.8923578751164958 | 0 |
| 20 | 0.3 | raw_similarity | 0 | 0.7222739981360671 | 0.8998136067101584 | 0 |
