# DDD Hard Negative Error Profile

- DDD num_cases: 761
- current Top1/Top3/Top5/Rank<=50: 0.3758/0.4967/0.5401/0.7438
- rank=1: 286
- rank 2-3: 92
- rank 4-5: 33
- rank 6-20: 112
- rank 21-50: 43
- rank>50: 195
- Top1 target gap: 80 cases
- Top3 target gap: 79 cases
- Top5 target gap: 69 cases
- gold in top50 but rank>5: 155
- gold not in top50: 195

## 诊断结论
- 主要是 top50 内排序问题，但仍存在一部分候选召回问题。
- Rank<=50 = 0.7438，说明 hard negative training 合理，尤其适合处理 gold 已在 top50 但排在错误近邻之后的病例。
- rank>50 有 195 例，后续仍需要 candidate expansion；但当前最大可恢复池主要来自 rank 6-50 和 rank 2-5 的局部排序。

## Rank Bucket
| bucket     | count | rate   |
| ---------- | ----- | ------ |
| rank=1     | 286   | 0.3758 |
| rank 2-3   | 92    | 0.1209 |
| rank 4-5   | 33    | 0.0434 |
| rank 6-20  | 112   | 0.1472 |
| rank 21-50 | 43    | 0.0565 |
| rank>50    | 195   | 0.2562 |
