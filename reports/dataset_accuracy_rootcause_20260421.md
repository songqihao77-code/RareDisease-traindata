# 数据集准确率低原因分析

## 输入依据

- 实验汇总:
  - `D:\RareDisease-traindata\outputs\stage2_finetune_v59\evaluation\best_20260420_170732_summary.json`
  - `D:\RareDisease-traindata\outputs\stage2_finetune_v59\evaluation\best_20260420_170732_details.csv`
- 疾病超边相似度分析:
  - `D:\RareDisease-traindata\outputs\dataset_hyperedge_similarity\20260421_152849\dataset_summary.csv`
  - `D:\RareDisease-traindata\outputs\dataset_hyperedge_similarity\20260421_152849\dataset_mondo_summary.csv`
  - `D:\RareDisease-traindata\outputs\dataset_hyperedge_similarity\20260421_152849\case_level_similarity.csv`

## 按数据集汇总

| 数据集 | Top1 | Top3 | Top5 | mean_weighted_hit_ratio | mean_case_hit_ratio | 全局训练标签重叠率 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| DDD | 0.338 | 0.456 | 0.512 | 0.331 | 0.494 | 0.173 | 超边相似度最高，但大量测试标签训练中没见过 |
| HMS | 0.160 | 0.240 | 0.280 | 0.125 | 0.213 | 1.000 | 训练见过标签，但病例与超边对齐差，样本量也很小 |
| LIRICAL | 0.356 | 0.441 | 0.542 | 0.200 | 0.480 | 1.000 | 中等偏上，超边匹配尚可 |
| MME | 0.800 | 0.800 | 0.800 | 0.184 | 0.489 | 1.000 | 样本很少，但标签覆盖完整，任务相对容易 |
| MyGene2 | 0.848 | 0.848 | 0.909 | 0.124 | 0.450 | 1.000 | 虽然超边覆盖不高，但标签重复充分，准确率高 |
| RAMEDIS | 0.700 | 0.825 | 0.885 | 0.087 | 0.204 | 1.000 | 超边相似度不高，但训练支持很强，模型能记住标签模式 |
| mimic_test | 0.212 | 0.334 | 0.389 | 0.089 | 0.126 | 0.796 | 标签多数训练见过，但病例表型和超边都很疏，噪声大 |

## 关键观察

### 1. 低准确率不只有一种原因

- `DDD` 的主要问题不是超边不对，而是 **测试标签大量未见**。
  - `test_unique_labels=756`
  - `seen_in_global_train=131`
  - `global_label_overlap_ratio=0.173`
  - `unseen_labels_global=625`
- `mimic_test` 的主要问题不是未见标签，而是 **训练见过很多，仍然难分**。
  - `test_unique_labels=353`
  - `seen_in_global_train=281`
  - `global_label_overlap_ratio=0.796`
  - 但 `Top1=0.212`
- `HMS` 的主要问题是 **样本太少 + 病例与超边匹配差**。
  - `num_cases=25`
  - `mean_weighted_hit_ratio=0.125`
  - `mean_case_hit_ratio=0.213`

### 2. DDD 为什么只有中等准确率

- 从超边角度看，`DDD` 是 7 个测试数据集中和超边最接近的:
  - `mean_weighted_hit_ratio=0.331`
  - `mean_case_hit_ratio=0.494`
  - `mean_jaccard=0.239`
- 但从训练覆盖看，`DDD` 的标签极度缺失:
  - 每个测试病例对应的训练支持均值只有 `0.503`
  - `unseen_label_cases=630 / 777`
  - `unseen_label_ratio=0.811`
- 结论:
  - `DDD` 的准确率不是被“病例和超边不相似”拉低的，而是被“测试标签大量不在训练集中”硬性压低的。
  - 换句话说，模型即使表型读得对，也经常没有正确标签可学。

### 3. mimic_test 为什么准确率最低之一

- `mimic_test` 的标签并不算缺训练支持:
  - 每个测试病例平均训练支持 `131.29`
  - `low_support_case_ratio_le10=0.240`
- 但它与疾病超边的对齐非常差:
  - `mean_weighted_hit_ratio=0.089`
  - `mean_case_hit_ratio=0.126`
  - `mean_jaccard=0.047`
- 这说明：
  - 病例 HPO 只覆盖到疾病超边里很小一部分
  - 大量病例表型更像是临床噪声、并发症、住院状态，而不是疾病核心表型
- 典型难标签即使训练支持很高也表现很差:
  - `MONDO:0018905`: `test_cases=101`, `train_case_support=386`, `top1=0.188`
  - `MONDO:0016264`: `test_cases=30`, `train_case_support=100`, `top1=0.000`
  - `MONDO:0005062`: `test_cases=30`, `train_case_support=92`, `top1=0.033`
  - `MONDO:0100480`: `test_cases=32`, `train_case_support=25`, `top1=0.000`
- 结论:
  - `mimic_test` 的低准确率主要来自 **数据分布偏移 + 病例表型噪声大 + 与疾病超边核心 HPO 对不齐**，而不是简单的标签冷启动。

### 4. HMS 为什么比 mimic_test 还差

- `HMS` 标签训练里都见过:
  - `global_label_overlap_ratio=1.0`
- 但整体超边匹配依旧弱:
  - `mean_weighted_hit_ratio=0.125`
  - `mean_case_hit_ratio=0.213`
  - `mean_jaccard=0.060`
- 并且样本数只有 `25`，波动很大:
  - `top1_wrong_count=21`
  - `top3` 相比 `top1` 只提升 `0.08`
  - `top5` 再只提升 `0.04`
- 这说明：
  - 模型不是“差一点排到前面”，而是很多 HMS 病例从一开始就没有抓到正确疾病区域
  - 更像是 HMS 病例表型本身和疾病超边定义不贴，或标注过于稀疏/偏临床化

### 5. 为什么 RAMEDIS / MyGene2 准确率反而更高

- `RAMEDIS`:
  - `Top1=0.700`
  - 虽然 `mean_weighted_hit_ratio=0.087` 很低
  - 但 `mean_train_case_support_per_test_case=32.49`
- `MyGene2`:
  - `Top1=0.848`
  - `mean_weighted_hit_ratio=0.124`
  - `mean_train_case_support_per_test_case=9.70`
- 这说明当前模型并不纯粹依赖“和超边有多像”，还很依赖 **训练中是否形成了稳定的标签模式记忆**。
- 因而低准确率通常是两类因素叠加：
  - 标签没见过或见得太少
  - 即使见过，病例 HPO 与疾病超边核心结构也不对齐

## 直接回答“准确率低的原因”

### DDD

- 主因: **测试标签大规模未见**
- 次因: 少数标签即使见过，支持次数也很低
- 不是主因: 超边相似度低

### HMS

- 主因: **病例 HPO 与疾病超边对齐差**
- 次因: **样本量太小，估计不稳定**
- 不是主因: 标签未见

### mimic_test

- 主因: **临床表型噪声大，和疾病超边核心 HPO 对齐差**
- 次因: **存在部分未见标签**
- 次因: **标签空间大，长尾重，疾病间容易混淆**

## 下一步建议

1. 对 `DDD`，优先解决标签覆盖问题。
   - 扩充 `DDD` 训练标签覆盖
   - 或在评估时区分 `seen-label` / `unseen-label` 两部分

2. 对 `mimic_test`，优先解决表型噪声和分布偏移。
   - 加强 HPO 清洗，只保留更接近罕见病核心表型的项
   - 对住院并发症、器官功能异常、治疗状态类 HPO 做降权
   - 强化病例到超边的 key-HPO 匹配，而不是只做整体 pooled scoring

3. 对 `HMS`，优先做样本级误差审计。
   - 核对错分病例的 HPO 是否本身稀疏、泛化或偏症状学
   - 核对真标签对应的超边是否过宽/过旧/权重不合理

4. 评估报告应补一个分桶。
   - `seen label`
   - `unseen label`
   - `high hyperedge similarity`
   - `low hyperedge similarity`

这样才能把“模型能力不足”和“数据本身不可学”分开。
