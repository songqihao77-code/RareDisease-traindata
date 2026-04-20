# top1<40 数据集审计报告（基于 `best_20260420_170732`）

## 1. 审计范围

本报告针对以下最新评估结果展开：

- `D:\RareDisease-traindata\outputs\stage2_finetune_v59\evaluation\best_20260420_170732_summary.json`
- `D:\RareDisease-traindata\outputs\stage2_finetune_v59\evaluation\best_20260420_170732_details.csv`

并结合以下数据与代码进行交叉核验：

- 测试集：`LLLdataset/dataset/processed/test/*`
- 训练集：`LLLdataset/dataset/processed/train/*`、`LLLdataset/dataset/processed/mimic_rag_0425.csv`
- 疾病超边：`LLLdataset/DiseaseHy/rare_disease_hgnn_clean_package_v59/v59DiseaseHy.npz`
- 关键代码：`src/data/build_hypergraph.py`、`src/data/dataset.py`

审计对象是 `top-1 < 40%` 的 4 个数据集：

- `HMS`: 16.0%
- `mimic_test`: 21.25%
- `DDD`: 33.77%
- `LIRICAL`: 35.59%

## 2. 执行摘要

结论不是“模型整体失效”，而是这 4 个低分集分别被不同类型的问题主导：

1. `mimic_test` 的主问题是数据标签噪声严重，尤其是同一个 `case_id` 对应多个 `mondo_label`，而当前管线会静默只取第一个标签，导致训练和评估都被污染。
2. `DDD` 的主问题是极端细粒度和同源样本覆盖不足。测试集几乎是一病一例，同源训练标签覆盖极低，很多标签只能靠跨源迁移。
3. `HMS` 的主问题是样本太少、每标签样本极稀、且长 phenotype 列表疑似引入噪声，导致模型难以稳定定位关键 HPO。
4. `LIRICAL` 的主问题不是标签没见过，而是每病样本仍然偏少，错误主要发生在 phenotype 较短、区分性不足的 case 上。
5. `v59DiseaseHy.npz` 确实是瓶颈之一，但不是唯一瓶颈。它整体很稀疏，特别是 `mimic_test` 的真标签对应超边非常短，放大了 frequent-label bias；但 `DDD/LIRICAL` 的主要问题仍然是数据分布与表型信息不足，而不是超边单点失效。

## 3. 关键证据

### 3.1 总览表

| 数据集 | case 数 | 有效标签数 | top-1 | top-3 | top-5 | 中位真实排名 | 主导问题 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HMS | 25 | 19 | 0.1600 | 0.2400 | 0.2800 | 35 | 小样本、低重复、长 phenotype 噪声 |
| mimic_test | 1873 | 323 | 0.2125 | 0.3337 | 0.3887 | 15 | 多标签脏数据、频繁标签偏置、超边过稀 |
| DDD | 761 | 741 | 0.3377 | 0.4560 | 0.5125 | 5 | 极端细粒度、同源覆盖极低、case 信息不足 |
| LIRICAL | 59 | 34 | 0.3559 | 0.4407 | 0.5424 | 4 | 每病样本少、短 phenotype 区分度不足 |

### 3.2 `v59DiseaseHy.npz` 的整体状态

`v59DiseaseHy.npz` 的形状是 `(19566, 16443)`，共有 `227,907` 个非零项。

- 平均每个疾病超边只连接 `13.86` 个 HPO
- 中位数只有 `7`
- `40.97%` 的疾病超边 HPO 数 `<= 5`

这说明 `v59` 的疾病先验整体偏稀疏。它不能单独解释全部低分问题，但会明显放大下面两类风险：

- 真标签本身超边很短，case 稍微多几个附加 HPO 就容易偏离真标签
- 训练样本多的疾病更容易在 embedding 空间里“吸走”邻近 case

### 3.3 当前代码对多标签同 `case_id` 的处理存在静默折叠

在 `src/data/build_hypergraph.py` 中：

- 第 `85` 行：`for case_id, group_df in df.groupby(case_id_col, sort=False):`
- 第 `86` 行：`mondo_id = group_df[label_col].iloc[0]`

这意味着只要一个 `case_id` 下存在多个 `mondo_label`，当前流程就只取第一条标签，其他标签被直接忽略。

而 `src/data/dataset.py` 会先把 `case_id` namespace 化，但不会阻止“同一源文件内同一 case_id 多标签”的情况：

- 第 `124-125` 行：只对 `case_id` 加 namespace
- 第 `140` 行：逐行把 `mondo_label` 映射到 `gold_disease_idx`

因此，**如果原始文件里一个 `case_id` 配了多个疾病标签，当前训练与评估都会被污染，而且不会报警**。

## 4. 分数据集审计结论

### 4.1 `mimic_test`

#### 证据

- `top-1 = 21.25%`
- 测试集中有 `227 / 1873 = 12.12%` 的 case 存在“同一 `case_id` 多个 `mondo_label`”
- 训练集 `mimic_rag_0425.csv` 中，这种问题更严重：`1031 / 7311 = 14.10%`
- 这些脏 case 不是随机脏数据，而是**完全相同的一组 HPO 被笛卡尔复制到多个标签**，`same_hpo_set_ratio = 1.0`
- 脏 case 上当前 top-1 只有 `9.69%`，干净 case 上 top-1 为 `22.84%`
- 如果把“命中该 case 的任一候选标签”都算正确，整体 top-1 可从 `21.25%` 提到 `23.22%`
- `70` 个标签在 `mimic_rag_0425.csv` 中同源训练计数为 `0`
- `30` 个标签在全训练集中都没见过
- 真标签在 `v59` 中很稀疏：
  - 平均超边 HPO 数 `18.88`
  - 中位数只有 `9`
  - `26.11%` 的 case 真标签超边 `<= 5` 个 HPO
  - `52.91%` 的 case 真标签超边 `<= 10` 个 HPO
- 错误预测明显向高频标签坍缩：
  - 错误样本的真标签平均训练样本数 `87.48`
  - 但错误 top-1 预测标签平均训练样本数 `158.61`
  - `67.46%` 的错误样本，其预测标签训练频次高于真标签

#### 原因判断

`mimic_test` 的低分主要不是单一“模型不够强”，而是三件事叠加：

1. **训练标签噪声**  
   `mimic_rag_0425.csv` 本身存在大规模“一组 HPO 对多个标签”的构造，这会把模型往互相冲突的监督信号上推。

2. **评估标签噪声**  
   `mimic_test.csv` 也存在同样问题，且当前评估代码只取第一个标签，导致部分样本被硬性算错。

3. **频繁标签偏置 + 真标签超边过稀**  
   很多真标签在 `v59` 中只有很短的超边，而训练集中某些标签样本数极多，模型更容易把边界模糊的 case 吸到这些高频标签上。

#### 优先改进项

1. 先清洗 `mimic_rag_0425.csv` 和 `mimic_test.csv`  
   同一个 phenotype 如果对应多个候选疾病，不能复用同一个 `case_id`。要么拆成不同 `case_id`，要么显式存成候选标签集合，不能交给现有管线静默折叠。

2. 在数据加载阶段加硬校验  
   对所有输入文件做 `case_id -> mondo_label` 一致性检查，只要 `nunique() > 1` 就报错终止。

3. 对 `mimic` 子集加 class-balance / logit-adjustment  
   当前 frequent-label bias 很明显，建议至少做一种：
   - 按标签频次重加权
   - balanced softmax / logit adjustment
   - 对 `mimic` 单独做采样均衡

4. 优先补 `mimic` 真标签在 `v59` 中过短的疾病超边  
   先从 `<= 5`、`<= 10` HPO 的疾病入手补充 key HPO。

5. 引入二阶段候选重排  
   第一阶段召回 20-50 个候选病，第二阶段用 key HPO / negation / source-specific reranker 重排，能明显缓解 frequent-label 吸附。

### 4.2 `DDD`

#### 证据

- `top-1 = 33.77%`
- `761` 个 case 对应 `741` 个有效标签，几乎是一病一例，`cases_per_label = 1.027`
- 测试集 `756` 个标签中，只有 `45` 个被同源训练集覆盖，同源标签覆盖率仅 `5.95%`
- 即使放到全训练集，也只有 `505 / 756 = 66.80%` 的标签出现过
- 按 case 统计：
  - `32.85%` 的测试 case 真标签在全训练集中是零样本
  - `92.51%` 的测试 case 真标签在同源训练集中是零样本
- 测试集存在 `16` 个多标签 `case_id`，占 `2.10%`
- 脏 case 当前 top-1 为 `12.5%`，显著低于干净 case 的 `34.23%`
- 但 `DDD` 的主问题不是 `v59` 超边太差：
  - 真标签 `v59` 覆盖比平均 `0.503`
  - 正确样本该值为 `0.673`
  - 错误样本该值为 `0.416`
- 正确样本 phenotype 更完整：
  - 正确 case 平均 `19.41` 个 HPO
  - 错误 case 平均 `16.66` 个 HPO

#### 原因判断

`DDD` 的低分本质上是**极端细粒度分类问题**：

1. 标签空间太细，几乎没有“同病多例”可以帮助模型稳定学到疾病边界。
2. 同源训练覆盖极低，大部分测试标签只能依赖跨数据源迁移，而不同源病例的表型写法并不一致。
3. phenotype 较短时，模型更容易在近邻综合征之间混淆；这从正确样本有更多 HPO、且真标签超边覆盖更高可以看出来。
4. 多标签 `case_id` 不是主因，但确实会进一步拉低分数。

#### 优先改进项

1. 优先补 `DDD` 同源训练覆盖  
   这是最直接的提升路径。当前同源覆盖只有 `5.95%`，先补测试标签对应疾病的同源病例。

2. 做 family-aware hard negative  
   `DDD` 的问题更像近邻综合征区分失败，不是简单 seen/unseen。建议在训练时针对同父类、同器官系统、相似 HPO 集的疾病做 harder negative。

3. 针对短 phenotype case 做 key HPO 强化  
   可以引入：
   - key HPO 提取
   - HPO 置信度/权重
   - 去掉过泛化的高频 HPO

4. 对 `DDD` 脏 case 做清洗  
   数量不大，但修复成本低，应该顺手处理。

### 4.3 `HMS`

#### 证据

- `top-1 = 16.0%`
- 只有 `25` 个 case、`19` 个标签
- 同源测试标签覆盖是 `100%`，所以问题不是“标签没见过”
- 但同源训练本身非常稀：
  - `63` 个训练 case 对应 `36` 个标签
  - `cases_per_label = 1.75`
  - 测试 case 的同源训练计数中位数只有 `2`
- `v59` 上真标签覆盖偏低：
  - 平均覆盖比 `0.224`
  - 中位数 `0.143`
- 正确和错误 case 的差异很明显：
  - 正确 case 平均 `10.0` 个 HPO
  - 错误 case 平均 `23.33` 个 HPO
  - 正确 case 的真标签覆盖比 `0.472`
  - 错误 case 的真标签覆盖比 `0.177`

#### 原因判断

`HMS` 的低分更像是**少样本 + phenotype 噪声过重**：

1. 虽然标签都见过，但每病训练样本仍然太少，不足以学稳。
2. 错误样本不是因为 HPO 太少，反而是 HPO 太多且与真标签超边对不上，说明其中包含不少非区分性或噪声 HPO。
3. `v59` 在这个集合上的真标签覆盖偏低，意味着图先验和 HMS 的 phenotype 写法/粒度存在偏差。

#### 优先改进项

1. 对 `HMS` 做 phenotype 去噪  
   先保留关键器官系统、进展性、起病年龄、结构畸形等高区分度 HPO，弱化泛症状。

2. 增加 `HMS` 同源样本  
   这个集合太小，模型的波动会非常大。

3. 做 source-aware weighting  
   `HMS` 的写法可能和其他源差异较大，建议在 finetune 阶段增加 source-aware batch 或小比例 source-specific 微调。

4. 补齐 `HMS` 常见标签的 `v59` 超边  
   优先检查错误最集中的标签及其近邻标签在 `v59` 中是否缺 key HPO。

### 4.4 `LIRICAL`

#### 证据

- `top-1 = 35.59%`
- `59` 个 case、`34` 个标签
- 测试标签在同源训练和全训练中都 `100%` 被覆盖
- 但同源训练依然很稀：
  - `311` 个训练 case 对应 `252` 个标签
  - `cases_per_label = 1.234`
  - 测试 case 的同源训练计数中位数只有 `3`
- `LIRICAL` 不是图先验完全失效：
  - 真标签平均覆盖比 `0.538`
  - 正确 case 的覆盖比 `0.693`
  - 错误 case 的覆盖比 `0.453`
- phenotype 长度和准确率强相关：
  - 正确 case 平均 `24.90` 个 HPO
  - 错误 case 平均 `13.26` 个 HPO

#### 原因判断

`LIRICAL` 的主要矛盾是：**标签虽然见过，但每病有效样本仍少，且短 phenotype 时区分度明显不够**。

这类 case 的失败更像是：

- 真病就在候选附近，但 top-1 推不上去
- 需要更多 key phenotype 或更强的近邻重排，而不是单纯扩大全局疾病池训练

#### 优先改进项

1. 优先增强短 phenotype case  
   对 HPO 数较少的病例，补充 key HPO、发病年龄、遗传模式、阴性表型。

2. 做近邻疾病 rerank  
   `LIRICAL` 很适合二阶段结构：先粗召回，再对 top-N 做细粒度排序。

3. 增强同类疾病 hard negative  
   让模型更专门地区分近邻病，而不是继续学习全局粗匹配。

## 5. 哪些问题最值得先改

### P0：必须先改，否则继续训练会反复吃脏监督

1. 清洗所有 “同一 `case_id` 对应多个 `mondo_label`” 的数据  
   当前最严重的是：
   - `mimic_rag_0425.csv`: `1031 / 7311`
   - `mimic_test.csv`: `227 / 1873`
   - `DDD.csv`: `30 / 1522`
   - `DDD` test: `16 / 761`

2. 在数据导入阶段增加 hard fail 校验  
   不允许这种数据继续进入训练或评估。

3. 明确 `mimic` 的语义  
   如果一个 phenotype 天生对应多个候选疾病，就不能继续用单标签 CE 当作真值。

### P1：最可能直接提高 top-1 的改动

1. `mimic` 做 class-balance / logit-adjustment  
2. `DDD` 补同源训练覆盖  
3. `HMS/LIRICAL` 做 key HPO 提取与 phenotype 去噪  
4. 为 `mimic` 中超边极短的疾病补 `v59` 先验

### P2：结构性升级

1. 二阶段召回 + 重排  
2. family-aware hard negative  
3. source-aware finetune / source-adapter  
4. 把单标签训练改成支持候选标签集合或软标签

## 6. 我认为最靠谱的提升路径

如果目标是尽快把 `top-1 < 40` 的集合拉起来，建议按这个顺序做：

1. 先修 `mimic` 和 `DDD` 的多标签同 `case_id` 问题  
   这是最硬的脏数据问题，不修会持续污染。

2. 对 `mimic` 加频次校正  
   这个集合的错误明显往高频标签坍缩，性价比最高。

3. 对 `DDD/LIRICAL` 增强短 phenotype case  
   这两个集合里，正确样本都明显拥有更多 HPO 与更高真标签覆盖。

4. 对 `HMS` 做 phenotype 去噪而不是继续盲目堆 HPO  
   该集合的错误样本反而更长，说明“更多 HPO”并不等于“更准”。

5. 补 `v59` 中 `mimic` 高频真标签的短超边  
   尤其优先看真标签超边 `<= 5` 或 `<= 10` 的疾病。

## 7. 风险与待确认项

1. 本报告已经确认多标签 `case_id` 是真实问题，但还没有对“清洗后重训”的提升幅度做正式 ablation。
2. `hpo_dropout_prob=0.2`、`hpo_corruption_prob=0.15` 对低信噪比集合可能偏激进，但这一点当前还属于推断，尚未单独做消融。
3. `FakeDisease.xlsx` 在当前 finetune 配置中被纳入训练，是否对 `mimic/HMS` 的边界造成额外扰动，也值得单独验证。

## 8. 最终结论

这 4 个低分数据集并不是同一种“低分”：

- `mimic_test` 是**脏标签 + 高频标签偏置 + 稀疏超边**主导
- `DDD` 是**极端细粒度 + 同源覆盖不足 + phenotype 不完整**主导
- `HMS` 是**小样本 + phenotype 噪声**主导
- `LIRICAL` 是**已见标签但区分信息不足**主导

因此不建议再用统一策略去“整体提分”。最有效的办法是：

- 先修数据一致性
- 再针对 `mimic` 做频次偏置修正
- 针对 `DDD/LIRICAL/HMS` 分别做同源补样、本体超边补全、短 phenotype 增强和重排

这是比继续盲目调网络超参更有把握的提分路径。
