# HGNN 框架根因分析与修复路线报告

本报告基于 `audit_current_pipeline_report.md` 作为主输入材料，并结合仓库关键热路径做了定向核对。核对范围刻意收敛在主链脚本、训练配置、`trainer/evaluator`、`ModelPipeline`、`build_hypergraph`、`dataset`、`readout`、`case_refiner`、`loss_builder` 上，目标不是重复表层审计，而是重建当前框架真实运行方式，归并根因，判断主次矛盾，并形成可直接指导后续代码修改与实验验证的技术方案。

本次定向核对涉及的关键文件包括：

- `run_full_train.cmd`
- `run_attention_residual_sweep.cmd`
- `configs/train.yaml`
- `configs/train_pretrain.yaml`
- `configs/train_finetune.yaml`
- `configs/train_finetune_attn_ru000.yaml`
- `configs/train_finetune_attn_ru010.yaml`
- `configs/train_finetune_attn_ru020.yaml`
- `configs/train_finetune_attn_ru100.yaml`
- `configs/data_llldataset_eval.yaml`
- `src/training/trainer.py`
- `src/evaluation/evaluator.py`
- `src/models/model_pipeline.py`
- `src/models/readout.py`
- `src/models/case_refiner.py`
- `src/models/scorer.py`
- `src/models/hgnn_encoder.py`
- `src/data/build_hypergraph.py`
- `src/data/dataset.py`
- `src/training/loss_builder.py`
- `src/training/hard_negative_miner.py`

---

## A. 《基于审计报告重建的当前框架真实形态》

### A.1 当前真实运行方式

当前仓库的真实主线不是“统一超图 `H=[H_case|H_disease]` 一起进入 encoder”，而是一个已经演化成 **disease-only encoder + case-side construction + full-pool disease matching** 的诊断框架。

从主入口看，真实实验主链是：

```text
run_full_train.cmd
  -> stage1: python -m src.training.trainer --config configs/train_pretrain.yaml
  -> stage2: python -m src.training.trainer --config configs/train_finetune.yaml
  -> eval  : python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/train_finetune.yaml --checkpoint_path outputs/stage2_finetune/checkpoints/best.pt
```

这意味着当前“默认入口”并不是 `configs/train.yaml`，而是脚本串起来的 staged pipeline。任何不先区分这两条入口的人，后续对 loss、数据、实验结果的理解都很容易跑偏。

从数学热路径看，当前前向流程是：

1. 静态知识侧只加载 `H_disease`，其形状是 `[num_hpo, num_disease]`，且是带权疾病超边。
2. `ModelPipeline.forward()` 实际只把 `H_disease` 送入 `HGNNEncoder`，得到全局 HPO 节点表示 `Z`。
3. 病例侧通过 `H_case` 从 `Z` 中构造 `case_repr`。如果启用了 `case_refiner`，则先对病例活跃 HPO 的节点表示做一次病例条件化细化，再进入 readout。
4. 疾病侧表示由 `H_disease^T @ Z` 直接读出，得到 `disease_repr`。
5. `CosineScorer` 对 `case_repr` 与 `disease_repr` 做归一化后点积，产生 `scores[B, num_disease]`。
6. 训练目标是 full-pool cross entropy，可叠加 poly term 和 hard negative ranking loss。
7. 训练和评估都在真正的全疾病池上竞争，不是局部候选池，也不是只在 batch 内负采样。

### A.2 与原始设计意图的关系

和你给定的设计意图对照，当前实现和设计意图其实大体同向：

- 疾病超边带权，病例超边二值，这一点成立。
- encoder 尽量只使用 `H_disease`，避免 `H_case` 污染 encoder，这一点已经在真实热路径中成立。
- 病例信息主要在 `readout / case_refiner / scorer` 之前的病例侧构造阶段进入，这一点基本成立。
- 任务是病例-疾病匹配，不是节点分类，这一点成立。
- scorer 是 cosine，这一点成立。
- 训练目标主体是 full-pool CE，可叠加 hard negative，这一点成立。

真正的矛盾，不在“设计本身是否错了”，而在 **实现层与实验治理层没有把这条新主线收敛成唯一可信口径**：

- 文档、注释、兼容字段还残留着旧的 unified hypergraph 叙事。
- 默认入口 `train.yaml` 与脚本主线配置不一致。
- 配置表面语义与代码 hidden default 不一致。
- 病例侧增强模块已经叠加不少，但 baseline 还没有冻结成唯一可信主线。

### A.3 当前框架的真实语义

如果用任务语义重述当前框架，它更像：

- 疾病侧是相对稳定的知识库编码器。
- 病例侧是 noisy query 的构造器，而不是知识图的一部分。
- 训练目标是让 noisy query 在全疾病知识池中找到正确疾病原型。

这是一个合理的罕见病诊断建模方向。因为在这个任务里：

- 疾病侧更适合被当作稳定知识。
- 病例侧 HPO 本身更噪、更稀疏、更依赖上下文。
- 把病例侧直接混进 encoder，天然比在 readout 后段引入更容易破坏稳定知识空间。

### A.4 为什么当前矛盾会影响最终诊断效果

影响效果的不只是“模型有没有 bug”，而是“实验结论是否可信、改动是否可归因、训练目标是否真的是你以为的那个目标”。

当前最伤结果解释力和迭代效率的，不是疾病侧 HGNN 数学定义，而是以下三类结构性问题：

- 入口与配置漂移，导致同名实验的真实含义不稳定。
- 索引与标识语义脆弱，导致结果虽然能跑通，但容易被静默误用。
- 病例侧复杂模块过早叠加，导致性能变化无法拆解成清晰因果。

这三类问题会共同造成一个后果：即便某次指标涨了，也很难知道究竟是 loss 变了、数据变了、病例侧模块变了、还是索引/装配细节漂了。对诊断任务来说，这比单点 bug 更致命，因为它直接阻断后续的可靠迭代。

---

## B. 《问题根因归并与影响链分析》

### B.1 根因归并结论

审计报告列出的问题可以归并到四个上游根因，而不是彼此独立的散点问题。

| 观察到的问题 | 更上游根因 | 直接影响 | 最终影响 |
| --- | --- | --- | --- |
| `run_full_train.cmd` 主链与 `train.yaml` 不一致 | 根因 R1：缺少唯一可信实验入口与单一事实来源 | 不同入口实际训练目标不同 | 结果不可复现，改动前后无法严谨对比 |
| `train.yaml` 表面 loss 与代码 hidden default 不一致 | 根因 R1：配置语义没有完全外显 | 用户以为是 CE，实际是 CE+poly+hard negative | 训练现象被误读，结论失真 |
| attention/residual sweep 不是单变量 | 根因 R1：实验治理缺失 | sweep 结果不可归因 | 错误选择模块方向，浪费算力与时间 |
| trainer/evaluator 各自复制 model config 组装逻辑 | 根因 R1：配置与装配没有统一收口 | 训练评估未来容易悄悄漂移 | checkpoint、评估、复现实验的信任度下降 |
| 文档仍在描述 unified `H` | 根因 R1：版本迁移未完成 | 团队会把旧叙事当成当前现实 | 错修问题，甚至把现在正确实现改坏 |
| `gold_disease_idx / gold_disease_cols_global / gold_disease_cols_local` 混杂 | 根因 R2：索引语义没有类型化、命名没有收口 | 依赖隐式映射才能保持正确 | 极易出现 silent bug，尤其在后续重构中 |
| `case_id` 只按 stem 命名，train/test 有碰撞风险 | 根因 R2：标识命名空间没有被程序强约束 | 后处理、join、误差分析被污染 | 会误判泄露、误判重复病例、误判模型错误 |
| `gold_disease_idx` 在 dataset 生成，但后续链路又靠标签重映射 | 根因 R2：接口边界不干净 | 上游字段存在但不构成下游真值接口 | 可维护性差，容易让人误以为某字段是主标签 |
| baseline 未冻结就叠加 attention / refiner / corruption / dropout / hard negative | 根因 R3：病例侧结构边界尚未收敛 | 性能变化来自多个耦合因素 | 无法判断真正有效模块，调参成本爆炸 |
| readout 与 case_refiner 都在做病例上下文加权 | 根因 R3：病例侧职责边界不清 | 模块作用重叠 | 复杂度增加，但不保证收益 |
| 16443 全疾病池、表型噪声、疾病相似度高 | 根因 R4：任务本身难 | baseline 上限不可能很高 | 指标提升更依赖稳定实验治理 |

### B.2 根因说明

#### 根因 R1：缺少唯一可信实验入口与单一事实来源

这是当前最上游的问题。代码已经从旧方案演化到新方案，但入口、配置、文档、实验脚本、装配逻辑没有一起收敛。结果是：

- 代码真实行为正确的部分，没有被稳定表达出来。
- 配置里写出来的东西，不一定就是实际生效的东西。
- 评估与训练现在大体同构，但未来很容易再次分叉。

这类问题本身未必立刻把 top1 打穿，但它会让所有“结果解释”和“下一步该修什么”的判断失去地基。

#### 根因 R2：索引与标识语义没有类型化

当前能跑通，很大程度依赖几个关键转换函数刚好把语义接上了，而不是因为接口定义天然清晰。最典型的是：

- `gold_disease_idx` 是疾病索引。
- `gold_disease_cols_global` 是组合矩阵列号。
- `gold_disease_cols_local` 是 scorer 空间局部索引。

这三者当前刚好通过 `_build_gold_local()` 接住了，但这不是稳健设计，而是“依赖实现细节维持正确”。这类问题最危险的地方在于，它往往不会第一时间报错，而是会在新实验、新统计脚本、新模块接入时静默出错。

#### 根因 R3：病例侧结构边界尚未冻结

现在 disease side 已经相对清晰，但 case side 仍处于“多个增强同时存在”的过渡态：

- `case_refiner` 在做病例条件化。
- `readout attention` 在做病例内 HPO 重加权。
- `hpo_dropout_prob / hpo_corruption_prob` 在改病例输入分布。
- `hard negative` 在改变优化压力。

这些东西都作用在 noisy query 侧，本来就容易互相耦合。如果 baseline 没有先冻结，再上这些模块，性能变化就很难定位。

#### 根因 R4：任务本身困难

罕见病诊断当然是难任务，full-pool disease matching 也比局部候选筛选难得多。但当前证据表明，它不是首要根因。因为目前最突出的风险，仍是实现正确性与实验治理问题，而不是“任务太难所以怎么做都差”。

### B.3 因果链总结

当前最典型的因果链可以概括为：

```text
没有唯一可信入口 / 配置语义不外显
-> 同名实验真实含义不同
-> sweep 不是单变量
-> 无法归因病例侧模块收益
-> 复杂模块继续叠加
-> baseline 始终不可信
-> 指标波动无法转化为稳定架构结论
```

以及：

```text
索引与 case_id 语义未收口
-> 当前靠隐式映射跑通
-> 后续分析 / 重构 / join 极易误用
-> 结果解释被污染
-> 进一步加剧“实验不可归因”
```

---

## C. 《当前效果不理想的主因判断（实现 / 设计 / 数据任务）》

### C.1 主次判断

当前效果不理想的主因排序，我的判断是：

1. **主因：实现与实验治理问题**
2. **次因：框架设计边界尚未收敛，尤其是病例侧复杂模块叠加过早**
3. **第三因：数据与任务本身难度**

### C.2 判断依据

之所以把“实现与实验治理问题”放在第一位，不是因为代码存在大量会直接崩溃的 bug，而是因为当前最关键的问题是：

- 你无法稳定地知道自己到底在比较什么。
- 你无法确认一次性能变化来自哪一层。
- 你无法把实验结论安全传递给下一个改代码的人。

这会直接拖累效果优化速度，甚至让部分“看起来有效”的改动只是伪增益。

之所以把“框架设计问题”放在第二位，是因为当前主路线本身并没有被证伪。相反，`disease-only encoder + case-side readout + full-pool cosine matching` 在任务语义上是合理的。真正的问题是：

- case side 的职责边界还没冻结。
- baseline 尚未作为唯一可信主线被建立。
- attention、refiner、corruption、hard negative 的组合关系没有被拆开验证。

之所以把“数据与任务难度”放在第三位，是因为虽然任务本身确实难，但从现有证据看，框架还没有到“已经被实现和实验治理充分整理，只剩任务上限”的阶段。现在把问题归因给任务难度，会过早地掩盖工程主问题。

### C.3 一句话结论

当前性能问题首先不是“HGNN 主路线错了”，也不是“数据太难无解”，而是 **框架虽然大方向合理，但实现正确性、配置治理、索引语义和实验可归因性还没有被整理成可信工程系统**。

---

## D. 《高优先级问题重排序》

这里不机械沿用原审计报告的 P0/P1/P2/P3，而是从“最影响后续效果提升”的角度重新排序。

### D.1 如果只能先修 3 个问题，建议先修这 3 个

#### 1. 建立唯一可信主线，消灭入口与配置语义漂移

这包括：

- 明确 `run_full_train.cmd + train_pretrain.yaml + train_finetune.yaml + data_llldataset_eval.yaml` 才是当前主实验链。
- 让 `train.yaml` 不再冒充主入口，或者让它与主线语义完全一致。
- 消灭 hidden default，尤其是 loss 相关默认值。
- 训练启动时打印完整展开后的最终有效配置。

这是第一优先级，因为如果这个问题不先修，后面所有实验对比都不可信。

#### 2. 冻结一个唯一可信 baseline，停止在 baseline 未稳定前叠加病例侧复杂模块

这包括：

- 明确一个最小可信 baseline。
- 把 `readout attention`、`case_refiner`、`hard negative`、`hpo_corruption` 的启用顺序拆开。
- 重做单变量实验，不再接受 confounded sweep。

这是第二优先级，因为当前最大的问题不是“模块不够多”，而是“根本不知道哪个模块有用”。

#### 3. 清理索引语义与样本标识语义，并把训练/评估装配逻辑收口

这包括：

- 重命名 `gold_disease_*` 三套索引。
- 对索引转换增加断言。
- `case_id` 加入 split/路径级命名空间。
- trainer/evaluator 共用一套 model config builder。

这是第三优先级，因为它直接决定后续重构是不是安全，评估是不是可信，分析是不是会被污染。

### D.2 为什么不是优先去改 encoder 或大改主架构

因为现有证据并不支持“disease-only encoder 是主问题”。相反，现有热路径已经相对贴合任务语义。现在最该做的不是重新发明架构，而是先把这条路线整理成一个可信、可归因、可复现实验系统。

---

## E. 《关键矛盾逐项分析与解决方案》

### E.1 `H_case` 是否应该进入 encoder？

当前阶段，我的建议是：**不应该直接进入同一个 encoder 主干**。

原因有三点：

- 疾病侧是相对稳定知识，病例侧是 noisy query。把两者放进同一超图 encoder，会让 query 噪声反向污染知识表示。
- 训练时病例集合是 batch 相关的，若 `H_case` 进 encoder，则疾病侧节点表示会随 batch 改变，破坏疾病侧表征稳定性。
- 评估时病例分布与训练不同，若 encoder 使用 `H_case`，训练链和评估链更难严格同构。

当前最合理的做法是：

- encoder 只编码 `H_disease`，把它看成疾病知识图编码器。
- `H_case` 只在病例表示构造阶段进入，即 `readout` 之前或 `case_refiner -> readout` 这一段。
- 如果未来要引入病例和疾病的更强交互，优先考虑 post-encoder 的 query-conditioned interaction，而不是让病例超边直接进入 disease encoder。

如果未来一定要尝试 `H_case` 入 encoder，必须满足以下工程约束：

- 疾病侧表示不能被单个 batch 的病例集合任意漂移。
- 训练和评估必须共享完全一致的 query injection 方式。
- 必须做严格的 ablation，与 current disease-only encoder 做一对一对照。
- 需要明确防止 query-to-query 之间的批内信息串扰。

结论是：当前不建议把 `H_case` 拉回 encoder。那不是当前最值得投入的修复方向。

### E.2 当前 `readout / case_refiner / scorer` 的角色边界是否清晰？

当前边界不够清晰，尤其是 `readout attention` 和 `case_refiner` 都在做“病例上下文条件化”的事情，存在职责重叠。

更合理的角色定义应该是：

- `readout`：唯一负责把节点表示聚合成 `case_repr` 与 `disease_repr`。
- `case_refiner`：只做病例侧节点表示的可选预处理，不改变疾病侧，不直接打分。
- `scorer`：只负责相似度定义与排序，不承担补救表示质量的职责。

建议的重定义方式：

- baseline 阶段，`readout` 先退回到最简单、最透明的聚合器。
- `case_refiner` 默认关闭，只有在 baseline 稳定后再单独验证。
- `scorer` 继续保持 cosine，不要在当前阶段引入可学习复杂 scorer。

如果要减少冗余，我更倾向于：

- 第一阶段保留 `disease_repr = H_disease^T @ Z` 不动。
- 病例侧先只保留一个简单 readout。
- 把 `case_refiner` 作为后续可选增强，而不是当前默认主线。

### E.3 当前 full-pool competition 是否合理？

合理，而且从任务定义上应该坚持。

罕见病诊断的真实目标不是“从小候选池中挑对”，而是让病例在全疾病池中获得正确排名。当前 full-pool competition 的方向是对的。但它要成立，必须满足以下前置条件：

- disease index 必须固定、连续、全局唯一。
- `scores` 的疾病列顺序必须在训练和评估中完全一致。
- `gold_disease_cols_local` 必须严格对应 scorer 空间，而不是历史遗留列空间。
- 训练和评估必须使用同一 scorer 定义。
- 不允许因为配置漂移，让某些实验实际比较的是不同 pool 或不同 loss。

当前最先该修的 full-pool 前置条件不是 scorer，而是：

1. 显式固定主入口和最终有效配置。
2. 清理索引语义与断言。
3. 统一 trainer/evaluator 的 model assembly。

### E.4 训练链与评估链是否应完全同构？

核心数学路径上，必须严格同构。

必须严格一致的部分包括：

- encoder 输入只能是 `H_disease`。
- 病例侧表示构造逻辑必须一致。
- 疾病侧 readout 逻辑必须一致。
- scorer 必须一致。
- 疾病列空间顺序必须一致。
- gold label 到 score pool 的映射必须一致。

允许存在工程差异的部分包括：

- 评估时可以缓存 `node_repr` 与 `disease_repr`。
- 评估时可以有更强的 index 校验和 skip reason 统计。
- 训练时可以有 dropout/corruption，评估时必须关掉。
- 批大小、日志、保存格式可以不同。

当前仓库里，训练与评估在核心数学路径上已经比较接近，但 duplicated model config builder 是潜在漂移源，必须尽快收口。

### E.5 hard negative 在这个项目中真正应该解决什么问题？

hard negative 不应该用来“救主线”，它真正应该解决的是：

- 在 full-pool 场景下，模型已经能学到基本可分性后，
- 进一步压低那些与真实疾病非常相近的高分错误疾病，
- 提升 top1 和前列排名质量。

它应该在什么阶段启用：

- 不应该在 baseline 还不稳定时默认启用。
- 更适合放在 baseline 跑通、简单 readout 已验证后，再作为后续增强项开启。

它会不会掩盖主问题：

- 会。

因为 hard negative 会明显改变梯度结构。如果 baseline 本身还存在配置漂移、病例侧结构混杂、索引语义不清，它完全可能制造一种“好像更强了”的假象，但你不知道它是在补真正的表示问题，还是在掩盖训练定义不稳定的问题。

建议是：

- 第一阶段关闭 hard negative。
- 第二阶段在固定 baseline 上做单变量验证。
- 只有确认它主要改善 near-miss disease 的排名，才保留。

### E.6 attention / case_refiner 是否真的有理论必要性？

在当前阶段，没有充分证据证明它们有“必须先上”的理论必要性。

从任务语义看，attention 或 case_refiner 可能有用的地方是：

- 病例 HPO 噪声较大，需要抑制无关表型。
- 同一病例内部不同 HPO 的判别价值不同。
- 需要从病例上下文中重新解释局部表型。

但在当前阶段，它们更可能带来的问题是：

- 增加优化不稳定性。
- 增加实验方差。
- 与 corruption/dropout/hard negative 一起耦合，导致难以归因。

所以当前判断是：

- 它们不是错误方向。
- 但它们也不是现在最该先押注的收益来源。
- 当前更值得做的是先建立一个简单、稳定、能复现实验结论的 baseline。

### E.7 当前“疾病超边 + 病例侧 readout”整体路线是否仍值得坚持？

值得坚持，而且不建议轻易推翻。

应该保留的部分：

- `H_disease` 驱动的 disease-only encoder。
- `H_disease^T @ Z` 的疾病侧 readout。
- full-pool disease matching 任务定义。
- cosine scorer 的简单稳定打分形式。

应该重构的部分：

- 配置入口和有效配置打印。
- trainer/evaluator 共享模型装配逻辑。
- 病例侧模块职责边界。
- 索引与 case id 的语义定义。

应该暂时删掉或默认关闭的部分：

- 旧的 unified `H` 叙事在主文档中的主地位。
- baseline 之前默认启用的 `case_refiner`。
- baseline 之前默认启用的 hard negative。
- confounded sweep 的继续扩展。

结论不是“推翻主路线”，而是“保留主路线、简化病例侧、修正工程骨架”。

---

## F. 《分阶段修复路线图》

### F.1 第一阶段：必须先修的实现正确性与实验治理问题

目标：

- 建立唯一可信主线。
- 让每次训练和评估都能明确知道自己真实跑了什么。
- 消除最危险的 silent mismatch 风险。

修改范围：

- 入口层：`run_full_train.cmd`、`train.yaml`、训练启动日志。
- 配置层：loss 默认值、最终有效配置打印、主线配置声明。
- 语义层：`gold_disease_*` 命名重构、`case_id` 命名空间修复、断言补齐。
- 装配层：trainer/evaluator 共用一套 model config builder。

成功判据：

- 主线实验入口只有一个可信口径。
- 训练启动时能打印最终 effective config。
- `train.yaml` 不再含有 hidden default 误导。
- 所有 gold index 都能被显式解释清楚。
- 训练和评估的模型组装来自同一实现。

失败时说明什么：

- 如果第一阶段都无法稳定，说明当前性能变化几乎不具备可解释性，继续做模块增强没有意义。

### F.2 第二阶段：在实现正确基础上优化框架结构

目标：

- 建立唯一可信 baseline。
- 明确病例侧最小有效结构。
- 验证 attention / case_refiner / hard negative 的真实边际收益。

修改范围：

- 默认 baseline 配置。
- 病例侧 readout 简化与单变量消融。
- case_refiner 的开关与默认值。
- hard negative 的启用阶段与实验对照。

成功判据：

- 至少有一条 baseline 能稳定复现。
- attention 是否有效有单变量证据。
- case_refiner 是否有效有单变量证据。
- hard negative 是否改善 near-miss 排名有清晰证据。

失败时说明什么：

- 如果连在简化 baseline 上也得不到稳定结果，才需要重新审视 disease-side encoder 本身或数据表示能力。

### F.3 第三阶段：在框架稳定后再尝试增强模块或更复杂方法

目标：

- 在可信 baseline 上追求增益，而不是用复杂度换不确定性。

修改范围：

- 更复杂的病例侧交互。
- 更精细的难负样本策略。
- 更强的 scorer 或分阶段训练策略。

成功判据：

- 复杂模块相对 baseline 有稳定、可重复、可解释的增益。
- 增益能在多个数据子集上保持一致方向。

失败时说明什么：

- 如果第三阶段复杂模块收益不稳定，不说明主路线错误，只说明当前任务信号不足以支撑更复杂 query-side 模型。

---

## G. 《最小验证实验设计》

下面这组实验不是为了“刷最高指标”，而是为了快速定位当前到底卡在哪里。

### 实验 1：主线配置真值校验实验

改什么：

- 明确只跑 `run_full_train.cmd` 主线。
- 训练启动时打印最终 effective config。
- 对 `train.yaml` 的 hidden default 做一次显式输出对照。

对照组是什么：

- 当前仓库默认行为。
- 修复后“显式配置等于实际生效配置”的行为。

期望观察到什么：

- 主线 staged pipeline 的 effective config 可被完整追踪。
- `train.yaml` 与脚本主线不会再被误当成同一实验。

如果结果不符合预期意味着什么：

- 说明当前配置层仍有未识别的隐式逻辑，必须继续清理，否则后续所有实验结论都不可信。

### 实验 2：最小 baseline 冻结实验

改什么：

- 保持 disease-only encoder。
- 关闭 `case_refiner`。
- 关闭 hard negative。
- 关闭 `hpo_corruption_prob`，必要时连 `hpo_dropout_prob` 也降到最小。
- 病例侧 readout 先用最简单聚合形式，优先测试 `residual_uniform=1.0` 或等价均匀聚合。

对照组是什么：

- 当前 `train_finetune.yaml` 主线。
- 简化后的 clean baseline。

期望观察到什么：

- clean baseline 的训练曲线更稳定，方差更小。
- 即便指标不一定立刻最高，也应更适合作为可信参照。

如果结果不符合预期意味着什么：

- 如果 clean baseline 比当前主线显著更差且不稳定，说明病例侧增强可能已经承载关键能力，需要重新检查 readout 设计而不是简单关闭模块。

### 实验 3：readout attention 单变量实验

改什么：

- 在固定 baseline 的前提下，只改 `residual_uniform` 或 attention 开关。
- 其余包括数据、loss、dropout、corruption、checkpoint 初始化都保持不变。

对照组是什么：

- 简单均匀聚合。
- leave-one-out attention readout。

期望观察到什么：

- 如果 attention 真实有效，应在不引入新变量的情况下带来稳定增益，尤其是 top1 或 true rank 改善。

如果结果不符合预期意味着什么：

- 说明 attention 不是当前主要瓶颈，或者其收益被病例噪声与结构复杂度抵消，应回退到简单聚合。

### 实验 4：case_refiner 单变量实验

改什么：

- 在“已经选定的 readout 方案”上，只开启 `case_refiner`。

对照组是什么：

- 最佳 baseline readout。
- baseline readout + `case_refiner`。

期望观察到什么：

- 若 `case_refiner` 真有效，应改善 noisy case 的匹配质量，而不是仅带来随机波动。

如果结果不符合预期意味着什么：

- 说明它当前更像复杂度源，而不是有效增益模块，应从默认主线中移除。

### 实验 5：hard negative 延后启用实验

改什么：

- 在 baseline 已稳定后，对比 CE-only 与 CE+hard negative。
- hard negative 从较晚 epoch 启动。

对照组是什么：

- CE-only baseline。
- CE + hard negative。

期望观察到什么：

- 若 hard negative 真有价值，应主要改善“高相似错误疾病”的排序，表现为 top1 或 true rank 的后期提升。

如果结果不符合预期意味着什么：

- 说明 hard negative 只是放大了训练噪声，或者 baseline 还没稳到值得引入它。

### 实验 6：索引与命名空间防错实验

改什么：

- 为 `gold_disease_idx / gold_disease_col_in_combined_h / gold_disease_idx_in_score_pool` 建立显式断言。
- `case_id` 改为带 split 或路径命名空间。

对照组是什么：

- 当前命名与索引实现。
- 收口命名与断言后的实现。

期望观察到什么：

- 所有 batch 的索引映射都可解释、可验证。
- train/test 的 `case_id` 不再发生命名碰撞。

如果结果不符合预期意味着什么：

- 说明当前已有分析结果中可能包含静默误差，需要回溯结果解释过程。

---

## H. 《是否需要推翻当前框架》

### H.1 结论

**不需要推翻当前框架。**

当前 HGNN 主路线仍然成立，尤其是以下核心判断是正确的：

- 疾病侧是更稳定的知识载体。
- 病例侧是 noisy query，更适合在后段进入。
- full-pool disease matching 是合理任务定义。
- cosine scorer 是当前阶段合适的稳定选择。

### H.2 哪些部分值得保留

- `H_disease` 驱动的 encoder。
- `H_disease^T @ Z` 的疾病表示读出。
- `H_case` 只在病例侧构造阶段进入。
- full-pool competition。
- 训练评估共用同一分数定义。

### H.3 哪些部分应该重构

- 实验入口与配置系统。
- trainer/evaluator 的模型装配。
- 索引与 `case_id` 的语义命名。
- 病例侧模块的职责边界与默认主线。

### H.4 哪些部分可以暂时删掉或降级

- 文档中对 unified `H` 的主叙事。
- baseline 之前默认启用的 `case_refiner`。
- baseline 之前默认启用的 hard negative。
- 任何不是单变量的 sweep。

### H.5 什么时候才考虑推翻主路线

只有当下面两件事都发生时，才值得考虑更大幅度改架构：

- 第一阶段和第二阶段已经完成，实验口径、baseline、索引语义都稳定。
- 在严格单变量验证后，disease-only encoder 的简单主线依然明显打不过更简单的非图 baseline，且问题不能归因于数据、loss、病例侧结构。

在这之前，大改方向属于过早优化。

---

## I. 《给后续代码修改者的执行建议》

这一节按“技术负责人给工程同学下修改单”的方式写。

### I.1 先做哪些代码层动作

- 把 trainer 和 evaluator 的 model config 组装逻辑提取到一个共享函数，禁止双份维护。
- 在训练启动时打印完整 effective config，包含 loss、hard negative、poly、readout、case_refiner 的最终取值。
- 明确在 README 或主报告中声明：当前热路径是 disease-only encoder，不再把 `H_case` 混入 encoder。
- 给 `gold_disease_*` 全部重命名，名称必须显式带上其所在空间。
- 给索引转换加断言，尤其是“global col 是否属于 disease col 空间”和“local idx 是否落在 score pool 内”。
- 把 `case_id` 改为带 split 或相对路径前缀的命名空间。

### I.2 再做哪些配置层动作

- 把 `train.yaml` 定位成样例配置、废弃配置，或者让它完全等价于主线配置。
- 对 loss 配置实行“写什么跑什么”，不允许 hidden default 偷偷改语义。
- 为 baseline 提供单独的 clean config。
- 为 attention、case_refiner、hard negative 各自提供独立 ablation config。

### I.3 训练策略建议

- 先跑 clean baseline。
- 再只开 attention。
- 再只开 case_refiner。
- 最后再开 hard negative。

不要做的事情：

- 不要继续沿用 confounded sweep。
- 不要在 baseline 未冻结前继续叠加病例侧复杂模块。
- 不要在未清理索引语义前改动 scorer 或输出字段。

### I.4 验证与日志建议

- 每次训练产物保存 effective config snapshot。
- 每次评估保存 disease index version、checkpoint path、test file manifest。
- 对每个 batch 的 `num_case`、`num_disease`、gold local/global 映射做轻量断言。
- 对 train/test `case_id` 交集做自动检查，但注意区分“命名碰撞”与“真实病例泄露”。

### I.5 一个推荐的主线 baseline 定义

建议先把下面这条线定义为“唯一可信 baseline 候选”：

- encoder：只用 `H_disease`
- disease readout：`H_disease^T @ Z`
- case readout：简单均匀聚合或最简 attention 聚合二选一
- scorer：cosine
- loss：先 CE-only
- hard negative：关闭
- case_refiner：关闭
- HPO augmentation：先最小化

先把这条线做稳定，再谈增强。

---

## J. 《最终结论：当前最应该先做什么》

当前最应该先做的，不是推翻 HGNN 主路线，也不是急着把 `H_case` 拉回 encoder，而是把现有路线整理成一个 **唯一可信、可归因、可复现** 的工程与实验系统。

最关键的三件事是：

1. **统一主线入口与有效配置语义**  
   明确当前主线就是 staged pipeline，并消灭 `train.yaml` hidden default 带来的语义漂移。

2. **冻结一个 clean baseline**  
   在 baseline 未稳定前，默认关闭 `case_refiner` 和 hard negative，停止继续使用 confounded sweep。

3. **收口索引与标识语义**  
   重命名 gold disease 三套索引、修复 `case_id` 命名空间、统一 trainer/evaluator 装配逻辑，并补足断言。

如果这三件事做完，后续你才有资格严肃讨论：

- attention 是否真正有效
- case_refiner 是否值得保留
- hard negative 是否能带来稳定收益
- 是否还需要进一步加强 disease-case interaction

在当前阶段，**建立可信 baseline** 的价值远高于继续引入新模块。
