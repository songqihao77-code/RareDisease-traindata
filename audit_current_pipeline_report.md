# audit_current_pipeline_report

## 1. 《项目概览》

### 1.1 当前真实项目根目录与生效版本

- 本次自动定位到的 Git 根目录是 `D:\RareDisease-traindata`。
- 同机还存在 `D:\RareDisease` 与 `D:\RareDisease - 副本` 两个近似项目；关键文件哈希均不同，说明三者不是同一份代码。
- 当前脚本入口 `run_full_train.cmd` 与 `run_attention_residual_sweep.cmd` 都显式 `cd /d D:\RareDisease-traindata`，因此本报告只审计 `D:\RareDisease-traindata` 这套代码。

### 1.2 当前代码的真实目标

这套代码当前实现的是一个“病例-疾病匹配”式的 HGNN 流程，而不是普通节点分类：

- HPO 节点先在 `H_disease` 上做 HGNN 编码，得到全局 HPO 节点表示。
- 病例侧通过 `H_case` 从这些节点表示读出 `case_repr`。
- 疾病侧通过 `H_disease^T @ Z` 读出 `disease_repr`。
- `CosineScorer` 在全疾病池上做相似度打分，训练用 full-pool CE，可选叠加 hard negative ranking loss。

### 1.3 本次审计确认的核心结论

1. 当前热路径里，`H_case` 没有进入 HGNN encoder；encoder 实际只吃 `H_disease`。
2. `H_case` 只在 `case_refiner` 和 `readout` 阶段进入病例侧表示构造，不会污染疾病侧表示。
3. 当前训练和评估的打分空间都是真正的全疾病池，当前实测大小为 `16443` 个疾病。
4. 训练与评估的 scorer 都是同一个 `CosineScorer`，分数定义一致；loss 里只是在训练时对 cosine score 做温度缩放。
5. 当前实现存在明显的“版本/配置/命名漂移”问题，尤其是：
   - 默认配置与脚本实际跑的配置不一致。
   - `H=[H_case|H_disease]` 仍然存在于文档和兼容字段里，但已不再是热路径。
   - 某些实验配置并不只改了一个变量，导致实验归因不成立。

### 1.4 本次执行验证得到的关键运行证据

我直接调用了当前仓库代码做单批前向，得到以下真实结果：

- 训练批：
  - `H_case.shape = (19566, 4)`
  - `H_disease.shape = (19566, 16443)`
  - `scores.shape = (4, 16443)`
  - `batch_graph` 中没有 `H`
  - 输出键为 `case_ids / case_labels / case_cols_global / disease_cols_global / gold_disease_cols_global / gold_disease_cols_local / scores`
- 评估批：
  - `H_case.shape = (19566, 4)`
  - `H_disease.shape = (19566, 16443)`
  - `scores.shape = (4, 16443)`
  - `precompute_disease_side()` 产出的
    - `node_repr.shape = (19566, 128)`
    - `disease_repr.shape = (16443, 128)`
- 稀疏矩阵统计：
  - `H_disease` 的 `nnz = 227907`，`min=0.0001135`，`max=0.9504152`，`non_unit_count=227907`，说明它确实是带权疾病超边。
  - `H_case` 的非零值唯一集合为 `[1.]`，说明它当前是严格二值病例超边。

## 2. 《真实执行链总览》

### 2.1 主入口

当前主入口不是 `main.py`，而是两个脚本：

- `run_full_train.cmd`
  - `python -m src.training.trainer --config configs/train_pretrain.yaml`
  - `python -m src.training.trainer --config configs/train_finetune.yaml`
  - `python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/train_finetune.yaml --checkpoint_path ...\outputs\stage2_finetune\checkpoints\best.pt`
- `run_attention_residual_sweep.cmd`
  - 多次调用 `trainer.py + evaluator.py`，分别跑不同 finetune 配置

因此，当前“脚本级真实主链”是：

```text
run_full_train.cmd
  ├─ stage1: src.training.trainer(main, train_pretrain.yaml)
  ├─ stage2: src.training.trainer(main, train_finetune.yaml)
  └─ eval : src.evaluation.evaluator(main, data_llldataset_eval.yaml + train_finetune.yaml + best.pt)
```

### 2.2 训练链路

```text
配置读取
  trainer.load_config()
    -> 解析 train_*.yaml

训练数据读入
  trainer.resolve_train_files()
    -> paths.train_files / train_dir
  dataset.load_case_files()
    -> 读 csv/xlsx
    -> case_id 加 source stem 前缀
    -> 按 disease_index 生成 gold_disease_idx
    -> 过滤不在 disease_index 中的标签

训练/验证切分
  trainer.split_train_val_by_case()
    -> 按 case_id 切分 train_df / val_df

按病例分 batch
  dataset.CaseBatchLoader
    -> 保证一个病例不会被拆到多个 batch

静态图加载
  build_hypergraph.load_static_graph()
    -> hpo_to_idx
    -> disease_to_idx
    -> H_disease

每个 batch 构图
  build_hypergraph.build_batch_hypergraph()
    -> build_case_incidence()
    -> 生成 H_case
    -> 附带 case_ids / case_labels / disease_cols_global / gold_disease_cols_global
    -> 当前热路径 include_combined_h=False，不再生成 H=[H_case|H_disease]

模型前向
  model_pipeline.ModelPipeline.forward()
    -> encoder(H_disease)
    -> 可选 case_refiner(node_repr, H_case)
    -> readout(node_repr, H_case, H_disease, refined_case_node_repr?)
    -> scorer(case_repr, disease_repr)
    -> 输出 scores 与 gold_disease_cols_local

loss
  hard_negative_miner.mine_hard_negatives()
    -> 从 full-pool scores 中挖 top-k 负类
  loss_builder.FullPoolCrossEntropyLoss.forward()
    -> full-pool cross entropy
    -> 可选 poly term
    -> 可选 hard ranking term

反向传播
  trainer.run_one_epoch()
    -> zero_grad
    -> backward
    -> clip_grad_norm_ (可选)
    -> optimizer.step()

指标
  trainer.compute_topk_metrics()
    -> 直接在 full-pool scores 上算 top1/top3/top5
```

### 2.3 评估链路

```text
配置读取
  evaluator.load_yaml_config(data_config_path, train_config_path)

静态资源加载
  evaluator.load_static_resources()
    -> load_static_graph()
    -> 额外校验 HPO / disease index 连续性

测试集读入
  evaluator.load_test_cases()
    -> 读 test_files
    -> case_id 加 source stem 前缀
    -> 聚合成 case_table[case_id, mondo_label, hpo_ids, source_file]
    -> 标记 skip_reason

评估 batch 构图
  build_hypergraph.build_batch_hypergraph(..., include_combined_h=False)

加载 checkpoint
  evaluator.load_checkpoint_model()
    -> 用 train_config 重建 ModelPipeline
    -> 加载 checkpoint state_dict

预计算 disease side
  model.precompute_disease_side(H_disease)
    -> encoder(H_disease)
    -> readout.build_disease_repr(node_repr, H_disease)

逐 batch 前向
  model.forward(batch_graph, node_repr_override, disease_repr_override)
    -> 只重算 case side
    -> disease side 复用缓存

排序与写回
  evaluator.evaluate()
    -> full-pool topk
    -> true rank
    -> per-dataset summary
  evaluator.save_results()
```

### 2.4 关键问答直答

#### B1. `H_case` 到底有没有进入模型？

有，但不进入 encoder。

- `trainer.run_one_epoch()` 和 `evaluator.evaluate()` 都调用 `build_batch_hypergraph(..., include_combined_h=False)`。
- `ModelPipeline.forward()` 在 `src/models/model_pipeline.py:305-317` 只对 `batch_graph["H_disease"]` 调 `self.encoder(...)`。
- `H_case` 只在：
  - `self.case_refiner(node_repr, batch_graph["H_case"])`
  - `self.readout(..., batch_graph["H_case"], H_disease, refined_case_node_repr=...)`
  中进入。

结论：`H_case` 参与模型，但只在病例侧细化与读出阶段进入。

#### B2. `H_disease` 是如何进入模型的？

完整路径如下：

1. `configs/train_*.yaml` 或 `configs/train.yaml` 提供 `paths.disease_incidence_path`。
2. `trainer.main()` / `evaluator.load_static_resources()` 调 `load_static_graph()`。
3. `load_static_graph()`：
   - `load_index_file(hpo_index_path, "hpo_id", "hpo_idx")`
   - `load_index_file(disease_index_path, "mondo_id", "disease_idx")`
   - `load_disease_incidence(disease_incidence_path)` 读入带权稀疏矩阵
4. 该矩阵放进 `static_graph["H_disease"]` / `resources["H_disease"]`
5. `ModelPipeline.forward()` 中通过 `_prepare_h_disease()` 转成当前 device 上的 torch sparse
6. `self.encoder(prepared_h_disease)` 真正执行 HGNN

#### B3. 当前真正送入 HGNN encoder 的 `H` 到底是什么？

是 `H_disease`，不是 `H=[H_case|H_disease]`。

证据：

- `src/models/model_pipeline.py:305-309`
- 单批实跑 `batch_graph` 中 `train_batch_has_H = false`、`eval_batch_has_H = false`

#### B4. readout 如何构造病例表示和疾病表示？

- 病例表示：
  - `src/models/readout.py:152-178`
  - 从 `H_case` 的活跃边里取出对应 HPO 节点表示
  - 用 leave-one-out attention 聚合成 `case_repr`
  - 若启用 `case_refiner`，则先把活跃 edge 的节点表示做病例条件化，再进入同一个 readout
- 疾病表示：
  - `src/models/readout.py:233-236`
  - 直接 `H_disease^T @ Z`

结论：当前代码确实是“病例-疾病匹配”，不是别的节点级分类逻辑。

#### B5. scorer 如何计算分数？训练与推理是否一致？

一致，都是 cosine。

- `src/models/scorer.py:16-26`
- `case_repr` 和 `disease_repr` 先 `F.normalize`
- 再做矩阵乘 `scores = c @ d.t()`

训练与评估都调用同一个 `CosineScorer`；loss 内部只额外做 `scores / temperature`，不会改推理排序定义。

#### B6. top-k 的对象是全疾病池还是局部候选池？

是全疾病池。

- 单批实跑 `scores.shape = (4, 16443)`
- `16443` 正好等于 `num_disease`
- `trainer.compute_topk_metrics()` 和 `evaluator.evaluate()` 都直接在这个 full-pool `scores` 上做 top-k

#### B7. hard negative 是否真的生效？

在训练时可以生效，且作用于 full-pool scores。

- `trainer.run_one_epoch():304-314`
- 条件：
  - `is_train == True`
  - `use_hard_negative == True`
  - `epoch >= start_epoch`
- `mine_hard_negatives()` 会先把正类位置置为 `-inf`，再对剩余全疾病分数取 top-k
- `loss_builder._compute_hard_rank_loss()` 对这些 hard negatives 叠加 ranking margin loss

#### B8. attention / case_refiner 是否真的起作用？

- attention：起作用于病例 readout，不作用于 encoder，不修改 `H_case` 权重。
- case_refiner：若配置启用，确实起作用，但只作用在病例侧 active edge 的节点表示上，不作用于疾病侧表示，不改 `H_case` 稀疏矩阵本身。

关键证据：

- `src/models/case_refiner.py:87-173`
- `src/models/readout.py:180-231`
- `src/models/model_pipeline.py:315-330`

#### B9. trainer 和 evaluator 的图构造、前向逻辑、目标空间是否一致？

主体一致。

- 都用 `build_batch_hypergraph(..., include_combined_h=False)`
- 都用 `ModelPipeline`
- 都用 full-pool `scores`
- 都用 `gold_disease_cols_local`

主要差异：

- evaluator 会额外做
  - index 连续性校验
  - 可评估病例过滤与 `skip_reason`
  - `precompute_disease_side()` 缓存

这些差异不改变数学定义，但会造成“训练端不报错、评估端先报错”的漂移风险。

#### B10. 是否存在数据泄露、索引错位、局部/全局列号错配等风险？

有风险点，但要分清“已确认”和“风险”：

- 已确认：
  - `gold_disease_idx`、`gold_disease_cols_global`、`gold_disease_cols_local` 是三套不同概念，命名非常容易误用。
  - train/test 的 `case_id` 命名空间只用 `stem + "_" + raw_case_id`；`DDD` train/test 间实测有 `761` 个 prefixed `case_id` 碰撞。
- 未确认但存在风险：
  - 代码没有自动验证 train/test 语义是否真正隔离，只依赖配置选文件。
  - 训练端对 index 一致性的校验弱于评估端。

补充说明：本次对 `DDD` train/test 碰撞样本抽检后发现，`761` 个碰撞的 `prefixed case_id` 在标签和 HPO 集合上都不相同，因此**不能直接判定为真实重复病例泄露**；更准确地说，这是**命名空间碰撞风险**，不是本次已确认的数据重复泄露。

## 3. 《核心模块逐文件审计》

### 3.1 `src/data/build_hypergraph.py`

职责：

- 加载静态索引和 `H_disease`
- 从病例表构建 `H_case`
- 按需构造兼容字段 `H=[H_case|H_disease]`

关键函数：

- `load_disease_incidence()`
  - 支持 `scipy.sparse.save_npz` 格式和自定义 `rows/cols/vals/shape` 格式
- `build_case_incidence()`
  - 以病例为单位聚合 `hpo_id`
  - 过滤不在 `disease_to_idx` 中的标签
  - 过滤无有效 HPO 的病例
  - 对 `H_case` 一律写入 `1.0`
  - 返回 `gold_disease_cols`，其值本质上是 `disease_idx`
- `build_batch_hypergraph()`
  - 当前热路径下 `include_combined_h=False`
  - 仍然构造 `case_cols_global / disease_cols_global / gold_disease_cols_global`

与上下游关系：

- 上游：trainer/evaluator
- 下游：ModelPipeline

审计结论：

- 真实热路径已不依赖 `H`，但文件文档仍强烈强调 `H=[H_case|H_disease]`，存在版本漂移。
- `gold_disease_cols_global` 的命名带有历史包袱；它不是纯粹的“全局 disease_idx”，而是“在批次拼接矩阵中的列号”。

### 3.2 `src/data/dataset.py`

职责：

- 读取 csv/xlsx 病例表
- 合并多文件训练数据
- 生成按病例分 batch 的 `CaseBatchLoader`

关键函数：

- `read_case_table_file()`
- `load_case_files()`
  - 对每个文件做 `stem + "_" + case_id`
  - 生成 `gold_disease_idx`
  - 过滤 disease index 外标签
- `CaseBatchLoader`
  - 以 case 为最小单位分 batch
  - 不会把一个病例拆到多个 batch

审计结论：

- `gold_disease_idx` 在这里被生成，但后续热路径并没有直接消费这个字段；后面又用 `mondo_label -> disease_to_idx` 重新映射一次。
- `case_id` 命名空间只依赖文件 stem，不依赖目录层级，所以 train/test 中同 stem 文件会发生碰撞。

### 3.3 `src/models/hgnn_encoder.py`

职责：

- 两层 HGNN 编码 HPO 节点，输出 `Z[num_hpo, hidden_dim]`

关键逻辑：

- `_propagate()` 实现 `D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X`
- `forward()` 可接 scipy sparse 或 torch tensor

审计结论：

- 数学上支持任意 `H`，但当前运行中实际输入只有 `H_disease`。
- 文件头注释仍在描述“前半列病例边、后半列疾病边”的统一超图输入，这与当前热路径不一致。

### 3.4 `src/models/case_refiner.py`

职责：

- 对病例侧 active edge 上的节点表示做病例条件化细化

关键逻辑：

- `CaseConditionedRefiner.forward()`
  - 用当前病例的 active HPO 求 `case_ctx`
  - 用 `z / ctx / z*ctx / |z-ctx|` 进入 MLP 门控
  - 产出 `RefinedCaseNodeState`

审计结论：

- 它不会改 `H_case` 权重，也不会改疾病侧表示。
- 它只对病例侧活跃 `case-HPO` 边的节点表示起作用。

### 3.5 `src/models/readout.py`

职责：

- 从节点表示构造 `case_repr` 和 `disease_repr`

关键逻辑：

- `build_case_repr()`
  - 从 `H_case` 活跃边取 `edge_repr`
  - 用 leave-one-out attention 聚合到 `case_repr`
- `build_case_repr_from_refined()`
  - 如果 case_refiner 开启，则用 refined edge 表示
- `build_disease_repr()`
  - `H_disease^T @ Z`

审计结论：

- 病例侧和疾病侧路径完全分离。
- 即使 case_refiner 开启，也只影响 case side。

### 3.6 `src/models/scorer.py`

职责：

- 计算 `case_repr` 与 `disease_repr` 的 cosine score

审计结论：

- scorer 非常纯，当前没有其它打分器分支。
- 如果未来想切换 scorer，现有 train/eval config 并没有开放这个入口。

### 3.7 `src/models/model_pipeline.py`

职责：

- 编排 `encoder -> case_refiner -> readout -> scorer`

关键逻辑：

- `_validate_batch_graph()`
- `_prepare_h_disease()`
- `_build_gold_local()`
- `precompute_disease_side()`
- `forward()`

审计结论：

- 当前真实前向核心在 `forward():305-330`
  - encoder 明确只用 `H_disease`
  - `H_case` 仅在 case_refiner/readout 中使用
- `scores_global`、`H`、`case_padding_value` 都是面向旧接口/调试兼容的遗留能力，热路径默认关闭

### 3.8 `src/training/loss_builder.py`

职责：

- 构造 full-pool CE + poly + hard ranking loss

关键逻辑：

- `compute_loss()`
  - `scaled_scores = scores / tau`
  - `ce_loss_none = F.cross_entropy(scaled_scores, targets, reduction="none")`
  - `poly_epsilon > 0` 时叠加 poly 项
  - `hard_neg_indices` 存在时叠加 hard ranking 项

审计结论：

- 当前训练目标不是单纯 CE；是否附加 poly/hard neg 受 config 与默认值共同决定。

### 3.9 `src/training/hard_negative_miner.py`

职责：

- 从全疾病池中在线选取 hardest negatives

关键逻辑：

- 先把正类位置置 `-inf`
- 再取 top-k

审计结论：

- 当前 hard negative 不是采样局部候选池，而是从全局疾病池挖 top-k。

### 3.10 `src/training/trainer.py`

职责：

- 训练入口
- 构建模型配置
- 训练/验证循环
- checkpoint 与 history 落盘

关键函数：

- `build_model_config()`
- `run_one_epoch()`
- `main()`

审计结论：

- 当前训练端明确采用 `include_combined_h=False`
- 默认模型配置被代码二次组装，导致部分 YAML 其实不会直接生效
- `build_model_config()` 与 evaluator 里 `_build_model_config()` 重复实现，存在漂移风险

### 3.11 `src/evaluation/evaluator.py`

职责：

- 评估入口
- 测试集加载
- 模型重建与 checkpoint 加载
- full-pool 排序评估与结果导出

关键函数：

- `load_static_resources()`
- `load_test_cases()`
- `load_checkpoint_model()`
- `evaluate()`
- `save_results()`

审计结论：

- 评估链路数学上与训练一致
- 但它自己复制了一套 model config 组装逻辑，且校验更严格
- 当前评估配置下 `2978` 个病例全部可评估，`skip_reason=0`

### 3.12 `configs/train.yaml`

职责：

- `trainer.py` 默认配置；但不是 `run_full_train.cmd` 的主用配置

审计结论：

- 它不是当前脚本主链真正使用的配置
- 它看起来只写了 `full_pool_ce + temperature`
- 但代码默认还会启用 `poly_epsilon=2.0`、`hard_negative=True`、`k=10`、`top_m=3`、`weight=0.5`

### 3.13 `configs/train_pretrain.yaml`

职责：

- `run_full_train.cmd` 第一阶段真实使用配置

审计结论：

- 显式关闭 hard negative
- 显式设 `poly_epsilon: 0.0`
- 使用 `case_refiner`

### 3.14 `configs/train_finetune.yaml`

职责：

- `run_full_train.cmd` 第二阶段真实使用配置

审计结论：

- 当前主线 finetune 配置
- 使用 `case_refiner`
- 开启 `hpo_corruption_prob: 0.15`
- 开启 hard negative 与 poly loss

### 3.15 `configs/data.yaml` 与 `configs/data_llldataset_eval.yaml`

职责：

- `evaluator.py` 默认使用 `data.yaml`
- `run_full_train.cmd` 和 sweep 脚本真实使用 `data_llldataset_eval.yaml`

审计结论：

- `data_llldataset_eval.yaml` 才是脚本主链实际评估配置
- `data.yaml` 更多像默认/手工运行入口配置

### 3.16 `run_full_train.cmd` / `run_attention_residual_sweep.cmd`

职责：

- 当前代码库真正的实验编排入口

审计结论：

- 若只看 `trainer.py` 默认参数，会误判当前真实训练链
- 实验审计必须以这些脚本为准，而不是只看 `configs/train.yaml`

### 3.17 `src/graph/incidence_builder.py`

职责：

- 离线构建 `H_disease` 稀疏 triplets 的工具脚本

审计结论：

- 它不在当前 trainer/evaluator 运行链里
- 但它解释了 `load_disease_incidence()` 为什么支持 `rows/cols/vals/shape` 格式

## 4. 《设计意图 vs 实际实现差异审计》

| 设计意图 | 当前实际实现 | 结论 | 证据 |
|---|---|---|---|
| encoder 原则上只用 `H_disease`，避免 `H_case` 污染疾病空间 | 当前热路径确实只把 `H_disease` 送进 encoder | 一致 | `src/models/model_pipeline.py:305-309` |
| `H_case` 只在 readout / case_refiner / scorer 阶段进入 | 当前 `H_case` 只进入 `case_refiner` 和 `readout`，不进入 scorer 的结构定义本身，但 scorer 吃的是 case side 输出 | 基本一致 | `src/models/model_pipeline.py:316-330` |
| 做病例-疾病匹配，不是节点分类 | 当前 `scores.shape=[num_case, num_disease]`，用 cosine 做病例-疾病匹配 | 一致 | 单批实跑 + `src/models/scorer.py` |
| full-pool competition 为主 | 当前 loss 和 top-k 都在全疾病池上算 | 一致 | `scores.shape=(B,16443)`，`loss_builder.py` |
| cosine scorer | 当前 scorer 被 trainer/evaluator 硬编码为 cosine | 一致，但配置不可切换 | `trainer.build_model_config()` / `evaluator._build_model_config()` |
| `H=[H_case|H_disease]` 只作为统一表示，不应影响当前热路径 | 当前热路径完全跳过 `H` | 代码实现已偏向简化版，但文档仍停留在旧语义 | `include_combined_h=False` |
| 注意力残差 sweep 应该隔离单变量 | 当前 sweep 配置并不只改 `residual_uniform`，还改了 `case_refiner` 与训练数据 | 不一致 | `train_finetune_attn_ru020.yaml` vs `ru000/010/100` |
| 配置应当“所见即所得” | 当前多处存在代码默认值覆盖/补齐，用户看 YAML 不足以知道真实行为 | 不一致 | `trainer.py:519-538` |

## 5. 《问题清单（按严重级别排序）》

### P0

本次未发现“当前主链必然直接跑崩或目标空间明显错误”的 P0 级代码问题。

### [P1] 残差注意力 sweep 实际同时改变了多个变量，实验结果不可归因

**严重级别**：P1  
**文件路径**：`configs/train_finetune_attn_ru020.yaml`、`configs/train_finetune_attn_ru000.yaml`、`configs/train_finetune_attn_ru010.yaml`、`configs/train_finetune_attn_ru100.yaml`、`run_attention_residual_sweep.cmd`  
**函数/类名**：配置级问题，无单一函数  
**关键代码位置**：

- `ru020`: `model.case_refiner` 存在，且 `train_files` 多了 `all_diseases_5_to_15_profiles_minimal.xlsx`
- `ru000/010/100`: 没有 `model.case_refiner`
- `run_attention_residual_sweep.cmd` 直接把这四个配置当成一组 sweep

**当前行为**：

脚本把这四个配置当作 `residual_uniform` sweep 来跑，但 `ru020` 不仅改了 `residual_uniform=0.2`，还额外启用了 `case_refiner`，并增加了一份训练数据；其它三个配置没有这些变化。

**为什么有问题**：

这会导致实验结果无法归因到 `residual_uniform` 本身。任何性能变化，都可能来自：

- 是否启用 `case_refiner`
- 是否加入额外训练文件
- `residual_uniform`

**对训练/评估的影响**：

当前这组 sweep 结果不适合拿来判断 attention residual 的真实贡献，也不适合作为后续模型选择依据。

**建议优先如何验证**：

先固定训练数据与 `case_refiner` 开关，只保留 `residual_uniform` 一个变量，重跑 sweep。

**证据链**：

- `run_attention_residual_sweep.cmd:13-24, 41-49`
- `configs/train_finetune_attn_ru020.yaml:1-35`
- `configs/train_finetune_attn_ru000.yaml:1-30`
- `configs/train_finetune_attn_ru010.yaml:1-30`
- `configs/train_finetune_attn_ru100.yaml:1-31`

### [P1] `configs/train.yaml` 的真实 loss 语义与文件表面写法不一致，默认会静默启用 poly loss 和 hard negative

**严重级别**：P1  
**文件路径**：`configs/train.yaml`、`src/training/trainer.py`、`src/training/loss_builder.py`  
**函数/类名**：`trainer.main()`、`build_loss()`  
**关键代码位置**：

- `configs/train.yaml:38-40` 只写了 `loss_name` 与 `temperature`
- `trainer.py:519-538` 用代码默认值补出 `hard_negative_cfg`
- `trainer.py:538` 把 `poly_epsilon` 默认补成 `2.0`

**当前行为**：

如果用户直接执行 `python -m src.training.trainer`，默认会读 `configs/train.yaml`。虽然 YAML 表面没有声明 hard negative 和 poly，但代码会默认打开：

- `poly_epsilon = 2.0`
- `use_hard_negative = True`
- `k = 10`
- `top_m = 3`
- `weight = 0.5`
- `margin = 0.1`

**为什么有问题**：

用户只看 `train.yaml` 会误以为这是“纯 full-pool CE”，实际上不是。实验复现和结果解释会被误导。

**对训练/评估的影响**：

默认入口与配置文件表述不一致，容易导致“同样叫 train.yaml，但实际训练目标不是预期目标”。

**建议优先如何验证**：

打印训练启动时的最终 loss 配置；确认默认入口是否还要保留这些隐式默认项。

**证据链**：

- `src/training/trainer.py:26, 455-458, 519-538`
- `configs/train.yaml:38-40`
- `src/training/loss_builder.py:88-149`

### [P1] `case_id` 命名空间只用文件 stem，train/test 中同 stem 文件会发生碰撞，跨阶段分析很容易误判为同一病例

**严重级别**：P1  
**文件路径**：`src/data/dataset.py`、`src/evaluation/evaluator.py`、`configs/train_finetune.yaml`、`configs/data_llldataset_eval.yaml`  
**函数/类名**：`load_case_files()`、`load_test_cases()`  
**关键代码位置**：

- `dataset.py:81-82`
- `evaluator.py:420-423`

**当前行为**：

训练和评估都用 `stem + "_" + case_id` 生成新 `case_id`。  
当前主线配置下，`train_finetune.yaml` 与 `data_llldataset_eval.yaml` 中都存在 `DDD.csv`；实测出现 `761` 个相同的 prefixed `case_id`，例如 `DDD_case_1`。

**为什么有问题**：

虽然本次抽检后发现这些碰撞样本的 label/HPO 集都不同，不能直接判定为真实重复病例，但它们会在日志、CSV、人工排查、跨阶段 join、外部分析脚本里被误认为同一病例。

**对训练/评估的影响**：

当前主链内部因为 train/eval 分开跑，暂时没直接打坏训练逻辑；但它会显著污染后续分析、误差归因与可追溯性。

**建议优先如何验证**：

优先把前缀从 `stem` 升级为包含 split 或相对路径层级的信息，例如 `train_DDD_case_1` / `test_DDD_case_1`。

**证据链**：

- `src/data/dataset.py:81-82`
- `src/evaluation/evaluator.py:421-423`
- 实测统计：
  - `prefixed_case_id_overlap = 761`
  - `DDD.csv <-> DDD.csv: 761`
  - `identical_label = 0`
  - `identical_hpo = 0`
  - `identical_both = 0`

### [P2] 文档与实现已经漂移：当前热路径不再把 `H_case` 混入 encoder，但多个核心文件仍在描述统一超图 `H=[H_case|H_disease]`

**严重级别**：P2  
**文件路径**：`src/models/hgnn_encoder.py`、`src/data/build_hypergraph.py`、`src/models/model_pipeline.py`、`src/training/trainer.py`、`src/evaluation/evaluator.py`  
**函数/类名**：`HGNNEncoder.forward()`、`build_batch_hypergraph()`、`ModelPipeline.forward()`  
**关键代码位置**：

- `hgnn_encoder.py:1-34`
- `build_hypergraph.py:1-5, 188-191`
- `trainer.py:270-281`
- `evaluator.py:612-623`
- `model_pipeline.py:305-330`

**当前行为**：

注释和字段名还在强调统一超图 `H`，但当前训练/评估热路径已经统一传 `include_combined_h=False`，且 encoder 只吃 `H_disease`。

**为什么有问题**：

这会严重误导后续审计、重构和新实验设计。仅看文档会得出错误结论：以为 `H_case` 还在污染 encoder。

**对训练/评估的影响**：

对当前数学结果无直接破坏，但对理解和后续改动风险很大。

**建议优先如何验证**：

在报告和注释层面先统一口径：当前热路径是 disease-only encoder；`H` 只是兼容字段。

**证据链**：

- 单批实跑 `batch_graph_has_H = false`
- `ModelPipeline.forward()` 只调用 `self.encoder(prepared_h_disease)`

### [P2] 索引命名混乱：`gold_disease_idx`、`gold_disease_cols_global`、`gold_disease_cols_local` 是三种不同空间，当前靠代码映射维持正确，但极易二次出错

**严重级别**：P2  
**文件路径**：`src/data/dataset.py`、`src/data/build_hypergraph.py`、`src/models/model_pipeline.py`  
**函数/类名**：`load_case_files()`、`build_batch_hypergraph()`、`_build_gold_local()`  
**关键代码位置**：

- `dataset.py:91-109`
- `build_hypergraph.py:224-236`
- `model_pipeline.py:231-250`

**当前行为**：

- `gold_disease_idx`：纯 disease index
- `gold_disease_cols_global`：批次拼接矩阵 `[H_case|H_disease]` 中的列号，值为 `num_case + disease_idx`
- `gold_disease_cols_local`：当前 `scores[:, disease]` 空间里的局部索引；在 full-pool 下它又等于 `disease_idx`

**为什么有问题**：

三者名字都像“疾病索引”，但语义不同；当前热路径因为 `_build_gold_local()` 显式做了一次映射，所以没出错，但只要后续有人直接拿错字段，就会出现典型的 `+num_case` 偏移问题。

**对训练/评估的影响**：

当前链路可跑通，但这是最容易继续踩坑的索引脆弱点之一。

**建议优先如何验证**：

先统一命名，例如：

- `gold_disease_idx`
- `gold_disease_col_in_combined_h`
- `gold_disease_idx_in_score_pool`

**证据链**：

- 单批实跑：
  - `train_first_gold_global = 5333`
  - `train_first_gold_local = 5329`
  - `train_first_disease_global_col = 4`
  - 说明 `gold_global = num_case + gold_local`

### [P2] trainer 与 evaluator 各自复制了一套模型配置装配逻辑，未来极易出现“训练结构”和“评估结构”悄悄漂移

**严重级别**：P2  
**文件路径**：`src/training/trainer.py`、`src/evaluation/evaluator.py`  
**函数/类名**：`build_model_config()`、`_build_model_config()`  
**关键代码位置**：

- `trainer.py:417-450`
- `evaluator.py:222-258`

**当前行为**：

训练和评估都自己拼一份 `ModelPipeline` 配置，代码目前几乎相同，但不是同一函数。

**为什么有问题**：

一旦未来某侧先改了 `case_refiner/readout/scorer/outputs` 拼装规则，另一侧没同步，就会出现：

- checkpoint 能训不能评
- 评估结构与训练结构不一致
- 同名配置实际含义不同

**对训练/评估的影响**：

当前尚未观察到直接不一致，但这是高概率的维护性风险。

**建议优先如何验证**：

把模型配置装配提取成单一共享函数；trainer/evaluator 都只调用这一份。

**证据链**：

- 两处函数实现内容高度重复

### [P3] 多个配置项当前没有完全按“用户看到的 YAML”方式生效，存在死参数/半死参数

**严重级别**：P3  
**文件路径**：`configs/train.yaml`、`src/training/trainer.py`、`src/evaluation/evaluator.py`、`src/models/model_pipeline.py`、`src/models/readout.py`  
**函数/类名**：`build_model_config()`、`_build_model_config()`、`ModelPipeline.forward()`  
**关键代码位置**：

- `trainer.py:423-436`
- `evaluator.py:231-244`
- `model_pipeline.py:361-391`
- `readout.py:144-149, 257-264`

**当前行为**：

- `train.num_workers` 没有被任何 DataLoader 使用
- scorer 类型被硬编码为 cosine，YAML 没有真实切换能力
- encoder 类型被硬编码为 hgnn
- `outputs.include_global_scores`、`outputs.return_intermediate` 没有从 YAML 透传
- `readout.return_attention` 即使开了，热路径默认也不会把 attention 信息返回给 trainer/evaluator

**为什么有问题**：

这会导致“我改了配置但为什么没效果”的隐性问题。

**对训练/评估的影响**：

当前主要影响可观测性和配置可信度，不是核心数学错误。

**建议优先如何验证**：

打印训练启动时的最终展开配置，并区分“用户 YAML”与“代码默认/强制项”。

**证据链**：

- `num_workers` 搜索无消费点
- `outputs` 由代码硬编码为 `False/True`

### [P3] `train_dir` 路径模式与 `train_files` 模式对文件格式支持不一致

**严重级别**：P3  
**文件路径**：`src/training/trainer.py`、`src/evaluation/evaluator.py`  
**函数/类名**：`list_case_files()`、`_resolve_case_files_from_paths()`  
**关键代码位置**：

- `trainer.py:38-57`
- `evaluator.py:47-86`

**当前行为**：

- `train_files` 模式支持 csv/xlsx，因为最终走 `read_case_table_file()`
- `train_dir` / evaluator 内部解析目录时却只扫描 `*.xlsx` 或有限后缀

**为什么有问题**：

同一个项目里，两种等价配置方式的行为不完全一致。

**对训练/评估的影响**：

当前主线配置都用 `train_files` / `test_files`，所以未直接触发；但它是明显的脆弱边角。

**建议优先如何验证**：

统一目录扫描和单文件读取的后缀支持规则。

**证据链**：

- `trainer.list_case_files()`
- `evaluator._resolve_case_files_from_paths()`

## 6. 《实现难点与脆弱点》

### 6.1 最难的是“同一个疾病标签在不同空间里有三种编号”

这是当前框架最容易继续出错的地方：

- 数据表里的 `gold_disease_idx`
- 批次拼接矩阵里的 `gold_disease_cols_global`
- scorer 空间里的 `gold_disease_cols_local`

当前代码能跑通，是因为 `ModelPipeline._build_gold_local()` 还在显式做映射；但这个设计天然脆弱。

### 6.2 当前项目已经从“统一超图 H”演化到了“disease-only encoder + case-side readout”，但文档和兼容字段还没同步清理

这会让任何新接手的人都误判真实链路，尤其容易错误地把 `H_case` 污染 encoder 当成当前 bug 去修，结果反而把现有正确实现改坏。

### 6.3 配置层存在多个版本并存，且默认入口与脚本入口不一致

当前至少存在三层入口：

- 直接跑 `trainer.py/evaluator.py` 的默认配置
- `run_full_train.cmd` 的 staged pipeline
- `run_attention_residual_sweep.cmd` 的 sweep pipeline

如果不先锁定“当前到底用哪套配置”，后续所有实验结论都可能对错对象。

### 6.4 病例侧增强模块天然很难归因

`case_refiner` 和 readout attention 都只作用在 case side，而 disease side 是固定全局空间。  
这意味着性能波动很容易同时受到以下因素影响：

- `hpo_dropout_prob`
- `hpo_corruption_prob`
- `case_refiner`
- `readout.attention`
- `residual_uniform`

如果实验配置不严格控变量，就很难知道提升来自哪一层。

### 6.5 数据集分割与命名空间没有被代码强约束

当前 train/test 的隔离主要依赖“你在 YAML 里选了哪些文件”，而不是程序自动验证：

- 是否同 stem
- 是否 case_id 碰撞
- 是否语义上同一病例
- 是否跨 split 混入

这使得数据问题更容易以“后处理统计异常”的形式出现，而不是训练时直接报错。

## 7. 《对当前实验结果影响最大的前三个问题》

### 7.1 残差注意力 sweep 被混入了 `case_refiner` 与额外训练数据变化

这会直接让当前 sweep 结论不可归因，是对实验结果解释伤害最大的已确认问题。

### 7.2 默认配置 `train.yaml` 的真实 loss 不是表面看到的 loss

如果有人直接用默认入口复现实验，很容易拿错目标函数，导致结果根本不是以为的那一组。

### 7.3 `case_id` 命名空间碰撞会污染跨阶段分析与结果对账

虽然这次没有直接证明 train/test 真实重复病例，但 `DDD_case_1` 这类碰撞已经实测存在，后续任何按 `case_id` 做 join 的分析都可能被误导。

## 8. 《建议优先验证的修复顺序》

1. 先冻结“当前真实实验入口”和“当前真实配置集合”。
   先明确报告、脚本、后续分析统一以 `run_full_train.cmd + train_pretrain.yaml + train_finetune.yaml + data_llldataset_eval.yaml` 为主线，避免继续拿 `train.yaml` 当主实验配置。

2. 先修实验可归因性，再谈模型结构改动。
   先把 attention residual sweep 改成真正单变量 sweep，否则后续所有分析都建立在不可靠实验上。

3. 先统一索引命名与断言，再做功能扩展。
   把 `gold_disease_idx / gold_disease_cols_global / gold_disease_cols_local` 重新命名并补断言，避免后续再出现列号/索引错用。

4. 统一 trainer/evaluator 的模型配置装配逻辑。
   先消除训练与评估的潜在结构漂移风险。

5. 最后再清理配置死参数和旧文档。
   这一步不一定立刻影响指标，但能显著降低后续维护误判率。

## 9. 《给 GPT Pro 的交接摘要》

当前 `D:\RareDisease-traindata` 的真实 HGNN 热路径已经不是“统一超图 `H=[H_case|H_disease]` 一起进 encoder”，而是明确的 **disease-only encoder**：`ModelPipeline.forward()` 只把 `H_disease` 送进 `HGNNEncoder`，得到全局 HPO 节点表示 `Z`；随后 `H_case` 只在 `case_refiner` 和 `readout` 阶段进入病例侧表示构造。疾病侧表示由 `H_disease^T @ Z` 直接读出，最终由 `CosineScorer` 产生 `scores[B, num_disease]`。我已实跑单批前向，训练与评估的 `scores.shape` 都是 `(4, 16443)`，说明当前确实是在全疾病池上做病例-疾病匹配，训练与评估的分数定义一致。`H_disease` 是真实带权稀疏矩阵，`H_case` 当前是严格二值矩阵。

当前最值得继续深挖的不是“`H_case` 是否污染 encoder”这个旧问题，因为就现在这版代码看，它没有进入 encoder；真正的问题在于 **配置/版本/命名漂移**。首先，`run_full_train.cmd` 的真实主链是 `train_pretrain.yaml -> train_finetune.yaml -> data_llldataset_eval.yaml`，不是默认 `train.yaml`；而 `train.yaml` 又存在隐式默认项，虽然表面只写了 `full_pool_ce + temperature`，代码实际上会默认补出 `poly_epsilon=2.0` 和一整套 hard negative 参数。其次，当前 `run_attention_residual_sweep.cmd` 所用四个 finetune 配置并不是单变量 sweep：`ru020` 不仅改了 `residual_uniform`，还额外启用了 `case_refiner` 并增加了一份训练数据，因此这组实验结果不可归因。再次，索引空间存在明显历史包袱：`gold_disease_idx`、`gold_disease_cols_global`、`gold_disease_cols_local` 是三种不同语义的索引，当前能跑通依赖 `ModelPipeline._build_gold_local()` 做转换，但命名非常容易误导后续开发。

关于数据问题，我没有直接证实 train/test 有真实重复病例泄露，但发现了一个必须警惕的命名空间风险：训练和评估都用 `stem + "_" + case_id` 生成新病例 ID；在当前主线配置下，`DDD.csv` 的 train/test 之间实测出现 `761` 个相同的 prefixed `case_id`，例如 `DDD_case_1`。进一步核查后，这些碰撞样本的 label 和 HPO 集都不同，因此更准确的结论是“**命名空间碰撞**”，而不是“已确认的重复病例泄露”。GPT Pro 如果要继续分析，优先建议沿着以下顺序推进：先锁定主实验入口和真正生效的配置，再修正 confounded sweep，然后统一索引命名与 trainer/evaluator 共享配置装配逻辑，最后再讨论更深层的结构优化或性能瓶颈。
