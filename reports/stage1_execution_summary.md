# 第一阶段执行说明

本轮工作严格以 [hgnn_root_cause_and_fix_plan.md](D:/RareDisease-traindata/reports/hgnn_root_cause_and_fix_plan.md) 作为执行规格，只落地第一阶段“实现正确性与实验治理”修复，不做第二阶段结构创新。

## 1. 本轮修改了哪些文件

### 修改文件

- `run_full_train.cmd`
- `configs/train_pretrain.yaml`
- `configs/train_finetune.yaml`
- `configs/data_llldataset_eval.yaml`
- `configs/train.yaml`
- `src/training/trainer.py`
- `src/evaluation/evaluator.py`
- `src/data/dataset.py`
- `src/data/build_hypergraph.py`
- `src/models/model_pipeline.py`
- `src/training/loss_builder.py`

### 新增文件

- `src/runtime_config.py`
- `configs/train_finetune_clean_baseline.yaml`

## 2. 每个文件改了什么

### `src/runtime_config.py`

- 提取 trainer/evaluator 共享的 `ModelPipeline` 配置 builder
- 提供显式 `loss` 解析逻辑，消灭会改变训练语义的 hidden default
- 提供 effective config 的打印与 YAML 保存工具
- 固化“当前唯一可信主线”描述

### `src/training/trainer.py`

- 去掉 `train.yaml` 的隐式默认入口，改为必须显式传 `--config`
- 改为调用共享 model builder
- 启动时解析显式 `loss` 语义，不再静默补齐 `poly=2.0`、`hard_negative=True`
- 训练开始前打印并保存 `effective_config.yaml`
- 训练时改用新字段 `gold_disease_idx_in_score_pool`

### `src/evaluation/evaluator.py`

- 去掉 `data.yaml/train.yaml` 的隐式默认入口，改为必须显式传参
- 改为调用共享 model builder
- 评估时打印 effective config 摘要，并在 `evaluation/` 下保存对应 YAML
- 加入 train/test `case_id` overlap 自动检查
- 评估时改用新字段 `gold_disease_idx_in_score_pool`

### `src/data/dataset.py`

- 增加统一的 namespaced `case_id` 构造逻辑
- 训练读入链路从 `stem + "_" + raw_case_id` 改为 `split::relative/path::raw_case_id`
- 增加只读取 namespaced `case_id` 集合的辅助函数，供 overlap 检查复用
- 明确 `gold_disease_idx` 只是纯 disease index 语义

### `src/data/build_hypergraph.py`

- 明确当前热路径不依赖 combined `H`
- 将 batch 图中的疾病标签语义拆清：
  - `gold_disease_idx`
  - `gold_disease_col_in_combined_h`
  - 兼容别名 `gold_disease_cols_global`
- 在关键位置增加索引合法性断言

### `src/models/model_pipeline.py`

- 明确三套 gold disease 索引语义
- `_build_gold_local()` 增加语义一致性断言
- 输出新字段：
  - `gold_disease_col_in_combined_h`
  - `gold_disease_idx_in_score_pool`
- 保留旧字段别名，避免第一阶段修复破坏主线

### `src/training/loss_builder.py`

- 把默认 `poly_epsilon` 改为 `0.0`
- 把默认 `hard_weight/top_m` 改为 no-op
- 避免 loss 模块自身继续暗带“增强默认值”

### `configs/train.yaml`

- 明确标注为“非主线样例配置”
- 写清当前唯一可信主线是 `run_full_train.cmd + train_pretrain.yaml + train_finetune.yaml + data_llldataset_eval.yaml`
- 显式写出 `poly=0.0`、`hard_negative=off`
- 改用独立 `save_dir`，避免继续伪装成主线默认输出

### `configs/train_pretrain.yaml` / `configs/train_finetune.yaml` / `configs/data_llldataset_eval.yaml`

- 在文件头部补充可信主线说明，减少入口歧义

### `run_full_train.cmd`

- 在脚本头部明确标注“当前唯一可信主线入口”

### `configs/train_finetune_clean_baseline.yaml`

- 新建第一阶段 clean baseline 候选配置
- 关键策略：
  - `case_refiner` 关闭
  - `poly` 关闭
  - `hard_negative` 关闭
  - `hpo_corruption_prob` 关闭
  - `hpo_dropout_prob` 关闭
  - `readout.residual_uniform=1.0`

## 3. 对应解决了报告中的哪个问题

### 问题：主线入口与默认配置漂移

对应修复：

- `trainer.py` / `evaluator.py` 改为必须显式传配置
- `run_full_train.cmd` 与主线配置头部增加说明
- `configs/train.yaml` 明确降级为非主线样例

### 问题：effective config 不可见，实验不可追踪

对应修复：

- 新增 `src/runtime_config.py`
- 训练和评估都打印并保存 `effective_config.yaml`

### 问题：hidden default 偷偷改变 loss 语义

对应修复：

- `resolve_loss_config()` 改为“写什么跑什么”
- `trainer.py` 不再用 `poly=2.0 / hard_negative=True` 这类隐式补齐
- `loss_builder.py` 默认值改为 no-op

### 问题：trainer/evaluator 双份 model builder，未来易漂移

对应修复：

- 提取共享 builder 到 `src/runtime_config.py`
- trainer/evaluator 统一调用这一份

### 问题：gold disease 三套索引语义混杂

对应修复：

- `build_hypergraph.py`、`model_pipeline.py`、`dataset.py` 中显式区分三种空间
- 在转换点加入断言
- 对外输出新字段，同时保留兼容别名

### 问题：`case_id` 命名空间碰撞风险

对应修复：

- 改为 `split::relative/path::raw_case_id`
- evaluator 中增加 train/test overlap 自动检查
- 明确这修的是命名空间风险，不等于已确认真实泄露

### 问题：缺少 clean baseline 候选

对应修复：

- 新增 `configs/train_finetune_clean_baseline.yaml`

## 4. 如何验证修改已生效

本轮做了最小验证，没有跑完整训练：

- `python -m compileall src` 通过，说明主要代码改动无语法错误
- `python -m src.training.trainer --help` 显示 `--config` 已变为必填
- `python -m src.evaluation.evaluator --help` 显示 `--data_config_path / --train_config_path` 已变为必填
- 直接用脚本检查 `train_finetune_clean_baseline.yaml`，确认：
  - encoder 仍是 `hgnn`
  - `case_refiner_enabled=False`
  - loss 为 CE-only，无 poly/hard negative
- 直接读取 `DDD` train/test，验证新 namespaced `case_id` 后：
  - train case 数量正常
  - test case 数量正常
  - overlap_count = 0

## 5. 还有哪些问题故意留到第二阶段

- 不改 disease-only encoder 主路线
- 不把 `H_case` 拉回 encoder
- 不改 scorer
- 不引入新 loss
- 不重写 readout 数学结构
- 不在本轮判断 attention / case_refiner 的最终去留
- 不在本轮做提分导向调参

这些都属于第二阶段“结构优化与单变量实验验证”的范围。

## 6. 下一步最推荐先做的 3 个实验

### 实验 1：主线当前配置 vs clean baseline

- 对照：
  - `configs/train_finetune.yaml`
  - `configs/train_finetune_clean_baseline.yaml`
- 目的：
  - 先确认 clean baseline 是否更稳定、更适合作为可信参照

### 实验 2：在 clean baseline 上只打开 readout attention 的单变量实验

- 对照：
  - clean baseline
  - clean baseline + `residual_uniform` 从 `1.0` 降到 `0.2` 或其他目标值
- 目的：
  - 检查 attention/readout weighting 是否真有增益

### 实验 3：在 clean baseline 上只打开 `case_refiner` 的单变量实验

- 对照：
  - clean baseline
  - clean baseline + `case_refiner.enabled=true`
- 目的：
  - 判断 `case_refiner` 是否值得进入第二阶段继续优化
