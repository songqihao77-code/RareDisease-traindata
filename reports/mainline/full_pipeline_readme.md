# RareDisease HGNN full mainline pipeline

## 一键运行命令

```cmd
D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode full
```

已有 finetune checkpoint 时，只重跑评估和后处理：

```cmd
D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode eval_only
```

## 每一步做什么

1. `stage1_pretrain`: 调用 `python -m src.training.trainer` 运行 pretrain，并把 `save_dir` 写到 `outputs\mainline_full_pipeline\stage1_pretrain`。
2. `stage2_finetune`: 调用 `python -m src.training.trainer` 运行 finetune，`init_checkpoint_path` 指向 stage1 的 `best.pt`。
3. `stage3_exact_eval`: 调用 `python -m src.evaluation.evaluator`，使用 stage2 的 `best.pt` 生成 exact evaluation。
4. `stage4_candidates`: 调用 `tools/export_top50_candidates.py`，分别导出 validation/test top50 candidates，并包含 DDD core-missing rerank 所需的核心 HPO 覆盖特征。
5. `stage5_ddd_rerank`: 调用 `tools/run_top50_evidence_rerank.py --protocol validation_select`，只在 validation candidates 上选择权重，再对 test candidates 固定评估一次。
6. `stage6_mimic_similar_case`: 调用 `tools/run_mimic_similar_case_aug.py`，默认启用 validation-selected gated SimilarCase，对 test candidates 固定评估一次。
7. final aggregation: DDD 使用 rerank，mimic 使用 SimilarCase，其它数据集使用 exact baseline。

## 主线保留的两个提分模块

DDD rerank 使用 `w_core_missing` 作为弱惩罚。候选导出阶段会计算 `core_missing_semantic_top5`，表示候选疾病 top5 加权核心 HPO 中未被病例精确或语义覆盖的比例。重排阶段会保护 HGNN 原始 rank <= 3 的候选，避免强行打掉高置信候选。

mimic SimilarCase 默认使用 gated rerank。配置入口在 `configs/mainline_full_pipeline.yaml` 的 `postprocess.mimic` 下，可以通过 `gated_rerank_enabled: false` 临时关闭；默认网格包含 `gated_sim_weight`、`gated_ic_weight`、`gated_agree_boost` 和 `gated_protect_bonus`。

## 为什么 DDD 是评估后 rerank

DDD 模块只读取 evaluation 后导出的 top50 candidates 和 validation 选择出的固定权重，不改变 HGNN encoder、loss、sampler 或训练 checkpoint。因此它是 dataset-specific post-processing，不属于 `trainer.py` 的训练逻辑。

## 为什么 mimic 是评估后 SimilarCase

mimic SimilarCase 模块只在 HGNN candidates 上融合相似病例证据，并且参数由 validation 选择后在 test 固定评估。它不反向传播、不更新 checkpoint，也不改变训练数据采样，因此属于 evaluation 后模块。

## 最终主表

论文主表读取：

`outputs/mainline_full_pipeline/mainline_final_metrics_with_sources.csv`

简版指标表是：

`outputs/mainline_full_pipeline/mainline_final_metrics.csv`

## 如何确认没有 checkpoint 混用

检查 `outputs/mainline_full_pipeline/run_manifest.json` 中的 `finetune_checkpoint`、`validation_candidates_metadata.checkpoint_path`、`test_candidates_metadata.checkpoint_path` 和 final table 的 `checkpoint_path`。这些路径必须一致。

## 如何确认 DDD 和 mimic 已进入最终表

检查 `mainline_final_metrics_with_sources.csv`：

- `DDD` 的 `module_applied` 应为 `ddd_validation_selected_grid_rerank`。
- mimic alias 数据集的 `module_applied` 应为 `similar_case_fixed_test`。
- `HMS/LIRICAL/MME/MyGene2/RAMEDIS` 的 `module_applied` 应为 `hgnn_exact_baseline`。
