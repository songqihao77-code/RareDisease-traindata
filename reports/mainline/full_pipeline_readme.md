# RareDisease HGNN full mainline pipeline

## 一键运行命令

```cmd
D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode full
```

已有 finetune checkpoint 时只重跑评估和后处理：

```cmd
D:\python\python.exe tools\run_full_mainline_pipeline.py --config-path configs\mainline_full_pipeline.yaml --mode eval_only
```

## 每一步做了什么

1. `stage1_pretrain`: 调用 `python -m src.training.trainer` 运行 pretrain，并把 `save_dir` 写到 `D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage1_pretrain`。
2. `stage2_finetune`: 调用 `python -m src.training.trainer` 运行 finetune，`init_checkpoint_path` 指向 stage1 的 `best.pt`。
3. `stage3_exact_eval`: 调用 `python -m src.evaluation.evaluator`，使用 stage2 的 `best.pt` 生成 exact evaluation。
4. `stage4_candidates`: 调用 `tools/export_top50_candidates.py`，分别导出 validation/test top50 candidates。
5. `stage5_ddd_rerank`: 调用 `tools/run_top50_evidence_rerank.py --protocol validation_select`，只在 validation candidates 上选权重，再对 test candidates 固定评估一次。
6. `stage6_mimic_similar_case`: 调用 `tools/run_mimic_similar_case_aug.py`，使用 validation-selected fixed SimilarCase 参数，对 test candidates 固定评估一次。
7. final aggregation: DDD 使用 rerank，mimic 使用 SimilarCase，其他数据集使用 exact baseline。

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

## 如何确认 mimic alias 已正确匹配

检查 `mainline_final_metrics_with_sources.csv` 中 `mimic_test`、`mimic_test_recleaned` 或 `mimic_test_recleaned_mondo_hpo_rows` 对应行，`module_applied` 应为 `similar_case_fixed_test`。

## 如何确认 DDD 和 mimic 已进入最终表

检查 `mainline_final_metrics_with_sources.csv`：

- `DDD` 的 `module_applied` 应为 `ddd_validation_selected_grid_rerank`。
- mimic alias 数据集的 `module_applied` 应为 `similar_case_fixed_test`。
- `HMS/LIRICAL/MME/MyGene2/RAMEDIS` 的 `module_applied` 应为 `hgnn_exact_baseline`。
