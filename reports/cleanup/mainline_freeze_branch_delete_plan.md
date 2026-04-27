# Mainline Freeze Branch Delete Plan

## 1. Final Mainline Decision

最终 DDD 主线固定为 `HGNN baseline + validation-selected grid rerank (DDD_top1)`。

- DDD baseline: top1/top3/top5/recall@50 = `0.3022/0.4442/0.4967/0.7451`
- DDD final grid rerank: top1/top3/top5/recall@50 = `0.3693/0.4875/0.5506/0.7451`
- DDD 不再使用 gated rerank 作为主线。
- DDD 不再使用 ontology-aware HN exact training 作为主线。
- DDD 不再使用 test-side exploratory grid/gate 作为正式结果。
- mimic 主线保持 `HGNN_SimilarCase_Aug`，不修改 mimic 代码或结果逻辑。

## 2. Mimic Mainline Location

已找到上次完成的 mimic 主线模块：

- mimic fixed config: `configs/mimic_similar_case_aug.yaml`
- frozen mimic config: `reports/mimic_next/frozen_similar_case_aug_config.json`
- mimic entry: `tools/run_mimic_similar_case_aug.py`
- all-dataset SimilarCase entry: `tools/run_similar_case_aug_all_datasets.py`
- mimic final report: `reports/mimic_next/final_similar_case_aug_report.md`
- all-dataset report: `reports/similar_case_all/final_similar_case_all_report.md`
- all-dataset output: `outputs/similar_case_all/similar_case_all_fixed_ranked_candidates.csv`

mimic 主线应只对 `mimic_test` 启用；DDD 使用 validation-selected grid rerank；其他 dataset 保持 HGNN baseline。

## 3. A. 必须保留

- HGNN encoder 和 `src/models/`。
- 原始 exact evaluation：`src/evaluation/evaluator.py` 和已有 exact evaluation 输出。
- DDD validation-selected grid rerank 所需代码：`tools/run_top50_evidence_rerank.py` 的 grid/fixed-eval 能力。
- DDD final 权重和 fixed 结果：
  - `outputs/rerank/ddd_val_selected_grid_weights.json`
  - `outputs/rerank/ddd_rerank_fixed_test_metrics.csv`
  - `outputs/rerank/ddd_rerank_fixed_test_by_dataset.csv`
- mimic 主线代码与产物：
  - `configs/mimic_similar_case_aug.yaml`
  - `reports/mimic_next/frozen_similar_case_aug_config.json`
  - `tools/run_mimic_similar_case_aug.py`
  - `reports/mimic_next/final_similar_case_aug_report.md`
- SimilarCase-Aug 主线代码与产物：
  - `tools/run_similar_case_aug_all_datasets.py`
  - `reports/similar_case_all/`
  - `outputs/similar_case_all/`
- 最终论文总结报告：
  - `reports/ddd_improvement/ddd_final_experiment_summary.md`
  - `reports/ddd_improvement/ddd_final_method_comparison.md`
  - `reports/ddd_improvement/ddd_paper_table_plan.md`
  - `reports/ddd_improvement/ddd_next_experiment_recommendations_after_hn.md`

## 4. B. 删除支线代码

删除或回滚以下 DDD ontology-aware HN 支线代码：

- `configs/train_finetune_ddd_ontology_hn.yaml`
- `src/training/hard_negative_pools.py`
- `reports/ddd_improvement/run_ddd_hn_candidate_pool_dryrun.py`

修复以下 import / runtime 依赖：

- 从 `src/training/trainer.py` 移除 `build_hard_negative_pool_builder` import 和 candidate pool 构建逻辑。
- 从 `src/runtime_config.py` 移除 `loss.hard_negative.candidate_pools` 配置透传，恢复为通用 hard negative 配置。
- 保留 `src/training/hard_negative_miner.py`，因为它是历史通用 hard negative miner，不作为本轮删除目标。

清理 gated rerank 支线逻辑：

- `tools/run_top50_evidence_rerank.py` 保留 grid / fixed-eval；移除 gated search 入口和 gate mask 逻辑。
- `reports/ddd_improvement/run_ddd_validation_selected_rerank.py` 改为 grid-only validation-selected rerank 复现脚本，不再输出 gated payload。

## 5. C. 移到 archive 或删除的支线输出

本轮直接删除以下支线输出，并先写入 manifest：

- `outputs/ddd_ontology_hn/`
- `reports/ddd_improvement/ddd_hn_candidate_pool_audit.md`
- `reports/ddd_improvement/ddd_hn_candidate_pool_stats.csv`
- `reports/ddd_improvement/ddd_hn_dryrun_samples.csv`

不删除以下文件，即使其中包含 gated 行：

- `outputs/rerank/ddd_rerank_fixed_test_metrics.csv`
- `outputs/rerank/ddd_rerank_fixed_test_by_dataset.csv`

原因：它们是用户指定必须保留的固定结果文件；最终主线入口只读取 `validation_grid_DDD_top1` 行。

## 6. D. 保留但标注为 supplementary 的报告

以下报告不进主线，但保留用于论文附录和追溯：

- `reports/ddd_improvement/ddd_hn_failure_analysis.md`
- `reports/ddd_improvement/ddd_final_method_comparison.md`
- `reports/ddd_improvement/ddd_paper_table_plan.md`
- `reports/ddd_improvement/ddd_final_experiment_summary.md`
- `reports/ddd_improvement/ddd_next_experiment_recommendations_after_hn.md`
- `reports/ddd_improvement/ddd_next_stage_execution_report.md`
- `reports/ddd_improvement/ddd_validation_selected_rerank_report.md`

## 7. New Mainline Entrypoint

新增统一入口：

- `configs/mainline_hgnn_val_grid_rerank.yaml`
- `tools/run_mainline_hgnn_val_grid_rerank_all.py`

该入口固定执行：

- `DDD`: validation-selected grid rerank (`DDD_top1`)
- `mimic_test`: existing SimilarCase-Aug mimic mainline
- other datasets: HGNN baseline

输出目录：

- `outputs/mainline_hgnn_val_grid_rerank_all/`
- `reports/mainline_hgnn_val_grid_rerank_all/`

## 8. Safety Checks

清理后必须确认：

- 不再引用 `train_finetune_ddd_ontology_hn.yaml`。
- 不再引用 `HardNegativeCandidatePoolBuilder`。
- 不再引用 `run_ddd_hn_candidate_pool_dryrun.py`。
- 主线仍能 `py_compile`。
- 不修改 HGNN encoder。
- 不覆盖原始 exact evaluation。
- 不删除 mimic 主线。
- 不删除 SimilarCase-Aug 主线。
