# Mainline Freeze After Branch Cleanup

## 1. Final Mainline

最终主线已固定为：

- `DDD`: `HGNN + validation-selected grid rerank (DDD_top1)`
- `mimic_test`: existing `HGNN_SimilarCase_Aug`
- all other datasets: `HGNN exact baseline`

本轮未训练新模型，未修改 HGNN encoder，未覆盖原始 exact evaluation，未做 test-side 调参。

## 2. Mainline Files

新增最终主线配置与入口：

- `configs/mainline_hgnn_val_grid_rerank.yaml`
- `tools/run_mainline_hgnn_val_grid_rerank_all.py`

最终输出：

- `outputs/mainline_hgnn_val_grid_rerank_all/mainline_final_case_ranks.csv`
- `outputs/mainline_hgnn_val_grid_rerank_all/hgnn_baseline_metrics.csv`
- `outputs/mainline_hgnn_val_grid_rerank_all/mainline_final_metrics.csv`
- `outputs/mainline_hgnn_val_grid_rerank_all/mainline_delta_vs_baseline.csv`
- `outputs/mainline_hgnn_val_grid_rerank_all/mainline_run_manifest.json`
- `reports/mainline_hgnn_val_grid_rerank_all/mainline_final_all_dataset_report.md`

## 3. Deleted Branch Files

删除 manifest：

- `reports/cleanup/deleted_branch_files_manifest.json`

已删除：

- `configs/train_finetune_ddd_ontology_hn.yaml`
- `src/training/hard_negative_pools.py`
- `reports/ddd_improvement/run_ddd_hn_candidate_pool_dryrun.py`
- `reports/ddd_improvement/ddd_hn_candidate_pool_audit.md`
- `reports/ddd_improvement/ddd_hn_candidate_pool_stats.csv`
- `reports/ddd_improvement/ddd_hn_dryrun_samples.csv`
- `outputs/ddd_ontology_hn/`

## 4. Import / Reference Cleanup

已从 `src/training/trainer.py` 移除：

- `from src.training.hard_negative_pools import build_hard_negative_pool_builder`
- batch 内 `hard_negative_pool_builder.build_for_batch(...)`
- trainer main 中的 `build_hard_negative_pool_builder(...)`

已从 `src/runtime_config.py` 移除：

- `loss.hard_negative.candidate_pools` 配置透传。

保留 `src/training/hard_negative_miner.py`，因为它是通用 hard negative miner，不是 DDD HN candidate pool branch 文件。

## 5. Rerank Cleanup

`tools/run_top50_evidence_rerank.py` 已收敛为 grid-only / fixed-eval 工具：

- 保留 validation-selected grid selection。
- 保留 fixed weights evaluation。
- 保留 HGNN preset baseline metrics。
- 移除 gated search 入口。
- 移除 test-side exploratory report 入口。

`reports/ddd_improvement/run_ddd_validation_selected_rerank.py` 已收敛为 grid-only 复现脚本，不再生成 gated payload。

## 6. Final All-Dataset Metrics

| Dataset | Top1 | Top3 | Top5 | Median Rank | Recall@50 |
|---|---:|---:|---:|---:|---:|
| DDD | 0.3693 | 0.4875 | 0.5506 | 4.0 | 0.7451 |
| HMS | 0.2800 | 0.4400 | 0.4800 | 6.0 | 0.7200 |
| LIRICAL | 0.5254 | 0.5932 | 0.6780 | 1.0 | 0.7797 |
| MME | 0.9000 | 0.9000 | 0.9000 | 1.0 | 0.9000 |
| MyGene2 | 0.8485 | 0.8788 | 0.8788 | 1.0 | 0.9697 |
| RAMEDIS | 0.7742 | 0.8664 | 0.9309 | 1.0 | 0.9908 |
| mimic_test | 0.2093 | 0.3300 | 0.3940 | 18.0 | 0.6498 |
| ALL | 0.3076 | 0.4234 | 0.4866 | 6.0 | 0.7065 |

ALL delta vs historical HGNN baseline:

- top1: `+0.0282`
- top3: `+0.0302`
- top5: `+0.0390`
- recall@50: `+0.0218`

## 7. Validation

已运行：

```powershell
D:\python\python.exe tools\run_mainline_hgnn_val_grid_rerank_all.py --config-path configs\mainline_hgnn_val_grid_rerank.yaml
```

已运行语法检查：

```powershell
D:\python\python.exe -m py_compile tools\run_top50_evidence_rerank.py tools\run_mainline_hgnn_val_grid_rerank_all.py tools\run_mimic_similar_case_aug.py tools\run_similar_case_aug_all_datasets.py src\training\trainer.py src\runtime_config.py reports\ddd_improvement\run_ddd_validation_selected_rerank.py
```

结果：通过。

## 8. Remaining Notes

- 源码、工具、配置中不再引用 `HardNegativeCandidatePoolBuilder`、`build_hard_negative_pool_builder`、`train_finetune_ddd_ontology_hn.yaml` 或 `run_ddd_hn_candidate_pool_dryrun.py`。
- 历史报告中仍保留 HN/gated/test-side exploratory 文字引用，用于论文附录和审计追溯；主线入口不使用这些分支。
- mimic SimilarCase-Aug 的 case namespace 与 HGNN baseline 使用不同文件名，主线入口按 `case_N` 后缀对齐，这是为了复用已固定的 mimic 主线输出。
