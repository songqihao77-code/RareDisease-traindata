# Parity audit: old baseline vs current final

- old baseline summary: `D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf\evaluation\best_20260423_202917_summary.json`
- old details: `D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf\evaluation\best_20260423_202917_details.csv`
- current exact summary: `D:\RareDisease-traindata\outputs\attn_beta_sweep\edge_log_beta02\evaluation\best_20260427_110758_summary.json`
- current details: `D:\RareDisease-traindata\outputs\attn_beta_sweep\edge_log_beta02\evaluation\best_20260427_110758_details.csv`
- old checkpoint: `D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf\checkpoints\best.pt`
- current checkpoint: `D:\RareDisease-traindata\outputs\attn_beta_sweep\edge_log_beta02\checkpoints\best.pt`
- data config same: `True`

## Focus datasets
| dataset_name | old_top1 | current_final_top1 | delta_final_vs_old_top1 | old_top5 | current_final_top5 | delta_final_vs_old_top5 | case_id_set_equal | label_mismatch_count | checkpoint_drift | module_applied_current_final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LIRICAL | 0.5423728813559322 | 0.4576271186440678 | -0.0847457627118644 | 0.6610169491525424 | 0.5254237288135594 | -0.13559322033898302 | True | 0 | True | hgnn_baseline |
| MME | 0.9 | 0.8 | -0.09999999999999998 | 0.9 | 0.8 | -0.09999999999999998 | True | 0 | True | hgnn_baseline |
| RAMEDIS | 0.7880184331797235 | 0.7695852534562212 | -0.018433179723502335 | 0.9262672811059908 | 0.8847926267281107 | -0.041474654377880116 | True | 0 | True | hgnn_baseline |

## Conclusion
- LIRICAL/RAMEDIS/MME 当前最终汇总没有应用 mimic SimilarCase，也没有应用 DDD rerank。
- 三者下降主要来自 checkpoint/config drift：旧 baseline 是 `g4b_weighting_idf`，当前重跑主线是 `train_finetune_attn_idf_main` / `attn_beta_sweep\edge_log_beta02`。
- case_id 与 label 在重叠集合内一致；HPO 行历史 hash 未记录，只能确认 summary 中解析到的 test file path 是否相同。