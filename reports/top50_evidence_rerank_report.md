# top50 evidence rerank report

- generated_at: 2026-04-26T14:58:57
- candidates_path: `D:\RareDisease-traindata\outputs\rerank\top50_candidates.csv`
- audit_data_config_path: `D:\RareDisease-traindata\configs\data_llldataset_eval.yaml`
- rank policy: gold absent from top50 is counted as rank 51; `rank_le_50` is therefore top50 candidate coverage.

## 权重预设

|preset|w_hgnn|w_ic|w_exact|w_case_cov|w_dis_cov|w_size|
|---|---|---|---|---|---|---|
|A_hgnn_only|1.0000|0.0000|0.0000|0.0000|0.0000|0.0000|
|B_hgnn_ic|0.7500|0.2500|0.0000|0.0000|0.0000|0.0000|
|C_hgnn_exact|0.7500|0.0000|0.2500|0.0000|0.0000|0.0000|
|D_hgnn_ic_coverage|0.6500|0.2500|0.0000|0.0500|0.0500|0.0000|
|E_hgnn_ic_exact_coverage|0.6000|0.2000|0.1000|0.0500|0.0500|0.0000|
|F_hgnn_ic_exact_coverage_size|0.6000|0.2000|0.1000|0.0500|0.0500|0.0200|

## 指标

|preset|dataset_name|num_cases|top1|top3|top5|median_rank|mean_rank|rank_le_50|
|---|---|---|---|---|---|---|---|---|
|A_hgnn_only|DDD|761|0.3022|0.4442|0.4967|6.0000|18.8436|0.7451|
|A_hgnn_only|mimic_test|1873|0.1917|0.2995|0.3540|20.0000|25.9541|0.6151|
|A_hgnn_only|HMS|25|0.2800|0.4400|0.4800|6.0000|18.7200|0.7200|
|A_hgnn_only|LIRICAL|59|0.5254|0.5932|0.6780|1.0000|14.4576|0.7797|
|A_hgnn_only|ALL|2978|0.2794|0.3932|0.4476|9.0000|21.8526|0.6847|
|B_hgnn_ic|DDD|761|0.3456|0.4625|0.5138|5.0000|18.1117|0.7451|
|B_hgnn_ic|mimic_test|1873|0.1869|0.3006|0.3502|20.0000|25.9477|0.6151|
|B_hgnn_ic|HMS|25|0.3200|0.4400|0.5200|4.0000|17.9600|0.7200|
|B_hgnn_ic|LIRICAL|59|0.5763|0.6441|0.7119|1.0000|13.8814|0.7797|
|B_hgnn_ic|ALL|2978|0.2874|0.4003|0.4506|9.0000|21.6441|0.6847|
|C_hgnn_exact|DDD|761|0.3351|0.4560|0.5085|5.0000|18.3219|0.7451|
|C_hgnn_exact|mimic_test|1873|0.1922|0.3022|0.3508|21.0000|25.9877|0.6151|
|C_hgnn_exact|HMS|25|0.3200|0.4400|0.5200|4.0000|18.2800|0.7200|
|C_hgnn_exact|LIRICAL|59|0.5424|0.6102|0.6949|1.0000|14.2712|0.7797|
|C_hgnn_exact|ALL|2978|0.2884|0.3989|0.4490|9.0000|21.7330|0.6847|
|D_hgnn_ic_coverage|DDD|761|0.3495|0.4757|0.5191|5.0000|17.8620|0.7451|
|D_hgnn_ic_coverage|mimic_test|1873|0.1863|0.3022|0.3492|21.0000|25.9669|0.6151|
|D_hgnn_ic_coverage|HMS|25|0.3200|0.4400|0.5200|4.0000|17.9200|0.7200|
|D_hgnn_ic_coverage|LIRICAL|59|0.5763|0.6949|0.7119|1.0000|13.7966|0.7797|
|D_hgnn_ic_coverage|ALL|2978|0.2864|0.4050|0.4506|8.0000|21.5950|0.6847|
|E_hgnn_ic_exact_coverage|DDD|761|0.3548|0.4770|0.5164|5.0000|17.8160|0.7451|
|E_hgnn_ic_exact_coverage|mimic_test|1873|0.1879|0.3006|0.3497|21.0000|26.0091|0.6151|
|E_hgnn_ic_exact_coverage|HMS|25|0.3200|0.4400|0.5200|4.0000|17.9200|0.7200|
|E_hgnn_ic_exact_coverage|LIRICAL|59|0.5763|0.7119|0.7119|1.0000|13.7797|0.7797|
|E_hgnn_ic_exact_coverage|ALL|2978|0.2881|0.4050|0.4503|9.0000|21.6101|0.6847|
|F_hgnn_ic_exact_coverage_size|DDD|761|0.3535|0.4717|0.5138|5.0000|17.9054|0.7451|
|F_hgnn_ic_exact_coverage_size|mimic_test|1873|0.1901|0.3033|0.3481|20.0000|26.0048|0.6151|
|F_hgnn_ic_exact_coverage_size|HMS|25|0.3200|0.4400|0.5200|5.0000|17.9600|0.7200|
|F_hgnn_ic_exact_coverage_size|LIRICAL|59|0.5593|0.6610|0.7119|1.0000|13.9492|0.7797|
|F_hgnn_ic_exact_coverage_size|ALL|2978|0.2871|0.4033|0.4496|9.0000|21.6343|0.6847|

## mimic_test 多标签审计

|preset|mimic_cases|multi_label_cases|multi_label_case_ratio|exact_top1|exact_top3|exact_top5|multi_label_any_hit@1|multi_label_any_hit@3|multi_label_any_hit@5|potential_top1_delta|potential_top3_delta|potential_top5_delta|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|A_hgnn_only|1873|227|0.1212|0.1917|0.2995|0.3540|0.2088|0.3219|0.3785|0.0171|0.0224|0.0246|
|B_hgnn_ic|1873|227|0.1212|0.1869|0.3006|0.3502|0.2040|0.3235|0.3753|0.0171|0.0230|0.0251|
|C_hgnn_exact|1873|227|0.1212|0.1922|0.3022|0.3508|0.2088|0.3251|0.3753|0.0166|0.0230|0.0246|
|D_hgnn_ic_coverage|1873|227|0.1212|0.1863|0.3022|0.3492|0.2034|0.3246|0.3743|0.0171|0.0224|0.0251|
|E_hgnn_ic_exact_coverage|1873|227|0.1212|0.1879|0.3006|0.3497|0.2050|0.3230|0.3737|0.0171|0.0224|0.0240|
|F_hgnn_ic_exact_coverage_size|1873|227|0.1212|0.1901|0.3033|0.3481|0.2066|0.3262|0.3732|0.0166|0.0230|0.0251|

## 结论问题

- 哪组权重 DDD top5 最高？`D_hgnn_ic_coverage`，DDD top5=0.5191。
- 哪组权重 DDD top1 最高？`E_hgnn_ic_exact_coverage`，DDD top1=0.3548。
- 是否存在 top5 上升但 top1 下降？不存在。
- mimic_test 是否因多标签压缩产生假错误？存在潜在假错误：最佳 DDD top5 预设 `D_hgnn_ic_coverage` 的 any-hit@5 比 exact top5 高 0.0251。
- 这个 reranker 是否值得接入正式 evaluation？值得：DDD top5 有提升，且 DDD top1 未低于 HGNN-only top50 基线。下一步才考虑训练 hard negative。
