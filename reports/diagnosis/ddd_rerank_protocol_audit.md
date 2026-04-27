# DDD Rerank Protocol Audit

## 关键结论
- `tools/run_top50_evidence_rerank.py` 已支持 `exploratory`、`validation_select`、`fixed_eval` 三种协议。
- `reports/top50_evidence_rerank_v2_report.md` 明确标记 test-side grid/gate 只能作为 exploratory upper bound，不能作为正式 test 结论。
- 当前已有 validation candidates: `outputs/rerank/top50_candidates_validation.csv`，也已有 `outputs/rerank/val_selected_weights.json` 与 `outputs/rerank/rerank_fixed_test_metrics.csv`。
- 但当前固定 test 结果来自 validation grid 选权重，不是完整 gated/mimic-safe gate；`outputs/rerank/rerank_validation_gated_results.csv` 是空文件，说明 gated validation selection 没有完成。

## Evidence Features
| Feature | Calculation | Normalization | Notes |
|---|---|---|---|
| `hgnn_score` | HGNN candidate score | case 内 min-max | 主模型分数 |
| `ic_weighted_overlap` | shared HPO IC / case HPO IC | case 内 min-max | 最有解释性的 no-train evidence |
| `exact_overlap` | shared_hpo_count / sqrt(case_hpo_count*disease_hpo_count) | case 内 min-max | 抑制疾病 HPO 数量差异 |
| `semantic_ic_overlap` | exact 或 HPO ancestor/descendant match 的 case IC coverage | case 内 min-max | 使用 `src/rerank/hpo_semantic.py` |
| `case_coverage` | shared / case_hpo_count | case 内 min-max | query 覆盖 |
| `disease_coverage` | shared / disease_hpo_count | case 内 min-max | disease 侧覆盖 |
| `size_penalty` | log1p(disease_hpo_count) | case 内 min-max 后负权重 | 惩罚过宽疾病 |

## Fusion Formula
`score = w_hgnn*hgnn + w_ic*ic + w_exact*exact + w_semantic*semantic + w_case_cov*case_cov + w_dis_cov*disease_cov - w_size*size_penalty`。排序只发生在 HGNN top50 candidates 内。

## Metrics Evidence
- HGNN baseline DDD: top1/top3/top5=0.3022/0.4442/0.4967, recall@50=0.7451。
- validation-selected fixed test DDD: top1/top3/top5=0.3430/0.4704/0.5138, recall@50=0.7451。
- selected weights: `{'w_hgnn': 0.75, 'w_ic': 0.05, 'w_exact': 0.0, 'w_semantic': 0.1, 'w_case_cov': 0.0, 'w_dis_cov': 0.03, 'w_size': 0.01}`；objective=`ALL_top1`；kind=`grid`。

## Protocol Classification
- exploratory test-side rerank: `outputs/rerank/rerank_v2_grid_results.csv`、`outputs/rerank/rerank_v2_gated_results.csv`、`reports/top50_evidence_rerank_v2_report.md`。只能作为诊断/附表。
- validation-selected rerank: `outputs/rerank/top50_candidates_validation.csv` 选权重，`outputs/rerank/rerank_fixed_test_metrics.csv` 固定 test 评估。可作为论文候选，但需要在文中说明 selection objective。
- test-set weight search: 不能进入论文主表；只能标注为 exploratory upper bound。

## 最小可行后续路径
1. 保持 `tools/export_top50_candidates.py --case-source validation` 生成 validation top50。
2. 在 validation 上完成 grid 或 gated selection 并保存 `selected_weights.json`。
3. 用 `--protocol fixed_eval` 对 test candidates 只评估一次。
4. 不覆盖 `outputs/rerank/top50_candidates_v2.csv` 或既有 exact evaluation。