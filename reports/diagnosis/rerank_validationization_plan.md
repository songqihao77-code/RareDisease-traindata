# Rerank Validationization Plan

## 当前状态
- `tools/export_top50_candidates.py` 支持 `--case-source validation`，可导出 validation top50 candidates。
- `tools/run_top50_evidence_rerank.py` 支持 `validation_select`、`fixed_eval`、`presets` 三种协议。
- validation grid search、best config JSON 保存、fixed test evaluation 已具备。
- 旧的 `outputs/rerank/rerank_v2_grid_results.csv` / `rerank_v2_gated_results.csv` 属于 test-side exploratory，不应作为正式 test 结果。

## 缺口
- 当前正式脚本只冻结 linear grid；gated/mimic-safe gate 如果要进主表，需要同样从 validation 选择阈值并固定 test 跑一次。
- best config 以前主要保存为 JSON，本次补充生成 `configs/rerank/best_val_selected_rerank.yaml`。

## 推荐表述
- Main table: `HGNN exact baseline` 与 `validation-selected fixed-test rerank`。
- Supplementary: test-side exploratory upper bound、gated ablation、any-label 或 relaxed diagnostics。
