# Mainline Dependency Audit

## 1. 工作区状态

- `git status --short` 显示当前工作区不是干净状态。
- 已存在的 tracked 修改：
  - `configs/data_llldataset_eval.yaml`
  - `src/runtime_config.py`
  - `src/training/hard_negative_miner.py`
  - `src/training/trainer.py`
- 本次清理没有修改上述 HGNN/training tracked 文件，也没有修改 HGNN encoder。
- `git diff --stat` 在清理前显示上述 4 个 tracked 文件共有 `195 insertions(+), 4 deletions(-)`；这些不是本次主线收敛的删除对象。

## 2. 主线保留范围

保留当前 verified mainline：

- 原始 HGNN exact baseline/evaluator 代码。
- `mimic_test_recleaned` 相关 fixed-test 结果。
- `tools/export_top50_candidates.py`：用于主线复现 HGNN top50 candidates。
- `tools/audit_mimic_cleaning.py`：用于 mimic 数据审计。
- `tools/run_mimic_similar_case_aug.py`：新的 SimilarCase-Aug 主线入口。
- `tools/finalize_similar_case_aug.py`：冻结配置、泄漏审计、validation stability、最终报告入口。
- `configs/mimic_similar_case_aug.yaml`：主线 frozen method 配置。
- `reports/mimic_next/frozen_similar_case_aug_config.json`
- `reports/mimic_next/final_similar_case_aug_report.md`
- `reports/mimic_next/similar_case_fixed_test.md`
- `reports/mimic_next/similar_case_fixed_test.csv`
- `reports/mimic_next/similar_case_fixed_test_ranked_candidates.csv`
- `reports/mimic_next/similar_case_leakage_audit.md`
- `reports/mimic_next/similar_case_leakage_audit.csv`
- `reports/mimic_next/similar_case_validation_stability.md`
- `reports/mimic_next/similar_case_validation_stability.csv`
- `reports/mimic_next/top5_gain_source_analysis.md`
- `reports/mimic_next/recovered_rank_gt50_cases.*`
- `reports/mimic_next/near_miss_top5_cases.*`
- `reports/mimic_next/mimic_rank_decomposition_recleaned.*`

## 3. 已删除的 mimic 支线入口

以下文件已删除，因为它们会继续暴露 broad augmentation、HPO noise、pairwise 或 test-side exploratory 入口：

- `tools/run_mimic_candidate_augmentation.py`
- `tools/analyze_mimic_similar_case_fast.py`
- `tools/postprocess_mimic_similar_selected.py`
- `tools/write_deeprare_reports.py`
- `reports/mimic_next/current_aug_result_summary.md`
- `reports/mimic_next/fast_module_validation_report.md`
- `reports/mimic_next/final_mimic_next_recommendation.md`
- `reports/mimic_next/hpo_filter_sanity_check.csv`
- `reports/mimic_next/mimic_aug_fixed_test.csv`
- `reports/mimic_next/mimic_aug_pairwise_ablation.csv`
- `reports/mimic_next/mimic_aug_pairwise_test.csv`
- `reports/mimic_next/mimic_aug_source_ablation.csv`
- `reports/mimic_next/mimic_aug_val_selection.csv`
- `reports/mimic_next/mimic_hpo_noise_ablation.csv`
- `reports/mimic_next/mimic_hpo_noise_fixed_test.csv`
- `reports/mimic_next/mimic_rank50_recovery.csv`
- `reports/mimic_next/similar_case_val_ablation.csv`

## 4. 依赖检查结果

清理后执行依赖搜索：

```text
rg -n "analyze_mimic_similar_case_fast|postprocess_mimic_similar_selected|run_mimic_candidate_augmentation|write_deeprare_reports|mimic_aug_|mimic_hpo_noise|similar_case_val_ablation|skip-hpo-noise|skip-pairwise|run-bm25|run-tfidf|run-semantic-ic|run-kb-overlap|run-synonym-expansion|test_side_exploratory|kb_component" -S tools configs reports\mimic_next
```

结果：无匹配，说明已删除的 mimic 支线脚本和临时开关不再被 active mimic 主线引用。

## 5. 保留但非 mimic 主线的通用脚本

以下文件仍包含 `pairwise` 或 `semantic_ic` 等关键词，但不是 mimic 专用支线入口，本次未删除：

- `tools/train_top50_pairwise_reranker.py`
- `tools/eval_top50_pairwise_reranker.py`
- `tools/run_top50_evidence_rerank.py`
- `tools/run_candidate_augmentation.py`
- `tools/run_diagnosis_audit.py`

标记：`kept_because_used_by_other_pipeline`。

原因：这些脚本命名和路径不是 mimic 专用；部分用于通用 top50 evidence rerank、diagnosis audit 或 DDD/DeepRare parity 相关工作。删除它们可能破坏非 mimic pipeline。当前 mimic 主线报告明确不建议训练 pairwise reranker，且 `configs/mimic_similar_case_aug.yaml` 中 `allow_pairwise_training: false`。

## 6. SimilarCase 主线脚本边界

`tools/run_mimic_similar_case_aug.py` 只暴露以下参数：

- `--similar-case-topk`
- `--similar-case-weight`
- `--score-type`
- `--use-frozen-config`
- `--frozen-config-path`
- `--validation-candidates-path`
- `--test-candidates-path`
- `--output-dir`

不暴露：

- BM25 disease retrieval
- TF-IDF disease retrieval
- `semantic_ic`
- `kb_overlap`
- `synonym_alt_replaced`
- HPO noise ablation
- pairwise reranker
- train candidate export
- test-side exploratory grid

说明：脚本内部使用 `TfidfVectorizer` 只用于 SimilarCase 的 HPO-set 相似病例检索，不是 broad TF-IDF disease retrieval source。

## 7. 验证

已运行：

```powershell
D:\python\python.exe -m py_compile tools\run_mimic_similar_case_aug.py tools\finalize_similar_case_aug.py tools\audit_mimic_cleaning.py tools\export_top50_candidates.py
D:\python\python.exe -c "import json; from pathlib import Path; p=Path(r'reports\mimic_next\frozen_similar_case_aug_config.json'); d=json.load(open(p,encoding='utf-8')); print(d['method_name'], d['similar_case_topk'], d['similar_case_weight'], d['score_type'], d['fixed_test_metrics']['top5'])"
```

验证结果：

- 主线脚本均通过 `py_compile`。
- frozen config 可正常读取。
- frozen config 内容为 `HGNN_SimilarCase_Aug`, `topk=10`, `weight=0.4`, `score_type=raw_similarity`, `fixed_test_top5=0.39402028830752805`。

## 8. 剩余风险

- 工作区仍有本次未处理的 tracked 修改和其它未跟踪报告/脚本；这些不属于 mimic SimilarCase 主线清理范围。
- patient/admission/note-level leakage 仍无法确认，需要原始 `subject_id/hadm_id/note_id` 映射。
- 通用 pairwise/rerank 脚本仍存在；它们不是 mimic 主线入口，但后续若要进一步收敛整个项目的所有非主线实验，需要单独确认 DDD/其它 pipeline 是否还依赖它们。
