# SimilarCase Mainline After Hyperedge Cleanup

## 1. Decision

不采用 `DiseaseHyperedge-SimAug`。当前保留主线为：

- HGNN exact baseline
- `mimic_test_recleaned`
- validation-selected fixed-test `SimilarCase-Aug`
- all-dataset `SimilarCase-Aug` fixed evaluation

## 2. Removed Branch

已删除以下受控负结果分支产物：

- `reports/hyperedge_sim_aug/`
- `outputs/hyperedge_sim_aug/`

这些目录只包含 `DiseaseHyperedge-SimAug` 的向量、邻居、validation selection、fixed test ranked candidates 和报告，不属于当前主线。

## 3. Kept Mainline

仍保留：

- `tools/run_mimic_similar_case_aug.py`
- `tools/finalize_similar_case_aug.py`
- `tools/run_similar_case_aug_all_datasets.py`
- `tools/audit_mimic_cleaning.py`
- `tools/export_top50_candidates.py`
- `configs/mimic_similar_case_aug.yaml`
- `reports/mimic_next/`
- `reports/similar_case_all/`
- `outputs/similar_case_all/`

## 4. Verification

已验证：

```powershell
Test-Path reports\hyperedge_sim_aug
Test-Path outputs\hyperedge_sim_aug
Test-Path reports\similar_case_all
Test-Path outputs\similar_case_all
Test-Path reports\mimic_next\frozen_similar_case_aug_config.json
D:\python\python.exe -m py_compile tools\run_mimic_similar_case_aug.py tools\finalize_similar_case_aug.py tools\run_similar_case_aug_all_datasets.py tools\audit_mimic_cleaning.py tools\export_top50_candidates.py
rg -n "hyperedge_sim_aug|DiseaseHyperedge-SimAug|run_hyperedge_sim_aug|build_disease_hyperedge_similarity" -S tools configs reports --glob '!reports/cleanup/**'
```

结果：

- `reports/hyperedge_sim_aug`: removed
- `outputs/hyperedge_sim_aug`: removed
- `reports/similar_case_all`: kept
- `outputs/similar_case_all`: kept
- frozen SimilarCase config: kept
- SimilarCase 主线脚本编译通过
- active `tools/configs/reports` 中无 `DiseaseHyperedge-SimAug` 入口残留

## 5. Not Removed

没有删除：

- 原始数据集
- `LLLdataset/DiseaseHy/` 原始 disease-HPO hyperedge 资源
- HGNN encoder
- 原始 exact baseline evaluator
- 已冻结 SimilarCase-Aug 结果

通用 `tools/audit_dataset_hyperedge_similarity.py` 未删除，因为它不是本次新增的 `DiseaseHyperedge-SimAug` 支线入口。
