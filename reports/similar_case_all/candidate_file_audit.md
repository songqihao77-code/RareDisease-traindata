# SimilarCase All-Dataset Candidate File Audit

- test candidates rows/cases: 148900 / 2978
- test candidates datasets: DDD, HMS, LIRICAL, MME, MyGene2, RAMEDIS, mimic_test_recleaned_mondo_hpo_rows
- baseline per-dataset datasets: DDD, HMS, LIRICAL, MME, MyGene2, RAMEDIS, mimic_test_recleaned_mondo_hpo_rows
- validation candidates rows/cases: 107300 / 2146
- validation candidates datasets: DDD, FakeDisease, HMS, LIRICAL, MME, MyGene2, RAMEDIS, mimic_rag_0425
- test candidates 是否覆盖 baseline 全部数据集: 是
- 每条 candidate 包含 `dataset_name` 和 `case_id` 字段，可直接按 dataset 聚合；`case_id` 也包含 split namespace。
- 当前候选文件已包含全部 test dataset，因此本轮未重新导出 all-test top50 candidates。

| file | dataset_name | rows | cases |
| --- | --- | --- | --- |
| test_candidates | DDD | 38050 | 761 |
| test_candidates | HMS | 1250 | 25 |
| test_candidates | LIRICAL | 2950 | 59 |
| test_candidates | MME | 500 | 10 |
| test_candidates | MyGene2 | 1650 | 33 |
| test_candidates | RAMEDIS | 10850 | 217 |
| test_candidates | mimic_test_recleaned_mondo_hpo_rows | 93650 | 1873 |
| validation_candidates | DDD | 8200 | 164 |
| validation_candidates | FakeDisease | 58150 | 1163 |
| validation_candidates | HMS | 150 | 3 |
| validation_candidates | LIRICAL | 1650 | 33 |
| validation_candidates | MME | 150 | 3 |
| validation_candidates | MyGene2 | 550 | 11 |
| validation_candidates | RAMEDIS | 2000 | 40 |
| validation_candidates | mimic_rag_0425 | 36450 | 729 |
| baseline_details | DDD | 761 | 761 |
| baseline_details | HMS | 25 | 25 |
| baseline_details | LIRICAL | 59 | 59 |
| baseline_details | MME | 10 | 10 |
| baseline_details | MyGene2 | 33 | 33 |
| baseline_details | RAMEDIS | 217 | 217 |
| baseline_details | mimic_test_recleaned_mondo_hpo_rows | 1873 | 1873 |
| baseline_per_dataset | DDD | 1 |  |
| baseline_per_dataset | HMS | 1 |  |
| baseline_per_dataset | LIRICAL | 1 |  |
| baseline_per_dataset | MME | 1 |  |
| baseline_per_dataset | MyGene2 | 1 |  |
| baseline_per_dataset | RAMEDIS | 1 |  |
| baseline_per_dataset | mimic_test_recleaned_mondo_hpo_rows | 1 |  |