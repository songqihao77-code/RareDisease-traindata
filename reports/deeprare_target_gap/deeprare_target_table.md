# DeepRare Target 对照表

mapping_status 采用保守口径。unavailable 数据集不强行映射到无关项目数据集。

| deeprare_dataset_name | target_recall1 | target_recall3 | target_recall5 | project_dataset_candidate           | mapping_status    | notes                                                                                      |
| --------------------- | -------------- | -------------- | -------------- | ----------------------------------- | ----------------- | ------------------------------------------------------------------------------------------ |
| RareBench-MME         | 0.7800         | 0.8500         | 0.9000         | MME                                 | exact_match       | 当前测试配置中存在 MME.xlsx，按 RareBench-MME 的对应候选处理；样本量很小，解释时需标注 high_variance。                     |
| RareBench-HMS         | 0.5700         | 0.6500         | 0.7100         | HMS                                 | exact_match       | 当前项目测试集中名称为 HMS，可与 RareBench-HMS 对齐；样本量很小，解释时需标注 high_variance。                            |
| RareBench-LIRICAL     | 0.5600         | 0.6500         | 0.6800         | LIRICAL                             | exact_match       | 当前项目测试集中名称为 LIRICAL，可与 RareBench-LIRICAL 对齐。                                               |
| RareBench-RAMEDIS     | 0.7300         | 0.8300         | 0.8500         | RAMEDIS                             | exact_match       | 当前项目测试集中名称为 RAMEDIS，可与 RareBench-RAMEDIS 对齐。                                               |
| MIMIC-IV-Rare         | 0.2900         | 0.3700         | 0.3900         | mimic_test_recleaned_mondo_hpo_rows | exact_match       | 当前 mainline 的 mimic test 数据集；只使用 strict exact Recall，不使用 supplementary any-label 指标。       |
| MyGene                | 0.7600         | 0.8000         | 0.8100         | MyGene2                             | approximate_match | 项目数据集名称为 MyGene2，作为当前可用的 MyGene 近似对应数据集，标记为 approximate_match。                             |
| DDD                   | 0.4800         | 0.6000         | 0.6300         | DDD                                 | exact_match       | 当前项目测试集中名称为 DDD，可直接对齐。                                                                     |
| Xinhua Hosp.          | 0.5800         | 0.7100         | 0.7400         |                                     | unavailable       | configs/data_llldataset_eval.yaml 和当前 final mainline metrics 中没有 Xinhua Hosp. 对应数据集，不做硬映射。 |
