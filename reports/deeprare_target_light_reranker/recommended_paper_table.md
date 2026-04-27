# Recommended Paper Table

1. current mainline 仍应作为主表 baseline：是。
2. light reranker 是否可作为主表新方法：暂不建议作为主表新方法；本次更适合作为附表/负结果分析。
3. 是否达到 DeepRare target：达到的数据集为 MME, MyGene2, RAMEDIS。
4. 达到 target 的 dataset：MME, MyGene2, RAMEDIS。
5. 仍未达到 target 的 dataset：DDD, HMS, LIRICAL, mimic_test_recleaned_mondo_hpo_rows。
6. 是否牺牲已达标数据集：否。
7. gated rerank 建议保留为附表，不与 strict exact 主结果混写。
8. any-label / relaxed MONDO 只能 supplementary，不能替代 strict exact。
9. 图对比学习仍后置。
10. 如果 DDD 仍未达标：是，下一步进入 ontology-aware hard negative training。
