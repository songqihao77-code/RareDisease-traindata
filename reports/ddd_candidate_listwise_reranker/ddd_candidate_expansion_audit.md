# DDD Candidate Expansion Audit

| split      | candidate_limit | num_cases | current_top50_recall | expanded_recall | avg_candidates_per_case | gold_recovered_beyond_top50_cases | hgnn_source_rows | mondo_expansion_rows | hpo_expansion_rows | similar_case_rows |
| ---------- | --------------- | --------- | -------------------- | --------------- | ----------------------- | --------------------------------- | ---------------- | -------------------- | ------------------ | ----------------- |
| train      | 50              | 1358      | 0.9698               | 0.9698          | 215.1097                | 18                                | 67900            | 28059                | 241937             | 18329             |
| train      | 100             | 1358      | 0.9698               | 0.9772          | 215.1097                | 18                                | 67900            | 28059                | 241937             | 18329             |
| train      | 200             | 1358      | 0.9698               | 0.9831          | 215.1097                | 18                                | 67900            | 28059                | 241937             | 18329             |
| validation | 50              | 164       | 0.7134               | 0.7134          | 216.3415                | 20                                | 8200             | 2698                 | 30011              | 1953              |
| validation | 100             | 164       | 0.7134               | 0.7805          | 216.3415                | 20                                | 8200             | 2698                 | 30011              | 1953              |
| validation | 200             | 164       | 0.7134               | 0.8232          | 216.3415                | 20                                | 8200             | 2698                 | 30011              | 1953              |
| test       | 50              | 761       | 0.7438               | 0.7438          | 214.8739                | 88                                | 38050            | 16106                | 135637             | 10502             |
| test       | 100             | 761       | 0.7438               | 0.8187          | 214.8739                | 88                                | 38050            | 16106                | 135637             | 10502             |
| test       | 200             | 761       | 0.7438               | 0.8581          | 214.8739                | 88                                | 38050            | 16106                | 135637             | 10502             |

## 判断
- expansion 保留 current top50，不丢弃已有候选。
- 是否值得进入 listwise reranker：只有当 validation expanded recall 高于 current top50 recall，且新增候选没有显著稀释 gold 排序时才值得。
- test expansion 只用于 fixed evaluation，不参与 topK/source/weight 选择。
