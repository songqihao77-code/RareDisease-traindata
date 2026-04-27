# similar_case fixed test

- 协议: validation-selected fixed weights，用 test 只评估一次。
- selected source_combination: HGNN + similar_case
- selected topk/weight/score_type: 10 / 0.5 / raw_similarity

| method | total_cases | top1 | top3 | top5 | rank_le_50 | rank_gt_50_cases | gold_in_top50_but_rank_gt5_cases | median_rank | mean_rank | any_label_at_1 | any_label_at_3 | any_label_at_5 | any_label_at_50 | selected_source_combination | selected_similar_case_topk | selected_similar_case_weight | selected_similar_case_score_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected_fixed_weights | 1873 | 0.18526428190069408 | 0.29898558462359853 | 0.3475707421249333 | 0.6219967965830219 | 708 | 514 | 28.0 | 3648.940736785905 | 0.20288307528029897 | 0.3224773091297384 | 0.37159636946075814 | 0.6567004805125467 | HGNN + similar_case | 10 | 0.5 | raw_similarity |