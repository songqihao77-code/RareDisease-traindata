# similar_case fixed test

- 协议: validation-selected fixed weights，用 test 只评估一次。
- selected source_combination: HGNN + similar_case
- selected topk/weight/score_type: 10 / 0.4 / raw_similarity

| method | total_cases | top1 | top3 | top5 | rank_le_50 | rank_gt_50_cases | gold_in_top50_but_rank_gt5_cases | median_rank | mean_rank | any_label_at_1 | any_label_at_3 | any_label_at_5 | any_label_at_50 | selected_source_combination | selected_similar_case_topk | selected_similar_case_weight | selected_similar_case_score_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected_fixed_weights | 1873 | 0.20928990923651897 | 0.32995194874532835 | 0.39402028830752805 | 0.6497597437266418 | 656 | 479 | 18.0 | 3397.430859583556 | 0.22957821676454884 | 0.35451147891083823 | 0.4223171382808329 | 0.6871329418045916 | HGNN + similar_case | 10 | 0.4 | raw_similarity |