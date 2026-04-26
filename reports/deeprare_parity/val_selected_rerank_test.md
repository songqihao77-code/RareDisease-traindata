# Validation-Selected Rerank Test

- `HGNN_exact_baseline` 来自原始 exact evaluation，不覆盖。
- `validation_selected_fixed_test` 的权重来自 validation candidates，test candidates 只做一次 fixed eval；本轮只完成 linear grid，完整 gated/mimic-safe gate 因运行超时未完成。
- `test_side_exploratory_upper_bound:*` 来自 test-side grid，只能作为附表或 upper bound，不能进入论文主表。

| dataset | method | top1 | top3 | top5 | rank_le_50 |
| --- | --- | --- | --- | --- | --- |
| DDD | HGNN_exact_baseline | 0.3022 | 0.4442 | 0.4967 | 0.7451 |
| HMS | HGNN_exact_baseline | 0.2800 | 0.4400 | 0.4800 | 0.7200 |
| LIRICAL | HGNN_exact_baseline | 0.5254 | 0.5932 | 0.6780 | 0.7797 |
| MME | HGNN_exact_baseline | 0.9000 | 0.9000 | 0.9000 | 0.9000 |
| MyGene2 | HGNN_exact_baseline | 0.8485 | 0.8788 | 0.8788 | 0.9697 |
| RAMEDIS | HGNN_exact_baseline | 0.7742 | 0.8664 | 0.9309 | 0.9908 |
| mimic_test | HGNN_exact_baseline | 0.1917 | 0.2995 | 0.3540 | 0.6151 |
| ALL | HGNN_exact_baseline | 0.2794 | 0.3932 | 0.4476 | 0.6847 |
| DDD | validation_selected_fixed_test | 0.3430 | 0.4704 | 0.5138 | 0.7451 |
| mimic_test | validation_selected_fixed_test | 0.1869 | 0.3006 | 0.3529 | 0.6151 |
| HMS | validation_selected_fixed_test | 0.3200 | 0.4400 | 0.5600 | 0.7200 |
| LIRICAL | validation_selected_fixed_test | 0.5932 | 0.6780 | 0.7119 | 0.7797 |
| ALL | validation_selected_fixed_test | 0.2848 | 0.4026 | 0.4527 | 0.6847 |
| DDD | test_side_exploratory_upper_bound:grid_1720 | 0.3784 | 0.4888 | 0.5532 | 0.7451 |
| mimic_test | test_side_exploratory_upper_bound:grid_1720 | 0.1644 | 0.2755 | 0.3348 | 0.6151 |
| HMS | test_side_exploratory_upper_bound:grid_1720 | 0.3600 | 0.4800 | 0.5600 | 0.7200 |
| LIRICAL | test_side_exploratory_upper_bound:grid_1720 | 0.5763 | 0.6949 | 0.7288 | 0.7797 |
| ALL | test_side_exploratory_upper_bound:grid_1720 | 0.2639 | 0.3852 | 0.4480 | 0.6847 |
