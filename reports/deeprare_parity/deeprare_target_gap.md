# DeepRare Target Gap

> 本表使用当前 exact HGNN baseline / rank decomposition 与 DeepRare 论文目标对齐。`rank<=50` 只是 top50 rerank 的理论上限，不代表正式可报告结果。

| dataset | current_cases | deeprare_paper_cases | split_mismatch | current_top1 | current_top3 | current_top5 | deeprare_top1 | deeprare_top3 | deeprare_top5 | gap_top1 | gap_top3 | gap_top5 | rank_le_50_upper_bound | top50_rerank_can_reach_top5_target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DDD | 761 | 2283 | True | 0.3022 | 0.4442 | 0.4967 | 0.4800 | 0.6000 | 0.6300 | -0.1778 | -0.1558 | -0.1333 | 0.7451 | True |
| mimic_test | 1873 | 1873 | False | 0.1917 | 0.2995 | 0.3540 | 0.2900 | 0.3700 | 0.3900 | -0.0983 | -0.0705 | -0.0360 | 0.6151 | True |
| LIRICAL | 59 | 370 | True | 0.5254 | 0.5932 | 0.6780 | 0.5600 | 0.6500 | 0.6800 | -0.0346 | -0.0568 | -0.0020 | 0.7797 | True |
| RAMEDIS | 217 | 624 | True | 0.7742 | 0.8664 | 0.9309 | 0.7100 | 0.8300 | 0.8500 | 0.0642 | 0.0364 | 0.0809 | 0.9908 | True |
| HMS | 25 | 88 | True | 0.2800 | 0.4400 | 0.4800 | 0.5700 | 0.6500 | 0.7100 | -0.2900 | -0.2100 | -0.2300 | 0.7200 | True |
| MME | 10 | 40 | True | 0.9000 | 0.9000 | 0.9000 | 0.7800 | 0.8500 | 0.9000 | 0.1200 | 0.0500 | 0.0000 | 0.9000 | True |
| MyGene2 | 33 | 146 | True | 0.8485 | 0.8788 | 0.8788 | 0.7600 | 0.8000 | 0.8100 | 0.0885 | 0.0788 | 0.0688 | 0.9697 | True |

## Split Mismatch

- `DDD`: 当前 761 vs 论文 2283
- `LIRICAL`: 当前 59 vs 论文 370
- `RAMEDIS`: 当前 217 vs 论文 624
- `HMS`: 当前 25 vs 论文 88
- `MME`: 当前 10 vs 论文 40
- `MyGene2`: 当前 33 vs 论文 146
