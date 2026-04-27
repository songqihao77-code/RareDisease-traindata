# Recommended Accuracy Improvement Plan

## 当前实验框架审计

| 阶段 | 入口脚本 | 主要输入 | 主要输出 | 说明 |
| --- | --- | --- | --- | --- |
| Pretrain | `python -m src.training.trainer --config configs\train_pretrain.yaml` | `configs/train_pretrain.yaml` 中的 train_files、Disease_index、HPO_index、v59DiseaseHy | `outputs/stage1_pretrain_v59` | 使用 validation split 监控 `val_top1` 保存 best/last |
| Finetune | `python -m src.training.trainer --config configs\train_finetune_attn_idf_main.yaml` | processed train、pretrain checkpoint、v59DiseaseHy | `outputs/attn_beta_sweep/edge_log_beta02` | 当前 mainline，监控 `val_real_macro_top5`，hard negative 为 current top-k |
| Exact eval | `python -m src.evaluation.evaluator --data_config_path configs\data_llldataset_eval.yaml --train_config_path configs\train_finetune_attn_idf_main.yaml` | data eval config、finetune config、checkpoint | `outputs/attn_beta_sweep/edge_log_beta02/evaluation/*` | exact rank 明细和 per-dataset metrics；未与 relaxed 混合 |
| Candidate export | `tools/export_top50_candidates.py` | checkpoint、test/validation case source | `outputs/rerank/top50_candidates*.csv` | 支持 test / train / validation candidate export |
| Rerank | `tools/run_top50_evidence_rerank.py` | top50 candidates、validation candidates、weights JSON/YAML | `outputs/rerank/*`, `reports/rerank/*` | 支持 validation grid selection 与 fixed test eval |
| Similar-case aug | `tools/run_mimic_similar_case_aug.py` | mimic top50 candidates、validation candidates | `reports/mimic_next/*` | 只能作为固定 protocol 或 exploratory，不能 test 调参 |

### 关键风险
- 旧 rerank v2/grid/gate 文件存在 test-side exploratory search，应标记为 upper bound/附表。
- 当前 validation candidate 已存在并可用于选权重；正式结果应只加载固定权重 test 一次。
- exact evaluation 由 `src.evaluation.evaluator` 的 `true_rank` 产生，relaxed/any-label/synonym/parent-child 只应作为 error analysis。
- Disease_index、HPO_index、v59DiseaseHy 维度一致性见 `mapping_issue_summary.md`；不同 dataset 均通过同一 v4/v59 资源评估。

## 当前准确率低的主要原因分解
- `DDD`: Recall@50 约 0.745，说明大量 gold 已在 top50 内但排序不足；适合 validation-selected rerank 和 hard negative training。
- `mimic_test`: Top1 低且 Recall@50 约 0.615，rank>50 比例高；主要是 candidate recall、multi-label exact 低估、HPO/KB overlap 不足。
- `ALL`: 被 DDD 和 mimic_test 两个大数据集主导，Top1 提升优先看这两个 dataset。
- `HMS/LIRICAL`: case 数较小且与论文 split 可能不一致，应先确认协议 parity。

## Dataset 瓶颈
| dataset | num_cases | top1 | top3 | top5 | top50 | rank_gt_50_ratio | top50_rank_gt5_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DDD | 761 | 0.3022 | 0.4442 | 0.4967 | 0.7451 | 0.2549 | 0.2484 |
| HMS | 25 | 0.2800 | 0.4400 | 0.4800 | 0.7200 | 0.2800 | 0.2400 |
| LIRICAL | 59 | 0.5254 | 0.5932 | 0.6780 | 0.7797 | 0.2203 | 0.1017 |
| MME | 10 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.1000 | 0.0000 |
| MyGene2 | 33 | 0.8485 | 0.8788 | 0.8788 | 0.9697 | 0.0303 | 0.0909 |
| RAMEDIS | 217 | 0.7742 | 0.8664 | 0.9309 | 0.9908 | 0.0092 | 0.0599 |
| mimic_test | 1873 | 0.1917 | 0.2995 | 0.3540 | 0.6151 | 0.3849 | 0.2611 |
| ALL | 2978 | 0.2794 | 0.3932 | 0.4476 | 0.6847 | 0.3153 | 0.2371 |

## 不需要训练即可提升或解释的问题
- 固定 validation-selected rerank 可改善 DDD/LIRICAL/HMS 的 top-k，但必须与 test-side exploratory 区分。
- mimic_test any-label、multi-label、synonym/parent-child 命中只能解释 exact 低估，不能覆盖 exact 主指标。
- MONDO/HPO obsolete、replacement、alt_id、xref one-to-many 应做 mapping audit 和数据清洗。

## 适合 no-train reranker 的问题
- gold 已在 top50 但 rank>5 的 DDD/LIRICAL/HMS 样本。
- HPO exact/IC/semantic overlap 强但 HGNN score 未排前的 near miss。
- 不适合解决 gold 不在 candidate pool 的 mimic rank>50 样本。

## 适合 hard negative training 的问题
- DDD same-parent/shared-ancestor/top50-above-gold confusion。
- HPO 高重叠 hard negative 造成的 top1 错误。
- mixed HN 可作为主实验，但需要先补 trainer candidate_pools。

## 需要数据清洗或 label mapping 修复的问题
- mimic_test multi-label exact gold 选择规则需明确；any-label 只做 supplementary。
- obsolete MONDO/HPO、replacement、alt_id、OMIM/ORPHA/ICD one-to-many 映射需单独修复并冻结版本。
- HPO parent-child/synonym 可解释命中不能混入 exact evaluation。

## 可进入论文主表
- `HGNN exact baseline`。
- `validation-selected fixed-test rerank`。
- 完成 candidate_pools 后的 `HN-current/HN-overlap/HN-sibling/HN-top50-above-gold/HN-mixed` 独立训练结果。

## 只能作为附表或 error analysis
- test-side grid/gate/weight search upper bound。
- mimic any-label hit@k、relaxed synonym/parent-child 命中。
- mapping audit、near-miss case list、hard negative candidate 类型统计。

## 推荐下一步实验顺序与预期
1. 固定 rerank protocol：预期 DDD Top1 +0.03~0.07，ALL Top1 +0.005~0.02；mimic Top1 不一定提升。
2. 修复并冻结 mapping/obsolete/alt_id：预期主要提升可解释性和 exact 可信度，Recall@50 小幅改善。
3. mimic candidate recovery / similar-case aug：预期 mimic Recall@50 和 Top5 改善，需 validation 选 source weights。
4. 实现 trainer candidate_pools 后跑 HN-current/overlap/sibling/top50-above-gold/mixed：预期 DDD Top1/Top3 提升，需监控 false negative。
5. HN-mixed + val-selected rerank：作为最终主表候选，预期 DDD Top1 和 ALL Top1 叠加提升。
