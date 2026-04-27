# DDD Accuracy Improvement Audit Report

> 重要风险先行: `configs/train_finetune_ontology_hn.yaml` 虽配置了 `HN-mixed`，但 `src/training/trainer.py::run_one_epoch` 当前没有向 `mine_configurable_hard_negatives` 传入 `candidate_pools`，因此 ontology-aware hard negative 会退化为 score-based `HN-current`。另一个边界是 `reports/top50_evidence_rerank_v2_report.md` 中的 grid/gate 是 test-side exploratory，不能作为正式 test 调参结论。

## 1. Executive Summary
DDD 当前 exact baseline 为 top1=0.3022, top3=0.4442, top5=0.4967, recall@50=0.7451。核心问题不是 gold 大面积缺失于候选全集，而是 top50 内排序不足: 567/761 个 gold 已进入 top50，但只有 230/761 排到 top1。无需训练即可做的优先路径包括 DDD label/MONDO normalization audit、disease-HPO coverage audit、validation-selected evidence rerank 和 near-miss slicing。reranker 适合处理 gold 已在 top50 但 rank>1 的 337 个样本，尤其是 top5 非 top1 和 top50 非 top5。hard negative training 适合处理 same_parent/shared_ancestor/高 HPO-overlap 混淆，但必须先把 ontology candidate pools 接入训练热路径。论文主表可放 HGNN baseline 与 validation-selected fixed rerank；test-side grid/gate 只能放附表或 error analysis。mimic 本轮只作为诊断对照，不作为优化主线。

## 2. Key Files and Entry Points
|Path|Role|Important Functions / Classes|Notes|
|---|---|---|---|
|configs/data_llldataset_eval.yaml|evaluation data config|`src.evaluation.evaluator::load_test_cases`|包含 DDD test 文件，不改 mimic 主线|
|configs/train_finetune_attn_idf_main.yaml|当前 HGNN finetune/baseline config|`src.training.trainer::main`|checkpoint 位于 `outputs/attn_beta_sweep/edge_log_beta02/checkpoints/best.pt`|
|src/data/dataset.py|case table 读取、DDD split 命名空间|`load_case_files`, `build_namespaced_case_id`, `CaseBatchLoader`|使用 split::relative/path::case_id|
|src/data/build_hypergraph.py|HPO/disease incidence 与 case HPO 处理|`load_static_graph`, `build_case_incidence`, `build_batch_hypergraph`|disease incidence 使用 v59 npz|
|src/evaluation/evaluator.py|exact evaluation|`evaluate`, `compute_topk_metrics`, `save_results`|输出 details/per_dataset/summary|
|tools/export_top50_candidates.py|HGNN top50 candidate + evidence export|`export_top50_candidates`, `_evidence_features`|生成 rerank top50 candidates|
|tools/run_top50_evidence_rerank.py|top50 evidence rerank/grid/gate/protocol|`score_matrix`, `run_validation_select`, `evaluate_fixed_payload`|只在 top50 内重排|
|src/rerank/hpo_semantic.py|HPO semantic overlap|`HpoSemanticMatcher`|使用本地 HPO ontology|
|src/training/hard_negative_miner.py|hard negative mining|`mine_hard_negatives`, `mine_configurable_hard_negatives`|ontology pools 接口存在但 trainer 未传入|
|src/models/hgnn_encoder.py|HGNN encoder|`HGNNEncoder`|本轮只读，不修改|

## 3. DDD Current Metrics
|Dataset|Method|Top1|Top3|Top5|Median Rank|Recall@50|Source File|Notes|
|---|---|---|---|---|---|---|---|---|
|DDD|HGNN_exact_baseline / A_hgnn_only|0.3022|0.4442|0.4967|6.0000|0.7451|D:\RareDisease-traindata\outputs\rerank\top50_candidates_v2.csv|与用户提供 baseline 对齐；mean_rank top50-capped=18.8463，full mean=271.9304|
|DDD|validation_selected_fixed_test|0.3430|0.4704|0.5138|5.0000|0.7451|D:\RareDisease-traindata\outputs\rerank\rerank_fixed_test_metrics.csv|validation grid 选权重，test fixed eval 一次；可作为候选但需说明 gated 未完成|
|DDD|test_side_exploratory:grid_1720|0.3784|0.4888|0.5532|nan|0.7451|D:\RareDisease-traindata\reports\top50_evidence_rerank_v2_report.md|test-side search，只能 error analysis/附表|

## 4. DDD Rank Decomposition
|bucket|num_cases|ratio|interpretation|
|---|---|---|---|
|rank = 1|230|0.3022|HGNN 已精确命中，非当前主要提升空间|
|rank <= 3|338|0.4442|top3 累计命中|
|rank <= 5|378|0.4967|top5 累计命中|
|rank <= 10|438|0.5756|少量重排即可进入可用诊断列表|
|rank <= 20|496|0.6518|top50 内排序提升的中短尾空间|
|rank <= 50|567|0.7451|candidate recall@50，上限为 top50 内 rerank 可达样本|
|rank > 50|194|0.2549|HGNN top50 candidate recall 未覆盖|
|gold absent from candidate universe|0|0.0000|疾病索引/候选全集缺失|
|top50 but rank > 5|189|0.2484|核心 reranker 目标|
|top5 but not top1|148|0.1945|top1 排序损失，适合 evidence rerank|

## 5. DDD Failure Mode Diagnosis
|Failure Mode|Evidence|Num Cases / Ratio|Severity|Why It Matters|
|---|---|---|---|---|
|top50 内排序不足|gold in top50=567/761，但 top1=230/761|337 / 44.28%|High|只需在 HGNN top50 内重排即可提升 top1/top3/top5|
|candidate recall 不足|rank>50 样本无法被 top50 rerank 修复|194 / 25.49%|Medium|需要数据/映射/候选生成或训练侧 hard negative 改善|
|label / mapping mismatch|gold 不在 disease index=0; obsolete=1|0 explicit|Low-Medium|当前未见大面积 index 缺失，但 alias/parent-child 仍需人工审计|
|HPO coverage 弱|top50 miss 中 zero exact overlap=26/194|26 / 13.40%|Medium|会同时影响 candidate recall 与 evidence rerank|
|score calibration / evidence 未融合|validation-selected fixed DDD=0.3430/0.4704/0.5138 vs baseline 0.3022/0.4442/0.4967|N/A|Medium|no-train evidence 可提升 top1，但协议必须 validation-selected|

## 6. DDD Near-Miss Analysis
|gold_disease_id|gold_disease_name|predicted_disease_id|predicted_disease_name|confusion_count|average_gold_rank|ontology_relation|
|---|---|---|---|---|---|---|
|MONDO:0010717|pyruvate dehydrogenase E1-alpha deficiency|MONDO:0018651|obsolete lipoyl transferase 2 deficiency|2|3.0000|unrelated_or_unknown|
|MONDO:0020242|hereditary macular dystrophy|MONDO:0018146|macular telangiectasia type 1|2|3.0000|shared_ancestor|
|MONDO:0005129|cataract|MONDO:0013067|cataract 34 multiple types|1|2.0000|candidate_descendant_of_gold|
|MONDO:0007039|NF2-related schwannomatosis|MONDO:0014630|familial adenomatous polyposis 3|1|2.0000|shared_ancestor|
|MONDO:0007585|exostoses, multiple, type 1|MONDO:0010846|exostoses, multiple, type III|1|2.0000|same_parent|
|MONDO:0007630|North Carolina macular dystrophy|MONDO:0018146|macular telangiectasia type 1|1|2.0000|shared_ancestor|
|MONDO:0007733|holoprosencephaly 3|MONDO:0007819|solitary median maxillary central incisor syndrome|1|2.0000|candidate_descendant_of_gold|
|MONDO:0007986|metatropic dysplasia|MONDO:0008701|achondrogenesis type IA|1|2.0000|shared_ancestor|
|MONDO:0008075|schwannomatosis|MONDO:0014299|LZTR1-related schwannomatosis|1|2.0000|candidate_descendant_of_gold|
|MONDO:0008209|Char syndrome|MONDO:0014213|CTCF-related neurodevelopmental disorder|1|2.0000|shared_ancestor|
|MONDO:0008244|piebaldism|MONDO:0013201|Waardenburg syndrome type 4B|1|2.0000|shared_ancestor|
|MONDO:0008318|Proteus syndrome|MONDO:0013125|CLAPO syndrome|1|2.0000|same_parent|
|MONDO:0008546|thanatophoric dysplasia type 1|MONDO:0008547|thanatophoric dysplasia type 2|1|2.0000|same_parent|
|MONDO:0008612|tuberous sclerosis 1|MONDO:0013199|tuberous sclerosis 2|1|2.0000|same_parent|
|MONDO:0008722|short chain acyl-CoA dehydrogenase deficiency|MONDO:0700250|mitochondrial complex IV deficiency, nuclear type 1|1|2.0000|shared_ancestor|
|MONDO:0008767|neuronal ceroid lipofuscinosis 3|MONDO:0008769|neuronal ceroid lipofuscinosis 2|1|2.0000|same_parent|
|MONDO:0008847|atrichia with papular lesions|MONDO:0007511|ectodermal dysplasia, trichoodontoonychial type|1|2.0000|shared_ancestor|
|MONDO:0008861|3-methylcrotonyl-CoA carboxylase 1 deficiency|MONDO:0009475|isovaleric acidemia|1|2.0000|shared_ancestor|
|MONDO:0008918|carnitine-acylcarnitine translocase deficiency|MONDO:0009282|multiple acyl-CoA dehydrogenase deficiency|1|2.0000|shared_ancestor|
|MONDO:0009130|Dyggve-Melchior-Clausen disease|MONDO:0008477|spondylometaphyseal dysplasia, Kozlowski type|1|2.0000|shared_ancestor|
|MONDO:0009353|homocystinuria due to methylene tetrahydrofolate reductase deficiency|MONDO:0009609|methylcobalamin deficiency type cblG|1|2.0000|shared_ancestor|
|MONDO:0009561|alpha-mannosidosis|MONDO:0018149|GM1 gangliosidosis|1|2.0000|same_parent|
|MONDO:0009603|3-hydroxyisobutyryl-CoA hydrolase deficiency|MONDO:0014314|sacral agenesis-abnormal ossification of the vertebral bodies-persistent notochordal canal syndrome|1|2.0000|shared_ancestor|
|MONDO:0009728|nephronophthisis 1|MONDO:0013302|nephronophthisis 11|1|2.0000|same_parent|
|MONDO:0009746|hereditary sensory and autonomic neuropathy type 4|MONDO:0012092|hereditary sensory and autonomic neuropathy type 5|1|2.0000|same_parent|

## 7. Reranker Audit
当前 reranker 使用 HGNN score、IC overlap、exact overlap、semantic IC overlap、case/disease coverage 与 disease size penalty。所有特征在 case 内 min-max normalization 后线性融合，最终只在 HGNN top50 内重排。当前 test-side grid/gate 已存在，但只能作为 exploratory；validation-selected 路径已有 `top50_candidates_validation.csv`、`val_selected_weights.json` 和 fixed test metrics。需要注意 validation gated 文件为空，正式 gated 方案仍需补跑 validation selection。

## 8. Hard Negative Readiness
已有 hard negative 基础实现和 ontology-aware 接口，但 trainer 未传 `candidate_pools`，所以 ontology-aware 策略目前没有真正生效。DDD 适合做 hard negative，因为 near-miss 中存在 same_parent/shared_ancestor/ancestor-descendant 与 HPO overlap 混淆。推荐先构建 above-gold、same_parent/sibling、shared_ancestor、高 HPO-overlap、query-overlap pools，再独立训练新模型，不能覆盖当前 baseline。

## 9. Improvement Opportunities
### A. 不训练新模型即可尝试
- DDD label normalization audit；MONDO obsolete/synonym/alias audit；candidate universe coverage check；disease-HPO coverage check；rerank feature normalization；validation-selected evidence rerank；DDD near-miss error slicing。

### B. 轻量训练 / reranker 可尝试
- linear reranker；pairwise reranker；listwise reranker；feature fusion MLP；calibration model。所有方案必须 train/validation 选型，test fixed eval 一次。

### C. 数据与标签修复
- MONDO / OMIM / ORPHA 映射对齐；obsolete ID replacement；synonym / alias 合并；parent-child relaxed evaluation；HPO version 对齐；disease hyperedge coverage 修复。

### D. 论文实验设计建议
- 主表只放 HGNN baseline、validation-selected rerank、后续真正接入 candidate_pools 后的 ontology-aware hard negative。test grid 只能作为附表或 error analysis；near-miss analysis 放附表；relaxed metric 放 supplementary；mimic 只作为对照诊断。

## 10. Concrete Next Experiments
|Experiment|Goal|Code Location to Modify Later|Expected Benefit|Risk|Priority|
|---|---|---|---|---|---|
|DDD label normalization audit|确认 alias/obsolete/parent-child 是否制造假错误|tools/audit_processed_mondo_mapping.py 或新增 reports 脚本|减少 label mismatch 假阴性|需要人工规则确认|P0|
|validation-selected linear evidence rerank|固定 validation 权重后 test 一次|tools/run_top50_evidence_rerank.py|提升 DDD top1/top3|objective 选择影响 ALL/DDD tradeoff|P0|
|complete validation gated rerank|验证 gated 是否保护弱证据/高置信 HGNN 样本|tools/run_top50_evidence_rerank.py|比纯 grid 更稳|当前 validation gated 输出为空，需跑完|P0|
|DDD top50 miss coverage repair list|定位 rank>50 的 HPO/gold coverage 弱点|src/data/build_disease_hyperedge_v4_assets.py / hyperedge build scripts|提高 recall@50|数据修复可能改变基线，需版本化|P1|
|above-gold hard negative pool|把排在 gold 前的 DDD near-miss 作为训练负样本|src/training/trainer.py + src/training/hard_negative_miner.py|提升 top1 排序|需要训练新模型，不能本轮执行|P1|
|MONDO sibling/shared-ancestor hard negatives|强化疾病家族内区分|src/training/hard_negative_miner.py|改善 same_parent/shared_ancestor 混淆|ontology 粗粒度父类可能引入噪声|P1|
|high HPO-overlap disease hard negatives|用 phenotype 相似疾病训练判别边界|src/training/hard_negative_miner.py|改善高重叠 near-miss|过拟合 HPO exact overlap|P1|
|pairwise reranker on train/val top50|学习 gold-vs-negative feature difference|tools/train_top50_pairwise_reranker.py / eval script|比手工权重更灵活|必须避免 gold leakage 特征|P1|
|parent-child relaxed supplementary metric|评估 ontology granularity 对 DDD 的影响|src/evaluation/evaluator.py 或独立 supplementary evaluator|解释 near-miss 是否医学可接受|不能替代 exact metric|P2|
|DDD error slicing by HPO count/noise|区分稀疏病例和噪声病例|reports/diagnosis analysis scripts|指导数据清洗优先级|只读分析不能直接提分|P2|

## 11. Questions / Missing Information
- DDD validation split 是否应单独从 DDD train 中切出，还是沿用全训练集 validation candidates。
- 论文主表是否按 DDD objective 选权重，还是按 ALL_top1 / macro objective 选权重。
- disease index 与 MONDO 版本是否固定为 `Disease_index_v4.xlsx` 与 MONDO 2025-06-03。
- 是否允许后续新增独立 validation-selected gated rerank 输出，不覆盖原始 `top50_candidates_v2.csv`。
- ontology-aware hard negative 的 candidate_pools 应在 batch 内动态构建还是预先离线导出。

## Generated Artifacts
- `reports/diagnosis/ddd_rank_decomposition.csv` / `.md`
- `reports/diagnosis/ddd_top50_miss_audit.csv` / `.md`
- `reports/diagnosis/ddd_nearmiss_cases.csv`
- `reports/diagnosis/ddd_nearmiss_pairs.csv`
- `reports/diagnosis/ddd_nearmiss_audit.md`
- `reports/diagnosis/ddd_rerank_protocol_audit.md`
- `reports/diagnosis/ddd_hard_negative_audit.md`

最终判断: DDD 下一步值得进入 validation-selected reranker 和 ontology-aware hard negative training；前者可立即按正式协议推进，后者必须先接入真实 ontology/query/top50 candidate pools 后再训练。