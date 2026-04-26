# 1. Executive Summary

本次审计未修改源代码。当前 RareDisease HGNN 主线是：`H_disease` 进入 `HGNNEncoder` 得到全局 HPO 节点表示，病例侧通过 `H_case` 做 `case_refiner` 和 attention readout，疾病侧通过 `H_disease^T @ Z` 得到疾病表示，最后用 cosine full-pool scoring 做全疾病池排序。这个主线与“避免训练病例超边污染疾病图”的目标一致，关键证据在 `src/models/model_pipeline.py:331-357` 和 `src/models/model_pipeline.py:301-315`。

主要瓶颈不是单一模型结构问题，而是四类因素叠加：

1. `mimic_test`：低 exact overlap、低 static retrieval、同一 `case_id` 多标签污染和临床噪声共同主导。当前代码在 `src/data/build_hypergraph.py:245-251` 对同一 case 只取 `group_df[label_col].iloc[0]`，多标签 case 会被静默压成单标签。
2. `HMS`：当前 test split 只有 25 cases。只读审计显示原始 `LLLdataset/dataset/HMS.jsonl` 为 88 行，处理全量 `LLLdataset/dataset/processed/HMS.xlsx` 为 88 cases，而测试集 `LLLdataset/dataset/processed/test/HMS.xlsx` 为 25 cases。当前 25 个样本更像 split/论文口径差异，不是运行时 mapping 过滤造成。
3. `DDD`：median_rank=6 但 top1 不高，说明召回到近邻区域后区分相似疾病失败；训练 hard negative 是在线 top-score negative，但缺少 ontology/family-aware hard negatives。
4. `LIRICAL`：median_rank=1 但 mean_rank=301.47，明显由少数 outlier 拉高。当前 details 中最严重 outlier 为 `case_344 / MONDO:0010879`，rank=14693，需要优先做 gold mapping 和超边覆盖审计。

高风险实现点：

- exact `MONDO` ID evaluation 不考虑 synonym、subtype、obsolete/merged ID；这会比 DeepRare 论文类 agentic evaluator 更严格。
- HPO/MONDO 格式已经统一为 `HP:ddddddd` / `MONDO:ddddddd`，但没有看到运行时 obsolete HPO 替换和 ancestor/descendant semantic matching。
- 疾病超边权重确实进入 encoder/readout，但 `HGNNEncoder._propagate()` 使用 `D_e^{-1}` 和 `D_v^{-1/2}` 归一化，且 disease readout 是未归一化的 `H_disease^T @ Z`；v59 每个疾病列权重和约为 1，因此“大超边总权重天然占优”被缓解，但“大超边语义稀释”和“小超边信息不足”仍是风险。

# 2. Current Framework Map

主入口与配置：

- 训练命令：`run_full_train.cmd`
- stage1 pretrain 配置：`configs/train_pretrain.yaml`
- stage2 finetune 配置：`configs/train_finetune_attn_idf_main.yaml`
- 评估配置：`configs/data_llldataset_eval.yaml`
- 训练入口：`src/training/trainer.py:887` 的 `main()`
- 评估入口：`src/evaluation/evaluator.py:997` 的 `main()`
- 运行时配置装配：`src/runtime_config.py:76` 的 `build_model_pipeline_config()`，`src/runtime_config.py:153` 的 `resolve_loss_config()`

数据预处理与图构建入口：

- 病例表 MONDO/HPO 构建：`src/data/build_llldataset_mondo_case_tables.py:633` 的 `main()`
- DDD/DDG2P 构建：`src/data/build_ddg2p_processed_excel.py:422` 的 `main()`
- 疾病超边 v3 构建：`src/data/build_disease_hyperedge_v3.py:179` 的 `main()`
- 疾病超边 v4 索引/稀疏资产：`src/data/build_disease_hyperedge_v4_assets.py:126` 的 `main()`
- 稀疏 triplets/npz 构建：`src/graph/incidence_builder.py:24` 的 `build_sparse_triplets()` 和 `src/graph/incidence_builder.py:145` 的 `main()`
- 训练/评估 batch hypergraph：`src/data/build_hypergraph.py:397` 的 `build_batch_hypergraph()`

模型与训练关键模块：

- HGNN encoder：`src/models/hgnn_encoder.py:27` `HGNNEncoder`
- case readout / disease readout：`src/models/readout.py:38` `HyperedgeReadout`
- case conditioned refiner：`src/models/case_refiner.py:64` `CaseConditionedRefiner`
- model pipeline：`src/models/model_pipeline.py:25` `ModelPipeline`
- similarity scoring：`src/models/scorer.py:12` `CosineScorer`
- loss：`src/training/loss_builder.py:78` `compute_loss()`，`src/training/loss_builder.py:142` `FullPoolCrossEntropyLoss`
- hard negative mining：`src/training/hard_negative_miner.py:6` `mine_hard_negatives()`
- training metrics：`src/training/trainer.py:205` `compute_topk_metrics()`
- evaluation metrics：`src/evaluation/evaluator.py:461` `compute_topk_metrics()`

# 3. Data Flow and Tensor Flow

当前数据流：

```text
raw case HPO / disease IDs
  -> build_llldataset_mondo_case_tables.py / build_ddg2p_processed_excel.py
  -> processed long table: [case_id, mondo_label, hpo_id]
  -> load_case_files() / load_test_cases()
  -> build_case_incidence(): H_case [num_hpo, B]
  -> load_static_graph(): H_disease [num_hpo, M]
  -> ModelPipeline.forward()
  -> encoder(H_disease): Z [num_hpo, hidden_dim]
  -> optional CaseConditionedRefiner(Z, H_case)
  -> HyperedgeReadout:
       case_repr [B, hidden_dim]
       disease_repr [M, hidden_dim]
  -> CosineScorer: scores [B, M]
  -> full-pool CE / rank metrics
```

关键张量形状：

| 模块 | 文件/函数 | 输入 | 输出 |
|---|---|---|---|
| 静态图加载 | `src/data/build_hypergraph.py:358 load_static_graph()` | index xlsx + `v59DiseaseHy.npz` | `H_disease [num_hpo, num_disease]` |
| 病例 incidence | `src/data/build_hypergraph.py:205 build_case_incidence()` | `case_df` long table | `H_case [num_hpo, num_case]` |
| batch graph | `src/data/build_hypergraph.py:397 build_batch_hypergraph()` | `H_case`, `H_disease` | dict with labels, cols, optional `H=[H_case|H_disease]` |
| encoder | `src/models/hgnn_encoder.py:84 forward()` | sparse `H_disease [N, M]` | `Z [N, d]` |
| case refiner | `src/models/case_refiner.py:92 forward()` | `Z [N,d]`, `H_case [N,B]` | active-edge `RefinedCaseNodeState` |
| case readout | `src/models/readout.py:231 build_case_repr()` | `Z [N,d]`, `H_case [N,B]` | `case_repr [B,d]` |
| disease readout | `src/models/readout.py:312 build_disease_repr()` | `Z [N,d]`, `H_disease [N,M]` | `disease_repr [M,d]` |
| scorer | `src/models/scorer.py:19 forward()` | `case_repr [B,d]`, `disease_repr [M,d]` | `scores [B,M]` |
| loss | `src/training/loss_builder.py:78 compute_loss()` | `scores [B,M]`, `targets [B]` | CE + optional hard rank loss |

# 4. Dataset and Candidate Pool Audit

只读统计结果：

| 数据集 | processed/test cases | labels | rows | valid HPO avg | 运行时 skipped |
|---|---:|---:|---:|---:|---:|
| `mimic_test` | 1873 | 353 | 21749 | 10.06 | 0 |
| `HMS` test | 25 | 19 | 530 | 21.20 | 0 |
| `HMS` processed full | 88 | 36 | 1710 | - | - |
| `HMS.jsonl` raw | 88 lines | - | - | - | - |
| `DDD` test | 761 | 756 | 13712 | 17.59 | 0 |
| `LIRICAL` test | 59 | 34 | 1027 | 17.41 | 0 |
| `LIRICAL` processed full | 370 | 252 | 5270 | - | - |

候选池：

- `Disease_index_v4.xlsx` 有 16443 个疾病，`H_disease` shape 为 `(19566, 16443)`。
- 当前评估是 full-pool exact ID 排名，`scores.shape == (num_case, resources["num_disease"])`，检查在 `src/evaluation/evaluator.py:754-760`。
- `gold_in_disease_pool=1.0` 对应代码层面仅是 `case_table["mondo_label"].isin(disease_to_idx)`，见 `src/evaluation/evaluator.py:575-582`；它不检查 synonym、subtype、obsolete/merged ID。

HMS 重点结论：

- HMS 当前只有 25 个 test cases，不是 `load_test_cases()` 或 `build_case_incidence()` 过滤导致，因为运行时 `num_skipped=0`。
- 与 raw 88 行相比，test split 25 cases 需要核对论文 DeepRare/HMS 的评估规模。如果论文报告的是全量 HMS 或不同 split，则当前结果不能直接横向比较。

# 5. HPO/Disease Mapping Audit

ID 格式：

- 当前 processed/test 文件中 `hpo_id` 均匹配 `^HP:\d{7}$`，`mondo_label` 均匹配 `^MONDO:\d{7}$`。
- index 文件也无格式异常：`HPO_index_v4.xlsx` 19566 rows，`Disease_index_v4.xlsx` 16443 rows。
- 构建侧 `normalize_prefixed_id()` 在 `src/data/build_llldataset_mondo_case_tables.py:151` 负责补齐前缀和位数，`extract_hpo_ids()` 在 `src/data/build_llldataset_mondo_case_tables.py:176` 只保留可规范化的 HP ID。

MONDO/ORPHA/OMIM 映射：

- `load_orpha_to_mondo_map()`：`src/data/build_llldataset_mondo_case_tables.py:242-260`
- `load_omim_to_mondo_map()`：`src/data/build_llldataset_mondo_case_tables.py:263-281`
- manual ORPHA overrides：`src/data/build_llldataset_mondo_case_tables.py:29-38`
- disease name synonym/name index：`src/data/build_llldataset_mondo_case_tables.py:226-234`

风险：

- 代码中有 `NAME_PREFIXES_TO_STRIP = ("obsolete:", "non rare in europe:")`，但这主要用于名称归一化，不等价于 obsolete MONDO/HPO replacement。
- `rg` 未发现主训练/评估热路径使用 HPO `alt_id`、`replaced_by`、ancestor/descendant 的逻辑。
- 对 DeepRare 类系统而言，疾病 normalizer 可能会通过 name/synonym/LLM evaluator 接受等价疾病；当前项目只按 exact `MONDO` ID 计算，指标会更严。

# 6. Hypergraph Construction Audit

疾病超边：

- v3 从 `orphanet -> GARD -> HPOA` 优先级合并，见 `src/data/build_disease_hyperedge_v3.py:20-25`。
- 同一 `mondo_id,hpo_id` 使用 `raw_weight.max()` 去重，见 `src/data/build_disease_hyperedge_v3.py:89-94`。
- 每个疾病内部按 `raw_weight / total_raw_weight` 归一化，见 `src/data/build_disease_hyperedge_v3.py:120-134`。
- 权重和校验在 `src/data/build_disease_hyperedge_v3.py:153-158`。
- triplets 写入 `rows/cols/vals/shape`，见 `src/graph/incidence_builder.py:56-62` 和 `src/graph/incidence_builder.py:102`。

v59 实际静态图只读统计：

- `H_disease`: `(19566, 16443)`, `nnz=227907`
- 权重范围：`0.0001135` 到 `0.9504`
- 疾病列权重和：min `0.0`，median `~1.0`，max `~1.0`
- 疾病 HPO 数：min `0`，median `7`，max `226`
- HPO disease frequency：min `0`，median `1`，max `2169`

风险判断：

- 权重确实进入 `H_disease`，并被 encoder/readout 使用。
- 但 `HGNNEncoder._propagate()` 当前固定 `W=I`，通过 `D_e^{-1}` 和 `D_v^{-1/2}` 做传播归一化，见 `src/models/hgnn_encoder.py:57-83`。这会降低大超边总权重优势，但可能稀释大超边中关键 HPO。
- `disease_repr = H_disease^T @ Z` 未再除以列权重，见 `src/models/readout.py:312-315`。由于 v59 多数列权重和已经为 1，这通常合理；但 `col_sum=0` 的空疾病会得到近零向量，应统计是否进入 candidate pool 并影响排序尾部。

# 7. Encoder/Readout/Scoring Audit

Encoder：

- `HGNNEncoder` 是两层传播，初始节点 embedding `X0 [num_hpo, hidden_dim]` 可学习。
- `ModelPipeline.forward()` 在无 override 时执行 `self.encoder(prepared_h_disease)`，证据 `src/models/model_pipeline.py:331-336`。
- 训练热路径没有把 `H_case` 拼进 encoder；`H` 仅为兼容字段，`src/models/model_pipeline.py:122-144` 也允许缺省 `H`。

Case side：

- `CaseConditionedRefiner` 输入 `node_repr [num_hpo, hidden_dim]` 与 `H_case [num_hpo, num_case]`，输出 active-edge 状态，见 `src/models/case_refiner.py:92-121`。
- `HyperedgeReadout` 用 leave-one-out context + attention 得到 `case_repr [num_case, hidden_dim]`，见 `src/models/readout.py:140-229`。
- case-side IDF weighting 已经进入 `H_case` 的 edge weight：`case_noise_control.enabled=true` 时，`_build_case_hpo_weights()` 用 disease-side HPO specificity，见 `src/data/build_hypergraph.py:95-136`；配置在 `configs/train_finetune_attn_idf_main.yaml`。

Disease side：

- `build_disease_repr()` 是 `H_disease^T @ Z`，见 `src/models/readout.py:312-315`。
- 评估阶段 `precompute_disease_side()` 缓存 `node_repr` 和 `disease_repr`，见 `src/models/model_pipeline.py:301-315`，数学等价。

Scoring：

- `CosineScorer` 对 `case_repr` 和 `disease_repr` 分别 L2 normalize，再矩阵乘法，见 `src/models/scorer.py:26-29`。
- scorer 无可学习参数，因此 top1 的进一步提升主要依赖 representation、case weighting、训练目标和 rerank/fusion。

# 8. Loss and Negative Sampling Audit

Loss：

- 主体是 full-pool CE：`F.cross_entropy(scores / tau, targets)`，见 `src/training/loss_builder.py:114-122`。
- `poly_epsilon` 可叠加 poly loss，见 `src/training/loss_builder.py:116-120`。
- finetune 配置：`temperature=0.18`，`poly_epsilon=2.0`。

Negative sampling：

- `mine_hard_negatives()` 从当前模型 `scores` 中屏蔽 gold 后取 top-k，见 `src/training/hard_negative_miner.py:6-27`。
- 训练时在 `epoch >= start_epoch` 启用，见 `src/training/trainer.py:716-724`。
- hard rank loss 用 margin violation：`relu(margin - pos + neg)`，见 `src/training/loss_builder.py:55-76`。

风险：

- 这不是随机负样本，已经是在线 hard negatives。
- 但它是“当前模型高分负例”，不是 disease family / ontology sibling / semantic-neighbor negatives。对 DDD 这种 median_rank 好但 top1 不高的 near-miss 场景，当前 hard negative 可能不够稳定地覆盖同父类和相似 HPO 疾病。

# 9. Evaluation Metric Audit

评估流程：

- `load_test_cases()` 聚合为每 case 一条记录，见 `src/evaluation/evaluator.py:371-458`。
- 可评估判定只检查 label 是否在 disease index、是否至少一个有效 HPO，见 `src/evaluation/evaluator.py:575-582`。
- 排名通过 `torch.argsort(scores, descending=True)` exact 定位 gold disease index，见 `src/evaluation/evaluator.py:763-782`。
- top-k/mean/median/rank_le_50 基于 1-indexed true rank，见 `src/evaluation/evaluator.py:461-482`。

关键风险：

- 多标签 `case_id` 在评估前会被压成 `group_df[label_col].iloc[0]`，见 `src/evaluation/evaluator.py:371-458` 的 case aggregation 和 `src/data/build_hypergraph.py:245-246` 的 batch 构建。这对 `mimic_test` 影响明显：只读统计显示 `mimic_test` 有 227/1873 multi-label cases。
- exact `MONDO` ID 评估不接受同义词、父子疾病、obsolete replacement、临床等价诊断；与 DeepRare 论文的 agentic reasoning/evaluator 口径可能不一致。
- `gold_in_disease_pool=1.0` 不代表 disease normalizer 正确，只代表 exact `mondo_label` 存在于 `Disease_index_v4.xlsx`。

# 10. Dataset-specific Root Cause Diagnosis

## mimic_test

判断：主要瓶颈是病例 HPO 噪声、低 exact overlap、multi-label 污染和 candidate/ranking 过度依赖 exact overlap，不是单纯 disease pool 缺失。

证据：

- `avg_valid_hpo_count=10.06`，`avg_overlap_count=1.19`，`overlap_zero_rate=0.3668`，`low_overlap<=1 ~0.697`。
- static none top5 仅 `0.0299`，rank_le_50 `0.1548`；IDF top5 将 rank_le_50 提到 `0.2061` 但 top5 降到 `0.0256`。
- 当前评估文件统计：`mimic_test` 1873 cases，227 multi-label cases。
- `src/data/build_hypergraph.py:245-251` 对同 case 多 label 只取第一个 label。

最先验证假设：

- 把 `mimic_test` multi-label case 单独评估为“任一候选 label 命中”或剔除 multi-label case，看 top1/top5 是否显著上升；如果上升，先修数据语义而不是继续调 HGNN。

## HMS

判断：主要瓶颈是 sample count/split 口径、病例 HPO 与疾病超边对齐差，以及小样本方差。

证据：

- raw `HMS.jsonl` 88 lines，processed full 88 cases，但 test split 25 cases。
- test 25 cases，19 labels，valid HPO avg 21.20，运行时 skipped=0。
- static none top5 `0.08`、rank_le_50 `0.40`；IDF top5 rank_le_50 可到 `0.72`，说明召回能被 key-HPO filtering 改善，但 top5 不稳定。

最先验证假设：

- 核对 DeepRare HMS 使用的是 88 全量还是 25 test split；若论文是全量/不同 split，先重建可比评估集。

## DDD

判断：主要瓶颈是相似疾病 near-miss 和细粒度标签区分；模型能召回局部区域，但 top1/top3 不够。

证据：

- 当前 top1 `0.3022`、top5 `0.4967`，median_rank=6，rank_le_50=0.745。
- static none top5 `0.490` 接近 HGNN top5，说明疾病超边本身对 DDD 有强信号。
- RRF/score fusion 可把 DDD top5 提高到约 `0.53+`，但不同 lambda 下 top1/top5 tradeoff 明显。

最先验证假设：

- 对 DDD top50 候选做 family-aware / IC-overlap reranker，看 median/top5 是否能在不显著损害 top1 的情况下提升。

## LIRICAL

判断：整体已较好，主要问题是少数 outlier 的 gold mapping、超边缺失或 exact-ID 口径。

证据：

- top1 `0.5254`，median_rank=1，但 mean_rank=301.47。
- details 中 outlier：`case_344 / MONDO:0010879` rank=14693，`case_161 / MONDO:0014857` rank=1386。
- overlap_zero_rate=0，说明不是无表型匹配，而是少数 case 的 label/候选池/超边语义可能错位。

最先验证假设：

- 审计 LIRICAL rank>100 的 cases：gold MONDO 是否是 obsolete/subtype、pred_top1 是否为同义或父子疾病、gold disease hyperedge 是否为空或缺 key HPO。

# 11. High-risk Bugs or Mismatches

| 检查项 | 风险判断 | 证据位置 | 建议验证方式 |
|---|---|---|---|
| disease ID / gold label / candidate pool 一致性 | 中风险；主链有断言，但命名复杂 | `src/data/build_hypergraph.py:456-466`, `src/models/model_pipeline.py:239-299` | 对每个 batch dump `mondo_label -> disease_idx -> score_col` 三元组抽样比对 |
| `gold_in_disease_pool` 是否检查 synonym/subtype/obsolete | 高风险；只查 exact ID | `src/evaluation/evaluator.py:575-582` | 加 synonym/subclass/obsolete audit，不直接改 metric |
| HPO ID 格式 | 低风险 | processed/index 统计均 0 bad format | 保持构建后格式断言 |
| obsolete HPO 替换 | 高风险；未见热路径替换 | `rg obsolete/alt_id/replaced_by` 未见训练评估使用 | 从 `raw_data/hp.json` 建 obsolete map，统计 processed 中命中数 |
| ancestor/descendant | 高风险；基本未用 | 未见 semantic closure 用于 scoring/eval | 做离线 semantic similarity audit |
| 疾病超边权重进入模型 | 已进入 | `src/graph/incidence_builder.py:56-58`, `src/models/hgnn_encoder.py:57-83`, `src/models/readout.py:312-315` | 对 weight shuffle/uniform 做 ablation |
| 权重被归一化冲掉 | 中风险 | 每疾病列权重和约 1；encoder 另有度归一化 | 比较 raw weighted vs binarized H_disease |
| 大超边天然占优 | 中低风险 | 列权重归一化；cosine scorer | 统计 top predictions 与 hyperedge size 相关性 |
| 小超边被惩罚 | 中风险 | 疾病 HPO count min 0、median 7 | 分 bucket 评估 gold hyperedge size vs rank |
| case hyperedge 进入 encoder | 低风险 | `src/models/model_pipeline.py:331-336` 只 encoder `H_disease` | 用 hook/assert 确认 encoder 输入 shape `[N,M]` |
| train/test case 泄漏 | 中风险；命名空间已修，但真实重复未审完 | `src/data/dataset.py:102-126`, `src/evaluation/evaluator.py:554-558` | 基于 HPO set + label 的 fingerprint 查重复 |
| negative sampling 只是随机 | 低风险；不是随机 | `src/training/hard_negative_miner.py:6-27` | 增加 ontology-aware negative 对照 |
| DDD near-miss | 高风险 | median_rank=6/top1=0.3022 | 分析 top5/top10 错误是否同父类/同系统 |
| mimic exact overlap 依赖 | 高风险 | static/top-k 与 overlap 指标很低 | 加 semantic match/fusion 离线评估 |
| exact disease ID vs paper evaluator | 高风险 | `src/evaluation/evaluator.py:763-782` | 对 top5 做 synonym/parent-child 宽松命中审计 |
| HMS 25 样本 | 高风险 | raw/full 88 vs test 25 | 核论文 split，重建 full-HMS evaluation |
| LIRICAL outlier | 高风险 | rank max 14693 | 对 rank>100 case 做 mapping/超边 audit |

# 12. Minimal Experiment Roadmap

最多 8 个小实验，按优先级：

1. HMS sample count/mapping audit  
改动位置：新增离线脚本或 notebook，读取 `LLLdataset/dataset/HMS.jsonl`、`processed/HMS.xlsx`、`processed/test/HMS.xlsx`。  
是否训练：否。  
指标：不直接提升，确认 top1/top5 可比性。  
主要数据集：HMS。  
成功下一步：按论文规模重建 evaluation config。  
失败停止：若论文确为 25 cases，则停止“样本规模不一致”方向。

2. LIRICAL outlier audit  
改动位置：离线分析 `outputs/..._details.csv` + `Disease_index_v4.xlsx` + disease hyperedge。  
是否训练：否。  
指标：mean_rank、rank_le_50，可能不影响 top1。  
主要数据集：LIRICAL。  
成功下一步：修 mapping/obsolete/subtype 口径或补 key HPO。  
失败停止：若 outlier 都是合理 hard case，则停止 mapping 怀疑。

3. Multi-label case evaluation audit  
改动位置：`src/evaluation/evaluator.py` 后处理或独立脚本，不先改主 metric。  
是否训练：否。  
指标：top1/top5 的 upper bound。  
主要数据集：mimic_test、DDD。  
成功下一步：设计 multi-label-aware eval 或清洗数据。  
失败停止：如果增益很小，停止把 mimic 主因归为多标签。

4. Score-level rank fusion: HGNN + IC overlap + semantic similarity  
改动位置：独立 rerank/fusion 脚本，复用 `outputs/..._details.csv` 和 `H_disease`。  
是否训练：否。  
指标：rank_le_50、top5、median_rank。  
主要数据集：mimic_test、HMS、DDD。  
成功下一步：只把最佳 fusion 作为 inference reranker 接入。  
失败停止：停止 score-level fusion 方向。

5. Case-side soft IDF/IC weighting readout ablation  
改动位置：`src/data/build_hypergraph.py` 的 `_build_case_hpo_weights()` / `case_noise_control` 配置。  
是否训练：建议先不训练做 eval-time ablation；再短训。  
指标：rank_le_50、median_rank，mimic top5 可能小幅变动。  
主要数据集：mimic_test、HMS。  
成功下一步：调 beta/normalization，不改结构。  
失败停止：停止继续强化 hard IDF filtering。

6. DDD top50 hard-negative reranker  
改动位置：离线 rerank top50；后续可放在 `src/evaluation` 的 optional reranker。  
是否训练：先否；若有效再训练轻量 reranker。  
指标：top1/top3/top5。  
主要数据集：DDD。  
成功下一步：将 family/IC features 作为 training hard negatives。  
失败停止：停止 DDD reranker，转向数据覆盖。

7. Training hard negative mining: semantic/family-aware negatives  
改动位置：`src/training/hard_negative_miner.py` 和 `src/training/trainer.py:716-724`。  
是否训练：需要。  
指标：DDD top1/top3，LIRICAL top1。  
主要数据集：DDD、LIRICAL。  
成功下一步：加入 ontology sibling 或 HPO similarity cache。  
失败停止：停止复杂 negative，保留在线 top-score negatives。

8. Disease hyperedge controlled augmentation  
改动位置：`src/data/build_disease_hyperedge_v3.py` 或新建离线补边资产，不直接覆盖 v59。  
是否训练：先静态 retrieval 验证，再训练。  
指标：mimic rank_le_50/top5、HMS top5、LIRICAL outlier mean_rank。  
主要数据集：mimic_test、HMS、LIRICAL。  
成功下一步：只对短超边/低覆盖 gold disease 做受控补边。  
失败停止：停止泛化补边，避免污染 disease graph。

# 13. Exact Files That Should Be Modified Later

后续若进入修复/实验阶段，建议优先修改或新增以下文件：

- `src/evaluation/evaluator.py`：增加 optional multi-label audit、synonym/subtype relaxed audit、outlier export；不要直接替换主 exact metric。
- `src/data/build_hypergraph.py`：改进 `build_case_incidence()` 对同一 `case_id` 多 `mondo_label` 的处理，从静默 `iloc[0]` 改为 hard fail 或显式 multi-label mode。
- `src/data/build_llldataset_mondo_case_tables.py`：补 obsolete MONDO/HPO、alt ID、synonym audit 输出。
- `src/training/hard_negative_miner.py`：增加 ontology/HPO-similarity aware hard negative mining。
- `src/models/readout.py`：若 soft IC 权重需要进入 attention prior，可扩展 `attn_prior_mode`，但应保持默认不变。
- `tools/audit_dataset_hyperedge_similarity.py`：扩展到 ancestor/descendant semantic similarity 和 outlier drilldown。
- `configs/data_llldataset_eval.yaml`：若确认 HMS 论文口径不是 25 test cases，新增 full-HMS eval config，而不是覆盖现有 split。
- `configs/train_finetune_attn_idf_main.yaml`：仅在离线证据充分后调整 `case_noise_control`、hard negative 和 sampler 参数。
