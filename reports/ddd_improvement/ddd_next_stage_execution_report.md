# DDD Next Stage Execution Report

## 1. Executive Summary

本轮补齐了 DDD 的 validation-selected rerank 输出，并生成了独立的新结果文件，未覆盖原始 exact evaluation。`tools/run_top50_evidence_rerank.py` 的 `exploratory`、`validation_select`、`fixed_eval` 三种协议均可用；本轮实际使用 validation candidates 选权重/门控，再对 test candidates 做一次 fixed evaluation。gated validation selection 已完成，输出 `5184` 条 validation gated 组合；本轮采用 validation 上的候选权重子集做 bounded gated search，没有使用 test set 选权重。DDD fixed test 最好结果来自 `validation_grid_DDD_top1`，DDD top1/top3/top5 = `0.3693/0.4875/0.5506`，recall@50 = `0.7451`。gated DDD objective 的 fixed test 为 `0.3693/0.4836/0.5453`，recall@50 不变。

ontology-aware hard negative 的 `candidate_pools` 已接入训练热路径，但本轮没有启动正式训练。新增 `HardNegativeCandidatePoolBuilder` 支持 `above_gold`、`sibling/same_parent`、`shared_ancestor`、`overlap`，并在 `trainer.py::run_one_epoch` 中传给 `mine_configurable_hard_negatives(...)`。dry-run 显示 `HN-mixed` 不再全部退化为 `HN-current`，推断 composition 为 `current=0.494; above_gold=0.150; sibling=0.181; shared_ancestor=0.100; overlap=0.075`。本轮未修改 mimic 数据、配置或结果，未修改 HGNN encoder。

## 2. Validation-selected Rerank Results

| Protocol | Selection Objective | DDD Top1 | DDD Top3 | DDD Top5 | Recall@50 | Source File | Paper Usability |
|---|---:|---:|---:|---:|---:|---|---|
| HGNN baseline | - | 0.3022 | 0.4442 | 0.4967 | 0.7451 | `outputs/rerank/top50_candidates_v2.csv` | 主表 baseline |
| validation-selected grid rerank | DDD_top1 | 0.3693 | 0.4875 | 0.5506 | 0.7451 | `outputs/rerank/ddd_rerank_fixed_test_metrics.csv` | 可作为论文候选 |
| validation-selected grid rerank | ALL_top1 | 0.3430 | 0.4704 | 0.5138 | 0.7451 | `outputs/rerank/ddd_rerank_fixed_test_metrics.csv` | 可作为论文候选 |
| validation-selected gated rerank | DDD_top1 | 0.3693 | 0.4836 | 0.5453 | 0.7451 | `outputs/rerank/ddd_rerank_fixed_test_metrics.csv` | 可作为论文候选，但需说明 bounded gated selection |
| validation-selected gated rerank | ALL_top1 | 0.3417 | 0.4691 | 0.5125 | 0.7451 | `outputs/rerank/ddd_rerank_fixed_test_metrics.csv` | 可作为论文候选，但收益略低 |
| test-side exploratory upper bound | DDD_top1 | 0.3784 | 0.4888 | 0.5532 | 0.7451 | `reports/top50_evidence_rerank_v2_report.md` | 不能进主表，只能 supplementary/error analysis |

相对 baseline，`validation_grid_DDD_top1` 的绝对提升为 top1 `+0.0670`、top3 `+0.0434`、top5 `+0.0539`。所有 top50 rerank 均不改变 recall@50，因为只在 HGNN top50 内重排。

## 3. Selected Weights / Gate Config

### Grid: DDD_top1

- preset: `grid_1617`
- weights: `w_hgnn=0.70, w_ic=0.20, w_exact=0.10, w_semantic=0.15, w_case_cov=0.05, w_dis_cov=0.03, w_size=0.02`
- gate: `None`
- validation DDD top1/top3/top5/recall@50: `0.4085/0.5366/0.5671/0.7012`
- fixed test DDD top1/top3/top5/recall@50: `0.3693/0.4875/0.5506/0.7451`

### Grid: ALL_top1

- preset: `grid_1787`
- weights: `w_hgnn=0.75, w_ic=0.05, w_exact=0.00, w_semantic=0.10, w_case_cov=0.00, w_dis_cov=0.03, w_size=0.01`
- gate: `None`
- validation ALL top1: `0.5461`
- fixed test DDD top1/top3/top5/recall@50: `0.3430/0.4704/0.5138/0.7451`

### Gated: DDD_top1

- preset: `gated_1920`
- weights: `w_hgnn=0.70, w_ic=0.20, w_exact=0.10, w_semantic=0.15, w_case_cov=0.05, w_dis_cov=0.03, w_size=0.02`
- gate: `dataset_gate=ALL, max_exact_threshold=2, max_ic_threshold=0.15, hgnn_margin_threshold=0.10`
- validation DDD top1/top3/top5/recall@50: `0.4085/0.5366/0.5671/0.7012`
- fixed test DDD top1/top3/top5/recall@50: `0.3693/0.4836/0.5453/0.7451`

### Gated: ALL_top1

- preset: `gated_4352`
- weights: `w_hgnn=0.80, w_ic=0.05, w_exact=0.00, w_semantic=0.15, w_case_cov=0.00, w_dis_cov=0.03, w_size=0.00`
- gate: `dataset_gate=ALL, max_exact_threshold=1, max_ic_threshold=0.15, hgnn_margin_threshold=0.10`
- validation ALL top1: `0.5461`
- fixed test DDD top1/top3/top5/recall@50: `0.3417/0.4691/0.5125/0.7451`

## 4. Hard Negative Candidate Pool Fix

| File | Function | Change | Why Needed | Risk |
|---|---|---|---|---|
| `src/training/hard_negative_pools.py` | `HardNegativeCandidatePoolBuilder` | 新增 candidate pool builder，支持 `above_gold`、`sibling/same_parent`、`shared_ancestor`、`overlap` | 让 `HN-mixed` 能拿到真实 ontology/query/top50 pools | pool 覆盖不足时会 fallback |
| `src/training/trainer.py` | `run_one_epoch` | 在训练 hard negative mining 前构建并传入 `candidate_pools` | 修复 `HN-mixed` 实际退化为 `HN-current` 的问题 | 仅 `candidate_pools.enabled=true` 时生效 |
| `src/training/trainer.py` | `main` | 初始化 `build_hard_negative_pool_builder(...)` | 让训练入口从配置创建 pool builder | 默认配置为空，不改变 baseline |
| `src/runtime_config.py` | `resolve_loss_config` | 保留 `loss.hard_negative.candidate_pools` 配置 | 避免配置解析时丢弃 pool 参数 | 默认 `{}` 不影响旧配置 |
| `tools/run_top50_evidence_rerank.py` | `gate_mask` / `_clean_gate` | 增加 `dataset_gate`，支持 `ALL`、`DDD_ONLY`、`NON_MIMIC` | gated rerank 支持 dataset gate | 默认 `ALL`，旧行为不变 |

这些修改不影响 mimic 数据、mimic 配置、mimic 结果，也不影响 HGNN encoder。回滚方式是撤销上述文件改动，并删除新增 DDD 专用配置和报告产物。

## 5. HN Dry-run Results

| Strategy | Num Cases | Avg Negatives | Source Composition | Fallback Rate | Notes |
|---|---:|---:|---|---:|---|
| HN-current | 16 | 10.0 | `current=0.994; overlap=0.006` | 0.9938 | baseline miner |
| HN-overlap | 16 | 10.0 | `above_gold=0.119; sibling=0.056; overlap=0.825` | 0.0000 | 候选集合有交叠，composition 按集合归属推断 |
| HN-sibling | 16 | 10.0 | `current=0.406; above_gold=0.075; sibling=0.512; overlap=0.006` | 0.4063 | sibling pool 不足时补 current |
| HN-shared-ancestor | 16 | 10.0 | `sibling=0.006; shared_ancestor=0.994` | 0.0000 | shared_ancestor 覆盖充分 |
| HN-above-gold | 16 | 10.0 | `current=0.725; above_gold=0.269; overlap=0.006` | 0.7250 | above_gold 每 case 数量较少 |
| HN-mixed | 16 | 10.0 | `current=0.494; above_gold=0.150; sibling=0.181; shared_ancestor=0.100; overlap=0.075` | 0.4938 | 已使用 configured pools |

dry-run 文件：

- `reports/ddd_improvement/ddd_hn_candidate_pool_audit.md`
- `reports/ddd_improvement/ddd_hn_candidate_pool_stats.csv`
- `reports/ddd_improvement/ddd_hn_dryrun_samples.csv`

明确结论：`HN-mixed` 已经使用 ontology/query/top50 pools，不再全部退化为 `HN-current`。仍有 fallback，因为 `above_gold` 等 pool 对部分 case 覆盖不足，这是预期行为，不是硬编码假数据。

## 6. DDD Dedicated Training Config

已新增 `configs/train_finetune_ddd_ontology_hn.yaml`，用于后续正式训练。该配置：

- 输出目录为 `outputs/ddd_ontology_hn/`；
- hard negative strategy 为 `HN-mixed`；
- candidate pool sources 包含 `above_gold`、`same_parent`、`sibling`、`shared_ancestor`、`hpo_overlap`；
- 保留 baseline checkpoint 对照路径 `outputs/attn_beta_sweep/edge_log_beta02/checkpoints/best.pt`；
- 不覆盖已有配置；
- 本轮未启动训练。

后续训练命令只作为准备，不在本轮执行：

```powershell
D:\python\python.exe -m src.training.trainer --config configs\train_finetune_ddd_ontology_hn.yaml
```

## 7. Paper Table Recommendation

可以进入论文主表：

- HGNN exact baseline；
- validation-selected fixed rerank，优先推荐 `validation_grid_DDD_top1` 作为 DDD-focused 结果；
- 后续真正完成训练后的 ontology-aware HN 结果，如果使用固定配置、独立输出目录和一次性 test evaluation。

不能进入论文主表：

- test-side grid；
- test-side gate；
- 未经 validation 选择的 exploratory 结果；
- HN dry-run 结果；
- 任何使用 test set 选权重或调 gate 的结果。

## 8. Next Commands

复现本轮 rerank：

```powershell
D:\python\python.exe reports\ddd_improvement\run_ddd_validation_selected_rerank.py
```

复现 HN candidate pool dry-run：

```powershell
D:\python\python.exe reports\ddd_improvement\run_ddd_hn_candidate_pool_dryrun.py
```

语法检查：

```powershell
D:\python\python.exe -m py_compile tools\run_top50_evidence_rerank.py src\training\hard_negative_pools.py src\training\trainer.py src\runtime_config.py reports\ddd_improvement\run_ddd_validation_selected_rerank.py reports\ddd_improvement\run_ddd_hn_candidate_pool_dryrun.py
```

## 9. Remaining Risks

- validation set DDD 只有 `164` 个 case，DDD-focused 权重可能比 ALL objective 更偏向 DDD。
- gated selection 本轮是 validation 上的 bounded gated search，已完成 `5184` 个组合，但不是对全部 grid 权重做无界枚举；报告和论文中需要说明。
- `above_gold` pool 依赖已有 train/validation top50 candidate 文件，gold 不在 top50 的 train case 无法贡献该 pool。
- MONDO relation 的 parent/ancestor 粒度可能偏粗，same_parent/shared_ancestor hard negatives 可能引入语义过宽负样本。
- DDD-focused rerank 明显提升 DDD，但会降低 mimic_test 与 ALL；本阶段不优化 mimic 主线，论文表需要按研究目标解释。
- 后续正式训练必须使用独立输出目录 `outputs/ddd_ontology_hn/`、固定 seed，并只在训练完成后做一次固定 test evaluation。

最终判断：DDD 可以进入正式 ontology-aware hard negative training，但必须使用本轮新增的 candidate_pools 接入路径、独立训练目录和 validation-selected/fixed-test 协议。
