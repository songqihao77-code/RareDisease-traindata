# Snapshot Consistency Audit

Generated at: `2026-04-30T09:14:38`

## 1. Executive Summary

- Audited run internal consistency: PASS (0 High/Critical internal failures).
- Current top-level config can reproduce audited run: NO (4 High/Critical mismatches).
- Locked config can serve as frozen reproduction config: YES (0 High/Critical locked-config mismatches).
- Checkpoint / candidate / final result mix detected inside audited run: NO.
- Locked config contains ignored keys: NO.
- Runner warning/strict config validation available: YES.
- Confirmed silently ignored config keys after runner validation: none.
- Multiple output snapshot mix risk: YES.
- Baseline recommendation: use the locked config plus frozen manifest for traceability; do not use the mutable top-level config as the reproduction contract.

The high-risk state has moved from unguarded config drift to a locked snapshot plus runner-level warning/strict validation. The mutable config still does not reproduce the audited snapshot.

## 2. Audited Snapshot

- run_dir: `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5`
- top_config: `D:\RareDisease-traindata\configs\mainline_full_pipeline.yaml`
- locked_config: `D:\RareDisease-traindata\configs\mainline_full_pipeline_hybrid_tag_v5.locked.yaml`
- runner: `D:\RareDisease-traindata\tools\run_full_mainline_pipeline.py`
- frozen manifest: `D:\RareDisease-traindata\outputs\snapshot_consistency_lock_audit\frozen_snapshot_manifest.json`

## 3. Repository State

| item | value | status | notes |
| --- | --- | --- | --- |
| cwd | D:\RareDisease-traindata | OK |  |
| audit_time | 2026-04-30T09:14:38 | OK |  |
| python_version | 3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)] | OK | Windows-11-10.0.26100-SP0 |
| git_status_short | M api_tset.py<br> M configs/mainline_full_pipeline.yaml<br>A  "reports/TAG Implementation Report.md"<br> M reports/mainline/full_pipeline_readme.md<br> M src/models/model_pipeline.py<br> M src/runtime_config.py<br> M src/training/trainer.py<br> M tools/run_full_mainline_pipeline.py<br>?? configs/mainline_full_pipeline_hybrid_tag_v5.locked.README.md<br>?? configs/mainline_full_pipeline_hybrid_tag_v5.locked.yaml<br>?? configs/train_finetune_dual_stream_frozen.yaml<br>?? models/<br>?? reports/current_framework_accuracy_audit/<br>?? reports/dual_stream_frozen_ablation/<br>?? reports/llm_reranker_top30_error_analysis.md<br>?? reports/llm_reranker_top30_v2_deep_analysis_and_v3_update.md<br>?? reports/llm_reranker_top30_v4_small_model_update.md<br>?? reports/snapshot_consistency_audit/<br>?? src/extract_biolord_hpo.py<br>?? src/hgnn_encoder_tag.py<br>?? src/models/dual_stream_fusion.py<br>?? tools/analyze_candidate_recall.py<br>?? tools/audit_current_framework_accuracy.py<br>?? tools/audit_snapshot_consistency.py<br>?? tools/run_dual_stream_frozen_ablation.py<br>?? tools/run_llm_reranker_top30.py<br>?? tools/run_llm_reranker_top30_v2.py<br>?? tools/run_llm_reranker_top30_v3.py<br>?? tools/run_llm_reranker_top30_v4.py<br>?? tools/run_llm_reranker_top30_v5.py<br>?? tools/run_llm_reranker_top30_v5_guarded.py<br>?? v59_rare_disease_authoritative_diagnostic_guide_final.zip<br>?? v59_rare_disease_authoritative_diagnostic_guide_final/<br>warning: unable to access 'C:\Users\admin/.config/git/ignore': Permission denied<br>warning: unable to access 'C:\Users\admin/.config/git/ignore': Permission denied | OK |  |
| git_rev_parse_HEAD | 4cb0dc995658e1a114eb49e24076c4dd421fed9c | OK |  |
| run_dir | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | OK |  |
| top_config | D:\RareDisease-traindata\configs\mainline_full_pipeline.yaml | OK |  |
| locked_config | D:\RareDisease-traindata\configs\mainline_full_pipeline_hybrid_tag_v5.locked.yaml | OK |  |
| runner | D:\RareDisease-traindata\tools\run_full_mainline_pipeline.py | OK |  |

## 4. Snapshot Inventory

Full table: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\snapshot_inventory.csv`

| type | path | exists | size_bytes | sha256 | notes |
| --- | --- | --- | --- | --- | --- |
| manifest | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\run_manifest.json | true | 11732 | a72b4eccf061bb566dcf3e4644b919ced1bc8538fd4e0dc86e494b2fb12b370c | run manifest |
| config | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml | true | 1835 | fcb17040eb4ffe1e74ccf9c37e9a6a854f454491ed0b8ecf0faffecbdb6b043d | stage1 pretrain config |
| config | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml | true | 2914 | d4ede580d7d30b32a34a788d37ee697ababe20170d57029d2615de9e2e0ef931 | stage2 finetune config |
| config | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage3_exact_eval_train.yaml | true | 2916 | 75f7946a622d4f5bf1f1921e41eef3e7a992cc8d102d23a1342d90b0ea686695 | stage3 eval train config |
| checkpoint | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage1_pretrain\checkpoints\best.pt | true | 93748043 | SKIPPED_HASH_LARGE_FILE | stage1 best checkpoint; SKIPPED_HASH_LARGE_FILE |
| checkpoint | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | true | 93749323 | SKIPPED_HASH_LARGE_FILE | stage2 best checkpoint; SKIPPED_HASH_LARGE_FILE |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv | true | 1141082 | a451eb064b97590658946a19d685c1df42dbc42ab8c9a7d25eed274f4b3feeaf | exact eval details |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_per_dataset.csv | true | 988 | 0519b33205bc0314ef66afe9d763dc16ffbb5ac0391f430eab21c670cab85bca | exact eval per dataset |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_summary.json | true | 19814 | cd1cb9c2e2efb58aa5b27e1f874255c8d8e719af1d951252cbd03c0cdb1c27a5 | exact eval summary |
| candidate | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_validation.csv | true | 30118438 | bc56a5b22c86014e74f222627e1ae7e34e06c731b93f82becba8ba96c1127f71 | validation top50 candidates |
| candidate | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_test.csv | true | 43605533 | 8a51ad3ed1179a7a385d9023507d5a23f1a5f569d4d1ed0ec93a5168883c5d3d | test top50 candidates |
| candidate_metadata | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_validation.metadata.json | true | 1748 | fca44e98c2fa48735cd85fb0c5b6412717f5f9ae8bb1cb97ddbd4a33b11998df | validation candidate metadata |
| candidate_metadata | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_test.metadata.json | true | 1736 | 006c2eb1b896618dda46f4b0e154375c5afbb2a30ebb06628f0fd2897c32a385 | test candidate metadata |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage5_ddd_rerank\rerank_fixed_test_metrics.csv | true | 1067 | 31dc0c9528dca1e24428dfdf5493670eedb26959e61542d5829d992ec46e3773 | DDD rerank fixed test metrics |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage5_ddd_rerank\ddd_val_selected_grid_weights.json | true | 1433 | 0c446ac425a857b5943e2f39fe658ff022c5c577c6cf78161c5878d03e821214 | DDD selected weights |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage6_mimic_similar_case\similar_case_fixed_test.csv | true | 568 | acb2808ce457179f4db9e240e6a54d89567d5915734fb0924e435d18cecdcc3b | mimic SimilarCase metrics |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage6_mimic_similar_case\similar_case_fixed_test_ranked_candidates.csv | true | 20405671 | a615656fff4e08f6212c5386c16a20a4456bdc44b3a1cb72b6ad4823ab36b0fc | mimic SimilarCase ranked candidates |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage6_mimic_similar_case\manifest.json | false |  | NOT_FOUND | mimic SimilarCase manifest |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_metrics.csv | true | 649 | 8c1b649248760b5e93eebec591396263c20b6c12ed33a96cf48b86e5af92e8a4 | final metrics |
| result | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_metrics_with_sources.csv | true | 3891 | 5441e5ca6d5323f50f873344ce127fcf5d24ad51948b431e9e75c85a1a79a8c4 | final metrics with sources |
| ... | ... | ... | ... | ... | ... |

## 5. Manifest Consistency Checks

Status summary: PASS=17

| check_id | check_name | expected | actual | status | severity | notes |
| --- | --- | --- | --- | --- | --- | --- |
| M001 | manifest output_dir equals audited run-dir | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | PASS | High |  |
| M002 | manifest finetune checkpoint exists | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | PASS | Critical |  |
| M003 | manifest finetune checkpoint is run-dir stage2 best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | PASS | High |  |
| M103 | manifest stage_configs.pretrain points inside audited run | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml | PASS | Medium |  |
| M104 | manifest stage_configs.finetune points inside audited run | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml | PASS | High |  |
| M105 | manifest stage_configs.eval_train points inside audited run | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage3_exact_eval_train.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage3_exact_eval_train.yaml | PASS | Medium |  |
| M020 | stage3 exact eval command uses stage2 best checkpoint | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | PASS | Critical |  |
| M021 | stage3 eval and candidate metadata use same data config | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | PASS | High |  |
| M03validation | validation candidate metadata checkpoint matches stage2 best | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | PASS | Critical | metadata key=validation_candidates_metadata |
| M04validation | validation candidate metadata output file exists | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_validation.csv | PASS | High |  |
| M03test | test candidate metadata checkpoint matches stage2 best | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | PASS | Critical | metadata key=test_candidates_metadata |
| M04test | test candidate metadata output file exists | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_test.csv | PASS | High |  |
| M05metrics | manifest final_outputs.metrics exists inside run-dir | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_metrics.csv | PASS | High |  |
| M05metrics_with_sources | manifest final_outputs.metrics_with_sources exists inside run-dir | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_metrics_with_sources.csv | PASS | High |  |
| M05case_ranks | manifest final_outputs.case_ranks exists inside run-dir | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv | PASS | High |  |
| M06DDD | DDD final source is expected postprocess inside run-dir | ddd_validation_selected_grid_rerank | {"module_applied": "ddd_validation_selected_grid_rerank", "source_result_path": "D:\\RareDisease-traindata\\outputs\\mainline_full_pipeline_hybrid_tag_v5\\stage5_ddd_rerank\\ddd_val_selected_grid_weights.json"} | PASS | High |  |
| M06mimic_test_recleaned_mondo_hpo_rows | mimic_test_recleaned_mondo_hpo_rows final source is expected postprocess inside run-dir | similar_case_fixed_test | {"module_applied": "similar_case_fixed_test", "source_result_path": "D:\\RareDisease-traindata\\outputs\\mainline_full_pipeline_hybrid_tag_v5\\stage6_mimic_similar_case\\similar_case_fixed_test.csv"} | PASS | High |  |

## 6. Top Config vs Snapshot

Status summary: FAIL=7, NOT_FOUND=6, PASS=8

| key_path | top_config_value | snapshot_value | status | severity | notes |
| --- | --- | --- | --- | --- | --- |
| paths.output_dir | D:\RareDisease-traindata\outputs\mainline_full_pipeline | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | FAIL | High |  |
| tag_encoder.enabled | False | True | FAIL | High |  |
| paths.data_eval_config | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | PASS | Info |  |
| resume.finetune_checkpoint | D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | FAIL | High |  |
| resume.skip_pretrain | True | NOT_FOUND | NOT_FOUND | High | snapshot manifest has no equivalent field; runner audit checks whether this is ignored |
| resume.skip_finetune | True | NOT_FOUND | NOT_FOUND | High | snapshot manifest has no equivalent field; runner audit checks whether this is ignored |
| pipeline.run_pretrain | True | True | PASS | Info | snapshot value inferred from run_manifest commands |
| pipeline.run_finetune | True | True | PASS | Info | snapshot value inferred from run_manifest commands |
| pipeline.run_exact_eval | True | True | PASS | Info | snapshot value inferred from run_manifest commands |
| pipeline.run_candidate_export | True | True | PASS | Info | snapshot value inferred from run_manifest commands |
| pipeline.run_ddd_rerank | True | True | PASS | Info | snapshot value inferred from run_manifest commands |
| pipeline.run_mimic_similar_case | True | True | PASS | Info | snapshot value inferred from run_manifest commands |
| pipeline.run_final_aggregation | True | True | PASS | Info | snapshot value inferred from run_manifest commands |
| data.random_seed | NOT_FOUND | 42 | NOT_FOUND | Low | top-level config does not expose training seed |
| model.hidden_dim | NOT_FOUND | 128 | NOT_FOUND | Low | top-level config does not expose model hidden_dim |
| model.encoder.use_tag_encoder | False | True | FAIL | High | top-level tag flag compared to effective stage2 model encoder |
| model.case_refiner.enabled | NOT_FOUND | True | NOT_FOUND | Low | top-level config does not expose case_refiner |
| dual_stream.enabled | NOT_FOUND | NOT_FOUND | NOT_FOUND | Low |  |
| stage2.train.init_checkpoint_path | D:\RareDisease-traindata\outputs\mainline_full_pipeline\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage1_pretrain\checkpoints\best.pt | FAIL | Medium | top resume checkpoint is not the same concept as stage2 pretrain init checkpoint |
| paths.pretrain_config | D:\RareDisease-traindata\configs\train_pretrain.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml | FAIL | Low | top config points to source template, snapshot points to generated effective config |
| paths.finetune_config | D:\RareDisease-traindata\configs\train_finetune_attn_idf_main.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml | FAIL | Low | top config points to source template, snapshot points to generated effective config |

## 7. Locked Config Verification

Status summary: PASS=13

| key_path | locked_config_value | snapshot_value | status | severity | notes |
| --- | --- | --- | --- | --- | --- |
| paths.output_dir | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | PASS | Info |  |
| tag_encoder.enabled | True | True | PASS | Info |  |
| model.encoder.use_tag_encoder | True | True | PASS | Info |  |
| resume.finetune_checkpoint | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt | PASS | Info |  |
| resume.pretrain_checkpoint | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage1_pretrain\checkpoints\best.pt | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage1_pretrain\checkpoints\best.pt | PASS | Info |  |
| paths.data_eval_config | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | D:\RareDisease-traindata\configs\data_llldataset_eval.yaml | PASS | Info |  |
| paths.pretrain_config | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml | PASS | Info |  |
| paths.finetune_config | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml | PASS | Info |  |
| paths.pretrain_config.sha256 | fcb17040eb4ffe1e74ccf9c37e9a6a854f454491ed0b8ecf0faffecbdb6b043d | fcb17040eb4ffe1e74ccf9c37e9a6a854f454491ed0b8ecf0faffecbdb6b043d | PASS | Info | hash comparison of effective stage1 config |
| paths.finetune_config.sha256 | d4ede580d7d30b32a34a788d37ee697ababe20170d57029d2615de9e2e0ef931 | d4ede580d7d30b32a34a788d37ee697ababe20170d57029d2615de9e2e0ef931 | PASS | Info | hash comparison of effective stage2 config |
| tag_encoder.pretrained_embed_path | D:\RareDisease-traindata\data\processed\biolord_hpo_embeds_v5.npy | D:\RareDisease-traindata\data\processed\biolord_hpo_embeds_v5.npy | PASS | Info |  |
| resume.skip_pretrain | ABSENT | ABSENT_EXPECTED | PASS | Info | locked config must not contain runner-validated unsupported control-flow keys |
| resume.skip_finetune | ABSENT | ABSENT_EXPECTED | PASS | Info | locked config must not contain runner-validated unsupported control-flow keys |

## 8. Suspicious / Ignored Config Keys

Status summary: CONSUMED_OR_MENTIONED=31, SUSPICIOUS=5

| key_path | value | status | severity | notes |
| --- | --- | --- | --- | --- |
| aggregation.ddd_source | ddd_rerank | SUSPICIOUS | Medium | control-flow-looking key is not directly mentioned in runner source; verify manually |
| aggregation.default_source | hgnn_exact_baseline | SUSPICIOUS | Medium | control-flow-looking key is not directly mentioned in runner source; verify manually |
| aggregation.mimic_source | similar_case | SUSPICIOUS | Medium | control-flow-looking key is not directly mentioned in runner source; verify manually |
| aggregation.no_test_side_tuning | True | SUSPICIOUS | Medium | control-flow-looking key is not directly mentioned in runner source; verify manually |
| paths.data_eval_config | configs/data_llldataset_eval.yaml | CONSUMED_OR_MENTIONED | Low |  |
| paths.finetune_config | configs/train_finetune_attn_idf_main.yaml | CONSUMED_OR_MENTIONED | Low |  |
| paths.output_dir | outputs/mainline_full_pipeline | CONSUMED_OR_MENTIONED | Low |  |
| paths.pretrain_config | configs/train_pretrain.yaml | CONSUMED_OR_MENTIONED | Low |  |
| pipeline.run_candidate_export | True | CONSUMED_OR_MENTIONED | Low |  |
| pipeline.run_ddd_rerank | True | CONSUMED_OR_MENTIONED | Low |  |
| pipeline.run_exact_eval | True | CONSUMED_OR_MENTIONED | Low |  |
| pipeline.run_final_aggregation | True | CONSUMED_OR_MENTIONED | Low |  |
| pipeline.run_finetune | True | CONSUMED_OR_MENTIONED | Low |  |
| pipeline.run_mimic_similar_case | True | CONSUMED_OR_MENTIONED | Low |  |
| pipeline.run_pretrain | True | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.ddd.enabled | True | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.ddd.fixed_test | True | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.ddd.module | ddd_validation_selected_grid_rerank | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.ddd.objective | ddd_top1 | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.ddd.select_on | validation | SUSPICIOUS | Medium | control-flow-looking key is not directly mentioned in runner source; verify manually |
| postprocess.mimic.dataset_aliases[0] | mimic_test | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.mimic.dataset_aliases[1] | mimic_test_recleaned | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.mimic.dataset_aliases[2] | mimic_test_recleaned_mondo_hpo_rows | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.mimic.enabled | True | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.mimic.fixed_test | True | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.mimic.module | similar_case_fixed_test | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.mimic.similarity_batch_size | 256 | CONSUMED_OR_MENTIONED | Low |  |
| postprocess.mimic.similarity_device | cuda | CONSUMED_OR_MENTIONED | Low |  |
| resume.finetune_checkpoint | outputs/mainline_full_pipeline/stage2_finetune/checkpoints/best.pt | CONSUMED_OR_MENTIONED | Low |  |
| resume.skip_finetune | True | CONSUMED_OR_MENTIONED | Medium |  |
| resume.skip_pretrain | True | CONSUMED_OR_MENTIONED | Medium |  |
| tag_encoder.batch_size | 64 | CONSUMED_OR_MENTIONED | Low |  |
| tag_encoder.enabled | False | CONSUMED_OR_MENTIONED | Low |  |
| tag_encoder.hpo_index_path | LLLdataset/DiseaseHy/processed/HPO_index_v5.xlsx | CONSUMED_OR_MENTIONED | Low |  |
| tag_encoder.model_name | models/BioLORD-2023 | CONSUMED_OR_MENTIONED | Low |  |
| tag_encoder.pretrained_embed_path | data/processed/biolord_hpo_embeds_v5.npy | CONSUMED_OR_MENTIONED | Low |  |

## 9. Runner Config Validation

Status summary: EXPECTED_STRICT_FAIL=2, PASS=4

| config_path | key_path | value | status | strict_would_fail | severity | notes |
| --- | --- | --- | --- | --- | --- | --- |
| D:\RareDisease-traindata\tools\run_full_mainline_pipeline.py | runner.validate_pipeline_config_keys | true | PASS | false | Info | runner exposes config-key validation function |
| D:\RareDisease-traindata\tools\run_full_mainline_pipeline.py | runner.--strict-config-keys | true | PASS | false | Info | runner exposes strict CLI guard |
| D:\RareDisease-traindata\configs\mainline_full_pipeline.yaml | resume.skip_pretrain | True | EXPECTED_STRICT_FAIL | true | High | unsupported control-flow key present; Use pipeline.run_pretrain or --mode instead. |
| D:\RareDisease-traindata\configs\mainline_full_pipeline.yaml | resume.skip_finetune | True | EXPECTED_STRICT_FAIL | true | High | unsupported control-flow key present; Use pipeline.run_finetune or --mode instead. |
| D:\RareDisease-traindata\configs\mainline_full_pipeline_hybrid_tag_v5.locked.yaml | resume.skip_pretrain | ABSENT | PASS | false | Info | key absent; strict validation passes for this key |
| D:\RareDisease-traindata\configs\mainline_full_pipeline_hybrid_tag_v5.locked.yaml | resume.skip_finetune | ABSENT | PASS | false | Info | key absent; strict validation passes for this key |

## 10. Candidate-Checkpoint Consistency

Status summary: PASS=2

| candidate_file | metadata_file | checkpoint_match | n_cases | n_rows | candidate_count_issue_cases | duplicate_candidate_cases | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_test.csv | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_test.metadata.json | true | 2978 | 148900 | 0 | 0 | PASS | rank_issue_cases=0; has_score=True;  |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_validation.csv | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_validation.metadata.json | true | 2146 | 107300 | 0 | 0 | PASS | rank_issue_cases=0; has_score=True;  |

## 11. Final Result Traceability

Status summary: PASS=8

| dataset | metric_source | source_path | source_exists | inside_run_dir | expected_source_type | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DDD | ddd_validation_selected_grid_rerank | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage5_ddd_rerank\ddd_val_selected_grid_weights.json | true | true | ddd_validation_selected_grid_rerank | PASS | case_rank_rows=761; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |
| HMS | hgnn_exact_baseline | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv | true | true | hgnn_exact_baseline | PASS | case_rank_rows=25; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |
| LIRICAL | hgnn_exact_baseline | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv | true | true | hgnn_exact_baseline | PASS | case_rank_rows=59; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |
| MME | hgnn_exact_baseline | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv | true | true | hgnn_exact_baseline | PASS | case_rank_rows=10; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |
| MyGene2 | hgnn_exact_baseline | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv | true | true | hgnn_exact_baseline | PASS | case_rank_rows=33; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |
| RAMEDIS | hgnn_exact_baseline | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv | true | true | hgnn_exact_baseline | PASS | case_rank_rows=217; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |
| mimic_test_recleaned_mondo_hpo_rows | similar_case_fixed_test | D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage6_mimic_similar_case\similar_case_fixed_test.csv | true | true | similar_case_fixed_test | PASS | case_rank_rows=1873; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |
| ALL | mixed | mixed | true | false | mixed | PASS | case_rank_rows=0; final_case_ranks=D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv |

## 12. Output Snapshot Comparison

| output_dir | has_manifest | has_stage2_checkpoint | has_final_metrics | ALL_top1 | DDD_top1 | mimic_top1 | mimic_top5 | mimic_rank_le_50 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline | true | true | true | 0.3106111484217596 | 0.37582128777923784 | 0.20928990923651897 | 0.402562733582488 | 0.6556326748531767 |  |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5 | true | true | true | 0.31296171927468097 | 0.3639947437582129 | 0.21783235451147892 | 0.404164442071543 | 0.6689802455953017 |  |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5_nodegate_dropout | true | true | true | 0.2988582941571524 | 0.3745072273324573 | 0.19434063000533902 | 0.3678590496529632 | 0.6332087560064068 |  |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5_scalar_rollback | true | true | true | 0.3035594358629953 | 0.3718791064388962 | 0.19914575547250402 | 0.3582487987186332 | 0.6300053390282968 |  |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline_tag | true | true | true | 0.24210879785090664 | 0.3311432325886991 | 0.1382808328884143 | 0.2375867592098238 | 0.5691404164442072 |  |
| D:\RareDisease-traindata\outputs\mainline_full_pipeline_tag_v5 | true | true | true | 0.2669576897246474 | 0.3639947437582129 | 0.16657768286171917 | 0.27282434596903365 | 0.584089695675387 |  |

## 13. Confirmed Problems

| severity | problem | evidence | recommended_fix |
| --- | --- | --- | --- |
| High | Current top-level config does not reproduce audited snapshot | paths.output_dir, tag_encoder.enabled, resume.finetune_checkpoint, model.encoder.use_tag_encoder | Create a locked config/manifest for the audited snapshot and do not reuse mutable top config as proof of reproduction. |
| Medium | Multiple mainline output snapshots contain final metrics | 6 mainline_full_pipeline* dirs with final metrics | Reference results by frozen manifest path and output dir in every report. |

## 14. Risks / Warnings

- The repository worktree is dirty; treat existing unrelated changes as separate user work.
- Several `mainline_full_pipeline*` output directories contain metrics; reports must cite the exact output directory and manifest.
- Config-template paths and generated stage config paths are intentionally different; this is acceptable only when documented by a frozen manifest.

## 15. Recommended Fixes

1. Treat `configs/mainline_full_pipeline_hybrid_tag_v5.locked.yaml` as the frozen config for this audited snapshot.
2. Keep `--strict-config-keys` enabled for formal config checks before starting any expensive run.
3. Require every final report to cite `run_manifest.json`, candidate metadata, final metrics source table, locked config, and frozen audit manifest.
4. Keep future audit outputs outside `outputs/mainline_full_pipeline*` to avoid overwriting experiment artifacts.

## 16. Commands Run

- `D:\python\python.exe -m compileall tools/audit_snapshot_consistency.py tools/run_full_mainline_pipeline.py`
- `D:\python\python.exe -c "from tools.run_full_mainline_pipeline import validate_pipeline_config_keys; import yaml; cfg=yaml.safe_load(open('configs/mainline_full_pipeline.yaml', encoding='utf-8')); validate_pipeline_config_keys(cfg, strict=False)"`
- `D:\python\python.exe -c "from tools.run_full_mainline_pipeline import validate_pipeline_config_keys; import yaml; cfg=yaml.safe_load(open('configs/mainline_full_pipeline_hybrid_tag_v5.locked.yaml', encoding='utf-8')); validate_pipeline_config_keys(cfg, strict=True)"`
- `D:\python\python.exe tools/audit_snapshot_consistency.py --run-dir outputs/mainline_full_pipeline_hybrid_tag_v5 --top-config configs/mainline_full_pipeline.yaml --locked-config configs/mainline_full_pipeline_hybrid_tag_v5.locked.yaml --runner tools/run_full_mainline_pipeline.py --out-report-dir reports/snapshot_consistency_lock_audit --out-output-dir outputs/snapshot_consistency_lock_audit`

## 17. Generated Files

- Markdown report: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\snapshot_consistency_audit.md`
- frozen manifest: `D:\RareDisease-traindata\outputs\snapshot_consistency_lock_audit\frozen_snapshot_manifest.json`
- repository_state: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\repository_state.csv`
- snapshot_inventory: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\snapshot_inventory.csv`
- manifest_consistency_checks: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\manifest_consistency_checks.csv`
- top_config_vs_snapshot: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\top_config_vs_snapshot.csv`
- locked_config_vs_snapshot: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\locked_config_vs_snapshot.csv`
- suspicious_config_keys: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\suspicious_config_keys.csv`
- runner_config_validation: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\runner_config_validation.csv`
- candidate_consistency: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\candidate_consistency.csv`
- final_result_traceability: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\final_result_traceability.csv`
- output_snapshot_comparison: `D:\RareDisease-traindata\reports\snapshot_consistency_lock_audit\tables\output_snapshot_comparison.csv`
