# RareDisease HGNN Current Framework Accuracy Audit

Generated at: 2026-04-29

Scope: read-only audit of the current RareDisease HGNN repository state. Primary audited run is `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5`, because the current audit manifest and tables point to this run. The repository also contains newer and older mainline output directories; those are treated as separate result snapshots.

## 1. Executive Summary

### Bottom Line

The low accuracy on `mimic_test_recleaned_mondo_hpo_rows` is not explained by missing HPO vocabulary, missing disease index coverage, candidate CSV corruption, duplicate candidates, or a simple sorting-direction bug. The strongest confirmed bottlenecks are:

1. `mimic_test` has much weaker case-to-gold disease HPO overlap than DDD. In the current audit, 36.68% of mimic cases have zero exact overlap with the gold disease v59 hyperedge, versus 5.65% for DDD.
2. Candidate recall is lower for mimic. Exact HGNN gold@50 is 0.6295 for mimic versus 0.7293 for DDD in the audited run. That means about 694 mimic cases are outside HGNN top50 before post-processing.
3. Many mimic failures are not sparse in raw HPO count, but their HPOs are adult clinical, generic, or comorbidity-like phenotypes such as hypertension, nausea, fever, fatigue, dyspnea, anemia, abdominal pain, and diarrhea. These terms map to the HPO index, but are weak diagnostic discriminators for rare disease ranking.
4. MONDO hygiene is a real issue. Current mimic test has 51 cases with obsolete MONDO gold labels and HGNN top1 predictions include obsolete MONDO terms in 63/1873 mimic cases. This is not the sole cause, but it adds label granularity and ontology-version noise.
5. Exact evaluation compresses multi-label cases to the first label. Current/recent audits show 227 mimic cases are multi-label. Any-label scoring raises final mimic top5 from 0.4042 to 0.4335 in the SimilarCase result, so exact metrics are modestly underestimated, but this does not explain most of the gap.
6. DDD and mimic have different bottleneck shapes. DDD has a substantial top50-internal ranking problem that evidence rerank improves. Mimic has both top50 recall and evidence quality problems. SimilarCase can recover some rank>50 cases into top50, but in the audited run it recovered 116 cases to top50 and 0 directly to top5.

### Result Trust Status

The audited run itself is internally traceable: exact eval, top50 candidates, DDD rerank, mimic SimilarCase, and final mixed metrics all point to `outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt`.

However, the repository state is not clean and the current top-level config is not the same snapshot as the audited `hybrid_tag_v5` run. `configs\mainline_full_pipeline.yaml` now points to `outputs/mainline_full_pipeline`, has `tag_encoder.enabled: false`, and contains `resume.skip_pretrain` / `resume.skip_finetune` keys that are not consumed by `tools\run_full_mainline_pipeline.py`. Therefore, any future formal run must freeze config, checkpoint, candidate files, and result paths before being considered paper-grade.

## 2. Current Pipeline Reconstruction

### Repository Inventory

Full inventory table: `D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables\repository_inventory.csv`

| Type | Path | Purpose | Exists | Notes |
|---|---|---|---|---|
| config | `D:\RareDisease-traindata\configs\mainline_full_pipeline.yaml` | top-level staged pipeline config | yes | Current file differs from audited `hybrid_tag_v5` manifest. |
| config | `D:\RareDisease-traindata\configs\train_pretrain.yaml` | base pretrain config | yes | Uses PubCaseFinder disease hyperedge + unseen profiles. |
| config | `D:\RareDisease-traindata\configs\train_finetune_attn_idf_main.yaml` | base finetune config | yes | Real + synthetic finetune, hard negative, case IDF weighting. |
| config | `D:\RareDisease-traindata\configs\data_llldataset_eval.yaml` | test evaluation data config | yes | Lists DDD, HMS, LIRICAL, mimic recleaned, MME, MyGene2, RAMEDIS. |
| config | `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml` | audited effective finetune config | yes | TAG enabled, save_dir captured under audited run. |
| script | `D:\RareDisease-traindata\tools\run_full_mainline_pipeline.py` | staged pipeline runner | yes | Builds stage configs, runs training/eval/candidates/rerank/final aggregation. |
| script | `D:\RareDisease-traindata\tools\export_top50_candidates.py` | candidate export | yes | Exports top50 with evidence features and metadata. |
| script | `D:\RareDisease-traindata\tools\run_top50_evidence_rerank.py` | DDD/evidence top50 rerank | yes | Validation-selected fixed-test protocol. |
| script | `D:\RareDisease-traindata\tools\run_mimic_similar_case_aug.py` | mimic SimilarCase post-processing | yes | Validation-selected fixed-test protocol, can add similar-case labels. |
| script | `D:\RareDisease-traindata\tools\run_eval_with_pretrain_alignment.py` | requested audit target | NOT_FOUND | No such file in current repo. |
| script | `D:\RareDisease-traindata\tools\audit_pretrain_vs_real_dataset_gap.py` | requested audit target | NOT_FOUND | Related existing file: `tools\analysis\deeprare_target_gap.py`. |
| src | `D:\RareDisease-traindata\src\data\dataset.py` | case table loader | yes | Namespaces case IDs, maps MONDO labels to disease index. |
| src | `D:\RareDisease-traindata\src\data\build_hypergraph.py` | HPO/case incidence builder | yes | Drops cases with no valid HPO; applies case-side IDF weights. |
| src | `D:\RareDisease-traindata\src\models\hgnn_encoder.py` | base HGNN encoder | yes | Sparse two-layer HGNN propagation. |
| src | `D:\RareDisease-traindata\src\hgnn_encoder_tag.py` | TAG encoder | yes | Frozen BioLORD HPO embeddings plus learnable gate. |
| src | `D:\RareDisease-traindata\src\models\model_pipeline.py` | encoder/readout/scorer orchestration | yes | Supports HGNN/TAG, case refiner, optional dual stream. |
| src | `D:\RareDisease-traindata\src\training\trainer.py` | training loop | yes | Full-pool CE, hard negative, source-balanced sampler. |
| src | `D:\RareDisease-traindata\src\evaluation\evaluator.py` | exact evaluation | yes | Full-rank ranking over disease index. |
| checkpoint | `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage2_finetune\checkpoints\best.pt` | audited finetune checkpoint | yes | Checkpoint epoch 10 in metadata. |
| candidate | `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_test.csv` | audited test top50 candidates | yes | 2978 cases, 148900 rows. |
| result | `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv` | exact full-rank eval details | yes | 2978/2978 evaluable, 0 skipped. |
| result | `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_metrics_with_sources.csv` | final mixed metrics with source paths | yes | DDD uses rerank, mimic uses SimilarCase, others exact. |
| dataset | `D:\RareDisease-traindata\LLLdataset\DiseaseHy\rare_disease_hgnn_clean_package_v59\v59DiseaseHy.npz` | v59 disease-HPO incidence matrix | yes | Shape 19566 HPO x 16443 diseases. |
| dataset | `D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx` | disease index | yes | MONDO disease space used by model. |
| dataset | `D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\HPO_index_v4.xlsx` | HPO index | yes | HPO vocabulary used by model. |

### Pipeline Table

| Stage | Input | Output | Key Script | Key Config | Possible Risk |
|---|---|---|---|---|---|
| raw data | `data\raw_data`, `raw_data`, `LLLdataset\dataset`, `LLLdataset\DiseaseHy` | source case tables and ontology resources | build scripts under `src\data` and `LLLdataset` | multiple local files | Mixed ontology versions and obsolete MONDO/HPO can persist. |
| cleaned/eval cases | `LLLdataset\dataset\processed\train`, `LLLdataset\dataset\processed\test` | train/test long tables with `case_id`, `mondo_label`, `hpo_id` | `src\data\dataset.py` | `configs\data_llldataset_eval.yaml`, stage train configs | Multi-label cases are compressed to first label during exact eval. |
| HPO/MONDO index | `HPO_index_v4.xlsx`, `Disease_index_v4.xlsx`, v59 npz | `hpo_to_idx`, `disease_to_idx`, `H_disease` | `src\data\build_hypergraph.py`, `src\evaluation\evaluator.py` | stage configs | Disease index includes obsolete MONDO terms. |
| pretrain | PubCaseFinder disease hyperedge + unseen profiles | stage1 checkpoint | `python -m src.training.trainer` | `outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml` | Synthetic/knowledge profiles differ from real mimic HPO distributions. |
| finetune | real train sets + mimic_rag + FakeDisease, init from pretrain | stage2 checkpoint | `python -m src.training.trainer` | `outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml` | Source-balanced sampler and hard negatives may not match mimic_test target distribution. |
| exact eval | test files + stage2 checkpoint | full-rank exact details and per-dataset metrics | `python -m src.evaluation.evaluator` | `configs\data_llldataset_eval.yaml`, stage3 train config | Single-label exact metric underestimates multi-label mimic cases. |
| top50 candidate export | same test/validation sources + same checkpoint | top50 candidates with evidence features | `tools\export_top50_candidates.py` | data config + stage2 config | Candidate recall@50 caps top50 rerank. |
| DDD rerank | validation/test top50 candidates | DDD fixed-test rerank metrics | `tools\run_top50_evidence_rerank.py` | selected weights JSON | Only reranks top50. Cannot recover gold absent from top50. |
| mimic SimilarCase | validation/test candidates + train/val/test case tables | mimic fixed-test ranked candidates and metrics | `tools\run_mimic_similar_case_aug.py` | validation-selected topk/weight/score type | Can expand beyond HGNN top50 via train labels, but current top5 gain remains limited. |
| final aggregation | exact details, DDD rerank, mimic SimilarCase | final mixed metrics/case ranks | `tools\run_full_mainline_pipeline.py` | top-level and stage configs | Multiple output dirs make "current" ambiguous unless manifest is used. |

### Text Flow

```text
raw data / ontology files
 -> cleaned case tables
 -> HPO_index_v4 + Disease_index_v4 + v59DiseaseHy.npz
 -> pretrain HGNN/TAG checkpoint
 -> finetune HGNN/TAG checkpoint
 -> exact full-rank evaluation
 -> validation/test top50 candidates with overlap/evidence features
 -> DDD validation-selected top50 evidence rerank
 -> mimic validation-selected SimilarCase augmentation
 -> final mixed metrics
```

## 3. Current Metrics Summary

Primary metric table: `D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables\current_metrics_summary.csv`

Primary audited exact eval: `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv`

Primary audited final mixed metrics: `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv`

| Dataset | Split/Result | n | top1 | top3 | top5 | top10 | top30 | top50/rank_le_50 | median_rank | result_path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DDD | exact baseline | 761 | 0.3035 | 0.4297 | 0.4678 | 0.5466 | 0.6715 | 0.7293 | 7 | `stage3_exact_eval\exact_details.csv` |
| mimic_test_recleaned_mondo_hpo_rows | exact baseline | 1873 | 0.1991 | 0.2974 | 0.3486 | 0.4180 | 0.5483 | 0.6295 | 20 | `stage3_exact_eval\exact_details.csv` |
| ALL | exact baseline | 2978 | 0.2858 | 0.3889 | 0.4355 | 0.5040 | 0.6216 | 0.6901 | 10 | `stage3_exact_eval\exact_details.csv` |
| DDD | final mixed | 761 | 0.3640 | 0.4783 | 0.5177 | 0.5979 | 0.6965 | 0.7293 | 4 | `mainline_final_case_ranks.csv` |
| mimic_test_recleaned_mondo_hpo_rows | final mixed | 1873 | 0.2178 | 0.3492 | 0.4042 | 0.4613 | 0.5830 | 0.6690 | 15 | `mainline_final_case_ranks.csv` |
| ALL | final mixed | 2978 | 0.3130 | 0.4338 | 0.4832 | 0.5443 | 0.6498 | 0.7149 | 6 | `mainline_final_case_ranks.csv` |

Other test datasets in exact baseline:

| Dataset | n | top1 | top3 | top5 | rank_le_50 | median_rank |
|---|---:|---:|---:|---:|---:|---:|
| HMS | 25 | 0.3600 | 0.4400 | 0.5600 | 0.7600 | 4 |
| LIRICAL | 59 | 0.5254 | 0.5763 | 0.6271 | 0.8475 | 1 |
| MME | 10 | 0.8000 | 0.9000 | 0.9000 | 0.9000 | 1 |
| MyGene2 | 33 | 0.8788 | 0.9091 | 0.9091 | 0.9394 | 1 |
| RAMEDIS | 217 | 0.7834 | 0.8756 | 0.9124 | 0.9770 | 1 |

### Multiple Output Snapshot Warning

The repository has multiple mainline output directories with different metrics:

| Output Dir | ALL top1 | DDD top1 | mimic top1 | mimic top5 | mimic rank_le_50 |
|---|---:|---:|---:|---:|---:|
| `mainline_full_pipeline_hybrid_tag_v5` | 0.3130 | 0.3640 | 0.2178 | 0.4042 | 0.6690 |
| `mainline_full_pipeline_hybrid_tag_v5_scalar_rollback` | 0.3036 | 0.3719 | 0.1991 | 0.3582 | 0.6300 |
| `mainline_full_pipeline_hybrid_tag_v5_nodegate_dropout` | 0.2989 | 0.3745 | 0.1943 | 0.3679 | 0.6332 |
| `mainline_full_pipeline` | 0.3106 | 0.3758 | 0.2093 | 0.4026 | 0.6556 |
| `mainline_full_pipeline_tag_v5` | 0.2670 | 0.3640 | 0.1666 | 0.2728 | 0.5841 |

Conclusion: current claims must identify the exact output directory and manifest. Do not merge numbers across these snapshots.

## 4. Candidate Recall Upper Bound Analysis

Candidate summary table: `D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables\candidate_recall_summary.csv`

| Dataset | Split | n | gold@1 | gold@3 | gold@5 | gold@10 | gold@30 | gold@50 | gold_not_top50 | duplicate cases | candidate<50 cases | candidate_file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DDD | validation | 164 | 0.3171 | 0.4390 | 0.5122 | 0.5793 | 0.6341 | 0.6890 | 51 | 0 | 0 | `top50_candidates_validation.csv` |
| mimic_rag_0425 | validation | 729 | 0.2236 | 0.3292 | 0.3813 | 0.4540 | 0.6008 | 0.6708 | 240 | 0 | 0 | `top50_candidates_validation.csv` |
| ALL | validation | 2146 | 0.5280 | 0.6230 | 0.6645 | 0.7134 | 0.7870 | 0.8234 | 379 | 0 | 0 | `top50_candidates_validation.csv` |
| DDD | test | 761 | 0.3035 | 0.4297 | 0.4678 | 0.5466 | 0.6715 | 0.7293 | 206 | 0 | 0 | `top50_candidates_test.csv` |
| mimic_test_recleaned_mondo_hpo_rows | test | 1873 | 0.1991 | 0.2974 | 0.3486 | 0.4180 | 0.5483 | 0.6295 | 694 | 0 | 0 | `top50_candidates_test.csv` |
| ALL | test | 2978 | 0.2858 | 0.3889 | 0.4355 | 0.5040 | 0.6216 | 0.6901 | 923 | 0 | 0 | `top50_candidates_test.csv` |

### Interpretation

DDD:

- 72.93% of DDD gold labels enter HGNN top50.
- Exact top1 is 30.35%, final DDD rerank top1 is 36.40%.
- Because 206/761 cases remain outside top50, top50 rerank alone cannot exceed 72.93% recall, but DDD still has large top50-internal ranking headroom.

mimic_test:

- Only 62.95% of mimic gold labels enter HGNN top50.
- 694/1873 cases are outside HGNN top50.
- Final SimilarCase raises rank_le_50 to 66.90%, but top1 only reaches 21.78%.
- Therefore mimic is not mainly a simple top50 sorting problem. It needs better candidate recall, better evidence alignment, and better disease/label granularity handling.

Excluded as current causes:

- Duplicate MONDO in current candidate files: 0 cases.
- Candidate count below 50: 0 cases.
- Gold missing from disease index in current test: 0 cases.

## 5. Dataset Quality Audit

Dataset quality table: `D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables\dataset_quality_summary.csv`

Case-level output: `D:\RareDisease-traindata\outputs\current_framework_accuracy_audit\dataset_quality_case_level.csv`

| Dataset | n_cases | unique_gold | hpo_mean | hpo_median | hpo<=2 | hpo<=5 | hpo_mapped_rate | gold_in_index_rate | gold_in_v59_rate | Main Issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DDD | 761 | 741 | 17.59 | 11 | 0.1196 | 0.2996 | 1.0000 | 1.0000 | 1.0000 | top50 recall + top50-internal ranking |
| HMS | 25 | 19 | 21.20 | 16 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | very small n |
| LIRICAL | 59 | 34 | 17.41 | 13 | 0.0000 | 0.1356 | 1.0000 | 1.0000 | 1.0000 | small n, ranking tail |
| MME | 10 | 4 | 11.50 | 10 | 0.0000 | 0.1000 | 1.0000 | 1.0000 | 1.0000 | very small n |
| MyGene2 | 33 | 13 | 9.00 | 7 | 0.0606 | 0.4848 | 1.0000 | 1.0000 | 1.0000 | sparse cases but high current accuracy |
| RAMEDIS | 217 | 48 | 9.66 | 8 | 0.0000 | 0.2811 | 1.0000 | 1.0000 | 1.0000 | high current accuracy |
| mimic_test_recleaned_mondo_hpo_rows | 1873 | 323 | 10.06 | 9 | 0.0256 | 0.1725 | 1.0000 | 1.0000 | 0.9995 | weak gold HPO overlap, generic clinical HPO, obsolete MONDO |

### HPO Coverage

All current test datasets have HPO mapped rate 1.0000 against the model HPO index. This excludes "massive unmapped HPO" as the primary current cause.

Obsolete HPO mentions exist but are not dominant in mimic:

- DDD obsolete HPO mentions: 72.
- mimic obsolete HPO mentions: 6.

### MONDO Coverage

Current test gold disease coverage:

- DDD: gold in disease index 100%, gold in v59 100%.
- mimic: gold in disease index 100%, gold in v59 99.9466%. One mimic case has a gold label in disease index but no v59 hyperedge: `case_1293`, `MONDO:0019297`, gold disease HPO count 0.

Current obsolete MONDO burden:

- DDD: 2 obsolete gold cases.
- mimic: 51 obsolete gold cases across 17 unique obsolete labels.
- mimic top1 predictions are obsolete in 63/1873 cases; 57 of those are wrong top1 predictions.

Top obsolete mimic gold labels include:

| MONDO | Name | Cases |
|---|---|---:|
| MONDO:0016343 | obsolete unclassified cardiomyopathy | 10 |
| MONDO:0016428 | obsolete multiple sclerosis variant | 10 |
| MONDO:0020078 | obsolete acute myeloid leukemia with recurrent genetic anomaly | 5 |
| MONDO:0800029 | obsolete interstitial lung disease 2 | 4 |
| MONDO:0017131 | obsolete hereditary cardiac anomaly | 3 |

### Top HPO Terms

Top HPO table: `D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables\top_hpo_by_dataset.csv`

DDD top terms include inheritance and neurodevelopmental terms:

| HPO | Name | Mentions |
|---|---|---:|
| HP:0000007 | Autosomal recessive inheritance | 316 |
| HP:0001249 | Intellectual disability | 200 |
| HP:0000006 | Autosomal dominant inheritance | 169 |
| HP:0001263 | Global developmental delay | 150 |
| HP:0001250 | Seizure | 141 |

mimic top terms are adult clinical and often non-specific:

| HPO | Name | Mentions |
|---|---|---:|
| HP:0000822 | Hypertension | 563 |
| HP:0002018 | Nausea | 322 |
| HP:0001945 | Fever | 320 |
| HP:0012378 | Fatigue | 320 |
| HP:0002094 | Dyspnea | 308 |
| HP:0001903 | Anemia | 285 |
| HP:0002027 | Abdominal pain | 284 |
| HP:0002014 | Diarrhea | 275 |

Interpretation: mimic does not fail because HPO IDs are absent. It fails because many mapped HPOs are low-discriminative clinical symptoms or comorbid findings relative to v59 rare disease hyperedges.

## 6. DDD vs mimic_test Gap Analysis

Gap table: `D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables\ddd_vs_mimic_gap.csv`

| Comparison Item | DDD | mimic_test | Difference | Possible Impact |
|---|---:|---:|---:|---|
| n_cases | 761 | 1873 | +1112 | mimic estimate is more stable but harder mix |
| hpo_mean | 17.59 | 10.06 | -7.54 | lower evidence density |
| hpo_le_5_rate | 0.2996 | 0.1725 | -0.1272 | mimic is not simply more low-count; it is lower-overlap |
| gold_case_overlap_zero_rate | 0.0565 | 0.3668 | +0.3103 | major candidate recall risk |
| gold_label_in_train_case_rate | 0.6610 | 0.9450 | +0.2840 | mimic labels often seen, but phenotype expression differs |
| unique_gold_in_train_rate | 0.6545 | 0.8854 | +0.2309 | disease coverage alone does not solve mimic |
| gold@50 | 0.7293 | 0.6295 | -0.0998 | lower top50 upper bound |
| top1_exact | 0.3035 | 0.1991 | -0.1044 | baseline ranking gap |
| median_rank_exact | 7 | 20 | +13 | heavier rank tail |

### Bucket Evidence

Case-level bucket audit from current outputs:

| Dataset | Bucket | n | gold@5 | gold@50 | median rank | mean gold overlap | mean Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|
| DDD | hpo<=5 | 228 | 0.320 | 0.649 | 16 | 1.56 | 0.120 |
| DDD | hpo>10 | 399 | 0.544 | 0.767 | 4 | 13.29 | 0.294 |
| DDD | gold_overlap_zero | 43 | 0.023 | 0.349 | >50 | 0.00 | 0.000 |
| DDD | gold_overlap>0 | 718 | 0.494 | 0.752 | 6 | 8.66 | 0.251 |
| mimic | hpo<=5 | 323 | 0.251 | 0.533 | 41 | 0.57 | 0.040 |
| mimic | hpo>10 | 782 | 0.368 | 0.669 | 18 | 1.56 | 0.047 |
| mimic | gold_overlap_zero | 687 | 0.189 | 0.491 | >50 | 0.00 | 0.000 |
| mimic | gold_overlap>0 | 1186 | 0.441 | 0.710 | 9 | 1.89 | 0.075 |

Why mimic is lower than DDD:

- DDD has more case HPOs on average and much stronger overlap with gold disease hyperedges.
- mimic has many HPOs that are valid but weakly disease-specific.
- mimic has more obsolete MONDO labels and obsolete top1 predictions.
- mimic top50 recall is about 10 percentage points lower than DDD in the current audited run.
- mimic is partly multi-label, but any-label scoring only recovers a few points, not the whole gap.

## 7. Failure Case Study

Failure case table: `D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables\failure_cases_mimic.csv`

Selected mimic cases:

| case_id | gold_mondo | gold_name | hpo_count | gold_rank | top1_mondo | top1_name | failure_type | Possible Cause |
|---|---|---|---:|---:|---|---|---|---|
| case_2 | MONDO:0009692 | primary myelofibrosis | 12 | 60 | MONDO:0018646 | sclerosing cholangitis | GOLD_NOT_IN_TOP50 | gold outside HGNN top50 |
| case_5 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 12 | 134 | MONDO:0016264 | autoimmune hepatitis | GOLD_NOT_IN_TOP50 | gold outside HGNN top50 |
| case_8 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 11 | 119 | MONDO:0015128 | primary adrenal insufficiency | GOLD_NOT_IN_TOP50 | related but wrong label/granularity |
| case_9 | MONDO:0019801 | acute adrenal insufficiency | 12 | 160 | MONDO:0018874 | acute myeloid leukemia | GOLD_NOT_IN_TOP50 | weak/ambiguous phenotype evidence |
| case_10 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 15 | 139 | MONDO:0015264 | cryptogenic organizing pneumonia | GOLD_NOT_IN_TOP50 | gold outside HGNN top50 |
| case_12 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 12 | 116 | MONDO:0011382 | sickle cell disease | GOLD_NOT_IN_TOP50 | gold outside HGNN top50 |
| case_16 | MONDO:0019801 | acute adrenal insufficiency | 14 | 101 | MONDO:0015128 | primary adrenal insufficiency | GOLD_NOT_IN_TOP50 | similar disease but exact miss |
| case_21 | MONDO:0019801 | acute adrenal insufficiency | 14 | 67 | MONDO:0017215 | calciphylaxis | GOLD_NOT_IN_TOP50 | gold outside HGNN top50 |
| case_1 | MONDO:0009692 | primary myelofibrosis | 13 | 15 | MONDO:0016264 | autoimmune hepatitis | RERANK_SORTING_FAILURE | recallable but ranked below top5 |
| case_11 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 14 | 48 | MONDO:0015128 | primary adrenal insufficiency | RERANK_SORTING_FAILURE | recallable but low rank |
| case_29 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 12 | 25 | MONDO:0015127 | pituitary deficiency | RERANK_SORTING_FAILURE | zero overlap with gold hyperedge |
| case_30 | MONDO:0015128 | primary adrenal insufficiency | 14 | 12 | MONDO:0015925 | interstitial lung disease | RERANK_SORTING_FAILURE | recallable but weak evidence |
| case_35 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 10 | 42 | MONDO:0007915 | systemic lupus erythematosus | RERANK_SORTING_FAILURE | recallable but low rank |
| case_37 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 11 | 15 | MONDO:0015128 | primary adrenal insufficiency | RERANK_SORTING_FAILURE | exact label granularity issue |
| case_3 | MONDO:0009692 | primary myelofibrosis | 10 | 4 | MONDO:0005789 | hepatitis D virus infection | SIMILAR_DISEASE_CONFUSION | near top but not top1 |
| case_6 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 5 | 3 | MONDO:0015127 | pituitary deficiency | SIMILAR_DISEASE_CONFUSION | low HPO count and zero overlap |
| case_7 | MONDO:0019801 | acute adrenal insufficiency | 15 | 5 | MONDO:0015128 | primary adrenal insufficiency | SIMILAR_DISEASE_CONFUSION | related adrenal disease |
| case_17 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 13 | 2 | MONDO:0017215 | calciphylaxis | SIMILAR_DISEASE_CONFUSION | near miss |
| case_18 | MONDO:0100480 | autoimmune primary adrenal insufficiency | 10 | 2 | MONDO:0015128 | primary adrenal insufficiency | SIMILAR_DISEASE_CONFUSION | exact label granularity issue |
| case_20 | MONDO:0015128 | primary adrenal insufficiency | 13 | 5 | MONDO:0011382 | sickle cell disease | SIMILAR_DISEASE_CONFUSION | zero overlap with gold hyperedge |

Failure type count in selected failure table:

| Failure Type | Count |
|---|---:|
| GOLD_NOT_IN_TOP50 | 10 |
| RERANK_SORTING_FAILURE | 7 |
| SIMILAR_DISEASE_CONFUSION | 6 |
| HPO_MAPPING_LOSS | 5 |
| LOW_HPO_COUNT | 2 |

## 8. Code-Level Risk Audit

| Risk Point | File | Lines/Function | Evidence | Severity | Confirmed |
|---|---|---|---|---|---|
| Current top-level config does not match audited result snapshot | `configs\mainline_full_pipeline.yaml` and `outputs\mainline_full_pipeline_hybrid_tag_v5\run_manifest.json` | config lines 11, 16, 48-51; manifest output_dir/tag_encoder | Current config says `output_dir: outputs/mainline_full_pipeline` and `tag_encoder.enabled: false`; manifest for audited run says output_dir `hybrid_tag_v5` and TAG enabled. | High | yes |
| `resume.skip_pretrain` and `resume.skip_finetune` appear in config but are not consumed by runner | `tools\run_full_mainline_pipeline.py` | no matches for `skip_pretrain` / `skip_finetune`; runner uses `--mode` and `pipeline.run_*` | Config keys can mislead future reruns. | Medium | yes |
| Loader drops labels outside disease index | `src\data\dataset.py` | `load_case_files`, around lines 174-186 | `gold_disease_idx = map(disease2idx)`, missing labels are warned and removed. Current test has 0 missing, but this can silently change training data. | Medium | behavior confirmed, current impact excluded |
| Batch hypergraph drops cases with no valid HPO | `src\data\build_hypergraph.py` | `build_case_incidence`, around lines 246-253 and 296-318 | Cases with no valid HPO are skipped. Current exact eval reports 0 skipped, but future data can be affected. | Medium | behavior confirmed, current impact excluded |
| Exact evaluation compresses multi-label cases to first label | `src\evaluation\evaluator.py` | `load_test_cases`, around lines 402-407 | `mondo_label` is `group_df[label_col].iloc[0]`; HPO IDs are aggregated. This underestimates multi-label mimic metrics. | High | yes |
| Exact evaluation sorting direction appears correct | `src\evaluation\evaluator.py` | lines 763-781 | Uses `torch.topk(scores)` and `torch.argsort(..., descending=True)`. No evidence of score-direction reversal. | Low | no issue found |
| Eval checkpoint/config mismatch is guarded | `src\evaluation\evaluator.py` | lines 238-257 | Loads state dict with `strict=False` but raises if any missing/unexpected keys remain. | Low | no mismatch found for audited run |
| Init checkpoint loading allows readout/case_refiner/dual-stream misses | `src\training\trainer.py` | lines 105-160 | Missing keys outside allowed prefixes raise; allowed prefixes warn and continue. Useful for upgrades but should be recorded. | Medium | behavior confirmed |
| Candidate export uses same score direction and writes metadata | `tools\export_top50_candidates.py` | lines 301-340, metadata lines 374-400 | Uses `torch.topk`; writes checkpoint path, epoch, data config, train config. | Low | no issue found |
| Pipeline checks candidate checkpoint consistency | `tools\run_full_mainline_pipeline.py` | `assert_same_checkpoint`, lines 826-835 | Candidate metadata checkpoint must equal finetune checkpoint. Audited manifest confirms same path. | Low | no issue found |
| DDD rerank is top50-capped | `tools\run_top50_evidence_rerank.py` | `ranks_from_scores`, lines 190-196 | Gold absent from top50 is assigned rank `top_k + 1`; cannot recover absent gold. | High | yes |
| mimic SimilarCase is not strictly top50-capped | `tools\run_mimic_similar_case_aug.py` | `similar_source`, `combine`, lines 352-424 | Similar training labels can be unioned with HGNN candidates. Current run recovers 116 rank>50 cases to top50, 0 to top5. | Medium | yes |
| Disease index includes obsolete MONDO labels | data + candidate outputs | computed from `data\raw_data\mondo.json` and current candidate file | mimic top1 obsolete predictions: 63/1873, 57 wrong; mimic obsolete gold cases: 51. | High | yes |
| Candidate file duplication/count corruption | candidate audit | current tables | Duplicate candidates 0, candidate count <50 0. | Low | excluded |
| Train/test case ID leakage by raw ID | `src\data\dataset.py`, `src\evaluation\evaluator.py`, exact summary | namespaced IDs and overlap check | Exact summary reports train/test overlap_count 0. | Low | excluded for audited run |

## 9. Experiment-Flow Trustworthiness Audit

| Experiment/Result | Trustworthy? | Reason | Need More Verification |
|---|---|---|---|
| `outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval` exact metrics | yes, for that snapshot | exact summary has 2978/2978 evaluable, 0 skipped, same checkpoint path | Re-run only after freezing config snapshot. |
| `hybrid_tag_v5` top50 candidates | yes, for that snapshot | metadata records same checkpoint, data config, train config, top_k=50, 2978 cases, 148900 rows | Keep metadata with candidate CSV in any report. |
| DDD evidence rerank in `hybrid_tag_v5` | mostly yes | selected on validation candidates, fixed test once; source weights JSON records validation selection | Ensure no later test-side DDD exploratory result is mixed into paper table. |
| mimic SimilarCase in `hybrid_tag_v5` | mostly yes | validation-selected topk=20, weight=0.5, raw_similarity; fixed test output exists | Need stronger validation/test domain shift analysis between `mimic_rag_0425` validation and `mimic_test` test. |
| final mixed metrics in `hybrid_tag_v5` | yes, with caveat | source table records DDD rerank, mimic SimilarCase, other exact baseline | Must state it is mixed post-processing, not pure HGNN exact. |
| current `configs\mainline_full_pipeline.yaml` as reproduction config | no | does not match audited output; resume skip keys are not consumed | P0: export immutable effective configs and command manifest for final run. |
| LLM reranker output directories under `hybrid_tag_v5\stage5_llm_*` | exploratory only | created after manifest finish; not part of final aggregation | Need separate validation protocol before using. |
| Older reports under `reports\diagnosis`, `reports\mimic_diagnosis`, `reports\pretrain_real_gap` | partial | useful supporting analysis but often from older output dirs | Recompute any quoted numbers against the chosen final run. |

## 10. Confirmed Root Causes

### Root Cause 1: mimic gold disease often has little or no overlap with case HPO

- Type: confirmed.
- Affects: mimic_test, ALL.
- Evidence: current dataset quality shows mimic gold overlap zero rate 0.3668 versus DDD 0.0565. mimic overlap-zero cases have gold@50 0.491 and median rank >50.
- Why it lowers accuracy: the HGNN candidate generator relies on HPO/disease hyperedge evidence. If the gold disease hyperedge shares no exact HPO with a case, cosine ranking has weak signal.
- Can reranker solve it: only partially. If gold is absent from top50, top50 rerank cannot. SimilarCase can add labels but current gain is limited.
- Priority: P1.
- Next step: candidate expansion using semantic/ancestor overlap, disease textual evidence, gene/inheritance/onset evidence, and ontology-aware label normalization.

### Root Cause 2: candidate recall@50 is substantially lower for mimic than DDD

- Type: confirmed.
- Affects: mimic_test, ALL.
- Evidence: current test gold@50: DDD 0.7293, mimic 0.6295. mimic has 694/1873 gold-not-top50 cases.
- Why it lowers accuracy: top50 reranking cannot place missing gold diseases at rank 1.
- Can reranker solve it: DDD top50 evidence reranker no; mimic SimilarCase partly, but current recovered-to-top5 count is 0 for rank>50 cases.
- Priority: P1.
- Next step: evaluate top100/top500 recall, candidate expansion, and hybrid retrieval before rerank.

### Root Cause 3: current mimic HPOs are valid but often low-discriminative clinical symptoms

- Type: confirmed.
- Affects: mimic_test.
- Evidence: top mimic HPOs are hypertension, nausea, fever, fatigue, dyspnea, anemia, abdominal pain, diarrhea. HPO mapped rate is 1.0, but Jaccard mean is only 0.0475.
- Why it lowers accuracy: these terms are common in inpatient notes and comorbid states, so they point to broad disease families rather than rare disease-specific diagnoses.
- Can reranker solve it: only if additional evidence is added. A pure HPO-overlap reranker will remain weak.
- Priority: P1.
- Next step: add disease textual evidence, gene/inheritance/onset/organ-system cues, HPO specificity filtering, and MIMIC-specific phenotype normalization.

### Root Cause 4: exact evaluation single-label compression underestimates mimic multi-label cases

- Type: confirmed.
- Affects: mimic_test.
- Evidence: `src\evaluation\evaluator.py` uses `group_df[label_col].iloc[0]`. SimilarCase final output reports any-label top5 0.4335 versus exact top5 0.4042.
- Why it lowers accuracy: if another valid label is predicted but not the first label, exact scoring marks it wrong.
- Can reranker solve it: no, it is an evaluation-label policy issue.
- Priority: P0/P1.
- Next step: keep strict single-label as main metric but add any-label and hierarchy-aware supplementary metrics. Also canonicalize primary label policy.

### Root Cause 5: obsolete MONDO labels and obsolete candidate predictions pollute mimic

- Type: confirmed.
- Affects: mimic_test more than DDD.
- Evidence: mimic has 51 obsolete gold cases; top1 predictions are obsolete in 63/1873 mimic cases, 57 wrong.
- Why it lowers accuracy: obsolete labels create granularity/version mismatches and can attract predictions to deprecated disease buckets.
- Can reranker solve it: not reliably without canonical replacement mapping.
- Priority: P0/P1.
- Next step: audit MONDO obsolete replacement/synonym/parent mappings and create non-destructive canonicalization diagnostics.

### Root Cause 6: DDD has a strong top50-internal ranking bottleneck

- Type: confirmed.
- Affects: DDD.
- Evidence: DDD exact gold@50 0.7293 but exact top1 0.3035. DDD evidence rerank improves top1 to 0.3640 without changing rank_le_50.
- Why it lowers accuracy: gold is often recallable but not ranked first.
- Can reranker solve it: yes, partly.
- Priority: P1/P2.
- Next step: validation-selected pairwise/listwise reranker or ontology-aware hard negatives, with fixed test protocol.

## 11. Hypotheses Requiring Validation

### Hypothesis 1: MONDO parent/child or sibling relaxed evaluation would explain part of mimic near misses

- Evidence: failure cases show adrenal insufficiency variants and similar disease predictions.
- Need validation: compute exact, any-label, parent/child, sibling, and synonym relaxed metrics using one frozen ontology version.

### Hypothesis 2: MIMIC phenotype extraction includes comorbid and nonspecific symptoms that should be downweighted

- Evidence: dominant mimic HPO terms are inpatient symptoms and comorbidities; low exact overlap persists despite 1.0 HPO mapped rate.
- Need validation: compare performance after validation-selected HPO specificity filters, disease-specific HPO weighting, or note-evidence provenance filtering.

### Hypothesis 3: Full-pool CE with current hard-negative strategy is not aligned with top-k diagnosis ranking

- Evidence: DDD top50-internal ranking gap; ontology-aware HN historical report was not strongly successful, but current hard negatives may still be too easy or unstable.
- Need validation: listwise/pairwise reranker on fixed top50, ontology-family hard negatives, and candidate-level calibration.

### Hypothesis 4: Text/gene/inheritance evidence is needed for similar diseases

- Evidence: HPO-only profiles confuse adrenal, autoimmune, hematologic, and pulmonary disease families.
- Need validation: RAG/evidence reranker with authoritative disease descriptions, genes, inheritance, onset, and phenotype frequency.

### Hypothesis 5: Current validation source `mimic_rag_0425` is not fully representative of `mimic_test`

- Evidence: validation mimic_rag and test mimic share top generic HPOs, but test performance remains low and label/extraction pipelines differ.
- Need validation: distributional tests on HPO, disease labels, case-level overlap, and similar-case leakage/coverage.

## 12. Already Excluded or No Current Evidence

| Issue | Status | Evidence |
|---|---|---|
| Gold MONDO absent from disease index is the main cause | excluded for current test | DDD and mimic gold_in_index_rate 1.0; exact summary skipped missing label 0. |
| Massive HPO unmapped loss | excluded for current test | hpo_mapped_rate 1.0 for current test datasets. |
| Candidate CSV duplicate MONDO corrupting ranks | excluded | duplicate candidate cases 0. |
| Candidate count below top50 | excluded | candidate<50 cases 0. |
| Train/test case_id leakage by raw ID | excluded for audited run | exact summary case_id_overlap_count 0. |
| Checkpoint-candidate mismatch in audited run | excluded | manifest and candidate metadata use same checkpoint path. |
| Score sorting direction reversed | no evidence | evaluator/export/rerank all rank by descending score. |
| Evaluation skipped many samples | excluded for exact audited test | num_skipped 0. |

## 13. Prioritized Fix Roadmap

| Priority | Direction | Solves | Modules | Expected Impact | Risk |
|---|---|---|---|---|---|
| P0 | Freeze one canonical experiment snapshot | Result trust | configs, run manifest, outputs naming, report template | Makes future results reproducible and comparable | Low |
| P0 | Add non-destructive MONDO obsolete/replacement audit | Ontology hygiene | audit script, disease index diagnostics | Prevents obsolete labels/predictions from polluting metrics | Medium, replacement mapping may be ambiguous |
| P0 | Add strict plus supplementary evaluation protocol | Multi-label and hierarchy ambiguity | evaluator/reporting only | Clarifies exact vs any-label vs hierarchy-aware metrics | Low |
| P0 | Verify config keys and remove/flag ignored resume fields | Reproducibility | `run_full_mainline_pipeline.py`, config docs | Avoids accidental retraining or wrong output dir | Low |
| P1 | Improve mimic candidate recall before rerank | mimic gold not in top50 | candidate generation, top100/top500, semantic expansion | Most likely to improve mimic top3/top5/top50 | Medium |
| P1 | Add MIMIC-specific phenotype weighting/filtering | generic clinical HPO noise | preprocessing/audit/rerank features | Better signal-to-noise for mimic | Medium, may hurt rare but valid HPOs |
| P1 | Add disease textual/gene/inheritance evidence reranker | similar disease confusion | reranker/RAG/evidence features | Better top1/top5 when gold is recallable | Medium |
| P1 | Validation-selected pairwise/listwise candidate reranker | DDD and mimic top50 ranking | top50 candidate training/eval | Improves ranking without changing encoder | Medium, overfit risk |
| P2 | Ontology-aware hard-negative curriculum | similar diseases | trainer/hard_negative_miner | Paper-method ablation; could improve family ranking | Medium/high based on previous mixed HN results |
| P2 | Multi-view reranker with HPO overlap + disease evidence + similar cases | ranking and recall | rerank modules | Strong paper module if protocol is frozen | Medium |
| P2 | Label granularity-aware evaluation | parent/child/sibling exact misses | evaluation reports | Better error analysis and paper supplement | Low |
| P3 | Multi-source knowledge graph and evidence extraction | long-term recall | data pipeline, KG, RAG | Robust rare disease diagnosis system | High |
| P3 | Uncertainty calibration and abstention | deployment trust | evaluation/scoring | Better clinical decision support | Medium |

## 14. Recommended Next Experiments

Do not start with large HGNN retraining. Recommended sequence:

1. P0 audit-only canonicalization:
   - Generate `obsolete_mondo_case_audit.csv`.
   - Generate `multi_label_exact_vs_any_label_metrics.csv`.
   - Generate `top100/top500_recall_upper_bound.csv` if full score details are available.
2. Candidate recall experiment:
   - Export top100/top500 from the same checkpoint.
   - Compare gold@50/gold@100/gold@500 by DDD and mimic.
   - Do not tune on test. Use validation split for weights.
3. MIMIC evidence quality experiment:
   - Bucket HPOs by IC, frequency, and clinical genericity.
   - Test validation-selected downweighting only.
4. Reranker design:
   - DDD: pairwise/listwise reranker on top50, validation-selected, fixed test.
   - mimic: reranker should operate after candidate expansion, not only top50.
5. GPT Pro deep analysis:
   - Use this report plus tables and failure cases to design a robust method plan.

## 15. GPT Pro Handoff Section

Copy this section to GPT Pro:

```text
Project: D:\RareDisease-traindata
Task: RareDisease HGNN/HPO-MONDO diagnosis framework audit and solution design.

Primary audited run:
- D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5
- Report: D:\RareDisease-traindata\reports\current_framework_accuracy_audit\current_framework_accuracy_audit.md
- Audit tables: D:\RareDisease-traindata\reports\current_framework_accuracy_audit\tables
- Case-level audit outputs: D:\RareDisease-traindata\outputs\current_framework_accuracy_audit

Key configs and code:
- configs\mainline_full_pipeline.yaml
- configs\train_pretrain.yaml
- configs\train_finetune_attn_idf_main.yaml
- configs\data_llldataset_eval.yaml
- outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage1_pretrain.yaml
- outputs\mainline_full_pipeline_hybrid_tag_v5\configs\stage2_finetune.yaml
- src\data\dataset.py
- src\data\build_hypergraph.py
- src\models\hgnn_encoder.py
- src\hgnn_encoder_tag.py
- src\models\model_pipeline.py
- src\training\trainer.py
- src\evaluation\evaluator.py
- tools\export_top50_candidates.py
- tools\run_top50_evidence_rerank.py
- tools\run_mimic_similar_case_aug.py
- tools\run_full_mainline_pipeline.py

Current exact baseline metrics for audited run:
- DDD: n=761, top1=0.3035, top3=0.4297, top5=0.4678, rank_le_50=0.7293, median_rank=7.
- mimic_test_recleaned_mondo_hpo_rows: n=1873, top1=0.1991, top3=0.2974, top5=0.3486, rank_le_50=0.6295, median_rank=20.
- ALL: n=2978, top1=0.2858, top3=0.3889, top5=0.4355, rank_le_50=0.6901, median_rank=10.

Current final mixed metrics:
- DDD uses validation-selected top50 evidence rerank: top1=0.3640, top5=0.5177, rank_le_50=0.7293.
- mimic uses validation-selected SimilarCase: top1=0.2178, top5=0.4042, rank_le_50=0.6690.
- ALL final mixed: top1=0.3130, top5=0.4832, rank_le_50=0.7149.

Confirmed problems:
1. mimic has low gold disease HPO overlap: overlap_zero_rate=0.3668 vs DDD=0.0565.
2. mimic candidate recall@50 is lower: 0.6295 vs DDD=0.7293.
3. mimic HPOs are mapped but clinically generic: hypertension, nausea, fever, fatigue, dyspnea, anemia, abdominal pain, diarrhea dominate.
4. exact evaluation compresses multi-label cases to first label. Any-label final mimic top5=0.4335 vs exact top5=0.4042.
5. obsolete MONDO hygiene issue: mimic obsolete gold cases=51; mimic top1 obsolete predictions=63, of which 57 are wrong.
6. DDD bottleneck is more top50-internal ranking; mimic bottleneck is recall + weak evidence + label granularity.
7. Current top-level config does not match the audited run, and ignored resume skip keys exist. Future experiments need frozen manifests.

Excluded as primary causes in current run:
- Gold missing from disease index: 0 current test cases.
- Massive unmapped HPO: hpo_mapped_rate=1.0.
- Candidate duplicates or candidate count <50: 0.
- Train/test case_id overlap: exact summary overlap_count=0.
- Checkpoint-candidate mismatch in audited run: metadata paths match.
- Sorting direction bug: no evidence.

Most important GPT Pro analysis questions:
1. How should candidate generation be redesigned for mimic so gold@50/100/500 improves without test tuning?
2. How should MONDO obsolete/parent-child/synonym mapping be handled while preserving strict and supplementary metrics?
3. What reranker architecture best combines HGNN score, HPO overlap, disease textual evidence, gene/inheritance/onset, phenotype frequency, and similar cases?
4. How should MIMIC phenotype noise/comorbidity be downweighted without deleting rare-disease evidence?
5. What validation protocol can select weights on DDD/mimic_rag validation while fixed-test remains clean?

Recommended next Codex tasks:
1. Create read-only audits for obsolete MONDO replacement candidates and exact vs any-label/hierarchy-aware metrics.
2. Export and audit top100/top500 recall from the same checkpoint if feasible without training.
3. Build a validation-only candidate expansion prototype and report recall upper bounds.
4. Only after P0 trust fixes, implement a small reranker ablation with fixed test protocol.
```

## 16. Audit Script Inventory

| Script | Purpose | Modifies Raw Data | Output Path |
|---|---|---|---|
| `D:\RareDisease-traindata\tools\audit_current_framework_accuracy.py` | Existing read-only audit script used to generate current inventory, metrics, candidate recall, dataset quality, top HPO, gap, and failure case tables. | no | `reports\current_framework_accuracy_audit\tables`, `outputs\current_framework_accuracy_audit` |
| `tools\run_eval_with_pretrain_alignment.py` | Requested by audit prompt | NOT_FOUND | NOT_FOUND |
| `tools\audit_pretrain_vs_real_dataset_gap.py` | Requested by audit prompt | NOT_FOUND | NOT_FOUND |

No core HGNN encoder, trainer, evaluator, model pipeline, config, checkpoint, dataset, or mainline output file was modified by this audit report.
