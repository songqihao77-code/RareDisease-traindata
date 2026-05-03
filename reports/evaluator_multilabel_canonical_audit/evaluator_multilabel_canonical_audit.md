# Evaluator Multilabel / Canonical MONDO Audit

Generated at: `2026-04-30T09:58:09`

## Executive Summary

- No training was run; this audit re-reads frozen evaluation artifacts and raw test labels.
- Frozen run: `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5`
- Locked config: `D:\RareDisease-traindata\configs\mainline_full_pipeline_hybrid_tag_v5.locked.yaml`
- Disease index: `D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\Disease_index_v4.xlsx`
- Multi-label cases: `243` total, `227` in mimic-like datasets.
- Obsolete-label cases: `67` total, `63` in mimic-like datasets.
- mimic final top5 strict primary vs any-label: `0.4042` -> `0.4335`.
- Primary metric remains strict primary-label; any-label and canonical/obsolete-aware metrics are supplementary.

## Inputs

- data config: `D:\RareDisease-traindata\configs\data_llldataset_eval.yaml`
- output manifest: `D:\RareDisease-traindata\outputs\evaluator_multilabel_canonical_audit\evaluator_multilabel_canonical_audit_manifest.json`
- exact details: `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage3_exact_eval\exact_details.csv`
- candidate metadata: `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\stage4_candidates\top50_candidates_test.metadata.json`
- final case ranks: `D:\RareDisease-traindata\outputs\mainline_full_pipeline_hybrid_tag_v5\mainline_final_case_ranks.csv`

## Evaluator Code Changes

- `src/evaluation/evaluator.py` now preserves `mondo_labels` for every case while keeping `mondo_label` as the strict primary label.
- Future evaluator runs keep existing `true_rank` / top-level strict metrics and add `any_label_rank`, `canonical_primary_rank`, and `canonical_any_label_rank` in details.
- `src/evaluation/mondo_canonicalizer.py` resolves MONDO alternative IDs and curated obsolete replacements for supplementary relaxed metrics.

## Label Coverage

| n_cases | multi_label_cases | obsolete_any_label_cases | primary_in_index_rate | any_label_in_index_rate |
| --- | --- | --- | --- | --- |
| 2978 | 243 | 67 | 1.0 | 1.0 |

## Metric Comparison

Full table: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\metric_comparison.csv`

| rank_source | dataset | scoring_mode | n | top1 | top3 | top5 | top10 | top30 | rank_le_50 | median_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_full_strict_primary | DDD | strict_primary | 761 | 0.3035479632063075 | 0.4296977660972405 | 0.46780551905387646 | 0.5466491458607096 | 0.671484888304862 | 0.7293035479632063 | 7.0 |
| exact_full_strict_primary | mimic_test_recleaned_mondo_hpo_rows | strict_primary | 1873 | 0.19914575547250402 | 0.29738387613454353 | 0.34863854778430325 | 0.4180459156433529 | 0.5483182060864923 | 0.6294714361986119 | 20.0 |
| exact_full_strict_primary | ALL | strict_primary | 2978 | 0.2857622565480188 | 0.3888515782404298 | 0.43552719946272667 | 0.5040295500335796 | 0.6215580926796508 | 0.6900604432505036 | 10.0 |
| hgnn_top50_candidates | DDD | strict_primary | 761 | 0.3035479632063075 | 0.4296977660972405 | 0.46780551905387646 | 0.5466491458607096 | 0.671484888304862 | 0.7293035479632063 | 7.0 |
| hgnn_top50_candidates | mimic_test_recleaned_mondo_hpo_rows | strict_primary | 1873 | 0.19914575547250402 | 0.29738387613454353 | 0.34863854778430325 | 0.4180459156433529 | 0.5483182060864923 | 0.6294714361986119 | 20.0 |
| hgnn_top50_candidates | ALL | strict_primary | 2978 | 0.2857622565480188 | 0.3888515782404298 | 0.43552719946272667 | 0.5040295500335796 | 0.6215580926796508 | 0.6900604432505036 | 10.0 |
| hgnn_top50_candidates | DDD | any_label | 761 | 0.30486202365308807 | 0.43101182654402104 | 0.47174770039421815 | 0.5479632063074902 | 0.6727989487516426 | 0.7306176084099869 | 7.0 |
| hgnn_top50_candidates | mimic_test_recleaned_mondo_hpo_rows | any_label | 1873 | 0.21302722904431393 | 0.31607047517351844 | 0.3694607581420182 | 0.4426054458088628 | 0.5814201815269621 | 0.6636412172984517 | 16.0 |
| hgnn_top50_candidates | ALL | any_label | 2978 | 0.29482874412357285 | 0.40094022834116855 | 0.4496306245802552 | 0.5198119543317663 | 0.6427132303559436 | 0.7118871725990598 | 9.0 |
| hgnn_top50_candidates | DDD | canonical_primary | 761 | 0.3035479632063075 | 0.4296977660972405 | 0.46780551905387646 | 0.5479632063074902 | 0.6727989487516426 | 0.7306176084099869 | 7.0 |
| hgnn_top50_candidates | mimic_test_recleaned_mondo_hpo_rows | canonical_primary | 1873 | 0.19914575547250402 | 0.29738387613454353 | 0.34863854778430325 | 0.4180459156433529 | 0.5483182060864923 | 0.6294714361986119 | 20.0 |
| hgnn_top50_candidates | ALL | canonical_primary | 2978 | 0.2857622565480188 | 0.3888515782404298 | 0.43552719946272667 | 0.5043653458697113 | 0.6218938885157824 | 0.6903962390866353 | 10.0 |
| hgnn_top50_candidates | DDD | canonical_any_label | 761 | 0.30486202365308807 | 0.43101182654402104 | 0.47174770039421815 | 0.5492772667542707 | 0.6741130091984231 | 0.7319316688567674 | 7.0 |
| hgnn_top50_candidates | mimic_test_recleaned_mondo_hpo_rows | canonical_any_label | 1873 | 0.21302722904431393 | 0.31607047517351844 | 0.3694607581420182 | 0.4426054458088628 | 0.5814201815269621 | 0.6636412172984517 | 16.0 |
| hgnn_top50_candidates | ALL | canonical_any_label | 2978 | 0.29482874412357285 | 0.40094022834116855 | 0.4496306245802552 | 0.5201477501678979 | 0.6430490261920753 | 0.7122229684351914 | 9.0 |
| ddd_rerank_top50 | DDD | strict_primary | 761 | 0.3639947437582129 | 0.4783180026281209 | 0.5177398160315374 | 0.5978975032851511 | 0.6964520367936925 | 0.7293035479632063 | 4.0 |
| ddd_rerank_top50 | ALL | strict_primary | 761 | 0.3639947437582129 | 0.4783180026281209 | 0.5177398160315374 | 0.5978975032851511 | 0.6964520367936925 | 0.7293035479632063 | 4.0 |
| ddd_rerank_top50 | DDD | any_label | 761 | 0.36530880420499345 | 0.480946123521682 | 0.5216819973718791 | 0.5992115637319316 | 0.6977660972404731 | 0.7306176084099869 | 4.0 |
| ddd_rerank_top50 | ALL | any_label | 761 | 0.36530880420499345 | 0.480946123521682 | 0.5216819973718791 | 0.5992115637319316 | 0.6977660972404731 | 0.7306176084099869 | 4.0 |
| ddd_rerank_top50 | DDD | canonical_primary | 761 | 0.3639947437582129 | 0.47963206307490147 | 0.519053876478318 | 0.5992115637319316 | 0.6977660972404731 | 0.7306176084099869 | 4.0 |
| ddd_rerank_top50 | ALL | canonical_primary | 761 | 0.3639947437582129 | 0.47963206307490147 | 0.519053876478318 | 0.5992115637319316 | 0.6977660972404731 | 0.7306176084099869 | 4.0 |
| ddd_rerank_top50 | DDD | canonical_any_label | 761 | 0.36530880420499345 | 0.48226018396846254 | 0.5229960578186597 | 0.6005256241787122 | 0.6990801576872536 | 0.7319316688567674 | 4.0 |
| ddd_rerank_top50 | ALL | canonical_any_label | 761 | 0.36530880420499345 | 0.48226018396846254 | 0.5229960578186597 | 0.6005256241787122 | 0.6990801576872536 | 0.7319316688567674 | 4.0 |
| mimic_similarcase_top50 | mimic_test_recleaned_mondo_hpo_rows | strict_primary | 1873 | 0.21783235451147892 | 0.34917245061398827 | 0.404164442071543 | 0.4612920448478377 | 0.5830218900160171 | 0.6689802455953017 | 15.0 |
| mimic_similarcase_top50 | ALL | strict_primary | 1873 | 0.21783235451147892 | 0.34917245061398827 | 0.404164442071543 | 0.4612920448478377 | 0.5830218900160171 | 0.6689802455953017 | 15.0 |
| mimic_similarcase_top50 | mimic_test_recleaned_mondo_hpo_rows | any_label | 1873 | 0.23384943940202882 | 0.37533368926855315 | 0.4335290977042178 | 0.49279231179925254 | 0.6139882541377469 | 0.7031500266951415 | 11.0 |
| mimic_similarcase_top50 | ALL | any_label | 1873 | 0.23384943940202882 | 0.37533368926855315 | 0.4335290977042178 | 0.49279231179925254 | 0.6139882541377469 | 0.7031500266951415 | 11.0 |
| mimic_similarcase_top50 | mimic_test_recleaned_mondo_hpo_rows | canonical_primary | 1873 | 0.21783235451147892 | 0.34917245061398827 | 0.404164442071543 | 0.4612920448478377 | 0.5830218900160171 | 0.6689802455953017 | 15.0 |
| mimic_similarcase_top50 | ALL | canonical_primary | 1873 | 0.21783235451147892 | 0.34917245061398827 | 0.404164442071543 | 0.4612920448478377 | 0.5830218900160171 | 0.6689802455953017 | 15.0 |
| mimic_similarcase_top50 | mimic_test_recleaned_mondo_hpo_rows | canonical_any_label | 1873 | 0.23384943940202882 | 0.37533368926855315 | 0.4335290977042178 | 0.49279231179925254 | 0.6139882541377469 | 0.7031500266951415 | 11.0 |
| mimic_similarcase_top50 | ALL | canonical_any_label | 1873 | 0.23384943940202882 | 0.37533368926855315 | 0.4335290977042178 | 0.49279231179925254 | 0.6139882541377469 | 0.7031500266951415 | 11.0 |
| final_mixed | DDD | strict_primary | 761 | 0.3639947437582129 | 0.4783180026281209 | 0.5177398160315374 | 0.5978975032851511 | 0.6964520367936925 | 0.7293035479632063 | 4.0 |
| final_mixed | mimic_test_recleaned_mondo_hpo_rows | strict_primary | 1873 | 0.21783235451147892 | 0.34917245061398827 | 0.404164442071543 | 0.4612920448478377 | 0.5830218900160171 | 0.6689802455953017 | 15.0 |
| final_mixed | ALL | strict_primary | 2978 | 0.31296171927468097 | 0.4338482202820685 | 0.4832102081934184 | 0.5443250503693754 | 0.6497649429147079 | 0.7149093351242445 | 6.0 |
| final_mixed | DDD | any_label | 761 | 0.36530880420499345 | 0.480946123521682 | 0.5216819973718791 | 0.5992115637319316 | 0.6977660972404731 | 0.7306176084099869 | 4.0 |
| final_mixed | mimic_test_recleaned_mondo_hpo_rows | any_label | 1873 | 0.23384943940202882 | 0.37533368926855315 | 0.4335290977042178 | 0.49279231179925254 | 0.6139882541377469 | 0.7031500266951415 | 11.0 |
| final_mixed | ALL | any_label | 2978 | 0.3233713901947616 | 0.45097380792478176 | 0.5026863666890531 | 0.5644728005372733 | 0.6695768972464742 | 0.7367360644728005 | 5.0 |
| final_mixed | DDD | canonical_primary | 761 | 0.3639947437582129 | 0.47963206307490147 | 0.519053876478318 | 0.5992115637319316 | 0.6977660972404731 | 0.7306176084099869 | 4.0 |
| final_mixed | mimic_test_recleaned_mondo_hpo_rows | canonical_primary | 1873 | 0.21783235451147892 | 0.34917245061398827 | 0.404164442071543 | 0.4612920448478377 | 0.5830218900160171 | 0.6689802455953017 | 15.0 |
| final_mixed | ALL | canonical_primary | 2978 | 0.31296171927468097 | 0.4341840161182001 | 0.48354600402955006 | 0.544660846205507 | 0.6501007387508395 | 0.7152451309603761 | 6.0 |
| final_mixed | DDD | canonical_any_label | 761 | 0.36530880420499345 | 0.48226018396846254 | 0.5229960578186597 | 0.6005256241787122 | 0.6990801576872536 | 0.7319316688567674 | 4.0 |
| final_mixed | mimic_test_recleaned_mondo_hpo_rows | canonical_any_label | 1873 | 0.23384943940202882 | 0.37533368926855315 | 0.4335290977042178 | 0.49279231179925254 | 0.6139882541377469 | 0.7031500266951415 | 11.0 |
| final_mixed | ALL | canonical_any_label | 2978 | 0.3233713901947616 | 0.4513096037609134 | 0.5030221625251847 | 0.564808596373405 | 0.6699126930826058 | 0.7370718603089321 | 5.0 |

## Obsolete MONDO Cases

Full table: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\obsolete_mondo_cases.csv`

| case_id | dataset | primary_label | obsolete_labels | hgnn_top1 | hgnn_any_label_rank | final_any_label_rank |
| --- | --- | --- | --- | --- | --- | --- |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_26 | DDD | MONDO:0014076 | [] | MONDO:0018340 | 13 | 15 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_48 | DDD | MONDO:0100229 | ["MONDO:0100229"] | MONDO:0007094 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_74 | DDD | MONDO:0013935 | ["MONDO:0013935"] | MONDO:0012662 | 13 | 18 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_132 | DDD | MONDO:0003608 | [] | MONDO:0014753 | 8 | 8 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_262 | DDD | MONDO:0016828 | [] | MONDO:0016363 | 2 | 3 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_303 | DDD | MONDO:0013245 | [] | MONDO:0018112 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_354 | DDD | MONDO:0012556 | [] | MONDO:0044234 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_534 | DDD | MONDO:0012699 | [] | MONDO:0018277 | 23 | 18 |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_630 | DDD | MONDO:0018651 | ["MONDO:0018651"] | MONDO:0032620 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/HMS.xlsx::case_18 | HMS | MONDO:0007798 | ["MONDO:0007798"] | MONDO:0007798 | 1 | 1 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_48 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0019527 | [] | MONDO:0800029 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_61 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0019635 | [] | MONDO:0800029 | 1000000000 | 28 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_64 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0017154 | ["MONDO:0017154"] | MONDO:0016264 | 1000000000 | 39 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_97 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0002571 | [] | MONDO:0020671 | 21 | 3 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_170 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0019434 | [] | MONDO:0015490 | 13 | 14 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_173 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0007713 | [] | MONDO:0016428 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_184 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0005021 | [] | MONDO:0016343 | 4 | 1 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_185 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0015924 | [] | MONDO:0016343 | 6 | 10 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_186 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0018987 | ["MONDO:0016428"] | MONDO:0017363 | 19 | 22 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_191 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0019100 | [] | MONDO:0015343 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_231 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0002571 | [] | MONDO:0016428 | 2 | 1 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_264 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0005021 | [] | MONDO:0016343 | 3 | 1 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_272 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0018157 | ["MONDO:0018157"] | MONDO:0008946 | 1000000000 | 1000000000 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_273 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0018157 | ["MONDO:0018157"] | MONDO:0018608 | 3 | 1 |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_340 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0019194 | [] | MONDO:0800029 | 29 | 33 |
| ... | ... | ... | ... | ... | ... | ... |

## Sample Cases

Full table: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\sample_cases.csv`

| case_id | dataset | primary_label | all_labels | top1_candidate | strict_primary_rank_hgnn | any_label_rank_hgnn | metric_source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_73 | DDD | MONDO:0035524 | ["MONDO:0035524", "MONDO:0007201"] | MONDO:0035521 | 3 | 2 | ddd_validation_selected_grid_rerank |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_187 | DDD | MONDO:0019766 | ["MONDO:0019766", "MONDO:0010653"] | MONDO:0012496 | 10 | 3 | ddd_validation_selected_grid_rerank |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_322 | DDD | MONDO:0016543 | ["MONDO:0016543", "MONDO:0009908"] | MONDO:0019335 | 8 | 4 | ddd_validation_selected_grid_rerank |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_382 | DDD | MONDO:0016091 | ["MONDO:0016091", "MONDO:0009499"] | MONDO:0009590 | 1000000000 | 5 | ddd_validation_selected_grid_rerank |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_526 | DDD | MONDO:0017226 | ["MONDO:0017226", "MONDO:0009843"] | MONDO:0009843 | 2 | 1 | ddd_validation_selected_grid_rerank |
| test::LLLdataset/dataset/processed/test/DDD.csv::case_741 | DDD | MONDO:0019141 | ["MONDO:0019141", "MONDO:0006602"] | MONDO:0008293 | 7 | 6 | ddd_validation_selected_grid_rerank |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_38 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0100480 | ["MONDO:0100480", "MONDO:0015128"] | MONDO:0016367 | 46 | 14 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_74 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0017604 | ["MONDO:0017604", "MONDO:8000010"] | MONDO:8000010 | 1000000000 | 1 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_186 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0018987 | ["MONDO:0018987", "MONDO:0016428"] | MONDO:0017363 | 1000000000 | 19 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_246 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0001633 | ["MONDO:0001633", "MONDO:0019741"] | MONDO:8000010 | 1000000000 | 12 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_265 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0007803 | ["MONDO:0007803", "MONDO:0009693"] | MONDO:0019065 | 1000000000 | 6 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_269 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0010789 | ["MONDO:0010789", "MONDO:0019635"] | MONDO:0004114 | 1000000000 | 3 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_273 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0018157 | ["MONDO:0018157", "MONDO:0019635"] | MONDO:0018608 | 1000000000 | 3 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_275 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0010789 | ["MONDO:0010789", "MONDO:0019635"] | MONDO:0009693 | 1000000000 | 3 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_278 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0010789 | ["MONDO:0010789", "MONDO:0019635"] | MONDO:0018608 | 1000000000 | 15 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_279 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0010789 | ["MONDO:0010789", "MONDO:0019635"] | MONDO:0009637 | 43 | 23 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_313 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0044067 | ["MONDO:0044067", "MONDO:0007915"] | MONDO:0018905 | 1000000000 | 37 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_317 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0002404 | ["MONDO:0002404", "MONDO:0018874"] | MONDO:0020077 | 1000000000 | 24 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_321 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0020553 | ["MONDO:0020553", "MONDO:0018874"] | MONDO:0018874 | 1000000000 | 1 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_324 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0005615 | ["MONDO:0005615", "MONDO:0009693"] | MONDO:0009693 | 1000000000 | 1 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_340 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0019194 | ["MONDO:0019194", "MONDO:0009637", "MONDO:0007108"] | MONDO:0800029 | 1000000000 | 29 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_353 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0007915 | ["MONDO:0007915", "MONDO:0008558"] | MONDO:0008558 | 2 | 1 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_373 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0011996 | ["MONDO:0011996", "MONDO:0004967"] | MONDO:0018874 | 45 | 38 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_378 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0019338 | ["MONDO:0019338", "MONDO:0015924"] | MONDO:0015924 | 43 | 1 | similar_case_fixed_test |
| test::LLLdataset/dataset/processed/test/mimic_test_recleaned_mondo_hpo_rows.csv::case_380 | mimic_test_recleaned_mondo_hpo_rows | MONDO:0016642 | ["MONDO:0016642", "MONDO:0018874"] | MONDO:0018874 | 1000000000 | 1 | similar_case_fixed_test |
| ... | ... | ... | ... | ... | ... | ... | ... |

## Generated Files

- case_label_audit: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\case_label_audit.csv`
- metric_comparison: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\metric_comparison.csv`
- hgnn_top50_multilabel_ranks: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\hgnn_top50_multilabel_ranks.csv`
- final_mixed_multilabel_ranks: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\final_mixed_multilabel_ranks.csv`
- obsolete_cases: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\obsolete_mondo_cases.csv`
- sample_cases: `D:\RareDisease-traindata\reports\evaluator_multilabel_canonical_audit\tables\sample_cases.csv`
- manifest: `D:\RareDisease-traindata\outputs\evaluator_multilabel_canonical_audit\evaluator_multilabel_canonical_audit_manifest.json`

## Commands Run

- `D:\python\python.exe -m compileall src\evaluation\evaluator.py src\evaluation\mondo_canonicalizer.py src\evaluation\multilabel_metrics.py tools\audit_evaluator_multilabel_canonical.py`
- `D:\python\python.exe tools\audit_evaluator_multilabel_canonical.py --run-dir outputs\mainline_full_pipeline_hybrid_tag_v5 --locked-config configs\mainline_full_pipeline_hybrid_tag_v5.locked.yaml --data-config configs\data_llldataset_eval.yaml --out-report-dir reports\evaluator_multilabel_canonical_audit --out-output-dir outputs\evaluator_multilabel_canonical_audit`
