@echo off
setlocal

set "PY=D:\python\python.exe"
if not exist "%PY%" set "PY=python"

cd /d D:\RareDisease-traindata

echo [1/12] Pretrain started...
"%PY%" -m src.training.trainer --config configs\train_pretrain.yaml
if errorlevel 1 (
    echo [ERROR] Pretrain failed.
    exit /b 1
)

echo [2/12] Sync pretrain checkpoint for finetune init...
copy /Y outputs\stage1_pretrain_v59\checkpoints\best.pt outputs\stage1_pretrain_v59\checkpoints\best_fixed_for_weighting_ablation.pt
if errorlevel 1 (
    echo [ERROR] Pretrain checkpoint sync failed.
    exit /b 1
)

echo [3/12] Finetune started...
"%PY%" -m src.training.trainer --config configs\train_finetune_attn_idf_main.yaml
if errorlevel 1 (
    echo [ERROR] Finetune failed.
    exit /b 1
)

echo [4/12] HGNN exact evaluation started...
"%PY%" -m src.evaluation.evaluator --data_config_path configs\data_llldataset_eval.yaml --train_config_path configs\train_finetune_attn_idf_main.yaml --checkpoint_path outputs\attn_beta_sweep\edge_log_beta02\checkpoints\best.pt
if errorlevel 1 (
    echo [ERROR] HGNN exact evaluation failed.
    exit /b 1
)

echo [5/12] Export test top50 candidates from latest finetune checkpoint...
"%PY%" tools\export_top50_candidates.py --data-config-path configs\data_llldataset_eval.yaml --train-config-path configs\train_finetune_attn_idf_main.yaml --checkpoint-path outputs\attn_beta_sweep\edge_log_beta02\checkpoints\best.pt --output-path outputs\rerank\top50_candidates_v2.csv --top-k 50 --case-source test
if errorlevel 1 (
    echo [ERROR] Test top50 candidate export failed.
    exit /b 1
)

echo [6/12] Copy test candidates for MIMIC SimilarCase module...
if not exist outputs\mimic_next mkdir outputs\mimic_next
copy /Y outputs\rerank\top50_candidates_v2.csv outputs\mimic_next\top50_candidates_recleaned_test.csv
if errorlevel 1 (
    echo [ERROR] MIMIC candidate copy failed.
    exit /b 1
)

echo [7/12] Export validation top50 candidates for fixed protocol selection...
"%PY%" tools\export_top50_candidates.py --data-config-path configs\data_llldataset_eval.yaml --train-config-path configs\train_finetune_attn_idf_main.yaml --checkpoint-path outputs\attn_beta_sweep\edge_log_beta02\checkpoints\best.pt --output-path outputs\rerank\top50_candidates_validation.csv --top-k 50 --case-source validation
if errorlevel 1 (
    echo [ERROR] Validation top50 candidate export failed.
    exit /b 1
)

echo [8/12] Build HGNN baseline case ranks from latest top50 candidates...
"%PY%" tools\build_top50_baseline_case_ranks.py --candidates-path outputs\rerank\top50_candidates_v2.csv --output-case-ranks outputs\rerank\top50_rerank_case_ranks.csv --output-metrics outputs\rerank\top50_rerank_metrics.csv
if errorlevel 1 (
    echo [ERROR] Baseline case-rank build failed.
    exit /b 1
)

echo [9/12] Select DDD validation grid rerank weights...
"%PY%" tools\run_top50_evidence_rerank.py --protocol validation_select --validation-candidates-path outputs\rerank\top50_candidates_validation.csv --test-candidates-path outputs\rerank\top50_candidates_v2.csv --output-dir outputs\rerank --selected-weights-path outputs\rerank\rerank_selected_grid_weights.json --selection-objective DDD_top1
if errorlevel 1 (
    echo [ERROR] DDD validation grid search failed.
    exit /b 1
)

echo [10/12] Finalize DDD validation-selected fixed rerank package...
"%PY%" reports\ddd_improvement\run_ddd_validation_selected_rerank.py
if errorlevel 1 (
    echo [ERROR] DDD fixed rerank finalization failed.
    exit /b 1
)

echo [11/12] Run MIMIC SimilarCase-Aug fixed test protocol...
"%PY%" tools\run_mimic_similar_case_aug.py --validation-candidates-path outputs\rerank\top50_candidates_validation.csv --test-candidates-path outputs\mimic_next\top50_candidates_recleaned_test.csv --output-dir reports\mimic_next
if errorlevel 1 (
    echo [ERROR] MIMIC SimilarCase-Aug failed.
    exit /b 1
)

echo [12/12] Build final all-dataset mainline report with DDD and MIMIC postprocess modules...
"%PY%" tools\run_mainline_hgnn_val_grid_rerank_all.py --config-path configs\mainline_hgnn_val_grid_rerank.yaml
if errorlevel 1 (
    echo [ERROR] Final mainline report failed.
    exit /b 1
)

echo [DONE] Full mainline with DDD/MIMIC postprocess completed successfully.
echo Final metrics: outputs\mainline_hgnn_val_grid_rerank_all\mainline_final_metrics.csv
echo Final report : reports\mainline_hgnn_val_grid_rerank_all\mainline_final_all_dataset_report.md
exit /b 0
