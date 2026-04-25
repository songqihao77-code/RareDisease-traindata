@echo off
setlocal

set CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /d D:\RareDisease-traindata

echo [1/4] prior=none deterministic train started...
python -m src.training.trainer --config configs/train_finetune_attn_none_deterministic.yaml
if errorlevel 1 (
    echo [ERROR] prior=none deterministic train failed.
    exit /b 1
)

echo [2/4] prior=none deterministic eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/train_finetune_attn_none_deterministic.yaml
if errorlevel 1 (
    echo [ERROR] prior=none deterministic eval failed.
    exit /b 1
)

echo [3/4] beta=0.2 deterministic train started...
python -m src.training.trainer --config configs/train_finetune_attn_idf_main.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 deterministic train failed.
    exit /b 1
)

echo [4/4] beta=0.2 deterministic eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/train_finetune_attn_idf_main.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 deterministic eval failed.
    exit /b 1
)

echo [DONE] Deterministic prior comparison completed successfully.
exit /b 0
