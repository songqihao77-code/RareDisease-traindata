@echo off
setlocal

set CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /d D:\RareDisease-traindata

echo [1/12] prior=none seed=42 train started...
python -m src.training.trainer --config configs/seed_sweep/train_finetune_none_seed42.yaml
if errorlevel 1 (
    echo [ERROR] prior=none seed=42 train failed.
    exit /b 1
)

echo [2/12] prior=none seed=42 eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/seed_sweep/train_finetune_none_seed42.yaml
if errorlevel 1 (
    echo [ERROR] prior=none seed=42 eval failed.
    exit /b 1
)

echo [3/12] prior=none seed=123 train started...
python -m src.training.trainer --config configs/seed_sweep/train_finetune_none_seed123.yaml
if errorlevel 1 (
    echo [ERROR] prior=none seed=123 train failed.
    exit /b 1
)

echo [4/12] prior=none seed=123 eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/seed_sweep/train_finetune_none_seed123.yaml
if errorlevel 1 (
    echo [ERROR] prior=none seed=123 eval failed.
    exit /b 1
)

echo [5/12] prior=none seed=3407 train started...
python -m src.training.trainer --config configs/seed_sweep/train_finetune_none_seed3407.yaml
if errorlevel 1 (
    echo [ERROR] prior=none seed=3407 train failed.
    exit /b 1
)

echo [6/12] prior=none seed=3407 eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/seed_sweep/train_finetune_none_seed3407.yaml
if errorlevel 1 (
    echo [ERROR] prior=none seed=3407 eval failed.
    exit /b 1
)

echo [7/12] beta=0.2 seed=42 train started...
python -m src.training.trainer --config configs/seed_sweep/train_finetune_beta02_seed42.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 seed=42 train failed.
    exit /b 1
)

echo [8/12] beta=0.2 seed=42 eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/seed_sweep/train_finetune_beta02_seed42.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 seed=42 eval failed.
    exit /b 1
)

echo [9/12] beta=0.2 seed=123 train started...
python -m src.training.trainer --config configs/seed_sweep/train_finetune_beta02_seed123.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 seed=123 train failed.
    exit /b 1
)

echo [10/12] beta=0.2 seed=123 eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/seed_sweep/train_finetune_beta02_seed123.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 seed=123 eval failed.
    exit /b 1
)

echo [11/12] beta=0.2 seed=3407 train started...
python -m src.training.trainer --config configs/seed_sweep/train_finetune_beta02_seed3407.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 seed=3407 train failed.
    exit /b 1
)

echo [12/12] beta=0.2 seed=3407 eval started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/seed_sweep/train_finetune_beta02_seed3407.yaml
if errorlevel 1 (
    echo [ERROR] beta=0.2 seed=3407 eval failed.
    exit /b 1
)

echo [DONE] Deterministic seed sweep prior comparison completed successfully.
exit /b 0
