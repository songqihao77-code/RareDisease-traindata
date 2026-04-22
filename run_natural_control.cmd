@echo off
setlocal

cd /d D:\RareDisease-traindata

echo [1/2] Natural-control finetune started...
python -m src.training.trainer --config configs/train_finetune_natural_control.yaml
if errorlevel 1 (
    echo [ERROR] Natural-control finetune failed.
    exit /b 1
)

echo [2/2] Natural-control evaluation started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/train_finetune_natural_control.yaml
if errorlevel 1 (
    echo [ERROR] Natural-control evaluation failed.
    exit /b 1
)

echo [DONE] Natural-control finetune and evaluation completed successfully.
exit /b 0
