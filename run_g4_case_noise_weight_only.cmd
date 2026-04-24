@echo off
setlocal

cd /d D:\RareDisease-traindata || goto :fail

set "PYTHON_EXE=D:\python\python.exe"
set "PRETRAIN_CFG=D:\RareDisease-traindata\configs\train_pretrain.yaml"
set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4_case_noise_weight_only.yaml"
set "EVAL_CFG=D:\RareDisease-traindata\configs\data_llldataset_eval.yaml"
set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4_case_noise_weight_only\checkpoints\best.pt"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    exit /b 1
)

echo [INFO] Current g4 case-noise weight_only mainline uses idf weighting.

echo [1/3] g4 pretrain started...
"%PYTHON_EXE%" -m src.training.trainer --config "%PRETRAIN_CFG%"
if errorlevel 1 (
    echo [ERROR] g4 pretrain failed.
    exit /b 1
)

echo [2/3] g4 finetune started...
"%PYTHON_EXE%" -m src.training.trainer --config "%FINETUNE_CFG%"
if errorlevel 1 (
    echo [ERROR] g4 finetune failed.
    exit /b 1
)

echo [3/3] g4 evaluation started...
"%PYTHON_EXE%" -m src.evaluation.evaluator --data_config_path "%EVAL_CFG%" --train_config_path "%FINETUNE_CFG%" --checkpoint_path "%CHECKPOINT_PATH%"
if errorlevel 1 (
    echo [ERROR] g4 evaluation failed.
    exit /b 1
)

echo [DONE] g4 pretrain, finetune and evaluation completed successfully.
exit /b 0

:fail
echo [ERROR] Script stopped before g4 started.
exit /b 1
