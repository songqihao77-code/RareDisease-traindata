@echo off
setlocal

cd /d D:\RareDisease-traindata

set "DATA_CONFIG=configs\data_llldataset_eval.yaml"
set "PRETRAIN_CKPT=D:\RareDisease-traindata\outputs\stage1_pretrain\checkpoints\best.pt"

if not exist "%PRETRAIN_CKPT%" (
    echo [ERROR] Missing pretrain checkpoint: %PRETRAIN_CKPT%
    exit /b 1
)

call :run_one attn_ru100_uniform configs\train_finetune_attn_ru100.yaml D:\RareDisease-traindata\outputs\attn_ru100_uniform
if errorlevel 1 exit /b 1

call :run_one attn_ru020_main configs\train_finetune_attn_ru020.yaml D:\RareDisease-traindata\outputs\attn_ru020_main
if errorlevel 1 exit /b 1

call :run_one attn_ru010_light configs\train_finetune_attn_ru010.yaml D:\RareDisease-traindata\outputs\attn_ru010_light
if errorlevel 1 exit /b 1

call :run_one attn_ru000_pure configs\train_finetune_attn_ru000.yaml D:\RareDisease-traindata\outputs\attn_ru000_pure
if errorlevel 1 exit /b 1

echo ==================================================
echo [DONE] All residual_uniform sweep runs finished.
echo Results:
echo   D:\RareDisease-traindata\outputs\attn_ru100_uniform
echo   D:\RareDisease-traindata\outputs\attn_ru020_main
echo   D:\RareDisease-traindata\outputs\attn_ru010_light
echo   D:\RareDisease-traindata\outputs\attn_ru000_pure
exit /b 0

:run_one
set "EXP_NAME=%~1"
set "CFG_PATH=%~2"
set "SAVE_DIR=%~3"

echo ==================================================
echo [%EXP_NAME%] Training...
python -m src.training.trainer --config "%CFG_PATH%"
if errorlevel 1 (
    echo [ERROR] %EXP_NAME% training failed.
    exit /b 1
)

echo [%EXP_NAME%] Evaluating...
python -m src.evaluation.evaluator --data_config_path "%DATA_CONFIG%" --train_config_path "%CFG_PATH%" --checkpoint_path "%SAVE_DIR%\checkpoints\best.pt"
if errorlevel 1 (
    echo [ERROR] %EXP_NAME% evaluation failed.
    exit /b 1
)

echo [%EXP_NAME%] Finished. Output: %SAVE_DIR%
exit /b 0
