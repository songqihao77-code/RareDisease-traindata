@echo off
setlocal

cd /d D:\RareDisease-traindata || goto :fail

set "PYTHON_EXE=D:\python\python.exe"
set "EVAL_CFG=D:\RareDisease-traindata\configs\data_llldataset_eval.yaml"
set "FIXED_PRETRAIN=D:\RareDisease-traindata\outputs\stage1_pretrain_v59\checkpoints\best_fixed_for_weighting_ablation.pt"
set "EXPERIMENT=%~1"
set "DRY_RUN=%~2"

if "%EXPERIMENT%"=="" goto :usage

if /I "%EXPERIMENT%"=="g4a" set "EXPERIMENT=g4a_weighting_sqrt_idf"
if /I "%EXPERIMENT%"=="g4b" set "EXPERIMENT=g4b_weighting_idf"
if /I "%EXPERIMENT%"=="g4c" set "EXPERIMENT=g4c_weighting_power_idf_alpha075"
if /I "%EXPERIMENT%"=="g4b_seed42" set "EXPERIMENT=g4b_weighting_idf_seed42"
if /I "%EXPERIMENT%"=="g4b_seed123" set "EXPERIMENT=g4b_weighting_idf_seed123"
if /I "%EXPERIMENT%"=="g4b_seed3407" set "EXPERIMENT=g4b_weighting_idf_seed3407"
if /I "%EXPERIMENT%"=="g4b_seed42_fixed15" set "EXPERIMENT=g4b_weighting_idf_seed42_fixed15"
if /I "%EXPERIMENT%"=="g4b_seed123_fixed15" set "EXPERIMENT=g4b_weighting_idf_seed123_fixed15"
if /I "%EXPERIMENT%"=="g4b_seed3407_fixed15" set "EXPERIMENT=g4b_weighting_idf_seed3407_fixed15"

if /I "%EXPERIMENT%"=="g4a_weighting_sqrt_idf" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4a_weighting_sqrt_idf.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4a_weighting_sqrt_idf\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4b_weighting_idf" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4b_weighting_idf.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4c_weighting_power_idf_alpha075" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4c_weighting_power_idf_alpha075.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4c_weighting_power_idf_alpha075\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4b_weighting_idf_seed42" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4b_weighting_idf_seed42.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf_seed42\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4b_weighting_idf_seed123" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4b_weighting_idf_seed123.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf_seed123\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4b_weighting_idf_seed3407" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4b_weighting_idf_seed3407.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf_seed3407\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4b_weighting_idf_seed42_fixed15" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4b_weighting_idf_seed42_fixed15.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf_seed42_fixed15\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4b_weighting_idf_seed123_fixed15" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4b_weighting_idf_seed123_fixed15.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf_seed123_fixed15\checkpoints\best.pt"
    goto :run
)

if /I "%EXPERIMENT%"=="g4b_weighting_idf_seed3407_fixed15" (
    set "FINETUNE_CFG=D:\RareDisease-traindata\configs\ablation_case_noise\g4b_weighting_idf_seed3407_fixed15.yaml"
    set "CHECKPOINT_PATH=D:\RareDisease-traindata\outputs\case_noise_ablation\g4b_weighting_idf_seed3407_fixed15\checkpoints\best.pt"
    goto :run
)

echo [ERROR] Unknown experiment: %EXPERIMENT%
goto :usage

:run
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    exit /b 1
)

if not exist "%FIXED_PRETRAIN%" (
    echo [ERROR] Fixed pretrain checkpoint not found: %FIXED_PRETRAIN%
    exit /b 1
)

echo [INFO] experiment=%EXPERIMENT%
echo [INFO] finetune_config=%FINETUNE_CFG%
echo [INFO] fixed_pretrain=%FIXED_PRETRAIN%
echo [INFO] eval_config=%EVAL_CFG%
echo [INFO] best_checkpoint_out=%CHECKPOINT_PATH%

if /I "%DRY_RUN%"=="--dry-run" (
    echo [DRY-RUN] "%PYTHON_EXE%" -m src.training.trainer --config "%FINETUNE_CFG%"
    echo [DRY-RUN] "%PYTHON_EXE%" -m src.evaluation.evaluator --data_config_path "%EVAL_CFG%" --train_config_path "%FINETUNE_CFG%" --checkpoint_path "%CHECKPOINT_PATH%"
    exit /b 0
)

echo [1/2] weighting finetune started...
"%PYTHON_EXE%" -m src.training.trainer --config "%FINETUNE_CFG%"
if errorlevel 1 (
    echo [ERROR] weighting finetune failed.
    exit /b 1
)

echo [2/2] weighting evaluation started...
"%PYTHON_EXE%" -m src.evaluation.evaluator --data_config_path "%EVAL_CFG%" --train_config_path "%FINETUNE_CFG%" --checkpoint_path "%CHECKPOINT_PATH%"
if errorlevel 1 (
    echo [ERROR] weighting evaluation failed.
    exit /b 1
)

echo [DONE] weighting finetune and evaluation completed successfully.
exit /b 0

:usage
echo Usage: run_weighting_ablation.cmd ^<g4a^|g4b^|g4c^|g4b_seed42^|g4b_seed123^|g4b_seed3407^|g4b_seed42_fixed15^|g4b_seed123_fixed15^|g4b_seed3407_fixed15^|g4a_weighting_sqrt_idf^|g4b_weighting_idf^|g4c_weighting_power_idf_alpha075^|g4b_weighting_idf_seed42^|g4b_weighting_idf_seed123^|g4b_weighting_idf_seed3407^|g4b_weighting_idf_seed42_fixed15^|g4b_weighting_idf_seed123_fixed15^|g4b_weighting_idf_seed3407_fixed15^> [--dry-run]
exit /b 1

:fail
echo [ERROR] Script stopped before weighting ablation started.
exit /b 1
