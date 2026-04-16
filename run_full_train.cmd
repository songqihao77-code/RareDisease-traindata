@echo off
setlocal

cd /d D:\RareDisease-traindata

echo [1/3] Pretrain started...
python -m src.training.trainer --config configs/train_pretrain.yaml
if errorlevel 1 (
    echo [ERROR] Pretrain failed.
    exit /b 1
)

echo [2/3] Finetune started...
python -m src.training.trainer --config configs/train_finetune.yaml
if errorlevel 1 (
    echo [ERROR] Finetune failed.
    exit /b 1
)

echo [3/3] Evaluation started...
python -m src.evaluation.evaluator --data_config_path configs/data_llldataset_eval.yaml --train_config_path configs/train_finetune.yaml --checkpoint_path D:\RareDisease-traindata\outputs\stage2_finetune\checkpoints\best.pt
if errorlevel 1 (
    echo [ERROR] Evaluation failed.
    exit /b 1
)

echo [DONE] All steps completed successfully.
exit /b 0
