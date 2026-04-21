@echo off
setlocal

cd /d D:\RareDisease-traindata

set CONFIG_A=configs\train_finetune_ab_a_natural.yaml
set CONFIG_B=configs\train_finetune_ab_b_source_balanced.yaml
set EVAL_CONFIG=configs\data_llldataset_eval.yaml
set CKPT_A=D:\RareDisease-traindata\outputs\ab_sampler_compare\A_natural\checkpoints\best.pt
set CKPT_B=D:\RareDisease-traindata\outputs\ab_sampler_compare\B_source_balanced\checkpoints\best.pt

echo [A/4] Group A train started...
python -m src.training.trainer --config %CONFIG_A%
if errorlevel 1 (
    echo [ERROR] Group A training failed.
    exit /b 1
)

echo [A/4] Group A evaluation started...
python -m src.evaluation.evaluator --data_config_path %EVAL_CONFIG% --train_config_path %CONFIG_A% --checkpoint_path %CKPT_A%
if errorlevel 1 (
    echo [ERROR] Group A evaluation failed.
    exit /b 1
)

echo [B/4] Group B train started...
python -m src.training.trainer --config %CONFIG_B%
if errorlevel 1 (
    echo [ERROR] Group B training failed.
    exit /b 1
)

echo [B/4] Group B evaluation started...
python -m src.evaluation.evaluator --data_config_path %EVAL_CONFIG% --train_config_path %CONFIG_B% --checkpoint_path %CKPT_B%
if errorlevel 1 (
    echo [ERROR] Group B evaluation failed.
    exit /b 1
)

echo [DONE] A/B sampler comparison completed successfully.
echo [A] outputs: D:\RareDisease-traindata\outputs\ab_sampler_compare\A_natural
echo [B] outputs: D:\RareDisease-traindata\outputs\ab_sampler_compare\B_source_balanced
exit /b 0
