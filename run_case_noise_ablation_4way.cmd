@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d D:\RareDisease-traindata || goto :fail

set "BASE_CFG=D:\RareDisease-traindata\configs\train_finetune.yaml"
set "EVAL_CFG=D:\RareDisease-traindata\configs\data_llldataset_eval.yaml"
set "VARIANT_DIR=D:\RareDisease-traindata\configs\ablation_case_noise"
set "OUT_ROOT=D:\RareDisease-traindata\outputs\case_noise_ablation"
set "PYTHON_EXE=D:\python\python.exe"
set "GEN_SCRIPT=%TEMP%\generate_case_noise_ablation_configs.py"

if not exist "%VARIANT_DIR%" mkdir "%VARIANT_DIR%"
if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    goto :fail
)

echo [1/3] Generate 4 finetune configs...
> "%GEN_SCRIPT%" echo from pathlib import Path
>> "%GEN_SCRIPT%" echo import copy
>> "%GEN_SCRIPT%" echo import yaml
>> "%GEN_SCRIPT%" echo.
>> "%GEN_SCRIPT%" echo root = Path^(r"D:\RareDisease-traindata"^)
>> "%GEN_SCRIPT%" echo base_cfg = root / "configs" / "train_finetune.yaml"
>> "%GEN_SCRIPT%" echo variant_dir = root / "configs" / "ablation_case_noise"
>> "%GEN_SCRIPT%" echo out_root = root / "outputs" / "case_noise_ablation"
>> "%GEN_SCRIPT%" echo variant_dir.mkdir^(parents=True, exist_ok=True^)
>> "%GEN_SCRIPT%" echo out_root.mkdir^(parents=True, exist_ok=True^)
>> "%GEN_SCRIPT%" echo.
>> "%GEN_SCRIPT%" echo with base_cfg.open^("r", encoding="utf-8"^) as f:
>> "%GEN_SCRIPT%" echo     base = yaml.safe_load^(f^)
>> "%GEN_SCRIPT%" echo.
>> "%GEN_SCRIPT%" echo def deep_update^(dst, src^):
>> "%GEN_SCRIPT%" echo     for k, v in src.items^(^):
>> "%GEN_SCRIPT%" echo         if isinstance^(v, dict^) and isinstance^(dst.get^(k^), dict^):
>> "%GEN_SCRIPT%" echo             deep_update^(dst[k], v^)
>> "%GEN_SCRIPT%" echo         else:
>> "%GEN_SCRIPT%" echo             dst[k] = v
>> "%GEN_SCRIPT%" echo.
>> "%GEN_SCRIPT%" echo variants = [
>> "%GEN_SCRIPT%" echo     ^("g1_legacy_like_case_noise_off", {
>> "%GEN_SCRIPT%" echo         "paths": {"save_dir": str^(out_root / "g1_legacy_like_case_noise_off"^)},
>> "%GEN_SCRIPT%" echo         "train": {"hpo_corruption_prob": 0.0, "hpo_dropout_prob": 0.2},
>> "%GEN_SCRIPT%" echo         "case_noise_control": {"enabled": False},
>> "%GEN_SCRIPT%" echo     }^),
>> "%GEN_SCRIPT%" echo     ^("g2_case_noise_on_prune_and_weight", {
>> "%GEN_SCRIPT%" echo         "paths": {"save_dir": str^(out_root / "g2_case_noise_on_prune_and_weight"^)},
>> "%GEN_SCRIPT%" echo         "train": {"hpo_corruption_prob": 0.0, "hpo_dropout_prob": 0.2},
>> "%GEN_SCRIPT%" echo         "case_noise_control": {"enabled": True, "mode": "prune_and_weight"},
>> "%GEN_SCRIPT%" echo     }^),
>> "%GEN_SCRIPT%" echo     ^("g3_case_noise_on_no_dropout", {
>> "%GEN_SCRIPT%" echo         "paths": {"save_dir": str^(out_root / "g3_case_noise_on_no_dropout"^)},
>> "%GEN_SCRIPT%" echo         "train": {"hpo_corruption_prob": 0.0, "hpo_dropout_prob": 0.0},
>> "%GEN_SCRIPT%" echo         "case_noise_control": {"enabled": True, "mode": "prune_and_weight"},
>> "%GEN_SCRIPT%" echo     }^),
>> "%GEN_SCRIPT%" echo     ^("g4_case_noise_weight_only", {
>> "%GEN_SCRIPT%" echo         "paths": {"save_dir": str^(out_root / "g4_case_noise_weight_only"^)},
>> "%GEN_SCRIPT%" echo         "train": {"hpo_corruption_prob": 0.0, "hpo_dropout_prob": 0.2},
>> "%GEN_SCRIPT%" echo         "case_noise_control": {"enabled": True, "mode": "weight_only"},
>> "%GEN_SCRIPT%" echo     }^),
>> "%GEN_SCRIPT%" echo ]
>> "%GEN_SCRIPT%" echo.
>> "%GEN_SCRIPT%" echo for name, patch in variants:
>> "%GEN_SCRIPT%" echo     cfg = copy.deepcopy^(base^)
>> "%GEN_SCRIPT%" echo     deep_update^(cfg, patch^)
>> "%GEN_SCRIPT%" echo     out_path = variant_dir / f"{name}.yaml"
>> "%GEN_SCRIPT%" echo     with out_path.open^("w", encoding="utf-8"^) as f:
>> "%GEN_SCRIPT%" echo         yaml.safe_dump^(cfg, f, sort_keys=False, allow_unicode=True^)
>> "%GEN_SCRIPT%" echo     print^(f"[CFG] {out_path}"^)
if errorlevel 1 goto :fail
"%PYTHON_EXE%" "%GEN_SCRIPT%"
if errorlevel 1 goto :fail
if /I "%~1"=="--prepare-only" goto :prepare_done

for %%G in (
    g1_legacy_like_case_noise_off
    g2_case_noise_on_prune_and_weight
    g3_case_noise_on_no_dropout
    g4_case_noise_weight_only
) do (
    echo.
    echo ============================================================
    echo [2/3] %%G - finetune
    "%PYTHON_EXE%" -m src.training.trainer --config "%VARIANT_DIR%\%%G.yaml"
    if errorlevel 1 goto :fail

    echo [3/3] %%G - evaluate
    "%PYTHON_EXE%" -m src.evaluation.evaluator --data_config_path "%EVAL_CFG%" --train_config_path "%VARIANT_DIR%\%%G.yaml" --checkpoint_path "%OUT_ROOT%\%%G\checkpoints\best.pt"
    if errorlevel 1 goto :fail
)

echo.
echo [DONE] All 4 experiments finished successfully.
pause
exit /b 0

:prepare_done
echo.
echo [DONE] Config generation finished successfully.
echo [INFO] Prepare-only mode: no training/evaluation started.
pause
exit /b 0

:fail
echo.
echo [ERROR] Script stopped. Check the log above.
pause
exit /b 1
