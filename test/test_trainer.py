try:
    from ._script_bootstrap import bootstrap_project
except ImportError:
    from _script_bootstrap import bootstrap_project

bootstrap_project(allow_omp_duplicate=True)

import torch
from unittest.mock import MagicMock, call

from src.models.model_pipeline import ModelPipeline
from src.training.trainer import (
    build_model_config,
    load_init_checkpoint,
    resolve_train_files,
    run_one_epoch,
)


def test_resolve_train_files_prefers_explicit_train_files(tmp_path) -> None:
    train_file = tmp_path / "train_a.xlsx"
    skipped_file = tmp_path / "~$temp.xlsx"
    train_dir = tmp_path / "train_dir"
    train_dir.mkdir()
    fallback_file = train_dir / "fallback.xlsx"

    train_file.write_text("a", encoding="utf-8")
    skipped_file.write_text("b", encoding="utf-8")
    fallback_file.write_text("c", encoding="utf-8")

    file_paths = resolve_train_files(
        {
            "train_files": [str(train_file), str(skipped_file)],
            "train_dir": str(train_dir),
        }
    )

    assert file_paths == [train_file]


def test_load_init_checkpoint_only_loads_model_weights(tmp_path) -> None:
    source_model = torch.nn.Linear(3, 2)
    target_model = torch.nn.Linear(3, 2)

    with torch.no_grad():
        source_model.weight.fill_(1.5)
        source_model.bias.fill_(0.25)

    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    resolved_path = load_init_checkpoint(target_model, checkpoint_path)

    assert resolved_path == checkpoint_path
    assert torch.allclose(target_model.weight, source_model.weight)
    assert torch.allclose(target_model.bias, source_model.bias)


def _make_mock_loader():
    """构造一个 mock CaseBatchLoader，只需要 set_epoch / __len__ / get_batch。"""
    import pandas as pd

    loader = MagicMock()
    loader.__len__ = MagicMock(return_value=0)
    loader.set_epoch = MagicMock()
    return loader


def test_train_epoch_calls_set_epoch_with_shuffle():
    """训练阶段 run_one_epoch 应调用 loader.set_epoch(shuffle=True)。"""
    loader = _make_mock_loader()
    model = MagicMock()
    model.train = MagicMock()
    model.parameters = MagicMock(return_value=iter([]))

    # total_steps=0 会触发 RuntimeError，但我们只关心 set_epoch 是否被调用
    try:
        run_one_epoch(
            epoch=1,
            model=model,
            loader=loader,
            static_graph={},
            loss_fn=MagicMock(),
            optimizer=MagicMock(),
            is_train=True,
            shuffle=True,
            random_seed=42,
            grad_clip_norm=None,
            log_every=1,
        )
    except RuntimeError:
        pass  # 预期：所有 batch 被跳过

    loader.set_epoch.assert_called_once_with(
        epoch=1, shuffle=True, random_seed=42,
    )


def test_val_epoch_calls_set_epoch_without_shuffle():
    """验证阶段 run_one_epoch 应调用 loader.set_epoch(shuffle=False)。"""
    loader = _make_mock_loader()
    model = MagicMock()
    model.eval = MagicMock()

    try:
        run_one_epoch(
            epoch=3,
            model=model,
            loader=loader,
            static_graph={},
            loss_fn=MagicMock(),
            optimizer=None,
            is_train=False,
            shuffle=False,
            random_seed=42,
            grad_clip_norm=None,
            log_every=1,
        )
    except RuntimeError:
        pass

    loader.set_epoch.assert_called_once_with(
        epoch=3, shuffle=False, random_seed=42,
    )

def test_build_model_config_readout_passthrough():
    cfg = {
        "model": {
            "hidden_dim": 256,
            "case_refiner": {
                "enabled": True,
                "mlp_hidden_dim": 64,
                "residual": 0.6,
            },
            "readout": {
                "attn_hidden_dim": 128,
                "context_mode": "leave_one_out"
            },
        }
    }
    out = build_model_config(cfg, num_hpo=10)
    assert out["model"]["readout"]["attn_hidden_dim"] == 128
    assert out["model"]["readout"]["context_mode"] == "leave_one_out"
    assert out["model"]["readout"]["hidden_dim"] == 256
    assert out["model"]["case_refiner"]["enabled"] is True
    assert out["model"]["case_refiner"]["hidden_dim"] == 256
    assert out["model"]["case_refiner"]["mlp_hidden_dim"] == 64
    assert out["model"]["case_refiner"]["residual"] == 0.6


def test_load_init_checkpoint_allows_readout_migration(tmp_path) -> None:
    cfg = {
        "model": {
            "hidden_dim": 8,
            "case_refiner": {
                "enabled": True,
                "mlp_hidden_dim": 8,
                "residual": 0.7,
            },
            "readout": {
                "attn_hidden_dim": 8,
                "context_mode": "leave_one_out",
            },
        }
    }
    source_model = ModelPipeline(build_model_config(cfg, num_hpo=5))
    target_model = ModelPipeline(build_model_config(cfg, num_hpo=5))

    with torch.no_grad():
        source_model.encoder.theta0.weight.fill_(1.25)

    checkpoint_state = source_model.state_dict()
    checkpoint_state = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith("readout.attn_mlp.") and not key.startswith("case_refiner.")
    }
    checkpoint_state["readout.attn.weight"] = torch.ones((1, 8), dtype=torch.float32)

    checkpoint_path = tmp_path / "migrated.pt"
    torch.save({"model_state_dict": checkpoint_state}, checkpoint_path)

    resolved_path = load_init_checkpoint(target_model, checkpoint_path)

    assert resolved_path == checkpoint_path
    assert torch.allclose(target_model.encoder.theta0.weight, source_model.encoder.theta0.weight)

