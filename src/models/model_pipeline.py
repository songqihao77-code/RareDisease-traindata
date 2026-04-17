"""统一模型前向流水线。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn

from src.models.case_refiner import CaseConditionedRefiner, RefinedCaseNodeState
from src.models.hgnn_encoder import HGNNEncoder
from src.models.readout import HyperedgeReadout
from src.models.scorer import CosineScorer


ENCODER_REGISTRY = {"hgnn": HGNNEncoder}
CASE_REFINER_REGISTRY = {"case_conditioned": CaseConditionedRefiner}
READOUT_REGISTRY = {"hyperedge": HyperedgeReadout}
SCORER_REGISTRY = {"cosine": CosineScorer}


class ModelPipeline(nn.Module):
    """只负责组织 encoder / readout / scorer，不处理训练循环。"""

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        encoder: nn.Module | None = None,
        readout: nn.Module | None = None,
        scorer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = self._normalize_config(config)
        self.output_config = self._get_block("outputs")
        self.case_weighting_config = self._normalize_case_weighting_config()

        self.encoder = encoder or self._build_module("encoder", ENCODER_REGISTRY, "hgnn")
        self.case_refiner = self._build_optional_case_refiner()
        self.readout = readout or self._build_module("readout", READOUT_REGISTRY, "hyperedge")
        self.scorer = scorer or self._build_module("scorer", SCORER_REGISTRY, "cosine")

        # 静态 H_disease 在训练/评估中会被反复使用。
        # 这里缓存其 torch sparse 形式，避免每个 batch 都从 scipy sparse 重新转换。
        self._cached_h_disease_src_id: int | None = None
        self._cached_h_disease_device: torch.device | None = None
        self._cached_h_disease_sparse: torch.Tensor | None = None
        self._cached_case_ic_src_id: int | None = None
        self._cached_case_ic_device: torch.device | None = None
        self._cached_case_ic_dtype: torch.dtype | None = None
        self._cached_case_ic_weight: torch.Tensor | None = None

    def _normalize_config(self, config: Mapping[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        if not isinstance(config, Mapping):
            raise TypeError(f"config 必须是 dict 或 Mapping，当前收到 {type(config).__name__}。")
        if isinstance(config.get("model"), Mapping):
            return dict(config["model"])
        return dict(config)

    def _get_block(self, name: str) -> dict[str, Any]:
        block = self.config.get(name, {})
        if block is None:
            return {}
        if not isinstance(block, Mapping):
            raise TypeError(f"config.{name} 必须是 Mapping，当前收到 {type(block).__name__}。")
        return dict(block)

    def _normalize_case_weighting_config(self) -> dict[str, Any]:
        block = self._get_block("case_weighting")
        config = {
            "enabled": bool(block.get("enabled", False)),
            "mode": str(block.get("mode", "fixed_ic")),
            "ic_source": str(block.get("ic_source", "h_disease_binary_df")),
            "eps": float(block.get("eps", 1e-8)),
            "clip_quantile_low": float(block.get("clip_quantile_low", 0.05)),
            "clip_quantile_high": float(block.get("clip_quantile_high", 0.95)),
            "weight_min": float(block.get("weight_min", 0.5)),
            "weight_max": float(block.get("weight_max", 1.5)),
            "zero_support_policy": str(block.get("zero_support_policy", "identity")),
        }

        if not config["enabled"]:
            return config

        if config["mode"] != "fixed_ic":
            raise ValueError(f"case_weighting.mode 当前只支持 'fixed_ic'，收到 {config['mode']!r}。")
        if config["ic_source"] != "h_disease_binary_df":
            raise ValueError(
                "case_weighting.ic_source 当前只支持 'h_disease_binary_df'，"
                f"收到 {config['ic_source']!r}。"
            )
        if config["eps"] <= 0:
            raise ValueError(f"case_weighting.eps 必须大于 0，当前为 {config['eps']}。")
        if not 0.0 <= config["clip_quantile_low"] <= 1.0:
            raise ValueError("case_weighting.clip_quantile_low 必须位于 [0, 1]。")
        if not 0.0 <= config["clip_quantile_high"] <= 1.0:
            raise ValueError("case_weighting.clip_quantile_high 必须位于 [0, 1]。")
        if config["clip_quantile_low"] > config["clip_quantile_high"]:
            raise ValueError("case_weighting.clip_quantile_low 不能大于 clip_quantile_high。")
        if config["weight_min"] > config["weight_max"]:
            raise ValueError("case_weighting.weight_min 不能大于 weight_max。")
        if config["zero_support_policy"] != "identity":
            raise ValueError(
                "case_weighting.zero_support_policy 当前只支持 'identity'，"
                f"收到 {config['zero_support_policy']!r}。"
            )
        return config

    def _build_module(
        self,
        name: str,
        registry: dict[str, type[nn.Module]],
        default_type: str,
    ) -> nn.Module:
        block = self._get_block(name)
        module_type = block.get("type", default_type)
        if module_type not in registry:
            raise ValueError(f"未知的 {name} 类型 {module_type!r}，可选值: {list(registry)}。")

        params = {k: v for k, v in block.items() if k not in {"type", "params"}}
        params.update(block.get("params", {}))
        return registry[module_type](**params)

    def _build_optional_case_refiner(self) -> nn.Module | None:
        block = self._get_block("case_refiner")
        if not block:
            return None

        enabled = block.get("enabled")
        if enabled is None:
            enabled = block.get("use_case_refiner", False)
        if not bool(enabled):
            return None

        module_type = block.get("type", "case_conditioned")
        if module_type not in CASE_REFINER_REGISTRY:
            raise ValueError(
                f"未知的 case_refiner 类型 {module_type!r}，可选值: {list(CASE_REFINER_REGISTRY)}。"
            )

        params = {
            k: v
            for k, v in block.items()
            if k not in {"type", "params", "enabled", "use_case_refiner"}
        }
        params.update(block.get("params", {}))
        return CASE_REFINER_REGISTRY[module_type](**params)

    def _shape2(self, value: Any, field_name: str) -> tuple[int, int]:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(f"{field_name} 必须是二维矩阵，当前收到 {type(value).__name__}。")
        return int(shape[0]), int(shape[1])

    def _check_seq_len(self, values: Any, expected: int, field_name: str) -> None:
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            raise ValueError(f"{field_name} 必须是可检查长度的序列。")
        if len(values) != expected:
            raise ValueError(f"{field_name} 长度应为 {expected}，当前为 {len(values)}。")

    def _validate_batch_graph(self, batch_graph: Mapping[str, Any]) -> tuple[int, int, int]:
        """最小但必要的输入校验。

        `H` 现在允许缺省：
        - 训练/评估热路径可以不再构造 `H = [H_case | H_disease]`
        - 若上游确实提供了 `H`，仍保留原有形状校验
        """
        if not isinstance(batch_graph, Mapping):
            raise TypeError(f"batch_graph 必须是 dict 或 Mapping，当前收到 {type(batch_graph).__name__}。")

        required = ("H_case", "H_disease")
        missing = [key for key in required if key not in batch_graph]
        if missing:
            raise KeyError(f"batch_graph 缺少必要字段: {', '.join(missing)}。")

        case_hpo, num_case = self._shape2(batch_graph["H_case"], "batch_graph['H_case']")
        disease_hpo, num_disease = self._shape2(batch_graph["H_disease"], "batch_graph['H_disease']")
        if case_hpo != disease_hpo:
            raise ValueError("H_case 与 H_disease 的行数必须对应同一组 HPO 节点。")

        if "H" in batch_graph:
            num_hpo, total_cols = self._shape2(batch_graph["H"], "batch_graph['H']")
            if num_hpo != case_hpo:
                raise ValueError("H、H_case、H_disease 的行数必须一致。")
            if total_cols != num_case + num_disease:
                raise ValueError("若提供 H，则必须满足 H = [H_case | H_disease]。")
        else:
            num_hpo = case_hpo

        if "case_ids" in batch_graph:
            self._check_seq_len(batch_graph["case_ids"], num_case, "batch_graph['case_ids']")
        if "case_labels" in batch_graph:
            self._check_seq_len(batch_graph["case_labels"], num_case, "batch_graph['case_labels']")
        if "gold_disease_cols_global" in batch_graph:
            self._check_seq_len(
                batch_graph["gold_disease_cols_global"],
                num_case,
                "batch_graph['gold_disease_cols_global']",
            )
        if "disease_cols_global" in batch_graph:
            self._check_seq_len(
                batch_graph["disease_cols_global"],
                num_disease,
                "batch_graph['disease_cols_global']",
            )

        return num_hpo, num_case, num_disease

    def _get_model_device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_h_disease(self, H_disease, device: torch.device) -> torch.Tensor:
        """把静态 disease incidence 转为当前 device 上的 torch sparse，并做缓存。"""
        if isinstance(H_disease, torch.Tensor):
            tensor = H_disease.float().to(device)
            return tensor.coalesce() if tensor.is_sparse else tensor.to_sparse().coalesce()

        if scipy.sparse.issparse(H_disease):
            if (
                self._cached_h_disease_sparse is not None
                and self._cached_h_disease_src_id == id(H_disease)
                and self._cached_h_disease_device == device
            ):
                return self._cached_h_disease_sparse

            H_coo = H_disease.tocoo()
            idx = torch.tensor(np.vstack([H_coo.row, H_coo.col]), dtype=torch.long, device=device)
            val = torch.tensor(H_coo.data, dtype=torch.float32, device=device)
            tensor = torch.sparse_coo_tensor(idx, val, H_coo.shape, device=device).coalesce()
            self._cached_h_disease_src_id = id(H_disease)
            self._cached_h_disease_device = device
            self._cached_h_disease_sparse = tensor
            return tensor

        raise TypeError(f"H_disease 只支持 scipy.sparse 或 torch.Tensor，当前为 {type(H_disease).__name__}。")

    def _build_fixed_ic_weights(self, H_disease) -> np.ndarray:
        """基于 H_disease 的二值 disease frequency 构造固定 IC 权重。"""
        num_hpo, num_disease = self._shape2(H_disease, "H_disease")
        if num_disease <= 0:
            raise ValueError("H_disease 的列数必须大于 0。")

        if scipy.sparse.issparse(H_disease):
            H_coo = H_disease.tocoo()
            positive_mask = H_coo.data > 0
            positive_rows = H_coo.row[positive_mask]
            df = np.bincount(positive_rows, minlength=num_hpo).astype(np.float64, copy=False)
        elif isinstance(H_disease, torch.Tensor):
            tensor = H_disease
            if tensor.is_sparse:
                tensor = tensor.coalesce()
                rows = tensor.indices()[0][tensor.values() > 0]
                df = (
                    torch.bincount(rows, minlength=num_hpo)
                    .to(dtype=torch.float64)
                    .cpu()
                    .numpy()
                )
            else:
                df = (tensor > 0).sum(dim=1).to(dtype=torch.float64).cpu().numpy()
        else:
            raise TypeError(f"H_disease 只支持 scipy.sparse 或 torch.Tensor，当前为 {type(H_disease).__name__}。")

        weights = np.ones(num_hpo, dtype=np.float32)
        support_mask = df > 0
        if not support_mask.any():
            return weights

        eps = float(self.case_weighting_config["eps"])
        p = (df[support_mask] + eps) / (float(num_disease) + eps)
        ic_raw = -np.log(p)

        q_low = float(self.case_weighting_config["clip_quantile_low"])
        q_high = float(self.case_weighting_config["clip_quantile_high"])
        clip_low = float(np.quantile(ic_raw, q_low))
        clip_high = float(np.quantile(ic_raw, q_high))
        clipped = np.clip(ic_raw, clip_low, clip_high)

        clipped_min = float(clipped.min())
        clipped_max = float(clipped.max())
        if clipped_max - clipped_min <= eps:
            normalized = np.full_like(clipped, 0.5, dtype=np.float64)
        else:
            normalized = (clipped - clipped_min) / (clipped_max - clipped_min)

        weight_min = float(self.case_weighting_config["weight_min"])
        weight_max = float(self.case_weighting_config["weight_max"])
        weights[support_mask] = (
            weight_min + normalized * (weight_max - weight_min)
        ).astype(np.float32, copy=False)

        if self.case_weighting_config["zero_support_policy"] == "identity":
            weights[~support_mask] = 1.0
        return weights

    def _get_case_ic_weight(
        self,
        H_disease,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.case_weighting_config["enabled"]:
            return None

        src_id = id(H_disease)
        if (
            self._cached_case_ic_weight is not None
            and self._cached_case_ic_src_id == src_id
            and self._cached_case_ic_device == device
            and self._cached_case_ic_dtype == dtype
        ):
            return self._cached_case_ic_weight

        weights_np = self._build_fixed_ic_weights(H_disease)
        weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
        self._cached_case_ic_src_id = src_id
        self._cached_case_ic_device = device
        self._cached_case_ic_dtype = dtype
        self._cached_case_ic_weight = weights
        return weights

    def _build_weighted_h_case(
        self,
        H_case,
        ic_weight: torch.Tensor | None,
        device: torch.device,
    ):
        """只在前向里临时缩放 H_case.active edge 的 values。"""
        if ic_weight is None:
            return H_case

        if scipy.sparse.issparse(H_case):
            H_coo = H_case.tocoo()
            idx = torch.tensor(np.vstack([H_coo.row, H_coo.col]), dtype=torch.long, device=device)
            val = torch.tensor(H_coo.data, dtype=ic_weight.dtype, device=device)
            weighted_val = val * ic_weight.index_select(0, idx[0])
            return torch.sparse_coo_tensor(idx, weighted_val, H_coo.shape, device=device).coalesce()

        if isinstance(H_case, torch.Tensor):
            tensor = H_case.to(device=device)
            if tensor.is_sparse:
                tensor = tensor.coalesce()
                indices = tensor.indices()
                values = tensor.values().to(dtype=ic_weight.dtype)
                weighted_val = values * ic_weight.index_select(0, indices[0]).to(dtype=values.dtype)
                return torch.sparse_coo_tensor(
                    indices,
                    weighted_val,
                    tensor.shape,
                    device=device,
                ).coalesce()

            tensor = tensor.to(dtype=ic_weight.dtype)
            scale = ic_weight.unsqueeze(1)
            return tensor * scale

        raise TypeError(f"H_case 只支持 scipy.sparse 或 torch.Tensor，当前为 {type(H_case).__name__}。")

    def _resolve_disease_cols_global(
        self,
        batch_graph: Mapping[str, Any],
        num_case: int,
        num_disease: int,
    ) -> list[int]:
        cols = batch_graph.get("disease_cols_global")
        if cols is None:
            return list(range(num_case, num_case + num_disease))
        return [int(i) for i in cols]

    def _build_scores_global(
        self,
        scores: torch.Tensor,
        batch_graph: Mapping[str, Any],
        num_case: int,
        num_disease: int,
    ) -> torch.Tensor:
        # 如果上游没有构造 H，就退回到最小必要列数。
        if "H" in batch_graph:
            _, total_cols = self._shape2(batch_graph["H"], "batch_graph['H']")
        else:
            total_cols = num_case + num_disease

        disease_cols_global = self._resolve_disease_cols_global(batch_graph, num_case, num_disease)
        if len(disease_cols_global) != num_disease:
            raise ValueError("disease_cols_global 长度必须与疾病数一致。")

        padding_value = self.output_config.get("case_padding_value")
        if padding_value is None:
            padding_value = torch.finfo(scores.dtype).min

        scores_global = scores.new_full((scores.shape[0], total_cols), padding_value)
        scores_global[:, torch.as_tensor(disease_cols_global, dtype=torch.long, device=scores.device)] = scores
        return scores_global

    def _build_gold_local(
        self,
        batch_graph: Mapping[str, Any],
        device: torch.device,
        num_case: int,
        num_disease: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gold_global = [int(i) for i in batch_graph["gold_disease_cols_global"]]
        disease_cols_global = self._resolve_disease_cols_global(batch_graph, num_case, num_disease)
        global_to_local = {col: idx for idx, col in enumerate(disease_cols_global)}

        try:
            gold_local = [global_to_local[col] for col in gold_global]
        except KeyError as exc:
            raise ValueError("gold_disease_cols_global 中存在不属于疾病列空间的索引。") from exc

        return (
            torch.as_tensor(gold_global, dtype=torch.long, device=device),
            torch.as_tensor(gold_local, dtype=torch.long, device=device),
        )

    def _validate_refined_case_node_repr(
        self,
        refined_case_node_repr: torch.Tensor | RefinedCaseNodeState,
        num_hpo: int,
        num_case: int,
        hidden_dim: int,
    ) -> None:
        if isinstance(refined_case_node_repr, RefinedCaseNodeState):
            if refined_case_node_repr.shape != (num_hpo, num_case, hidden_dim):
                raise ValueError(
                    "case_refiner 返回的 RefinedCaseNodeState 形状不正确，"
                    f"得到 {refined_case_node_repr.shape}，期望 {(num_hpo, num_case, hidden_dim)}。"
                )
            return

        if (
            not isinstance(refined_case_node_repr, torch.Tensor)
            or refined_case_node_repr.ndim != 3
            or refined_case_node_repr.shape != (num_hpo, num_case, hidden_dim)
        ):
            raise ValueError("case_refiner 必须返回形状为 [num_hpo, num_case, hidden_dim] 的结果。")

    def precompute_disease_side(self, H_disease) -> dict[str, torch.Tensor]:
        """在评估阶段预计算静态 disease side。

        这是严格等价的缓存：
        - eval pass 内模型参数固定
        - H_disease 固定
        因此 `encoder(H_disease)` 与 `build_disease_repr(H_disease^T @ Z)` 只需算一次。
        """
        device = self._get_model_device()
        prepared_h_disease = self._prepare_h_disease(H_disease, device)
        self._get_case_ic_weight(H_disease, device=device, dtype=prepared_h_disease.dtype)
        node_repr = self.encoder(prepared_h_disease)
        disease_repr = self.readout.build_disease_repr(node_repr, prepared_h_disease)
        return {
            "node_repr": node_repr,
            "disease_repr": disease_repr,
        }

    def forward(
        self,
        batch_graph: Mapping[str, Any],
        return_intermediate: bool | None = None,
        node_repr_override: torch.Tensor | None = None,
        disease_repr_override: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        num_hpo, num_case, num_disease = self._validate_batch_graph(batch_graph)
        include_intermediate = self.output_config.get("return_intermediate", False)
        if return_intermediate is not None:
            include_intermediate = bool(return_intermediate)

        device = self._get_model_device()
        prepared_h_disease = None
        if node_repr_override is None or disease_repr_override is None:
            prepared_h_disease = self._prepare_h_disease(batch_graph["H_disease"], device)

        if node_repr_override is None:
            node_repr = self.encoder(prepared_h_disease)
        else:
            node_repr = node_repr_override
        if not isinstance(node_repr, torch.Tensor) or node_repr.ndim != 2 or node_repr.shape[0] != num_hpo:
            raise ValueError("encoder 必须返回形状为 [num_hpo, hidden_dim] 的张量。")

        weighted_h_case = self._build_weighted_h_case(
            batch_graph["H_case"],
            ic_weight=self._get_case_ic_weight(
                batch_graph["H_disease"],
                device=device,
                dtype=node_repr.dtype,
            ),
            device=device,
        )

        refined_case_node_repr: torch.Tensor | RefinedCaseNodeState | None = None
        if self.case_refiner is not None:
            refined_case_node_repr = self.case_refiner(node_repr, weighted_h_case)
            self._validate_refined_case_node_repr(
                refined_case_node_repr=refined_case_node_repr,
                num_hpo=num_hpo,
                num_case=num_case,
                hidden_dim=node_repr.shape[1],
            )

        readout_out = self.readout(
            node_repr,
            weighted_h_case,
            prepared_h_disease if prepared_h_disease is not None else batch_graph["H_disease"],
            refined_case_node_repr=refined_case_node_repr,
            disease_repr_override=disease_repr_override,
        )
        if not isinstance(readout_out, Mapping) or "case_repr" not in readout_out or "disease_repr" not in readout_out:
            raise ValueError("readout 必须返回包含 case_repr 和 disease_repr 的 dict。")

        case_repr = readout_out["case_repr"]
        disease_repr = readout_out["disease_repr"]
        if case_repr.ndim != 2 or case_repr.shape[0] != num_case:
            raise ValueError("case_repr 的形状应为 [num_case, hidden_dim]。")
        if disease_repr.ndim != 2 or disease_repr.shape[0] != num_disease:
            raise ValueError("disease_repr 的形状应为 [num_disease, hidden_dim]。")

        gold_global = None
        gold_local = None
        if "gold_disease_cols_global" in batch_graph:
            gold_global, gold_local = self._build_gold_local(
                batch_graph=batch_graph,
                device=case_repr.device,
                num_case=num_case,
                num_disease=num_disease,
            )

        scorer_out = self.scorer(case_repr, disease_repr)
        if not isinstance(scorer_out, Mapping) or "scores" not in scorer_out:
            raise ValueError("scorer 必须返回包含 scores 的 dict。")
        scores = scorer_out["scores"]
        if not isinstance(scores, torch.Tensor) or scores.shape != (num_case, num_disease):
            raise ValueError("scores 的形状应为 [num_case, num_disease]。")

        outputs: dict[str, Any] = {"scores": scores}

        if self.output_config.get("include_global_scores", False):
            outputs["scores_global"] = self._build_scores_global(scores, batch_graph, num_case, num_disease)

        if self.output_config.get("include_metadata", True):
            for key in ("case_ids", "case_labels", "case_cols_global", "disease_cols_global"):
                if key in batch_graph:
                    outputs[key] = batch_graph[key]
            if "gold_disease_cols_global" in batch_graph:
                if gold_global is None or gold_local is None:
                    gold_global, gold_local = self._build_gold_local(
                        batch_graph=batch_graph,
                        device=scores.device,
                        num_case=num_case,
                        num_disease=num_disease,
                    )
                outputs["gold_disease_cols_global"] = gold_global
                outputs["gold_disease_cols_local"] = gold_local

        if include_intermediate:
            outputs["node_repr"] = node_repr
            if refined_case_node_repr is not None:
                outputs["refined_case_node_repr"] = (
                    refined_case_node_repr.to_dense()
                    if isinstance(refined_case_node_repr, RefinedCaseNodeState)
                    else refined_case_node_repr
                )
            outputs["case_repr"] = case_repr
            outputs["disease_repr"] = disease_repr
            for key, val in readout_out.items():
                if key not in {"case_repr", "disease_repr"}:
                    outputs[key] = val

        return outputs


__all__ = ["ModelPipeline"]
