"""
模型前向流水线。

这个文件只负责把 encoder、readout、scorer 串起来，形成统一前向入口。
当前项目默认使用统一超图 H = [H_case | H_disease]：
前半部分列是病例超边，后半部分列是疾病超边。
这里不负责训练循环、优化器、数据读取，也默认不在 pipeline 中计算 loss。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from src.models.hgnn_encoder import HGNNEncoder
from src.models.readout import HyperedgeReadout

try:
    from src.models.scorer import CosineScorer
except ImportError:
    from src.models.readout import CosineScorer


ENCODER_REGISTRY = {"hgnn": HGNNEncoder}
READOUT_REGISTRY = {"hyperedge": HyperedgeReadout}
SCORER_REGISTRY = {"cosine": CosineScorer}


class ModelPipeline(nn.Module):
    """精简的模型流水线，只做模块组织与结果拼装。"""

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

        self.encoder = encoder or self._build_module("encoder", ENCODER_REGISTRY, "hgnn")
        self.readout = readout or self._build_module("readout", READOUT_REGISTRY, "hyperedge")
        self.scorer = scorer or self._build_module("scorer", SCORER_REGISTRY, "cosine")

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

    def _shape2(self, value: Any, field_name: str) -> tuple[int, int]:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(f"{field_name} 必须是二维矩阵对象，当前收到 {type(value).__name__}。")
        return int(shape[0]), int(shape[1])

    def _check_seq_len(self, values: Any, expected: int, field_name: str) -> None:
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            raise ValueError(f"{field_name} 必须是长度可检查的序列。")
        if len(values) != expected:
            raise ValueError(f"{field_name} 长度应为 {expected}，当前为 {len(values)}。")

    def _validate_batch_graph(self, batch_graph: Mapping[str, Any]) -> tuple[int, int, int]:
        if not isinstance(batch_graph, Mapping):
            raise TypeError(f"batch_graph 必须是 dict 或 Mapping，当前收到 {type(batch_graph).__name__}。")

        required = ("H", "H_case", "H_disease")
        missing = [key for key in required if key not in batch_graph]
        if missing:
            raise KeyError(f"batch_graph 缺少必要字段: {', '.join(missing)}。")

        num_hpo, total_cols = self._shape2(batch_graph["H"], "batch_graph['H']")
        case_hpo, num_case = self._shape2(batch_graph["H_case"], "batch_graph['H_case']")
        disease_hpo, num_disease = self._shape2(batch_graph["H_disease"], "batch_graph['H_disease']")

        if num_hpo != case_hpo or num_hpo != disease_hpo:
            raise ValueError("H、H_case、H_disease 的行数必须对应同一套 HPO 节点空间。")
        if total_cols != num_case + num_disease:
            raise ValueError("统一超图必须满足 H = [H_case | H_disease]，即 H 列数 = case 列数 + disease 列数。")

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

    def _resolve_disease_cols_global(self, batch_graph: Mapping[str, Any], num_case: int, num_disease: int) -> list[int]:
        # 若上游没显式传入全局疾病列索引，就回退到当前统一超图的列顺序约定。
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
        _, total_cols = self._shape2(batch_graph["H"], "batch_graph['H']")
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

    def forward(
        self,
        batch_graph: Mapping[str, Any],
        return_intermediate: bool | None = None,
    ) -> dict[str, Any]:
        """
        统一前向流程：
        1. encoder 对统一超图 H 编码，得到 HPO 节点表示 node_repr
        2. readout 用 H_case / H_disease 聚合出病例表示与疾病表示
        3. scorer 计算病例到疾病的分数矩阵 scores
        """
        num_hpo, num_case, num_disease = self._validate_batch_graph(batch_graph)
        include_intermediate = self.output_config.get("return_intermediate", False)
        if return_intermediate is not None:
            include_intermediate = bool(return_intermediate)

        node_repr = self.encoder(batch_graph["H"])
        if not isinstance(node_repr, torch.Tensor) or node_repr.ndim != 2 or node_repr.shape[0] != num_hpo:
            raise ValueError("encoder 必须返回形状为 [num_hpo, hidden_dim] 的张量。")

        readout_out = self.readout(node_repr, batch_graph["H_case"], batch_graph["H_disease"])
        if not isinstance(readout_out, Mapping) or "case_repr" not in readout_out or "disease_repr" not in readout_out:
            raise ValueError("readout 必须返回包含 case_repr 和 disease_repr 的 dict。")
        case_repr = readout_out["case_repr"]
        disease_repr = readout_out["disease_repr"]

        if case_repr.ndim != 2 or case_repr.shape[0] != num_case:
            raise ValueError("case_repr 的形状应为 [num_case, hidden_dim]。")
        if disease_repr.ndim != 2 or disease_repr.shape[0] != num_disease:
            raise ValueError("disease_repr 的形状应为 [num_disease, hidden_dim]。")

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
            outputs["case_repr"] = case_repr
            outputs["disease_repr"] = disease_repr

        return outputs


__all__ = ["ModelPipeline"]
