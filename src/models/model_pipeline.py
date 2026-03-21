"""
端到端模型流水线：
dataset -> build_hypergraph -> hgnn_encoder -> readout -> scorer。

该流水线同时维护两种分数视图：
1. scores_local：仅针对疾病列的 logits，形状为 [batch_size, disease_count]
2. scores：带全局列索引的 logits，形状为 [batch_size, batch_size + disease_count]
   其中前 batch_size 列对应 case 超边，这些位置会被填充为一个很大的负数，
   这样就可以直接使用 gold_disease_cols_global 作为监督目标。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hgnn_encoder import HGNNEncoder
from src.models.readout import HyperedgeReadout
from src.models.scorer import CosineScorer


class ModelPipeline(nn.Module):
    """对编码器、读出层、打分器和损失计算的轻量封装。"""

    def __init__(
        self,
        num_hpo: int,
        hidden_dim: int = 128,
        case_padding_value: float = -1e9,
    ) -> None:
        super().__init__()
        self.encoder = HGNNEncoder(num_hpo=num_hpo, hidden_dim=hidden_dim)
        self.readout = HyperedgeReadout()
        self.scorer = CosineScorer()
        self.case_padding_value = float(case_padding_value)

    @staticmethod
    def _num_case(batch_graph: dict[str, Any]) -> int:
        return int(batch_graph["H_case"].shape[1])

    def _build_global_scores(
        self,
        scores_local: torch.Tensor,
        num_case: int,
    ) -> torch.Tensor:
        # 将 case 对应的列补到前面，和全局超图中的列顺序保持一致。
        if num_case <= 0:
            return scores_local
        pad = scores_local.new_full(
            (scores_local.shape[0], num_case),
            self.case_padding_value,
        )
        return torch.cat([pad, scores_local], dim=1)

    def forward(self, batch_graph: dict[str, Any]) -> dict[str, torch.Tensor | list[str]]:
        # H 是完整超图关联矩阵，H_case / H_disease 分别用于聚合 case 和 disease 表示。
        H = batch_graph["H"]
        H_case = batch_graph["H_case"]
        H_disease = batch_graph["H_disease"]
        num_case = self._num_case(batch_graph)

        Z = self.encoder(H)
        readout_out = self.readout(Z, H_case, H_disease)
        case_repr = readout_out["case_repr"]
        disease_repr = readout_out["disease_repr"]

        scores_local = self.scorer(case_repr, disease_repr)["scores"]
        # 在局部分数前拼接 case 列占位，使标签可以直接使用全局列索引。
        scores = self._build_global_scores(scores_local, num_case)

        out: dict[str, torch.Tensor | list[str]] = {
            "Z": Z,
            "case_repr": case_repr,
            "disease_repr": disease_repr,
            "scores_local": scores_local,
            "scores": scores,
            "case_ids": batch_graph["case_ids"],
            "case_labels": batch_graph["case_labels"],
        }

        if "gold_disease_cols_global" in batch_graph:
            gold_global = torch.as_tensor(
                batch_graph["gold_disease_cols_global"],
                dtype=torch.long,
                device=scores.device,
            )
            # 全局列索引减去 case 列数后，才能映射到仅含疾病列的局部索引。
            gold_local = gold_global - num_case
            loss = F.cross_entropy(scores, gold_global)
            out["gold_disease_cols_global"] = gold_global
            out["gold_disease_cols_local"] = gold_local
            out["loss"] = loss

        return out
