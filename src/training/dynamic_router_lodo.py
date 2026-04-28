import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


# ==============================================================================
# 1. 极简 Meta-feature Router 网络
# ==============================================================================
class DynamicMetaRouter(nn.Module):
    """
    轻量级的自适应路由网络。
    输入：患者维度的抽象元特征 (例如：HPO数量，平均IC，系统分支数等)
    输出：证据特征（如HGNN, Semantic IC, Case Cov等）的Softmax融合权重
    """

    def __init__(self, meta_feat_dim: int = 7, hidden_dim: int = 16, num_evidence: int = 5, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(meta_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_evidence)
        )

    def forward(self, meta_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            meta_features: [batch_size, meta_feat_dim]
        Returns:
            weights: [batch_size, num_evidence] (经过softmax归一化的权重)
        """
        logits = self.mlp(meta_features)
        weights = F.softmax(logits, dim=-1)
        return weights


# ==============================================================================
# 2. Pairwise Margin Ranking Loss (针对 Hard Negatives)
# ==============================================================================
class PairwiseMarginRankingLoss(nn.Module):
    """
    用于优化诊断候选排序的 Pairwise Loss。
    目标：确保 Gold Disease 的融合得分比 Hard Negative 疾病的得分高出一个 margin。
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_scores: [batch_size, 1] (Gold disease 的最终融合得分)
            neg_scores: [batch_size, num_negatives] (Hard negatives 的融合得分)
        Returns:
            loss: 标量
        """
        # 利用广播机制计算所有正负样本对的 hinge loss
        # max(0, neg_score - pos_score + margin)
        differences = neg_scores - pos_scores + self.margin
        loss = F.relu(differences)
        # 取所有 hard negatives 对的均值
        return loss.mean()


# ==============================================================================
# 3. LODO (Leave-One-Dataset-Out) 训练与评估骨架
# ==============================================================================
class LODOTrainer:
    def __init__(self, datasets: List[str], meta_feat_dim: int, num_evidence: int):
        self.datasets = datasets
        self.meta_feat_dim = meta_feat_dim
        self.num_evidence = num_evidence

    def build_mock_dataloader(self, dataset_names: List[str], batch_size: int = 32):
        """
        【请替换为您本地真实的 DataLoader】
        为了演示，这里生成 Mock 数据。在实际应用中，您需要将对应数据集的数据合并。
        返回的数据格式应为：
        - meta_features: [batch_size, meta_feat_dim]
        - pos_evidence: [batch_size, num_evidence] (Gold disease的特征分数)
        - neg_evidence: [batch_size, num_negatives, num_evidence] (Hard negatives的特征分数)
        """
        # 生成随机数据模拟
        num_batches = 10
        for _ in range(num_batches):
            meta_features = torch.randn(batch_size, self.meta_feat_dim)
            pos_evidence = torch.rand(batch_size, self.num_evidence)
            neg_evidence = torch.rand(batch_size, 5, self.num_evidence)  # 5个hard negatives
            yield meta_features, pos_evidence, neg_evidence

    def compute_fused_scores(self, weights: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        """
        利用生成的权重对证据进行线性融合。
        Args:
            weights: [batch_size, num_evidence]
            evidence: [batch_size, num_evidence] 或 [batch_size, num_negatives, num_evidence]
        """
        if evidence.dim() == 2:
            return torch.sum(weights * evidence, dim=-1, keepdim=True)  # [batch_size, 1]
        elif evidence.dim() == 3:
            # weights 增加一个维度以适配 num_negatives: [batch_size, 1, num_evidence]
            return torch.sum(weights.unsqueeze(1) * evidence, dim=-1)  # [batch_size, num_negatives]

    def train_fold(self, test_dataset: str, epochs: int = 5, lr: float = 1e-3):
        print(f"\n{'=' * 50}\nStarting LODO Fold - Holdout Test Set: {test_dataset}\n{'=' * 50}")

        # 1. 划分训练集和测试集
        train_datasets = [d for d in self.datasets if d != test_dataset]
        print(f"Training on pooled datasets: {train_datasets}")

        # 2. 初始化模型与优化器
        model = DynamicMetaRouter(self.meta_feat_dim, hidden_dim=16, num_evidence=self.num_evidence)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = PairwiseMarginRankingLoss(margin=0.1)

        # 3. 训练循环
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            weight_vars = []

            # 【此处替换为真实的 train_dataloader】
            train_loader = self.build_mock_dataloader(train_datasets)

            for batch_idx, (meta_features, pos_evidence, neg_evidence) in enumerate(train_loader):
                optimizer.zero_grad()

                # 前向传播：获取动态权重
                weights = model(meta_features)

                # 监控指标：计算 Weight Variance 防坍塌 (Collapse)
                # 计算 batch 内同一证据权重的方差，如果接近0说明模型在输出常数静态权重
                batch_weight_variance = torch.var(weights, dim=0).mean().item()
                weight_vars.append(batch_weight_variance)

                # 融合得分
                pos_scores = self.compute_fused_scores(weights, pos_evidence)
                neg_scores = self.compute_fused_scores(weights, neg_evidence)

                # 计算 Pairwise Loss 并反向传播
                loss = criterion(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (batch_idx + 1)
            avg_var = np.mean(weight_vars)

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Avg Weight Variance: {avg_var:.6f}")
            if avg_var < 1e-5:
                print(
                    "⚠️ WARNING: Weight variance is extremely low. The router might have collapsed to static weights!")

        # 4. 评估循环 (在 holdout 数据集上测试)
        print(f"\n--- Evaluating on Unseen Target Dataset: {test_dataset} ---")
        model.eval()
        with torch.no_grad():
            # 【此处替换为真实的 test_dataloader，输出预测排名】
            test_loader = self.build_mock_dataloader([test_dataset], batch_size=16)
            test_weights = []
            for meta_features, _, _ in test_loader:
                weights = model(meta_features)
                test_weights.append(weights)

            final_weights = torch.cat(test_weights, dim=0)
            print(f"Test Set Average Weights: {final_weights.mean(dim=0).numpy()}")
            print(f"Test Set Weight Variance: {torch.var(final_weights, dim=0).mean().item():.6f}")
            print("Evaluation Complete. (Integration with exact Top-K metrics required locally)")

        return model

    def run_all_folds(self):
        models = {}
        for test_dataset in self.datasets:
            models[test_dataset] = self.train_fold(test_dataset)
        return models


if __name__ == "__main__":
    # 配置参数
    DATASETS = ["DDD", "MIMIC-IV-Rare", "HMS", "LIRICAL", "MME"]
    META_FEAT_DIM = 7  # 患者元特征维度数
    NUM_EVIDENCE = 5  # 待融合的候选证据特征数 (如 HGNN, IC_overlap 等)

    # 启动全量 LODO 实验
    trainer = LODOTrainer(DATASETS, META_FEAT_DIM, NUM_EVIDENCE)
    trainer.run_all_folds()