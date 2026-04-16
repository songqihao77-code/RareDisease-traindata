import os
import sys
import unittest

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.loss_builder import FullPoolCrossEntropyLoss, build_loss


class TestLossBuilder(unittest.TestCase):
    # 正常前向：检查返回字段、shape 和关键统计量
    def test_full_pool_ce_forward_basic(self):
        scores = torch.tensor(
            [
                [1.0, 3.0, 2.0, -1.0],
                [0.5, 0.1, 0.2, 0.0],
                [-1.0, 0.0, 2.5, 2.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([1, 0, 3], dtype=torch.long)

        loss_fn = build_loss("full_pool_ce", temperature=1.0, reduction="mean")
        with torch.no_grad():
            out = loss_fn(scores, targets)

        pred = scores.argmax(dim=1)
        correct = pred == targets
        batch_acc = correct.float().mean()
        target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)

        self.assertIn("loss", out)
        self.assertIn("pred", out)
        self.assertIn("correct", out)
        self.assertIn("batch_acc", out)
        self.assertIn("target_scores", out)

        self.assertEqual(out["pred"].shape, (3,))
        self.assertEqual(out["correct"].shape, (3,))
        self.assertEqual(out["target_scores"].shape, (3,))
        self.assertEqual(out["loss"].ndim, 0)

        self.assertTrue(torch.equal(out["pred"], pred))
        self.assertTrue(torch.equal(out["correct"], correct))
        self.assertTrue(torch.allclose(out["batch_acc"], batch_acc))
        self.assertTrue(torch.allclose(out["target_scores"], target_scores))

        print("1. 正常前向测试")
        print("scores=\n", scores)
        print("targets=", targets)
        print("pred=", out["pred"])
        print("correct=", out["correct"])
        print("batch_acc=", out["batch_acc"])
        print("target_scores=", out["target_scores"])
        print("loss=", out["loss"])

    # 与官方 cross_entropy 对齐
    def test_full_pool_ce_matches_torch_ce(self):
        scores = torch.tensor(
            [
                [2.0, 0.0, -1.0, 3.0],
                [0.1, 0.2, 1.8, -0.5],
                [1.0, 2.5, 0.3, 0.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([3, 2, 1], dtype=torch.long)

        loss_fn = build_loss("full_pool_ce", temperature=1.0, reduction="mean")
        with torch.no_grad():
            loss = loss_fn(scores, targets)["loss"]
            expected = F.cross_entropy(scores, targets, reduction="mean")

        diff = (loss - expected).abs()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6, rtol=1e-6))

        print("2. 与官方 CE 对齐测试")
        print("module_loss=", loss)
        print("torch_ce_loss=", expected)
        print("abs_diff=", diff)

    # temperature 改变后，loss 应发生变化
    def test_full_pool_ce_temperature_effect(self):
        scores = torch.tensor(
            [
                [1.2, 0.4, -0.7, 2.1],
                [0.3, 1.5, 0.8, -0.2],
                [2.0, 0.1, 1.0, -1.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([3, 1, 2], dtype=torch.long)

        with torch.no_grad():
            loss_t1 = build_loss("full_pool_ce", temperature=1.0, reduction="mean")(scores, targets)["loss"]
            loss_t05 = build_loss("full_pool_ce", temperature=0.5, reduction="mean")(scores, targets)["loss"]

        self.assertFalse(torch.allclose(loss_t1, loss_t05))

        print("3. temperature 生效测试")
        print("scores=\n", scores)
        print("loss_t1=", loss_t1)
        print("loss_t0_5=", loss_t05)

    # build_loss 应返回正确实例
    def test_build_loss_returns_correct_instance(self):
        loss_fn = build_loss("full_pool_ce")

        self.assertIsInstance(loss_fn, FullPoolCrossEntropyLoss)

        print("4. build_loss 接口测试")
        print("loss_fn_type=", type(loss_fn).__name__)

    # 非法 loss_name 应明确报错
    def test_build_loss_invalid_name(self):
        with self.assertRaises(ValueError) as cm:
            build_loss("xxx")

        print("5. 非法 loss_name 测试")
        print("invalid_loss_name_error=", str(cm.exception))

    # scores 不是 2 维时应报错
    def test_invalid_scores_dim(self):
        scores = torch.randn(3, 4, 1)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        loss_fn = build_loss("full_pool_ce")

        with self.assertRaises(ValueError) as cm:
            loss_fn(scores, targets)

        print("6. 非法 scores shape 测试")
        print("invalid_scores_dim_error=", str(cm.exception))

    # targets 不是 1 维时应报错
    def test_invalid_targets_dim(self):
        scores = torch.randn(3, 4)
        targets = torch.tensor([[0], [1], [2]], dtype=torch.long)
        loss_fn = build_loss("full_pool_ce")

        with self.assertRaises(ValueError) as cm:
            loss_fn(scores, targets)

        print("7. 非法 targets shape 测试")
        print("invalid_targets_dim_error=", str(cm.exception))

    # targets 越界时应报错
    def test_targets_out_of_range(self):
        scores = torch.tensor(
            [
                [1.0, 0.5, -0.2],
                [0.3, 1.2, 0.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 3], dtype=torch.long)
        loss_fn = build_loss("full_pool_ce")

        with self.assertRaises(ValueError) as cm:
            loss_fn(scores, targets)

        print("8. targets 越界测试")
        print("targets_out_of_range_error=", str(cm.exception))

    # temperature 非法时应在初始化阶段报错
    def test_invalid_temperature(self):
        with self.assertRaises(ValueError) as cm_zero:
            FullPoolCrossEntropyLoss(temperature=0)
        print("9. temperature 非法测试")
        print("temperature_zero_error=", str(cm_zero.exception))

        with self.assertRaises(ValueError) as cm_negative:
            FullPoolCrossEntropyLoss(temperature=-0.5)
        print("temperature_negative_error=", str(cm_negative.exception))


if __name__ == "__main__":
    unittest.main()
