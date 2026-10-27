"""Continuous direction training, checkpoint compatibility and predictor contract."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinna import config, model, train


class RegressionTests(unittest.TestCase):
    def test_cosine_geometry(self):
        m = model.DirectionMLP(4, 3, output_kind="regression")
        v = torch.eye(3)
        self.assertAlmostEqual(float(model.direction_loss(m, v, v, v)), 0)
        self.assertAlmostEqual(float(model.direction_loss(m, -v, v, v)), 2)

    def test_training_and_checkpoint(self):
        ir = np.zeros((3, 80), np.float32)
        ir[:, 10] = 1
        ir[1, 22], ir[2, 37] = .4, -.3
        v = torch.eye(3, device=config.DEVICE)
        cfg = train.TrainConfig(output_kind="regression", steps=2, batch=2,
                                eval_every=1, sources=("white",), preroll=128)
        fitted, history, info = train.fit(ir, v, np.arange(3), cfg, log=lambda _: None)
        self.assertTrue(np.isfinite(info["best_val_loss"]))
        waves = torch.randn(3, 2048, device=config.DEVICE)
        expected, conf = train.predictor(fitted, v)(waves)
        self.assertIsNone(conf)
        torch.testing.assert_close(expected.norm(dim=-1), torch.ones(3, device=config.DEVICE))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.pt"
            model.save(p, fitted)
            restored, ck = model.load(p, config.DEVICE)
            actual, conf = train.predictor(restored, v)(waves)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(ck["config"]["output_kind"], "regression")

    def test_old_checkpoint_and_classifier_unchanged(self):
        torch.manual_seed(7)
        default = model.DirectionMLP(4, 6)
        torch.manual_seed(7)
        explicit = model.DirectionMLP(4, 6, output_kind="classification")
        x = torch.randn(3, 4)
        torch.testing.assert_close(default(x), explicit(x), rtol=0, atol=0)
        old_config = {k: v for k, v in default.config.items() if k != "output_kind"}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "old.pt"
            torch.save({"config": old_config, "state_dict": default.state_dict()}, p)
            restored, _ = model.load(p)
            torch.testing.assert_close(restored(x), default(x), rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
