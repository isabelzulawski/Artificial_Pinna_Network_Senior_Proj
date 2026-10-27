import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinna import config, model, train
from pinna.features import N_FEATURES, features
from pinna.phase_features import extract, feature_count


class TemporalTests(unittest.TestCase):
    def test_features_preserve_magnitude_and_mask_silence(self):
        torch.manual_seed(1)
        w = torch.randn(2, 4096, device=config.DEVICE)
        x = extract(w, "temporal_phase")
        self.assertEqual(x.shape, (2, feature_count("temporal_phase")))
        torch.testing.assert_close(x[:, :N_FEATURES], features(w), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(x).all())
        self.assertLessEqual(float(x[:, N_FEATURES:].abs().max()), 1.00001)
        torch.testing.assert_close(extract(2*w, "temporal_phase")[:, N_FEATURES:],
                                   x[:, N_FEATURES:], atol=1e-5, rtol=1e-5)
        silent = extract(torch.zeros_like(w), "temporal_phase")
        self.assertEqual(int(silent[:, N_FEATURES:].count_nonzero()), 0)
        control = extract(w, "temporal_zero")
        torch.testing.assert_close(control[:, :N_FEATURES], x[:, :N_FEATURES])
        self.assertEqual(int(control[:, N_FEATURES:].count_nonzero()), 0)

    def test_initial_logits_frozen_baseline_and_checkpoint(self):
        base = model.DirectionMLP(N_FEATURES, 3).to(config.DEVICE)
        fused = model.TemporalPhaseClassifier(base.config).to(config.DEVICE)
        fused.magnitude.load_state_dict(base.state_dict())
        x = extract(torch.randn(2, 4096, device=config.DEVICE), "temporal_phase")
        torch.testing.assert_close(fused(x), base(x[:, :N_FEATURES]), atol=0, rtol=0)
        before = {k: v.clone() for k, v in fused.magnitude.state_dict().items()}
        opt = torch.optim.AdamW([p for p in fused.parameters() if p.requires_grad], lr=.001)
        for _ in range(2):
            opt.zero_grad()
            fused(x).square().mean().backward()
            opt.step()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in fused.phase.parameters()))
        for k, v in fused.magnitude.state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)
        fused.eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "phase.pt"
            model.save(path, fused, feature_mode="temporal_phase")
            restored, _ = model.load(path, config.DEVICE)
            torch.testing.assert_close(restored(x), fused(x))

    def test_fit_and_wrong_protocol_rejection(self):
        ir = np.zeros((3, 80), np.float32)
        ir[:, 10] = 1
        v = torch.eye(3, device=config.DEVICE)
        base = model.DirectionMLP(N_FEATURES, 3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "base.pt"
            model.save(path, base, feature_mode="magnitude", class_vectors=v.cpu(),
                       info={"config": {"preroll": 128}})
            for mode in ("temporal_phase", "temporal_zero"):
                cfg = train.TrainConfig(feature_mode=mode, baseline_checkpoint=str(path),
                        steps=2, batch=2, eval_every=1, preroll=128, sources=("white",))
                fitted, history, info = train.fit(ir, v, np.arange(3), cfg, log=lambda _: None)
                self.assertEqual(history[0]["step"], 0)
                self.assertLessEqual(info["best_val_err_mean"], history[0]["val_err_mean"] + 1e-5)
                pred, conf = train.predictor(fitted, v, mode)(torch.randn(2, 4096, device=config.DEVICE))
                self.assertTrue(torch.isfinite(pred).all())
                self.assertTrue(torch.isfinite(conf).all())
            with self.assertRaisesRegex(ValueError, "directions"):
                train.fit(ir, v, np.array([1, 0, 2]), cfg, log=lambda _: None)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
