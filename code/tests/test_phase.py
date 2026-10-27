import unittest

import numpy as np
import torch

from pinna import config, features, model, simulate, train
from pinna.phase_features import extract


class PhaseTests(unittest.TestCase):
    def test_magnitude_unchanged_and_phase_finite(self):
        w = torch.randn(3, 2048, device=config.DEVICE)
        original = features.features(w)
        self.assertTrue(torch.equal(extract(w), original))
        phase = extract(w, "phase")
        self.assertEqual(phase.shape, (3, 573))
        self.assertTrue(torch.equal(phase[:, :191], original))
        self.assertTrue(torch.isfinite(phase).all())
        self.assertLessEqual(float(phase[:, 191:].abs().max()), 1.00001)
        zero = extract(w, "phase_zero")
        self.assertEqual(int(torch.count_nonzero(zero[:, 191:])), 0)
        self.assertTrue(torch.equal(zero[:, :191], original))
        self.assertTrue(torch.isfinite(extract(torch.zeros_like(w), "phase")).all())

    def test_phase_gain_invariance_and_delay_sensitivity(self):
        # An impulse away from boundaries gives a controlled linear phase slope.
        w = torch.zeros(2, 4096, device=config.DEVICE)
        w[0, 1024], w[1, 1041] = 1, 1
        a = extract(w, "phase")[:, 191:]
        b = extract(w * 2, "phase")[:, 191:]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        self.assertGreater(float((a[0] - a[1]).abs().max()), .01)

    def test_training_checkpoint_and_predictor_all_modes(self):
        ir = np.zeros((3, 80), np.float32)
        ir[:, 10] = 1
        ir[1, 22], ir[2, 37] = .4, -.3
        v = torch.eye(3, device=config.DEVICE)
        idx = np.arange(3)
        for mode in ("magnitude", "phase", "phase_zero"):
            cfg = train.TrainConfig(feature_mode=mode, preroll=128, steps=2,
                                    batch=2, eval_every=1, sources=("white",))
            fitted, history, info = train.fit(ir, v, idx, cfg, log=lambda _: None)
            self.assertEqual(len(history), 2)
            self.assertTrue(np.isfinite(info["best_val_loss"]))
            for pool_size in (None, 4):
                if pool_size:
                    cfg.train_pool = pool_size
                    fitted, _, _ = train.fit(ir, v, idx, cfg, log=lambda _: None)
            waves = simulate.make_batch(ir, idx, ["white"] * 3,
                       torch.Generator(device=config.DEVICE).manual_seed(2), 30, preroll=128)
            vp, cf = train.predictor(fitted, v, mode)(waves)
            self.assertEqual(vp.shape, (3, 3))
            self.assertTrue(torch.isfinite(cf).all())
            torch.testing.assert_close(vp.norm(dim=-1), torch.ones(3, device=config.DEVICE))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
