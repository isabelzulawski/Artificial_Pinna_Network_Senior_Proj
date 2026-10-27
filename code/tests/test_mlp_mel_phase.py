import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pinna import config, features, model, train
from pinna.original_features import original_features
from pinna.phase_features import extract, feature_count


class MLPMelPhaseTests(unittest.TestCase):
    def test_direct_phase_encoding_and_unchanged_magnitude(self):
        w = torch.randn(2, 4096, device=config.DEVICE)
        expected = F.adaptive_avg_pool2d(original_features(w)[:, 2:], (8, 8)).flatten(1)
        x = extract(w, 'mlp_mel_phase')
        self.assertEqual(x.shape, (2, 319))
        self.assertEqual(feature_count('mlp_mel_phase'), 319)
        torch.testing.assert_close(x[:, :191], features.features(w), rtol=0, atol=0)
        torch.testing.assert_close(x[:, 191:], expected, rtol=0, atol=0)
        zero = extract(w, 'mlp_mel_zero')
        torch.testing.assert_close(x[:, :191], zero[:, :191], rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(zero[:, 191:])), 0)
        self.assertLessEqual(float(x[:, 191:].abs().max()), 1.00001)
        self.assertTrue(torch.isfinite(extract(torch.zeros_like(w), 'mlp_mel_phase')).all())

    def test_lightweight_mlp_only_and_checkpoint_both_modes(self):
        ir = np.zeros((3, 80), np.float32)
        ir[:, 10] = 1
        ir[1, 22], ir[2, 37] = .4, -.3
        v = torch.eye(3, device=config.DEVICE)
        for mode in ('mlp_mel_zero', 'mlp_mel_phase'):
            cfg = train.TrainConfig(feature_mode=mode, hidden=232, select_by='angular',
                                    steps=2, batch=2, eval_every=1, sources=('white',), preroll=128)
            fitted, history, info = train.fit(ir, v, np.arange(3), cfg, log=lambda _: None)
            self.assertIsInstance(fitted, model.DirectionMLP)
            self.assertEqual(fitted.config['n_in'], 319)
            self.assertEqual(fitted.config['hidden'], 232)
            self.assertTrue(np.isfinite(info['best_val_err_mean']))
            w = torch.randn(2, 4096, device=config.DEVICE)
            expected, _ = train.predictor(fitted, v, mode)(w)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'mlp.pt'
                model.save(path, fitted, feature_mode=mode)
                loaded, ck = model.load(path, config.DEVICE)
                self.assertIsInstance(loaded, model.DirectionMLP)
                actual, _ = train.predictor(loaded, v, ck['feature_mode'])(w)
                torch.testing.assert_close(expected, actual)
        budget = model.DirectionMLP(319, 632, hidden=232)
        self.assertEqual(sum(p.numel() for p in budget.parameters()), 276190)
        self.assertLessEqual(sum(p.numel() for p in budget.parameters()), 277750)


if __name__ == '__main__':
    unittest.main()
