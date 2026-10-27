import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinna import config, features, model, simulate, train
from pinna.phase_features import extract, feature_count
from pinna.raw_phase import relative_angles, reference_bin


class RawPhaseTests(unittest.TestCase):
    def test_known_relative_phase_reference_and_gain(self):
        t = torch.arange(4800, device=config.DEVICE) / config.SR
        w = (torch.cos(2*math.pi*1000*t+.3) + torch.cos(2*math.pi*3000*t-.7))[None]
        j = int(np.where(np.isclose(features.FREQS, 3000))[0][0])
        a, valid = relative_angles(w, 1000)
        self.assertTrue(valid[0, j])
        self.assertAlmostEqual(float(a[0, j]), -1/math.pi, places=4)
        b, _ = relative_angles(w*3, 1000)
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        c, _ = relative_angles(w, 3000)
        self.assertAlmostEqual(float(c[0, j]), 0, places=6)
        self.assertEqual(reference_bin(1000, 65536)[0], 1365)
        for bad in (0, -1, 24000, float('nan')):
            with self.assertRaises(ValueError):
                reference_bin(bad, 65536)

    def test_masks_shapes_and_matching_encoding(self):
        w = torch.randn(2, 4096, device=config.DEVICE)
        mag = features.features(w)
        raw = extract(w, 'raw_phase', 750)
        encoded = extract(w, 'raw_phase_sincos', 750)
        self.assertEqual(raw.shape, (2, 383))
        self.assertEqual(encoded.shape, (2, 575))
        self.assertEqual(feature_count('raw_phase'), 383)
        torch.testing.assert_close(raw[:, :191], mag, rtol=0, atol=0)
        torch.testing.assert_close(encoded[:, 191:383], torch.cos(math.pi*raw[:, 191:]))
        torch.testing.assert_close(encoded[:, 383:], torch.sin(math.pi*raw[:, 191:]))
        self.assertLessEqual(float(raw[:, 191:].abs().max()), 1)
        for mode in ('raw_phase', 'raw_phase_zero', 'raw_phase_sincos'):
            silence = extract(torch.zeros_like(w), mode)
            self.assertTrue(torch.isfinite(silence).all())
            self.assertEqual(int(torch.count_nonzero(silence[:, 191:])), 0)
        self.assertEqual(int(torch.count_nonzero(extract(w, 'raw_phase_zero')[:, 191:])), 0)

    def test_training_checkpoint_predictor_uses_reference(self):
        ir = np.zeros((3, 80), np.float32)
        ir[:, 10] = 1
        ir[1, 22], ir[2, 37] = .4, -.3
        v, idx = torch.eye(3, device=config.DEVICE), np.arange(3)
        cfg = train.TrainConfig(feature_mode='raw_phase', phase_reference_hz=4000,
                               steps=2, batch=2, eval_every=1, sources=('white',), preroll=128)
        fitted, history, info = train.fit(ir, v, idx, cfg, log=lambda _: None)
        self.assertEqual(info['config']['phase_reference_hz'], 4000)
        waves = simulate.make_batch(ir, idx, ['white']*3,
                    torch.Generator(device=config.DEVICE).manual_seed(2), 30, preroll=128)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.pt'
            model.save(path, fitted, feature_mode='raw_phase', info=info)
            loaded, ck = model.load(path, config.DEVICE)
            direct = model.direction_prediction(loaded, loaded(extract(waves, 'raw_phase', 4000)), v)[0]
            predicted = train.predictor(loaded, v, 'raw_phase', ck['info']['config']['phase_reference_hz'])(waves)[0]
            torch.testing.assert_close(direct, predicted)


if __name__ == '__main__':
    unittest.main()
