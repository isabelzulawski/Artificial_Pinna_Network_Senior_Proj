import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinna import config
from pinna.original_cnn import PinnaResNet
from pinna.original_features import mel_bank, original_features
from compare_original_cnn_phase import fit, phase_input, predictor


class OriginalCNNTests(unittest.TestCase):
    def test_control_changes_only_phase_and_silence_is_finite(self):
        w = torch.randn(2, 4096, device=config.DEVICE)
        full = original_features(w)
        control = original_features(w, include_phase=False)
        self.assertEqual(full.shape, (2, 4, 64, 256))
        torch.testing.assert_close(full[:, :2], control[:, :2], rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(control[:, 2:])), 0)
        torch.testing.assert_close(phase_input(full, False), control, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(original_features(torch.zeros_like(w))).all())

    @unittest.skipUnless(importlib.util.find_spec('librosa'), 'Optional local notebook-reference check requires librosa')
    def test_frontend_matches_notebook_librosa(self):
        import librosa
        bank = librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=20, fmax=24000)
        np.testing.assert_allclose(mel_bank(), bank, atol=1e-9, rtol=1e-5)
        w = np.random.default_rng(2).normal(size=65536).astype(np.float32)
        spectrum = librosa.stft(w, n_fft=1024, hop_length=256, win_length=1024,
                                window='hann', center=True, pad_mode='reflect')
        mel = bank @ spectrum
        log = np.log10(np.abs(mel)+1e-10).astype(np.float32)
        reference = np.stack((log, librosa.feature.delta(log, mode='nearest'),
                              np.cos(np.angle(mel)), np.sin(np.angle(mel))))[..., :256]
        actual = original_features(torch.from_numpy(w[None])).numpy()[0]
        # Float32 FFT/complex-matmul implementations differ most where mel
        # components nearly cancel. Check both typical and worst-case error.
        self.assertLess(float(np.sqrt(np.mean((actual-reference)**2, axis=(1, 2))).max()), 3e-5)
        np.testing.assert_allclose(actual, reference, atol=2e-3, rtol=1e-4)

    def test_fit_predictor_and_checkpoint_both_modes(self):
        ir = np.zeros((3, 80), np.float32)
        ir[:, 10] = 1
        ir[1, 22], ir[2, 37] = .4, -.3
        vectors = torch.eye(3, device=config.DEVICE)
        w = torch.randn(2, 4096, device=config.DEVICE)
        validation = (original_features(w).cpu(), vectors[:2].cpu())
        counts = []
        for enabled in (False, True):
            fitted, history, info = fit(ir, vectors, np.arange(3), validation, enabled,
                                        seed=0, steps=2, batch=2, eval_every=1, log=lambda *a, **k: None)
            self.assertEqual(len(history), 2)
            self.assertTrue(np.isfinite(info['best_val_err_mean']))
            counts.append(info['n_params'])
            expected, confidence = predictor(fitted, enabled, chunk=1)(w)
            self.assertIsNone(confidence)
            self.assertTrue(torch.isfinite(expected).all())
            torch.testing.assert_close(expected.norm(dim=-1), torch.ones(2, device=config.DEVICE), atol=1e-5, rtol=1e-5)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'cnn.pt'
                torch.save(fitted.state_dict(), path)
                loaded = PinnaResNet().to(config.DEVICE).eval()
                loaded.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True))
                actual, _ = predictor(loaded, enabled, chunk=1)(w)
                torch.testing.assert_close(actual, expected)
        self.assertEqual(counts[0], counts[1])


if __name__ == '__main__':
    unittest.main()
