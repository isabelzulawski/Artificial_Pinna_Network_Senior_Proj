"""Verify resumability rather than merely loading model weights."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pinna import train


class ResumeTests(unittest.TestCase):
    def test_exact_resume_and_config_guard(self):
        device = train.DEVICE
        vectors = torch.eye(3, device=device)
        val = {'x': torch.arange(24, device=device, dtype=torch.float32).reshape(6, 4) / 24,
               'dir': np.array([0, 1, 2, 0, 1, 2])}
        cfg = train.TrainConfig(steps=20, batch=4, hidden=8, eval_every=5, patience=20,
                                select_by='angular')
        def waves(ir, idx, kinds, gen, snr, preroll):
            return torch.randn(len(idx), 4, generator=gen, device=device)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(train, 'feature_count', return_value=4), \
             patch.object(train, 'extract', side_effect=lambda x, *args: x), \
             patch.object(train.simulate, 'make_batch', side_effect=waves):
            checkpoint = Path(tmp) / 'latest.pt'
            full, history, info = train.fit(None, vectors, np.arange(3), cfg, val=val, log=lambda x: None)
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 6: raise InterruptedError('simulated job termination')
                return waves(*args, **kwargs)
            with patch.object(train.simulate, 'make_batch', side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    train.fit(None, vectors, np.arange(3), cfg, val=val, checkpoint=checkpoint, log=lambda x: None)
            resumed, rh, ri = train.fit(None, vectors, np.arange(3), cfg, val=val,
                                        checkpoint=checkpoint, resume=True, log=lambda x: None)
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            for expected, actual in zip(history, rh):
                for key in expected.keys() - {'seconds'}: self.assertEqual(expected[key], actual[key])
            self.assertEqual(info['best_step'], ri['best_step'])
            with self.assertRaisesRegex(ValueError, 'Resume config'):
                train.fit(None, vectors, np.arange(3), replace(cfg, steps=40), val=val,
                          checkpoint=checkpoint, resume=True, log=lambda x: None)


if __name__ == '__main__': unittest.main()


class SweepTests(unittest.TestCase):
    def test_validation_ranking_and_fresh_confirmation(self):
        import sweep_mlp_scaling as sweep
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sofa = root / 'sample.sofa'; sofa.write_bytes(b'test fixture')
            calls = []
            def trial(output, stage, width, steps, seed, *args):
                calls.append((stage, width, steps, seed))
                # Larger models deliberately win validation, regardless of test.
                return dict(hidden=width, steps=steps, val_mean=1 / width + 1 / steps,
                            n_params=width, test_mean=width)
            argv = ['sweep', '--sofa', str(sofa), '--output', str(root / 'run')]
            with patch('sys.argv', argv), patch.object(sweep.data, 'load_sofa', return_value=(None, None, None, 48000)), \
                 patch.object(sweep.data, 'split', return_value=(np.arange(3), np.arange(3))), \
                 patch.object(sweep.data, 'unit_vec', return_value=np.eye(3)), \
                 patch.object(sweep.train, 'make_validation_set', return_value={}), \
                 patch.object(sweep, 'run_one', side_effect=trial):
                sweep.main()
            self.assertEqual(len(calls), 18)
            self.assertEqual(calls[:9], [('screen', w, s, 0) for w in (256, 512, 1024) for s in (3000, 6000, 12000)])
            self.assertEqual(calls[9:], [('confirm', 1024, steps, seed) for steps in (12000, 6000, 3000) for seed in (0, 1, 2)])
            self.assertTrue((root / 'run' / 'selected.json').exists())
