import unittest
from dataclasses import replace

import numpy as np
import torch

from diagnose_phase import template_scores
from pinna import config, evaluate


class DiagnosticTests(unittest.TestCase):
    def test_paired_clean_waveform_for_band_snr_diagnostic(self):
        ir = np.zeros((2,32),np.float32)
        ir[:,5] = 1
        cond = evaluate.Cond("core","paired",window=4096,preroll=128,snr_db=30)
        for kind in ("band","harmonic"):
            gen = torch.Generator().manual_seed(31)
            state = gen.get_state()
            noisy = evaluate.synth(cond,ir,np.arange(2),kind,config.SR,gen)
            clean_gen = torch.Generator().set_state(state)
            clean = evaluate.synth(replace(cond,snr_db=None),ir,np.arange(2),kind,config.SR,clean_gen)
            snr = 10*torch.log10(clean.square().mean(-1)/(noisy-clean).square().mean(-1))
            torch.testing.assert_close(snr,torch.full_like(snr,30),atol=1e-3,rtol=0)

    def test_known_templates_and_no_information(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        x = torch.eye(3)[labels]
        scores = template_scores(x, x, x, labels, 3)
        self.assertEqual(scores["accuracy_pct"], 100)
        self.assertEqual(scores["shifted_accuracy_pct"], 100)
        self.assertEqual(scores["confusion"], [[2,0,0], [0,2,0], [0,0,2]])
        scores = template_scores(x*0, x*0, x*0, labels, 3)
        self.assertAlmostEqual(scores["accuracy_pct"], 100/3, places=4)


if __name__ == "__main__":
    unittest.main()
