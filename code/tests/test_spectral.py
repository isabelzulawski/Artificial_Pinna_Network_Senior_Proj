import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinna import config, model, train
from pinna.features import features
from pinna.phase_features import extract
from pinna.spectral_features import COUNTS, N_SPECTRAL


class SpectralTests(unittest.TestCase):
    def test_frontend_and_gain(self):
        w = torch.randn(2,4096,device=config.DEVICE)
        x = extract(w,"spectral_mlp")
        self.assertEqual(x.shape,(2,N_SPECTRAL))
        torch.testing.assert_close(x,extract(w,"spectral_cnn"),rtol=0,atol=0)
        coarse = x[:,:2*COUNTS[0]].reshape(2,2,COUNTS[0])
        torch.testing.assert_close(coarse[:,1,1:],features(w),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(x,extract(2*w,"spectral_mlp"),atol=2e-6,rtol=1e-4)
        self.assertTrue(torch.isfinite(extract(torch.zeros_like(w),"spectral_cnn")).all())

    def test_training_and_checkpoint(self):
        ir = np.zeros((3,80),np.float32)
        ir[:,10] = 1
        v = torch.eye(3,device=config.DEVICE)
        for mode in ("spectral_mlp","spectral_cnn"):
            cfg = train.TrainConfig(feature_mode=mode, steps=2,batch=2,eval_every=1,
                                    preroll=128,sources=("white",))
            fitted,_,info = train.fit(ir,v,np.arange(3),cfg,log=lambda _:None)
            self.assertTrue(np.isfinite(info["best_val_loss"]))
            waves = torch.randn(2,4096,device=config.DEVICE)
            expected,_ = train.predictor(fitted,v,mode)(waves)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/"model.pt"
                model.save(path,fitted)
                restored,_ = model.load(path,config.DEVICE)
                actual,_ = train.predictor(restored,v,mode)(waves)
                torch.testing.assert_close(actual,expected)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
