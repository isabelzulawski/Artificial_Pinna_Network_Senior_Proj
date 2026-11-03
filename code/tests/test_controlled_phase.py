import unittest
import torch
from pinna import controlled_phase as p
from pinna.features import features
from pinna.model import DirectionMLP
from pinna.raw_phase import relative_angles

class ControlledFeatures(unittest.TestCase):
    def test_shapes_controls_and_reference(self):
        w=torch.randn(2,65536)
        mag=features(w)
        counts=[]
        for mode in p.MODES:
            x=p.extract(w,mode)
            self.assertEqual(x.shape,(2,p.feature_count(mode)))
            self.assertTrue(torch.isfinite(x).all())
            torch.testing.assert_close(x[:,:191],mag)
            if mode!='magnitude': counts.append(sum(v.numel() for v in DirectionMLP(x.shape[-1],632,512).parameters()))
        self.assertEqual(len(set(counts)),1)
        self.assertEqual(torch.count_nonzero(p.extract(w,'zero')[:,191:]),0)
        torch.testing.assert_close(p.extract(w,'reference_1k')[:,191:383],relative_angles(w,1000.)[0])
        self.assertTrue(torch.isfinite(p.extract(torch.zeros_like(w),'curvature')).all())
    def test_curvature_phase_algebra(self):
        z=torch.polar(torch.ones(192),torch.randn(192))
        def curvature(t): return t[2:]*t[:-2]*t[1:-1].conj().square()
        shifted=z*torch.exp(-1j*.13*torch.arange(192))
        torch.testing.assert_close(curvature(z),curvature(shifted),atol=1e-5,rtol=1e-5)
