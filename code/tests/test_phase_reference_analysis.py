import unittest
import numpy as np
from analyze_phase_reference import phase_views, wrap


class PhaseAlgebraTests(unittest.TestCase):
    def test_linear_delay_and_branch(self):
        n=65536; sr=48000
        f=np.fft.rfftfreq(n,1/sr)
        z=np.exp(-2j*np.pi*f*50/sr)*(1+.3*np.exp(-2j*np.pi*f*7/sr))
        a,_=phase_views(z,f)
        b,_=phase_views(z*np.exp(-2j*np.pi*f*37/sr),f)
        self.assertLess(np.nanmax(abs(wrap(a['phase_curvature']-b['phase_curvature']))),1e-10)
        self.assertLess(np.nanmax(abs(wrap(a['delay_detrended']-b['delay_detrended']))),1e-9)
        self.assertGreater(np.nanmean(abs(wrap(a['reference_1k']-b['reference_1k']))),1.)
        self.assertGreater(np.nanmean(abs(wrap(a['scaled_unwrapped']-a['scaled_branch_plus_2pi']))),1.)

    def test_silence_and_missing_reference(self):
        f=np.fft.rfftfreq(65536,1/48000)
        result,meta=phase_views(np.zeros_like(f,dtype=complex),f)
        self.assertFalse(meta['reference_valid'])
        self.assertTrue(np.isnan(result['reference_1k']).all())
        z=np.ones_like(f,dtype=complex)
        z[np.argmin(abs(f-1000))]=0
        result,meta=phase_views(z,f)
        self.assertFalse(meta['reference_valid'])
        self.assertTrue(np.isnan(result['scaled_unwrapped']).all())


if __name__ == '__main__': unittest.main()
