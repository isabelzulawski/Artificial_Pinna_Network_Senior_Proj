"""Measured pinna responses and the direction grid.

The SOFA file holds one impulse response per measured direction (632: 18 azimuths x 35 elevations + 2 poles,
480 taps at 48 kHz). Directions are unit vectors (x front, y left, z up; azimuth counter-clockwise from +x).
"""
import h5py
import numpy as np

from . import config


def load_sofa(path=None):
    """-> ir (D, taps) float32, azimuth (D,), elevation (D,) in degrees, sample rate."""
    with h5py.File(path or config.sofa_path(), "r") as f:
        pos = f["SourcePosition"][:].astype(np.float64)
        ir = f["Data.IR"][:, 0, :].astype(np.float32)
        sr = int(f["Data.SamplingRate"][:].ravel()[0])
    assert sr == config.SR, f"expected {config.SR} Hz responses, got {sr}"
    return ir, pos[:, 0], pos[:, 1], sr


def unit_vec(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def vec_to_az_el(v):
    v = np.asarray(v, np.float64)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    return np.mod(np.degrees(np.arctan2(v[:, 1], v[:, 0])), 360.0), np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))


def split(protocol, az, el):
    """Direction split -> (training direction indices, test direction indices).
    seen       train and test on all 632 directions (new sounds at the measured directions)
    unseen_el  train on elevations that are multiples of 10 deg (+ poles, 308); test the 324 in between (5 deg away)
    unseen_az  every third azimuth column (az 20, 80, ..., 320; 210 directions) held out (20 deg from training)"""
    if protocol == "seen":
        return np.arange(len(az)), np.arange(len(az))
    if protocol == "unseen_el":
        on10 = np.isclose(np.round(el / 10) * 10, el) | (np.abs(el) > 89)
        return np.where(on10)[0], np.where(~on10)[0]
    if protocol == "unseen_az":
        held = (np.round(az / 20).astype(int) % 18 % 3 == 1) & (np.abs(el) < 89)
        return np.where(~held)[0], np.where(held)[0]
    raise ValueError(protocol)


def mirror_elevation_index(az, el):
    """Index of the grid direction with the same azimuth and the opposite elevation (poles swap)."""
    key = {(round(a) % 360, round(e)): i for i, (a, e) in enumerate(zip(az, el))}
    return np.array([key.get((round(a) % 360 if abs(e) < 89 else 0, -round(e))) for a, e in zip(az, el)])
