"""Earlier notebook's direct sine/cosine phase representation adapted to an MLP.

Keep the original 191 MLP magnitude features. Compute complex-mel phase as in
the notebook, encode cos/sin before pooling, then compress to an 8x8 map per
channel: 128 additional values. Pooling is a lossy lightweight adaptation, not
an exact reproduction of the CNN's full phase input. No reference frequency,
neighbor-frequency differences or learned convolutional branch is used.
"""
import torch
import torch.nn.functional as F

from .original_features import T_TARGET, complex_mel

MLP_MEL_MODES = ('mlp_mel_phase', 'mlp_mel_zero')
PHASE_GRID = (8, 8)
N_MLP_MEL_PHASE = 2 * PHASE_GRID[0] * PHASE_GRID[1]


def mlp_mel_phase(w, mode):
    if mode == 'mlp_mel_zero':
        return w.new_zeros((len(w), N_MLP_MEL_PHASE))
    if mode != 'mlp_mel_phase':
        raise ValueError(mode)
    angle = complex_mel(w).angle()
    phase = torch.stack((angle.cos(), angle.sin()), 1)[..., :T_TARGET]
    if phase.shape[-1] < T_TARGET:
        phase = F.pad(phase, (0, T_TARGET-phase.shape[-1]))
    return F.adaptive_avg_pool2d(phase, PHASE_GRID).flatten(1)
