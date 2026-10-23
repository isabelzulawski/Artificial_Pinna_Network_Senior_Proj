"""Monaural direction-of-arrival decoder for the artificial pinna.

Modules, in pipeline order:
  config     paths, device, constants
  data       measured pinna responses (SOFA), direction grid, train/test direction splits
  simulate   source signals and the forward model (source -> pinna -> microphone)
  features   input features (frequency-differenced Welch spectrum, 2-20 kHz)
  model      the MLP classifier, its soft-label loss and decoder
  train      training with a fixed validation set, early stopping and best-checkpoint selection
  evaluate   the test battery (conditions) and metrics
  plots      figures used by the notebook
  report     tables and saved run outputs
"""
