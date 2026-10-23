"""Paths, device and the constants shared by every module.

The SOFA file (measured pinna responses) is not in the repository. Tell the code where it is, either
  - in code/local_paths.json (untracked; copy local_paths.example.json), key "sofa", or
  - with the environment variable PINNA_SOFA.
"""
import json
import os
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = CODE_DIR / "runs"  # notebook outputs: checkpoints, metrics, figures (untracked)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR = 48_000  # sample rate of the measured responses (checked when the SOFA file is loaded)
WINDOW = 65_536  # analysis window: 1.37 s at 48 kHz (the previous team's window)
SOURCE_TYPES = ("white", "pink", "band", "harmonic", "bursts", "clicktrain")


def sofa_path():
    if os.environ.get("PINNA_SOFA"):
        return Path(os.environ["PINNA_SOFA"])
    cfg = CODE_DIR / "local_paths.json"
    if cfg.exists():
        return Path(json.loads(cfg.read_text())["sofa"])
    raise FileNotFoundError("SOFA path not set: copy local_paths.example.json to local_paths.json and edit it, "
                            "or set PINNA_SOFA")
