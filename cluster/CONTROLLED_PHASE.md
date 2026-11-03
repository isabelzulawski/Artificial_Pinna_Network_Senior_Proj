# Controlled 512-width phase comparison

Run `code/compare_controlled_phase.py` from a GPU allocation. It runs A–E serially:
191-dimensional magnitude anchor, and matched 573-dimensional zero, compact,
1 kHz reference, and curvature conditions. Two 512-unit hidden layers throughout.
Historical feature modes, generator, objective and scaling directories are unchanged.

The five 5-step smoke runs must all pass before the five seed-0 3000-step screens.
A phase condition within 10% of the zero control's validation error is confirmed;
if none passes, confirm the best phase as a seed-variability check. This permissive
rule is predeclared, not evidence of improvement. Both controls are always confirmed.
Confirmation starts fresh at seeds 0,1,2 with independent 12000-step OneCycle
schedules. No screen checkpoint is extended with an incompatible LR horizon.
Checkpoint selection uses validation angular error; early stopping is disabled.
Classification CE, batch 256, lr .002, preroll 4096 and uniform 10–40 dB remain.
The fixed validation source generator uses seed 90004. No test results select models.

All phase inputs are microphone-derived. Curvature uses the second circular
frequency difference of full-window FFT phase at uniformly spaced original band
centers; division by pi scales its wrapped angle. This cancels an ideal linear
phase ramp, not arbitrary source phase or finite-window changes. Invalid bins are
zero-filled at the existing relative amplitude threshold. This mask is not SNR.
D pads 192 phase angles and E pads 190 angles to 382 extra inputs. B–E share
LayerNorm and parameter counts; zero-control comparison accounts for normalization
and capacity changes. A is the historical architecture anchor.

SNR 0,5,10,20,30,40 diagnostics use fixed validation draws with identical source,
gain and noise RNG consumption. They are exploratory validation diagnostics, not
new independent test data. Seed SD is optimization variability, not physical-session
uncertainty. Recommend continuing phase only for repeatable matched-control gains.

```bash
export PINNA_PYTHON=/zooper2/isabel.zulawski/jupyter_venv/bin/python3
export PINNA_SOFA='/zooper2/isabel.zulawski/senior projects/Artificial_Pinna_Network_Senior_Proj/data/ArtificialPinnaB2_v2 1.sofa'
phase_output="$PWD/code/runs/controlled_phase_$(date -u +%Y%m%dT%H%M%SZ)_$$"
sbatch cluster/controlled_phase.sbatch --output "$phase_output"
```

Inside an existing GPU Jupyter Slurm allocation where sbatch is unavailable:

```bash
nohup env OMP_NUM_THREADS=2 MPLBACKEND=Agg "$PINNA_PYTHON" -u code/compare_controlled_phase.py --sofa "$PINNA_SOFA" --require-cuda --output "$phase_output" > "${phase_output}.log" 2>&1 < /dev/null &
tail -f "${phase_output}.log"
```

`summary.json`, `report.md`, validation/loss curves update after each completed
trial. `selection.json` records the screening decision. Each trial retains a best
model, full optimizer/scheduler/RNG checkpoint, training log and result JSON.
Existing root/trial directories are rejected rather than overwritten. An interrupted
sweep currently requires a fresh output root; saved training states are retained
for explicit recovery, not silently restarted or skipped.
