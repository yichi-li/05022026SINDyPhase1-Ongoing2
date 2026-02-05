# Phase 1: SINDy Recovery for Epidemic Dynamics

This project focuses on **Phase 1 (Recovery)**: can SINDy recover SIR/SEIR dynamics from synthetic data under increasing noise?

All experiments use **synthetic data only** (no real-world datasets).

## Quick Start (macOS / Linux)

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the batch grid (recommended for reproducibility)
python run_phase1_batch.py
```

## Output

Results are written to `resultsSAVE/<timestamp>_.../`:

- `terminal_output.txt` (full log)
- `fig*.png` (plots)

These outputs can be large, so they are typically excluded from version control.

## Configuration (Environment Variables)

You can control experiments with environment variables (see `run_phase1.py`):

- `PHASE1_MODEL_TYPE`: `sir` or `seir`
- `PHASE1_SCENARIO`: `fast`, `medium`, `slow`
- `PHASE1_T_END_SIM`, `PHASE1_T_END_FIT`, `PHASE1_DT`
- `PHASE1_LIBRARY_MODE`: `polynomial` or `restricted`
- `PHASE1_RESTRICTED_SET`: `basic` or `extended` (when restricted)
- `PHASE1_POLY_DEGREE`, `PHASE1_THRESHOLD`
- `PHASE1_OPTIMIZER`: `stlsq` or `sr3`
- `PHASE1_REDUCED_COORDS`: `0` or `1`
- `PHASE1_SHOW_PLOTS`: `0` or `1`
- `PHASE1_T_SWITCH`, `PHASE1_T_SWITCH_ALT`

## Folder Structure

- `src/seir_sim.py`: SIR/SEIR simulator (synthetic ground truth)
- `src/phase1_recovery_pysindy.py`: Phase 1 experiments
- `src/plots.py`: small plotting helpers
- `src/visualization.py`: visualization utilities
- `run_phase1.py`: single-run entry point
- `run_phase1_batch.py`: batch grid runner
- `run_phase1_symbolic.py`: symbolic regression baseline (SIR)
