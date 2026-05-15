# RL Airfoil Optimization (TD3-first, XAI-ready)

This repository contains a modular reinforcement-learning pipeline for airfoil optimization with:
- pluggable evaluators (`surrogate`, `xfoil`),
- TD3 training/evaluation flow,
- strict XAI-oriented logging outputs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main train --algorithm td3 --evaluator surrogate --surrogate-model-name S-1D
python -m src.main evaluate --algorithm td3 --evaluator surrogate --run-dir logs/td3/run_<id>
```

## Output layout

Per run, outputs are written under:

- `logs/td3/run_<id>/...`

with all required XAI files:
- `experiment_metadata.json`
- `rollout_step_logs.csv`
- `policy_outputs.csv`
- `replay_sample_logs.csv`
- `episode_summary.csv`
- `training_update_logs.csv`
- `xfoil_validation_logs.csv`

