from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple
import json
import time
import subprocess


@dataclass
class Constraints:
    cm_bounds: Tuple[float, float] = (-0.12, 0.02)
    tc_bounds: Tuple[float, float] = (0.08, 0.18)


@dataclass
class RewardWeights:
    w1: float = 1.0
    w2: float = 0.5
    w3: float = 0.5


@dataclass
class ExperimentConfig:
    algorithm: str = "td3"
    evaluator: str = "surrogate"
    surrogate_model_name: str = "S-1D"
    surrogate_checkpoint_path: str = "checkpoints/surrogate_s1d.pt"
    scaler_json_path: str = "checkpoints/scalers.json"
    rl_checkpoint_path: str = "checkpoints/td3_model.zip"
    seed: int = 42
    total_timesteps: int = 200_000
    episode_max_steps: int = 25
    action_range: Tuple[float, float] = (-1.0, 1.0)
    action_scale: float = 0.02
    aoa: float = 2.0
    re: float = 1e6
    cd_lower_bound: float = 1e-6
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    constraints: Constraints = field(default_factory=Constraints)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def create_run_dir(base: Path, algorithm: str) -> Path:
    run_id = f"run_{int(time.time())}"
    run_dir = base / algorithm.lower() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_experiment_metadata(cfg: ExperimentConfig, run_dir: Path, normalization_stats: Dict) -> None:
    metadata = {
        "experiment_id": run_dir.name,
        "algorithm": cfg.algorithm.upper(),
        "evaluator": cfg.evaluator,
        "surrogate_model_name": cfg.surrogate_model_name,
        "surrogate_checkpoint_path": cfg.surrogate_checkpoint_path,
        "rl_checkpoint_path": cfg.rl_checkpoint_path,
        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "episode_max_steps": cfg.episode_max_steps,
        "action_range": list(cfg.action_range),
        "action_scale": cfg.action_scale,
        "AoA": cfg.aoa,
        "Re": cfg.re,
        "reward_weights": asdict(cfg.reward_weights),
        "constraints": {"CM": list(cfg.constraints.cm_bounds), "t/c": list(cfg.constraints.tc_bounds)},
        "cd_lower_bound": cfg.cd_lower_bound,
        "normalization_stats": normalization_stats,
        "code_commit_hash": git_commit_hash(),
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
