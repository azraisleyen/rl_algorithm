from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple

import json
import subprocess
import time


@dataclass
class Constraints:
    cm_bounds: Tuple[float, float] = (-0.12, 0.02)
    tc_bounds: Tuple[float, float] = (0.08, 0.18)


@dataclass
class RewardWeights:
    w1: float = 1.0
    w2: float = 10.0
    w3: float = 10.0
    w_local_thickness: float = 50.0

    # TD3/SAC gibi continuous-control algoritmalarında
    # actor'ın action saturation yapmasını azaltmak için kullanılır.
    # action vektörü [-1, 1]^8 aralığında olduğu için:
    # max sum(action^2) = 8
    # w_action=0.01 ise max penalty=0.08
    # w_action=0.05 ise max penalty=0.40
    w_action: float = 0.05


@dataclass
class TD3Hyperparameters:
    learning_rate: float = 1e-3
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.03
    target_noise_clip: float = 0.07
    action_noise_sigma: float = 0.02


@dataclass
class GeometryConfig:
    n_points: int = 201
    min_local_thickness_required: float = 1e-4
    max_abs_surface_y: float = 0.75


@dataclass
class ExperimentConfig:
    algorithm: str = "td3"
    evaluator: str = "surrogate"

    surrogate_model_name: str = "S-1D"
    surrogate_checkpoint_path: str = "checkpoints/surrogate_s1d.pt"
    scaler_json_path: str = "checkpoints/scalers.json"

    rl_checkpoint_path: str = "checkpoints/td3_surrogate_s-1d.zip"

    seed: int = 42
    total_timesteps: int = 200_000
    episode_max_steps: int = 25

    action_range: Tuple[float, float] = (-1.0, 1.0)
    action_scale: float = 0.003
    cst_bounds: Tuple[float, float] = (-0.35, 0.35)

    # Güvenli başlangıç airfoil'i.
    # Kendi paper'daki baseline CST değerleriniz varsa burayı onunla değiştirin.
    initial_cst: Tuple[float, float, float, float, float, float, float, float] = (
        0.20, 0.18, 0.14, 0.10,
        -0.12, -0.10, -0.08, -0.05,
    )
    initial_cst_noise_std: float = 0.005

    aoa: float = 2.0
    re: float = 1e6

    cd_lower_bound: float = 1e-6

    invalid_geometry_penalty: float = 100.0
    solver_error_penalty: float = 100.0

    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    constraints: Constraints = field(default_factory=Constraints)
    td3: TD3Hyperparameters = field(default_factory=TD3Hyperparameters)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def create_run_dir(base: Path, algorithm: str) -> Path:
    run_id = f"run_{int(time.time())}"
    run_dir = base / algorithm.lower() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_experiment_metadata(
    cfg: ExperimentConfig,
    run_dir: Path,
    normalization_stats: Dict,
) -> None:
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
        "cst_bounds": list(cfg.cst_bounds),
        "initial_cst": list(cfg.initial_cst),
        "initial_cst_noise_std": cfg.initial_cst_noise_std,
        "AoA": cfg.aoa,
        "Re": cfg.re,
        "reward_weights": asdict(cfg.reward_weights),
        "constraints": {
            "CM": list(cfg.constraints.cm_bounds),
            "t/c": list(cfg.constraints.tc_bounds),
        },
        "cd_lower_bound": cfg.cd_lower_bound,
        "invalid_geometry_penalty": cfg.invalid_geometry_penalty,
        "solver_error_penalty": cfg.solver_error_penalty,
        "td3_hyperparameters": asdict(cfg.td3),
        "geometry_config": asdict(cfg.geometry),
        "normalization_stats": normalization_stats,
        "code_commit_hash": git_commit_hash(),
    }

    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)