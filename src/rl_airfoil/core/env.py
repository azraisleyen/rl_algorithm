from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple
from rl_airfoil.config.schema import ExperimentConfig
from rl_airfoil.evaluators.base import Evaluator


class AirfoilEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: ExperimentConfig, evaluator: Evaluator):
        super().__init__()
        self.cfg = cfg
        self.evaluator = evaluator
        self.max_steps = cfg.episode_max_steps
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.step_count = 0
        self.cst = np.zeros(8, dtype=np.float32)
        self.last_aero = None

    def _obs(self):
        cl = self.last_aero.cl if self.last_aero else 0.0
        cd = self.last_aero.cd if self.last_aero else 0.01
        cm = self.last_aero.cm if self.last_aero else -0.05
        tc = self.last_aero.tc if self.last_aero else 0.12
        cl_cd = cl / max(cd, self.cfg.cd_lower_bound)
        return np.array([*self.cst.tolist(), self.cfg.aoa, self.cfg.re, np.log10(self.cfg.re), cl, cd, cm, cl_cd, tc], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cst = np.random.uniform(-0.05, 0.05, size=8).astype(np.float32)
        self.last_aero = self.evaluator.evaluate(self.cst, self.cfg.aoa, self.cfg.re)
        return self._obs(), {"done_reason": "reset"}

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        prev_cst = self.cst.copy()
        self.cst = np.clip(self.cst + action * self.cfg.action_scale, -0.3, 0.3)
        aero = self.evaluator.evaluate(self.cst, self.cfg.aoa, self.cfg.re)
        self.last_aero = aero

        cl_cd = aero.cl / max(aero.cd, self.cfg.cd_lower_bound)
        cm_lo, cm_hi = self.cfg.constraints.cm_bounds
        tc_lo, tc_hi = self.cfg.constraints.tc_bounds
        cm_pen = max(0.0, cm_lo - aero.cm) + max(0.0, aero.cm - cm_hi)
        tc_pen = max(0.0, tc_lo - aero.tc) + max(0.0, aero.tc - tc_hi)
        reward_obj = self.cfg.reward_weights.w1 * cl_cd
        reward_cm_pen = self.cfg.reward_weights.w2 * cm_pen
        reward_tc_pen = self.cfg.reward_weights.w3 * tc_pen
        reward = reward_obj - reward_cm_pen - reward_tc_pen

        terminated = False
        truncated = self.step_count >= self.max_steps
        done_reason = "max_episode_steps" if truncated else "running"
        info: Dict = {
            "prev_cst": prev_cst,
            "next_cst": self.cst.copy(),
            "delta_cst": self.cst - prev_cst,
            "CL": aero.cl, "CD": aero.cd, "CM": aero.cm, "t_c": aero.tc, "CL_CD": cl_cd,
            "reward_objective_term": reward_obj,
            "reward_CM_penalty": reward_cm_pen,
            "reward_tc_penalty": reward_tc_pen,
            "done_reason": done_reason,
            "is_CM_feasible": cm_pen == 0.0,
            "is_tc_feasible": tc_pen == 0.0,
            "is_geometry_valid": aero.is_geometry_valid,
            "CM_lower_violation": max(0.0, cm_lo - aero.cm),
            "CM_upper_violation": max(0.0, aero.cm - cm_hi),
            "tc_lower_violation": max(0.0, tc_lo - aero.tc),
            "tc_upper_violation": max(0.0, aero.tc - tc_hi),
        }
        return self._obs(), float(reward), terminated, truncated, info
