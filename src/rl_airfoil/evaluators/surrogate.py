from __future__ import annotations
import numpy as np
from .base import Evaluator, AeroOutput


class SurrogateEvaluator(Evaluator):
    name = "surrogate"

    def __init__(self, checkpoint_path: str, model_name: str):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        camber = float(np.mean(cst[:4]) - np.mean(cst[4:]))
        thickness = float(np.clip(0.12 + 0.2 * np.std(cst), 0.04, 0.22))
        cl = 0.8 + 0.1 * aoa + 0.7 * camber
        cd = max(1e-6, 0.012 + 0.015 * (camber**2) + 0.02 * abs(thickness - 0.12))
        cm = -0.08 - 0.1 * camber
        return AeroOutput(cl=cl, cd=cd, cm=cm, tc=thickness)
