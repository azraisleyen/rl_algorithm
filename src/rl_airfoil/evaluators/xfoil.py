from __future__ import annotations
import numpy as np
from .base import Evaluator, AeroOutput


class XFOILEvaluator(Evaluator):
    name = "xfoil"

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        camber = float(np.mean(cst[:4]) - np.mean(cst[4:]))
        thickness = float(np.clip(0.11 + 0.18 * np.std(cst), 0.04, 0.22))
        cl = 0.75 + 0.11 * aoa + 0.65 * camber
        cd = max(1e-6, 0.011 + 0.013 * (camber**2) + 0.025 * abs(thickness - 0.12))
        cm = -0.07 - 0.11 * camber
        return AeroOutput(cl=cl, cd=cd, cm=cm, tc=thickness)
