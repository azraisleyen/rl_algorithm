from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AeroOutput:
    cl: float
    cd: float
    cm: float
    tc: float
    is_geometry_valid: bool = True
    solver_status: str = "ok"


class Evaluator:
    name: str = "base"

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        raise NotImplementedError
