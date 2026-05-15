from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import pickle

from .base import Evaluator, AeroOutput


@dataclass
class SurrogateArtifacts:
    model: torch.nn.Module
    x_scaler: Optional[Any]
    y_scaler: Optional[Any]


class SurrogateEvaluator(Evaluator):
    name = "surrogate"

    def __init__(self, checkpoint_path: str, model_name: str, scaler_x_path: str = "checkpoints/scaler_x.pkl", scaler_y_path: str = "checkpoints/scaler_y.pkl", device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.scaler_x_path = Path(scaler_x_path)
        self.scaler_y_path = Path(scaler_y_path)
        self.device = torch.device(device)
        self.artifacts = self._load_artifacts()

    def _load_pickle(self, p: Path):
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def _load_artifacts(self) -> SurrogateArtifacts:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
            model = ckpt["model"]
        else:
            # fallback: try torchscript
            model = torch.jit.load(str(self.checkpoint_path), map_location=self.device)
        model.eval()

        x_scaler = self._load_pickle(self.scaler_x_path)
        y_scaler = self._load_pickle(self.scaler_y_path)
        return SurrogateArtifacts(model=model, x_scaler=x_scaler, y_scaler=y_scaler)

    def _featurize(self, cst: np.ndarray, aoa: float, re: float) -> np.ndarray:
        x = np.concatenate([cst.astype(np.float32), np.array([aoa, np.log10(re)], dtype=np.float32)], axis=0).reshape(1, -1)
        if self.artifacts.x_scaler is not None:
            x = self.artifacts.x_scaler.transform(x)
        return x.astype(np.float32)

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        x = self._featurize(cst, aoa, re)
        with torch.no_grad():
            xin = torch.from_numpy(x).to(self.device)
            y = self.artifacts.model(xin)
            if isinstance(y, tuple):
                y = y[0]
            y = y.detach().cpu().numpy()

        if self.artifacts.y_scaler is not None:
            y = self.artifacts.y_scaler.inverse_transform(y)

        cl, cd, cm = [float(v) for v in y.reshape(-1)[:3]]
        cd = max(1e-6, cd)

        # Geometry-derived t/c proxy (replace with exact geometry routine if available)
        tc = float(np.clip(np.max(cst[:4] - cst[4:]) + 0.12, 0.02, 0.30))
        is_valid = bool(np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm) and np.isfinite(tc))

        return AeroOutput(cl=cl, cd=cd, cm=cm, tc=tc, is_geometry_valid=is_valid, solver_status="ok" if is_valid else "nan_output")
