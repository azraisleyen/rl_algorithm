from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np
import torch

from .base import Evaluator, AeroOutput


@dataclass
class JsonScaler:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    use_log_re: bool

    @classmethod
    def from_json(cls, path: Path) -> "JsonScaler":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            x_mean=np.asarray(raw["x_mean"], dtype=np.float32),
            x_scale=np.asarray(raw["x_scale"], dtype=np.float32),
            y_mean=np.asarray(raw["y_mean"], dtype=np.float32),
            y_std=np.asarray(raw["y_std"], dtype=np.float32),
            use_log_re=bool(raw.get("use_log_re", True)),
        )

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / np.where(self.x_scale == 0.0, 1.0, self.x_scale)

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


@dataclass
class SurrogateArtifacts:
    model: torch.nn.Module
    scaler: JsonScaler


class SurrogateEvaluator(Evaluator):
    name = "surrogate"

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        scaler_json_path: str = "checkpoints/scalers.json",
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.scaler_json_path = Path(scaler_json_path)
        self.device = torch.device(device)
        self.artifacts = self._load_artifacts()

    def _load_model(self) -> torch.nn.Module:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found: {self.checkpoint_path}")

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(ckpt, torch.nn.Module):
                model = ckpt
            elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
                model = ckpt["model"]
            else:
                model = torch.jit.load(str(self.checkpoint_path), map_location=self.device)
        except Exception:
            model = torch.jit.load(str(self.checkpoint_path), map_location=self.device)

        model.eval()
        return model

    def _load_artifacts(self) -> SurrogateArtifacts:
        if not self.scaler_json_path.exists():
            raise FileNotFoundError(f"scalers.json not found: {self.scaler_json_path}")
        scaler = JsonScaler.from_json(self.scaler_json_path)
        model = self._load_model()
        return SurrogateArtifacts(model=model, scaler=scaler)

    def _featurize(self, cst: np.ndarray, aoa: float, re: float) -> np.ndarray:
        re_feat = np.log10(re) if self.artifacts.scaler.use_log_re else re
        x_raw = np.concatenate(
            [cst.astype(np.float32), np.array([aoa, re_feat], dtype=np.float32)], axis=0
        ).reshape(1, -1)
        x = self.artifacts.scaler.transform_x(x_raw)
        return x.astype(np.float32)

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        x = self._featurize(cst, aoa, re)
        with torch.no_grad():
            xin = torch.from_numpy(x).to(self.device)
            y = self.artifacts.model(xin)
            if isinstance(y, tuple):
                y = y[0]
            y = y.detach().cpu().numpy()

        y = self.artifacts.scaler.inverse_y(y)
        cl, cd, cm = [float(v) for v in y.reshape(-1)[:3]]
        cd = max(1e-6, cd)

        tc = float(np.clip(np.max(cst[:4] - cst[4:]) + 0.12, 0.02, 0.30))
        is_valid = bool(np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm) and np.isfinite(tc))
        status = "ok" if is_valid else "nan_output"
        return AeroOutput(cl=cl, cd=cd, cm=cm, tc=tc, is_geometry_valid=is_valid, solver_status=status)
