from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

DEFAULT_FEATURE_ORDER: Tuple[str, ...] = ("gx", "gy", "yaw", "pitch", "roll", "tz")
MODEL_VERSION = 1


class Calibrator:
    """Ridge regression calibrator for gaze mapping."""

    def __init__(self, feature_order: Optional[Sequence[str]] = None) -> None:
        self.feature_order: Tuple[str, ...] = tuple(feature_order or DEFAULT_FEATURE_ORDER)
        self.W: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.alpha: float = 0.5
        self.version: int = MODEL_VERSION

    def is_ready(self) -> bool:
        return self.W is not None and self.mu is not None and self.sigma is not None

    def fit(self, X: Iterable[Iterable[float]], Y: Iterable[Iterable[float]], alpha: float = 0.5) -> bool:
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(Y, dtype=np.float64)
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            return False
        if x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[0] < 8:
            return False
        if y_arr.shape[1] != 2 or x_arr.shape[1] != len(self.feature_order):
            return False
        if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
            return False

        mu = np.mean(x_arr, axis=0)
        sigma = np.std(x_arr, axis=0)
        sigma[sigma < 1e-6] = 1.0
        x_norm = (x_arr - mu) / sigma
        x_bias = np.hstack((np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm))

        alpha = max(float(alpha), 0.0)
        reg = np.eye(x_bias.shape[1], dtype=np.float64)
        reg[0, 0] = 0.0
        gram = x_bias.T @ x_bias + alpha * reg
        rhs = x_bias.T @ y_arr
        try:
            w = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(gram) @ rhs

        if not np.isfinite(w).all():
            return False

        self.W = w.astype(np.float64)
        self.mu = mu.astype(np.float64)
        self.sigma = sigma.astype(np.float64)
        self.alpha = alpha
        self.version = MODEL_VERSION
        return True

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        if not self.is_ready() or self.W is None or self.mu is None or self.sigma is None:
            raise ValueError("Calibrator is not fitted.")

        x_arr = np.asarray(X, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2 or x_arr.shape[1] != len(self.feature_order):
            raise ValueError("Input features have invalid shape.")
        if not np.isfinite(x_arr).all():
            raise ValueError("Input features contain non-finite values.")

        x_norm = (x_arr - self.mu) / self.sigma
        x_bias = np.hstack((np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm))
        pred = x_bias @ self.W
        return np.clip(pred, 0.0, 1.0).astype(np.float32)

    def save_npz(self, path: str) -> None:
        if not self.is_ready() or self.W is None or self.mu is None or self.sigma is None:
            raise ValueError("Calibrator is not fitted.")

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        np.savez(
            path,
            W=self.W.astype(np.float64),
            mu=self.mu.astype(np.float64),
            sigma=self.sigma.astype(np.float64),
            alpha=np.array([self.alpha], dtype=np.float64),
            feature_order=np.array(self.feature_order, dtype="<U32"),
            version=np.array([self.version], dtype=np.int32),
        )

    @classmethod
    def load_npz(cls, path: str) -> "Calibrator":
        with np.load(path, allow_pickle=False) as data:
            feature_order = tuple(str(v) for v in data["feature_order"].tolist())
            calibrator = cls(feature_order=feature_order)
            calibrator.W = np.asarray(data["W"], dtype=np.float64)
            calibrator.mu = np.asarray(data["mu"], dtype=np.float64)
            calibrator.sigma = np.asarray(data["sigma"], dtype=np.float64)
            calibrator.alpha = float(np.asarray(data["alpha"], dtype=np.float64).reshape(-1)[0])
            calibrator.version = int(np.asarray(data["version"], dtype=np.int32).reshape(-1)[0])

        if calibrator.W is None or calibrator.mu is None or calibrator.sigma is None:
            raise ValueError("Model file is incomplete.")
        if calibrator.W.ndim != 2 or calibrator.W.shape[1] != 2:
            raise ValueError("Model weights shape is invalid.")
        if calibrator.mu.ndim != 1 or calibrator.sigma.ndim != 1:
            raise ValueError("Normalization vectors are invalid.")
        if calibrator.mu.shape[0] != len(calibrator.feature_order):
            raise ValueError("Feature order size does not match normalization vectors.")
        if calibrator.W.shape[0] != len(calibrator.feature_order) + 1:
            raise ValueError("Weights size does not match feature count.")
        if not np.isfinite(calibrator.W).all():
            raise ValueError("Model weights contain non-finite values.")
        if not np.isfinite(calibrator.mu).all() or not np.isfinite(calibrator.sigma).all():
            raise ValueError("Model normalization contains non-finite values.")
        calibrator.sigma[calibrator.sigma < 1e-6] = 1.0

        return calibrator
