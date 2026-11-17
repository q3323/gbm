from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PlattCalibrator:
    coefficient: float
    intercept: float

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = _logit(probabilities)
        linear = self.coefficient * logits + self.intercept
        return 1.0 / (1.0 + np.exp(-linear))


def fit_platt(
    probabilities: np.ndarray,
    actual_up: np.ndarray,
    *,
    minimum_samples: int = 60,
) -> PlattCalibrator | None:
    if len(probabilities) < minimum_samples:
        return None
    actual = np.asarray(actual_up, dtype=int)
    if len(np.unique(actual)) < 2:
        return None
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(_logit(probabilities).reshape(-1, 1), actual)
    return PlattCalibrator(
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def prequential_platt(
    probabilities: np.ndarray,
    actual_up: np.ndarray,
    trade_dates: list[object],
    *,
    minimum_samples: int = 60,
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    calibrated = raw.copy()
    dates = np.asarray(trade_dates)
    unique_dates = sorted(set(trade_dates))
    for current_date in unique_dates:
        current_mask = dates == current_date
        previous_mask = dates < current_date
        calibrator = fit_platt(
            raw[previous_mask],
            np.asarray(actual_up)[previous_mask],
            minimum_samples=minimum_samples,
        )
        if calibrator is not None:
            calibrated[current_mask] = calibrator.transform(raw[current_mask])
    return calibrated


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))
