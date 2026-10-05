"""Weighted recalibration and calibration diagnostics for a binary horizon outcome."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    out: np.ndarray = np.log(p / (1 - p))
    return out


@dataclass
class PlattRecalibrator:
    """Weighted logistic recalibration: p' = sigmoid(a + b * logit(p)).

    Two parameters, fitted on out-of-fold predictions with case-control weights, map the
    sample-scale probability onto the population scale (and fix over/under-confidence).
    """

    intercept_: float = 0.0
    slope_: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> PlattRecalibrator:
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(_logit(p)[:, None], y, sample_weight=w)
        self.intercept_ = float(lr.intercept_[0])
        self.slope_ = float(lr.coef_[0, 0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        out: np.ndarray = 1 / (1 + np.exp(-(self.intercept_ + self.slope_ * _logit(p))))
        return out


def intercept_shift(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Shift ``b`` on the logit scale so that the weighted mean of sigmoid(logit(p) + b) equals
    the weighted observed rate (calibration-in-the-large only; ranking is untouched)."""
    from scipy.optimize import brentq

    target = float(np.average(y, weights=w))
    if target <= 0:
        return 0.0
    z = _logit(p)

    def gap(b: float) -> float:
        return float(np.average(1 / (1 + np.exp(-(z + b))), weights=w)) - target

    return float(brentq(gap, -15, 15))


def apply_shift(p: np.ndarray, shift: float) -> np.ndarray:
    out: np.ndarray = 1 / (1 + np.exp(-(_logit(p) + shift)))
    return out


@dataclass(frozen=True)
class CalibrationSummary:
    observed_rate: float
    predicted_rate: float
    o_e_ratio: float
    calibration_intercept: float
    calibration_slope: float
    brier: float


def calibration_summary(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> CalibrationSummary:
    observed = float(np.average(y, weights=w))
    predicted = float(np.average(p, weights=w))
    # Slope/intercept of a weighted logistic fit of the outcome on logit(p): ideal (0, 1).
    fit = PlattRecalibrator().fit(p, y, w)
    return CalibrationSummary(
        observed_rate=observed,
        predicted_rate=predicted,
        o_e_ratio=observed / predicted if predicted > 0 else float("nan"),
        calibration_intercept=fit.intercept_,
        calibration_slope=fit.slope_,
        brier=float(np.average((p - y) ** 2, weights=w)),
    )


def reliability_table(
    p: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Weighted observed vs predicted rate in bins of equal weighted size (risk quantiles)."""
    order = np.argsort(p)
    cum = np.cumsum(w[order]) / w.sum()
    bins = np.minimum((cum * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame(
        {
            "bin": bins,
            "pw": p[order] * w[order],
            "yw": y[order] * w[order],
            "w": w[order],
            "y": y[order],
        }
    )
    sums = df.groupby("bin")[["pw", "yw", "w", "y"]].sum()
    return pd.DataFrame(
        {
            "predicted": sums["pw"] / sums["w"],
            "observed": sums["yw"] / sums["w"],
            "rows": df.groupby("bin").size(),
            "events": sums["y"],
        }
    )
