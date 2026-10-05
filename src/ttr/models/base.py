"""Shared model interface, preprocessing and baseline-hazard estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SurvivalData:
    """Features plus right-censored outcome for a set of landmark rows."""

    X: pd.DataFrame
    time: np.ndarray
    event: np.ndarray
    landmark: np.ndarray
    weight: np.ndarray | None = None

    @classmethod
    def from_table(cls, table: pd.DataFrame, features: list[str]) -> SurvivalData:
        return cls(
            X=table[features].reset_index(drop=True),
            time=table["time"].to_numpy(dtype=float),
            event=table["event"].to_numpy(dtype=int),
            landmark=table["landmark"].to_numpy(),
            weight=table["weight"].to_numpy(dtype=float) if "weight" in table else None,
        )

    def __len__(self) -> int:
        return len(self.X)


class SurvivalModel(ABC):
    """Fit on right-censored data; score risk (higher = sooner event) and event probability."""

    name: str

    @abstractmethod
    def fit(self, data: SurvivalData, valid: SurvivalData | None = None) -> SurvivalModel:
        """Fit on ``data``; ``valid`` may be used for early stopping, never for fitting."""

    @abstractmethod
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Monotone risk score: only the ordering is meaningful."""

    @abstractmethod
    def predict_event_prob(self, X: pd.DataFrame, horizon: float) -> np.ndarray:
        """P(event within ``horizon`` days) in the *training sample's* event-rate scale.

        The sample is case-control: these are not population probabilities until reweighted
        or recalibrated (ADR 0003).
        """

    def params(self) -> dict[str, object]:
        return {}


@dataclass
class Preprocessor:
    """Median imputation with missingness indicators, winsorising and standardisation.

    Fitted on training data only. Tree models do not need it; linear (Cox) models do.
    """

    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    medians_: pd.Series = field(init=False)
    lower_: pd.Series = field(init=False)
    upper_: pd.Series = field(init=False)
    mean_: pd.Series = field(init=False)
    scale_: pd.Series = field(init=False)
    indicators_: list[str] = field(init=False)

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        self.indicators_ = [c for c in X.columns if X[c].isna().any()]
        self.medians_ = X.median()
        filled = X.fillna(self.medians_)
        self.lower_ = filled.quantile(self.clip_quantiles[0])
        self.upper_ = filled.quantile(self.clip_quantiles[1])
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        self.mean_ = clipped.mean()
        self.scale_ = clipped.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        filled = X.fillna(self.medians_).clip(self.lower_, self.upper_, axis=1)
        out = (filled - self.mean_) / self.scale_
        for c in self.indicators_:
            out[f"{c}__missing"] = X[c].isna().astype(float).to_numpy()
        return out


def breslow_cumulative_hazard(
    time: np.ndarray, event: np.ndarray, risk: np.ndarray, weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow estimate of the baseline cumulative hazard for a proportional-hazards score.

    Args:
        risk: exp(linear predictor) per row.
    Returns:
        (event times, cumulative hazard at each).
    """
    w = np.ones_like(time, dtype=float) if weight is None else weight
    order = np.argsort(time)
    t, e, r, w = time[order], event[order], risk[order] * w[order], w[order]
    # Risk set at time s: rows with time >= s. Reverse cumulative sum gives it for every row.
    at_risk = np.cumsum(r[::-1])[::-1]
    times = np.unique(t[e == 1])
    first = np.searchsorted(t, times, side="left")
    deaths = np.array([w[(t == s) & (e == 1)].sum() for s in times])
    return times, np.cumsum(deaths / at_risk[first])


def hazard_at(times: np.ndarray, cumhaz: np.ndarray, horizon: float) -> float:
    idx = np.searchsorted(times, horizon, side="right") - 1
    return float(cumhaz[idx]) if idx >= 0 else 0.0
