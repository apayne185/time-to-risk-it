"""Ranking metrics for right-censored, landmark-stacked predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: share of comparable pairs where the higher-risk row has the earlier event."""
    if event.sum() == 0:
        return float("nan")
    return float(concordance_index(time, -risk, event))


def within_landmark_c(
    time: np.ndarray, event: np.ndarray, risk: np.ndarray, landmark: np.ndarray, min_events: int = 5
) -> float:
    """Event-weighted mean of Harrell's C computed separately at each landmark.

    Pooled C rewards a model for ranking whole landmarks against each other, which in this
    case-control sample reflects the sampling design, not player behaviour (ADR 0003).
    """
    scores, weights = [], []
    for lm in np.unique(landmark):
        m = landmark == lm
        n_events = int(event[m].sum())
        if n_events >= min_events and n_events < m.sum():
            scores.append(harrell_c(time[m], event[m], risk[m]))
            weights.append(n_events)
    return float(np.average(scores, weights=weights)) if scores else float("nan")


@dataclass(frozen=True)
class Metrics:
    c_within_landmark: float
    c_pooled: float
    mean_pred: float
    event_rate: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v for k, v in self.__dict__.items()}


def evaluate(
    time: np.ndarray, event: np.ndarray, risk: np.ndarray, prob: np.ndarray, landmark: np.ndarray
) -> Metrics:
    return Metrics(
        c_within_landmark=within_landmark_c(time, event, risk, landmark),
        c_pooled=harrell_c(time, event, risk),
        mean_pred=float(prob.mean()),
        event_rate=float(event.mean()),
    )


def bootstrap_ci(
    frame: pd.DataFrame, n_boot: int = 200, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Player-level bootstrap CI for within-landmark C.

    ``frame`` has columns user_id, time, event, risk, landmark. Players are resampled (all their
    landmark rows together), because rows of the same player are not independent.
    """
    rng = np.random.default_rng(seed)
    groups = {uid: idx for uid, idx in frame.groupby("user_id").indices.items()}
    ids = np.array(list(groups))
    stats = []
    for _ in range(n_boot):
        rows = np.concatenate([groups[u] for u in rng.choice(ids, size=len(ids))])
        b = frame.iloc[rows]
        stats.append(
            within_landmark_c(
                b["time"].to_numpy(),
                b["event"].to_numpy(),
                b["risk"].to_numpy(),
                b["landmark"].to_numpy(),
            )
        )
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
