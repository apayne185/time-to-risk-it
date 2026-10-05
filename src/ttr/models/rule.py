"""Operator-style rule baseline: the bar a learned model has to clear.

Scores players on three signals from the EDA (betting frequency, live-action share and recent
escalation) as the mean of their percentile ranks against the training population. Nothing is
learned from outcomes except the mapping from score decile to event probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from ttr.models.base import SurvivalData, SurvivalModel

RULE_FEATURES = ("active_frac_30d", "live_day_share_90d", "bet_days_trend_30_90")


class RuleBaseline(SurvivalModel):
    name = "rule_baseline"

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def fit(self, data: SurvivalData, valid: SurvivalData | None = None) -> RuleBaseline:
        X = data.X[list(RULE_FEATURES)].fillna(0.0)
        self.reference_ = {c: np.sort(X[c].to_numpy()) for c in RULE_FEATURES}
        score = self.predict_risk(data.X)
        self.edges_ = np.unique(np.quantile(score, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        self.bin_data_ = (np.digitize(score, self.edges_), data.time, data.event)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        X = X[list(RULE_FEATURES)].fillna(0.0)
        ranks = [
            np.searchsorted(self.reference_[c], X[c].to_numpy(), side="right")
            / len(self.reference_[c])
            for c in RULE_FEATURES
        ]
        return np.mean(ranks, axis=0)

    def predict_event_prob(self, X: pd.DataFrame, horizon: float) -> np.ndarray:
        bins, time, event = self.bin_data_
        probs = np.zeros(len(self.edges_) + 1)
        for b in range(len(probs)):
            mask = bins == b
            if mask.any():
                km = KaplanMeierFitter().fit(time[mask], event[mask])
                probs[b] = 1 - float(km.survival_function_at_times(horizon).iloc[0])
        out: np.ndarray = probs[np.digitize(self.predict_risk(X), self.edges_)]
        return out

    def params(self) -> dict[str, object]:
        return {"features": ",".join(RULE_FEATURES), "n_bins": self.n_bins}
