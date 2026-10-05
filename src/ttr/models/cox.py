"""Penalised Cox proportional-hazards model, stratified by landmark (ADR 0003)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from ttr.models.base import Preprocessor, SurvivalData, SurvivalModel


class StratifiedCox(SurvivalModel):
    """Separate baseline hazard per landmark; shared coefficients.

    New landmarks have no baseline of their own, so event probabilities use the average of the
    training landmarks' baseline cumulative hazards. Risk ranking is unaffected.
    """

    name = "cox"

    def __init__(self, penalizer: float = 0.05, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def fit(self, data: SurvivalData, valid: SurvivalData | None = None) -> StratifiedCox:
        self.pre_ = Preprocessor().fit(data.X)
        Z = self.pre_.transform(data.X)
        df = Z.assign(
            _time=data.time,
            _event=data.event,
            _landmark=pd.Series(data.landmark).astype(str).to_numpy(),
        )
        kwargs: dict[str, object] = {}
        if data.weight is not None:
            df["_weight"] = data.weight
            kwargs = {"weights_col": "_weight", "robust": True}
        self.model_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio).fit(
            df, duration_col="_time", event_col="_event", strata=["_landmark"], **kwargs
        )
        return self

    def _linear(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.pre_.transform(X)[self.model_.params_.index]
        centred = Z.to_numpy() - self.model_._norm_mean.to_numpy()
        out: np.ndarray = centred @ self.model_.params_.to_numpy()
        return out

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self._linear(X)

    def predict_event_prob(self, X: pd.DataFrame, horizon: float) -> np.ndarray:
        H0 = self.model_.baseline_cumulative_hazard_
        h = H0[H0.index <= horizon].ffill().iloc[-1].mean() if (H0.index <= horizon).any() else 0
        return 1 - np.exp(-float(h) * np.exp(self._linear(X)))

    def coefficients(self) -> pd.Series:
        coef: pd.Series = self.model_.params_.sort_values(key=np.abs, ascending=False)
        return coef

    def params(self) -> dict[str, object]:
        return {"penalizer": self.penalizer, "l1_ratio": self.l1_ratio}
