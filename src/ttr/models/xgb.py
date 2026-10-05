"""Gradient-boosted survival models (XGBoost Cox and AFT objectives)."""

from __future__ import annotations

from math import log, sqrt
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import erf

from ttr.models.base import (
    SurvivalData,
    SurvivalModel,
    breslow_cumulative_hazard,
    hazard_at,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "eta": 0.03,
    "max_depth": 3,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "tree_method": "hist",
    "seed": 0,
}


class _XGBSurvival(SurvivalModel):
    objective: str
    eval_metric: str

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.xgb_params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def _dmatrix(self, data: SurvivalData) -> xgb.DMatrix:
        raise NotImplementedError

    def fit(self, data: SurvivalData, valid: SurvivalData | None = None) -> _XGBSurvival:
        self.features_ = list(data.X.columns)
        params = {**self.xgb_params, "objective": self.objective, "eval_metric": self.eval_metric}
        dtrain = self._dmatrix(data)
        evals = [(dtrain, "train")]
        stopping = None
        if valid is not None:
            evals.append((self._dmatrix(valid), "valid"))
            stopping = self.early_stopping_rounds
        self.booster_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=stopping,
            verbose_eval=False,
        )
        self.best_iteration_ = (
            self.booster_.best_iteration if valid is not None else self.num_boost_round - 1
        )
        self._after_fit(data)
        return self

    def _after_fit(self, data: SurvivalData) -> None:
        pass

    def _margin(self, X: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(X[self.features_])
        out: np.ndarray = self.booster_.predict(
            d, output_margin=True, iteration_range=(0, self.best_iteration_ + 1)
        )
        return out

    def importance(self) -> pd.Series:
        gain = self.booster_.get_score(importance_type="total_gain")
        return (
            pd.Series(gain, dtype=float)
            .reindex(self.features_)
            .fillna(0)
            .sort_values(ascending=False)
        )

    def params(self) -> dict[str, object]:
        return {**self.xgb_params, "best_iteration": self.best_iteration_}


class XGBCox(_XGBSurvival):
    """XGBoost with the Cox partial-likelihood objective; Breslow baseline for probabilities."""

    name = "xgb_cox"
    objective = "survival:cox"
    eval_metric = "cox-nloglik"

    def _dmatrix(self, data: SurvivalData) -> xgb.DMatrix:
        # survival:cox encodes censoring as a negative time.
        label = np.where(data.event == 1, data.time, -data.time)
        return xgb.DMatrix(data.X, label=label, weight=data.weight)

    def _after_fit(self, data: SurvivalData) -> None:
        self.base_times_, self.base_cumhaz_ = breslow_cumulative_hazard(
            data.time, data.event, np.exp(self._margin(data.X)), data.weight
        )

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self._margin(X)

    def predict_event_prob(self, X: pd.DataFrame, horizon: float) -> np.ndarray:
        h0 = hazard_at(self.base_times_, self.base_cumhaz_, horizon)
        return 1 - np.exp(-h0 * np.exp(self._margin(X)))


class XGBAFT(_XGBSurvival):
    """XGBoost accelerated failure time model: log T = f(x) + sigma * Normal noise."""

    name = "xgb_aft"
    objective = "survival:aft"
    eval_metric = "aft-nloglik"

    def __init__(
        self, params: dict[str, Any] | None = None, sigma: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(
            {
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": sigma,
                **(params or {}),
            },
            **kwargs,
        )
        self.sigma = sigma

    def _dmatrix(self, data: SurvivalData) -> xgb.DMatrix:
        d = xgb.DMatrix(data.X, weight=data.weight)
        d.set_float_info("label_lower_bound", data.time)
        d.set_float_info("label_upper_bound", np.where(data.event == 1, data.time, np.inf))
        return d

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return -self._margin(X)  # shorter predicted time = higher risk

    def predict_event_prob(self, X: pd.DataFrame, horizon: float) -> np.ndarray:
        z = (log(horizon) - self._margin(X)) / self.sigma
        out: np.ndarray = 0.5 * (1 + erf(z / sqrt(2)))
        return out
