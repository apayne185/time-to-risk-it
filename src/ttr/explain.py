"""Per-player feature contributions to the risk score, oriented so positive = more risk.

- XGBoost: exact TreeSHAP values from XGBoost itself (``pred_contribs``).
- Cox: linear contributions, coefficient x (standardised value - training mean). Missingness
  indicators are folded back into their feature.
- Models with their own ``contributions`` method (the PyTorch hazard net: gradient x input).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ttr.models.base import SurvivalModel
from ttr.models.cox import StratifiedCox
from ttr.models.xgb import XGBAFT, XGBCox


def contributions(model: SurvivalModel, X: pd.DataFrame) -> pd.DataFrame:
    """One column per input feature; rows sum (with an intercept) to the risk score."""
    if isinstance(model, (XGBCox, XGBAFT)):
        d = xgb.DMatrix(X[model.features_])
        raw = model.booster_.predict(
            d, pred_contribs=True, iteration_range=(0, model.best_iteration_ + 1)
        )
        sign = -1.0 if isinstance(model, XGBAFT) else 1.0  # AFT margin is log time
        return pd.DataFrame(sign * raw[:, :-1], columns=model.features_, index=X.index)
    if isinstance(model, StratifiedCox):
        Z = model.pre_.transform(X)[model.model_.params_.index]
        contrib = (Z - model.model_._norm_mean) * model.model_.params_
        base = [c.removesuffix("__missing") for c in contrib.columns]
        out: pd.DataFrame = contrib.T.groupby(base).sum().T.reindex(columns=list(X.columns))
        return out.set_axis(X.index)
    own = getattr(model, "contributions", None)  # e.g. the PyTorch hazard net (optional extra)
    if callable(own):
        result: pd.DataFrame = own(X)
        return result
    raise TypeError(f"no contribution method for {type(model).__name__}")


def global_importance(contrib: pd.DataFrame) -> pd.Series:
    """Mean absolute contribution per feature."""
    return contrib.abs().mean().sort_values(ascending=False)


def top_drivers(
    contrib: pd.DataFrame, X: pd.DataFrame, k: int = 3
) -> list[list[dict[str, object]]]:
    """For each row, the ``k`` features pushing risk up the most, with their values."""
    out = []
    values = contrib.to_numpy()
    for i in range(len(contrib)):
        idx = np.argsort(-values[i])[:k]
        out.append(
            [
                {
                    "feature": contrib.columns[j],
                    "contribution": float(values[i, j]),
                    "value": None
                    if pd.isna(X.iloc[i][contrib.columns[j]])
                    else float(X.iloc[i][contrib.columns[j]]),
                }
                for j in idx
                if values[i, j] > 0
            ]
        )
    return out
