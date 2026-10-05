from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttr.models.base import Preprocessor, SurvivalData, breslow_cumulative_hazard
from ttr.models.cox import StratifiedCox
from ttr.models.rule import RuleBaseline
from ttr.models.xgb import XGBAFT, XGBCox


def _simulate(n: int = 1500, seed: int = 0) -> SurvivalData:
    """Exponential event times whose hazard rises with the rule features; 90-day censoring."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "active_frac_30d": rng.uniform(0, 1, n),
            "live_day_share_90d": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 1, n)),
            "bet_days_trend_30_90": rng.lognormal(0, 0.4, n),
            "noise": rng.normal(size=n),
        }
    )
    lp = 2.0 * X["active_frac_30d"] + 1.0 * X["live_day_share_90d"].fillna(0)
    t_event = rng.exponential(200 / np.exp(lp))
    time = np.minimum(t_event, 90).round().clip(1)
    event = (t_event <= 90).astype(int)
    landmark = rng.choice(pd.to_datetime(["2009-01-01", "2009-02-01"]), n)
    return SurvivalData(X=X, time=time, event=event, landmark=landmark)


def _cindex(model, data: SurvivalData) -> float:  # type: ignore[no-untyped-def]
    from lifelines.utils import concordance_index

    return float(concordance_index(data.time, -model.predict_risk(data.X), data.event))


@pytest.mark.parametrize(
    "model",
    [RuleBaseline(), StratifiedCox(), XGBCox(num_boost_round=200), XGBAFT(num_boost_round=200)],
    ids=lambda m: m.name,
)
def test_models_rank_and_return_probabilities(model) -> None:  # type: ignore[no-untyped-def]
    train, test = _simulate(seed=1), _simulate(seed=2)
    model.fit(train, valid=test if model.name.startswith("xgb") else None)
    assert _cindex(model, test) > 0.6
    p = model.predict_event_prob(test.X, 90)
    assert p.shape == (len(test),)
    assert np.all((p >= 0) & (p <= 1))
    # Higher risk score -> higher (or equal) event probability.
    order = np.argsort(model.predict_risk(test.X))
    assert np.corrcoef(np.arange(len(order)), p[order])[0, 1] > 0.5
    # Mean predicted probability is close to the observed 90-day event share (no censoring
    # before 90 days in this simulation).
    assert abs(p.mean() - test.event.mean()) < 0.1


def test_preprocessor_imputes_flags_and_scales() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, 10.0, 10.0, 10.0]})
    pre = Preprocessor(clip_quantiles=(0.0, 1.0)).fit(X)
    Z = pre.transform(X)
    assert list(Z.columns) == ["a", "b", "a__missing"]
    assert Z["a__missing"].tolist() == [0, 0, 1, 0]
    assert not Z.isna().any().any()
    assert Z["b"].abs().max() == 0  # constant column does not blow up


def test_breslow_matches_nelson_aalen_without_covariates() -> None:
    time = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    event = np.array([1, 1, 0, 1, 0])
    times, H = breslow_cumulative_hazard(time, event, np.ones(5))
    assert times.tolist() == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(H, [1 / 5, 1 / 5 + 1 / 4, 1 / 5 + 1 / 4 + 1 / 2])


def test_cox_recovers_effect_direction() -> None:
    model = StratifiedCox(penalizer=0.01).fit(_simulate(n=3000))
    coef = model.coefficients()
    assert coef["active_frac_30d"] > 0
    assert abs(coef["noise"]) < abs(coef["active_frac_30d"]) / 5
