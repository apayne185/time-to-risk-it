from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from lifelines.utils import concordance_index

from ttr.models.base import SurvivalData
from ttr.models.torch_hazard import DiscreteTimeHazardNet, interval_targets, masked_nll

EDGES = (0, 15, 30, 45, 60, 75, 90)


def _simulate(n: int = 2000, seed: int = 0) -> SurvivalData:
    """Exponential event times whose hazard rises with two features; 90-day censoring."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, n),
            "b": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 1, n)),
            "noise": rng.normal(size=n),
        }
    )
    lp = 2.0 * X["a"] + 1.0 * X["b"].fillna(0)
    t_event = rng.exponential(200 / np.exp(lp))
    time = np.minimum(t_event, 90).round().clip(1)
    return SurvivalData(X=X, time=time, event=(t_event <= 90).astype(int), landmark=np.zeros(n))


@pytest.mark.parametrize(
    ("time", "event", "y", "mask"),
    [
        (20, 1, [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]),  # event in the 2nd interval
        (1, 1, [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
        (90, 1, [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]),
        (30, 0, [0] * 6, [1, 1, 0, 0, 0, 0]),  # censored on an edge: two intervals survived
        (25, 0, [0] * 6, [1, 0, 0, 0, 0, 0]),  # censored mid-interval: only completed ones
        (90, 0, [0] * 6, [1] * 6),
    ],
)
def test_interval_targets(time: int, event: int, y: list[int], mask: list[int]) -> None:
    got_y, got_mask = interval_targets(np.array([float(time)]), np.array([event]), EDGES)
    assert got_y[0].tolist() == y and got_mask[0].tolist() == mask


def test_masked_nll_is_the_survival_likelihood() -> None:
    hazards = np.array([[0.1, 0.2, 0.3, 0.1, 0.1, 0.1], [0.05, 0.05, 0.1, 0.2, 0.2, 0.2]])
    y, mask = interval_targets(np.array([20.0, 30.0]), np.array([1, 0]), EDGES)
    logits = torch.logit(torch.tensor(hazards, dtype=torch.float32))
    nll = float(masked_nll(logits, torch.from_numpy(y), torch.from_numpy(mask), torch.ones(2)))
    # row 1: survive interval 1, event in interval 2; row 2: survive intervals 1 and 2
    expected = -(math.log(0.9) + math.log(0.2) + math.log(0.95) + math.log(0.95)) / 2
    assert nll == pytest.approx(expected, rel=1e-5)


@pytest.fixture(scope="module")
def fitted() -> tuple[DiscreteTimeHazardNet, SurvivalData]:
    train, valid = _simulate(seed=1), _simulate(seed=2)
    return DiscreteTimeHazardNet(hidden=32, max_epochs=100, patience=10).fit(train, valid), valid


def test_ranks_and_calibrates_on_simulated_data(
    fitted: tuple[DiscreteTimeHazardNet, SurvivalData],
) -> None:
    model, test = fitted
    c = concordance_index(test.time, -model.predict_risk(test.X), test.event)
    assert c > 0.65
    p90, p30 = model.predict_event_prob(test.X, 90), model.predict_event_prob(test.X, 30)
    assert np.all((p30 >= 0) & (p30 <= p90 + 1e-9) & (p90 <= 1))
    assert abs(p90.mean() - test.event.mean()) < 0.05
    observed_30 = ((test.event == 1) & (test.time <= 30)).mean()
    assert abs(p30.mean() - observed_30) < 0.05


def test_early_stopping_and_fixed_epoch_refit(
    fitted: tuple[DiscreteTimeHazardNet, SurvivalData],
) -> None:
    model, _ = fitted
    assert 0 <= model.best_iteration_ < 100
    assert min(model.history_) == model.history_[model.best_iteration_]
    refit = DiscreteTimeHazardNet(hidden=32, num_boost_round=model.best_iteration_ + 1)
    refit.fit(_simulate(seed=1))
    assert refit.best_iteration_ == model.best_iteration_
    assert refit.history_ == []  # no validation set used when the epoch count is fixed


def test_deterministic_for_seed() -> None:
    data = _simulate(n=500, seed=3)
    a = DiscreteTimeHazardNet(hidden=16, max_epochs=5).fit(data).predict_risk(data.X)
    b = DiscreteTimeHazardNet(hidden=16, max_epochs=5).fit(data).predict_risk(data.X)
    np.testing.assert_allclose(a, b)
