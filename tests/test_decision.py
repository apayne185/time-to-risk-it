import numpy as np
import pandas as pd
import pytest

from ttr.evaluate.calibration import PlattRecalibrator, calibration_summary, reliability_table
from ttr.evaluate.decision import capacity_curve, net_benefit, net_benefit_contact_all
from ttr.evaluate.weights import case_control_weights, control_weight


def test_control_weight_restores_population_rate() -> None:
    w = control_weight(100, 100, 0.01)
    assert 100 / (100 + w * 100) == pytest.approx(0.01)
    table = pd.DataFrame({"user_id": [1, 1, 2, 3], "rg_case": [1, 1, 0, 0]})
    weights = case_control_weights(table, 0.2)
    assert weights[0] == 1 and weights[2] == pytest.approx(control_weight(1, 2, 0.2))
    with pytest.raises(ValueError):
        control_weight(1, 1, 0)


def test_platt_recovers_known_miscalibration() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.01, 0.3, 20_000)
    y = (rng.random(true_p.size) < true_p).astype(int)
    inflated = 1 / (1 + np.exp(-(np.log(true_p / (1 - true_p)) + 1.5)))  # overconfident shift
    cal = PlattRecalibrator().fit(inflated, y, np.ones_like(y, dtype=float))
    assert cal.intercept_ == pytest.approx(-1.5, abs=0.15)
    assert cal.slope_ == pytest.approx(1.0, abs=0.1)
    s = calibration_summary(cal.transform(inflated), y, np.ones(y.size))
    assert s.o_e_ratio == pytest.approx(1.0, abs=0.05)
    table = reliability_table(cal.transform(inflated), y, np.ones(y.size))
    assert len(table) == 10
    assert np.allclose(table["predicted"], table["observed"], atol=0.03)


def test_net_benefit_reference_strategies() -> None:
    y = np.array([1, 0, 0, 0])
    w = np.ones(4)
    t = np.array([0.1, 0.5])
    perfect = net_benefit(y.astype(float), y, w, t)
    assert perfect == pytest.approx([0.25, 0.25])  # every case reached, no false contacts
    contact_all = net_benefit_contact_all(y, w, t)
    assert contact_all[0] == pytest.approx(0.25 - 0.75 * 0.1 / 0.9)
    assert net_benefit(np.ones(4), y, w, t) == pytest.approx(contact_all)


def test_capacity_curve_ranks_within_each_landmark() -> None:
    risk = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([1, 0, 0, 1])
    landmark = np.array([0, 0, 1, 1])
    out = capacity_curve(risk, y, np.ones(4), landmark, (0.5,)).iloc[0]
    # top half at each landmark: landmark 0 catches its case, landmark 1 misses it
    assert out["recall"] == 0.5
    assert out["precision"] == 0.5
    assert out["contacts_per_case"] == 2
