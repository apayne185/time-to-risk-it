import numpy as np
import pandas as pd
import pytest

from ttr.evaluate.metrics import bootstrap_ci, harrell_c, within_landmark_c


def test_harrell_c_perfect_and_reversed() -> None:
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 0])
    risk = np.array([4.0, 3.0, 2.0, 1.0])  # earliest event = highest risk
    assert harrell_c(time, event, risk) == 1.0
    assert harrell_c(time, event, -risk) == 0.0
    assert np.isnan(harrell_c(time, np.zeros(4, dtype=int), risk))


def test_within_landmark_c_ignores_between_landmark_ordering() -> None:
    """A score that only separates landmarks scores 0.5 within them, but well pooled."""
    rng = np.random.default_rng(0)
    n = 400
    landmark = np.repeat([0, 1], n // 2)
    # landmark 1 has many early events, landmark 0 few: pooled C rewards ranking by landmark.
    event = (rng.random(n) < np.where(landmark == 1, 0.8, 0.1)).astype(int)
    time = np.where(event == 1, rng.integers(1, 90, n), 90).astype(float)
    risk = landmark.astype(float)
    assert harrell_c(time, event, risk) > 0.7
    assert within_landmark_c(time, event, risk, landmark) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    n = 300
    risk = rng.normal(size=n)
    event = (rng.random(n) < 1 / (1 + np.exp(-2 * risk))).astype(int)
    time = np.where(event == 1, 10.0, 90.0)
    frame = pd.DataFrame(
        {
            "user_id": np.arange(n) // 2,
            "time": time,
            "event": event,
            "risk": risk,
            "landmark": np.zeros(n),
        }
    )
    lo, hi = bootstrap_ci(frame, n_boot=100)
    point = within_landmark_c(time, event, risk, np.zeros(n))
    assert lo < point < hi
    assert hi - lo < 0.2
