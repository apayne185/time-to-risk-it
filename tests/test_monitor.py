import numpy as np
import pandas as pd
import pytest

from ttr.monitor import feature_drift, psi, status


def test_psi_zero_for_identical_and_large_for_shifted() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(size=5000))
    assert psi(ref, ref) == pytest.approx(0, abs=1e-9)
    assert psi(ref, pd.Series(rng.normal(size=5000))) < 0.02
    assert psi(ref, pd.Series(rng.normal(1.0, 1, 5000))) > 0.25


def test_psi_counts_missingness_changes() -> None:
    ref = pd.Series(np.r_[np.arange(900.0), [np.nan] * 100])
    cur = pd.Series(np.r_[np.arange(500.0), [np.nan] * 500])
    assert psi(ref, cur) > 0.25


def test_feature_drift_table() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    cur = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(2, 1, 2000)})
    out = feature_drift(ref, cur, ["a", "b"])
    assert out["feature"].tolist() == ["b", "a"]
    assert out["status"].tolist() == ["major", "stable"]
    assert status(0.15) == "moderate"
