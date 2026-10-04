from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ttr.analysis.indices import betting_indices, index_columns
from ttr.analysis.replication import assign_index_dates, discriminant_analysis
from ttr.analysis.report import build_eda_report
from ttr.config import load_data_config
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate

TS = pd.Timestamp


def _daily(rows: list[tuple[int, str, int, float, float, int]]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows, columns=["user_id", "date", "product_type", "turnover", "hold", "n_bets"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return clean_daily(df, load_data_config())


def test_indices_follow_paper_definitions() -> None:
    daily = _daily(
        [
            (1, "2008-01-01", 2, 10.0, 10.0, 2),
            (1, "2008-01-05", 2, 30.0, -20.0, 3),
            (1, "2008-01-10", 2, 0.0, 15.0, 0),  # settlement: counts in loss and duration only
        ]
    )
    row = betting_indices(daily).loc[1]
    assert row["sum_stakes_liveaction"] == 40
    assert row["sum_bets_liveaction"] == 5
    assert row["bettingdays_liveaction"] == 2
    assert row["duration_liveaction"] == 10
    assert row["frequency_liveaction"] == pytest.approx(0.2)
    assert row["bets_per_day_liveaction"] == 2.5
    assert row["euros_per_bet_liveaction"] == 8
    assert row["net_loss_liveaction"] == 5
    assert row["percent_lost_liveaction"] == pytest.approx(5 / 40)


def test_cutoff_excludes_activity_on_and_after_date() -> None:
    daily = _daily([(1, "2008-01-01", 1, 10.0, 10.0, 1), (1, "2008-02-01", 1, 99.0, 99.0, 9)])
    row = betting_indices(daily, cutoff=pd.Series({1: TS("2008-02-01")})).loc[1]
    assert row["sum_stakes_fixedodds"] == 10


def test_zero_stake_ratios_are_missing_not_infinite() -> None:
    row = betting_indices(_daily([(1, "2008-01-01", 1, 0.0, 0.0, 2)])).loc[1]
    assert np.isnan(row["percent_lost_fixedodds"])
    assert np.isfinite(row["euros_per_bet_fixedodds"])


def test_controls_borrow_index_date_from_same_deposit_day_case() -> None:
    players = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "rg_case": [1, 0, 1, 0],
            "first_deposit_date": [TS("2005-01-01")] * 2 + [TS("2006-01-01")] * 2,
            "rg_first_date": [TS("2009-01-01"), pd.NaT, TS("2009-06-01"), pd.NaT],
        }
    )
    idx = assign_index_dates(players)
    assert idx[2] == TS("2009-01-01")
    assert idx[4] == TS("2009-06-01")


def test_discriminant_analysis_finds_planted_signal() -> None:
    rng = np.random.default_rng(0)
    n = 400
    y = pd.Series(np.repeat([1, 0], n // 2))
    X = pd.DataFrame(rng.gamma(2.0, 1.0, size=(n, 3)), columns=["noise_a", "signal", "noise_b"])
    X["signal"] += 3 * y
    result = discriminant_analysis(X, y, "planted")
    assert result.cv_auc > 0.9
    assert result.structure.index[0] == "signal"


@pytest.fixture(scope="module")
def synthetic_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_data_config()
    tables = generate(SyntheticSpec(n_pairs=150, seed=11), cfg)
    daily = clean_daily(tables["daily"], cfg)
    return daily, build_players(tables["demographics"], tables["rg"], daily)


def test_eda_report_on_synthetic(
    synthetic_clean: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    daily, players = synthetic_clean
    out = build_eda_report(daily, players, tmp_path)
    names = [r.name for r in out.results]
    assert names == ["full_history", "pre_index", "pre_index_primary"]
    assert all(p.exists() and p.stat().st_size > 10_000 for p in out.figures)
    text = out.report.read_text()
    assert "Already-closed accounts" in text
    # synthetic cases escalate before their event, so the signal must be found
    assert out.results[1].cv_auc > 0.6


def test_index_columns_cover_27_indices() -> None:
    assert len(index_columns()) == 27
