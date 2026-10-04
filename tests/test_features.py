from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ttr.config import PROJECT_ROOT, load_data_config, load_landmarks_config
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate
from ttr.features import FEATURES, build_feature_table, compute_features, feature_docs_markdown
from ttr.labels import get_definition
from ttr.landmarks import build_landmarks

TS = pd.Timestamp
CFG = load_data_config()


def _synthetic(seed: int, n_pairs: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = generate(SyntheticSpec(n_pairs=n_pairs, seed=seed), CFG)
    daily = clean_daily(tables["daily"], CFG)
    players = build_players(tables["demographics"], tables["rg"], daily)
    lm = build_landmarks(players, daily, CFG, load_landmarks_config(), get_definition("primary"))
    return daily, lm


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _synthetic(seed=5, n_pairs=80)


def _assert_same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        a.sort_values(["user_id", "landmark"]).reset_index(drop=True),
        b.sort_values(["user_id", "landmark"]).reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
    )


def test_features_ignore_activity_on_or_after_landmark(
    synthetic: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Per landmark, features from the full data equal features from data truncated at it."""
    daily, lm = synthetic
    full = compute_features(daily, lm)
    for landmark, rows in lm.groupby("landmark"):
        truncated = compute_features(daily[daily["date"] < landmark], rows)
        _assert_same(full[full["landmark"] == landmark], truncated)


@settings(max_examples=4, deadline=None)
@given(seed=st.integers(0, 10_000), shift=st.integers(0, 60))
def test_future_activity_cannot_change_features(seed: int, shift: int) -> None:
    """Injecting extreme activity on/after a landmark must not move that landmark's features."""
    daily, lm = _synthetic(seed)
    if lm.empty:
        return
    base = compute_features(daily, lm)
    for landmark, rows in lm.groupby("landmark"):
        future = rows[["user_id"]].assign(
            date=pd.Timestamp(landmark) + pd.Timedelta(days=shift),  # type: ignore[arg-type]
            product_type=2,
            n_bets=500,
            turnover=1e5,
            hold=5e4,
            product_family="live_action",
            money_valid=True,
            is_bet_day=True,
        )
        poisoned = pd.concat([daily, future], ignore_index=True)
        _assert_same(base[base["landmark"] == landmark], compute_features(poisoned, rows))


def test_hand_computed_features() -> None:
    rows = [  # (date, product_type, family, n_bets, turnover, hold)
        ("2008-12-10", 1, "fixed_odds", 2, 10.0, 10.0),  # lost 10
        ("2008-12-20", 2, "live_action", 4, 30.0, -5.0),  # stake up after a loss; won 5
        ("2008-12-25", 8, "casino", 10, 5.0, 5.0),  # first ever casino play
        ("2009-01-01", 2, "live_action", 9, 99.0, 99.0),  # on the landmark: must be ignored
    ]
    daily = pd.DataFrame(
        [
            {
                "user_id": 1,
                "date": TS(d),
                "product_type": p,
                "product_family": f,
                "n_bets": n,
                "turnover": t,
                "hold": h,
                "money_valid": True,
                "is_bet_day": True,
            }
            for d, p, f, n, t, h in rows
        ]
    )
    lm = pd.DataFrame(
        {
            "user_id": [1],
            "landmark": [TS("2009-01-01")],
            "days_since_last_bet": [7],
            "tenure_days": [100],
        }
    )
    f = compute_features(daily, lm).iloc[0]
    assert f["bet_days_30d"] == 3
    assert f["active_frac_30d"] == pytest.approx(3 / 30)
    assert f["live_day_share_30d"] == pytest.approx(1 / 3)
    assert f["sports_bets_per_day_30d"] == 3  # (2 + 4) bets over 2 sportsbook days
    assert f["euros_per_sports_bet_30d"] == pytest.approx(40 / 6)
    assert f["live_stake_share_30d"] == pytest.approx(30 / 40)
    assert f["log_stakes_30d"] == pytest.approx(np.log1p(45))
    assert f["pct_lost_30d"] == pytest.approx(10 / 45)
    assert f["chase_rate_90d"] == 1  # only 12-20 follows a loss (12-25 follows a win); stake rose
    assert f["max_families_30d"] == 1
    assert f["casino_started_30"] == 1
    assert f["days_since_first_live"] == 12


def test_inactive_window_gives_zero_counts_and_missing_ratios() -> None:
    daily = pd.DataFrame(
        [
            {
                "user_id": 1,
                "date": TS("2008-06-01"),
                "product_type": 1,
                "product_family": "fixed_odds",
                "n_bets": 1,
                "turnover": 5.0,
                "hold": 5.0,
                "money_valid": True,
                "is_bet_day": True,
            }
        ]
    )
    lm = pd.DataFrame(
        {
            "user_id": [1],
            "landmark": [TS("2009-01-01")],
            "days_since_last_bet": [214],
            "tenure_days": [300],
        }
    )
    f = compute_features(daily, lm).iloc[0]
    assert f["bet_days_30d"] == 0 and f["bet_days_365d"] == 1
    assert np.isnan(f["sports_bets_per_day_30d"])


def test_feature_table_keeps_every_landmark_row(
    synthetic: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    daily, lm = synthetic
    table = build_feature_table(lm, daily)
    assert len(table) == len(lm)
    assert set(FEATURES) <= set(table.columns)
    assert {"event", "time", "split", "fold"} <= set(table.columns)


def test_no_demographic_features() -> None:
    banned = ("age", "gender", "country", "language", "site", "birth", "demographic")
    assert not [f for f in FEATURES if any(b in f for b in banned)]


def test_feature_docs_up_to_date() -> None:
    docs = PROJECT_ROOT / "docs" / "features.md"
    assert docs.read_text() == feature_docs_markdown(), "run `uv run ttr features --docs`"
