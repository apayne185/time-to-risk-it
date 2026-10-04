from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ttr.config import load_data_config, load_landmarks_config
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate
from ttr.labels import get_definition
from ttr.landmarks import build_landmarks, landmark_dates, player_fold, split_bounds

TS = pd.Timestamp
DATA_CFG = load_data_config()
LM_CFG = load_landmarks_config()
PRIMARY = get_definition("primary")


def _player(
    uid: int,
    case: int,
    deposit: str,
    rg: str | None = None,
    event_type: int = 10,
    intervention: int = 1,
) -> dict[str, object]:
    return {
        "user_id": uid,
        "rg_case": case,
        "first_deposit_date": TS(deposit),
        "rg_first_date": TS(rg) if rg else pd.NaT,
        "event_type_first": event_type if case else pd.NA,
        "intervention_type_first": intervention if case else pd.NA,
        "exclusion_reason": pd.NA,
    }


def _build(players: list[dict[str, object]], bets: list[tuple[int, str]]) -> pd.DataFrame:
    p = pd.DataFrame(players).astype(
        {"event_type_first": "Int64", "intervention_type_first": "Int64"}
    )
    d = pd.DataFrame(bets, columns=["user_id", "date"]).assign(
        date=lambda x: pd.to_datetime(x["date"]), is_bet_day=True
    )
    return build_landmarks(p, d, DATA_CFG, LM_CFG, PRIMARY)


def _has(lm: pd.DataFrame, uid: int, landmark: str) -> bool:
    return bool(((lm["user_id"] == uid) & (lm["landmark"] == TS(landmark))).any())


def _row(lm: pd.DataFrame, uid: int, landmark: str) -> pd.Series:
    match = lm[(lm["user_id"] == uid) & (lm["landmark"] == TS(landmark))]
    assert len(match) == 1
    return match.iloc[0]


def test_case_followed_until_event_then_leaves_risk_set() -> None:
    lm = _build([_player(1, 1, "2008-01-01", "2009-01-15")], [(1, "2008-11-20"), (1, "2008-12-20")])
    assert _row(lm, 1, "2008-12-01")[["event", "time"]].tolist() == [1, 46]
    assert _row(lm, 1, "2009-01-01")[["event", "time"]].tolist() == [1, 15]
    assert not _has(lm, 1, "2009-02-01")


def test_event_beyond_horizon_is_censored_at_horizon() -> None:
    lm = _build([_player(1, 1, "2008-01-01", "2009-03-15")], [(1, "2008-10-20")])
    row = _row(lm, 1, "2008-11-01")
    assert (row["event"], row["time"]) == (0, LM_CFG.horizon_days)


def test_outcomes_censored_at_next_split_boundary() -> None:
    # Train landmark 2009-04-01; event in May belongs to the validation period.
    lm = _build([_player(1, 1, "2008-01-01", "2009-05-10")], [(1, "2009-03-20")])
    row = _row(lm, 1, "2009-04-01")
    assert row["split"] == "train"
    assert (row["event"], row["time"], row["follow_end"]) == (0, 30, TS("2009-05-01"))
    assert _row(lm, 1, "2009-05-01")["event"] == 1


def test_controls_censored_at_window_end() -> None:
    lm = _build([_player(2, 0, "2008-01-01")], [(2, "2009-09-20")])
    row = _row(lm, 2, "2009-10-01")
    assert (row["event"], row["follow_end"]) == (0, TS("2009-12-01"))
    assert row["time"] == 61


def test_inactive_players_and_non_harm_cases_are_not_scored() -> None:
    lm = _build(
        [
            _player(1, 0, "2008-01-01"),  # last bet too long ago
            _player(2, 1, "2008-01-01", "2009-01-15", intervention=2),  # re-opening: excluded
            _player(3, 0, "2008-12-15"),
        ],  # deposits mid-window
        [(1, "2008-06-01"), (2, "2008-12-20"), (3, "2008-12-16")],
    )
    assert set(lm["user_id"]) == {3}
    assert lm["landmark"].min() == TS("2009-01-01")


def test_features_window_excludes_landmark_day() -> None:
    lm = _build([_player(1, 0, "2008-01-01")], [(1, "2008-12-01")])
    # A bet on the landmark day itself is not history for that landmark.
    assert not _has(lm, 1, "2008-12-01")
    assert _row(lm, 1, "2009-01-01")["days_since_last_bet"] == 31


def test_split_bounds_and_folds() -> None:
    bounds = split_bounds(LM_CFG, TS(DATA_CFG.event_window.end)).set_index("split")
    assert bounds.loc["train", "censor_at"] == TS("2009-05-01")
    assert bounds.loc["test", "censor_at"] == TS("2009-12-01")
    assert len(landmark_dates(LM_CFG)) == 12
    folds = player_fold(pd.Series(range(1000)), 5)
    assert set(folds) == set(range(5))
    assert folds.equals(player_fold(pd.Series(range(1000)), 5))


@settings(max_examples=6, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_outcome_invariants_on_synthetic(seed: int) -> None:
    tables = generate(SyntheticSpec(n_pairs=60, seed=seed), DATA_CFG)
    daily = clean_daily(tables["daily"], DATA_CFG)
    players = build_players(tables["demographics"], tables["rg"], daily)
    lm = build_landmarks(players, daily, DATA_CFG, LM_CFG, PRIMARY)
    if lm.empty:
        pytest.skip("no eligible rows for this seed")
    assert lm["time"].between(1, LM_CFG.horizon_days).all()
    assert (lm["follow_end"] > lm["landmark"]).all()
    ev = lm[lm["event"] == 1]
    assert ((ev["event_date"] >= ev["landmark"]) & (ev["event_date"] < ev["follow_end"])).all()
    assert not (lm["event_date"] < lm["landmark"]).any()
    assert (lm.loc[lm["rg_case"] == 0, "event"] == 0).all()
    assert not lm.duplicated(["user_id", "landmark"]).any()
    # a player is in exactly one fold
    assert lm.groupby("user_id")["fold"].nunique().eq(1).all()
