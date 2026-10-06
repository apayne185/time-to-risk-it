from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ttr.config import load_data_config, load_landmarks_config
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate
from ttr.evaluate.calibration import PlattRecalibrator
from ttr.features import FEATURES, build_feature_table
from ttr.labels import get_definition
from ttr.landmarks import build_landmarks, landmark_dates
from ttr.models.base import SurvivalData
from ttr.models.xgb import XGBCox
from ttr.serve.scoring import Scorer
from ttr.stream.events import ActivityEvent, events_from_frame
from ttr.stream.processor import StreamScorer

CFG = load_data_config()
LM_CFG = load_landmarks_config()


@dataclass(frozen=True)
class World:
    raw: pd.DataFrame
    players: pd.DataFrame
    table: pd.DataFrame
    scorer: Scorer


@pytest.fixture(scope="module")
def world() -> World:
    tables = generate(SyntheticSpec(n_pairs=120, seed=17), CFG)
    daily = clean_daily(tables["daily"], CFG)
    players = build_players(tables["demographics"], tables["rg"], daily)
    lm = build_landmarks(players, daily, CFG, LM_CFG, get_definition("primary"))
    table = build_feature_table(lm, daily)
    model = XGBCox(num_boost_round=40).fit(SurvivalData.from_table(table, list(FEATURES)))
    scorer = Scorer(
        {
            "family": "xgb_cox",
            "model": model,
            "features": list(FEATURES),
            "recalibrator": PlattRecalibrator(),
            "horizon_days": 30,
            "policy": {"capacity_share": 0.01, "threshold": 0.3},
        }
    )
    return World(tables["daily"], players, table, scorer)


def _stream(world: World) -> tuple[StreamScorer, pd.DataFrame]:
    players = world.players
    proc = StreamScorer(
        world.scorer,
        players[["user_id", "first_deposit_date"]],
        landmark_dates(LM_CFG),
        CFG,
        LM_CFG.eligibility_lookback_days,
    )
    events = events_from_frame(world.raw)
    scores = [*proc.run(events), *proc.flush()]
    frame = pd.DataFrame(
        [
            {
                "user_id": s.user_id,
                "landmark": pd.Timestamp(s.landmark),
                "probability": s.probability_30d,
                "flagged": s.flagged,
            }
            for s in scores
        ]
    )
    return proc, frame


def test_stream_scores_equal_offline_scores(world: World) -> None:
    """Every offline landmark row gets the same score online as in batch."""
    proc, online = _stream(world)
    table = world.table
    scorer = world.scorer
    offline = table[["user_id", "landmark"]].assign(
        probability=scorer.score(table, explain=0)["probability"].to_numpy()
    )
    merged = offline.merge(
        online, on=["user_id", "landmark"], how="left", suffixes=("_offline", "_online")
    )
    assert merged["probability_online"].notna().all(), "stream missed offline risk-set rows"
    np.testing.assert_allclose(
        merged["probability_online"], merged["probability_offline"], rtol=1e-6
    )
    assert proc.stats.landmarks_fired == len(landmark_dates(LM_CFG))
    assert proc.stats.late_events == 0


def test_stream_scores_only_active_depositors(world: World) -> None:
    _, online = _stream(world)
    raw = world.raw
    players = world.players
    deposit = players.set_index("user_id")["first_deposit_date"]
    for row in online.sample(50, random_state=0).itertuples():
        bets = raw[
            (raw["user_id"] == row.user_id) & (raw["n_bets"] > 0) & (raw["date"] < row.landmark)
        ]
        landmark = pd.Timestamp(str(row.landmark))
        assert (bets["date"] >= landmark - pd.Timedelta(days=90)).any()
        assert deposit[row.user_id] < landmark


def test_landmark_fires_only_once_its_day_arrives(world: World) -> None:
    players = world.players
    uid = int(players["user_id"].iloc[0])
    proc = StreamScorer(
        world.scorer,
        pd.DataFrame({"user_id": [uid], "first_deposit_date": [pd.Timestamp("2008-01-01")]}),
        [pd.Timestamp("2009-01-01")],
        CFG,
    )

    def ev(d: str) -> ActivityEvent:
        return ActivityEvent(
            user_id=uid,
            date=date.fromisoformat(d),
            product_type=2,
            n_bets=3,
            turnover=10.0,
            hold=1.0,
        )

    assert proc.process(ev("2008-12-31")) == []
    fired = proc.process(ev("2009-01-01"))  # first event of the landmark day completes Dec 31
    assert [s.landmark for s in fired] == [date(2009, 1, 1)]
    assert proc.process(ev("2008-12-30")) == [] and proc.stats.late_events == 1
    assert proc.next_landmark is None
