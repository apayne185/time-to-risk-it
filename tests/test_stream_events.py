import pandas as pd
import pytest
from pydantic import ValidationError

from ttr.stream.events import ActivityEvent, events_from_frame


def test_round_trip_and_key() -> None:
    e = ActivityEvent(
        user_id=42,
        date=pd.Timestamp("2009-01-02").date(),
        product_type=2,
        n_bets=3,
        turnover=10.5,
        hold=-2.0,
    )
    assert ActivityEvent.from_bytes(e.to_bytes()) == e
    assert e.key == b"42"


def test_rejects_bad_events() -> None:
    with pytest.raises(ValidationError):
        ActivityEvent(user_id=1, date=pd.Timestamp("2009-01-02").date(), product_type=99, n_bets=1)
    with pytest.raises(ValidationError):
        ActivityEvent.from_bytes(
            b'{"user_id": 1, "date": "2009-01-02", "product_type": 1, "n_bets": 1, "extra": 1}'
        )


def test_events_from_frame_orders_by_time_and_keeps_missing_money() -> None:
    df = pd.DataFrame(
        {
            "user_id": [2, 1],
            "date": pd.to_datetime(["2009-01-02", "2009-01-01"]),
            "product_type": [10, 1],
            "n_bets": [4.0, None],
            "turnover": [None, 5.0],
            "hold": [None, 1.0],
        }
    )
    events = events_from_frame(df)
    assert [e.user_id for e in events] == [1, 2]
    assert events[0].n_bets == 0 and events[1].turnover is None
