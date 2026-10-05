"""Event schemas on the wire.

``bets.daily``: one message per (player, day, product) aggregate, keyed by player id so a
player's events stay ordered within one partition.
``rg.scores``: one message per scored player per landmark.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ACTIVITY_TOPIC = "bets.daily"
SCORES_TOPIC = "rg.scores"


class ActivityEvent(BaseModel):
    """A daily betting aggregate, as the operator's settlement system would emit it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int
    date: date
    product_type: int = Field(ge=1, le=25)
    n_bets: float = Field(ge=0)
    turnover: float | None = Field(default=None, ge=0)
    hold: float | None = None

    @property
    def key(self) -> bytes:
        return str(self.user_id).encode()

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> ActivityEvent:
        return cls.model_validate_json(raw)


class ScoreEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int
    landmark: date
    risk_score: float
    probability_30d: float
    flagged: bool
    top_drivers: list[dict[str, Any]]
    model: str

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode()


def _num(x: Any) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def events_from_frame(daily: pd.DataFrame) -> list[ActivityEvent]:
    """Raw daily-aggregate rows (interim table) to events, in event-time order."""
    ordered = daily.sort_values(["date", "user_id", "product_type"], kind="stable")
    return [
        ActivityEvent(
            user_id=int(uid),
            date=pd.Timestamp(d).date(),
            product_type=int(pt),
            n_bets=0.0 if pd.isna(n) else float(n),
            turnover=_num(t),
            hold=_num(h),
        )
        for uid, d, pt, n, t, h in zip(
            ordered["user_id"],
            ordered["date"],
            ordered["product_type"],
            ordered["n_bets"],
            ordered["turnover"],
            ordered["hold"],
            strict=True,
        )
    ]
