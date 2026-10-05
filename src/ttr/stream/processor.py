"""Online scoring over an activity event stream.

Event-time semantics: activity events are end-of-day aggregates, so an event dated on or after
the next landmark means every day before it is complete. That landmark then *fires*: all eligible
players are scored from the state accumulated so far, which by construction holds only activity
strictly before the landmark (the same leakage rule as the offline pipeline).

Eligibility matches the offline risk set's activity rule: deposited before the landmark and at
least one bet in the preceding ``lookback_days``. Outcome-based exclusions (a player's earlier RG
event) need the operator's intervention log and are applied downstream.

State keeps each player's full history so features are computed by exactly the offline SQL. A
production deployment would hold a 365-day window plus lifetime accumulators in a feature store;
the parity tests pin the behaviour either implementation must reproduce.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import pandas as pd

from ttr.config import DataConfig
from ttr.features import features_from_activity
from ttr.serve.scoring import Scorer
from ttr.stream.events import ActivityEvent, ScoreEvent

log = logging.getLogger(__name__)

_COLUMNS = ["user_id", "date", "product_type", "turnover", "hold", "n_bets"]


@dataclass
class StreamStats:
    events: int = 0
    late_events: int = 0
    landmarks_fired: int = 0
    players_scored: int = 0
    flagged: int = 0


@dataclass
class StreamScorer:
    scorer: Scorer
    players: pd.DataFrame  # user_id, first_deposit_date (a compacted topic in production)
    landmarks: list[pd.Timestamp]
    data_cfg: DataConfig
    lookback_days: int = 90
    stats: StreamStats = field(default_factory=StreamStats)

    def __post_init__(self) -> None:
        self._pending = sorted(pd.Timestamp(x) for x in self.landmarks)
        self._fired: list[pd.Timestamp] = []
        self._rows: list[tuple[object, ...]] = []
        self._deposit = self.players.set_index("user_id")["first_deposit_date"]

    @property
    def next_landmark(self) -> pd.Timestamp | None:
        return self._pending[0] if self._pending else None

    def process(self, event: ActivityEvent) -> list[ScoreEvent]:
        """Ingest one event; returns scores for any landmark it completes."""
        when = pd.Timestamp(event.date)
        out: list[ScoreEvent] = []
        while self._pending and when >= self._pending[0]:
            out += self._fire(self._pending.pop(0))
        if self._fired and when < self._fired[-1]:
            # Arrived after a landmark it belongs before: kept for later landmarks, but the
            # scores already emitted did not see it.
            self.stats.late_events += 1
        self._rows.append(
            (event.user_id, when, event.product_type, event.turnover, event.hold, event.n_bets)
        )
        self.stats.events += 1
        return out

    def run(self, events: Iterable[ActivityEvent]) -> Iterator[ScoreEvent]:
        for event in events:
            yield from self.process(event)

    def flush(self, until: pd.Timestamp | None = None) -> list[ScoreEvent]:
        """End of stream: fire remaining landmarks up to ``until`` (all, if None)."""
        out: list[ScoreEvent] = []
        while self._pending and (until is None or self._pending[0] <= until):
            out += self._fire(self._pending.pop(0))
        return out

    def _fire(self, landmark: pd.Timestamp) -> list[ScoreEvent]:
        self._fired.append(landmark)
        self.stats.landmarks_fired += 1
        activity = pd.DataFrame(self._rows, columns=_COLUMNS).astype(
            {"turnover": float, "hold": float, "n_bets": float}
        )
        bets = activity[(activity["n_bets"] > 0) & (activity["date"] < landmark)]
        recent = bets[bets["date"] >= landmark - pd.Timedelta(days=self.lookback_days)]
        eligible = [
            uid
            for uid in recent["user_id"].unique()
            if uid in self._deposit.index and self._deposit[uid] < landmark
        ]
        if not eligible:
            log.info("landmark %s: no eligible players", landmark.date())
            return []
        players = pd.DataFrame(
            {"user_id": eligible, "first_deposit_date": self._deposit.loc[eligible].to_numpy()}
        )
        feats = features_from_activity(
            activity[activity["user_id"].isin(eligible)], players, landmark, self.data_cfg
        )
        scored = self.scorer.score(feats)
        family = str(self.scorer.bundle.get("family"))
        events = [
            ScoreEvent(
                user_id=int(uid),
                landmark=landmark.date(),
                risk_score=float(r["risk_score"]),
                probability_30d=float(r["probability"]),
                flagged=bool(r["flagged"]),
                top_drivers=list(r["top_drivers"]),
                model=family,
            )
            for uid, (_, r) in zip(feats["user_id"], scored.iterrows(), strict=True)
        ]
        self.stats.players_scored += len(events)
        self.stats.flagged += sum(e.flagged for e in events)
        log.info(
            "landmark %s: scored %d players, %d flagged",
            landmark.date(),
            len(events),
            sum(e.flagged for e in events),
        )
        return events
