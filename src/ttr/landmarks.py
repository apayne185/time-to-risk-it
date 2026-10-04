"""Landmark table: one row per (eligible player, scoring date) with a censored outcome.

At landmark ``L`` a player is in the risk set if they had deposited before ``L``, have not yet
had a qualifying RG event, and placed a bet in the ``eligibility_lookback_days`` before ``L``.
They are followed from ``L`` until the earliest of:

- ``L + horizon_days``
- the first landmark of the next temporal split (so labels never cross a split boundary)
- the end of the event window (controls are only known to be event-free inside it)

``time`` counts days from ``L``: an event on day ``L`` has ``time = 1``; a player followed
for the full horizon without an event has ``time = horizon_days`` and ``event = 0``.

Features are not built here. Anything computed for a row must use activity strictly before
``landmark`` (enforced by tests in the feature stage).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ttr.config import DataConfig, LabelDefinition, LandmarksConfig
from ttr.labels import label_status

log = logging.getLogger(__name__)

ONE_DAY = pd.Timedelta(days=1)


def landmark_dates(cfg: LandmarksConfig) -> list[pd.Timestamp]:
    lm = cfg.landmarks
    return list(
        pd.date_range(pd.Timestamp(lm.first), pd.Timestamp(lm.last), freq=f"{lm.every_months}MS")
    )


def split_bounds(cfg: LandmarksConfig, window_end: pd.Timestamp) -> pd.DataFrame:
    """Per split: first and last landmark, and the (exclusive) date its outcomes are censored at."""
    names = list(cfg.splits)
    rows = []
    for i, name in enumerate(names):
        rng = cfg.splits[name]
        censor = (
            pd.Timestamp(cfg.splits[names[i + 1]].first)
            if i + 1 < len(names)
            else window_end + ONE_DAY
        )
        rows.append(
            {
                "split": name,
                "first": pd.Timestamp(rng.first),
                "last": pd.Timestamp(rng.last),
                "censor_at": censor,
            }
        )
    return pd.DataFrame(rows)


def assign_split(landmarks: pd.Series, bounds: pd.DataFrame) -> pd.Series:
    split = pd.Series(pd.NA, index=landmarks.index, dtype="string")
    for row in bounds.itertuples():
        split[(landmarks >= row.first) & (landmarks <= row.last)] = row.split
    return split


def player_fold(user_id: pd.Series, n_folds: int) -> pd.Series:
    """Deterministic player-level fold (Knuth multiplicative hash), stable across runs."""
    ids = user_id.to_numpy(dtype=np.uint64)
    folds = (ids * np.uint64(2654435761)) % np.uint64(2**32) % np.uint64(n_folds)
    return pd.Series(folds.astype(np.int64), index=user_id.index)


def build_landmarks(
    players: pd.DataFrame,
    daily: pd.DataFrame,
    data_cfg: DataConfig,
    lm_cfg: LandmarksConfig,
    definition: LabelDefinition,
) -> pd.DataFrame:
    window_end = pd.Timestamp(data_cfg.event_window.end)
    bounds = split_bounds(lm_cfg, window_end)

    cohort = players[players["exclusion_reason"].isna()].copy()
    cohort["label_status"] = label_status(cohort, definition)
    cohort = cohort[cohort["label_status"] != "excluded"]
    cohort["event_date"] = cohort["rg_first_date"].where(cohort["label_status"] == "event")
    cohort = cohort.set_index("user_id")

    bets = (
        daily.loc[daily["is_bet_day"] & daily["user_id"].isin(cohort.index), ["user_id", "date"]]
        .drop_duplicates()
        .sort_values("date")
    )
    lookback = pd.Timedelta(days=lm_cfg.eligibility_lookback_days)
    horizon = pd.Timedelta(days=lm_cfg.horizon_days)

    frames = []
    for landmark in landmark_dates(lm_cfg):
        recent = bets[(bets["date"] >= landmark - lookback) & (bets["date"] < landmark)]
        last_bet = recent.groupby("user_id")["date"].max()
        c = cohort.loc[cohort.index.intersection(last_bet.index)]
        c = c[(c["first_deposit_date"] < landmark) & ~(c["event_date"] < landmark).fillna(False)]
        frames.append(
            pd.DataFrame(
                {
                    "user_id": c.index,
                    "landmark": landmark,
                    "rg_case": c["rg_case"].to_numpy(),
                    "label_status": c["label_status"].to_numpy(),
                    "event_date": c["event_date"].to_numpy(),
                    "first_deposit_date": c["first_deposit_date"].to_numpy(),
                    "last_bet_date": last_bet.reindex(c.index).to_numpy(),
                }
            )
        )
    lm = pd.concat(frames, ignore_index=True)

    lm["split"] = assign_split(lm["landmark"], bounds)
    lm = lm[lm["split"].notna()].reset_index(drop=True)
    censor_at = lm["split"].map(bounds.set_index("split")["censor_at"])
    follow_end = pd.concat(
        [lm["landmark"] + horizon, censor_at.astype("datetime64[ns]")], axis=1
    ).min(axis=1)

    lm["event"] = (lm["event_date"] < follow_end).fillna(False).astype("int8")
    event_time = (lm["event_date"] - lm["landmark"]).dt.days + 1
    censor_time = (follow_end - lm["landmark"]).dt.days
    lm["time"] = np.where(lm["event"] == 1, event_time, censor_time).astype("int64")
    lm["follow_end"] = follow_end
    lm["tenure_days"] = (lm["landmark"] - lm["first_deposit_date"]).dt.days
    lm["days_since_last_bet"] = (lm["landmark"] - lm["last_bet_date"]).dt.days
    lm["fold"] = player_fold(lm["user_id"], lm_cfg.n_folds).astype("int8")

    cols = [
        "user_id",
        "landmark",
        "split",
        "fold",
        "rg_case",
        "label_status",
        "event",
        "time",
        "event_date",
        "follow_end",
        "tenure_days",
        "days_since_last_bet",
    ]
    return lm[cols].sort_values(["landmark", "user_id"]).reset_index(drop=True)


def summarise(lm: pd.DataFrame) -> pd.DataFrame:
    """Rows, players, events and event rate per landmark."""
    g = lm.groupby(["split", "landmark"], sort=False)
    out = g.agg(
        rows=("user_id", "size"), events=("event", "sum"), median_follow_days=("time", "median")
    )
    out["event_rate"] = out["events"] / out["rows"]
    return out.reset_index().sort_values("landmark")
