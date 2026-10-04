"""Monthly betting activity aligned on each player's index date (cases vs matched controls)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.4375
SPORTSBOOK = ("fixed_odds", "live_action")


def player_months(
    daily: pd.DataFrame, index_dates: pd.Series, months_before: int, months_after: int
) -> pd.DataFrame:
    """One row per (player, relative month) with any betting. Month 0 starts on the index date."""
    idx = index_dates.dropna()
    d = daily[daily["user_id"].isin(idx.index) & daily["is_bet_day"]].copy()
    d["month"] = np.floor((d["date"] - d["user_id"].map(idx)).dt.days / DAYS_PER_MONTH)
    d = d[d["month"].between(-months_before, months_after - 1)].astype({"month": int})

    keys = ["user_id", "month"]
    sports = d[d["product_family"].isin(SPORTSBOOK)]
    live = d["product_family"] == "live_action"
    return pd.concat(
        {
            "bet_days": d.groupby(keys)["date"].nunique(),
            "sports_bets": sports.groupby(keys)["n_bets"].sum(),
            "sports_days": sports.groupby(keys)["date"].nunique(),
            "sports_stakes": sports.groupby(keys)["turnover"].sum(),
            "live_stakes": d[live].groupby(keys)["turnover"].sum(),
        },
        axis=1,
    ).fillna(0)


def monthly_trajectories(
    daily: pd.DataFrame,
    players: pd.DataFrame,
    index_dates: pd.Series,
    months_before: int = 24,
    months_after: int = 6,
) -> pd.DataFrame:
    """Per group and relative month:

    - ``active_share``: share of exposed players (first deposit before the month ends) who bet
    - ``bet_days_per_player``: mean betting days per exposed player
    - ``sports_bets_per_day``: median sportsbook bets per sportsbook betting day, among players
      who bet on sport that month
    - ``live_share``: median live-action share of sportsbook stakes, same players
    """
    idx = index_dates.dropna()
    group = (
        players.set_index("user_id")["rg_case"].map({1: "case", 0: "control"}).reindex(idx.index)
    )

    pm = player_months(daily, idx, months_before, months_after).reset_index()
    pm["group"] = pm["user_id"].map(group)
    sports = pm[(pm["sports_days"] > 0)].assign(
        bpd=lambda x: x["sports_bets"] / x["sports_days"],
        live=lambda x: x["live_stakes"] / x["sports_stakes"].where(x["sports_stakes"] > 0),
    )

    months = np.arange(-months_before, months_after)
    fd = players.set_index("user_id")["first_deposit_date"].reindex(idx.index).to_numpy()
    # Exposed in a month = first deposit before the month ends (otherwise they could not bet).
    ends = (
        idx.to_numpy()[:, None]
        + pd.to_timedelta((months + 1) * DAYS_PER_MONTH, unit="D").to_numpy()
    )
    exposed = (
        pd.DataFrame(fd[:, None] < ends, index=idx.index, columns=months)
        .groupby(group)
        .sum()
        .rename_axis("group")
        .reset_index()
        .melt(id_vars="group", var_name="month", value_name="exposed")
        .astype({"month": int})
        .set_index(["group", "month"])
    )

    by = ["group", "month"]
    out = pd.concat(
        [
            exposed,
            pm.groupby(by)["user_id"].nunique().rename("active"),
            pm.groupby(by)["bet_days"].sum().rename("bet_days"),
            sports.groupby(by)["bpd"].median().rename("sports_bets_per_day"),
            sports.groupby(by)["live"].median().rename("live_share"),
        ],
        axis=1,
    ).fillna({"active": 0, "bet_days": 0})
    out["active_share"] = out["active"] / out["exposed"]
    out["bet_days_per_player"] = out["bet_days"] / out["exposed"]
    result: pd.DataFrame = out.reset_index()
    return result
