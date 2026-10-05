"""Model features per landmark row, computed with DuckDB from activity strictly before it.

Aggregation lives in ``sql/features/*.sql``; this module runs it, pivots the look-back windows
into columns and derives ratio and trend features. ``FEATURES`` is the single registry of
feature names and meanings: the output columns, the model inputs and ``docs/features.md`` are
all generated from it.

Demographics (age, gender, country) are deliberately not features. See ADR 0004.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ttr.config import PROJECT_ROOT, DataConfig

log = logging.getLogger(__name__)

SQL_DIR = PROJECT_ROOT / "sql" / "features"
WINDOWS = (30, 90, 365)

# Features computed in every look-back window: name -> description ({w} = window length).
_PER_WINDOW: dict[str, str] = {
    "bet_days": "Days with at least one bet in the last {w} days",
    "active_frac": "Share of the last {w} days with a bet",
    "live_day_share": "Share of betting days in the last {w} days with live-action bets",
    "casino_days": "Days with casino play in the last {w} days",
    "sports_bets_per_day": "Sportsbook bets per sportsbook betting day, last {w} days",
    "euros_per_sports_bet": "Average sportsbook stake per bet (EUR), last {w} days",
    "log_stakes": "log(1 + total stakes in EUR), last {w} days",
    "live_stake_share": "Live-action share of sportsbook stakes, last {w} days",
    "pct_lost": "Net loss as a share of stakes, last {w} days",
    "log_max_day_stakes": "log(1 + largest single-day stake in EUR), last {w} days",
    "max_families": "Most product families used on one day, last {w} days",
}
# Only meaningful over longer windows.
_LONG_WINDOW: dict[str, str] = {
    "stake_cv": "Coefficient of variation of daily stakes on betting days, last {w} days",
    "chase_rate": "Share of betting days after a losing day on which stakes rose, last {w} days",
}
_OTHER: dict[str, str] = {
    "bet_days_trend_30_365": "Betting-day rate in the last 30 days relative to the last year",
    "bet_days_trend_30_90": "Betting-day rate in the last 30 days relative to the last 90",
    "stakes_trend_30_365": "log ratio of last-30-day stakes to the last year's monthly average",
    "live_share_change_30_365": "Change in live-action day share, last 30 days vs last year",
    "days_since_last_bet": "Days since the most recent bet",
    "tenure_days": "Days since first deposit",
    "log_bet_days_all": "log(1 + betting days over the whole history)",
    "log_stakes_all": "log(1 + total stakes over the whole history, EUR)",
    "days_since_first_live": "Days since first live-action bet (missing if never)",
    "casino_started_30": "First ever casino play was in the last 30 days (0/1)",
}

FEATURES: dict[str, str] = {
    **{f"{k}_{w}d": v.format(w=w) for w in WINDOWS for k, v in _PER_WINDOW.items()},
    **{f"{k}_{w}d": v.format(w=w) for w in WINDOWS[1:] for k, v in _LONG_WINDOW.items()},
    **_OTHER,
}


def _sql(name: str) -> str:
    return (SQL_DIR / f"{name}.sql").read_text()


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


def _aggregate(daily: pd.DataFrame, keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    try:
        con.register("daily", daily)
        con.register("landmarks", keys)
        con.register("windows", pd.DataFrame({"window_days": list(WINDOWS)}))
        windowed = con.execute(_sql("window_aggregates")).df()
        history = con.execute(_sql("history")).df()
    finally:
        con.close()
    return windowed, history


def compute_features(daily: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    """Features for each (user_id, landmark) row of ``landmarks``, in ``FEATURES`` order."""
    keys = landmarks[["user_id", "landmark"]].drop_duplicates()
    cols = [
        "user_id",
        "date",
        "n_bets",
        "turnover",
        "hold",
        "product_family",
        "money_valid",
        "is_bet_day",
    ]
    d = daily[daily["user_id"].isin(keys["user_id"])][cols].astype({"product_family": "string"})
    windowed, history = _aggregate(d, keys)

    idx = ["user_id", "landmark"]
    metrics = [c for c in windowed.columns if c not in (*idx, "window_days")]
    # pivot, not pivot_table: keys are unique and duplicates should fail rather than aggregate.
    wide = windowed.pivot(index=idx, columns="window_days").reindex(  # noqa: PD010
        index=pd.MultiIndex.from_frame(keys),
        # A window with no activity for anyone has no rows; keep its columns anyway.
        columns=pd.MultiIndex.from_product([metrics, list(WINDOWS)]),
    )
    out = pd.DataFrame(index=wide.index)

    def col(metric: str, w: int) -> pd.Series:
        # Players with no activity in a window have no SQL row: counts and sums are zero.
        return wide[(metric, w)].astype(float).fillna(0.0)

    for w in WINDOWS:
        bet_days = col("bet_days", w)
        sports_bets, sports_days = col("sports_bets", w), col("sports_days", w)
        stakes, sports_stakes = col("stakes", w), col("sports_stakes", w)
        out[f"bet_days_{w}d"] = bet_days
        out[f"active_frac_{w}d"] = bet_days / w
        out[f"live_day_share_{w}d"] = _div(col("live_days", w), bet_days)
        out[f"casino_days_{w}d"] = col("casino_days", w)
        out[f"sports_bets_per_day_{w}d"] = _div(sports_bets, sports_days)
        out[f"euros_per_sports_bet_{w}d"] = _div(sports_stakes, sports_bets)
        out[f"log_stakes_{w}d"] = np.log1p(stakes.clip(lower=0))
        out[f"live_stake_share_{w}d"] = _div(col("live_stakes", w), sports_stakes)
        out[f"pct_lost_{w}d"] = _div(col("net_loss", w), stakes)
        out[f"log_max_day_stakes_{w}d"] = np.log1p(col("max_day_stakes", w).clip(lower=0))
        out[f"max_families_{w}d"] = col("max_families", w)
    for w in WINDOWS[1:]:
        out[f"stake_cv_{w}d"] = _div(
            wide[("sd_day_stakes", w)].astype(float), wide[("mean_day_stakes", w)].astype(float)
        )
        out[f"chase_rate_{w}d"] = _div(col("chase_days", w), col("days_after_loss", w))

    bd30, bd90, bd365 = (col("bet_days", w) for w in WINDOWS)
    out["bet_days_trend_30_365"] = (bd30 + 1) / (bd365 * 30 / 365 + 1)
    out["bet_days_trend_30_90"] = (bd30 + 1) / (bd90 * 30 / 90 + 1)
    out["stakes_trend_30_365"] = np.log1p(col("stakes", 30)) - np.log1p(col("stakes", 365) / 12)
    out["live_share_change_30_365"] = out["live_day_share_30d"] - out["live_day_share_365d"]

    h = history.set_index(idx).reindex(out.index)
    lm = landmarks.set_index(idx)[["days_since_last_bet", "tenure_days"]].reindex(out.index)
    landmark = pd.Series(out.index.get_level_values("landmark"), index=out.index)
    out["days_since_last_bet"] = lm["days_since_last_bet"].astype(float)
    out["tenure_days"] = lm["tenure_days"].astype(float)
    out["log_bet_days_all"] = np.log1p(h["bet_days_all"].astype(float).fillna(0))
    out["log_stakes_all"] = np.log1p(h["stakes_all"].astype(float).fillna(0).clip(lower=0))
    out["days_since_first_live"] = (landmark - pd.to_datetime(h["first_live_date"])).dt.days.astype(
        float
    )
    out["casino_started_30"] = (
        (landmark - pd.to_datetime(h["first_casino_date"])).dt.days <= 30
    ).astype(float)

    missing = set(FEATURES) - set(out.columns)
    extra = set(out.columns) - set(FEATURES)
    if missing or extra:
        raise RuntimeError(f"feature registry mismatch: missing {missing}, extra {extra}")
    return out[list(FEATURES)].reset_index()


def build_feature_table(landmarks: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Landmark rows (outcomes, splits, folds) joined to their features."""
    feats = compute_features(daily, landmarks)
    base = landmarks.drop(columns=[c for c in FEATURES if c in landmarks.columns])
    return base.merge(feats, on=["user_id", "landmark"], how="left", validate="1:1")


def feature_docs_markdown() -> str:
    lines = [
        "# Features",
        "",
        "Generated from `ttr.features.FEATURES` (`uv run ttr features --docs`). Every feature uses",
        "only activity strictly before the landmark; `tests/test_features.py` enforces this.",
        "Demographics are not features (ADR 0004).",
        "",
        "| Feature | Description |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {desc} |" for name, desc in FEATURES.items()]
    return "\n".join(lines) + "\n"


def write_feature_docs(path: Path) -> Path:
    path.write_text(feature_docs_markdown())
    return path


def features_from_activity(
    activity: pd.DataFrame, players: pd.DataFrame, landmark: pd.Timestamp, data_cfg: DataConfig
) -> pd.DataFrame:
    """Features at one landmark straight from raw daily activity (as used by the scoring API).

    ``activity`` has the raw daily-aggregate columns (user_id, date, product_type, turnover,
    hold, n_bets); ``players`` has user_id and first_deposit_date. Runs the same cleaning and SQL
    as the offline pipeline, so online and batch features cannot drift apart.
    """
    from ttr.data.clean import clean_daily

    daily = clean_daily(activity, data_cfg)
    before = daily[daily["is_bet_day"] & (daily["date"] < landmark)]
    last_bet = before.groupby("user_id")["date"].max()
    rows = players[["user_id", "first_deposit_date"]].assign(landmark=landmark)
    rows["days_since_last_bet"] = (landmark - rows["user_id"].map(last_bet)).dt.days
    rows["tenure_days"] = (landmark - rows["first_deposit_date"]).dt.days
    return compute_features(daily, rows.drop(columns="first_deposit_date"))
