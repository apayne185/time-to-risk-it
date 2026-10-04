"""Interim tables -> clean modelling inputs.

Each rule below was checked against the paper's analytic dataset (see
``tests/test_realdata.py``):

- Split records: rows sharing (user, date, product) are parts of one day's activity and are
  summed. Summing reproduces the paper's per-player totals exactly for every player with split
  fixed-odds records; keeping one row does not.
- Bet days: a day counts as a betting day only if bets were placed (``n_bets > 0``). Rows with
  zero bets but non-zero hold are settlements of earlier bets; they count towards net loss but
  not towards activity.
- Empty rows (no bets, no stake, no hold) carry no information and are dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ttr.config import DataConfig

log = logging.getLogger(__name__)

# The raw export truncates long country names.
COUNTRY_FIXES = {
    "Bosnia and Herzego": "Bosnia and Herzegovina",
    "United Arab Emirat": "United Arab Emirates",
}

DEMOGRAPHIC_COLUMNS = ["country", "language", "year_of_birth", "registration_date"]


@dataclass(frozen=True)
class CohortStep:
    step: str
    n_players: int
    n_cases: int
    n_controls: int


def clean_daily(daily: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Merge split records, drop empty rows and add product metadata."""
    n_in = len(daily)
    # min_count=1 keeps vendor products' money columns missing instead of summing to 0.
    out = (
        daily.assign(n_bets=daily["n_bets"].fillna(0))
        .groupby(["user_id", "date", "product_type"], as_index=False)[
            ["n_bets", "turnover", "hold"]
        ]
        .sum(min_count=1)
    )
    n_merged = n_in - len(out)

    empty = (out["n_bets"] == 0) & (out["turnover"].fillna(0) == 0) & (out["hold"].fillna(0) == 0)
    out = out.loc[~empty].copy()

    out["n_bets"] = out["n_bets"].astype("int64")
    out["product_family"] = out["product_type"].map(cfg.products.family_of()).astype("category")
    out["money_valid"] = out["product_type"].isin(cfg.products.money_valid)
    out["is_bet_day"] = out["n_bets"] > 0
    log.info(
        "daily: %d rows in, %d merged, %d empty dropped, %d out",
        n_in,
        n_merged,
        int(empty.sum()),
        len(out),
    )
    return out.sort_values(["user_id", "date", "product_type"]).reset_index(drop=True)


def split_country(raw: pd.Series) -> pd.DataFrame:
    """'Greece.BAW' -> country 'Greece', site 'BAW'. Plain names have no site suffix."""
    country = raw.str.replace(r"\.[A-Z]+$", "", regex=True).replace(COUNTRY_FIXES)
    site = raw.str.extract(r"\.([A-Z]+)$")[0]
    return pd.DataFrame({"country": country, "site": site}, index=raw.index)


def build_players(
    demographics: pd.DataFrame, rg: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    """One row per player: demographics, RG event details and activity span."""
    players = demographics.rename(columns={"language_name": "language"}).copy()
    players[["country", "site"]] = split_country(players.pop("country_name"))
    # Audit-only flag. Missing demographics occur only among controls, so this column (or any
    # imputation that reveals it) must never be a model feature.
    players["demographics_missing"] = players["country"].isna()

    bets = daily.loc[daily["is_bet_day"]].groupby("user_id")["date"]
    activity = pd.DataFrame(
        {
            "first_bet_date": bets.min(),
            "last_bet_date": bets.max(),
            "n_bet_days": bets.nunique(),
        }
    )
    players = players.merge(rg, on="user_id", how="left").merge(
        activity, left_on="user_id", right_index=True, how="left"
    )
    players["n_bet_days"] = players["n_bet_days"].fillna(0).astype("int64")
    players["exclusion_reason"] = exclusion_reasons(players)
    cols = [
        "user_id",
        "rg_case",
        "first_deposit_date",
        "registration_date",
        "gender",
        "year_of_birth",
        "country",
        "site",
        "language",
        "demographics_missing",
        "rg_n_events",
        "rg_first_date",
        "rg_last_date",
        "event_type_first",
        "intervention_type_first",
        "first_bet_date",
        "last_bet_date",
        "n_bet_days",
        "exclusion_reason",
    ]
    return players[cols].sort_values("user_id").reset_index(drop=True)


def exclusion_reasons(players: pd.DataFrame) -> pd.Series:
    """First applicable reason a player cannot enter the modelling cohort, else missing."""
    is_case = players["rg_case"] == 1
    rules = [
        ("no_betting_activity", players["n_bet_days"] == 0),
        ("case_without_rg_date", is_case & players["rg_first_date"].isna()),
        # Need at least one betting day strictly before the event to build features.
        ("no_pre_event_history", is_case & (players["rg_first_date"] <= players["first_bet_date"])),
    ]
    reason = pd.Series(pd.NA, index=players.index, dtype="string")
    for name, mask in reversed(rules):
        reason = reason.mask(mask.fillna(False), name)
    return reason


def cohort_flow(players: pd.DataFrame) -> list[CohortStep]:
    def step(name: str, df: pd.DataFrame) -> CohortStep:
        return CohortStep(
            name, len(df), int((df["rg_case"] == 1).sum()), int((df["rg_case"] == 0).sum())
        )

    steps = [step("all players", players)]
    remaining = players
    for reason in ["no_betting_activity", "case_without_rg_date", "no_pre_event_history"]:
        remaining = remaining.loc[remaining["exclusion_reason"].ne(reason).fillna(True)]
        steps.append(step(f"excluding {reason.replace('_', ' ')}", remaining))
    return steps


def clean(
    cfg: DataConfig, interim_dir: Path | None = None, out_dir: Path | None = None
) -> list[CohortStep]:
    interim_dir = cfg.resolve(interim_dir or cfg.interim_dir)
    out_dir = cfg.resolve(out_dir or cfg.processed_dir)

    daily = clean_daily(pd.read_parquet(interim_dir / "daily.parquet"), cfg)
    players = build_players(
        pd.read_parquet(interim_dir / "demographics.parquet"),
        pd.read_parquet(interim_dir / "rg.parquet"),
        daily,
    )
    flow = cohort_flow(players)

    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_dir / "daily.parquet", index=False)
    players.to_parquet(out_dir / "players.parquet", index=False)
    (out_dir / "cohort_flow.json").write_text(
        json.dumps([asdict(s) for s in flow], indent=2) + "\n"
    )
    return flow
