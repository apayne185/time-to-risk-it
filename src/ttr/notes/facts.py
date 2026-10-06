"""Turn a score's top drivers into plain-language facts with human-readable values.

Values are formatted here, deterministically, so a note writer (template or LLM) only has to
phrase them, and the guard can check every number in a note against this list.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

# Feature-name patterns -> how a person would say the value.
_DAYS_IN = {"30d": "last 30 days", "90d": "last 90 days", "365d": "last year"}


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    statement: str  # e.g. "Largest single-day stake in the last year: €215,000"


class PlayerFacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    probability_30d: float
    flagged: bool
    facts: list[Fact]


def _window(feature: str) -> str:
    return next(
        (label for suffix, label in _DAYS_IN.items() if feature.endswith(suffix)), "overall"
    )


def _money(x: float) -> str:
    return f"€{x:,.0f}"


def humanise(feature: str, value: float | None, description: str) -> str:
    """One sentence stating the behaviour and the player's value, without model jargon."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{description}: not available"
    w = _window(feature)
    if feature.startswith("log_max_day_stakes"):
        return f"Largest single-day stake in the {w}: {_money(math.expm1(value))}"
    if feature.startswith("log_stakes_all"):
        return f"Total staked over the whole account history: {_money(math.expm1(value))}"
    if feature.startswith("log_stakes"):
        return f"Total staked in the {w}: {_money(math.expm1(value))}"
    if feature.startswith("log_bet_days_all"):
        return f"Days with a bet over the whole account history: {math.expm1(value):,.0f}"
    if feature.startswith("active_frac"):
        return f"Share of days with a bet in the {w}: {value:.0%}"
    if feature.startswith("live_day_share") or feature.startswith("live_stake_share"):
        what = "betting days" if "day" in feature else "sportsbook stakes"
        return f"Share of {what} on live (in-play) betting in the {w}: {value:.0%}"
    if feature.startswith("bet_days_trend"):
        return f"Betting days in the last 30 days versus the usual rate: {value:.1f} times"
    if feature.startswith("stakes_trend"):
        ratio = math.exp(value)
        return f"Stakes in the last 30 days versus the usual monthly amount: {ratio:.1f} times"
    if feature.startswith("max_families"):
        return f"Most types of product used on a single day in the {w}: {value:.0f}"
    if feature.startswith("days_since_last_bet"):
        return f"Days since the most recent bet: {value:.0f}"
    if feature.startswith("tenure_days"):
        return f"Days since the first deposit: {value:,.0f}"
    if feature.startswith("days_since_first_live"):
        return f"Days since the first live (in-play) bet: {value:,.0f}"
    if feature.startswith("chase_rate"):
        return f"Share of betting days after a losing day when stakes went up, {w}: {value:.0%}"
    if feature.startswith("stake_cv"):
        return f"Day-to-day variability of stakes in the {w}: {value:.1f} (1 = typical swings)"
    if feature.startswith("pct_lost"):
        return f"Share of stakes lost in the {w}: {value:.0%}"
    if feature.startswith("bet_days"):
        return f"Days with a bet in the {w}: {value:.0f}"
    if feature.startswith("casino_days"):
        return f"Days with casino play in the {w}: {value:.0f}"
    if feature.startswith("sports_bets_per_day"):
        return f"Sportsbook bets per betting day in the {w}: {value:.1f}"
    if feature.startswith("euros_per_sports_bet"):
        return f"Average sportsbook stake per bet in the {w}: {_money(value)}"
    if feature == "casino_started_30":
        return (
            "Started playing casino games for the first time in the last 30 days"
            if value
            else "No new casino play in the last 30 days"
        )
    return f"{description}: {value:,.2f}"


def player_facts(
    player_id: int, probability: float, flagged: bool, drivers: list[dict[str, Any]]
) -> PlayerFacts:
    """Facts from a scorer's ``top_drivers`` entries (feature, value, description optional)."""
    return PlayerFacts(
        player_id=player_id,
        probability_30d=probability,
        flagged=flagged,
        facts=[
            Fact(
                feature=str(d["feature"]),
                statement=humanise(
                    str(d["feature"]), d.get("value"), str(d.get("description", d["feature"]))
                ),
            )
            for d in drivers
        ],
    )
