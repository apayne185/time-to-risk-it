"""Schema-faithful synthetic version of the bwin RG dataset.

The licensed data cannot be redistributed, so CI, tests and demos run on this generator
instead. It reproduces the study design (cases with a first RG event inside the event
window, controls matched on first-deposit date) and the quirks the pipeline must handle:

- cases escalate betting intensity and shift towards live action before their RG event
- some activity continues after the RG event (must never leak into features)
- vendor products carry bet counts but no money values
- some (user, date, product) records are split across two rows
- a block of controls has missing demographics (the dataset's leakage trap)
- a few cases have no RG date, or an RG event before any betting activity

The numbers are plausible, not fitted: the generator exists to exercise code paths,
not to stand in for the real data in any reported result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ttr.config import DataConfig
from ttr.data.raw import write_raw

DATA_START = pd.Timestamp("2000-05-01")
DATA_END = pd.Timestamp("2010-11-10")

COUNTRIES = {
    "Germany.COM": "German",
    "Germany": "German",
    "France.COM": "French",
    "Poland": "Polish",
    "Spain": "Spanish",
    "Austria": "German",
    "Italy.IT": "Italian",
    "Greece.BAW": "Greek",
}
COUNTRY_WEIGHTS = np.array([0.38, 0.06, 0.1, 0.09, 0.09, 0.08, 0.1, 0.1])

# First RG event types and their rough real-data frequencies (codebook Appendix 2).
EVENT_TYPES = np.array([2, 10, 4, 9, 11, 6, 1, 8, 3, 7])
EVENT_WEIGHTS = np.array([932, 334, 308, 274, 105, 41, 23, 20, 19, 10], dtype=float)
INTERVENTIONS = np.array([1, 6, 8, 11, 13, 14, 16, 18])

PRODUCTS = {"fixed_odds": 1, "live_action": 2, "casino": 8, "poker": 10, "games": 15}


@dataclass(frozen=True)
class SyntheticSpec:
    n_pairs: int = 300
    seed: int = 0
    escalation_days: int = 180  # cases ramp up over this many days before the event
    escalation_factor: float = 3.0  # peak multiplier on daily activity probability
    p_missing_demographics: float = 0.08  # share of controls with missing demographics
    p_case_no_rg_date: float = 0.01
    p_case_event_before_activity: float = 0.01
    p_control_no_activity: float = 0.01
    p_split_record: float = 0.01


def _dates_between(
    rng: np.random.Generator, lo: pd.Timestamp, hi: pd.Timestamp, n: int
) -> pd.DatetimeIndex:
    days = rng.integers(0, max((hi - lo).days, 1) + 1, size=n)
    return pd.DatetimeIndex(lo + pd.to_timedelta(days, unit="D"))


def _player_days(
    rng: np.random.Generator,
    spec: SyntheticSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_p: float,
    rg_date: pd.Timestamp | None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Active days for one player and, per day, how far into escalation they are (0..1)."""
    if end < start:
        return pd.DatetimeIndex([]), np.array([])
    days = pd.date_range(start, end, freq="D")
    ramp = np.zeros(len(days))
    if rg_date is not None:
        until = (rg_date - days).days.to_numpy()
        in_ramp = (until >= 0) & (until < spec.escalation_days)
        ramp[in_ramp] = 1 - until[in_ramp] / spec.escalation_days
    p = np.clip(base_p * (1 + (spec.escalation_factor - 1) * ramp), 0, 0.95)
    active = rng.random(len(days)) < p
    return days[active], ramp[active]


def _activity_rows(
    rng: np.random.Generator,
    user_id: int,
    days: pd.DatetimeIndex,
    ramp: np.ndarray,
    intensity: float,
    stake_scale: float,
    casino_affinity: float,
) -> pd.DataFrame:
    n = len(days)
    if n == 0:
        return pd.DataFrame()
    # Product mix shifts from fixed odds towards live action as escalation rises.
    live_share = np.clip(0.3 + 0.4 * ramp, 0, 0.9)
    u = rng.random(n)
    product = np.where(u < live_share, PRODUCTS["live_action"], PRODUCTS["fixed_odds"])
    rows = [pd.DataFrame({"date": days, "product_type": product, "ramp": ramp})]
    # Occasional same-day secondary products.
    extra = rng.random(n) < casino_affinity
    rows.append(
        pd.DataFrame({"date": days[extra], "product_type": PRODUCTS["casino"], "ramp": ramp[extra]})
    )
    vendor = rng.random(n) < 0.08
    rows.append(
        pd.DataFrame(
            {
                "date": days[vendor],
                "product_type": rng.choice(
                    [PRODUCTS["poker"], PRODUCTS["games"]], size=vendor.sum()
                ),
                "ramp": ramp[vendor],
            }
        )
    )
    df = pd.concat(rows, ignore_index=True)

    m = len(df)
    df["n_bets"] = (1 + rng.poisson(intensity * (1 + 2 * df["ramp"].to_numpy()), m)).astype(float)
    per_bet = np.exp(rng.normal(np.log(stake_scale), 0.8, m)) * (1 + df["ramp"].to_numpy())
    df["turnover"] = np.round(df["n_bets"] * per_bet, 2)
    # Fraction of stake lost: house edge on average, wide spread, wins are negative.
    lost = np.clip(rng.normal(0.07, 0.5, m), -3.0, 1.0)
    df["hold"] = np.round(df["turnover"] * lost, 2)
    vendor_rows = df["product_type"].isin([PRODUCTS["poker"], PRODUCTS["games"]])
    df.loc[vendor_rows, ["turnover", "hold"]] = np.nan
    df["user_id"] = user_id
    return df.drop(columns="ramp")


def _split_records(rng: np.random.Generator, daily: pd.DataFrame, p: float) -> pd.DataFrame:
    """Split some sportsbook records into two rows, as seen in the real export."""
    cand = daily["product_type"].isin([PRODUCTS["fixed_odds"], PRODUCTS["live_action"]])
    pick = cand & (rng.random(len(daily)) < p) & (daily["n_bets"] >= 2)
    if not pick.any():
        return daily
    a = daily[pick].copy()
    b = a.copy()
    frac = rng.uniform(0.2, 0.8, len(a))
    a["n_bets"] = np.floor(a["n_bets"] * frac).clip(lower=1)
    b["n_bets"] = b["n_bets"] - a["n_bets"]
    for col in ("turnover", "hold"):
        a[col] = np.round(a[col] * frac, 2)
        b[col] = np.round(b[col] - a[col], 2)
    return pd.concat([daily[~pick], a, b], ignore_index=True)


def generate(spec: SyntheticSpec, cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """Generate canonical demographics, daily and rg tables."""
    rng = np.random.default_rng(spec.seed)
    w_start = pd.Timestamp(cfg.event_window.start)
    w_end = pd.Timestamp(cfg.event_window.end)
    n = spec.n_pairs

    # --- matched pairs -------------------------------------------------------------
    first_deposit = _dates_between(
        rng, pd.Timestamp("2000-05-08"), w_end - pd.Timedelta(days=60), n
    )
    fd = np.concatenate([first_deposit, first_deposit])
    rg_case = np.repeat([1, 0], n)
    user_id = rng.choice(np.arange(30_000, 9_900_000), size=2 * n, replace=False)

    # --- demographics --------------------------------------------------------------
    countries = rng.choice(list(COUNTRIES), size=2 * n, p=COUNTRY_WEIGHTS)
    reg_lag = rng.geometric(0.5, size=2 * n) - 1
    demographics = pd.DataFrame(
        {
            "user_id": user_id,
            "rg_case": rg_case,
            "country_name": countries,
            "language_name": [COUNTRIES[c] for c in countries],
            "gender": np.where(rng.random(2 * n) < 0.9, "M", "F"),
            "year_of_birth": pd.array(rng.integers(1945, 1991, size=2 * n), dtype="Int64"),
            "registration_date": pd.DatetimeIndex(fd) - pd.to_timedelta(reg_lag, unit="D"),
            "first_deposit_date": pd.DatetimeIndex(fd),
        }
    )
    missing_demo = (rg_case == 0) & (rng.random(2 * n) < spec.p_missing_demographics)
    demographics.loc[
        missing_demo, ["country_name", "language_name", "year_of_birth", "registration_date"]
    ] = None

    # --- RG events for cases -------------------------------------------------------
    rg_lo = (first_deposit + pd.Timedelta(days=30)).where(
        first_deposit + pd.Timedelta(days=30) > w_start, w_start
    )
    span = (w_end - rg_lo).days.to_numpy().clip(min=0)
    rg_first = rg_lo + pd.to_timedelta(np.floor(rng.random(n) * (span + 1)), unit="D")
    n_events = rng.geometric(0.8, size=n)
    rg_last = rg_first + pd.to_timedelta(
        np.where(n_events > 1, rng.integers(1, 120, size=n), 0), unit="D"
    )
    rg_last = rg_last.where(rg_last < w_end, w_end)
    event_type = rng.choice(EVENT_TYPES, size=n, p=EVENT_WEIGHTS / EVENT_WEIGHTS.sum())
    intervention = rng.choice(INTERVENTIONS, size=n)
    reopen = (event_type == 2) & (rng.random(n) < 0.6)  # re-openings after earlier closure
    intervention[reopen] = 2
    rg = pd.DataFrame(
        {
            "user_id": user_id[:n],
            "rg_n_events": n_events,
            "rg_first_date": rg_first,
            "rg_last_date": rg_last,
            "event_type_first": pd.array(event_type, dtype="Int64"),
            "intervention_type_first": pd.array(intervention, dtype="Int64"),
        }
    )
    no_date = rng.random(n) < spec.p_case_no_rg_date
    rg.loc[no_date, ["rg_first_date", "rg_last_date"]] = pd.NaT
    event_before_activity = ~no_date & (rng.random(n) < spec.p_case_event_before_activity)

    # --- daily activity ------------------------------------------------------------
    frames = []
    churn = rng.exponential(900, size=2 * n).astype(int)
    for i in range(2 * n):
        is_case = i < n
        start = pd.Timestamp(fd[i])
        end = min(start + pd.Timedelta(days=int(churn[i]) + 30), DATA_END)
        rg_date: pd.Timestamp | None = None
        if is_case and not no_date[i]:
            rg_date = pd.Timestamp(rg_first[i])
            # Cases stay around until the event; ~25% stop betting shortly before it.
            if rng.random() < 0.25:
                end = rg_date - pd.Timedelta(days=int(rng.integers(1, 60)))
            else:
                end = max(end, rg_date + pd.Timedelta(days=int(rng.integers(0, 200))))
            end = min(end, DATA_END)
            if event_before_activity[i]:
                start = rg_date + pd.Timedelta(days=1)
        if not is_case and rng.random() < spec.p_control_no_activity:
            continue
        base_p = float(np.clip(rng.gamma(1.5, 0.06), 0.005, 0.6))
        days, ramp = _player_days(rng, spec, start, end, base_p, rg_date)
        frames.append(
            _activity_rows(
                rng,
                int(user_id[i]),
                days,
                ramp,
                intensity=float(rng.gamma(1.5, 2.0)),
                stake_scale=float(rng.lognormal(1.5, 0.7)),
                casino_affinity=float(rng.beta(1, 8)),
            )
        )
    daily = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    daily = _split_records(rng, daily, spec.p_split_record)
    daily = daily[["user_id", "date", "product_type", "turnover", "hold", "n_bets"]]
    daily = daily.sort_values(["user_id", "date", "product_type"], kind="stable")

    return {
        "demographics": demographics.reset_index(drop=True),
        "daily": daily.reset_index(drop=True),
        "rg": rg.reset_index(drop=True),
    }


def write_synthetic(spec: SyntheticSpec, cfg: DataConfig, out_dir: Path) -> dict[str, int]:
    """Generate synthetic tables and write them in the raw distribution format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = generate(spec, cfg)
    for table, df in tables.items():
        write_raw(df, out_dir / getattr(cfg.files, table), table)  # type: ignore[arg-type]
    return {table: len(df) for table, df in tables.items()}
