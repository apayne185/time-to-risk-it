"""The nine betting-activity indices used by Gray, LaPlante & Shaffer (2012).

Computed per player and product family, optionally only from activity before a per-player
cutoff date. Definitions follow the codebook's analytic dataset.
"""

from __future__ import annotations

import pandas as pd

# The paper's three product groupings (codebook: "fixed odds, live action and casino-type").
PAPER_FAMILIES: dict[str, tuple[int, ...]] = {
    "fixedodds": (1,),
    "liveaction": (2,),
    "casino": (4, 8, 17),
}

METRICS = (
    "sum_stakes",
    "sum_bets",
    "bettingdays",
    "duration",
    "frequency",
    "bets_per_day",
    "euros_per_bet",
    "net_loss",
    "percent_lost",
)


def _family_indices(rows: pd.DataFrame) -> pd.DataFrame:
    g = rows[rows["is_bet_day"]].groupby("user_id")
    # Duration spans every row, settlements included; this matches the paper's figures.
    span = rows.groupby("user_id")["date"]
    out = pd.DataFrame(
        {
            "sum_stakes": g["turnover"].sum(),
            "sum_bets": g["n_bets"].sum(),
            "bettingdays": g["date"].nunique(),
        }
    )
    out["duration"] = ((span.max() - span.min()).dt.days + 1).reindex(out.index)
    out["frequency"] = out["bettingdays"] / out["duration"]
    out["bets_per_day"] = out["sum_bets"] / out["bettingdays"]
    out["euros_per_bet"] = out["sum_stakes"] / out["sum_bets"].where(out["sum_bets"] > 0)
    # Net loss includes settlement rows (zero bets, non-zero hold).
    out["net_loss"] = rows.groupby("user_id")["hold"].sum().reindex(out.index)
    # Some players placed only zero-stake (free) bets: ratios are undefined, not infinite.
    out["percent_lost"] = out["net_loss"] / out["sum_stakes"].where(out["sum_stakes"] > 0)
    return out[list(METRICS)]


def betting_indices(daily: pd.DataFrame, cutoff: pd.Series | None = None) -> pd.DataFrame:
    """Wide table (one row per player who bet) of ``{metric}_{family}`` columns.

    Args:
        daily: cleaned daily aggregates.
        cutoff: optional per-player date (indexed by user_id); only activity strictly before
            it is used. Players without a cutoff are dropped.
    """
    if cutoff is not None:
        limit = daily["user_id"].map(cutoff)
        daily = daily[daily["date"] < limit]
    frames = []
    for family, products in PAPER_FAMILIES.items():
        fam = _family_indices(daily[daily["product_type"].isin(products)])
        frames.append(fam.add_suffix(f"_{family}"))
    return pd.concat(frames, axis=1).rename_axis("user_id")


def index_columns() -> list[str]:
    return [f"{m}_{f}" for f in PAPER_FAMILIES for m in METRICS]
