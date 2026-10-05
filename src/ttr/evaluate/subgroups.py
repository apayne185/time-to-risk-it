"""Model performance by demographic subgroup (demographics are not features: ADR 0004)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ttr.config import DecisionConfig


def attach_groups(table: pd.DataFrame, players: pd.DataFrame, cfg: DecisionConfig) -> pd.DataFrame:
    demo = players.set_index("user_id")[["gender", "year_of_birth", "country"]]
    out = table.join(demo, on="user_id")
    age = out["landmark"].dt.year - out["year_of_birth"].astype("Float64")
    bands = cfg.age_bands
    labels = [f"{lo}-{hi - 1}" if hi < 120 else f"{lo}+" for lo, hi in pairwise(bands)]
    out["age_band"] = pd.cut(age.astype(float), bins=list(bands), labels=labels, right=False)
    top = out.drop_duplicates("user_id")["country"].value_counts().index[: cfg.top_countries]
    out["country_group"] = out["country"].where(out["country"].isin(top), "Other")
    out.loc[out["country"].isna(), "country_group"] = np.nan
    return out


def subgroup_table(
    frame: pd.DataFrame, group_cols: tuple[str, ...] = ("gender", "age_band", "country_group")
) -> pd.DataFrame:
    """Per subgroup: rows, 30-day events, AUC and weighted observed/expected ratio.

    ``frame`` needs columns y, p (calibrated probability), risk and w. Players with missing
    demographics (all controls) are excluded, which the report states.
    """
    rows = []
    for col in group_cols:
        for value, g in frame.dropna(subset=[col]).groupby(col, observed=True):
            has_both = g["y"].nunique() == 2
            rows.append(
                {
                    "attribute": col.replace("_group", "").replace("_", " "),
                    "group": str(value),
                    "rows": len(g),
                    "events": int(g["y"].sum()),
                    "auc": roc_auc_score(g["y"], g["risk"]) if has_both else np.nan,
                    "o_e": np.average(g["y"], weights=g["w"]) / np.average(g["p"], weights=g["w"]),
                }
            )
    return pd.DataFrame(rows)
