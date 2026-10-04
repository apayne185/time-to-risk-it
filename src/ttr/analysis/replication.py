"""Replicate the paper's discriminant function analysis, then test it for look-ahead.

Gray, LaPlante & Shaffer (2012) compared cases and controls on betting indices aggregated over
each player's *entire* history, which includes activity after the RG intervention. That answers
"how do these groups differ?" but not "could we have seen it coming?". Here the same analysis is
run twice:

- ``full_history``: as in the paper.
- ``pre_index``: only activity before each player's index date. A case's index date is its first
  RG event; a control inherits the index date of a case with the same first-deposit date (the
  paper's matching variable), so both groups are observed over the same calendar span.

The gap between the two is the part of the paper's separation that was not available at the time
an operator would have had to act.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ttr.analysis.indices import betting_indices, index_columns
from ttr.labels import get_definition, label_status


def assign_index_dates(players: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Index date per player: cases use their first RG date; each control borrows the RG date of
    a randomly chosen case with the same first-deposit date."""
    rng = np.random.default_rng(seed)
    cases = players[(players["rg_case"] == 1) & players["rg_first_date"].notna()]
    case_dates = cases.groupby("first_deposit_date")["rg_first_date"].apply(list)

    index = cases.set_index("user_id")["rg_first_date"]
    controls = players[players["rg_case"] == 0]
    borrowed = {}
    for uid, fd in zip(controls["user_id"], controls["first_deposit_date"], strict=True):
        options = case_dates.get(fd)
        if options:
            borrowed[uid] = options[rng.integers(len(options))]
    dates: pd.Series = pd.concat([index, pd.Series(borrowed, dtype="datetime64[ns]")])
    return dates.rename("index_date")


def _sqrt_shifted(x: np.ndarray) -> np.ndarray:
    # Paper: square-root transform, with a constant added first to remove negative values.
    out: np.ndarray = np.sqrt(x - np.minimum(x.min(axis=0), 0) + 1)
    return out


@dataclass(frozen=True)
class DFAResult:
    name: str
    n_cases: int
    n_controls: int
    cv_accuracy: float
    cv_auc: float
    structure: pd.Series  # correlation of each index with the discriminant score (cases > 0)


def discriminant_analysis(
    X: pd.DataFrame, y: pd.Series, name: str, seed: int = 0, folds: int = 5
) -> DFAResult:
    """Linear discriminant analysis on sqrt-transformed indices, as in the paper's DFA.

    Reports cross-validated accuracy and AUC rather than in-sample classification rates.
    """
    model = make_pipeline(
        FunctionTransformer(_sqrt_shifted), StandardScaler(), LinearDiscriminantAnalysis()
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X.to_numpy(), y.to_numpy(), cv=cv, method="predict_proba")[
        :, 1
    ]

    model.fit(X.to_numpy(), y.to_numpy())
    score = pd.Series(model.decision_function(X.to_numpy()), index=X.index)
    structure = X.apply(
        lambda col: np.corrcoef(_sqrt_shifted(col.to_numpy()[:, None])[:, 0], score)[0, 1]
    )
    return DFAResult(
        name=name,
        n_cases=int(y.sum()),
        n_controls=int((1 - y).sum()),
        cv_accuracy=float(((proba >= 0.5) == y.to_numpy()).mean()),
        cv_auc=float(roc_auc_score(y, proba)),
        structure=structure.sort_values(key=np.abs, ascending=False),
    )


def replication_inputs(
    daily: pd.DataFrame, players: pd.DataFrame, seed: int = 0
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Feature matrices for both analyses on the same players.

    Players are the modelling cohort with an index date and at least one bet before it; indices
    for products a player never used are zero, as in the paper's first DFA.
    """
    cohort = players[players["exclusion_reason"].isna()].set_index("user_id")
    index_dates = assign_index_dates(players, seed).reindex(cohort.index).dropna()

    pre = betting_indices(daily, cutoff=index_dates)
    eligible = pre.index.intersection(index_dates.index)
    full = betting_indices(daily[daily["user_id"].isin(eligible)])

    cols = index_columns()
    y = cohort.loc[eligible, "rg_case"].astype(int)
    # Same pre-index features, but only harm-onset cases (primary label) against controls.
    status = label_status(cohort.loc[eligible], get_definition("primary"))
    primary = status.index[status != "excluded"]
    return {
        "full_history": (full.reindex(eligible)[cols].fillna(0.0), y),
        "pre_index": (pre.reindex(eligible)[cols].fillna(0.0), y),
        "pre_index_primary": (pre.reindex(primary)[cols].fillna(0.0), y.loc[primary]),
    }


def run_replication(daily: pd.DataFrame, players: pd.DataFrame, seed: int = 0) -> list[DFAResult]:
    return [
        discriminant_analysis(X, y, name, seed)
        for name, (X, y) in replication_inputs(daily, players, seed).items()
    ]
