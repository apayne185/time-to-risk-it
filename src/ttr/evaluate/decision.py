"""Decision-analytic metrics: net benefit and capacity-constrained outreach."""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_benefit(p: np.ndarray, y: np.ndarray, w: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit of contacting everyone with risk >= t, per active player (Vickers & Elkin).

    NB(t) = TP/N - FP/N * t / (1 - t): true contacts minus false contacts, with false contacts
    valued at the odds of the threshold. A threshold of t means one would accept
    (1 - t) / t unnecessary contacts to reach one player who goes on to trigger an RG event.
    """
    total = w.sum()
    out = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        contact = p >= t
        tp = w[contact & (y == 1)].sum() / total
        fp = w[contact & (y == 0)].sum() / total
        out[i] = tp - fp * t / (1 - t)
    return out


def net_benefit_contact_all(y: np.ndarray, w: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = np.average(y, weights=w)
    out: np.ndarray = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    return out


def capacity_curve(
    risk: np.ndarray, y: np.ndarray, w: np.ndarray, landmark: np.ndarray, shares: tuple[float, ...]
) -> pd.DataFrame:
    """Contact the top ``share`` of active players at every landmark (by weighted count).

    Returns, per share: recall (share of next-month RG cases reached), precision (share of
    contacts who go on to trigger an RG event) and contacts per case reached.
    """
    rows = []
    for share in shares:
        tp = contacts = events = 0.0
        for lm in np.unique(landmark):
            m = landmark == lm
            order = np.argsort(-risk[m], kind="stable")
            wm, ym = w[m][order], y[m][order]
            budget = share * wm.sum()
            take = np.cumsum(wm) <= budget
            if not take.any():  # always contact at least the top player
                take[0] = True
            tp += wm[take & (ym == 1)].sum()
            contacts += wm[take].sum()
            events += wm[ym == 1].sum()
        rows.append(
            {
                "share": share,
                "recall": tp / events if events else float("nan"),
                "precision": tp / contacts if contacts else float("nan"),
                "contacts_per_case": contacts / tp if tp else float("inf"),
            }
        )
    return pd.DataFrame(rows)
