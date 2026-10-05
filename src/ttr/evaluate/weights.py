"""Case-control weights: turn the matched sample back into an active-player population.

The sample holds every harm-onset RG case from the event window plus about one control per
case, so cases make up roughly half the players instead of a fraction of a percent. Giving each
control player a weight ``w`` makes the weighted share of case players equal an assumed
population rate ``pi``:

    n_cases / (n_cases + w * n_controls) = pi   =>   w = (n_cases / n_controls) * (1 - pi) / pi

Cases keep weight 1. Every landmark row inherits its player's weight. Ranking metrics (AUC,
C-index) are unchanged, because weights are constant within each class; probabilities,
calibration, net benefit and capacity are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def control_weight(
    n_case_players: int, n_control_players: int, population_case_rate: float
) -> float:
    if not 0 < population_case_rate < 1:
        raise ValueError("population_case_rate must be in (0, 1)")
    return (n_case_players / n_control_players) * (1 - population_case_rate) / population_case_rate


def case_control_weights(table: pd.DataFrame, population_case_rate: float) -> np.ndarray:
    """Per-row weights for a landmark table with ``user_id`` and ``rg_case`` columns."""
    players = table.drop_duplicates("user_id")
    w = control_weight(
        int((players["rg_case"] == 1).sum()),
        int((players["rg_case"] == 0).sum()),
        population_case_rate,
    )
    return np.where(table["rg_case"].to_numpy() == 1, 1.0, w)
