"""Outcome definitions from the RG event taxonomy in ``configs/labels.yaml``."""

from __future__ import annotations

import pandas as pd

from ttr.config import LabelDefinition, load_labels_config


def get_definition(name: str | None = None) -> LabelDefinition:
    cfg = load_labels_config()
    return cfg.definitions[name or cfg.default]


def label_status(players: pd.DataFrame, definition: LabelDefinition) -> pd.Series:
    """Per player: ``"event"`` (qualifying RG case), ``"control"``, or ``"excluded"``.

    Cases whose first RG event does not qualify are excluded rather than treated as controls:
    they did trigger an intervention, just not one this definition counts.
    """
    is_case = players["rg_case"] == 1
    qualifies = players["event_type_first"].isin(definition.event_types).fillna(False) & ~players[
        "intervention_type_first"
    ].isin(definition.exclude_interventions).fillna(False)
    status = pd.Series("control", index=players.index, dtype="string")
    status[is_case & qualifies] = "event"
    status[is_case & ~qualifies] = "excluded"
    return status
