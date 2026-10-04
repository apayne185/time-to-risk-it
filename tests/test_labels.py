import pandas as pd

from ttr.config import LabelDefinition
from ttr.labels import get_definition, label_status


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rg_case": [1, 1, 1, 1, 1, 0],
            "event_type_first": pd.array([10, 2, 2, 6, 2, None], dtype="Int64"),
            "intervention_type_first": pd.array([1, 2, 16, 1, 14, None], dtype="Int64"),
        }
    )


def test_primary_label_excludes_reopenings_and_non_harm_events() -> None:
    status = label_status(_players(), get_definition("primary"))
    # problem reported, re-opened, remains closed, heavy complainer, closure imposed, control
    assert status.tolist() == ["event", "excluded", "excluded", "excluded", "event", "control"]


def test_broad_label_counts_every_case() -> None:
    status = label_status(_players(), get_definition("broad"))
    assert status.tolist() == ["event", "event", "event", "event", "event", "control"]


def test_missing_event_type_is_excluded() -> None:
    players = _players()
    players.loc[0, "event_type_first"] = pd.NA
    definition = LabelDefinition(description="", event_types=(10,))
    assert label_status(players, definition)[0] == "excluded"
