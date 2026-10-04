from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from ttr.config import EventWindow, Products, load_data_config, load_labels_config


def test_data_config_loads() -> None:
    cfg = load_data_config()
    assert cfg.event_window.start == date(2008, 11, 2)
    assert cfg.products.family_of()[2] == "live_action"
    assert set(cfg.products.money_valid) <= set(cfg.products.family_of())


def test_labels_config_loads() -> None:
    cfg = load_labels_config()
    primary = cfg.definitions[cfg.default]
    assert 6 not in primary.event_types  # "heavy complainer" is not a harm event
    assert {2, 16} <= set(primary.exclude_interventions)  # already closed: prevalent cases


def test_event_window_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        EventWindow(start=date(2009, 1, 1), end=date(2008, 1, 1))


def test_product_families_must_be_disjoint() -> None:
    with pytest.raises(ValidationError):
        Products(money_valid=(1,), families={"a": (1, 2), "b": (2,)})


def test_relative_paths_resolve_to_project_root() -> None:
    cfg = load_data_config()
    assert cfg.resolve(Path("x")).parent == Path(__file__).resolve().parents[1]
