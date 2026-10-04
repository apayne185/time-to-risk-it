"""Typed project configuration loaded from ``configs/*.yaml``."""

from __future__ import annotations

from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

TableName = Literal["demographics", "daily", "rg", "analytic"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RawFiles(_Frozen):
    demographics: str
    daily: str
    rg: str
    analytic: str


class Checksums(_Frozen):
    demographics: str
    daily: str
    rg: str
    analytic: str


class EventWindow(_Frozen):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> EventWindow:
        if self.start >= self.end:
            raise ValueError("event_window.start must be before event_window.end")
        return self


class Products(_Frozen):
    money_valid: tuple[int, ...]
    families: dict[str, tuple[int, ...]]

    @model_validator(mode="after")
    def _disjoint(self) -> Products:
        seen: set[int] = set()
        for codes in self.families.values():
            overlap = seen.intersection(codes)
            if overlap:
                raise ValueError(f"product codes in more than one family: {sorted(overlap)}")
            seen.update(codes)
        return self

    def family_of(self) -> dict[int, str]:
        return {code: fam for fam, codes in self.families.items() for code in codes}


class DataConfig(_Frozen):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    files: RawFiles
    checksums: Checksums
    event_window: EventWindow
    products: Products

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


class LabelDefinition(_Frozen):
    description: str
    event_types: tuple[int, ...]
    exclude_interventions: tuple[int, ...] = ()


class LabelsConfig(_Frozen):
    definitions: dict[str, LabelDefinition]
    default: str

    @model_validator(mode="after")
    def _default_exists(self) -> LabelsConfig:
        if self.default not in self.definitions:
            raise ValueError(f"default label {self.default!r} is not defined")
        return self


def _read_yaml(path: Path) -> object:
    with path.open() as fh:
        return yaml.safe_load(fh)


@cache
def load_data_config(path: Path = CONFIG_DIR / "data.yaml") -> DataConfig:
    return DataConfig.model_validate(_read_yaml(path))


@cache
def load_labels_config(path: Path = CONFIG_DIR / "labels.yaml") -> LabelsConfig:
    return LabelsConfig.model_validate(_read_yaml(path))
