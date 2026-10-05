"""Typed project configuration loaded from ``configs/*.yaml``."""

from __future__ import annotations

import os
from datetime import date
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

# Source checkout by default; installed copies (e.g. the Docker image) set TTR_PROJECT_ROOT.
PROJECT_ROOT = Path(os.environ.get("TTR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
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


class LandmarkDates(_Frozen):
    first: date
    last: date
    every_months: int = 1

    @model_validator(mode="after")
    def _ordered(self) -> LandmarkDates:
        if self.first > self.last:
            raise ValueError("landmarks.first must not be after landmarks.last")
        if self.every_months < 1:
            raise ValueError("landmarks.every_months must be >= 1")
        return self


class SplitRange(_Frozen):
    first: date
    last: date


class LandmarksConfig(_Frozen):
    landmarks: LandmarkDates
    horizon_days: int
    eligibility_lookback_days: int
    splits: dict[str, SplitRange]
    n_folds: int = 5

    @model_validator(mode="after")
    def _splits_ordered(self) -> LandmarksConfig:
        ranges = list(self.splits.values())
        for prev, nxt in pairwise(ranges):
            if prev.last >= nxt.first:
                raise ValueError("splits must be listed in time order and must not overlap")
        if self.horizon_days < 1 or self.eligibility_lookback_days < 1:
            raise ValueError("horizon_days and eligibility_lookback_days must be positive")
        return self


class ModelFamily(_Frozen):
    grid: dict[str, list[float | int | str]] = {}


class MLflowConfig(_Frozen):
    experiment: str
    tracking_uri: str
    artifact_location: str


class ModelsConfig(_Frozen):
    horizon_days: int
    selection_metric: Literal["c_within_landmark", "c_pooled"]
    bootstrap_reps: int
    families: dict[str, ModelFamily]
    mlflow: MLflowConfig


class Thresholds(_Frozen):
    min: float
    max: float
    n: int


class DecisionConfig(_Frozen):
    horizon_days: int
    population_case_rate: float
    population_case_rate_grid: tuple[float, ...]
    capacity_shares: tuple[float, ...]
    headline_capacity_share: float
    dca_thresholds: Thresholds
    age_bands: tuple[int, ...]
    top_countries: int
    models: tuple[str, ...]

    @model_validator(mode="after")
    def _rates(self) -> DecisionConfig:
        for r in (self.population_case_rate, *self.population_case_rate_grid):
            if not 0 < r < 1:
                raise ValueError("population case rates must be in (0, 1)")
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


@cache
def load_landmarks_config(path: Path = CONFIG_DIR / "landmarks.yaml") -> LandmarksConfig:
    return LandmarksConfig.model_validate(_read_yaml(path))


@cache
def load_models_config(path: Path = CONFIG_DIR / "models.yaml") -> ModelsConfig:
    return ModelsConfig.model_validate(_read_yaml(path))


@cache
def load_decision_config(path: Path = CONFIG_DIR / "decision.yaml") -> DecisionConfig:
    return DecisionConfig.model_validate(_read_yaml(path))
