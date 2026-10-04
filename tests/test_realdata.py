"""Checks against the licensed dataset. Skipped automatically when it is not present."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ttr.config import DataConfig, load_data_config
from ttr.data.ingest import ingest

CFG = load_data_config()
RAW = CFG.resolve(CFG.raw_dir)

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (RAW / CFG.files.daily).exists(), reason="licensed bwin data not available"
    ),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


@pytest.fixture(scope="module")
def interim(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("interim")
    ingest(CFG, out_dir=out)
    return out


def test_counts_match_codebook(interim: Path) -> None:
    demo = pd.read_parquet(interim / "demographics.parquet")
    daily = pd.read_parquet(interim / "daily.parquet")
    rg = pd.read_parquet(interim / "rg.parquet")
    analytic = pd.read_parquet(interim / "analytic.parquet")
    assert len(demo) == 4134
    assert (demo["rg_case"] == 1).sum() == 2068
    assert len(daily) == 981_782
    assert daily["user_id"].nunique() == 4113
    assert len(rg) == 2068
    assert len(analytic) == 4132


def test_rg_dates_fall_in_event_window(interim: Path, data_cfg: DataConfig = CFG) -> None:
    rg = pd.read_parquet(interim / "rg.parquet")
    first = rg["rg_first_date"].dropna()
    assert first.min() >= pd.Timestamp(data_cfg.event_window.start)
    assert first.max() <= pd.Timestamp(data_cfg.event_window.end)
    assert rg["rg_first_date"].isna().sum() == 3


def test_missing_demographics_only_in_controls(interim: Path) -> None:
    """The leakage trap: demographic missingness alone identifies 167 controls."""
    demo = pd.read_parquet(interim / "demographics.parquet")
    missing = demo["country_name"].isna()
    assert missing.sum() == 167
    assert (demo.loc[missing, "rg_case"] == 0).all()
