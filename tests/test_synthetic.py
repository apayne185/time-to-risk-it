from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ttr.config import DataConfig, load_data_config
from ttr.data.ingest import ingest
from ttr.data.synthetic import SyntheticSpec, generate, write_synthetic

SPEC = SyntheticSpec(n_pairs=60, seed=7, p_missing_demographics=0.2, p_split_record=0.05)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return generate(SPEC, load_data_config())


def test_deterministic_for_seed(tables: dict[str, pd.DataFrame], data_cfg: DataConfig) -> None:
    again = generate(SPEC, data_cfg)
    for name, df in tables.items():
        pd.testing.assert_frame_equal(df, again[name])


def test_passes_ingest_contracts(tmp_path: Path, data_cfg: DataConfig) -> None:
    raw = tmp_path / "raw"
    counts = write_synthetic(SPEC, data_cfg, raw)
    report = ingest(data_cfg, raw, tmp_path / "interim", verify=False, include_analytic=False)
    assert report.n_cases == report.n_controls == SPEC.n_pairs
    assert report.n_daily_rows == counts["daily"]


def test_matched_on_first_deposit(tables: dict[str, pd.DataFrame]) -> None:
    demo = tables["demographics"]
    by_group = demo.groupby("rg_case")["first_deposit_date"].apply(lambda s: sorted(s))
    assert by_group[0] == by_group[1]


def test_rg_dates_in_window(tables: dict[str, pd.DataFrame], data_cfg: DataConfig) -> None:
    first = tables["rg"]["rg_first_date"].dropna()
    assert first.min() >= pd.Timestamp(data_cfg.event_window.start)
    assert first.max() <= pd.Timestamp(data_cfg.event_window.end)


def test_reproduces_dataset_quirks(tables: dict[str, pd.DataFrame]) -> None:
    demo, daily, rg = tables["demographics"], tables["daily"], tables["rg"]
    # leakage trap: missing demographics only among controls
    missing = demo["country_name"].isna()
    assert missing.any()
    assert (demo.loc[missing, "rg_case"] == 0).all()
    # split records
    assert daily.duplicated(["user_id", "date", "product_type"]).any()
    # activity after the RG event exists (features must exclude it)
    merged = daily.merge(rg[["user_id", "rg_first_date"]], on="user_id")
    assert (merged["date"] > merged["rg_first_date"]).any()
    # re-openings after an earlier closure
    assert ((rg["event_type_first"] == 2) & (rg["intervention_type_first"] == 2)).any()


def test_cases_escalate_before_event(tables: dict[str, pd.DataFrame]) -> None:
    """There must be a learnable signal, or downstream model tests are meaningless."""
    daily, rg, demo = tables["daily"], tables["rg"], tables["demographics"]
    merged = daily.merge(rg[["user_id", "rg_first_date"]], on="user_id")
    window = merged[
        (merged["date"] < merged["rg_first_date"])
        & (merged["date"] >= merged["rg_first_date"] - pd.Timedelta(days=90))
    ]
    controls = daily[daily["user_id"].isin(demo.loc[demo["rg_case"] == 0, "user_id"])]
    assert (window["product_type"] == 2).mean() > (controls["product_type"] == 2).mean()
