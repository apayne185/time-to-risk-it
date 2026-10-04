from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ttr.config import DataConfig, load_data_config
from ttr.data.raw import write_raw

TS = pd.Timestamp


@pytest.fixture
def data_cfg() -> DataConfig:
    return load_data_config()


@pytest.fixture
def tiny_tables() -> dict[str, pd.DataFrame]:
    """Two matched pairs in canonical form, including the dataset's known quirks."""
    demographics = pd.DataFrame(
        {
            "user_id": [101, 102, 201, 202],
            "rg_case": [1, 0, 1, 0],
            "country_name": ["Germany", None, "Italy.IT", "Austria"],
            "language_name": ["German", None, "Italian", "German"],
            "gender": ["M", "M", "F", "M"],
            "year_of_birth": pd.array([1980, None, 1975, 1990], dtype="Int64"),
            "registration_date": [TS("2007-01-01"), pd.NaT, TS("2008-03-01"), TS("2008-03-01")],
            "first_deposit_date": [
                TS(d) for d in ["2007-01-02", "2007-01-02", "2008-03-01", "2008-03-01"]
            ],
        }
    )
    daily = pd.DataFrame(
        {
            "user_id": [101, 101, 101, 102, 201, 201],
            "date": [
                TS(d)
                for d in [
                    "2007-01-02",
                    "2007-01-02",
                    "2009-01-05",
                    "2007-02-01",
                    "2008-03-02",
                    "2008-03-02",
                ]
            ],
            "product_type": [1, 2, 10, 1, 2, 2],  # 10 = vendor poker (no money values)
            "turnover": [10.0, 5.5, None, 2.0, 7.0, 3.0],
            "hold": [10.0, -4.5, None, 1.0, 7.0, 3.0],
            "n_bets": [1.0, 3.0, 4.0, 1.0, 2.0, 1.0],
        }
    )
    rg = pd.DataFrame(
        {
            "user_id": [101, 201],
            "rg_n_events": [1, 2],
            "rg_first_date": [TS("2009-02-01"), TS("2008-12-01")],
            "rg_last_date": [TS("2009-02-01"), TS("2009-03-01")],
            "event_type_first": pd.array([10, 2], dtype="Int64"),
            "intervention_type_first": pd.array([1, None], dtype="Int64"),
        }
    )
    return {"demographics": demographics, "daily": daily, "rg": rg}


@pytest.fixture
def tiny_raw_dir(
    tmp_path: Path, tiny_tables: dict[str, pd.DataFrame], data_cfg: DataConfig
) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for table, df in tiny_tables.items():
        write_raw(df, raw / getattr(data_cfg.files, table), table)  # type: ignore[arg-type]
    return raw
