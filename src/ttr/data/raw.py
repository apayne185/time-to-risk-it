"""Reading and writing the raw Transparency Project file format.

The files are tab-separated with a header row, a single space for missing values and
US-style dates (``m/d/YYYY``). This module is the only place that knows about that format;
everything downstream works with snake_case columns and proper dtypes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ttr.config import TableName

RAW_NA = " "
# Text columns use the SAS missing marker "." instead of a blank.
RAW_NA_VALUES = [RAW_NA, ".", ""]
RAW_DATE_FORMAT = "%m/%d/%Y"

# Raw header -> canonical column name, in raw file order.
COLUMNS: dict[TableName, dict[str, str]] = {
    "demographics": {
        "USERID": "user_id",
        "RG_case": "rg_case",
        "CountryName": "country_name",
        "LanguageName": "language_name",
        "Gender": "gender",
        "YearofBirth": "year_of_birth",
        "Registration_date": "registration_date",
        "First_Deposit_Date": "first_deposit_date",
    },
    "daily": {
        "UserID": "user_id",
        "Date": "date",
        "ProductType": "product_type",
        "Turnover": "turnover",
        "Hold": "hold",
        "NumberofBets": "n_bets",
    },
    "rg": {
        "UserID": "user_id",
        "RGsumevents": "rg_n_events",
        "RGFirst_Date": "rg_first_date",
        "RGLast_date": "rg_last_date",
        "Event_type_first": "event_type_first",
        "Interventiontype_first": "intervention_type_first",
    },
    # The analytic file has ~100 columns; only the keys are renamed.
    "analytic": {"UserID": "user_id", "RG_case": "rg_case"},
}

DATE_COLUMNS: dict[TableName, tuple[str, ...]] = {
    "demographics": ("registration_date", "first_deposit_date"),
    "daily": ("date",),
    "rg": ("rg_first_date", "rg_last_date"),
    "analytic": (),
}

NULLABLE_INT_COLUMNS: dict[TableName, tuple[str, ...]] = {
    "demographics": ("year_of_birth",),
    "daily": (),
    "rg": ("event_type_first", "intervention_type_first"),
    "analytic": (),
}


def read_raw(path: Path, table: TableName) -> pd.DataFrame:
    """Read one raw file into a typed frame with canonical column names."""
    df = pd.read_csv(path, sep="\t", na_values=RAW_NA_VALUES, keep_default_na=False)
    mapping = COLUMNS[table]
    missing = set(mapping) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {sorted(missing)}")
    df = df.rename(columns=mapping)
    if table != "analytic":
        df = df[list(mapping.values())]
    for col in DATE_COLUMNS[table]:
        df[col] = pd.to_datetime(df[col], format=RAW_DATE_FORMAT)
    for col in NULLABLE_INT_COLUMNS[table]:
        df[col] = df[col].astype("Int64")
    return df


def _format_date(s: pd.Series) -> pd.Series:
    out = (
        s.dt.month.astype("Int64").astype(str)
        + "/"
        + s.dt.day.astype("Int64").astype(str)
        + "/"
        + s.dt.year.astype("Int64").astype(str)
    )
    formatted: pd.Series = out.where(s.notna(), None)
    return formatted


def write_raw(df: pd.DataFrame, path: Path, table: TableName) -> None:
    """Write a canonical frame back out in the raw distribution format."""
    out = df.copy()
    for col in DATE_COLUMNS[table]:
        out[col] = _format_date(out[col])
    inverse = {v: k for k, v in COLUMNS[table].items()}
    out = out[list(inverse)].rename(columns=inverse)
    out.to_csv(path, sep="\t", index=False, na_rep=RAW_NA)
