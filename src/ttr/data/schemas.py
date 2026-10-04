"""Pandera contracts for the canonical (renamed, typed) raw tables.

Facts encoded here come from the dataset codebook; a violation means either the files are
not the published ones or a parsing assumption is wrong.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from ttr.config import load_data_config

ALL_PRODUCTS = sorted(load_data_config().products.family_of())
MONEY_VALID = list(load_data_config().products.money_valid)


class Demographics(pa.DataFrameModel):
    user_id: Series[int] = pa.Field(gt=0, unique=True)
    rg_case: Series[int] = pa.Field(isin=[0, 1])
    country_name: Series[str] = pa.Field(nullable=True)
    language_name: Series[str] = pa.Field(nullable=True)
    gender: Series[str] = pa.Field(isin=["M", "F"])
    year_of_birth: Series[pd.Int64Dtype] = pa.Field(ge=1900, le=2000, nullable=True)
    registration_date: Series[pd.Timestamp] = pa.Field(nullable=True)
    first_deposit_date: Series[pd.Timestamp]

    class Config:
        strict = True
        coerce = True


class Daily(pa.DataFrameModel):
    user_id: Series[int] = pa.Field(gt=0)
    date: Series[pd.Timestamp]
    product_type: Series[int] = pa.Field(isin=ALL_PRODUCTS)
    turnover: Series[float] = pa.Field(ge=0, nullable=True)
    hold: Series[float] = pa.Field(nullable=True)
    n_bets: Series[float] = pa.Field(ge=0, nullable=True)

    @pa.dataframe_check
    @classmethod
    def money_only_for_valid_products(cls, df: pd.DataFrame) -> pd.Series[bool]:
        """Turnover/hold are present exactly for bwin's own (non-vendor) products."""
        valid = df["product_type"].isin(MONEY_VALID)
        return df["turnover"].notna() == valid

    @pa.dataframe_check(raise_warning=True)
    @classmethod
    def hold_not_above_turnover(cls, df: pd.DataFrame) -> pd.Series[bool]:
        """Losses should not exceed stakes on a day.

        Warning only: the real data has 53 such rows, all casino product 8 (likely bonus
        accounting). They are handled in cleaning rather than rejected at ingest.
        """
        return ~(df["hold"] > df["turnover"] + 1e-6)

    class Config:
        strict = True
        coerce = True


class RG(pa.DataFrameModel):
    user_id: Series[int] = pa.Field(gt=0, unique=True)
    rg_n_events: Series[int] = pa.Field(ge=1)
    rg_first_date: Series[pd.Timestamp] = pa.Field(nullable=True)
    rg_last_date: Series[pd.Timestamp] = pa.Field(nullable=True)
    event_type_first: Series[pd.Int64Dtype] = pa.Field(ge=1, le=13, nullable=True)
    intervention_type_first: Series[pd.Int64Dtype] = pa.Field(ge=1, le=18, nullable=True)

    @pa.dataframe_check
    @classmethod
    def last_not_before_first(cls, df: pd.DataFrame) -> pd.Series[bool]:
        return ~(df["rg_last_date"] < df["rg_first_date"])

    class Config:
        strict = True
        coerce = True


class Analytic(pa.DataFrameModel):
    user_id: Series[int] = pa.Field(gt=0, unique=True)
    rg_case: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = False
