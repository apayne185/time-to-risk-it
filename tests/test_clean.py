from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ttr.config import DataConfig
from ttr.data.clean import build_players, clean, clean_daily, split_country
from ttr.data.ingest import ingest
from ttr.data.synthetic import SyntheticSpec, write_synthetic

TS = pd.Timestamp


def _daily(
    rows: list[tuple[int, str, int, float | None, float | None, float | None]],
) -> pd.DataFrame:
    df = pd.DataFrame(
        rows, columns=["user_id", "date", "product_type", "turnover", "hold", "n_bets"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_split_records_are_summed(data_cfg: DataConfig) -> None:
    out = clean_daily(
        _daily([(1, "2008-01-01", 2, 20.5, 20.5, 2), (1, "2008-01-01", 2, 100.0, 100.0, 10)]),
        data_cfg,
    )
    assert len(out) == 1
    assert out[["turnover", "hold", "n_bets"]].iloc[0].tolist() == [120.5, 120.5, 12]


def test_vendor_money_stays_missing_after_merge(data_cfg: DataConfig) -> None:
    out = clean_daily(
        _daily([(1, "2008-01-01", 10, None, None, 2), (1, "2008-01-01", 10, None, None, 3)]),
        data_cfg,
    )
    assert out.loc[0, "n_bets"] == 5
    assert pd.isna(out.loc[0, "turnover"]) and pd.isna(out.loc[0, "hold"])
    assert not out.loc[0, "money_valid"]


def test_empty_rows_dropped_settlements_kept(data_cfg: DataConfig) -> None:
    out = clean_daily(
        _daily(
            [
                (1, "2008-01-01", 1, 10.0, 10.0, 1),
                (1, "2008-01-02", 1, 0.0, 0.0, 0),  # empty
                (1, "2008-01-03", 1, 0.0, -25.0, 0),
            ]
        ),  # settled win
        data_cfg,
    )
    assert out["date"].tolist() == [TS("2008-01-01"), TS("2008-01-03")]
    assert out["is_bet_day"].tolist() == [True, False]


def test_missing_bet_count_treated_as_zero(data_cfg: DataConfig) -> None:
    out = clean_daily(_daily([(1, "2008-01-01", 1, 5.0, 5.0, None)]), data_cfg)
    assert out.loc[0, "n_bets"] == 0


def test_product_family(data_cfg: DataConfig) -> None:
    out = clean_daily(
        _daily([(1, "2008-01-01", 2, 1.0, 1.0, 1), (1, "2008-01-01", 8, 1.0, 1.0, 1)]), data_cfg
    )
    assert out["product_family"].astype(str).tolist() == ["live_action", "casino"]


@pytest.mark.parametrize(
    ("raw", "country", "site"),
    [
        ("Greece.BAW", "Greece", "BAW"),
        ("Germany", "Germany", None),
        ("Bosnia and Herzego", "Bosnia and Herzegovina", None),
        (None, None, None),
    ],
)
def test_split_country(raw: str | None, country: str | None, site: str | None) -> None:
    out = split_country(pd.Series([raw], dtype="string"))
    assert (out.loc[0, "country"] if pd.notna(out.loc[0, "country"]) else None) == country
    assert (out.loc[0, "site"] if pd.notna(out.loc[0, "site"]) else None) == site


def test_exclusion_reasons(tiny_tables: dict[str, pd.DataFrame], data_cfg: DataConfig) -> None:
    demo, rg = tiny_tables["demographics"], tiny_tables["rg"].copy()
    daily = clean_daily(tiny_tables["daily"], data_cfg)
    # 201's first bet is 2008-03-02; move its RG event onto that day -> no pre-event history.
    rg.loc[rg["user_id"] == 201, "rg_first_date"] = TS("2008-03-02")
    players = build_players(demo, rg, daily).set_index("user_id")
    assert pd.isna(players.loc[101, "exclusion_reason"])
    assert players.loc[201, "exclusion_reason"] == "no_pre_event_history"
    assert players.loc[202, "exclusion_reason"] == "no_betting_activity"
    assert players.loc[102, "demographics_missing"]


def test_clean_end_to_end_on_synthetic(tmp_path: Path, data_cfg: DataConfig) -> None:
    write_synthetic(SyntheticSpec(n_pairs=40, seed=3), data_cfg, tmp_path / "raw")
    ingest(data_cfg, tmp_path / "raw", tmp_path / "interim", verify=False, include_analytic=False)
    flow = clean(data_cfg, tmp_path / "interim", tmp_path / "processed")
    assert flow[0].n_players == 80
    assert [s.n_players for s in flow] == sorted((s.n_players for s in flow), reverse=True)
    daily = pd.read_parquet(tmp_path / "processed" / "daily.parquet")
    assert not daily.duplicated(["user_id", "date", "product_type"]).any()
    assert (~daily["is_bet_day"]).any()  # settlement rows survive cleaning
