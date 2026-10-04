from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from ttr.config import DataConfig
from ttr.data.ingest import IntegrityError, ingest, sha256, verify_checksums
from ttr.data.raw import read_raw, write_raw


def _ingest(cfg: DataConfig, raw: Path, out: Path) -> None:
    ingest(cfg, raw, out, verify=False, include_analytic=False)


def test_raw_round_trip(
    tiny_raw_dir: Path, tiny_tables: dict[str, pd.DataFrame], data_cfg: DataConfig
) -> None:
    for table, expected in tiny_tables.items():
        got = read_raw(tiny_raw_dir / getattr(data_cfg.files, table), table)  # type: ignore[arg-type]
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_ingest_writes_parquet_and_report(
    tiny_raw_dir: Path, tmp_path: Path, data_cfg: DataConfig
) -> None:
    out = tmp_path / "interim"
    report = ingest(data_cfg, tiny_raw_dir, out, verify=False, include_analytic=False)
    assert (report.n_cases, report.n_controls, report.n_daily_rows) == (2, 2, 6)
    assert {p.name for p in out.iterdir()} == {
        "demographics.parquet",
        "daily.parquet",
        "rg.parquet",
        "ingest_report.json",
    }
    assert json.loads((out / "ingest_report.json").read_text())["n_players"] == 4
    daily = pd.read_parquet(out / "daily.parquet")
    assert pd.api.types.is_datetime64_any_dtype(daily["date"])


def test_checksum_mismatch_is_rejected(tiny_raw_dir: Path, data_cfg: DataConfig) -> None:
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        verify_checksums(tiny_raw_dir, data_cfg, ("demographics",))


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"abc")
    assert sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_rg_details_must_match_cases(
    tiny_raw_dir: Path, tiny_tables: dict[str, pd.DataFrame], tmp_path: Path, data_cfg: DataConfig
) -> None:
    rg = tiny_tables["rg"].iloc[:1]
    write_raw(rg, tiny_raw_dir / data_cfg.files.rg, "rg")
    with pytest.raises(IntegrityError, match="cases without details"):
        _ingest(data_cfg, tiny_raw_dir, tmp_path / "out")


def test_daily_users_must_exist(
    tiny_raw_dir: Path, tiny_tables: dict[str, pd.DataFrame], tmp_path: Path, data_cfg: DataConfig
) -> None:
    daily = tiny_tables["daily"].copy()
    daily.loc[0, "user_id"] = 999
    write_raw(daily, tiny_raw_dir / data_cfg.files.daily, "daily")
    with pytest.raises(IntegrityError, match="not in demographics"):
        _ingest(data_cfg, tiny_raw_dir, tmp_path / "out")


def test_money_on_vendor_product_fails_schema(
    tiny_raw_dir: Path, tiny_tables: dict[str, pd.DataFrame], tmp_path: Path, data_cfg: DataConfig
) -> None:
    daily = tiny_tables["daily"].copy()
    daily.loc[2, ["turnover", "hold"]] = [1.0, 1.0]  # product 10 is a vendor product
    write_raw(daily, tiny_raw_dir / data_cfg.files.daily, "daily")
    with pytest.raises(pandera.errors.SchemaErrors, match="money_only_for_valid_products"):
        _ingest(data_cfg, tiny_raw_dir, tmp_path / "out")
