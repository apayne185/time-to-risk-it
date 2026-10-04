"""Raw files -> validated, typed parquet tables in the interim directory.

Ingest does not change values; it verifies integrity (checksums, schemas, cross-table
keys) and records what it saw. Cleaning decisions live in a separate step.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args

import pandas as pd
import pandera.pandas as pa

from ttr.config import DataConfig, TableName
from ttr.data import schemas
from ttr.data.raw import read_raw

log = logging.getLogger(__name__)

SCHEMAS: dict[TableName, type[pa.DataFrameModel]] = {
    "demographics": schemas.Demographics,
    "daily": schemas.Daily,
    "rg": schemas.RG,
    "analytic": schemas.Analytic,
}


class IntegrityError(ValueError):
    """Raw data failed a checksum or cross-table consistency check."""


@dataclass(frozen=True)
class IngestReport:
    n_players: int
    n_cases: int
    n_controls: int
    n_daily_rows: int
    n_players_with_daily: int
    daily_date_min: str
    daily_date_max: str
    n_rg_rows: int
    checksums_verified: bool


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(raw_dir: Path, cfg: DataConfig, tables: tuple[TableName, ...]) -> None:
    for table in tables:
        path = raw_dir / getattr(cfg.files, table)
        expected = getattr(cfg.checksums, table)
        actual = sha256(path)
        if actual != expected:
            raise IntegrityError(
                f"{path.name}: checksum mismatch (expected {expected[:12]}…, got {actual[:12]}…). "
                "Re-download the file or pass verify_checksums=False for synthetic data."
            )


def check_referential_integrity(tables: dict[TableName, pd.DataFrame]) -> None:
    demo = tables["demographics"]
    players = set(demo["user_id"])
    cases = set(demo.loc[demo["rg_case"] == 1, "user_id"])

    unknown_daily = set(tables["daily"]["user_id"]) - players
    if unknown_daily:
        raise IntegrityError(f"{len(unknown_daily)} daily-aggregate users not in demographics")

    rg_users = set(tables["rg"]["user_id"])
    if rg_users != cases:
        raise IntegrityError(
            f"RG details do not match RG cases: {len(rg_users - cases)} non-case rows, "
            f"{len(cases - rg_users)} cases without details"
        )

    if "analytic" in tables:
        unknown_analytic = set(tables["analytic"]["user_id"]) - players
        if unknown_analytic:
            raise IntegrityError(f"{len(unknown_analytic)} analytic users not in demographics")


def load_raw_tables(
    raw_dir: Path, cfg: DataConfig, tables: tuple[TableName, ...]
) -> dict[TableName, pd.DataFrame]:
    out: dict[TableName, pd.DataFrame] = {}
    for table in tables:
        df = read_raw(raw_dir / getattr(cfg.files, table), table)
        out[table] = SCHEMAS[table].validate(df, lazy=True)
        log.info("validated %s: %d rows", table, len(df))
    return out


def ingest(
    cfg: DataConfig,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    verify: bool = True,
    include_analytic: bool = True,
) -> IngestReport:
    """Validate the raw files and write them as parquet. Returns a summary report."""
    raw_dir = cfg.resolve(raw_dir or cfg.raw_dir)
    out_dir = cfg.resolve(out_dir or cfg.interim_dir)
    tables: tuple[TableName, ...] = tuple(
        t for t in get_args(TableName) if include_analytic or t != "analytic"
    )

    if verify:
        verify_checksums(raw_dir, cfg, tables)
    frames = load_raw_tables(raw_dir, cfg, tables)
    check_referential_integrity(frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    for table, df in frames.items():
        df.to_parquet(out_dir / f"{table}.parquet", index=False)

    demo, daily = frames["demographics"], frames["daily"]
    report = IngestReport(
        n_players=len(demo),
        n_cases=int((demo["rg_case"] == 1).sum()),
        n_controls=int((demo["rg_case"] == 0).sum()),
        n_daily_rows=len(daily),
        n_players_with_daily=int(daily["user_id"].nunique()),
        daily_date_min=str(daily["date"].min().date()),
        daily_date_max=str(daily["date"].max().date()),
        n_rg_rows=len(frames["rg"]),
        checksums_verified=verify,
    )
    (out_dir / "ingest_report.json").write_text(json.dumps(asdict(report), indent=2) + "\n")
    return report
