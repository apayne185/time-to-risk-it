"""Command-line entry point: ``ttr <command>``."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ttr.config import load_data_config

app = typer.Typer(help="time-to-risk-it: player-risk early warning pipeline.", no_args_is_help=True)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Directory for the raw-format files.")] = Path(
        "data/raw/synthetic"
    ),
    n_pairs: Annotated[int, typer.Option(help="Number of case-control pairs.")] = 300,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Generate a synthetic dataset in the raw distribution format."""
    from ttr.data.synthetic import SyntheticSpec, write_synthetic

    counts = write_synthetic(SyntheticSpec(n_pairs=n_pairs, seed=seed), load_data_config(), out)
    typer.echo(f"wrote synthetic data to {out}: {counts}")


@app.command()
def ingest(
    raw_dir: Annotated[Path | None, typer.Option(help="Defaults to configs/data.yaml.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Defaults to configs/data.yaml.")] = None,
    synthetic: Annotated[
        bool, typer.Option(help="Skip checksums and the analytic file (synthetic data).")
    ] = False,
) -> None:
    """Validate raw files and write typed parquet tables."""
    from ttr.data.ingest import ingest as run_ingest

    report = run_ingest(
        load_data_config(),
        raw_dir,
        out_dir,
        verify=not synthetic,
        include_analytic=not synthetic,
    )
    for key, value in asdict(report).items():
        typer.echo(f"{key:>22}: {value}")


@app.command()
def clean(
    interim_dir: Annotated[Path | None, typer.Option(help="Defaults to configs/data.yaml.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Defaults to configs/data.yaml.")] = None,
) -> None:
    """Merge split records, drop empty rows, build the player table and cohort flow."""
    from ttr.data.clean import clean as run_clean

    for step in run_clean(load_data_config(), interim_dir, out_dir):
        typer.echo(
            f"{step.step:>40}: {step.n_players:>5} ({step.n_cases} cases, "
            f"{step.n_controls} controls)"
        )
