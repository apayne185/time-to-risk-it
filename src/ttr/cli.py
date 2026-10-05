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


@app.command()
def eda(
    processed_dir: Annotated[
        Path | None, typer.Option(help="Defaults to configs/data.yaml.")
    ] = None,
    out_dir: Annotated[Path, typer.Option(help="Where the report and figures go.")] = Path(
        "reports"
    ),
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Replicate the paper's DFA and write the exploratory report with figures."""
    import pandas as pd

    from ttr.analysis.report import build_eda_report

    cfg = load_data_config()
    src = cfg.resolve(processed_dir or cfg.processed_dir)
    outputs = build_eda_report(
        pd.read_parquet(src / "daily.parquet"),
        pd.read_parquet(src / "players.parquet"),
        out_dir,
        seed,
    )
    for r in outputs.results:
        typer.echo(f"{r.name:>14}: AUC {r.cv_auc:.3f}, accuracy {r.cv_accuracy:.1%}")
    typer.echo(f"report: {outputs.report}")


@app.command()
def landmarks(
    processed_dir: Annotated[
        Path | None, typer.Option(help="Defaults to configs/data.yaml.")
    ] = None,
    label: Annotated[
        str | None, typer.Option(help="Label definition from configs/labels.yaml.")
    ] = None,
) -> None:
    """Build the landmark table: risk sets, censored outcomes, temporal splits and folds."""
    import pandas as pd

    from ttr.config import load_labels_config, load_landmarks_config
    from ttr.labels import get_definition
    from ttr.landmarks import build_landmarks, summarise

    cfg = load_data_config()
    src = cfg.resolve(processed_dir or cfg.processed_dir)
    name = label or load_labels_config().default
    lm = build_landmarks(
        pd.read_parquet(src / "players.parquet"),
        pd.read_parquet(src / "daily.parquet"),
        cfg,
        load_landmarks_config(),
        get_definition(name),
    )
    out = src / f"landmarks_{name}.parquet"
    lm.to_parquet(out, index=False)
    summary = summarise(lm)
    summary.to_csv(src / f"landmarks_{name}_summary.csv", index=False)
    typer.echo(summary.to_string(index=False))
    typer.echo(
        f"{len(lm):,} rows, {lm['user_id'].nunique():,} players, "
        f"{int(lm['event'].sum()):,} events -> {out}"
    )


@app.command()
def features(
    processed_dir: Annotated[
        Path | None, typer.Option(help="Defaults to configs/data.yaml.")
    ] = None,
    label: Annotated[
        str | None, typer.Option(help="Label definition from configs/labels.yaml.")
    ] = None,
    docs: Annotated[bool, typer.Option(help="Only regenerate docs/features.md.")] = False,
) -> None:
    """Compute model features for every landmark row (activity strictly before it)."""
    from ttr.config import PROJECT_ROOT, load_labels_config
    from ttr.features import FEATURES, build_feature_table, write_feature_docs

    if docs:
        typer.echo(f"wrote {write_feature_docs(PROJECT_ROOT / 'docs' / 'features.md')}")
        return

    import pandas as pd

    cfg = load_data_config()
    src = cfg.resolve(processed_dir or cfg.processed_dir)
    name = label or load_labels_config().default
    table = build_feature_table(
        pd.read_parquet(src / f"landmarks_{name}.parquet"),
        pd.read_parquet(src / "daily.parquet"),
    )
    out = src / f"features_{name}.parquet"
    table.to_parquet(out, index=False)
    typer.echo(f"{len(table):,} rows x {len(FEATURES)} features -> {out}")
