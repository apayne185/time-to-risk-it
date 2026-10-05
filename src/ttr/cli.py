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


@app.command()
def train(
    processed_dir: Annotated[
        Path | None, typer.Option(help="Defaults to configs/data.yaml.")
    ] = None,
    label: Annotated[
        str | None, typer.Option(help="Label definition from configs/labels.yaml.")
    ] = None,
    family: Annotated[
        list[str] | None, typer.Option(help="Only these model families (repeatable).")
    ] = None,
    model_dir: Annotated[Path | None, typer.Option(help="Defaults to models/<label>.")] = None,
    report: Annotated[
        Path | None, typer.Option(help="Defaults to reports/models_<label>.md.")
    ] = None,
) -> None:
    """Select, refit and evaluate survival models; log runs to MLflow."""
    from ttr.config import PROJECT_ROOT, load_labels_config, load_models_config
    from ttr.train import train_all

    cfg = load_data_config()
    src = cfg.resolve(processed_dir or cfg.processed_dir)
    name = label or load_labels_config().default
    results = train_all(
        src / f"features_{name}.parquet",
        name,
        load_models_config(),
        model_dir or PROJECT_ROOT / "models" / name,
        report or PROJECT_ROOT / "reports" / f"models_{name}.md",
        families=family,
    )
    for r in results:
        typer.echo(
            f"{r.family:>14}: val C {r.validation.c_within_landmark:.3f} | "
            f"test C {r.test.c_within_landmark:.3f} "
            f"({r.test_ci[0]:.3f}-{r.test_ci[1]:.3f})"
        )


@app.command()
def evaluate(
    processed_dir: Annotated[
        Path | None, typer.Option(help="Defaults to configs/data.yaml.")
    ] = None,
    label: Annotated[
        str | None, typer.Option(help="Label definition from configs/labels.yaml.")
    ] = None,
    model_dir: Annotated[Path | None, typer.Option(help="Defaults to models/<label>.")] = None,
    out_dir: Annotated[Path, typer.Option(help="Report and figures.")] = Path("reports"),
) -> None:
    """Decision layer: calibrated 30-day risk, net benefit, capacity, subgroups, drivers."""
    import pandas as pd

    from ttr.config import PROJECT_ROOT, load_decision_config, load_labels_config
    from ttr.evaluate.report import write_report
    from ttr.evaluate.run import run_evaluation, save_decision_bundle

    cfg = load_data_config()
    dcfg = load_decision_config()
    src = cfg.resolve(processed_dir or cfg.processed_dir)
    name = label or load_labels_config().default
    mdir = model_dir or PROJECT_ROOT / "models" / name
    result = run_evaluation(
        pd.read_parquet(src / f"features_{name}.parquet"),
        pd.read_parquet(src / "players.parquet"),
        mdir,
        dcfg,
    )
    bundle = save_decision_bundle(result, mdir, dcfg, name)
    report = write_report(result, dcfg, out_dir, mdir, name)
    ev = result.evaluations[result.decision_family]
    cap = ev.recall_at(dcfg.headline_capacity_share)
    typer.echo(
        f"decision model: {result.decision_family} (test AUC {ev.test_auc:.3f}); "
        f"top {dcfg.headline_capacity_share:.1%} reaches {cap:.0%} of next-month cases"
    )
    typer.echo(f"bundle: {bundle}\nreport: {report}")


@app.command()
def serve(
    bundle: Annotated[Path | None, typer.Option(help="Decision bundle (.joblib).")] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Run the HTTP scoring service."""
    import os

    import uvicorn

    if bundle is not None:
        os.environ["TTR_MODEL_BUNDLE"] = str(bundle.resolve())
    uvicorn.run("ttr.serve.app:app", host=host, port=port)


def _default_bundle(label: str | None) -> Path:
    from ttr.config import PROJECT_ROOT, load_labels_config

    return (
        PROJECT_ROOT
        / "models"
        / (label or load_labels_config().default)
        / ("decision_model.joblib")
    )


@app.command()
def score(
    input: Annotated[Path, typer.Option(help="Feature table (parquet) to score.")],
    output: Annotated[Path, typer.Option(help="Where to write scores (parquet).")],
    bundle: Annotated[Path | None, typer.Option(help="Decision bundle (.joblib).")] = None,
    landmark: Annotated[
        str | None, typer.Option(help="Only score rows at this landmark (YYYY-MM-DD).")
    ] = None,
) -> None:
    """Batch-score a feature table with the same scorer as the HTTP service."""
    import json

    import pandas as pd

    from ttr.serve.scoring import Scorer

    scorer = Scorer.load(bundle or _default_bundle(None))
    table = pd.read_parquet(input)
    if landmark is not None:
        table = table[table["landmark"] == pd.Timestamp(landmark)]
    scores = scorer.score(table.reset_index(drop=True))
    keys = [c for c in ("user_id", "landmark") if c in table.columns]
    out = pd.concat([table[keys].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    out["top_drivers"] = out["top_drivers"].map(json.dumps)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    typer.echo(f"scored {len(out):,} rows, {int(out['flagged'].sum()):,} flagged -> {output}")


@app.command()
def monitor(
    processed_dir: Annotated[
        Path | None, typer.Option(help="Defaults to configs/data.yaml.")
    ] = None,
    label: Annotated[str | None, typer.Option()] = None,
    bundle: Annotated[Path | None, typer.Option(help="Decision bundle (.joblib).")] = None,
    reference_split: Annotated[str, typer.Option()] = "train",
    current_split: Annotated[str, typer.Option()] = "test",
    report: Annotated[Path, typer.Option()] = Path("reports/monitoring.md"),
    write_shift: Annotated[
        bool, typer.Option(help="Store the recommended intercept shift in the bundle.")
    ] = False,
) -> None:
    """Input and score drift (PSI), plus the monthly intercept update from latest outcomes."""
    import joblib
    import pandas as pd

    from ttr.config import load_decision_config, load_labels_config
    from ttr.evaluate.run import horizon_outcome
    from ttr.evaluate.weights import case_control_weights
    from ttr.monitor import monitor as run_monitor
    from ttr.monitor import monitoring_markdown
    from ttr.serve.scoring import Scorer

    cfg = load_data_config()
    dcfg = load_decision_config()
    name = label or load_labels_config().default
    path = bundle or _default_bundle(name)
    scorer = Scorer.load(path)
    table = pd.read_parquet(
        cfg.resolve(processed_dir or cfg.processed_dir) / f"features_{name}.parquet"
    )
    ref = table[table["split"] == reference_split].reset_index(drop=True)
    cur = table[table["split"] == current_split].reset_index(drop=True)
    ref_scores, cur_scores = scorer.score(ref, explain=0), scorer.score(cur, explain=0)

    weights = case_control_weights(table, dcfg.population_case_rate)
    latest = cur["landmark"].max()
    mask = (cur["landmark"] == latest).to_numpy()
    outcomes = pd.DataFrame(
        {
            # Undo the active shift so the recommendation is absolute, not incremental.
            "probability": scorer.bundle["recalibrator"].transform(
                scorer.bundle["model"].predict_event_prob(
                    cur.loc[mask, scorer.features], scorer.bundle["horizon_days"]
                )
            ),
            "y": horizon_outcome(cur[mask], scorer.bundle["horizon_days"]),
            "w": weights[(table["split"] == current_split).to_numpy()][mask],
        }
    )
    result = run_monitor(ref, cur, scorer.features, ref_scores, cur_scores, outcomes)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(monitoring_markdown(result, reference_split, current_split))
    typer.echo(
        f"score PSI {result.score_psi:.3f}; "
        f"recommended shift {result.recommended_shift:+.3f}; report: {report}"
    )
    if write_shift and result.recommended_shift is not None:
        scorer.bundle["intercept_shift"] = result.recommended_shift
        joblib.dump(scorer.bundle, path)
        typer.echo(f"intercept shift written to {path}")
