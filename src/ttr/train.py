"""Train, select and evaluate survival models; track everything in MLflow.

1. For each model family, fit every grid point on the train split and score it on validation
   (XGBoost early-stops on validation).
2. Pick each family's best configuration by the selection metric (within-landmark C).
3. Refit it on train + validation (XGBoost with the early-stopped number of rounds) and evaluate
   once on the test split, with a player-level bootstrap CI.

The test split is touched exactly once per family, after selection.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from ttr.config import PROJECT_ROOT, ModelsConfig
from ttr.evaluate.metrics import Metrics, bootstrap_ci, evaluate
from ttr.features import FEATURES
from ttr.models.base import SurvivalData, SurvivalModel
from ttr.models.cox import StratifiedCox
from ttr.models.rule import RuleBaseline
from ttr.models.xgb import XGBAFT, XGBCox

log = logging.getLogger(__name__)


def build_model(
    family: str, params: dict[str, Any], num_boost_round: int | None = None
) -> SurvivalModel:
    params = dict(params)
    boost = {} if num_boost_round is None else {"num_boost_round": num_boost_round}
    if family == "rule_baseline":
        return RuleBaseline()
    if family == "cox":
        return StratifiedCox(**params)
    if family == "xgb_cox":
        return XGBCox(params=params, **boost)
    if family == "xgb_aft":
        sigma = float(params.pop("sigma", 1.0))
        return XGBAFT(params=params, sigma=sigma, **boost)
    raise ValueError(f"unknown model family {family!r}")


def grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def split_data(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(name): df.reset_index(drop=True) for name, df in table.groupby("split")}


def _score(model: SurvivalModel, data: SurvivalData, horizon: int) -> Metrics:
    return evaluate(
        data.time,
        data.event,
        model.predict_risk(data.X),
        model.predict_event_prob(data.X, horizon),
        data.landmark,
    )


@dataclass
class FamilyResult:
    family: str
    params: dict[str, Any]
    validation: Metrics
    test: Metrics
    test_ci: tuple[float, float]
    model: SurvivalModel


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _setup_mlflow(cfg: ModelsConfig) -> None:
    uri = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db = PROJECT_ROOT / uri.removeprefix("sqlite:///")
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(cfg.mlflow.experiment) is None:
        artifacts = (PROJECT_ROOT / cfg.mlflow.artifact_location).resolve()
        mlflow.create_experiment(cfg.mlflow.experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment)


def train_family(
    family: str,
    grid: dict[str, list[Any]],
    splits: dict[str, pd.DataFrame],
    cfg: ModelsConfig,
    features: list[str],
) -> FamilyResult:
    h = cfg.horizon_days
    train = SurvivalData.from_table(splits["train"], features)
    valid = SurvivalData.from_table(splits["validation"], features)
    test = SurvivalData.from_table(splits["test"], features)

    best: tuple[float, dict[str, Any], SurvivalModel, Metrics] | None = None
    for params in grid_points(grid):
        model = build_model(family, params).fit(train, valid=valid)
        val = _score(model, valid, h)
        with mlflow.start_run(run_name=f"{family}:{params}", nested=True):
            mlflow.set_tags({"family": family, "stage": "selection"})
            mlflow.log_params(model.params())
            mlflow.log_metrics(val.as_dict("val_"))
        key = getattr(val, cfg.selection_metric)
        if best is None or key > best[0]:
            best = (key, params, model, val)
    assert best is not None
    _, params, selected, val = best

    rounds = getattr(selected, "best_iteration_", None)
    final = build_model(family, params, None if rounds is None else rounds + 1)
    refit_table = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    final.fit(SurvivalData.from_table(refit_table, features))
    test_metrics = _score(final, test, h)
    ci_frame = splits["test"][["user_id", "time", "event", "landmark"]].assign(
        risk=final.predict_risk(test.X)
    )
    ci = bootstrap_ci(ci_frame, n_boot=cfg.bootstrap_reps)
    return FamilyResult(family, params, val, test_metrics, ci, final)


def _explain(result: FamilyResult) -> pd.Series | None:
    model = result.model
    if isinstance(model, StratifiedCox):
        return model.coefficients()
    if isinstance(model, (XGBCox, XGBAFT)):
        return model.importance()
    return None


_TABLE_HEAD = (
    "| Model | Validation C (within landmark) | Test C (within landmark, 95% CI) "
    "| Test C (pooled) | Selected params |\n|---|---|---|---|---|"
)


def results_markdown(results: list[FamilyResult], label: str, n: dict[str, int]) -> str:
    rows = "\n".join(
        f"| {r.family} | {r.validation.c_within_landmark:.3f} | "
        f"{r.test.c_within_landmark:.3f} [{r.test_ci[0]:.3f}, {r.test_ci[1]:.3f}] | "
        f"{r.test.c_pooled:.3f} | {_fmt_params(r.params)} |"
        for r in results
    )
    return f"""# Model comparison ({label} label)

Generated by `ttr train --label {label}`. Rows: train {n["train"]:,}, validation
{n["validation"]:,}, test {n["test"]:,}. Selection by within-landmark C on validation; the chosen
configuration is refit on train + validation and evaluated once on test. Test CI: player-level
bootstrap (percentile, 95%).

{_TABLE_HEAD}
{rows}

Within-landmark C compares players scored on the same date, which is what an RG team does. Pooled
C also rewards ranking late landmarks above early ones, which in this case-control sample reflects
the sampling design (ADR 0003), so it is shown for reference only. Probability-scale metrics
(calibration, decision curves) need case-control weights and are in the evaluation report.
"""


def _fmt_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "-"


def train_all(
    features_path: Path,
    label: str,
    cfg: ModelsConfig,
    model_dir: Path,
    report_path: Path | None = None,
    families: list[str] | None = None,
) -> list[FamilyResult]:
    table = pd.read_parquet(features_path)
    splits = split_data(table)
    features = list(FEATURES)
    _setup_mlflow(cfg)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with mlflow.start_run(run_name=f"train:{label}"):
        mlflow.set_tags(
            {"label": label, "git_sha": _git_sha(), "features_sha256": _file_sha(features_path)}
        )
        mlflow.log_params(
            {
                "horizon_days": cfg.horizon_days,
                "n_features": len(features),
                **{f"n_{k}": len(v) for k, v in splits.items()},
            }
        )
        for family, spec in cfg.families.items():
            if families and family not in families:
                continue
            log.info("training %s", family)
            with mlflow.start_run(run_name=family, nested=True):
                r = train_family(family, spec.grid, splits, cfg, features)
                mlflow.set_tags({"family": family, "stage": "final"})
                mlflow.log_params({f"best_{k}": v for k, v in r.model.params().items()})
                mlflow.log_metrics(
                    {
                        **r.validation.as_dict("val_"),
                        **r.test.as_dict("test_"),
                        "test_c_within_landmark_lo": r.test_ci[0],
                        "test_c_within_landmark_hi": r.test_ci[1],
                    }
                )
                path = model_dir / f"{family}.joblib"
                joblib.dump(
                    {"model": r.model, "features": features, "label": label, "params": r.params},
                    path,
                )
                mlflow.log_artifact(str(path))
                explain = _explain(r)
                if explain is not None:
                    exp_path = model_dir / f"{family}_explain.csv"
                    explain.rename("value").to_csv(exp_path)
                    mlflow.log_artifact(str(exp_path))
            results.append(r)

        summary = {
            r.family: {
                "params": r.params,
                **r.validation.as_dict("val_"),
                **r.test.as_dict("test_"),
                "test_ci": r.test_ci,
            }
            for r in results
        }
        (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                results_markdown(results, label, {k: len(v) for k, v in splits.items()})
            )
            mlflow.log_artifact(str(report_path))
    return results
