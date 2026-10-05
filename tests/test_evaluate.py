from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ttr.config import (
    load_data_config,
    load_decision_config,
    load_landmarks_config,
    load_models_config,
)
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate
from ttr.evaluate.calibration import apply_shift, intercept_shift
from ttr.evaluate.report import write_report
from ttr.evaluate.run import horizon_outcome, run_evaluation, save_decision_bundle
from ttr.explain import contributions, top_drivers
from ttr.features import build_feature_table
from ttr.labels import get_definition
from ttr.landmarks import build_landmarks
from ttr.train import train_all

FAMILIES = ("rule_baseline", "cox", "xgb_cox")


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    root = tmp_path_factory.mktemp("eval")
    cfg = load_data_config()
    tables = generate(SyntheticSpec(n_pairs=300, seed=33), cfg)
    daily = clean_daily(tables["daily"], cfg)
    players = build_players(tables["demographics"], tables["rg"], daily)
    lm = build_landmarks(players, daily, cfg, load_landmarks_config(), get_definition("primary"))
    table = build_feature_table(lm, daily)
    feat_path = root / "features_primary.parquet"
    table.to_parquet(feat_path)

    mcfg = load_models_config()
    mcfg = mcfg.model_copy(
        update={
            "bootstrap_reps": 10,
            "families": {
                k: v.model_copy(update={"grid": {}})
                for k, v in mcfg.families.items()
                if k in FAMILIES
            },
            "mlflow": mcfg.mlflow.model_copy(
                update={
                    "tracking_uri": f"sqlite:///{root / 'mlflow.db'}",
                    "artifact_location": str(root / "artifacts"),
                }
            ),
        }
    )
    train_all(feat_path, "primary", mcfg, root / "models" / "primary")
    return table, players, root


def test_evaluation_end_to_end(trained: tuple[pd.DataFrame, pd.DataFrame, Path]) -> None:
    table, players, root = trained
    dcfg = load_decision_config().model_copy(update={"models": FAMILIES})
    result = run_evaluation(table, players, root / "models" / "primary", dcfg)

    assert result.decision_family in ("cox", "xgb_cox")
    assert set(result.evaluations) == set(FAMILIES)
    for ev in result.evaluations.values():
        assert 0.5 < ev.test_auc <= 1
        assert ev.capacity["recall"].is_monotonic_increasing
        assert np.all((ev.test_prob > 0) & (ev.test_prob < 1))
    assert result.sensitivity["population_case_rate"].tolist() == list(
        dcfg.population_case_rate_grid
    )
    # contacts per reached case fall as the assumed population case rate rises
    assert result.sensitivity["contacts_per_case"].is_monotonic_decreasing
    assert len(result.rolling_shifts) == 3

    bundle_path = save_decision_bundle(result, root / "models" / "primary", dcfg, "primary")
    bundle = joblib.load(bundle_path)
    assert 0 < bundle["policy"]["threshold"] < 1
    report = write_report(result, dcfg, root / "reports", root / "models" / "primary", "primary")
    text = report.read_text()
    assert "## Headline" in text and "{" not in text.split("## Headline")[1][:400]
    assert {p.name for p in (root / "reports" / "figures").iterdir()} == {
        "capacity.png",
        "decision_curve.png",
        "calibration.png",
        "drivers.png",
    }


@pytest.mark.parametrize("family", ["cox", "xgb_cox"])
def test_contributions_explain_the_risk_score(
    trained: tuple[pd.DataFrame, pd.DataFrame, Path], family: str
) -> None:
    table, _, root = trained
    saved = joblib.load(root / "models" / "primary" / f"{family}.joblib")
    model, X = saved["model"], table[saved["features"]].head(200)
    contrib = contributions(model, X)
    assert list(contrib.columns) == saved["features"]
    # contributions differ from the risk score only by a constant
    gap = model.predict_risk(X) - contrib.sum(axis=1).to_numpy()
    assert float(np.ptp(gap)) < 1e-3
    drivers = top_drivers(contrib, X, k=3)
    assert all(len(d) <= 3 for d in drivers)
    assert all(float(str(x["contribution"])) > 0 for d in drivers for x in d)


def test_horizon_outcome_rejects_early_censoring() -> None:
    table = pd.DataFrame({"event": [0, 1], "time": [10, 5]})
    with pytest.raises(ValueError, match="censored before"):
        horizon_outcome(table, 30)
    assert horizon_outcome(
        pd.DataFrame({"event": [0, 1, 1], "time": [30, 5, 40]}), 30
    ).tolist() == [0, 1, 0]


def test_intercept_shift_matches_observed_rate() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.02, 5000)
    y = (rng.random(5000) < 3 * p).astype(int)
    w = np.ones(5000)
    shifted = apply_shift(p, intercept_shift(p, y, w))
    assert shifted.mean() == pytest.approx(y.mean(), rel=1e-3)
    assert np.array_equal(np.argsort(shifted), np.argsort(p))
