from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd
import pytest

from ttr.config import ModelsConfig, load_data_config, load_landmarks_config, load_models_config
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate
from ttr.features import build_feature_table
from ttr.labels import get_definition
from ttr.landmarks import build_landmarks
from ttr.train import grid_points, train_all


@pytest.fixture(scope="module")
def features_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = load_data_config()
    tables = generate(SyntheticSpec(n_pairs=250, seed=21), cfg)
    daily = clean_daily(tables["daily"], cfg)
    players = build_players(tables["demographics"], tables["rg"], daily)
    lm = build_landmarks(players, daily, cfg, load_landmarks_config(), get_definition("primary"))
    path = tmp_path_factory.mktemp("feat") / "features_primary.parquet"
    build_feature_table(lm, daily).to_parquet(path)
    return path


def _cfg(tmp_path: Path) -> ModelsConfig:
    base = load_models_config()
    return base.model_copy(
        update={
            "bootstrap_reps": 20,
            "families": {
                k: v for k, v in base.families.items() if k in ("rule_baseline", "xgb_cox")
            },
            "mlflow": base.mlflow.model_copy(
                update={
                    "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                    "artifact_location": str(tmp_path / "artifacts"),
                }
            ),
        }
    )


def test_grid_points() -> None:
    assert grid_points({}) == [{}]
    assert grid_points({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_train_all_end_to_end(features_path: Path, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    report = tmp_path / "report.md"
    results = train_all(features_path, "primary", cfg, tmp_path / "models", report)

    assert [r.family for r in results] == ["rule_baseline", "xgb_cox"]
    for r in results:
        assert 0.5 < r.test.c_within_landmark <= 1
        assert r.test_ci[0] <= r.test.c_within_landmark <= r.test_ci[1]
    # synthetic cases escalate before their event: the learned model should find it
    assert results[1].validation.c_within_landmark > 0.6

    saved = joblib.load(tmp_path / "models" / "xgb_cox.joblib")
    table = pd.read_parquet(features_path)
    risk = saved["model"].predict_risk(table[saved["features"]])
    assert len(risk) == len(table)
    assert "| xgb_cox |" in report.read_text()

    runs = mlflow.search_runs(experiment_names=[cfg.mlflow.experiment])
    assert {"selection", "final"} <= set(runs["tags.stage"].dropna())
    assert runs["tags.git_sha"].notna().any()
