from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ttr.config import load_data_config, load_landmarks_config
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate
from ttr.evaluate.calibration import PlattRecalibrator
from ttr.features import FEATURES, build_feature_table, features_from_activity
from ttr.labels import get_definition
from ttr.landmarks import build_landmarks
from ttr.models.base import SurvivalData
from ttr.models.xgb import XGBCox
from ttr.serve.app import create_app
from ttr.serve.scoring import Scorer

CFG = load_data_config()


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Synthetic data, a small trained model and a decision bundle on disk."""
    tables = generate(SyntheticSpec(n_pairs=150, seed=8), CFG)
    daily = clean_daily(tables["daily"], CFG)
    players = build_players(tables["demographics"], tables["rg"], daily)
    lm = build_landmarks(players, daily, CFG, load_landmarks_config(), get_definition("primary"))
    table = build_feature_table(lm, daily)
    model = XGBCox(num_boost_round=50).fit(SurvivalData.from_table(table, list(FEATURES)))
    bundle = {
        "family": "xgb_cox",
        "model": model,
        "features": list(FEATURES),
        "recalibrator": PlattRecalibrator(),
        "horizon_days": 30,
        "label": "primary",
        "policy": {"capacity_share": 0.01, "threshold": 0.5},
        "intercept_shift": 0.0,
    }
    path = tmp_path_factory.mktemp("bundle") / "decision_model.joblib"
    joblib.dump(bundle, path)
    return {"raw": tables["daily"], "players": players, "table": table, "bundle": path}


@pytest.fixture(scope="module")
def client(world: dict[str, object]) -> Iterator[TestClient]:
    with TestClient(create_app(world["bundle"])) as c:  # type: ignore[arg-type]
        yield c


def _feature_payload(row: pd.Series) -> dict[str, float | None]:
    return {f: (None if pd.isna(row[f]) else float(row[f])) for f in FEATURES}


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model").json()
    assert info["family"] == "xgb_cox" and info["n_features"] == len(FEATURES)


def test_score_matches_batch_scorer(client: TestClient, world: dict[str, object]) -> None:
    table: pd.DataFrame = world["table"]  # type: ignore[assignment]
    rows = table.head(5)
    body = {
        "players": [
            {"player_id": int(r.user_id), "features": _feature_payload(r)}
            for _, r in rows.iterrows()
        ]
    }
    res = client.post("/score", json=body)
    assert res.status_code == 200
    api = [s["probability_30d"] for s in res.json()["scores"]]
    batch = Scorer.load(world["bundle"]).score(rows.reset_index(drop=True))  # type: ignore[arg-type]
    np.testing.assert_allclose(api, batch["probability"], rtol=1e-6)
    for s in res.json()["scores"]:
        assert len(s["top_drivers"]) <= 3
        assert all(d["contribution"] > 0 and d["description"] for d in s["top_drivers"])


def test_score_rejects_wrong_features(client: TestClient) -> None:
    features: dict[str, float | None] = {f: 0.0 for f in FEATURES}
    features.pop("bet_days_30d")
    features["age"] = 30.0
    res = client.post("/score", json={"players": [{"player_id": 1, "features": features}]})
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail["missing_features"] == ["bet_days_30d"] and detail["unknown_features"] == ["age"]
    assert client.post("/score", json={"players": []}).status_code == 422


def test_activity_endpoint_matches_offline_features(
    client: TestClient, world: dict[str, object]
) -> None:
    """Online feature computation from raw activity equals the offline pipeline's features."""
    table: pd.DataFrame = world["table"]  # type: ignore[assignment]
    raw: pd.DataFrame = world["raw"]  # type: ignore[assignment]
    players: pd.DataFrame = world["players"]  # type: ignore[assignment]
    landmark = table["landmark"].iloc[0]
    rows = table[table["landmark"] == landmark].head(4)
    ids = rows["user_id"].tolist()
    p = players[players["user_id"].isin(ids)][["user_id", "first_deposit_date"]]
    activity = raw[raw["user_id"].isin(ids) & (raw["date"] < landmark)]

    online = features_from_activity(activity, p, landmark, CFG).set_index("user_id")
    offline = rows.set_index("user_id")[list(FEATURES)]
    pd.testing.assert_frame_equal(
        online.loc[offline.index, list(FEATURES)], offline, check_exact=False, rtol=1e-9
    )

    body = {
        "landmark": str(landmark.date()),
        "players": [
            {"player_id": int(uid), "first_deposit_date": str(pd.Timestamp(fd).date())}
            for uid, fd in zip(p["user_id"], p["first_deposit_date"], strict=True)
        ],
        "activity": [
            {
                "player_id": int(r.user_id),
                "date": str(r.date.date()),
                "product_type": int(r.product_type),
                "n_bets": 0.0 if pd.isna(r.n_bets) else float(r.n_bets),
                "turnover": None if pd.isna(r.turnover) else float(r.turnover),
                "hold": None if pd.isna(r.hold) else float(r.hold),
            }
            for r in activity.itertuples()
        ],
    }
    res = client.post("/score/activity", json=body)
    assert res.status_code == 200, res.text
    batch = Scorer.load(world["bundle"]).score(  # type: ignore[arg-type]
        offline.reset_index(drop=True)
    )
    by_id = {s["player_id"]: s["probability_30d"] for s in res.json()["scores"]}
    np.testing.assert_allclose([by_id[i] for i in offline.index], batch["probability"], rtol=1e-6)


def test_activity_endpoint_validates_input(client: TestClient) -> None:
    base = {
        "landmark": "2009-01-01",
        "players": [{"player_id": 1, "first_deposit_date": "2008-01-01"}],
    }
    row = {"player_id": 1, "date": "2008-12-01", "n_bets": 1, "turnover": 5.0, "hold": 1.0}
    bad_product = {**base, "activity": [{**row, "product_type": 11}]}
    assert client.post("/score/activity", json=bad_product).status_code == 422
    stranger = {**base, "activity": [{**row, "player_id": 2, "product_type": 1}]}
    assert client.post("/score/activity", json=stranger).status_code == 422
    negative = {**base, "activity": [{**row, "product_type": 1, "turnover": -1}]}
    assert client.post("/score/activity", json=negative).status_code == 422


def test_missing_bundle_returns_503(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "nope.joblib")) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/model").status_code == 503


def test_score_with_agent_notes(client: TestClient, world: dict[str, object]) -> None:
    table: pd.DataFrame = world["table"]  # type: ignore[assignment]
    row = table.iloc[0]
    body = {"players": [{"player_id": int(row.user_id), "features": _feature_payload(row)}]}
    plain = client.post("/score", json=body).json()["scores"][0]
    assert plain["agent_note"] is None
    noted = client.post("/score?notes=true", json=body).json()["scores"][0]
    note = noted["agent_note"]
    assert note["source"] == "template" and note["violations"] == []
    cited = {o["feature"] for o in note["note"]["observations"]}
    assert cited <= {d["feature"] for d in noted["top_drivers"]}
