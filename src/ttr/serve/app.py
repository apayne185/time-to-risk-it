"""HTTP scoring service.

Endpoints:

- ``GET /health``: liveness, and whether a model is loaded.
- ``GET /model``: model metadata, policy threshold and the active intercept shift.
- ``POST /score``: score players from precomputed features (the feature pipeline's output).
- ``POST /score/activity``: compute features from raw daily activity, then score. Uses the same
  cleaning and SQL as the offline pipeline.

The model bundle path comes from ``TTR_MODEL_BUNDLE`` (default
``models/primary/decision_model.joblib``). Scores support a human review and a supportive
contact; they are not for restricting, targeting or marketing to players.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ttr.config import PROJECT_ROOT, load_data_config
from ttr.features import features_from_activity
from ttr.serve.scoring import Scorer, clean_number, describe

log = logging.getLogger(__name__)

DEFAULT_BUNDLE = PROJECT_ROOT / "models" / "primary" / "decision_model.joblib"
MAX_PLAYERS = 10_000


class Driver(BaseModel):
    feature: str
    description: str
    value: float | None
    contribution: float


class PlayerScore(BaseModel):
    player_id: int
    risk_score: float
    probability_30d: Annotated[float, Field(ge=0, le=1)]
    flagged: bool = Field(description="Above the monthly contact-capacity threshold")
    top_drivers: list[Driver]


class ScoreResponse(BaseModel):
    model: dict[str, Any]
    scores: list[PlayerScore]


class PlayerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: int
    features: dict[str, float | None]


class ScoreRequest(BaseModel):
    players: Annotated[list[PlayerFeatures], Field(min_length=1, max_length=MAX_PLAYERS)]


class Player(BaseModel):
    player_id: int
    first_deposit_date: date


class ActivityRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: int
    date: date
    product_type: Annotated[int, Field(ge=1, le=25)]
    n_bets: Annotated[float, Field(ge=0)]
    turnover: Annotated[float, Field(ge=0)] | None = None
    hold: float | None = None


class ActivityRequest(BaseModel):
    landmark: date = Field(description="Scoring date; only activity before it is used")
    players: Annotated[list[Player], Field(min_length=1, max_length=MAX_PLAYERS)]
    activity: list[ActivityRow]


def _response(scorer: Scorer, ids: list[int], X: pd.DataFrame) -> ScoreResponse:
    scored = scorer.score(X)
    scores = [
        PlayerScore(
            player_id=pid,
            risk_score=float(row["risk_score"]),
            probability_30d=float(row["probability"]),
            flagged=bool(row["flagged"]),
            top_drivers=[
                Driver(
                    feature=str(d["feature"]),
                    description=describe(str(d["feature"])),
                    value=clean_number(d["value"]),
                    contribution=float(d["contribution"]),
                )
                for d in row["top_drivers"]
            ],
        )
        for pid, (_, row) in zip(ids, scored.iterrows(), strict=True)
    ]
    return ScoreResponse(model=scorer.info(), scores=scores)


def create_app(bundle_path: Path | None = None) -> FastAPI:
    path = bundle_path or Path(os.environ.get("TTR_MODEL_BUNDLE", DEFAULT_BUNDLE))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.scorer = Scorer.load(path)
            log.info("loaded model bundle %s", path)
        except FileNotFoundError:
            app.state.scorer = None
            log.error("no model bundle at %s; scoring endpoints return 503", path)
        yield

    app = FastAPI(
        title="time-to-risk-it",
        version="0.1.0",
        lifespan=lifespan,
        description="30-day responsible-gambling risk scores with explanations.",
    )

    def scorer(request: Request) -> Scorer:
        s: Scorer | None = request.app.state.scorer
        if s is None:
            raise HTTPException(503, "model bundle not loaded")
        return s

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "model_loaded": request.app.state.scorer is not None}

    @app.get("/model")
    def model_info(request: Request) -> dict[str, Any]:
        return scorer(request).info()

    @app.post("/score")
    def score(request: Request, body: ScoreRequest) -> ScoreResponse:
        s = scorer(request)
        expected = set(s.features)
        for p in body.players:
            missing, unknown = expected - set(p.features), set(p.features) - expected
            if missing or unknown:
                raise HTTPException(
                    422,
                    {
                        "player_id": p.player_id,
                        "missing_features": sorted(missing),
                        "unknown_features": sorted(unknown),
                    },
                )
        X = pd.DataFrame(
            [
                {k: (float("nan") if v is None else v) for k, v in p.features.items()}
                for p in body.players
            ]
        )
        return _response(s, [p.player_id for p in body.players], X)

    @app.post("/score/activity")
    def score_activity(request: Request, body: ActivityRequest) -> ScoreResponse:
        s = scorer(request)
        players = pd.DataFrame(
            {
                "user_id": [p.player_id for p in body.players],
                "first_deposit_date": pd.to_datetime([p.first_deposit_date for p in body.players]),
            }
        )
        if players["user_id"].duplicated().any():
            raise HTTPException(422, "duplicate player_id")
        activity = pd.DataFrame(
            [
                {
                    "user_id": a.player_id,
                    "date": pd.Timestamp(a.date),
                    "product_type": a.product_type,
                    "turnover": a.turnover,
                    "hold": a.hold,
                    "n_bets": a.n_bets,
                }
                for a in body.activity
            ],
            columns=["user_id", "date", "product_type", "turnover", "hold", "n_bets"],
        ).astype({"turnover": float, "hold": float, "n_bets": float})
        known_products = set(load_data_config().products.family_of())
        bad = set(activity["product_type"]) - known_products
        if bad:
            raise HTTPException(422, f"unknown product_type codes: {sorted(bad)}")
        unknown = set(activity["user_id"]) - set(players["user_id"])
        if unknown:
            raise HTTPException(422, f"activity for unknown players: {sorted(unknown)[:10]}")
        feats = features_from_activity(
            activity, players, pd.Timestamp(body.landmark), load_data_config()
        )
        feats = feats.set_index("user_id").reindex(players["user_id"])
        return _response(s, players["user_id"].tolist(), feats.reset_index(drop=True))

    return app


app = create_app()
