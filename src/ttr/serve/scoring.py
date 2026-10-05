"""Scoring core shared by the HTTP service and batch scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ttr.evaluate.calibration import apply_shift
from ttr.explain import contributions, top_drivers
from ttr.features import FEATURES


def fetch_bundle(location: str | Path, cache_dir: Path | None = None) -> Path:
    """Local path, or ``s3://bucket/key`` downloaded to ``cache_dir`` (needs the aws extra)."""
    loc = str(location)
    if not loc.startswith("s3://"):
        return Path(loc)
    import tempfile

    import boto3

    bucket, _, key = loc.removeprefix("s3://").partition("/")
    if not bucket or not key:
        raise ValueError(f"not an s3://bucket/key URI: {loc}")
    target = (cache_dir or Path(tempfile.gettempdir()) / "ttr-bundles") / Path(key).name
    target.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(target))
    return target


@dataclass
class Scorer:
    """Wraps a decision bundle: model, recalibration, monthly intercept shift and policy."""

    bundle: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Scorer:
        bundle = joblib.load(path)
        missing = {"model", "features", "recalibrator", "horizon_days", "policy"} - set(bundle)
        if missing:
            raise ValueError(f"{path} is not a decision bundle (missing {sorted(missing)})")
        return cls(bundle)

    @property
    def features(self) -> list[str]:
        return list(self.bundle["features"])

    @property
    def threshold(self) -> float:
        return float(self.bundle["policy"]["threshold"])

    @property
    def intercept_shift(self) -> float:
        return float(self.bundle.get("intercept_shift", 0.0))

    def info(self) -> dict[str, Any]:
        b = self.bundle
        return {
            "family": b.get("family"),
            "label": b.get("label"),
            "horizon_days": b["horizon_days"],
            "population_case_rate": b.get("population_case_rate"),
            "policy": b["policy"],
            "intercept_shift": self.intercept_shift,
            "git_sha": b.get("git_sha"),
            "created_at": b.get("created_at"),
            "n_features": len(self.features),
        }

    def score(self, X: pd.DataFrame, explain: int = 3) -> pd.DataFrame:
        """Risk score, calibrated probability, policy flag and top drivers per row."""
        X = X[self.features].astype(float)
        model = self.bundle["model"]
        raw = model.predict_event_prob(X, self.bundle["horizon_days"])
        prob = apply_shift(self.bundle["recalibrator"].transform(raw), self.intercept_shift)
        out = pd.DataFrame(
            {
                "risk_score": model.predict_risk(X),
                "probability": prob,
                "flagged": prob >= self.threshold,
            },
            index=X.index,
        )
        if explain:
            drivers = top_drivers(contributions(model, X), X, k=explain)
            out["top_drivers"] = pd.Series(drivers, index=out.index, dtype=object)
        return out


def describe(feature: str) -> str:
    return FEATURES.get(feature, feature)


def clean_number(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)
