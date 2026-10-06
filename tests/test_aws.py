"""AWS integration points, tested offline (moto for S3, cfn-lint for templates)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import boto3
import joblib
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from ttr.serve.app import create_app
from ttr.serve.scoring import fetch_bundle

BUCKET, KEY = "ttr-models-test", "bundles/primary/decision_model.joblib"


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        client = boto3.client("s3")
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
        )
        bundle = {
            "family": "xgb_cox",
            "label": "primary",
            "model": None,
            "features": ["a"],
            "recalibrator": None,
            "horizon_days": 30,
            "policy": {"capacity_share": 0.01, "threshold": 0.01},
        }
        local = tmp_path / "bundle.joblib"
        joblib.dump(bundle, local)
        client.upload_file(str(local), BUCKET, KEY)
        yield


def test_fetch_bundle_downloads_from_s3(s3: None, tmp_path: Path) -> None:
    path = fetch_bundle(f"s3://{BUCKET}/{KEY}", cache_dir=tmp_path / "cache")
    assert path == tmp_path / "cache" / "decision_model.joblib"
    assert joblib.load(path)["family"] == "xgb_cox"


def test_fetch_bundle_passes_local_paths_through(tmp_path: Path) -> None:
    assert fetch_bundle(tmp_path / "x.joblib") == tmp_path / "x.joblib"
    with pytest.raises(ValueError, match="s3://bucket/key"):
        fetch_bundle("s3://only-bucket")


def test_service_loads_bundle_from_s3(s3: None) -> None:
    with TestClient(create_app(f"s3://{BUCKET}/{KEY}")) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/model").json()["family"] == "xgb_cox"
        assert client.get("/ready").json() == {"status": "ready"}


def test_service_stays_up_when_s3_object_is_missing(s3: None) -> None:
    with TestClient(create_app(f"s3://{BUCKET}/missing.joblib")) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert client.get("/model").status_code == 503
        assert client.get("/ready").status_code == 503
