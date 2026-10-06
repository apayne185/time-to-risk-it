.DEFAULT_GOAL := help
SYN_RAW := data/raw/synthetic
SYN_INTERIM := data/interim/synthetic
SYN_PROCESSED := data/processed/synthetic

export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help install lint typecheck test check demo ingest clean-data eda landmarks features train evaluate monitor serve docker-build docker-demo stream-demo aws-lint notes-eval mlflow-ui

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies (incl. training extras) and git hooks
	uv sync --all-extras
	uv run pre-commit install

lint:  ## Lint, check formatting and validate CloudFormation
	uv run ruff check . && uv run ruff format --check .
	uv run cfn-lint infra/aws/*.yaml

typecheck:  ## Static type-check
	uv run mypy src tests

test:  ## Run tests (real-data tests run only if the licensed files are present)
	uv run pytest

check: lint typecheck test  ## Everything CI runs

demo:  ## Run the pipeline on synthetic data (no data licence needed)
	uv run ttr synth --out $(SYN_RAW)
	uv run ttr ingest --raw-dir $(SYN_RAW) --out-dir $(SYN_INTERIM) --synthetic
	uv run ttr clean --interim-dir $(SYN_INTERIM) --out-dir $(SYN_PROCESSED)
	uv run ttr eda --processed-dir $(SYN_PROCESSED) --out-dir data/reports/synthetic
	uv run ttr landmarks --processed-dir $(SYN_PROCESSED)
	uv run ttr features --processed-dir $(SYN_PROCESSED)
	uv run ttr train --processed-dir $(SYN_PROCESSED) --family rule_baseline --family xgb_cox \
		--model-dir data/models/synthetic --report data/reports/synthetic/models.md
	uv run ttr evaluate --processed-dir $(SYN_PROCESSED) --model-dir data/models/synthetic \
		--out-dir data/reports/synthetic
	uv run ttr monitor --processed-dir $(SYN_PROCESSED) --bundle data/models/synthetic/decision_model.joblib \
		--report data/reports/synthetic/monitoring.md

ingest:  ## Ingest the real dataset from data/raw/bwin_rg
	uv run ttr ingest

clean-data: ingest  ## Clean the real dataset into data/processed
	uv run ttr clean

eda: clean-data  ## Replicate the paper and write reports/eda.md with figures
	uv run ttr eda

landmarks: clean-data  ## Build landmark tables for the primary and broad labels
	uv run ttr landmarks --label primary
	uv run ttr landmarks --label broad

features: landmarks  ## Build feature tables for the primary and broad labels
	uv run ttr features --label primary
	uv run ttr features --label broad

train: features  ## Select, refit and evaluate all model families (logs to MLflow)
	uv run ttr train --label primary
	uv run ttr train --label broad

evaluate: train  ## Decision layer: calibration, net benefit, capacity, subgroups
	uv run ttr evaluate --label primary

monitor:  ## Feature/score drift and the recommended monthly intercept shift
	uv run ttr monitor

serve:  ## Run the scoring API locally on :8000
	uv run ttr serve

docker-build:  ## Build the scoring-service image
	docker build -t time-to-risk-it:latest .

docker-demo: demo docker-build  ## Serve the synthetic-data model in Docker on :8000
	MODELS_DIR=./data/models MODEL_NAME=synthetic docker compose up api

stream-demo: demo  ## Replay synthetic activity through Redpanda and score it online
	docker compose --profile streaming up -d --wait redpanda
	uv run ttr stream replay --source $(SYN_INTERIM)/daily.parquet
	uv run ttr stream score --players $(SYN_PROCESSED)/players.parquet \
		--bundle data/models/synthetic/decision_model.joblib
	uv run ttr stream read --topic rg.scores

aws-lint:  ## Validate the CloudFormation templates offline
	uv run cfn-lint infra/aws/*.yaml

notes-eval:  ## Agent notes on the top-25 players (WRITER=claude calls the API and costs money)
	uv run ttr notes eval --writer $(or $(WRITER),template)

mlflow-ui:  ## Browse experiment runs
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
