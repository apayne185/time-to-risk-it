.DEFAULT_GOAL := help
SYN_RAW := data/raw/synthetic
SYN_INTERIM := data/interim/synthetic
SYN_PROCESSED := data/processed/synthetic

.PHONY: help install lint typecheck test check demo ingest clean-data

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

lint:  ## Lint and check formatting
	uv run ruff check . && uv run ruff format --check .

typecheck:  ## Static type-check
	uv run mypy src tests

test:  ## Run tests (real-data tests run only if the licensed files are present)
	uv run pytest

check: lint typecheck test  ## Everything CI runs

demo:  ## Run the pipeline on synthetic data (no data licence needed)
	uv run ttr synth --out $(SYN_RAW)
	uv run ttr ingest --raw-dir $(SYN_RAW) --out-dir $(SYN_INTERIM) --synthetic
	uv run ttr clean --interim-dir $(SYN_INTERIM) --out-dir $(SYN_PROCESSED)

ingest:  ## Ingest the real dataset from data/raw/bwin_rg
	uv run ttr ingest

clean-data: ingest  ## Clean the real dataset into data/processed
	uv run ttr clean
