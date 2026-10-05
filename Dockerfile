# Scoring service image. The model bundle is mounted (or fetched from S3) at runtime, never baked
# in: it is trained on licensed data. Includes the aws extra (boto3) for s3:// bundles.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so code changes do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra aws --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra aws --no-editable

FROM python:3.12-slim
# libgomp: OpenMP runtime for XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY configs ./configs
COPY sql ./sql
ENV PATH=/app/.venv/bin:$PATH \
    TTR_PROJECT_ROOT=/app \
    TTR_MODEL_BUNDLE=/app/models/primary/decision_model.joblib \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "ttr.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
