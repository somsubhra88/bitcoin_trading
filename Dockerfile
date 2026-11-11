# Multi-stage build: dependencies are installed from uv.lock in the builder and
# the runtime image contains only the installed application and runtime files.
FROM python:3.12.8-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /app

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "uv==0.11.30"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN /opt/venv/bin/uv sync --frozen --no-dev --extra dashboard --no-editable \
    && /opt/venv/bin/pip uninstall --yes uv


FROM python:3.12.8-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# curl is the health-check probe; tzdata keeps zoneinfo behaviour consistent on
# minimal Linux hosts; ca-certificates is required for HTTPS exchange requests.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 btc

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=btc:btc alembic.ini ./alembic.ini
COPY --chown=btc:btc migrations ./migrations
COPY --chown=btc:btc dashboard ./dashboard

# Non-root: nothing in this image needs to write outside the venv or /tmp.
USER btc

EXPOSE 8000 8501

# Overridden per service in docker-compose.yml.
CMD ["uvicorn", "btc_trader.main:app", "--host", "0.0.0.0", "--port", "8000"]
