# syntax=docker/dockerfile:1
# The brain: its HTTP API (uvicorn), its CLI (`brain migrate`, ...) and the world CLIs that load
# data through it (`world-seed`). Build from the repo root:
#   docker build -f brain/Dockerfile .
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Third-party dependencies first, so source changes reuse this layer.
COPY pyproject.toml uv.lock ./
COPY contracts/pyproject.toml contracts/
COPY brain/pyproject.toml brain/
COPY realtime/pyproject.toml realtime/
COPY world/pyproject.toml world/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package moe-brain --package moe-world --no-install-workspace
# The whole repo (see .dockerignore): workspace packages install in place and find
# contracts/identity.json and the migrations from their own paths.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package moe-brain --package moe-world \
    && rm -rf board package.json pnpm-lock.yaml pnpm-workspace.yaml

FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /app /app
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
# Caddy serves the brain under /api and strips the prefix; --root-path keeps the docs' links right.
CMD ["uvicorn", "brain.main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
