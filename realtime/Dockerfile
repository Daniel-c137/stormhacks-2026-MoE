# syntax=docker/dockerfile:1
# The LiveKit agent worker (`realtime start`). Build from the repo root:
#   docker build -f realtime/Dockerfile .
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY contracts/pyproject.toml contracts/
COPY brain/pyproject.toml brain/
COPY realtime/pyproject.toml realtime/
COPY world/pyproject.toml world/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package moe-realtime --no-install-workspace
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package moe-realtime \
    && rm -rf board package.json pnpm-lock.yaml pnpm-workspace.yaml

FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --user-group --create-home --shell /usr/sbin/nologin app
COPY --from=build /app /app
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
CMD ["realtime", "start"]
