# syntax=docker/dockerfile:1
# The demo world: one image for the mock GitHub and Jira MCP servers (`world-github-mcp`,
# `world-jira-mcp`) and the world CLIs. Build from the repo root:
#   docker build -f world/Dockerfile .
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
    uv sync --frozen --no-dev --package moe-world --no-install-workspace
# The world reads world/snapshots, world/world.json and mock-data from the repo layout.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package moe-world \
    && rm -rf board package.json pnpm-lock.yaml pnpm-workspace.yaml

FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/overlay && chown app:app /data/overlay
COPY --from=build /app /app
WORKDIR /app
# The write journal lives on a volume mounted here; the servers listen beyond localhost so their
# neighbours on the compose network can reach them (they are never published).
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 \
    WORLD_OVERLAY_DIR=/data/overlay WORLD_MCP_HOST=0.0.0.0
USER app
EXPOSE 8101 8102
CMD ["world-github-mcp"]
