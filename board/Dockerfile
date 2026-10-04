# syntax=docker/dockerfile:1
# The board (Next.js standalone server). Build from the repo root:
#   docker build -f board/Dockerfile --build-arg NEXT_PUBLIC_API_URL=/api .
# NEXT_PUBLIC_* values are baked into the browser bundle at build time.
ARG NODE_IMAGE=node:24.21.0-bookworm-slim

FROM ${NODE_IMAGE} AS build
ENV PNPM_HOME=/pnpm NEXT_TELEMETRY_DISABLED=1
ENV PATH=$PNPM_HOME:$PATH
RUN npm install --global --no-fund --no-audit pnpm@12.3.4
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY board/package.json board/
COPY contracts/package.json contracts/
RUN --mount=type=cache,target=/pnpm/store \
    pnpm install --frozen-lockfile --store-dir /pnpm/store --filter board...
COPY contracts contracts
COPY board board
ARG NEXT_PUBLIC_API_URL=/api
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN pnpm --filter board build

FROM ${NODE_IMAGE}
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 HOSTNAME=0.0.0.0 PORT=3000
WORKDIR /app
# outputFileTracingRoot is the repo root, so the server sits at board/server.js.
COPY --from=build /app/board/.next/standalone ./
COPY --from=build /app/board/.next/static ./board/.next/static
USER node
EXPOSE 3000
CMD ["node", "board/server.js"]
