import path from "node:path";
import { loadEnvConfig } from "@next/env";
import type { NextConfig } from "next";

const repoRoot = path.resolve(process.cwd(), "..");

// One .env at the repo root serves every part. Next has already loaded (and cached) the board
// folder's env files by the time this runs, so the root .env needs a forced reload to be read.
loadEnvConfig(repoRoot, process.env.NODE_ENV !== "production", undefined, true);

// With a relative NEXT_PUBLIC_API_URL (e.g. /api, as production serves the brain on the board's
// origin), `next dev` forwards that path to BRAIN_URL itself, so no reverse proxy is needed locally.
// The Docker image is built without BRAIN_URL, so there Caddy alone routes /api.
const apiPath = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, "");
const brainUrl = process.env.BRAIN_URL?.trim().replace(/\/+$/, "");

const nextConfig: NextConfig = {
  transpilePackages: ["@moe/contracts"],
  // A self-contained server for the Docker image; tracing from the repo root includes contracts.
  output: "standalone",
  outputFileTracingRoot: repoRoot,
  // The tasks page used to live at /memory, next to the decisions it no longer shows.
  async redirects() {
    return [{ source: "/memory", destination: "/tasks", permanent: true }];
  },
  async rewrites() {
    if (!apiPath?.startsWith("/") || !brainUrl) return [];
    return [{ source: `${apiPath}/:path*`, destination: `${brainUrl}/:path*` }];
  },
};

export default nextConfig;
