import path from "node:path";
import { loadEnvConfig } from "@next/env";
import type { NextConfig } from "next";

// One .env at the repo root serves every part.
loadEnvConfig(path.resolve(process.cwd(), ".."));

// With a relative NEXT_PUBLIC_API_URL (e.g. /api, as production serves the brain on the board's
// origin), `next dev` forwards that path to BRAIN_URL itself, so no reverse proxy is needed locally.
const apiPath = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, "");
const brainUrl = process.env.BRAIN_URL?.trim().replace(/\/+$/, "");

const nextConfig: NextConfig = {
  transpilePackages: ["@moe/contracts"],
  async rewrites() {
    if (!apiPath?.startsWith("/") || !brainUrl) return [];
    return [{ source: `${apiPath}/:path*`, destination: `${brainUrl}/:path*` }];
  },
};

export default nextConfig;
