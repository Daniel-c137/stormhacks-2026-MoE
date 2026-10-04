import path from "node:path";
import { loadEnvConfig } from "@next/env";
import type { NextConfig } from "next";

const repoRoot = path.resolve(process.cwd(), "..");

// One .env at the repo root serves every part.
loadEnvConfig(repoRoot);

const nextConfig: NextConfig = {
  transpilePackages: ["@moe/contracts"],
  // A self-contained server for the Docker image; tracing from the repo root includes contracts.
  output: "standalone",
  outputFileTracingRoot: repoRoot,
};

export default nextConfig;
