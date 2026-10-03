import path from "node:path";
import { loadEnvConfig } from "@next/env";
import type { NextConfig } from "next";

// One .env at the repo root serves every part.
loadEnvConfig(path.resolve(process.cwd(), ".."));

const nextConfig: NextConfig = {
  transpilePackages: ["@moe/contracts"],
};

export default nextConfig;
