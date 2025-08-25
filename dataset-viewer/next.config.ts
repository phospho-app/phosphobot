import type { NextConfig } from "next";
import packageJson from './package.json';

const nextConfig: NextConfig = {

  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  generateBuildId: () => packageJson.version,

  // Expose environment variables to server components
  env: {
    HF_TOKEN: process.env.HF_TOKEN,
    DATASET_URL: process.env.DATASET_URL,
  },
};

export default nextConfig;
