import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/basker-energy-website',
  assetPrefix: '/basker-energy-website/',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

export default nextConfig;
