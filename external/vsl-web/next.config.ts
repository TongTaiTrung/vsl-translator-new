import { networkInterfaces } from "node:os";
import type { NextConfig } from "next";

const API_ORIGIN = (process.env.VSL_API_ORIGIN || "http://127.0.0.1:8000").replace(/\/+$/, "");

const lanHosts = Object.values(networkInterfaces())
  .flat()
  .filter((net) => net && net.family === "IPv4" && !net.internal)
  .map((net) => net!.address);

const nextConfig: NextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  allowedDevOrigins: lanHosts,
  async rewrites() {
    return [{ source: "/vsl-api/:path*", destination: `${API_ORIGIN}/:path*` }];
  },
};

export default nextConfig;
