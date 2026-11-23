import { networkInterfaces } from "node:os";
import type { NextConfig } from "next";

// Flask API (python -m src.app.server) chỉ cần nghe ở 127.0.0.1: trình duyệt gọi /vsl-api/... cùng origin với
// trang, Next chuyển tiếp tới đây. Nhờ vậy điện thoại trong mạng LAN không phải gọi thẳng API (không lộ cổng
// 8000, không dính mixed content khi trang chạy HTTPS).
const API_ORIGIN = (process.env.VSL_API_ORIGIN || "http://127.0.0.1:8000").replace(/\/+$/, "");

// IPv4 LAN của máy: dev server chặn tài nguyên /_next/* khi trang được mở từ host khác localhost.
const lanHosts = Object.values(networkInterfaces())
  .flat()
  .filter((net) => net && net.family === "IPv4" && !net.internal)
  .map((net) => net!.address);

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Nút "N" của dev server nằm góc dưới, đè lên nút Dừng của camera toàn màn hình trên điện thoại (npm run dev:lan).
  devIndicators: false,
  allowedDevOrigins: lanHosts,
  async rewrites() {
    return [{ source: "/vsl-api/:path*", destination: `${API_ORIGIN}/:path*` }];
  },
};

export default nextConfig;
