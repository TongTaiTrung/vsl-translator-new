// Chạy `next dev` bằng HTTPS trên mọi card mạng để điện thoại/máy khác cùng mạng LAN mở được web app.
// Trình duyệt chỉ cho dùng camera trên trang HTTPS (hoặc localhost), nên cần chứng chỉ: script tự tạo chứng chỉ
// tự ký bằng openssl cho localhost + các IP LAN hiện tại vào certificates/ (tạo lại khi IP đổi).
//
//   npm run dev:lan            # cổng 3000, thay cho `npm run dev` (Next chỉ cho một dev server mỗi thư mục)
//   PORT=3443 npm run dev:lan
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { networkInterfaces } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const certDir = join(root, "certificates");
const keyPath = join(certDir, "lan-key.pem");
const certPath = join(certDir, "lan-cert.pem");
const sanPath = join(certDir, "lan-san.txt");
const port = process.env.PORT || "3000";

const ips = Object.values(networkInterfaces())
  .flat()
  .filter((net) => net && net.family === "IPv4" && !net.internal)
  .map((net) => net.address);
const san = ["DNS:localhost", "IP:127.0.0.1", ...ips.map((ip) => `IP:${ip}`)].join(",");

const current = existsSync(sanPath) ? readFileSync(sanPath, "utf8") : "";
if (current !== san || !existsSync(keyPath) || !existsSync(certPath)) {
  mkdirSync(certDir, { recursive: true });
  // iOS từ chối chứng chỉ TLS có hạn > 825 ngày.
  execFileSync("openssl", [
    "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "825",
    "-keyout", keyPath, "-out", certPath, "-subj", "/CN=vsl-web LAN dev",
    "-addext", `subjectAltName=${san}`, "-addext", "extendedKeyUsage=serverAuth",
  ], { stdio: "ignore" });
  writeFileSync(sanPath, san);
  console.log(`Đã tạo chứng chỉ tự ký cho: ${san}`);
}

console.log("\nMở trên điện thoại / máy khác cùng mạng:");
for (const ip of ips) console.log(`  https://${ip}:${port}`);
console.log("Chứng chỉ tự ký: trình duyệt sẽ cảnh báo, chọn \"Nâng cao → Tiếp tục\" (Safari: \"Truy cập trang web này\").\n");

const next = spawn(
  process.execPath,
  [
    join(root, "node_modules/next/dist/bin/next"), "dev",
    "--hostname", "0.0.0.0", "--port", port,
    "--experimental-https", "--experimental-https-key", keyPath, "--experimental-https-cert", certPath,
  ],
  { cwd: root, stdio: "inherit" },
);
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => next.kill(signal));
next.on("exit", (code) => process.exit(code ?? 0));
