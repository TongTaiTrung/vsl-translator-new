/** Run the dev server over HTTPS on the LAN with a self-signed certificate. */

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
