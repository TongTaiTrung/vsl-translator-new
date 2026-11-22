import type { Metadata, Viewport } from "next";
import { Be_Vietnam_Pro } from "next/font/google";
import "./globals.css";

// Be Vietnam Pro: thiết kế cho tiếng Việt, dấu thanh rõ ở cỡ lớn (phụ đề) lẫn cỡ nhỏ.
const sans = Be_Vietnam_Pro({
  subsets: ["vietnamese", "latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "UrSound — Phiên dịch ngôn ngữ ký hiệu",
  description: "Phiên dịch ngôn ngữ ký hiệu Việt Nam từ webcam thành câu hoàn chỉnh, theo thời gian thực.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#eef2f0" },
    { media: "(prefers-color-scheme: dark)", color: "#0d1512" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" className={sans.variable}>
      <body>{children}</body>
    </html>
  );
}
