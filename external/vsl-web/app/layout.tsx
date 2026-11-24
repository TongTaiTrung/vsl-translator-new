import type { Metadata, Viewport } from "next";
import { Be_Vietnam_Pro } from "next/font/google";
import { THEME_SCRIPT } from "@/lib/theme";
import "./globals.css";

// Be Vietnam Pro: thiết kế cho tiếng Việt, dấu thanh rõ ở cỡ lớn (phụ đề) lẫn cỡ nhỏ.
const sans = Be_Vietnam_Pro({
  subsets: ["vietnamese", "latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "Phiên dịch ngôn ngữ ký hiệu",
  description: "Phiên dịch ngôn ngữ ký hiệu Việt Nam từ webcam thành câu hoàn chỉnh, theo thời gian thực.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Camera toàn màn hình trên điện thoại phủ cả vùng tai thỏ; nội dung tự né bằng env(safe-area-inset-*).
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#eef2f0" },
    { media: "(prefers-color-scheme: dark)", color: "#0d1512" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // suppressHydrationWarning: script trong <head> đặt data-theme trước khi React hydrate.
  return (
    <html lang="vi" className={sans.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
