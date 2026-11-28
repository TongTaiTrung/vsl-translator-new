/** Light/dark theme toggle. */

"use client";

import { useLayoutEffect, useState } from "react";
import { type Theme, applyStoredTheme, currentTheme, setTheme, watchSystemTheme } from "@/lib/theme";

export function ThemeToggle() {
  const [theme, setThemeState] = useState<Theme | null>(null);

  useLayoutEffect(() => {
    applyStoredTheme();
    setThemeState(currentTheme());
    return watchSystemTheme(() => setThemeState(currentTheme()));
  }, []);

  const toggle = () => {
    const next: Theme = currentTheme() === "dark" ? "light" : "dark";
    setTheme(next);
    setThemeState(next);
  };

  const label = theme === null
    ? "Đổi giao diện sáng/tối"
    : theme === "dark" ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối";

  return (
    <button className="icon-button theme-toggle" onClick={toggle} aria-label={label} title={label}>
      <svg className="icon theme-sun" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor"
        strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4" />
      </svg>
      <svg className="icon theme-moon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor"
        strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />
      </svg>
    </button>
  );
}
