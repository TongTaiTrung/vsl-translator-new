// Chế độ sáng/tối: mặc định theo prefers-color-scheme của hệ thống; người dùng chọn thì lưu localStorage và đặt
// data-theme="light" | "dark" trên <html> (globals.css đọc thuộc tính này). Chọn trùng với hệ thống thì xoá lựa
// chọn → lại theo hệ thống (kể cả khi hệ thống đổi sau đó).

export type Theme = "light" | "dark";

export const THEME_KEY = "vsl-web:theme";

/** Chạy đồng bộ trong <head> trước khi vẽ (layout.tsx) để không nháy màu khi tải trang. */
export const THEME_SCRIPT =
  `(function(){try{var t=localStorage.getItem(${JSON.stringify(THEME_KEY)});` +
  `if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}})()`;

const DARK_QUERY = "(prefers-color-scheme: dark)";

export function systemTheme(): Theme {
  return window.matchMedia(DARK_QUERY).matches ? "dark" : "light";
}

export function storedTheme(): Theme | null {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;
  }
}

/** Giao diện đang hiển thị: lựa chọn đã lưu, không có thì theo hệ thống. */
export function currentTheme(): Theme {
  const attr = document.documentElement.getAttribute("data-theme");
  return attr === "light" || attr === "dark" ? attr : systemTheme();
}

/** Áp dụng lựa chọn đã lưu (hoặc bỏ thuộc tính để theo hệ thống). */
export function applyStoredTheme(): void {
  const t = storedTheme();
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
}

export function setTheme(theme: Theme): void {
  const followSystem = theme === systemTheme();
  try {
    if (followSystem) localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, theme);
  } catch {
    // localStorage bị chặn (chế độ riêng tư…): vẫn đổi giao diện cho phiên này.
  }
  if (followSystem) document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
}

/** Gọi `onChange` khi hệ thống đổi sáng/tối; trả về hàm huỷ. */
export function watchSystemTheme(onChange: () => void): () => void {
  const media = window.matchMedia(DARK_QUERY);
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}
