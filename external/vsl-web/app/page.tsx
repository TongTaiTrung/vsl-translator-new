"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_OPTIONS,
  Recognizer,
  SPEED_PRESETS,
  type Phase,
  type RecognizerState,
  type Speed,
} from "@/lib/recognizer";
import { DEFAULT_MODEL, naiveSentence } from "@/lib/sentence";

// ------------------------------------------------------------------ cài đặt (lưu trên trình duyệt)

interface Settings {
  apiUrl: string;
  groqKey: string;
  rememberKey: boolean;
  model: string;
  mirrorPreview: boolean;
  mirrorData: boolean;
  speed: Speed;
  minConsecutive: number;
  sentenceGapSeconds: number;
}

const DEFAULT_SETTINGS: Settings = {
  apiUrl: DEFAULT_OPTIONS.apiUrl,
  groqKey: "",
  rememberKey: false,
  model: DEFAULT_MODEL,
  mirrorPreview: true,
  mirrorData: DEFAULT_OPTIONS.mirrorData,
  speed: "accurate",
  minConsecutive: DEFAULT_OPTIONS.minConsecutive,
  sentenceGapSeconds: DEFAULT_OPTIONS.sentenceGapMs / 1000,
};

const SETTINGS_KEY = "vsl-web:settings";
const GROQ_KEY = "vsl-web:groq-key";

function loadSettings(): Settings {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    const groqKey = localStorage.getItem(GROQ_KEY) || "";
    const speed = saved.speed in SPEED_PRESETS ? saved.speed : DEFAULT_SETTINGS.speed;
    return { ...DEFAULT_SETTINGS, ...saved, speed, groqKey, rememberKey: Boolean(groqKey) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(settings: Settings): void {
  const { groqKey, rememberKey, ...rest } = settings;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(rest));
  if (rememberKey && groqKey) localStorage.setItem(GROQ_KEY, groqKey);
  else localStorage.removeItem(GROQ_KEY);
}

// ------------------------------------------------------------------ câu

type SentenceStatus = "idle" | "pending" | "ok" | "fallback" | "error";

interface Sentence {
  text: string;
  status: SentenceStatus;
  note?: string;
}

interface FinishedSentence extends Sentence {
  id: number;
  words: string[];
  time: string;
}

async function composeSentence(words: string[], settings: Settings): Promise<Sentence> {
  const fallback = naiveSentence(words);
  if (!settings.groqKey.trim()) {
    return { text: fallback, status: "fallback", note: "Chưa có Groq key — câu ghép thô" };
  }
  try {
    const response = await fetch("/api/compose", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-groq-key": settings.groqKey.trim() },
      body: JSON.stringify({ words, model: settings.model }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    return { text: body.sentence, status: body.status === "ok" ? "ok" : "fallback" };
  } catch (err) {
    return { text: fallback, status: "error", note: (err as Error).message };
  }
}

// ------------------------------------------------------------------ hiển thị

const PHASE_TEXT: Record<Phase, string> = {
  stopped: "Chưa bắt đầu",
  starting: "Đang mở camera…",
  buffering: "Đang gom frame…",
  idle: "Đang chờ — đưa tay lên để ký",
  entering: "Đang vào ký hiệu…",
  signing: "Đang nhận diện",
};

const REJECT_TEXT: Record<string, string> = {
  ambiguous: "chưa chắc (2 lớp quá sát)",
  low_confidence: "độ tin cậy thấp",
};

interface Health {
  ok: boolean;
  classes?: string[];
  checkpoint?: string;
  error?: string;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recognizerRef = useRef<Recognizer | null>(null);
  const settingsRef = useRef<Settings>(DEFAULT_SETTINGS);
  const composeVersion = useRef(0);
  const nextId = useRef(1);

  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [loaded, setLoaded] = useState(false);
  const [state, setState] = useState<RecognizerState | null>(null);
  const [sentence, setSentence] = useState<Sentence>({ text: "", status: "idle" });
  const [history, setHistory] = useState<FinishedSentence[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [copied, setCopied] = useState<number | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  settingsRef.current = settings;

  const updateCurrent = useCallback(async (words: string[]) => {
    const version = ++composeVersion.current;
    if (words.length === 0) {
      setSentence({ text: "", status: "idle" });
      return;
    }
    setSentence((prev) => ({ text: prev.text || naiveSentence(words), status: "pending" }));
    const result = await composeSentence(words, settingsRef.current);
    if (version === composeVersion.current) setSentence(result);
  }, []);

  const finishSentence = useCallback(async (words: string[]) => {
    if (words.length === 0) return;
    const id = nextId.current++;
    const time = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setHistory((prev) => [{ id, words, time, text: naiveSentence(words), status: "pending" }, ...prev]);
    const result = await composeSentence(words, settingsRef.current);
    setHistory((prev) => prev.map((item) => (item.id === id ? { ...item, ...result } : item)));
  }, []);

  // Tạo recognizer một lần, gắn với thẻ <video>.
  useEffect(() => {
    if (!videoRef.current) return;
    const recognizer = new Recognizer(videoRef.current, {
      onState: setState,
      onWords: (words) => void updateCurrent(words),
      onSentenceEnd: (words) => void finishSentence(words),
    });
    recognizerRef.current = recognizer;
    return () => recognizer.stop();
  }, [updateCurrent, finishSentence]);

  // Nạp cài đặt đã lưu (chỉ chạy trên trình duyệt).
  useEffect(() => {
    const saved = loadSettings();
    setSettings(saved);
    setSettingsOpen(!saved.groqKey);      // chưa có key: mở sẵn phần cài đặt
    setLoaded(true);
  }, []);

  // Lưu cài đặt và đẩy tham số xuống recognizer.
  useEffect(() => {
    if (!loaded) return;
    saveSettings(settings);
    const { predictFrames, minFrames, strideFrames, minHandFrames } = SPEED_PRESETS[settings.speed];
    recognizerRef.current?.setOptions({
      apiUrl: settings.apiUrl,
      predictFrames,
      minFrames,
      strideFrames,
      minHandFrames,
      mirrorData: settings.mirrorData,
      minConsecutive: settings.minConsecutive,
      sentenceGapMs: settings.sentenceGapSeconds * 1000,
    });
  }, [settings, loaded]);

  // Kiểm tra API dự đoán mỗi khi đổi địa chỉ.
  useEffect(() => {
    if (!loaded) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const url = `${settings.apiUrl.replace(/\/+$/, "")}/api/health`;
        const response = await fetch(url, { signal: controller.signal });
        const body = await response.json();
        setHealth({ ok: response.ok, classes: body.classes, checkpoint: body.checkpoint });
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setHealth({ ok: false, error: "Không kết nối được API dự đoán" });
        }
      }
    }, 400);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [settings.apiUrl, loaded]);

  const update = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    setSettings((prev) => ({ ...prev, [key]: value }));

  const running = state !== null && state.phase !== "stopped";
  const words = state?.words ?? [];
  const prediction = state?.prediction ?? null;

  const toggle = () => {
    const recognizer = recognizerRef.current;
    if (!recognizer) return;
    if (recognizer.running) {
      recognizer.stop();
      void finishSentence(recognizer.finishSentence());
    } else {
      void recognizer.start();
    }
  };

  const endSentence = () => {
    const recognizer = recognizerRef.current;
    if (recognizer) void finishSentence(recognizer.finishSentence());
  };

  const copy = async (id: number, text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 1200);
  };

  const speak = (text: string) => {
    if (!("speechSynthesis" in window) || !text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "vi-VN";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>VSL Translator</h1>
          <p className="muted">Dịch ngôn ngữ ký hiệu Việt Nam từ webcam thành câu hoàn chỉnh</p>
        </div>
        <span className={`pill ${health?.ok ? "ok" : health ? "bad" : ""}`}>
          {health === null
            ? "Đang kiểm tra API…"
            : health.ok
              ? `API sẵn sàng · ${health.classes?.length ?? 0} ký hiệu`
              : health.error ?? "API lỗi"}
        </span>
      </header>

      <div className="layout">
        <section className="stage">
          <div className={`video-wrap phase-${state?.phase ?? "stopped"}`}>
            <video
              ref={videoRef}
              className={settings.mirrorPreview ? "mirror" : undefined}
              playsInline
              muted
            />
            {!running && (
              <div className="video-empty">
                <p>Camera đang tắt</p>
                <p className="muted">Bấm “Bắt đầu nhận diện” để mở camera</p>
              </div>
            )}
            {running && (
              <div className="hud">
                <span className="hud-status">
                  <i className="dot" />
                  {PHASE_TEXT[state!.phase]}
                  {state!.inFlight && <span className="spinner" aria-label="đang gọi API" />}
                </span>
                {state!.resolution && (
                  <span className="hud-meta">
                    {state!.resolution}
                    <span title={SPEED_PRESETS[settings.speed].hint}>· {SPEED_PRESETS[settings.speed].label}</span>
                    {state!.mode && (
                      <span title={state!.mode === "stream" ? "Luồng liên tục: chỉ gửi frame mới" : "Gửi cả cửa sổ 30 frame (server cũ)"}>
                        · {state!.mode === "stream" ? "stream" : "window"}
                      </span>
                    )}
                    {state!.latencyMs != null && <span>· {state!.latencyMs} ms</span>}
                  </span>
                )}
              </div>
            )}
            {running && state!.phase === "buffering" && (
              <div className="buffer-bar">
                <div style={{ width: `${(state!.buffered / state!.windowFrames) * 100}%` }} />
              </div>
            )}
          </div>

          <div className="controls">
            <button className={`primary ${running ? "danger" : ""}`} onClick={toggle} disabled={state?.phase === "starting"}>
              {running ? "Dừng nhận diện" : "Bắt đầu nhận diện"}
            </button>
            <button onClick={endSentence} disabled={words.length === 0}>Kết thúc câu</button>
            <button onClick={() => recognizerRef.current?.undo()} disabled={words.length === 0}>Xoá từ cuối</button>
            <button onClick={() => recognizerRef.current?.clear()} disabled={words.length === 0}>Xoá hết</button>
          </div>

          {state?.error && <p className="error">{state.error}</p>}

          <div className="card prediction">
            <div className="card-head">
              <h2>Dự đoán gần nhất</h2>
              {state?.latencyMs != null && <span className="muted small">{state.latencyMs} ms</span>}
            </div>
            {prediction ? (
              <>
                <p className="pred-label">
                  {prediction.accepted ? (
                    <strong>{prediction.label}</strong>
                  ) : (
                    <span className="muted">Từ chối: {REJECT_TEXT[prediction.rejection_reason ?? ""] ?? "—"}</span>
                  )}
                  {prediction.hand_frames !== undefined && (
                    <span className="muted small"> · tay trong {prediction.hand_frames}/{prediction.frames} frame</span>
                  )}
                </p>
                <ul className="bars">
                  {prediction.top.map((item) => (
                    <li key={item.label}>
                      <span className="bar-label">{item.label}</span>
                      <span className="bar"><span style={{ width: percent(item.confidence) }} /></span>
                      <span className="bar-value">{percent(item.confidence)}</span>
                    </li>
                  ))}
                </ul>
                <p className="muted small">
                  Phiếu: {state?.candidate ? `“${state.candidate}” ${state.count}/${state.required}` : "—"}
                </p>
              </>
            ) : (
              <p className="muted">Chưa có dự đoán.</p>
            )}
          </div>
        </section>

        <section className="side">
          <div className="card current">
            <div className="card-head">
              <h2>Câu đang ký</h2>
              <span className={`status status-${sentence.status}`}>
                {{ idle: "", pending: "đang ghép…", ok: "Groq", fallback: "ghép thô", error: "lỗi Groq" }[sentence.status]}
              </span>
            </div>
            <p className={`sentence ${sentence.text ? "" : "placeholder"}`}>
              {sentence.text || "Các từ nhận diện được sẽ tự ghép thành câu ở đây."}
            </p>
            {sentence.note && <p className="muted small">{sentence.note}</p>}
            <div className="chips">
              {words.map((word, i) => (
                <span key={`${word}-${i}`} className={`chip ${i === words.length - 1 && word === state?.lastCommitted ? "fresh" : ""}`}>
                  {word}
                </span>
              ))}
            </div>
            <p className="muted small">
              {settings.sentenceGapSeconds > 0
                ? `Nghỉ tay ${settings.sentenceGapSeconds} giây để kết thúc câu.`
                : "Bấm “Kết thúc câu” để chốt câu."}
            </p>
          </div>

          <div className="card">
            <div className="card-head">
              <h2>Câu hoàn chỉnh</h2>
              {history.length > 0 && (
                <button className="link" onClick={() => setHistory([])}>Xoá lịch sử</button>
              )}
            </div>
            {history.length === 0 ? (
              <p className="muted">Chưa có câu nào.</p>
            ) : (
              <ol className="history">
                {history.map((item) => (
                  <li key={item.id}>
                    <p className="sentence small-sentence">
                      {item.text}
                      {item.status === "pending" && <span className="muted small"> (đang ghép…)</span>}
                    </p>
                    <p className="muted small">
                      {item.time} · {item.words.join(" | ")}
                      {item.status === "error" && ` · lỗi Groq: ${item.note}`}
                      {item.status === "fallback" && " · ghép thô"}
                    </p>
                    <div className="row-actions">
                      <button className="link" onClick={() => void copy(item.id, item.text)}>
                        {copied === item.id ? "Đã chép" : "Chép"}
                      </button>
                      <button className="link" onClick={() => speak(item.text)}>Đọc</button>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </div>

          <details
            className="card settings"
            open={settingsOpen}
            onToggle={(e) => setSettingsOpen((e.target as HTMLDetailsElement).open)}
          >
            <summary><h2>Cài đặt</h2></summary>
            <label>
              Groq API key
              <input
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder="gsk_…"
                value={settings.groqKey}
                onChange={(e) => update("groqKey", e.target.value)}
              />
            </label>
            <label className="check">
              <input type="checkbox" checked={settings.rememberKey} onChange={(e) => update("rememberKey", e.target.checked)} />
              Ghi nhớ key trên trình duyệt này
            </label>
            <p className="muted small">
              Key chỉ được gửi kèm từng yêu cầu ghép câu tới Groq, không lưu trên server. Không có key thì câu được ghép thô.
            </p>
            <label>
              Model Groq
              <input value={settings.model} spellCheck={false} onChange={(e) => update("model", e.target.value)} />
            </label>
            <label>
              Địa chỉ API dự đoán
              <input value={settings.apiUrl} spellCheck={false} onChange={(e) => update("apiUrl", e.target.value)} />
            </label>
            <fieldset className="speed">
              <legend>Tốc độ nhận diện</legend>
              {(Object.keys(SPEED_PRESETS) as Speed[]).map((key) => (
                <label key={key} className={`speed-option ${settings.speed === key ? "selected" : ""}`}>
                  <input
                    type="radio" name="speed" value={key}
                    checked={settings.speed === key}
                    onChange={() => update("speed", key)}
                  />
                  <span>
                    <strong>{SPEED_PRESETS[key].label}</strong>
                    <span className="muted small">{SPEED_PRESETS[key].hint}</span>
                  </span>
                </label>
              ))}
              <p className="muted small">Nhanh hơn thì kém chính xác hơn. Áp dụng ngay, kể cả khi đang nhận diện; cần server có /api/stream.</p>
            </fieldset>
            <div className="grid2">
              <label>
                Số phiếu để chốt từ
                <input
                  type="number" min={1} max={6}
                  value={settings.minConsecutive}
                  onChange={(e) => update("minConsecutive", Math.max(1, Math.min(6, Number(e.target.value) || 1)))}
                />
              </label>
              <label>
                Nghỉ tay để kết thúc câu (giây, 0 = tắt)
                <input
                  type="number" min={0} max={30} step={0.5}
                  value={settings.sentenceGapSeconds}
                  onChange={(e) => update("sentenceGapSeconds", Math.max(0, Math.min(30, Number(e.target.value) || 0)))}
                />
              </label>
            </div>
            <label className="check">
              <input type="checkbox" checked={settings.mirrorPreview} onChange={(e) => update("mirrorPreview", e.target.checked)} />
              Lật gương hình xem trước
            </label>
            <label className="check">
              <input type="checkbox" checked={settings.mirrorData} onChange={(e) => update("mirrorData", e.target.checked)} />
              Lật dữ liệu gửi đi (mirror_data — phải giống lúc thu dữ liệu)
            </label>
            <button className="link" onClick={() => setSettings({ ...DEFAULT_SETTINGS, groqKey: settings.groqKey, rememberKey: settings.rememberKey })}>
              Khôi phục mặc định
            </button>
          </details>
        </section>
      </div>
    </main>
  );
}
