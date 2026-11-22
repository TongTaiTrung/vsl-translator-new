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
  stopped: "Camera đang tắt",
  starting: "Đang tải MediaPipe, mở camera",
  buffering: "Đang chuẩn bị",
  idle: "Đưa tay lên để ký",
  entering: "Đang vào ký hiệu",
  signing: "Đang nhận diện",
};

const REJECT_TEXT: Record<string, string> = {
  ambiguous: "Hai ký hiệu quá giống nhau",
  low_confidence: "Chưa đủ chắc chắn",
};

const STATUS_TEXT: Record<SentenceStatus, string> = {
  idle: "",
  pending: "Đang ghép câu",
  ok: "Đã ghép bằng Groq",
  fallback: "Ghép thô",
  error: "Groq lỗi, đã ghép thô",
};

interface Health {
  ok: boolean;
  classes?: string[];
  checkpoint?: string;
  error?: string;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

function Icon({ name }: { name: "settings" | "close" | "copy" | "check" | "speaker" | "undo" | "trash" | "stop" | "play" | "flag" }) {
  const paths: Record<string, React.ReactNode> = {
    settings: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
      </>
    ),
    close: <path d="M6 6l12 12M18 6L6 18" />,
    copy: (
      <>
        <rect x="9" y="9" width="11" height="11" rx="2" />
        <path d="M5 15V5a2 2 0 0 1 2-2h8" />
      </>
    ),
    check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
    speaker: (
      <>
        <path d="M11 5L6 9H3v6h3l5 4z" />
        <path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" />
      </>
    ),
    undo: <path d="M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3" />,
    trash: <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />,
    stop: <rect x="6" y="6" width="12" height="12" rx="2.5" />,
    play: <path d="M8 5.5v13l10.5-6.5z" />,
    flag: <path d="M5 21V4M5 4h11l-2 4 2 4H5" />,
  };
  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {paths[name]}
    </svg>
  );
}

/** Thay đổi `value` → render lại con với key mới để chạy hoạt ảnh vào. */
function Swap({ value, className, children }: { value: string; className?: string; children: React.ReactNode }) {
  return <span key={value} className={`swap ${className ?? ""}`}>{children}</span>;
}

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
    return () => recognizer.dispose();
  }, [updateCurrent, finishSentence]);

  // Nạp cài đặt đã lưu (chỉ chạy trên trình duyệt).
  useEffect(() => {
    const saved = loadSettings();
    setSettings(saved);
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

  const phase: Phase = state?.phase ?? "stopped";
  const running = phase !== "stopped";
  const words = state?.words ?? [];
  const prediction = state?.prediction ?? null;
  const showCandidate = running && phase === "signing" && state?.candidate && state.count > 0;
  const bufferProgress = state && state.windowFrames > 0 ? Math.min(1, state.buffered / state.windowFrames) : 0;

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
    setTimeout(() => setCopied(null), 1400);
  };

  const speak = (text: string) => {
    if (!("speechSynthesis" in window) || !text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "vi-VN";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  // Sheet cài đặt: Esc để đóng, đưa focus vào sheet khi mở.
  const sheetRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!settingsOpen) return;
    sheetRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setSettingsOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsOpen]);

  const healthText = health === null
    ? "Đang kết nối"
    : health.ok
      ? `${health.classes?.length ?? 0} ký hiệu sẵn sàng`
      : health.error ?? "API lỗi";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /></span>
          <span className="brand-name">VSL Translator</span>
        </div>
        <div className="topbar-right">
          <span className={`health ${health?.ok ? "ok" : health ? "bad" : "wait"}`} title={health?.checkpoint}>
            <i className="health-dot" />
            {healthText}
          </span>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Mở cài đặt">
            <Icon name="settings" />
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="stage" aria-label="Camera">
          <div className={`frame phase-${phase}`}>
            <video
              ref={videoRef}
              className={settings.mirrorPreview ? "mirror" : undefined}
              playsInline
              muted
            />
            <span className="corner tl" /><span className="corner tr" />
            <span className="corner bl" /><span className="corner br" />

            {!running && (
              <div className="frame-empty">
                <button className="start" onClick={toggle} aria-label="Bắt đầu phiên dịch">
                  <span className="start-ring" aria-hidden="true" />
                  <Icon name="play" />
                </button>
                <p className="frame-empty-title">Bắt đầu phiên dịch</p>
                <p className="frame-empty-note">Camera chỉ dùng để nhận diện, không ghi lại hình ảnh.</p>
              </div>
            )}
            {phase === "starting" && <div className="frame-empty"><p className="frame-empty-title">Đang mở camera…</p></div>}

            {running && phase !== "starting" && (
              <>
                <div className="hud-phase" role="status">
                  <span className="ring" style={{ "--p": phase === "buffering" ? bufferProgress : 1 } as React.CSSProperties}>
                    <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" /></svg>
                    <i />
                  </span>
                  <Swap value={phase}>{PHASE_TEXT[phase]}</Swap>
                </div>
                {state?.latencyMs != null && (
                  <span className="hud-latency" title={`Từ lúc chụp frame tới lúc có kết quả; MediaPipe ${state.delegate ?? ""} trên trình duyệt ${state.detectMs ?? "–"} ms/frame`}>
                    {state.latencyMs} ms
                  </span>
                )}
                {showCandidate && (
                  <div className="hud-candidate" key={state!.candidate}>
                    <span>{state!.candidate}</span>
                    <span className="votes" aria-label={`${state!.count}/${state!.required} phiếu`}>
                      {Array.from({ length: state!.required }, (_, i) => (
                        <i key={i} className={i < state!.count ? "on" : ""} />
                      ))}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="controls">
            {running ? (
              <button className="btn btn-stop" onClick={toggle} key="stop">
                <span className="live-dot" aria-hidden="true" />
                Dừng
              </button>
            ) : (
              <span aria-hidden="true" />
            )}
            <div className="controls-group">
              <button className="btn btn-ghost" onClick={endSentence} disabled={words.length === 0}>
                <Icon name="flag" /> Kết thúc câu
              </button>
              <button className="btn btn-ghost" onClick={() => recognizerRef.current?.undo()} disabled={words.length === 0}>
                <Icon name="undo" /> Xoá từ cuối
              </button>
              <button className="btn btn-ghost" onClick={() => recognizerRef.current?.clear()} disabled={words.length === 0}>
                <Icon name="trash" /> Xoá hết
              </button>
            </div>
          </div>

          {state?.error && <p className="error" role="alert">{state.error}</p>}

          <div className={`insight ${prediction ? "" : "is-empty"}`} aria-live="polite">
            <div className="insight-head">
              <span className="insight-title">Mô hình đang thấy</span>
              {prediction && !prediction.accepted && (
                <Swap value={prediction.rejection_reason ?? ""} className="insight-reject">
                  {REJECT_TEXT[prediction.rejection_reason ?? ""] ?? "Chưa chắc chắn"}
                </Swap>
              )}
            </div>
            <ol className="bars">
              {(prediction?.top ?? [{ label: "—", confidence: 0 }, { label: "—", confidence: 0 }, { label: "—", confidence: 0 }])
                .slice(0, 3)
                .map((item, i) => (
                  <li key={i} className={i === 0 && prediction?.accepted ? "lead" : ""}>
                    <span className="bar-label">{item.label}</span>
                    <span className="bar"><span style={{ transform: `scaleX(${item.confidence})` }} /></span>
                    <span className="bar-value">{prediction ? percent(item.confidence) : ""}</span>
                  </li>
                ))}
            </ol>
          </div>
        </section>

        <section className="caption" aria-label="Phụ đề">
          <div className="caption-head">
            <h2>Đang ký</h2>
            {sentence.status !== "idle" && (
              <Swap value={sentence.status} className={`caption-status status-${sentence.status}`}>
                {STATUS_TEXT[sentence.status]}
              </Swap>
            )}
          </div>

          <p className={`caption-text ${sentence.text ? "" : "placeholder"}`} aria-live="polite">
            <Swap value={sentence.text || "empty"}>
              {sentence.text || "Câu dịch sẽ hiện ở đây khi bạn bắt đầu ký."}
            </Swap>
          </p>
          {sentence.note && <p className="caption-note">{sentence.note}</p>}

          <div className="tokens" aria-label="Các từ đã nhận diện">
            {words.map((word, i) => (
              <span key={`${i}-${word}`} className="token">{word}</span>
            ))}
            {showCandidate && (
              <span
                key={`ghost-${state!.candidate}`}
                className="token ghost"
                style={{ "--fill": state!.count / state!.required } as React.CSSProperties}
              >
                {state!.candidate}
              </span>
            )}
          </div>
          <p className="caption-hint">
            {settings.sentenceGapSeconds > 0
              ? `Hạ tay ${settings.sentenceGapSeconds} giây để kết thúc câu.`
              : "Bấm “Kết thúc câu” để chốt câu."}
          </p>

          <div className="transcript">
            <div className="transcript-head">
              <h2>Hội thoại</h2>
              {history.length > 0 && (
                <button className="link" onClick={() => setHistory([])}>Xoá hội thoại</button>
              )}
            </div>
            {history.length === 0 ? (
              <p className="transcript-empty">Các câu đã hoàn chỉnh sẽ được lưu ở đây để chép hoặc đọc to.</p>
            ) : (
              <ol className="history">
                {history.map((item) => (
                  <li key={item.id}>
                    <p className={`history-text ${item.status === "pending" ? "pending" : ""}`}>
                      <Swap value={item.status === "pending" ? "p" : item.text}>{item.text}</Swap>
                    </p>
                    <div className="history-meta">
                      <time>{item.time}</time>
                      <span className="history-words">{item.words.join(" / ")}</span>
                      {item.status === "error" && <span className="history-warn">Groq lỗi: {item.note}</span>}
                      <span className="history-actions">
                        <button className="icon-button small" onClick={() => void copy(item.id, item.text)}
                          aria-label={copied === item.id ? "Đã chép" : "Chép câu"}>
                          <Swap value={copied === item.id ? "c" : "n"}>
                            <Icon name={copied === item.id ? "check" : "copy"} />
                          </Swap>
                        </button>
                        <button className="icon-button small" onClick={() => speak(item.text)} aria-label="Đọc to">
                          <Icon name="speaker" />
                        </button>
                      </span>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </section>
      </main>

      <div className={`sheet-backdrop ${settingsOpen ? "open" : ""}`} onClick={() => setSettingsOpen(false)} aria-hidden="true" />
      <aside
        className={`sheet ${settingsOpen ? "open" : ""}`}
        aria-label="Cài đặt"
        aria-hidden={!settingsOpen}
        inert={!settingsOpen}
        ref={sheetRef}
        tabIndex={-1}
      >
        <div className="sheet-head">
          <h2>Cài đặt</h2>
          <button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Đóng cài đặt">
            <Icon name="close" />
          </button>
        </div>

        <div className="sheet-body">
          <section className="group">
            <h3>Ghép câu</h3>
            <label className="field">
              <span>Groq API key</span>
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
              <span>Ghi nhớ key trên trình duyệt này</span>
            </label>
            <p className="help">
              Key chỉ đi kèm từng yêu cầu ghép câu tới Groq, server không lưu. Không có key thì các từ được ghép thô.
            </p>
            <label className="field">
              <span>Model Groq</span>
              <input value={settings.model} spellCheck={false} onChange={(e) => update("model", e.target.value)} />
            </label>
          </section>

          <section className="group">
            <h3>Nhận diện</h3>
            <fieldset className="segmented">
              <legend>Tốc độ</legend>
              <div
                className="segmented-track"
                style={{
                  "--i": (Object.keys(SPEED_PRESETS) as Speed[]).indexOf(settings.speed),
                  "--n": Object.keys(SPEED_PRESETS).length,
                } as React.CSSProperties}
              >
                <span className="segmented-thumb" aria-hidden="true" />
                {(Object.keys(SPEED_PRESETS) as Speed[]).map((key) => (
                  <label key={key} className={settings.speed === key ? "selected" : ""}>
                    <input
                      type="radio" name="speed" value={key}
                      checked={settings.speed === key}
                      onChange={() => update("speed", key)}
                    />
                    {SPEED_PRESETS[key].label.replace(/\s*\(.*\)/, "")}
                  </label>
                ))}
              </div>
              <Swap value={settings.speed} className="help">{SPEED_PRESETS[settings.speed].hint}</Swap>
            </fieldset>
            <div className="grid2">
              <label className="field">
                <span>Số phiếu để chốt từ</span>
                <input
                  type="number" min={1} max={6}
                  value={settings.minConsecutive}
                  onChange={(e) => update("minConsecutive", Math.max(1, Math.min(6, Number(e.target.value) || 1)))}
                />
              </label>
              <label className="field">
                <span>Hạ tay để kết thúc câu (giây)</span>
                <input
                  type="number" min={0} max={30} step={0.5}
                  value={settings.sentenceGapSeconds}
                  onChange={(e) => update("sentenceGapSeconds", Math.max(0, Math.min(30, Number(e.target.value) || 0)))}
                />
              </label>
            </div>
            <p className="help">Đặt 0 giây để chỉ kết thúc câu bằng nút.</p>
          </section>

          <section className="group">
            <h3>Camera và kết nối</h3>
            <label className="check">
              <input type="checkbox" checked={settings.mirrorPreview} onChange={(e) => update("mirrorPreview", e.target.checked)} />
              <span>Lật gương hình xem trước</span>
            </label>
            <label className="check">
              <input type="checkbox" checked={settings.mirrorData} onChange={(e) => update("mirrorData", e.target.checked)} />
              <span>Lật ảnh trước khi nhận diện (phải giống lúc thu dữ liệu)</span>
            </label>
            <label className="field">
              <span>Địa chỉ API nhận diện</span>
              <input value={settings.apiUrl} spellCheck={false} onChange={(e) => update("apiUrl", e.target.value)} />
            </label>
            {state?.resolution && (
              <p className="help">
                Camera: {state.resolution}, pad về 1280×1280. MediaPipe chạy trên trình duyệt
                {state.delegate ? ` (${state.delegate}${state.detectMs != null ? `, ${state.detectMs} ms/frame` : ""})` : ""}, chỉ gửi landmark lên API.
              </p>
            )}
          </section>

          <button className="link" onClick={() => setSettings({ ...DEFAULT_SETTINGS, groqKey: settings.groqKey, rememberKey: settings.rememberKey })}>
            Khôi phục mặc định
          </button>
        </div>
      </aside>
    </div>
  );
}
