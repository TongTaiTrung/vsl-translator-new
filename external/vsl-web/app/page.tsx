"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_OPTIONS,
  type CommittedWindow,
  Recognizer,
  SPEED_PRESETS,
  TURBO_LEVELS,
  TURBO_PRESETS,
  type Phase,
  type RecognizerState,
  type Speed,
  type TurboLevel,
} from "@/lib/recognizer";
import {
  addSample,
  clearSamples,
  deleteLabel,
  deleteSample,
  listSamples,
  memoryPayload,
  newSampleId,
  type Sample,
  type SampleSource,
  submitSample,
} from "@/lib/memory";
import type { FrameLandmarks } from "@/lib/landmarks";
import { DEFAULT_MODEL, naiveSentence } from "@/lib/sentence";
import { PersonalizeSheet } from "./personalize";
import { Slider } from "./slider";
import { Switch } from "./switch";
import { ThemeToggle } from "./theme-toggle";

// ------------------------------------------------------------------ cài đặt (lưu trên trình duyệt)

interface Settings {
  apiUrl: string;
  groqKey: string;
  rememberKey: boolean;
  model: string;
  mirrorPreview: boolean;
  mirrorData: boolean;
  debugCrop: boolean;
  speed: Speed;
  /** Siêu tốc (TURBO_PRESETS): bật thì thay cho `speed`. */
  turbo: boolean;
  turboLevel: TurboLevel;
  minConsecutive: number;
  sentenceGapSeconds: number;
  /** Dùng mẫu cá nhân khi nhận diện. */
  personalize: boolean;
  /** Mức ảnh hưởng của mẫu cá nhân (0–1). */
  memoryStrength: number;
  /** Tự lưu mẫu từ các từ người dùng giữ lại (không xoá/undo). */
  autoLearn: boolean;
  /** "realtime": học mỗi từ ngay sau khi chốt (hoàn tác thì xoá mẫu); "sentence": học khi kết thúc câu. */
  learnMode: LearnMode;
}

type LearnMode = "realtime" | "sentence";

const DEFAULT_SETTINGS: Settings = {
  apiUrl: DEFAULT_OPTIONS.apiUrl,
  groqKey: "",
  rememberKey: false,
  model: DEFAULT_MODEL,
  mirrorPreview: true,
  mirrorData: DEFAULT_OPTIONS.mirrorData,
  debugCrop: false,
  speed: "accurate",
  turbo: false,
  turboLevel: 4,
  minConsecutive: DEFAULT_OPTIONS.minConsecutive,
  sentenceGapSeconds: DEFAULT_OPTIONS.sentenceGapMs / 1000,
  personalize: true,
  memoryStrength: 0.5,
  autoLearn: true,
  learnMode: "realtime",
};

/** Học thời gian thực: chờ chừng này ms sau khi chốt từ (hoàn tác trong lúc này thì không học). */
const REALTIME_GRACE_MS = 2500;

/** Tự học: chỉ lấy từ mô hình đã khá chắc (giảm rủi ro học nhầm) … */
const AUTO_MIN_CONFIDENCE = 0.8;
/** … và bỏ qua cửa sổ đã gần như trùng một mẫu sẵn có của chính từ đó (không thêm thông tin). */
const AUTO_SKIP_SIMILARITY = 0.97;

/** Ghi mẫu: đếm ngược bao nhiêu giây, thu bao nhiêu frame (30 frame = 2 giây ở 15 fps, đúng độ dài mẫu train). */
const COUNTDOWN_SECONDS = 3;
const SAMPLE_FRAMES = 30;

interface Recording {
  label: string;
  stage: "countdown" | "capture" | "saving";
  count: number;
}

interface Notice {
  text: string;
  tone: "ok" | "warn";
}

const SETTINGS_KEY = "vsl-web:settings";
const GROQ_KEY = "vsl-web:groq-key";

function loadSettings(): Settings {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    const groqKey = localStorage.getItem(GROQ_KEY) || "";
    const speed = saved.speed in SPEED_PRESETS ? saved.speed : DEFAULT_SETTINGS.speed;
    const learnMode = saved.learnMode === "sentence" ? "sentence" : DEFAULT_SETTINGS.learnMode;
    const turbo = saved.turbo === true;
    const turboLevel = TURBO_LEVELS.includes(saved.turboLevel) ? saved.turboLevel : DEFAULT_SETTINGS.turboLevel;
    return {
      ...DEFAULT_SETTINGS, ...saved, speed, turbo, turboLevel, learnMode, groqKey, rememberKey: Boolean(groqKey),
    };
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
    return { text: fallback, status: "fallback", note: "Chưa có Groq key: ghép thô" };
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
  starting: "Đang tải MediaPipe…",
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
  featureDim?: number;
  error?: string;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

function Icon({ name }: { name: "settings" | "close" | "copy" | "check" | "speaker" | "undo" | "trash" | "stop" | "play" | "flag" | "user" | "flip" }) {
  const paths: Record<string, React.ReactNode> = {
    // Máy ảnh với hai mũi tên xoay: đổi camera trước / sau.
    flip: (
      <>
        <path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2l1.5-2h6l1.5 2h2A1.5 1.5 0 0 1 20 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z" />
        <path d="M9 12.5a3 3 0 0 1 5.2-2M15 13a3 3 0 0 1-5.2 2" />
        <path d="M14.4 8.6l-.2 1.9-1.9-.3M9.6 17.4l.2-1.9 1.9.3" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21a8 8 0 0 1 16 0" />
      </>
    ),
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

/** Tách chữ theo grapheme để dấu tiếng Việt không bị tách khỏi chữ cái. */
function graphemes(text: string): string[] {
  if (typeof Intl !== "undefined" && "Segmenter" in Intl) {
    return Array.from(new Intl.Segmenter("vi", { granularity: "grapheme" }).segment(text), (s) => s.segment);
  }
  return Array.from(text.normalize("NFC"));
}

/** Thay đổi `value` → render lại con với key mới để chạy hoạt ảnh vào. */
function Swap({ value, className, children }: { value: string; className?: string; children: React.ReactNode }) {
  return <span key={value} className={`swap ${className ?? ""}`}>{children}</span>;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
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
  const [personalizeOpen, setPersonalizeOpen] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  // Thiết bị cảm ứng (điện thoại / máy tính bảng): bố cục camera toàn màn hình khi đang chạy.
  const [touch, setTouch] = useState(false);
  useEffect(() => {
    const query = matchMedia("(pointer: coarse)");
    const sync = () => setTouch(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  settingsRef.current = settings;
  /** Cửa sổ đã được tự học → id mẫu, để xoá mẫu đó nếu người dùng sửa nhãn cửa sổ này sau. */
  const autoSaved = useRef(new WeakMap<CommittedWindow, string>());
  /** Cửa sổ người dùng đã hoàn tác/xoá: không học nữa. */
  const discarded = useRef(new WeakSet<CommittedWindow>());
  /** Lượt học thời gian thực đang chờ hết REALTIME_GRACE_MS. */
  const pendingLearn = useRef(new Map<CommittedWindow, ReturnType<typeof setTimeout>>());
  const learnRef = useRef<{
    committed: (w: CommittedWindow) => void;
    sentence: (windows: CommittedWindow[]) => void;
    discarded: (windows: CommittedWindow[]) => void;
  }>({ committed: () => undefined, sentence: () => undefined, discarded: () => undefined });
  const healthRef = useRef<Health | null>(null);
  healthRef.current = health;

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
      onSentenceWindows: (windows) => learnRef.current.sentence(windows),
      onWordCommitted: (w) => learnRef.current.committed(w),
      onWordsDiscarded: (windows) => learnRef.current.discarded(windows),
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
    const preset = settings.turbo ? TURBO_PRESETS[settings.turboLevel] : SPEED_PRESETS[settings.speed];
    const { predictFrames, minFrames, strideFrames, minHandFrames } = preset;
    recognizerRef.current?.setOptions({
      apiUrl: settings.apiUrl,
      predictFrames,
      minFrames,
      strideFrames,
      minHandFrames,
      idleHandFrames: preset.idleHandFrames ?? DEFAULT_OPTIONS.idleHandFrames,
      fill: preset.fill ?? DEFAULT_OPTIONS.fill,
      mirrorData: settings.mirrorData,
      minConsecutive: settings.minConsecutive,
      sentenceGapMs: settings.sentenceGapSeconds * 1000,
    });
    recognizerRef.current?.setDebugCanvas(settings.debugCrop ? overlayRef.current : null);
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
        setHealth({ ok: response.ok, classes: body.classes, checkpoint: body.checkpoint, featureDim: body.feature_dim });
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setHealth({ ok: false, error: "Không kết nối được API" });
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

  // ---------------------------------------------------------------- cá nhân hoá (lib/memory.ts)

  const reloadSamples = useCallback(async () => {
    try {
      setSamples(await listSamples());
    } catch (err) {
      setNotice({ text: `Không đọc được mẫu cá nhân: ${(err as Error).message}`, tone: "warn" });
    }
  }, []);

  useEffect(() => {
    void reloadSamples();
  }, [reloadSamples]);

  // Đẩy bộ mẫu xuống recognizer (mở lại phiên server); chờ một nhịp để kéo thanh mức ảnh hưởng không mở phiên liên tục.
  const featureDim = health?.featureDim;
  useEffect(() => {
    if (!loaded) return;
    const timer = setTimeout(() => {
      const payload = settings.personalize ? memoryPayload(samples, settings.memoryStrength, featureDim) : null;
      recognizerRef.current?.setMemory(payload);
    }, 300);
    return () => clearTimeout(timer);
  }, [samples, settings.personalize, settings.memoryStrength, featureDim, loaded]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  /** Gửi landmark lên server để kiểm tra + trích feature; trả mẫu sẵn sàng lưu, hoặc lý do không lưu. */
  const prepareSample = async (label: string, frames: FrameLandmarks[], source: SampleSource) => {
    const result = await submitSample(settings.apiUrl, label, frames);
    const sample: Sample | null = result.ok ? {
      id: newSampleId(), label, features: result.features, frames: result.frames,
      featureDim: featureDim ?? Math.round(atob(result.features).length / 4 / result.frames),
      handFrames: result.hand_frames, createdAt: Date.now(), source,
    } : null;
    return { result, sample };
  };

  /** Ghi mẫu thủ công (nút Ghi mẫu / Lưu làm mẫu). */
  const storeSample = async (label: string, frames: FrameLandmarks[]) => {
    const { result, sample } = await prepareSample(label, frames, "manual");
    if (!sample) {
      setNotice({ text: `Chưa lưu: tay xuất hiện quá ngắn (${result.hand_frames}/${result.frames} frame). Hãy ký lại.`, tone: "warn" });
      return;
    }
    await addSample(sample);
    await reloadSamples();
    const top = result.prediction?.top[0];
    setNotice(result.warning === "model_disagrees" && top
      ? { text: `Đã lưu mẫu “${label}”. Mô hình gốc thấy giống “${top.label}” (${Math.round(top.confidence * 100)}%); nếu ký nhầm, hãy xoá và ghi lại.`, tone: "warn" }
      : { text: `Đã lưu mẫu “${label}”.`, tone: "ok" });
  };

  const recordSample = async (label: string) => {
    const recognizer = recognizerRef.current;
    if (!recognizer?.running || recording) return;
    setPersonalizeOpen(false);
    try {
      for (let n = COUNTDOWN_SECONDS; n > 0; n--) {
        setRecording({ label, stage: "countdown", count: n });
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (!recognizer.running) throw new Error("Camera đã tắt");
      }
      setRecording({ label, stage: "capture", count: 0 });
      const frames = await recognizer.captureSample(SAMPLE_FRAMES);
      setRecording({ label, stage: "saving", count: 0 });
      await storeSample(label, frames);
    } catch (err) {
      setNotice({ text: `Không ghi được mẫu: ${(err as Error).message}`, tone: "warn" });
    } finally {
      setRecording(null);
      setPersonalizeOpen(true);
    }
  };

  const canLearn = () => {
    const current = settingsRef.current;
    return current.personalize && current.autoLearn && Boolean(healthRef.current?.ok);
  };

  /** Tự học: lưu cửa sổ của các từ người dùng giữ lại (chạy nền, tuần tự). */
  const learnWindows = (windows: CommittedWindow[], realtime: boolean) => {
    const picked = windows.filter((w) => !discarded.current.has(w) && !autoSaved.current.has(w) &&
      w.confidence >= AUTO_MIN_CONFIDENCE && w.frames.length >= 8 &&
      !(w.memory?.label === w.word && w.memory.similarity >= AUTO_SKIP_SIMILARITY));
    if (picked.length === 0) return;
    void (async () => {
      const learned: string[] = [];
      for (const w of picked) {
        try {
          const { result, sample } = await prepareSample(w.word, w.frames, "auto");
          // Mô hình gốc rất chắc là từ khác → có thể chính phiên này đã nhận nhầm: không học.
          if (!sample || result.warning === "model_disagrees" || discarded.current.has(w)) continue;
          if (await addSample(sample)) {
            autoSaved.current.set(w, sample.id);
            learned.push(w.word);
            // Người dùng hoàn tác đúng lúc đang lưu: bỏ mẫu vừa lưu.
            if (discarded.current.has(w)) await forget([w]);
          }
        } catch {
          return;          // server không hỗ trợ / mất kết nối: bỏ qua lần tự học này
        }
      }
      if (learned.length > 0) {
        await reloadSamples();
        setNotice({
          text: realtime ? `Đã học mẫu “${learned.join("”, “")}”.` : `Đã tự học ${learned.length} mẫu từ câu vừa ký.`,
          tone: "ok",
        });
      }
    })();
  };

  /** Bỏ các cửa sổ người dùng đã hoàn tác/xoá: huỷ lượt học đang chờ, xoá mẫu đã học từ chúng. */
  const forget = async (windows: CommittedWindow[]) => {
    let removed = 0;
    for (const w of windows) {
      discarded.current.add(w);
      const timer = pendingLearn.current.get(w);
      if (timer) clearTimeout(timer);
      pendingLearn.current.delete(w);
      const id = autoSaved.current.get(w);
      if (id) {
        await deleteSample(id);
        autoSaved.current.delete(w);
        removed += 1;
      }
    }
    if (removed > 0) await reloadSamples();
  };

  learnRef.current = {
    committed: (w) => {
      if (settingsRef.current.learnMode !== "realtime" || !canLearn()) return;
      // Chờ một nhịp: từ bị hoàn tác ngay sau khi chốt thì không học (và không phải xoá lại).
      pendingLearn.current.set(w, setTimeout(() => {
        pendingLearn.current.delete(w);
        if (canLearn()) learnWindows([w], true);
      }, REALTIME_GRACE_MS));
    },
    sentence: (windows) => {
      if (settingsRef.current.learnMode === "sentence" && canLearn()) learnWindows(windows, false);
    },
    discarded: (windows) => void forget(windows),
  };

  const saveCorrection = async (label: string) => {
    const committed = recognizerRef.current?.lastCommitted;
    if (!committed || recording) return;
    setRecording({ label, stage: "saving", count: 0 });
    try {
      // Cửa sổ này đã (hoặc sắp) được tự học với nhãn sai → bỏ mẫu đó trước.
      await forget([committed]);
      await storeSample(label, committed.frames);
    } catch (err) {
      setNotice({ text: `Không lưu được mẫu: ${(err as Error).message}`, tone: "warn" });
    } finally {
      setRecording(null);
    }
  };

  const removeLabel = async (label: string) => {
    await deleteLabel(label);
    await reloadSamples();
  };

  const removeAll = async () => {
    await clearSamples();
    await reloadSamples();
    setNotice({ text: "Đã xoá mẫu cá nhân trên trình duyệt này.", tone: "ok" });
  };

  const usableSamples = featureDim === undefined ? samples : samples.filter((s) => s.featureDim === featureDim);

  const phase: Phase = state?.phase ?? "stopped";
  const running = phase !== "stopped";
  // Điện thoại đang chạy: camera phủ gần hết màn hình, phụ đề và nút nằm đè ở góc (như app camera).
  const immersive = touch && running;
  const front = (state?.facing ?? "user") === "user";
  // Chỉ lật gương hình xem trước của camera trước; camera sau hiện đúng như mắt người cầm máy thấy.
  const mirrorView = settings.mirrorPreview && front;
  const words = state?.words ?? [];
  const prediction = state?.prediction ?? null;
  const showCandidate = running && phase === "signing" && state?.candidate && state.count > 0;
  const bufferProgress = state && state.windowFrames > 0 ? Math.min(1, state.buffered / state.windowFrames) : 0;

  // Mỗi lần chốt thêm một từ → tăng id để chạy lại hoạt ảnh nhấn mạnh (xoá/undo thì không).
  const [commit, setCommit] = useState<{ id: number; word: string } | null>(null);
  const prevCount = useRef(0);
  const commitId = useRef(0);
  const wordCount = words.length;
  const lastWord = words[wordCount - 1];
  useEffect(() => {
    if (wordCount > prevCount.current && lastWord) setCommit({ id: ++commitId.current, word: lastWord });
    else if (wordCount === 0) setCommit(null); // tránh phát lại khi bật camera lại
    prevCount.current = wordCount;
  }, [wordCount, lastWord]);

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

  const switchCamera = () => void recognizerRef.current?.switchCamera();

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

  // Khoá cuộn trang phía sau khung camera toàn màn hình.
  useEffect(() => {
    document.documentElement.classList.toggle("immersive", immersive);
    return () => document.documentElement.classList.remove("immersive");
  }, [immersive]);

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
        <div className="topbar-right">
          <span className={`health ${health?.ok ? "ok" : health ? "bad" : "wait"}`} title={health?.checkpoint}>
            <i className="health-dot" />
            <span className="health-text">{healthText}</span>
          </span>
          <ThemeToggle />
          <button
            className={`icon-button${settings.personalize && usableSamples.length > 0 ? " is-on" : ""}`}
            onClick={() => setPersonalizeOpen(true)}
            aria-label="Mở cá nhân hoá"
            title={settings.personalize && usableSamples.length > 0 ? `Cá nhân hoá: ${usableSamples.length} mẫu` : "Cá nhân hoá"}
          >
            <Icon name="user" />
          </button>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Mở cài đặt">
            <Icon name="settings" />
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="stage" aria-label="Camera">
          <div className={`frame phase-${phase}${immersive ? " immersive" : ""}`}>
            <video
              ref={videoRef}
              className={mirrorView ? "mirror" : undefined}
              playsInline
              muted
            />
            {settings.debugCrop && (
              <canvas ref={overlayRef} className={`debug-overlay${mirrorView ? " mirror" : ""}`} aria-hidden="true" />
            )}
            <span className="corner tl" /><span className="corner tr" />
            <span className="corner bl" /><span className="corner br" />

            {!running && (
              <div className="frame-empty">
                <button className="start" onClick={toggle} aria-label="Bắt đầu phiên dịch">
                  <span className="start-ring" aria-hidden="true" />
                  <Icon name="play" />
                </button>
                <p className="frame-empty-title">Bắt đầu phiên dịch</p>
                <p className="frame-empty-note">Camera chỉ dùng để nhận diện, không lưu hình.</p>
              </div>
            )}
            {phase === "starting" && <div className="frame-empty"><p className="frame-empty-title">Đang mở camera…</p></div>}

            {recording && (
              <div className="record-overlay" role="status">
                <p className="record-label">Mẫu: <strong>{recording.label}</strong></p>
                {recording.stage === "countdown" && (
                  <>
                    <span key={recording.count} className="record-count">{recording.count}</span>
                    <p className="record-hint">Chuẩn bị ký, bắt đầu khi hết đếm</p>
                  </>
                )}
                {recording.stage === "capture" && (
                  <>
                    <span className="record-bar">
                      <span style={{ transform: `scaleX(${(state?.capture?.done ?? 0) / SAMPLE_FRAMES})` }} />
                    </span>
                    <p className="record-hint">Đang ghi, ký một lần trọn vẹn</p>
                    <button className="btn btn-ghost" onClick={() => recognizerRef.current?.cancelCapture()}>Huỷ</button>
                  </>
                )}
                {recording.stage === "saving" && <p className="record-hint">Đang kiểm tra mẫu…</p>}
              </div>
            )}

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
                  <span
                    className={`hud-latency${state.captureFps != null && state.captureFps < 12 ? " slow" : ""}`}
                    title={`Từ lúc chụp frame tới lúc có kết quả; MediaPipe ${state.delegate ?? ""} trên trình duyệt ${state.detectMs ?? "–"} ms/frame; thu ${state.captureFps ?? "–"} fps (chuẩn 15)`}
                  >
                    {state.latencyMs} ms{state.captureFps != null ? ` · ${Math.round(state.captureFps)} fps` : ""}
                  </span>
                )}
                {commit && (
                  <>
                    <span key={`flash-${commit.id}`} className="frame-flash" aria-hidden="true" />
                    <span key={`wave-${commit.id}`} className="frame-wave" aria-hidden="true" />
                    <span key={`stamp-${commit.id}`} className="hud-stamp" aria-hidden="true">
                      <svg className="stamp-check" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" pathLength={1} />
                        <path d="M7.5 12.5l3 3 6-6.5" pathLength={1} />
                      </svg>
                      <span className="stamp-word">
                        {graphemes(commit.word).map((ch, i) => (
                          <span key={i} style={{ "--i": i } as React.CSSProperties}>{ch}</span>
                        ))}
                      </span>
                    </span>
                  </>
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

            {immersive && (
              <>
                {/* Phụ đề ở góc dưới trái: các từ vừa ký (từ đang chờ chốt mờ dần), bên dưới là câu đã ghép. */}
                <div className="live-caption" aria-live="polite">
                  {words.length > 0 || showCandidate ? (
                    <>
                      <div className="live-tokens">
                        {words.map((word, i) => (
                          <span key={`${i}-${word}`} className="live-token">{word}</span>
                        ))}
                        {showCandidate && (
                          <span
                            key={`ghost-${state!.candidate}`}
                            className="live-token ghost"
                            style={{ "--fill": state!.count / state!.required } as React.CSSProperties}
                          >
                            {state!.candidate}
                          </span>
                        )}
                      </div>
                      {sentence.text && <p className="live-text">{sentence.text}</p>}
                    </>
                  ) : history[0] ? (
                    <>
                      <span className="live-label">Câu vừa xong</span>
                      <p className="live-text">{history[0].text}</p>
                    </>
                  ) : (
                    <p className="live-hint">Đưa tay vào khung hình để ký.</p>
                  )}
                </div>

                {state?.error && <p className="live-error" role="alert">{state.error}</p>}

                {/* Nút dọc mép phải, trong tầm ngón cái. */}
                <div className="live-actions">
                  {(state?.cameraCount ?? 0) > 1 && (
                    <button
                      className="round-btn"
                      onClick={switchCamera}
                      disabled={state?.switching || phase === "starting"}
                      aria-label={front ? "Chuyển sang camera sau" : "Chuyển sang camera trước"}
                    >
                      <Icon name="flip" />
                    </button>
                  )}
                  <button className="round-btn" onClick={() => recognizerRef.current?.undo()} disabled={words.length === 0} aria-label="Xoá từ cuối">
                    <Icon name="undo" />
                  </button>
                  <button className="round-btn" onClick={endSentence} disabled={words.length === 0} aria-label="Kết thúc câu">
                    <Icon name="flag" />
                  </button>
                  <button className="round-btn stop" onClick={toggle} aria-label="Dừng phiên dịch">
                    <Icon name="stop" />
                  </button>
                </div>
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
          {notice && <p className={`notice notice-${notice.tone}`} role="status">{notice.text}</p>}

          <div className={`insight ${prediction ? "" : "is-empty"}`} aria-live="polite">
            <div className="insight-head">
              <span className="insight-title">Mô hình đang thấy</span>
              {prediction && !prediction.accepted && (
                <Swap value={prediction.rejection_reason ?? ""} className="insight-reject">
                  {REJECT_TEXT[prediction.rejection_reason ?? ""] ?? "Chưa chắc chắn"}
                </Swap>
              )}
              {prediction?.memory && prediction.memory.similarity >= 0.5 && (
                <span className="insight-memory" title="Độ giống với mẫu cá nhân gần nhất">
                  Giống mẫu “{prediction.memory.label}” {percent(prediction.memory.similarity)}
                </span>
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
              {sentence.text || "Câu dịch sẽ hiện ở đây."}
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
              <p className="transcript-empty">Câu hoàn chỉnh sẽ lưu ở đây để chép hoặc đọc to.</p>
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

      <PersonalizeSheet
        open={personalizeOpen}
        onClose={() => setPersonalizeOpen(false)}
        classes={health?.classes ?? []}
        samples={usableSamples}
        staleCount={samples.length - usableSamples.length}
        enabled={settings.personalize}
        strength={settings.memoryStrength}
        running={running}
        busy={recording !== null}
        activeExemplars={state?.memoryExemplars ?? null}
        memoryError={state?.memoryError ?? null}
        committed={commit ? recognizerRef.current?.lastCommitted ?? null : null}
        autoLearn={settings.autoLearn}
        learnMode={settings.learnMode}
        onLearnMode={(value) => update("learnMode", value)}
        onAutoLearn={(value) => update("autoLearn", value)}
        onEnabled={(value) => update("personalize", value)}
        onStrength={(value) => update("memoryStrength", value)}
        onRecord={(label) => void recordSample(label)}
        onSaveCorrection={(label) => void saveCorrection(label)}
        onDeleteLabel={(label) => void removeLabel(label)}
        onClearAll={() => void removeAll()}
      />

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
            <Switch
              label="Ghi nhớ key trên trình duyệt này"
              checked={settings.rememberKey} onChange={(value) => update("rememberKey", value)}
            />
            <p className="help">
              Key chỉ gửi kèm yêu cầu ghép câu tới Groq, server không lưu. Không có key thì ghép thô.
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
                className={`segmented-track${settings.turbo ? " inactive" : ""}`}
                style={{
                  "--i": (Object.keys(SPEED_PRESETS) as Speed[]).indexOf(settings.speed),
                  "--n": Object.keys(SPEED_PRESETS).length,
                } as React.CSSProperties}
              >
                <span className="segmented-thumb" aria-hidden="true" />
                {(Object.keys(SPEED_PRESETS) as Speed[]).map((key) => (
                  <label key={key} className={settings.speed === key && !settings.turbo ? "selected" : ""}>
                    <input
                      type="radio" name="speed" value={key}
                      checked={settings.speed === key}
                      onChange={() => setSettings((prev) => ({ ...prev, speed: key, turbo: false }))}
                      onClick={() => update("turbo", false)}
                    />
                    {SPEED_PRESETS[key].label.replace(/\s*\(.*\)/, "")}
                  </label>
                ))}
              </div>
              <Swap value={settings.speed} className={`help${settings.turbo ? " dimmed" : ""}`}>
                {SPEED_PRESETS[settings.speed].hint}
              </Swap>
            </fieldset>
            <div className={`turbo${settings.turbo ? " on" : ""}`}>
              <Switch
                label="Siêu tốc" hint="Nhanh hơn cả Rất nhanh, đổi lại dễ nhầm hơn."
                checked={settings.turbo} onChange={(value) => update("turbo", value)}
              />
              <div className="turbo-panel" inert={!settings.turbo}>
                <div className="turbo-panel-inner">
                  <div className="turbo-panel-content">
                    <Slider
                      stops min={0} max={TURBO_LEVELS.length - 1} step={1}
                      value={TURBO_LEVELS.indexOf(settings.turboLevel)}
                      ariaLabel="Mức tăng tốc" valueText={TURBO_PRESETS[settings.turboLevel].label}
                      onChange={(i) => update("turboLevel", TURBO_LEVELS[i])}
                    />
                    <div className="slider-labels" aria-hidden="true">
                      {TURBO_LEVELS.map((level, i) => (
                        <button
                          key={level} type="button" tabIndex={-1}
                          className={settings.turboLevel === level ? "selected" : ""}
                          style={{ "--s": i / (TURBO_LEVELS.length - 1) } as React.CSSProperties}
                          onClick={() => update("turboLevel", level)}
                        >
                          {TURBO_PRESETS[level].label}
                        </button>
                      ))}
                    </div>
                    <Swap value={`turbo-${settings.turboLevel}`} className="help">
                      {TURBO_PRESETS[settings.turboLevel].hint}
                    </Swap>
                  </div>
                </div>
              </div>
            </div>
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
            <Switch label="Lật gương hình xem trước" checked={settings.mirrorPreview} onChange={(value) => update("mirrorPreview", value)} />
            <Switch label="Lật ảnh khi nhận diện (khớp lúc thu dữ liệu)" checked={settings.mirrorData} onChange={(value) => update("mirrorData", value)} />
            <Switch label="Hiện khung crop người ký (debug)" checked={settings.debugCrop} onChange={(value) => update("debugCrop", value)} />
            <label className="field">
              <span>Địa chỉ API nhận diện</span>
              <input value={settings.apiUrl} spellCheck={false} onChange={(e) => update("apiUrl", e.target.value)} />
            </label>
            {state?.resolution && (
              <p className="help">
                Camera: {state.resolution}, pad về 1280×1280. MediaPipe chạy trên trình duyệt
                {state.delegate ? ` (${state.delegate}${state.detectMs != null ? `, ${state.detectMs} ms/frame` : ""})` : ""}, chỉ gửi landmark lên API.
                {state.captureFps != null && ` Thu ${state.captureFps} fps (chuẩn 15; máy chậm vẫn dự đoán đủ 2 giây nhưng ít chi tiết chuyển động hơn).`}
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
