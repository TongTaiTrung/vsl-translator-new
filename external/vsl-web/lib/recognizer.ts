// Nhận diện liên tục từ webcam bằng Flask API.
//
// - Thu frame ở 15 fps (nhịp lúc thu dữ liệu); cửa sổ dự đoán 30 frame = 2 giây (độ dài một mẫu train).
// - Chế độ "stream" (mặc định, server có /api/stream): mở một phiên, mỗi request chỉ gửi các frame MỚI;
//   server giữ tracking MediaPipe và cửa sổ 30 frame nên mỗi frame chỉ trích một lần (~0.15–0.3 s/request).
//   Server cũ không có /api/stream thì tự chuyển về chế độ "window": gửi cả 30 frame tới /api/predict.
// - Luôn chỉ có MỘT request đang chạy; gửi khi đã có thêm `strideFrames` frame mới. Mỗi phản hồi là một
//   phiếu cho WordVoter.
// - `hand_frames` (số frame có tay đang ký) phân biệt: nghỉ tay (release, cho lặp từ) / đang vào ký hiệu
//   (đứt chuỗi) / đủ tay (bỏ phiếu). Server cũ không trả trường này thì coi như đủ tay.
// - Chốt một từ → xoá cửa sổ để ký hiệu kế tiếp bắt đầu từ frame mới; phản hồi của cửa sổ cũ bị bỏ.
// - Nghỉ tay liên tục `sentenceGapMs` sau khi đã có từ → kết thúc câu (onSentenceEnd).

import { WordVoter } from "./voter";

export interface TopItem {
  label: string;
  confidence: number;
}

export interface Prediction {
  label: string | null;
  confidence: number;
  probability_gap: number;
  accepted: boolean;
  rejection_reason: "ambiguous" | "low_confidence" | null;
  top: TopItem[];
  frames: number;
  hand_frames?: number;
  elapsed_ms: number;
}

/** Phản hồi của POST /api/stream/<id>: có kết quả dự đoán khi `ready`. */
type StreamResponse = Partial<Prediction> & {
  ready: boolean;
  buffered: number;
  window: number;
  hand_frames: number;
  elapsed_ms: number;
};

export type Phase = "stopped" | "starting" | "buffering" | "idle" | "entering" | "signing";
export type Mode = "stream" | "window";

export interface RecognizerOptions {
  apiUrl: string;
  /** Dùng /api/stream nếu server hỗ trợ (nhanh hơn nhiều). */
  streaming: boolean;
  captureFps: number;
  windowFrames: number;
  strideFrames: number;
  maxSide: number;
  jpegQuality: number;
  mirrorData: boolean;
  minConsecutive: number;
  /** <= số này frame có tay: người dùng đang nghỉ. */
  idleHandFrames: number;
  /** >= số này frame có tay: đủ để bỏ phiếu. */
  minHandFrames: number;
  /** Nghỉ tay bao lâu (ms) thì kết thúc câu; 0 = tắt. */
  sentenceGapMs: number;
}

export const DEFAULT_OPTIONS: RecognizerOptions = {
  apiUrl: process.env.NEXT_PUBLIC_VSL_API_URL || "http://127.0.0.1:8000",
  streaming: true,
  captureFps: 15,
  windowFrames: 30,
  strideFrames: 5,
  maxSide: 1280,
  jpegQuality: 0.85,
  mirrorData: false,
  minConsecutive: 2,
  idleHandFrames: 3,
  minHandFrames: 12,
  sentenceGapMs: 3000,
};

/** Tối đa số frame một request /api/stream (giống MAX_STREAM_FRAMES của server). */
const MAX_STREAM_FRAMES = 60;

export interface RecognizerState {
  phase: Phase;
  mode: Mode | null;
  buffered: number;
  windowFrames: number;
  inFlight: boolean;
  prediction: Prediction | null;
  candidate: string | null;
  count: number;
  required: number;
  words: string[];
  lastCommitted: string | null;
  resolution: string | null;
  latencyMs: number | null;
  error: string | null;
}

interface Frame {
  data: string;
  t: number;
}

export interface RecognizerEvents {
  onState: (state: RecognizerState) => void;
  onWords: (words: string[]) => void;
  onSentenceEnd: (words: string[]) => void;
}

class HttpError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export class Recognizer {
  private options: RecognizerOptions;
  private voter: WordVoter;
  private stream: MediaStream | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private canvas: HTMLCanvasElement | null = null;
  /** Chế độ window: 30 frame gần nhất. Chế độ stream: các frame chưa gửi lên server. */
  private buffer: Frame[] = [];
  private sinceSend = 0;
  private generation = 0;
  private inFlight: AbortController | null = null;
  private idleSince: number | null = null;
  private mode: Mode | null = null;
  private sessionId: string | null = null;
  /** Số frame cửa sổ phía server đang giữ (chế độ stream). */
  private serverBuffered = 0;
  /** Request stream kế tiếp phải xoá cửa sổ server (sau khi chốt từ / xoá hết). */
  private resetNext = false;
  private state: RecognizerState;

  constructor(
    private video: HTMLVideoElement,
    private events: RecognizerEvents,
    options: Partial<RecognizerOptions> = {},
  ) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.voter = new WordVoter(this.options.minConsecutive);
    this.state = {
      phase: "stopped",
      mode: null,
      buffered: 0,
      windowFrames: this.options.windowFrames,
      inFlight: false,
      prediction: null,
      candidate: null,
      count: 0,
      required: this.options.minConsecutive,
      words: [],
      lastCommitted: null,
      resolution: null,
      latencyMs: null,
      error: null,
    };
  }

  get running(): boolean {
    return this.state.phase !== "stopped";
  }

  setOptions(partial: Partial<RecognizerOptions>): void {
    const apiChanged = partial.apiUrl !== undefined && partial.apiUrl !== this.options.apiUrl;
    this.options = { ...this.options, ...partial };
    this.voter.minConsecutive = this.options.minConsecutive;
    if (this.timer && partial.captureFps !== undefined) this.restartTimer();
    if (apiChanged && this.running) {
      // Phiên cũ nằm trên server cũ: dò lại chế độ ở request kế tiếp.
      this.inFlight?.abort();
      this.closeSession();
      this.mode = null;
      this.resetWindow();
    }
    this.emit({ required: this.options.minConsecutive, windowFrames: this.options.windowFrames });
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.emit({ phase: "starting", error: null });
    try {
      // Độ phân giải 16:9 lớn nhất camera hỗ trợ (giống src/features/camera.py); ảnh gửi đi thu về maxSide.
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 3840 }, height: { ideal: 2160 }, aspectRatio: { ideal: 16 / 9 } },
        audio: false,
      });
    } catch (err) {
      this.emit({ phase: "stopped", error: `Không mở được camera: ${(err as Error).message}` });
      return;
    }
    this.video.srcObject = this.stream;
    await this.video.play().catch(() => undefined);
    this.canvas = document.createElement("canvas");
    this.mode = null;                // dò ở request đầu tiên
    this.resetWindow();
    this.emit({
      phase: "buffering",
      resolution: `${this.video.videoWidth}×${this.video.videoHeight}`,
    });
    this.restartTimer();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.inFlight?.abort();
    this.inFlight = null;
    this.closeSession();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.resetWindow();
    this.voter.interrupt();
    this.idleSince = null;
    this.emit({ phase: "stopped", inFlight: false, candidate: null, count: 0, buffered: 0 });
  }

  /** Lấy chuỗi từ hiện tại và bắt đầu câu mới (dùng khi bấm "Kết thúc câu"). */
  finishSentence(): string[] {
    const words = [...this.voter.sequence];
    this.voter.clear();
    this.idleSince = null;
    this.emitWords();
    return words;
  }

  undo(): void {
    if (this.voter.undo() !== null) this.emitWords();
  }

  clear(): void {
    this.voter.clear();
    this.resetWindow();
    this.emitWords();
  }

  // ------------------------------------------------------------------ nội bộ

  private get base(): string {
    return this.options.apiUrl.replace(/\/+$/, "");
  }

  private restartTimer(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 1000 / this.options.captureFps);
  }

  private resetWindow(): void {
    this.generation += 1;          // phản hồi của cửa sổ cũ sẽ bị bỏ qua
    this.buffer = [];
    this.sinceSend = 0;
    this.serverBuffered = 0;
    this.resetNext = true;
  }

  private closeSession(): void {
    if (!this.sessionId) return;
    const url = `${this.base}/api/stream/${encodeURIComponent(this.sessionId)}`;
    this.sessionId = null;
    void fetch(url, { method: "DELETE", keepalive: true }).catch(() => undefined);
  }

  private tick(): void {
    this.checkSentenceGap(performance.now());
    const frame = this.grab();
    if (!frame) return;
    this.buffer.push(frame);
    // Chế độ window giữ 30 frame gần nhất; chế độ stream giữ frame chưa gửi (tối đa một request).
    const limit = this.mode === "stream" ? MAX_STREAM_FRAMES : this.options.windowFrames;
    if (this.buffer.length > limit) this.buffer.splice(0, this.buffer.length - limit);
    this.sinceSend += 1;
    this.emitBuffered();
    this.maybePredict();
  }

  private emitBuffered(): void {
    const { windowFrames } = this.options;
    const buffered = this.mode === "stream"
      ? Math.min(windowFrames, this.serverBuffered + this.buffer.length)
      : this.buffer.length;
    this.emit({ buffered });
  }

  private grab(): Frame | null {
    const { video, canvas } = this;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!canvas || !w || !h || video.readyState < 2) return null;
    // Gửi frame gốc (không lật); server tự lật nếu mirror_data = true.
    const scale = Math.min(1, this.options.maxSide / Math.max(w, h));
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
    return { data: canvas.toDataURL("image/jpeg", this.options.jpegQuality), t: performance.now() };
  }

  private maybePredict(): void {
    if (this.inFlight || this.sinceSend < this.options.strideFrames) return;
    if (this.mode === "window" || (this.mode === null && !this.options.streaming)) {
      if (this.buffer.length < this.options.windowFrames) return;
      this.mode = "window";
      this.sinceSend = 0;
      void this.run((signal) => this.predictWindow(signal));
    } else {
      // stream (hoặc chưa dò): gửi mọi frame chưa gửi.
      const frames = this.buffer;
      this.buffer = [];
      this.sinceSend = 0;
      void this.run((signal) => this.predictStream(frames, signal));
    }
  }

  /** Chạy một request (chỉ một request tại một thời điểm); bỏ kết quả nếu cửa sổ đã bị xoá. */
  private async run(request: (signal: AbortSignal) => Promise<StreamResponse | Prediction | null>): Promise<void> {
    const controller = new AbortController();
    const generation = this.generation;
    this.inFlight = controller;
    this.emit({ inFlight: true });
    const started = performance.now();
    try {
      const body = await request(controller.signal);
      if (body && generation === this.generation && this.running) {
        this.handle(body, performance.now() - started);
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        this.emit({ error: `Lỗi gọi API dự đoán: ${(err as Error).message}` });
      }
    } finally {
      if (this.inFlight === controller) this.inFlight = null;
      this.emit({ inFlight: false });
    }
  }

  private async post<T>(path: string, body: unknown, signal: AbortSignal): Promise<T> {
    const response = await fetch(`${this.base}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new HttpError(data.error || `HTTP ${response.status}`, response.status);
    return data as T;
  }

  private async predictWindow(signal: AbortSignal): Promise<Prediction> {
    const frames = this.buffer.slice(-this.options.windowFrames);
    const span = frames[frames.length - 1].t - frames[0].t;
    const fps = span > 0
      ? Math.min(240, Math.max(1, ((frames.length - 1) * 1000) / span))
      : this.options.captureFps;
    return this.post<Prediction>("/api/predict", {
      frames: frames.map((f) => f.data), fps, mirror_data: this.options.mirrorData, top_k: 3,
    }, signal);
  }

  private async predictStream(frames: Frame[], signal: AbortSignal): Promise<StreamResponse | null> {
    if (!this.sessionId) {
      try {
        const opened = await this.post<{ session_id: string; window: number }>("/api/stream", {}, signal);
        this.sessionId = opened.session_id;
        this.mode = "stream";
        this.serverBuffered = 0;
        this.resetNext = false;          // phiên mới vốn rỗng
        this.emit({ mode: "stream" });
      } catch (err) {
        if (err instanceof HttpError && (err.status === 404 || err.status === 405)) {
          // Server cũ không có /api/stream → gửi cả cửa sổ tới /api/predict.
          this.mode = "window";
          this.buffer = [...frames, ...this.buffer].slice(-this.options.windowFrames);
          this.emit({ mode: "window" });
          return null;
        }
        throw err;
      }
    }
    const reset = this.resetNext;
    this.resetNext = false;
    try {
      const body = await this.post<StreamResponse>(`/api/stream/${encodeURIComponent(this.sessionId)}`, {
        frames: frames.map((f) => f.data),
        timestamps: frames.map((f) => f.t),
        reset,
        fps: this.options.captureFps,
        mirror_data: this.options.mirrorData,
        top_k: 3,
      }, signal);
      return body;
    } catch (err) {
      if (err instanceof HttpError && err.status === 404) {
        // Phiên hết hạn (server khởi động lại, nghỉ quá lâu): mở phiên mới ở request kế tiếp.
        this.sessionId = null;
        this.serverBuffered = 0;
        return null;
      }
      if ((err as Error).name !== "AbortError") this.resetNext ||= reset;   // gửi lại reset ở lần sau
      throw err;
    }
  }

  private handle(response: StreamResponse | Prediction, latencyMs: number): void {
    const { idleHandFrames, minHandFrames } = this.options;
    const now = performance.now();
    if ("ready" in response) this.serverBuffered = response.buffered;

    if ("ready" in response && !response.ready) {
      // Cửa sổ server chưa đủ frame: chưa có dự đoán, chỉ theo dõi nghỉ tay để kết thúc câu.
      if (response.hand_frames === 0) {
        this.voter.release();
        this.idleSince = this.idleSince ?? now;
      } else {
        this.idleSince = null;
      }
      this.emit({ phase: "buffering", latencyMs: Math.round(latencyMs), error: null });
      this.emitBuffered();
      this.checkSentenceGap(now);
      return;
    }

    const prediction = response as Prediction;
    const hands = prediction.hand_frames ?? prediction.frames;
    let phase: Phase;
    let committed: string | null = null;

    if (hands <= idleHandFrames) {
      phase = "idle";
      this.voter.release();
    } else if (hands < minHandFrames) {
      phase = "entering";
      this.voter.interrupt();
    } else {
      phase = "signing";
      committed = this.voter.vote(prediction.accepted ? prediction.label : null);
    }

    this.idleSince = phase === "idle" ? (this.idleSince ?? now) : null;

    this.emit({
      phase,
      prediction,
      latencyMs: Math.round(latencyMs),
      error: null,
      candidate: this.voter.candidate,
      count: this.voter.count,
      ...(committed ? { lastCommitted: committed } : {}),
    });

    if (committed) {
      this.resetWindow();          // ký hiệu kế tiếp bắt đầu từ cửa sổ mới
      this.emitBuffered();
      this.emitWords();
    } else {
      this.checkSentenceGap(now);
    }
  }

  /** Nghỉ tay đủ lâu sau khi đã có từ → kết thúc câu. Gọi cả ở mỗi tick để không phải chờ request kế. */
  private checkSentenceGap(now: number): void {
    const { sentenceGapMs } = this.options;
    if (
      sentenceGapMs > 0 && this.idleSince !== null && this.voter.sequence.length > 0 &&
      now - this.idleSince >= sentenceGapMs
    ) {
      this.events.onSentenceEnd(this.finishSentence());
    }
  }

  private emitWords(): void {
    const words = [...this.voter.sequence];
    this.emit({ words, candidate: this.voter.candidate, count: this.voter.count });
    this.events.onWords(words);
  }

  private emit(partial: Partial<RecognizerState>): void {
    this.state = { ...this.state, ...partial };
    this.events.onState(this.state);
  }
}
