// Nhận diện liên tục từ webcam: MediaPipe chạy trên trình duyệt, chỉ gửi landmark lên Flask API.
//
// - Thu frame ở 15 fps (nhịp lúc thu dữ liệu); cửa sổ dự đoán 30 frame = 2 giây (độ dài một mẫu train).
// - Mỗi frame được pad về 1280x1280 (giống frame_padding của Python) rồi chạy pose + hand landmarker
//   (lib/landmarks.ts, chế độ VIDEO giữ tracking giữa các frame). Chỉ landmark (~3 KB JSON/frame) đi lên server,
//   không còn ảnh: server chỉ chạy extract + model, không chạy MediaPipe.
// - Mở một phiên `POST /api/stream` (phản hồi phải có `landmarks: true`), mỗi frame gửi NGAY khi detect xong
//   tới `POST /api/stream/<id>` với `predict: false`; mỗi `strideFrames` frame mới bật `predict` → server
//   chạy model trên cửa sổ đang giữ.
// - Luôn chỉ có MỘT request đang chạy. Mỗi phản hồi có dự đoán là một phiếu cho WordVoter.
// - `hand_frames` (số frame có tay đang ký) phân biệt: nghỉ tay (release, cho lặp từ) / đang vào ký hiệu
//   (đứt chuỗi) / đủ tay (bỏ phiếu).
// - Chốt một từ → xoá cửa sổ để ký hiệu kế tiếp bắt đầu từ frame mới; phản hồi của cửa sổ cũ bị bỏ.
// - Tốc độ (SPEED_PRESETS): dự đoán trên `predictFrames` frame gần nhất ngay khi có `minFrames` frame,
//   gửi mỗi `strideFrames` frame → có kết quả sớm hơn nhưng model thấy ít ký hiệu hơn.
// - Nghỉ tay liên tục `sentenceGapMs` sau khi đã có từ → kết thúc câu (onSentenceEnd).

import { type Delegate, type FrameLandmarks, LandmarkDetector } from "./landmarks";
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

/** Phản hồi của POST /api/stream/<id>: có kết quả dự đoán khi `predicted`. */
type StreamResponse = Partial<Prediction> & {
  ready: boolean;
  predicted: boolean;
  buffered: number;
  window: number;
  hand_frames: number;
  elapsed_ms: number;
};

export type Phase = "stopped" | "starting" | "buffering" | "idle" | "entering" | "signing";
export type Speed = "accurate" | "fast" | "fastest";

export interface SpeedPreset {
  label: string;
  hint: string;
  /** Dự đoán trên tối đa chừng này frame gần nhất (server nội suy về 30). */
  predictFrames: number;
  /** Có chừng này frame là bắt đầu dự đoán (sau khi bắt đầu / sau mỗi từ). */
  minFrames: number;
  strideFrames: number;
  minHandFrames: number;
}

// Thời gian từ lúc bắt đầu ký tới lúc chốt từ, đo trên trình duyệt (video clip thu dữ liệu làm camera,
// 2 phiếu): chuẩn ~1.5 s, nhanh ~1.0 s, rất nhanh ~0.8 s. Phát lại offline các clip holdout không thấy giảm
// độ chính xác, nhưng clip thu sạch hơn camera thật: trên camera thật bản nhanh dễ nhầm hơn (nhất là lúc
// đưa tay lên / chuyển ký hiệu). Giảm "Số phiếu để chốt từ" xuống 1 còn nhanh hơn nữa (và dễ nhầm hơn).
export const SPEED_PRESETS: Record<Speed, SpeedPreset> = {
  accurate: {
    label: "Chuẩn (~1.5 giây)",
    hint: "Chính xác nhất: chờ đủ 2 giây (cả ký hiệu) mới dự đoán.",
    predictFrames: 30, minFrames: 30, strideFrames: 5, minHandFrames: 12,
  },
  fast: {
    label: "Nhanh (~1 giây)",
    hint: "Dự đoán trên 1.3 giây gần nhất, bắt đầu sau 0.8 giây. Đôi khi nhầm lúc chuyển ký hiệu.",
    predictFrames: 20, minFrames: 12, strideFrames: 3, minHandFrames: 8,
  },
  fastest: {
    label: "Rất nhanh (~0.8 giây)",
    hint: "Dự đoán trên 1 giây gần nhất, bắt đầu sau 0.5 giây. Nhanh nhất nhưng dễ nhầm nhất, nên ký dứt khoát.",
    predictFrames: 15, minFrames: 8, strideFrames: 2, minHandFrames: 6,
  },
};

export interface RecognizerOptions {
  apiUrl: string;
  captureFps: number;
  /** Dự đoán trên tối đa chừng này frame gần nhất. */
  predictFrames: number;
  /** Có chừng này frame thì bắt đầu dự đoán. */
  minFrames: number;
  strideFrames: number;
  /** Lật gương frame trước khi chạy MediaPipe; phải giống MIRROR_DATA lúc thu dữ liệu. */
  mirrorData: boolean;
  /** MediaPipe trên trình duyệt: GPU (WebGL, nhanh) hoặc CPU (WASM). GPU lỗi thì tự về CPU. */
  delegate: Delegate;
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
  captureFps: 15,
  predictFrames: SPEED_PRESETS.accurate.predictFrames,
  minFrames: SPEED_PRESETS.accurate.minFrames,
  strideFrames: SPEED_PRESETS.accurate.strideFrames,
  mirrorData: false,
  delegate: "GPU",
  minConsecutive: 2,
  idleHandFrames: 3,
  minHandFrames: SPEED_PRESETS.accurate.minHandFrames,
  sentenceGapMs: 3000,
};

/** Tối đa số frame một request /api/stream (giống MAX_STREAM_FRAMES của server). */
const MAX_STREAM_FRAMES = 60;
const OLD_SERVER = "Server chưa nhận landmark: cập nhật và chạy lại src/app/server.py";

export interface RecognizerState {
  phase: Phase;
  buffered: number;
  /** Số frame cần gom trước khi có dự đoán đầu tiên. */
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
  /** Thời gian MediaPipe trên trình duyệt cho frame gần nhất (ms). */
  detectMs: number | null;
  /** Delegate MediaPipe đang chạy (GPU / CPU), null khi chưa tải. */
  delegate: Delegate | null;
  error: string | null;
}

interface Frame {
  lm: FrameLandmarks;
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
  private detector: LandmarkDetector | null = null;
  /** Các frame (landmark) chưa gửi lên server. */
  private buffer: Frame[] = [];
  /** Số frame thu được từ lần dự đoán trước. */
  private sinceSend = 0;
  private generation = 0;
  private inFlight: AbortController | null = null;
  private idleSince: number | null = null;
  private sessionId: string | null = null;
  /** Số frame cửa sổ phía server đang giữ. */
  private serverBuffered = 0;
  /** Request stream kế tiếp phải xoá cửa sổ server (sau khi chốt từ / xoá hết). */
  private resetNext = false;
  /** Thời điểm thu frame cuối của request stream gần nhất (để đo độ trễ thật). */
  private lastFrameAt: number | null = null;
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
      buffered: 0,
      windowFrames: Math.min(this.options.minFrames, this.options.predictFrames),
      inFlight: false,
      prediction: null,
      candidate: null,
      count: 0,
      required: this.options.minConsecutive,
      words: [],
      lastCommitted: null,
      resolution: null,
      latencyMs: null,
      detectMs: null,
      delegate: null,
      error: null,
    };
  }

  get running(): boolean {
    return this.state.phase !== "stopped";
  }

  setOptions(partial: Partial<RecognizerOptions>): void {
    const apiChanged = partial.apiUrl !== undefined && partial.apiUrl !== this.options.apiUrl;
    const mirrorChanged = partial.mirrorData !== undefined && partial.mirrorData !== this.options.mirrorData;
    this.options = { ...this.options, ...partial };
    this.voter.minConsecutive = this.options.minConsecutive;
    if (this.timer && partial.captureFps !== undefined) this.restartTimer();
    if (apiChanged && this.running) {
      // Phiên cũ nằm trên server cũ: mở phiên mới ở request kế tiếp.
      this.inFlight?.abort();
      this.closeSession();
      this.resetWindow();
    } else if (mirrorChanged && this.running) {
      this.resetWindow();          // landmark trước/sau khi lật gương không trộn chung một cửa sổ
    }
    this.emit({ required: this.options.minConsecutive });
    this.emitBuffered();
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.emit({ phase: "starting", error: null });
    if (!this.detector) {
      try {
        this.detector = await LandmarkDetector.create(this.options.delegate);
      } catch (err) {
        this.emit({ phase: "stopped", error: `Không tải được MediaPipe: ${(err as Error).message}` });
        return;
      }
      this.emit({ delegate: this.detector.delegate });
    }
    try {
      // Độ phân giải 16:9 lớn nhất camera hỗ trợ (giống src/features/camera.py); frame được pad về 1280x1280.
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

  /** Giải phóng MediaPipe (khi gỡ component); start() sau đó sẽ tải lại. */
  dispose(): void {
    this.stop();
    this.detector?.close();
    this.detector = null;
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
    if (frame) this.addFrame(frame);
  }

  private addFrame(frame: Frame): void {
    this.buffer.push(frame);
    // Giữ các frame chưa gửi (tối đa một request); server chậm thì bỏ frame cũ nhất.
    if (this.buffer.length > MAX_STREAM_FRAMES) this.buffer.splice(0, this.buffer.length - MAX_STREAM_FRAMES);
    this.sinceSend += 1;
    this.emitBuffered();
    this.maybePredict();
  }

  private emitBuffered(): void {
    const { predictFrames, minFrames } = this.options;
    const target = Math.min(minFrames, predictFrames);
    this.emit({ buffered: Math.min(target, this.serverBuffered + this.buffer.length), windowFrames: target });
  }

  /** Detect landmark của frame camera hiện tại (đồng bộ; GPU ~5–15 ms, CPU ~20–40 ms). */
  private grab(): Frame | null {
    const { video, detector } = this;
    if (!detector || !video.videoWidth || !video.videoHeight || video.readyState < 2) return null;
    const t = performance.now();
    try {
      const lm = detector.detect(video, t, this.options.mirrorData);
      this.emit({ detectMs: Math.round(performance.now() - t) });
      return { lm, t };
    } catch (err) {
      this.emit({ error: `Lỗi MediaPipe: ${(err as Error).message}` });
      return null;
    }
  }

  private maybePredict(): void {
    if (this.inFlight || this.buffer.length === 0) return;
    const due = this.sinceSend >= this.options.strideFrames;
    const frames = this.buffer;
    this.buffer = [];
    if (due) this.sinceSend = 0;
    void this.run((signal) => this.predictStream(frames, due, signal));
  }

  /** Chạy một request (chỉ một request tại một thời điểm); bỏ kết quả nếu cửa sổ đã bị xoá. */
  private async run(request: (signal: AbortSignal) => Promise<StreamResponse | null>): Promise<void> {
    const controller = new AbortController();
    const generation = this.generation;
    this.inFlight = controller;
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
      // Frame thu trong lúc chờ: gửi tiếp ngay thay vì đợi tick kế.
      if (this.running && this.timer) this.maybePredict();
    }
  }

  private async post<T>(path: string, body: unknown, signal: AbortSignal): Promise<T> {
    // Preflight CORS của JSON được trình duyệt cache 1 ngày (Access-Control-Max-Age của server).
    const response = await fetch(`${this.base}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new HttpError(data.error || `HTTP ${response.status}`, response.status);
    return data as T;
  }

  private async predictStream(frames: Frame[], predict: boolean, signal: AbortSignal): Promise<StreamResponse | null> {
    if (!this.sessionId) {
      let opened: { session_id: string; window: number; landmarks?: boolean };
      try {
        opened = await this.post("/api/stream", {}, signal);
      } catch (err) {
        if (err instanceof HttpError && (err.status === 404 || err.status === 405)) throw new Error(OLD_SERVER);
        throw err;
      }
      if (opened.landmarks !== true) {
        void fetch(`${this.base}/api/stream/${encodeURIComponent(opened.session_id)}`, { method: "DELETE" }).catch(() => undefined);
        throw new Error(OLD_SERVER);
      }
      this.sessionId = opened.session_id;
      this.serverBuffered = 0;
      this.resetNext = false;          // phiên mới vốn rỗng
    }
    const reset = this.resetNext;
    this.resetNext = false;
    const payload = {
      landmarks: frames.map((f) => f.lm),
      timestamps: frames.map((f) => f.t),
      reset,
      predict,
      window: this.options.predictFrames,
      min_frames: Math.min(this.options.minFrames, this.options.predictFrames),
      fps: this.options.captureFps,
      top_k: 3,
    };
    this.lastFrameAt = frames[frames.length - 1].t;
    try {
      return await this.post<StreamResponse>(`/api/stream/${encodeURIComponent(this.sessionId)}`, payload, signal);
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

  private handle(response: StreamResponse, requestMs: number): void {
    const { idleHandFrames, minHandFrames } = this.options;
    const now = performance.now();
    // Độ trễ người dùng cảm nhận: từ lúc thu frame mới nhất (trước MediaPipe) tới lúc có kết quả.
    const latencyMs = this.lastFrameAt !== null ? now - this.lastFrameAt : requestMs;
    this.serverBuffered = response.buffered;
    // Chỉ thêm landmark (predict: false) khi cửa sổ đã đủ: không có phiếu, chỉ cập nhật số frame.
    if (response.ready && !response.predicted) {
      this.emitBuffered();
      return;
    }

    if (!response.ready) {
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
