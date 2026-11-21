// Nhận diện liên tục từ webcam bằng Flask API.
//
// - Thu frame ở 15 fps (nhịp lúc thu dữ liệu); cửa sổ dự đoán 30 frame = 2 giây (độ dài một mẫu train).
// - Frame được mã hoá JPEG bất đồng bộ (OffscreenCanvas.convertToBlob / canvas.toBlob) nên không chặn
//   main thread (toDataURL đồng bộ tốn ~10–30 ms mỗi frame, làm giật hoạt ảnh).
// - Chế độ "stream" (mặc định, server có /api/stream): mở một phiên, mỗi request chỉ gửi các frame MỚI;
//   server giữ tracking MediaPipe và cửa sổ 30 frame nên mỗi frame chỉ trích một lần.
//   Server báo `binary` thì frame đi dạng multipart nhị phân (không base64, không preflight CORS) và được
//   gửi NGAY khi thu (`predict: false`): server trích feature dần, chỉ chạy model mỗi `strideFrames` frame →
//   request dự đoán chỉ còn trích 1 frame + chạy model (~50 ms thay vì ~170 ms khi gửi 5 frame một lần).
//   Server cũ không có /api/stream thì tự chuyển về chế độ "window": gửi cả 30 frame tới /api/predict.
// - Luôn chỉ có MỘT request đang chạy. Mỗi phản hồi có dự đoán là một phiếu cho WordVoter.
// - `hand_frames` (số frame có tay đang ký) phân biệt: nghỉ tay (release, cho lặp từ) / đang vào ký hiệu
//   (đứt chuỗi) / đủ tay (bỏ phiếu). Server cũ không trả trường này thì coi như đủ tay.
// - Chốt một từ → xoá cửa sổ để ký hiệu kế tiếp bắt đầu từ frame mới; phản hồi của cửa sổ cũ bị bỏ.
// - Tốc độ (SPEED_PRESETS, chỉ chế độ stream): dự đoán trên `predictFrames` frame gần nhất ngay khi có
//   `minFrames` frame, gửi mỗi `strideFrames` frame → có kết quả sớm hơn nhưng model thấy ít ký hiệu hơn.
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

/** Phản hồi của POST /api/stream/<id>: có kết quả dự đoán khi `predicted` (server cũ: khi `ready`). */
type StreamResponse = Partial<Prediction> & {
  ready: boolean;
  predicted?: boolean;
  buffered: number;
  window: number;
  hand_frames: number;
  elapsed_ms: number;
};

export type Phase = "stopped" | "starting" | "buffering" | "idle" | "entering" | "signing";
export type Mode = "stream" | "window";
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
  /** Dùng /api/stream nếu server hỗ trợ (nhanh hơn nhiều). */
  streaming: boolean;
  captureFps: number;
  /** Số frame gửi tới /api/predict ở chế độ window (server yêu cầu đúng 30). */
  windowFrames: number;
  /** Chế độ stream: dự đoán trên tối đa chừng này frame gần nhất. */
  predictFrames: number;
  /** Chế độ stream: có chừng này frame thì bắt đầu dự đoán. */
  minFrames: number;
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
  predictFrames: SPEED_PRESETS.accurate.predictFrames,
  minFrames: SPEED_PRESETS.accurate.minFrames,
  strideFrames: SPEED_PRESETS.accurate.strideFrames,
  maxSide: 1280,
  jpegQuality: 0.85,
  mirrorData: false,
  minConsecutive: 2,
  idleHandFrames: 3,
  minHandFrames: SPEED_PRESETS.accurate.minHandFrames,
  sentenceGapMs: 3000,
};

/** Tối đa số frame một request /api/stream (giống MAX_STREAM_FRAMES của server). */
const MAX_STREAM_FRAMES = 60;

export interface RecognizerState {
  phase: Phase;
  mode: Mode | null;
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
  error: string | null;
}

interface Frame {
  blob: Blob;
  t: number;
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
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
  private canvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  /** Đang mã hoá frame trước (bỏ tick nếu máy chậm hơn nhịp thu). */
  private grabbing = false;
  /** Chế độ window: 30 frame gần nhất. Chế độ stream: các frame chưa gửi lên server. */
  private buffer: Frame[] = [];
  /** Số frame thu được từ lần dự đoán trước. */
  private sinceSend = 0;
  /** Server nhận multipart + cờ predict: gửi từng frame ngay khi thu. */
  private binary = false;
  private generation = 0;
  private inFlight: AbortController | null = null;
  private idleSince: number | null = null;
  private mode: Mode | null = null;
  private sessionId: string | null = null;
  /** Số frame cửa sổ phía server đang giữ (chế độ stream). */
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
      mode: null,
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
    this.emit({ required: this.options.minConsecutive });
    this.emitBuffered();
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
    this.canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : document.createElement("canvas");
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
    if (this.grabbing) return;
    this.grabbing = true;
    const generation = this.generation;
    void this.grab()
      .then((frame) => {
        if (frame && this.running && this.timer) this.addFrame(frame, generation);
      })
      .catch(() => undefined)
      .finally(() => {
        this.grabbing = false;
      });
  }

  private addFrame(frame: Frame, generation: number): void {
    // Frame chụp trước khi cửa sổ bị xoá (chốt từ) vẫn thuộc ký hiệu cũ: bỏ.
    if (generation !== this.generation) return;
    this.buffer.push(frame);
    // Chế độ window giữ 30 frame gần nhất; chế độ stream giữ frame chưa gửi (tối đa một request).
    const limit = this.mode === "stream" ? MAX_STREAM_FRAMES : this.options.windowFrames;
    if (this.buffer.length > limit) this.buffer.splice(0, this.buffer.length - limit);
    this.sinceSend += 1;
    this.emitBuffered();
    this.maybePredict();
  }

  private emitBuffered(): void {
    const { windowFrames, predictFrames, minFrames } = this.options;
    if (this.mode === "window") {
      this.emit({ buffered: this.buffer.length, windowFrames });
      return;
    }
    const target = Math.min(minFrames, predictFrames);
    this.emit({ buffered: Math.min(target, this.serverBuffered + this.buffer.length), windowFrames: target });
  }

  private async grab(): Promise<Frame | null> {
    const { video, canvas } = this;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!canvas || !w || !h || video.readyState < 2) return null;
    // Gửi frame gốc (không lật); server tự lật nếu mirror_data = true.
    const scale = Math.min(1, this.options.maxSide / Math.max(w, h));
    const width = Math.round(w * scale);
    const height = Math.round(h * scale);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const t = performance.now();
    const quality = this.options.jpegQuality;
    if (canvas instanceof HTMLCanvasElement) {
      canvas.getContext("2d")!.drawImage(video, 0, 0, width, height);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
      return blob ? { blob, t } : null;
    }
    (canvas.getContext("2d") as OffscreenCanvasRenderingContext2D).drawImage(video, 0, 0, width, height);
    return { blob: await canvas.convertToBlob({ type: "image/jpeg", quality }), t };
  }

  private maybePredict(): void {
    if (this.inFlight) return;
    const due = this.sinceSend >= this.options.strideFrames;
    if (this.mode === "window" || (this.mode === null && !this.options.streaming)) {
      if (!due || this.buffer.length < this.options.windowFrames) return;
      this.mode = "window";
      this.sinceSend = 0;
      void this.run((signal) => this.predictWindow(signal));
      return;
    }
    // stream (hoặc chưa dò): server nhị phân nhận frame ngay khi thu; server cũ thì gom đủ stride.
    if (this.buffer.length === 0 || (!due && !(this.binary && this.sessionId))) return;
    const frames = this.buffer;
    this.buffer = [];
    if (due) this.sinceSend = 0;
    void this.run((signal) => this.predictStream(frames, due, signal));
  }

  /** Chạy một request (chỉ một request tại một thời điểm); bỏ kết quả nếu cửa sổ đã bị xoá. */
  private async run(request: (signal: AbortSignal) => Promise<StreamResponse | Prediction | null>): Promise<void> {
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
      // Frame thu trong lúc chờ: gửi tiếp ngay (chế độ nhị phân) thay vì đợi tick kế.
      if (this.running && this.timer) this.maybePredict();
    }
  }

  private async post<T>(path: string, body: unknown, signal: AbortSignal): Promise<T> {
    // FormData: trình duyệt tự đặt Content-Type multipart (request "đơn giản", không cần preflight CORS).
    const response = await fetch(`${this.base}${path}`, body instanceof FormData
      ? { method: "POST", body, signal }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
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
    const data = await Promise.all(frames.map((f) => blobToDataUrl(f.blob)));
    return this.post<Prediction>("/api/predict", {
      frames: data, fps, mirror_data: this.options.mirrorData, top_k: 3,
    }, signal);
  }

  private async predictStream(frames: Frame[], predict: boolean, signal: AbortSignal): Promise<StreamResponse | null> {
    if (!this.sessionId) {
      try {
        const opened = await this.post<{ session_id: string; window: number; binary?: boolean }>("/api/stream", {}, signal);
        this.sessionId = opened.session_id;
        this.binary = opened.binary === true;
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
    const meta = {
      timestamps: frames.map((f) => f.t),
      reset,
      predict,
      window: this.options.predictFrames,
      min_frames: Math.min(this.options.minFrames, this.options.predictFrames),
      fps: this.options.captureFps,
      mirror_data: this.options.mirrorData,
      top_k: 3,
    };
    let payload: FormData | Record<string, unknown>;
    if (this.binary) {
      payload = new FormData();
      payload.append("meta", JSON.stringify(meta));
      frames.forEach((f, i) => (payload as FormData).append("frames", f.blob, `${i}.jpg`));
    } else {
      payload = { ...meta, frames: await Promise.all(frames.map((f) => blobToDataUrl(f.blob))) };
    }
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

  private handle(response: StreamResponse | Prediction, requestMs: number): void {
    const { idleHandFrames, minHandFrames } = this.options;
    const now = performance.now();
    // Độ trễ người dùng cảm nhận: từ lúc thu frame mới nhất tới lúc có kết quả.
    const latencyMs = "ready" in response && this.lastFrameAt !== null ? now - this.lastFrameAt : requestMs;
    if ("ready" in response) {
      this.serverBuffered = response.buffered;
      // Chỉ trích feature (predict: false) khi cửa sổ đã đủ: không có phiếu, chỉ cập nhật số frame.
      if (response.ready && !(response.predicted ?? true)) {
        this.emitBuffered();
        return;
      }
    }

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
