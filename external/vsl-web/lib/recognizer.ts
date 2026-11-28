/** Recognition loop: capture frames, send landmarks to the streaming API, vote and emit words. */

import { type Delegate, type FrameLandmarks, LandmarkDetector } from "./landmarks";
import type { MemoryPayload } from "./memory";
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
  memory?: { label: string; similarity: number };
}

type StreamResponse = Partial<Prediction> & {
  ready: boolean;
  predicted: boolean;
  buffered: number;
  window: number;
  hand_frames: number;
  elapsed_ms: number;
};

export type Phase = "stopped" | "starting" | "buffering" | "idle" | "entering" | "signing";
export type Facing = "user" | "environment";
export type Speed = "accurate" | "fast" | "fastest";

export interface SpeedPreset {
  label: string;
  hint: string;
  predictFrames: number;
  minFrames: number;
  strideFrames: number;
  minHandFrames: number;
  idleHandFrames?: number;
  fill?: FillMode;
}

export type FillMode = "interp" | "repeat";

export const SPEED_PRESETS: Record<Speed, SpeedPreset> = {
  accurate: {
    label: "Chuẩn (~1.5 giây)",
    hint: "Chính xác nhất: chờ đủ 2 giây mới dự đoán.",
    predictFrames: 30, minFrames: 30, strideFrames: 5, minHandFrames: 12,
  },
  fast: {
    label: "Nhanh (~1 giây)",
    hint: "Dự đoán sau 0.8 giây trên 1.3 giây gần nhất. Đôi khi nhầm lúc chuyển ký hiệu.",
    predictFrames: 20, minFrames: 12, strideFrames: 3, minHandFrames: 8,
  },
  fastest: {
    label: "Rất nhanh (~0.8 giây)",
    hint: "Dự đoán sau 0.5 giây trên 1 giây gần nhất. Nhanh nhất, dễ nhầm nhất: nên ký dứt khoát.",
    predictFrames: 15, minFrames: 8, strideFrames: 2, minHandFrames: 6,
  },
};

export type TurboLevel = 4 | 6 | 8 | 20 | 30 | 40;
export const TURBO_LEVELS: TurboLevel[] = [4, 6, 8, 20, 30, 40];

export const TURBO_PRESETS: Record<TurboLevel, SpeedPreset> = {
  4: {
    label: "x4",
    hint: "Dự đoán sau 0.5 giây trên 0.8 giây gần nhất. Dễ nhầm hơn Rất nhanh, nhất là ký hiệu có chuyển động.",
    predictFrames: 12, minFrames: 7, strideFrames: 2, minHandFrames: 5,
  },
  6: {
    label: "x6",
    hint: "Dự đoán sau 0.33 giây trên 0.67 giây gần nhất, mỗi frame một lần. Chỉ nên dùng khi ký dứt khoát.",
    predictFrames: 10, minFrames: 5, strideFrames: 1, minHandFrames: 4,
  },
  8: {
    label: "x8",
    hint: "Dự đoán sau 0.27 giây trên 0.53 giây gần nhất, mỗi frame một lần. Nhanh nhất và dễ nhầm nhất.",
    predictFrames: 8, minFrames: 4, strideFrames: 1, minHandFrames: 4,
  },
  20: {
    label: "x20",
    hint: "Dự đoán ngay khi có 2 frame (0.13 giây), lặp 3 frame gần nhất cho đủ cửa sổ. Chỉ hợp với ký hiệu tĩnh như chữ cái; từ có chuyển động dễ sai.",
    predictFrames: 3, minFrames: 2, strideFrames: 1, minHandFrames: 2, idleHandFrames: 0, fill: "repeat",
  },
  30: {
    label: "x30",
    hint: "Dự đoán từ frame đầu tiên, lặp 2 frame gần nhất cho đủ cửa sổ. Gần như chỉ nhìn tư thế tay: từ có chuyển động thường bị bỏ qua hoặc sai.",
    predictFrames: 2, minFrames: 1, strideFrames: 1, minHandFrames: 1, idleHandFrames: 0, fill: "repeat",
  },
  40: {
    label: "x40",
    hint: "Mỗi frame dự đoán riêng: chỉ 1 frame mới nhất, lặp cho đủ cửa sổ. Nhanh nhất có thể ở 15 fps, dễ nhầm nhất, nhất là lúc chuyển ký hiệu.",
    predictFrames: 1, minFrames: 1, strideFrames: 1, minHandFrames: 1, idleHandFrames: 0, fill: "repeat",
  },
};

export interface RecognizerOptions {
  apiUrl: string;
  captureFps: number;
  predictFrames: number;
  minFrames: number;
  strideFrames: number;
  mirrorData: boolean;
  delegate: Delegate;
  numPoses: number;
  minConsecutive: number;
  idleHandFrames: number;
  minHandFrames: number;
  sentenceGapMs: number;
  fill: FillMode;
}

export const DEFAULT_OPTIONS: RecognizerOptions = {
  apiUrl: process.env.NEXT_PUBLIC_VSL_API_URL || "/vsl-api",
  captureFps: 15,
  predictFrames: SPEED_PRESETS.accurate.predictFrames,
  minFrames: SPEED_PRESETS.accurate.minFrames,
  strideFrames: SPEED_PRESETS.accurate.strideFrames,
  mirrorData: false,
  delegate: "GPU",
  numPoses: 1,
  minConsecutive: 2,
  idleHandFrames: 3,
  minHandFrames: SPEED_PRESETS.accurate.minHandFrames,
  sentenceGapMs: 3000,
  fill: "interp",
};

const MAX_STREAM_FRAMES = 60;
const MAX_WINDOW_FRAMES = 30;
const OLD_SERVER = "Server chưa nhận landmark: cập nhật và chạy lại src/app/server.py";

export interface RecognizerState {
  phase: Phase;
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
  facing: Facing;
  cameraCount: number;
  switching: boolean;
  latencyMs: number | null;
  detectMs: number | null;
  captureFps: number | null;
  delegate: Delegate | null;
  memoryExemplars: number | null;
  capture: { done: number; total: number } | null;
  memoryError: string | null;
  error: string | null;
}

export interface CommittedWindow {
  word: string;
  frames: FrameLandmarks[];
  confidence: number;
  memory?: { label: string; similarity: number };
}

interface Capture {
  frames: Frame[];
  total: number;
  resolve: (frames: FrameLandmarks[]) => void;
  reject: (err: Error) => void;
}

interface Frame {
  lm: FrameLandmarks;
  t: number;
}

export interface RecognizerEvents {
  onState: (state: RecognizerState) => void;
  onWords: (words: string[]) => void;
  onSentenceEnd: (words: string[]) => void;
  onSentenceWindows?: (windows: CommittedWindow[]) => void;
  onWordCommitted?: (window: CommittedWindow) => void;
  onWordsDiscarded?: (windows: CommittedWindow[]) => void;
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
  private timer: ReturnType<typeof setTimeout> | null = null;
  private loopId = 0;
  private frameInterval: number | null = null;
  private lastFrameT: number | null = null;
  private detector: LandmarkDetector | null = null;
  private debugCanvas: HTMLCanvasElement | null = null;
  private buffer: Frame[] = [];
  private lastPredictAt: number | null = null;
  private generation = 0;
  private inFlight: AbortController | null = null;
  private idleSince: number | null = null;
  private sessionId: string | null = null;
  private serverBuffered = 0;
  private resetNext = false;
  private lastFrameAt: number | null = null;
  private memory: MemoryPayload | null = null;
  private capture: Capture | null = null;
  private windowLandmarks: Frame[] = [];
  private committedWindow: CommittedWindow | null = null;
  private sentenceWindows: CommittedWindow[] = [];
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
      facing: "user",
      cameraCount: 0,
      switching: false,
      latencyMs: null,
      detectMs: null,
      captureFps: null,
      delegate: null,
      memoryExemplars: null,
      capture: null,
      memoryError: null,
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
      this.inFlight?.abort();
      this.closeSession();
      this.resetWindow();
    } else if (mirrorChanged && this.running) {
      this.resetWindow();
    }
    this.emit({ required: this.options.minConsecutive });
    this.emitBuffered();
  }

  setMemory(memory: MemoryPayload | null): void {
    this.memory = memory;
    if (!this.running || !this.sessionId) return;
    if (this.memorySync) {
      this.memoryDirty = true;
      return;
    }
    void this.syncMemory(this.sessionId);
  }

  private openedWith: MemoryPayload | null = null;
  private memorySync = false;
  private memoryDirty = false;

  private async syncMemory(sessionId: string): Promise<void> {
    this.memorySync = true;
    try {
      do {
        this.memoryDirty = false;
        const controller = new AbortController();
        try {
          const body = await this.post<{ memory: { exemplars: number } | null }>(
            `/api/stream/${encodeURIComponent(sessionId)}/memory`, { memory: this.memory }, controller.signal,
          );
          if (this.sessionId === sessionId) this.emit({ memoryExemplars: body.memory?.exemplars ?? null, memoryError: null });
        } catch (err) {
          if (this.sessionId !== sessionId) return;
          if (err instanceof HttpError && err.status === 400) {
            this.emit({ memoryError: err.message });
          } else if (err instanceof HttpError && (err.status === 404 || err.status === 405)) {
            this.inFlight?.abort();
            this.closeSession();
            this.resetWindow();
            this.emitBuffered();
            return;
          }
        }
      } while (this.memoryDirty && this.sessionId === sessionId);
    } finally {
      this.memorySync = false;
    }
  }

  captureSample(total: number): Promise<FrameLandmarks[]> {
    if (!this.running || !this.timer) return Promise.reject(new Error("Bật camera trước khi ghi mẫu"));
    this.capture?.reject(new Error("Đã bắt đầu lần ghi khác"));
    return new Promise((resolve, reject) => {
      this.capture = { frames: [], total, resolve, reject };
      this.voter.interrupt();
      this.emit({ capture: { done: 0, total }, candidate: null, count: 0 });
    });
  }

  cancelCapture(): void {
    this.endCapture(new Error("Đã huỷ ghi mẫu"));
  }

  get lastCommitted(): CommittedWindow | null {
    return this.committedWindow;
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.emit({ phase: "starting", error: null });
    if (!this.detector) {
      try {
        this.detector = await LandmarkDetector.create(this.options.delegate, this.options.numPoses);
        this.detector.setDebugCanvas(this.debugCanvas);
      } catch (err) {
        this.emit({ phase: "stopped", error: `Không tải được MediaPipe: ${(err as Error).message}` });
        return;
      }
      this.emit({ delegate: this.detector.delegate });
    }
    try {
      await this.openCamera();
    } catch (err) {
      this.emit({ phase: "stopped", error: `Không mở được camera: ${(err as Error).message}` });
      return;
    }
    if (this.state.phase !== "starting") return;
    this.resetWindow();
    this.emit({ phase: "buffering" });
    this.restartTimer();
  }

  async switchCamera(): Promise<void> {
    if (!this.running || this.state.phase === "starting" || this.state.switching) return;
    const previous = this.state.facing;
    this.emit({ switching: true, facing: previous === "user" ? "environment" : "user" });
    try {
      await this.openCamera();
    } catch (err) {
      this.emit({ facing: previous, error: `Không mở được camera ${previous === "user" ? "sau" : "trước"}: ${(err as Error).message}` });
      if (this.running) await this.openCamera().catch(() => undefined);
    } finally {
      this.emit({ switching: false });
    }
    if (!this.running) return;
    this.detector?.reset();
    this.endCapture(new Error("Đã đổi camera"));
    this.resetWindow();
    this.voter.interrupt();
    this.emit({ candidate: null, count: 0 });
    this.emitBuffered();
  }

  private async openCamera(): Promise<void> {
    const mobile = typeof matchMedia !== "undefined" && matchMedia("(pointer: coarse)").matches;
    const facing = this.state.facing;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    let stream: MediaStream;
    if (mobile) {
      const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { ...size, facingMode: { exact: facing } }, audio: false });
      } catch (err) {
        if ((err as Error).name !== "OverconstrainedError") throw err;
        stream = await navigator.mediaDevices.getUserMedia({ video: { ...size, facingMode: facing }, audio: false });
      }
    } else {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 3840 }, height: { ideal: 2160 }, aspectRatio: { ideal: 16 / 9 } },
        audio: false,
      });
    }
    if (!this.running) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    this.stream = stream;
    this.video.srcObject = stream;
    await this.video.play().catch(() => undefined);
    const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
    this.emit({
      resolution: `${this.video.videoWidth}×${this.video.videoHeight}`,
      cameraCount: devices.filter((d) => d.kind === "videoinput").length,
    });
  }

  stop(): void {
    this.endCapture(new Error("Camera đã tắt"));
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.loopId += 1;
    this.inFlight?.abort();
    this.inFlight = null;
    this.closeSession();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.resetWindow();
    this.voter.interrupt();
    this.idleSince = null;
    this.emit({ phase: "stopped", inFlight: false, candidate: null, count: 0, buffered: 0, memoryExemplars: null });
  }

  finishSentence(): string[] {
    const words = [...this.voter.sequence];
    const windows = this.sentenceWindows.filter((w, i) => w.word === words[i]);
    this.sentenceWindows = [];
    if (windows.length > 0) this.events.onSentenceWindows?.(windows);
    this.voter.clear();
    this.idleSince = null;
    this.emitWords();
    return words;
  }

  setDebugCanvas(canvas: HTMLCanvasElement | null): void {
    this.debugCanvas = canvas;
    this.detector?.setDebugCanvas(canvas);
  }

  dispose(): void {
    this.stop();
    this.detector?.close();
    this.detector = null;
  }

  undo(): void {
    if (this.voter.undo() !== null) {
      const removed = this.sentenceWindows.pop();
      if (removed) this.events.onWordsDiscarded?.([removed]);
      this.emitWords();
    }
  }

  clear(): void {
    this.voter.clear();
    if (this.sentenceWindows.length > 0) this.events.onWordsDiscarded?.(this.sentenceWindows);
    this.sentenceWindows = [];
    this.resetWindow();
    this.emitWords();
  }

  private get base(): string {
    return this.options.apiUrl.replace(/\/+$/, "");
  }

  private restartTimer(): void {
    if (this.timer) clearTimeout(this.timer);
    const id = ++this.loopId;
    this.frameInterval = null;
    this.lastFrameT = null;
    const loop = () => {
      if (id !== this.loopId) return;
      const started = performance.now();
      this.tick();
      if (id !== this.loopId || !this.timer) return;
      const wait = 1000 / this.options.captureFps - (performance.now() - started);
      this.timer = setTimeout(loop, Math.max(0, wait));
    };
    this.timer = setTimeout(loop, 0);
  }

  private get frameMs(): number {
    return 1000 / this.options.captureFps;
  }

  private resetWindow(): void {
    this.generation += 1;
    this.buffer = [];
    this.windowLandmarks = [];
    this.lastPredictAt = null;
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
    const frame = this.grab();
    if (frame) this.measureFps(frame.t);
    if (this.capture) {
      if (frame) this.captureFrame(frame);
      return;
    }
    this.checkSentenceGap(performance.now());
    if (frame) this.addFrame(frame);
  }

  private measureFps(t: number): void {
    if (this.lastFrameT !== null) {
      const dt = t - this.lastFrameT;
      this.frameInterval = this.frameInterval === null ? dt : this.frameInterval * 0.9 + dt * 0.1;
      this.emit({ captureFps: Math.round(10000 / this.frameInterval) / 10 });
    }
    this.lastFrameT = t;
  }

  private captureFrame(frame: Frame): void {
    const capture = this.capture!;
    capture.frames.push(frame);
    const span = frame.t - capture.frames[0].t;
    const done = Math.min(capture.total, Math.round(span / this.frameMs) + 1);
    this.emit({ capture: { done, total: capture.total } });
    if (done >= capture.total) this.endCapture(null);
  }

  private endCapture(err: Error | null): void {
    const capture = this.capture;
    if (!capture) return;
    this.capture = null;
    if (err) capture.reject(err);
    else capture.resolve(capture.frames.map((f) => f.lm));
    this.resetWindow();
    this.idleSince = null;
    this.emit({ capture: null });
    this.emitBuffered();
  }

  private addFrame(frame: Frame): void {
    this.windowLandmarks.push(frame);
    const oldest = frame.t - (MAX_WINDOW_FRAMES - 0.5) * this.frameMs;
    while (this.windowLandmarks.length > 1 && this.windowLandmarks[0].t < oldest) this.windowLandmarks.shift();
    this.buffer.push(frame);
    if (this.buffer.length > MAX_STREAM_FRAMES) this.buffer.splice(0, this.buffer.length - MAX_STREAM_FRAMES);
    this.lastPredictAt ??= frame.t - this.frameMs;
    this.emitBuffered();
    this.maybePredict();
  }

  private emitBuffered(): void {
    const { predictFrames, minFrames } = this.options;
    const target = Math.min(minFrames, predictFrames);
    this.emit({ buffered: Math.min(target, this.serverBuffered + this.buffer.length), windowFrames: target });
  }

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
    const frames = this.buffer;
    const last = frames[frames.length - 1].t;
    const due = this.lastPredictAt === null || last - this.lastPredictAt >= (this.options.strideFrames - 0.5) * this.frameMs;
    this.buffer = [];
    if (due) this.lastPredictAt = last;
    void this.run((signal) => this.predictStream(frames, due, signal));
  }

  private async run(request: (signal: AbortSignal) => Promise<StreamResponse | null>): Promise<void> {
    const controller = new AbortController();
    const generation = this.generation;
    this.inFlight = controller;
    const started = performance.now();
    try {
      const body = await request(controller.signal);
      if (body && generation === this.generation && this.running && !this.capture) {
        this.handle(body, performance.now() - started);
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        this.emit({ error: `Lỗi gọi API dự đoán: ${(err as Error).message}` });
      }
    } finally {
      if (this.inFlight === controller) this.inFlight = null;
      if (this.running && this.timer) this.maybePredict();
    }
  }

  private async post<T>(path: string, body: unknown, signal: AbortSignal): Promise<T> {
    const response = await fetch(`${this.base}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new HttpError(data.error || `HTTP ${response.status}`, response.status);
    return data as T;
  }

  private async predictStream(frames: Frame[], predict: boolean, signal: AbortSignal): Promise<StreamResponse | null> {
    if (!this.sessionId) {
      let opened: { session_id: string; window: number; landmarks?: boolean; memory?: { exemplars: number } | null };
      try {
        opened = await this.openStream(signal);
      } catch (err) {
        if (err instanceof HttpError && (err.status === 404 || err.status === 405)) throw new Error(OLD_SERVER);
        throw err;
      }
      this.emit({ memoryExemplars: opened.memory?.exemplars ?? null });
      if (opened.landmarks !== true) {
        void fetch(`${this.base}/api/stream/${encodeURIComponent(opened.session_id)}`, { method: "DELETE" }).catch(() => undefined);
        throw new Error(OLD_SERVER);
      }
      this.sessionId = opened.session_id;
      if (this.memory !== this.openedWith) this.setMemory(this.memory);
      this.serverBuffered = 0;
      this.resetNext = false;
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
      fill: this.options.fill,
      fps: this.options.captureFps,
      top_k: 3,
    };
    this.lastFrameAt = frames[frames.length - 1].t;
    try {
      return await this.post<StreamResponse>(`/api/stream/${encodeURIComponent(this.sessionId)}`, payload, signal);
    } catch (err) {
      if (err instanceof HttpError && err.status === 404) {
        this.sessionId = null;
        this.serverBuffered = 0;
        return null;
      }
      if ((err as Error).name !== "AbortError") this.resetNext ||= reset;
      throw err;
    }
  }

  private async openStream(signal: AbortSignal) {
    type Opened = { session_id: string; window: number; landmarks?: boolean; memory?: { exemplars: number } | null };
    this.openedWith = this.memory;
    if (!this.memory) return this.post<Opened>("/api/stream", {}, signal);
    try {
      const opened = await this.post<Opened>("/api/stream", { memory: this.memory }, signal);
      this.emit({ memoryError: null });
      return opened;
    } catch (err) {
      if (!(err instanceof HttpError) || err.status !== 400) throw err;
      const opened = await this.post<Opened>("/api/stream", {}, signal);
      this.emit({ memoryError: err.message });
      return opened;
    }
  }

  private handle(response: StreamResponse, requestMs: number): void {
    const { idleHandFrames, minHandFrames } = this.options;
    const now = performance.now();
    const latencyMs = this.lastFrameAt !== null ? now - this.lastFrameAt : requestMs;
    this.serverBuffered = response.buffered;
    if (response.ready && !response.predicted) {
      this.emitBuffered();
      return;
    }

    if (!response.ready) {
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
      const since = (this.windowLandmarks.at(-1)?.t ?? 0) - (this.options.predictFrames - 0.5) * this.frameMs;
      this.committedWindow = {
        word: committed, frames: this.windowLandmarks.filter((f) => f.t >= since).map((f) => f.lm),
        confidence: prediction.confidence, memory: prediction.memory,
      };
      this.sentenceWindows.push(this.committedWindow);
      this.events.onWordCommitted?.(this.committedWindow);
      const extra = this.sentenceWindows.length - this.voter.sequence.length;
      if (extra > 0) this.sentenceWindows.splice(0, extra);
      this.resetWindow();
      this.emitBuffered();
      this.emitWords();
    } else {
      this.checkSentenceGap(now);
    }
  }

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
