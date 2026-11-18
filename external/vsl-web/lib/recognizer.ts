// Nhận diện liên tục từ webcam bằng Flask API (/api/predict nhận đúng 30 frame).
//
// - Thu frame ở 15 fps (nhịp lúc thu dữ liệu) vào cửa sổ trượt 30 frame = 2 giây (độ dài một mẫu train).
// - Luôn chỉ có MỘT request đang chạy; khi nó trả về và đã có thêm `strideFrames` frame mới thì gửi
//   cửa sổ mới nhất. Mỗi phản hồi là một phiếu cho WordVoter.
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

export type Phase = "stopped" | "starting" | "buffering" | "idle" | "entering" | "signing";

export interface RecognizerOptions {
  apiUrl: string;
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

export class Recognizer {
  private options: RecognizerOptions;
  private voter: WordVoter;
  private stream: MediaStream | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private buffer: Frame[] = [];
  private sinceSend = 0;
  private generation = 0;
  private inFlight: AbortController | null = null;
  private idleSince: number | null = null;
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
    this.options = { ...this.options, ...partial };
    this.voter.minConsecutive = this.options.minConsecutive;
    if (this.timer && partial.captureFps !== undefined) this.restartTimer();
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
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.resetWindow();
    this.voter.interrupt();
    this.idleSince = null;
    this.emit({ phase: "stopped", inFlight: false, candidate: null, count: 0 });
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

  private restartTimer(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 1000 / this.options.captureFps);
  }

  private resetWindow(): void {
    this.generation += 1;          // phản hồi của cửa sổ cũ sẽ bị bỏ qua
    this.buffer = [];
    this.sinceSend = 0;
  }

  private tick(): void {
    this.checkSentenceGap(performance.now());
    const frame = this.grab();
    if (!frame) return;
    this.buffer.push(frame);
    if (this.buffer.length > this.options.windowFrames) {
      this.buffer.splice(0, this.buffer.length - this.options.windowFrames);
    }
    this.sinceSend += 1;
    this.emit({ buffered: this.buffer.length });
    this.maybePredict();
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
    const { windowFrames, strideFrames } = this.options;
    if (this.inFlight || this.buffer.length < windowFrames || this.sinceSend < strideFrames) return;
    const frames = this.buffer.slice(-windowFrames);
    const span = frames[frames.length - 1].t - frames[0].t;
    const fps = span > 0 ? Math.min(240, Math.max(1, ((frames.length - 1) * 1000) / span)) : this.options.captureFps;
    this.sinceSend = 0;
    void this.predict(frames.map((f) => f.data), fps, this.generation);
  }

  private async predict(frames: string[], fps: number, generation: number): Promise<void> {
    const controller = new AbortController();
    this.inFlight = controller;
    this.emit({ inFlight: true });
    const started = performance.now();
    try {
      const response = await fetch(`${this.options.apiUrl.replace(/\/+$/, "")}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames, fps, mirror_data: this.options.mirrorData, top_k: 3 }),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
      if (generation === this.generation && this.running) {
        this.handle(body as Prediction, performance.now() - started);
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

  private handle(prediction: Prediction, latencyMs: number): void {
    const { idleHandFrames, minHandFrames } = this.options;
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

    const now = performance.now();
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
