// Nhận diện liên tục từ webcam: MediaPipe chạy trên trình duyệt, chỉ gửi landmark lên Flask API.
//
// - Thu frame ở 15 fps (nhịp lúc thu dữ liệu); cửa sổ dự đoán 30 frame = 2 giây (độ dài một mẫu train).
//   Máy chậm (điện thoại: MediaPipe lâu hơn 1 nhịp) thu được ít fps hơn: mọi thứ tính theo THỜI GIAN chứ không
//   theo số frame — server giữ cửa sổ 2 giây theo timestamp, nhịp dự đoán / ghi mẫu / cửa sổ đã chốt cũng theo
//   ms — nên model luôn thấy đúng 2 giây cử chỉ như lúc train. Vòng thu tự giãn (setTimeout sau mỗi frame,
//   không dồn tick) để vẫn nhường event loop cho mạng và giao diện.
// - Mỗi frame chạy pose + hand landmarker trên ảnh crop quanh người chính (lib/subject.ts), toạ độ đổi về canvas
//   pad 1280x1280 giống frame_padding của Python (lib/landmarks.ts, chế độ VIDEO giữ tracking giữa các frame). Chỉ landmark (~3 KB JSON/frame) đi lên server,
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
  /** Có khi phiên dùng mẫu cá nhân: từ có mẫu giống cửa sổ này nhất và độ giống (cos, 1 = giống hệt). */
  memory?: { label: string; similarity: number };
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
/** Camera trước (người dùng tự ký) / sau (quay người khác ký). Cả hai đều cho ảnh KHÔNG lật: tay phải người ký
 *  nằm bên trái ảnh, đúng như webcam lúc thu dữ liệu, nên mirrorData không đổi theo camera. */
export type Facing = "user" | "environment";
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
  /** numPoses của PoseLandmarker trên ảnh crop người chính (lib/subject.ts): 1 = tracking ổn định như lúc thu
   *  dữ liệu, 2 = loại được người đi ngang phía sau. Chỉ áp dụng khi tải MediaPipe (start() đầu tiên). */
  numPoses: number;
  minConsecutive: number;
  /** <= số này frame có tay: người dùng đang nghỉ. */
  idleHandFrames: number;
  /** >= số này frame có tay: đủ để bỏ phiếu. */
  minHandFrames: number;
  /** Nghỉ tay bao lâu (ms) thì kết thúc câu; 0 = tắt. */
  sentenceGapMs: number;
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
};

/** Tối đa số frame một request /api/stream (giống MAX_STREAM_FRAMES của server). */
const MAX_STREAM_FRAMES = 60;
/** Cửa sổ dự đoán dài nhất (predictFrames của preset chuẩn = 30 frame = 2 giây). */
const MAX_WINDOW_FRAMES = 30;
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
  /** Camera đang dùng (điện thoại). */
  facing: Facing;
  /** Số camera của máy (biết sau khi được cấp quyền); >= 2 thì cho đổi camera. */
  cameraCount: number;
  /** Đang đổi camera. */
  switching: boolean;
  latencyMs: number | null;
  /** Thời gian MediaPipe trên trình duyệt cho frame gần nhất (ms). */
  detectMs: number | null;
  /** Nhịp thu thật (fps, trung bình trượt): < captureFps nghĩa là máy không kịp chạy MediaPipe mỗi nhịp. */
  captureFps: number | null;
  /** Delegate MediaPipe đang chạy (GPU / CPU), null khi chưa tải. */
  delegate: Delegate | null;
  /** Số mẫu cá nhân phiên server đang dùng; null = phiên không dùng mẫu (hoặc chưa mở phiên). */
  memoryExemplars: number | null;
  /** Đang ghi mẫu cá nhân: số frame đã thu / cần thu; null = không ghi. */
  capture: { done: number; total: number } | null;
  /** Server từ chối bộ mẫu cá nhân (phiên chạy mô hình gốc). */
  memoryError: string | null;
  error: string | null;
}

/** Từ vừa chốt kèm landmark cửa sổ đã sinh ra nó (để người dùng sửa nhãn và lưu làm mẫu). */
export interface CommittedWindow {
  word: string;
  frames: FrameLandmarks[];
  /** Xác suất của từ ở phiếu chốt. */
  confidence: number;
  /** Mẫu cá nhân giống cửa sổ này nhất (nếu phiên dùng mẫu). */
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
  /** Câu vừa kết thúc: cửa sổ của các từ người dùng giữ lại (không bị xoá/undo) — dùng để tự học mẫu. */
  onSentenceWindows?: (windows: CommittedWindow[]) => void;
  /** Vừa chốt một từ (tự học thời gian thực). */
  onWordCommitted?: (window: CommittedWindow) => void;
  /** Người dùng hoàn tác / xoá các từ này: mẫu đã học từ chúng phải bỏ. */
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
  /** Vòng thu frame (setTimeout nối tiếp); null = đã dừng. */
  private timer: ReturnType<typeof setTimeout> | null = null;
  private loopId = 0;
  /** Khoảng cách giữa hai frame thu được (ms, trung bình trượt) để báo fps thật. */
  private frameInterval: number | null = null;
  private lastFrameT: number | null = null;
  private detector: LandmarkDetector | null = null;
  private debugCanvas: HTMLCanvasElement | null = null;
  /** Các frame (landmark) chưa gửi lên server. */
  private buffer: Frame[] = [];
  /** Thời điểm (ms) của frame cuối trong request dự đoán gần nhất; null = cửa sổ vừa xoá. */
  private lastPredictAt: number | null = null;
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
  /** Mẫu cá nhân gửi kèm khi mở phiên; null = mô hình gốc. */
  private memory: MemoryPayload | null = null;
  private capture: Capture | null = null;
  /** Landmark của cửa sổ hiện tại (từ lần xoá cửa sổ gần nhất), tối đa MAX_WINDOW_FRAMES nhịp (2 giây). */
  private windowLandmarks: Frame[] = [];
  private committedWindow: CommittedWindow | null = null;
  /** Cửa sổ của từng từ trong câu hiện tại, cùng thứ tự với voter.sequence. */
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

  /** Đổi bộ mẫu cá nhân (null = tắt). Phiên đang mở được cập nhật tại chỗ (POST /api/stream/<id>/memory,
   *  không mất cửa sổ đang ký); server cũ không có endpoint đó thì đóng phiên, phiên kế tiếp mở với mẫu mới. */
  setMemory(memory: MemoryPayload | null): void {
    this.memory = memory;
    if (!this.running || !this.sessionId) return;      // phiên kế tiếp tự mở với this.memory
    if (this.memorySync) {
      this.memoryDirty = true;                          // gửi bản mới nhất khi lần cập nhật trước xong
      return;
    }
    void this.syncMemory(this.sessionId);
  }

  /** Bộ mẫu đã gửi lúc mở phiên hiện tại. */
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
            // Phiên hết hạn hoặc server cũ: mở phiên mới (kèm mẫu) ở request kế tiếp.
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

  /** Thu landmark `total` frame kế tiếp làm mẫu (không dự đoán trong lúc thu). Camera phải đang chạy. */
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

  /** Từ chốt gần nhất và cửa sổ landmark của nó; null nếu chưa chốt từ nào. */
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
    if (this.state.phase !== "starting") return;      // đã bấm dừng trong lúc chờ quyền camera
    this.resetWindow();
    this.emit({ phase: "buffering" });
    this.restartTimer();
  }

  /** Đổi camera trước ↔ sau khi đang chạy. Cửa sổ ký hiệu và khung người chính bắt đầu lại từ đầu. */
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

  /** Mở camera theo state.facing (thay stream cũ nếu có) và gắn vào <video>.
   *  Máy tính: độ phân giải 16:9 lớn nhất camera hỗ trợ (giống src/features/camera.py). Điện thoại: 720p — model chỉ
   *  nhận ảnh crop ≤ 512 px quanh người, ảnh 4K chỉ làm chậm bước cắt ảnh + FaceDetector (fps thấp mới là thứ làm
   *  nhận diện kém). Frame nào cũng được pad về 1280x1280. */
  private async openCamera(): Promise<void> {
    const mobile = typeof matchMedia !== "undefined" && matchMedia("(pointer: coarse)").matches;
    const facing = this.state.facing;
    // iOS chỉ mở được một camera một lúc: tắt stream cũ trước khi xin camera kia.
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
    if (!this.running) {                                  // đã dừng trong lúc chờ camera
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

  /** Lấy chuỗi từ hiện tại và bắt đầu câu mới (dùng khi bấm "Kết thúc câu"). */
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

  /** Bật (canvas overlay phủ lên <video>) / tắt (null) vẽ khung crop người chính để kiểm tra. */
  setDebugCanvas(canvas: HTMLCanvasElement | null): void {
    this.debugCanvas = canvas;
    this.detector?.setDebugCanvas(canvas);
  }

  /** Giải phóng MediaPipe (khi gỡ component); start() sau đó sẽ tải lại. */
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

  // ------------------------------------------------------------------ nội bộ

  private get base(): string {
    return this.options.apiUrl.replace(/\/+$/, "");
  }

  /** Vòng thu: nhịp kế tiếp tính từ lúc BẮT ĐẦU frame trước. Máy nhanh giữ đúng captureFps; máy chậm (MediaPipe
   *  lâu hơn một nhịp) thu liên tục nhưng không dồn tick như setInterval, và luôn nhường event loop một lượt
   *  (phản hồi mạng, vẽ giao diện) giữa hai frame. */
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
    this.generation += 1;          // phản hồi của cửa sổ cũ sẽ bị bỏ qua
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

  /** Mẫu cá nhân = `total` nhịp (2 giây với 30 frame ở 15 fps) tính theo thời gian, máy chậm thì ít frame hơn. */
  private captureFrame(frame: Frame): void {
    const capture = this.capture!;
    capture.frames.push(frame);
    const span = frame.t - capture.frames[0].t;
    const done = Math.min(capture.total, Math.round(span / this.frameMs) + 1);
    this.emit({ capture: { done, total: capture.total } });
    if (done >= capture.total) this.endCapture(null);
  }

  /** Kết thúc lần ghi mẫu: err = null là thu đủ. Cửa sổ dự đoán bắt đầu lại từ đầu sau đó. */
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
    // Giữ các frame chưa gửi (tối đa một request); server chậm thì bỏ frame cũ nhất.
    if (this.buffer.length > MAX_STREAM_FRAMES) this.buffer.splice(0, this.buffer.length - MAX_STREAM_FRAMES);
    // Frame đầu của cửa sổ mới tính như đã dự đoán một nhịp trước nó (giống đếm frame: dự đoán ở frame thứ stride).
    this.lastPredictAt ??= frame.t - this.frameMs;
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
    const frames = this.buffer;
    const last = frames[frames.length - 1].t;
    // Dự đoán mỗi strideFrames nhịp THỜI GIAN (máy chậm không phải chờ đủ strideFrames frame).
    const due = this.lastPredictAt === null || last - this.lastPredictAt >= (this.options.strideFrames - 0.5) * this.frameMs;
    this.buffer = [];
    if (due) this.lastPredictAt = last;
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
      if (body && generation === this.generation && this.running && !this.capture) {
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
      // Bộ mẫu đổi trong lúc đang mở phiên: cập nhật ngay cho phiên mới.
      if (this.memory !== this.openedWith) this.setMemory(this.memory);
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

  /** Mở phiên kèm mẫu cá nhân; server từ chối mẫu (400) thì báo lỗi và mở phiên không mẫu. */
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
      const since = (this.windowLandmarks.at(-1)?.t ?? 0) - (this.options.predictFrames - 0.5) * this.frameMs;
      this.committedWindow = {
        word: committed, frames: this.windowLandmarks.filter((f) => f.t >= since).map((f) => f.lm),
        confidence: prediction.confidence, memory: prediction.memory,
      };
      this.sentenceWindows.push(this.committedWindow);
      this.events.onWordCommitted?.(this.committedWindow);
      // Giữ cùng độ dài với voter.sequence (voter cắt bớt từ đầu khi quá maxWords).
      const extra = this.sentenceWindows.length - this.voter.sequence.length;
      if (extra > 0) this.sentenceWindows.splice(0, extra);
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
