// Chạy MediaPipe (pose + hand landmarker, chế độ VIDEO) ngay trên trình duyệt thay vì gửi ảnh lên server.
//
// Trước khi detect, frame camera được vẽ vào canvas FRAME_SIZE x FRAME_SIZE giống hệt
// src/features/frame.py:frame_padding (scale đều cho cạnh dài = 1280, đặt giữa, nền đen) nên toạ độ
// chuẩn hoá cùng tỉ lệ x/y như lúc trích đặc trưng train. mirrorData lật ảnh trước khi pad (giống server).
// Model .task là đúng file của Python (scripts/copy-mediapipe.mjs chép vào public/mediapipe/).
//
// Kết quả mỗi frame đúng định dạng src/features/landmarks.py: { pose, hands, hands_world }.

import type { HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

export const FRAME_SIZE = 1280;
const ASSETS = "/mediapipe";
/** Giống LandmarkerLoader (Python): detection / presence / tracking confidence. */
const CONFIDENCE = 0.5;
const NUM_HANDS = 2;

export interface FrameLandmarks {
  /** 33 điểm [x, y, z, visibility]; null = không thấy người. */
  pose: number[][] | null;
  /** 0–2 tay, mỗi tay 21 điểm [x, y, z] (toạ độ ảnh chuẩn hoá). */
  hands: number[][][];
  /** Cùng số tay, toạ độ mét (hand_world_landmarks); thiếu thì server dùng landmark ảnh cho thủ hình. */
  hands_world?: number[][][];
}

export type Delegate = "GPU" | "CPU";

/** 6 chữ số thập phân: sai lệch feature ~1e-4, nhỏ hơn nhiều nhiễu MediaPipe; JSON gọn hơn ~2 lần. */
const round = (v: number) => Math.round(v * 1e6) / 1e6;
const xyz = (points: { x: number; y: number; z: number }[]) => points.map((p) => [round(p.x), round(p.y), round(p.z)]);

export class LandmarkDetector {
  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  private lastTimestamp = -1;

  private constructor(
    private pose: PoseLandmarker,
    private hand: HandLandmarker,
    readonly delegate: Delegate,
  ) {
    this.canvas = typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(FRAME_SIZE, FRAME_SIZE)
      : Object.assign(document.createElement("canvas"), { width: FRAME_SIZE, height: FRAME_SIZE });
    this.ctx = this.canvas.getContext("2d", { alpha: false }) as CanvasRenderingContext2D;
  }

  /** Tải WASM + 2 model (lần đầu ~17 MB). GPU lỗi (không có WebGL2…) thì tự chuyển CPU. */
  static async create(delegate: Delegate = "GPU"): Promise<LandmarkDetector> {
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(`${ASSETS}/wasm`);
    const build = async (d: Delegate) => {
      const pose = await vision.PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${ASSETS}/pose_landmarker_full.task`, delegate: d },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: CONFIDENCE,
        minPosePresenceConfidence: CONFIDENCE,
        minTrackingConfidence: CONFIDENCE,
      });
      try {
        const hand = await vision.HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${ASSETS}/hand_landmarker.task`, delegate: d },
          runningMode: "VIDEO",
          numHands: NUM_HANDS,
          minHandDetectionConfidence: CONFIDENCE,
          minHandPresenceConfidence: CONFIDENCE,
          minTrackingConfidence: CONFIDENCE,
        });
        return new LandmarkDetector(pose, hand, d);
      } catch (err) {
        pose.close();
        throw err;
      }
    };
    if (delegate === "CPU") return build("CPU");
    try {
      return await build("GPU");
    } catch (err) {
      console.warn("MediaPipe GPU lỗi, chuyển sang CPU:", err);
      return build("CPU");
    }
  }

  /** Pad frame về 1280x1280 (lật gương nếu mirror) rồi detect; `timestamp` ms, tự đảm bảo tăng ngặt. */
  detect(video: HTMLVideoElement, timestamp: number, mirror: boolean): FrameLandmarks {
    const w = video.videoWidth;
    const h = video.videoHeight;
    // frame_padding: scale = 1280 / cạnh dài, kích thước làm tròn, lề trên/trái = phần nguyên của nửa phần dư.
    const scale = FRAME_SIZE / Math.max(w, h);
    const nw = Math.max(1, Math.min(FRAME_SIZE, Math.round(w * scale)));
    const nh = Math.max(1, Math.min(FRAME_SIZE, Math.round(h * scale)));
    const left = Math.floor((FRAME_SIZE - nw) / 2);
    const top = Math.floor((FRAME_SIZE - nh) / 2);
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, FRAME_SIZE, FRAME_SIZE);
    if (mirror) ctx.setTransform(-1, 0, 0, 1, left + nw, top);   // cv2.flip(frame, 1) rồi pad
    else ctx.setTransform(1, 0, 0, 1, left, top);
    ctx.drawImage(video, 0, 0, nw, nh);

    const t = Math.max(Math.round(timestamp), this.lastTimestamp + 1);   // VIDEO mode: timestamp tăng ngặt
    this.lastTimestamp = t;
    const pose = this.pose.detectForVideo(this.canvas, t);
    const hand = this.hand.detectForVideo(this.canvas, t);

    const person = pose.landmarks[0];
    const hands = hand.landmarks.slice(0, NUM_HANDS);
    const world = hand.worldLandmarks ?? [];
    return {
      pose: person ? person.map((p) => [round(p.x), round(p.y), round(p.z), round(p.visibility ?? 1)]) : null,
      hands: hands.map(xyz),
      ...(world.length >= hands.length ? { hands_world: hands.map((_, i) => xyz(world[i])) } : {}),
    };
  }

  close(): void {
    this.pose.close();
    this.hand.close();
  }
}
