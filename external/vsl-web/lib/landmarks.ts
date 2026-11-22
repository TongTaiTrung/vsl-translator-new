// Chạy MediaPipe (pose + hand landmarker, chế độ VIDEO) ngay trên trình duyệt thay vì gửi ảnh lên server.
//
// Nhiều người trong khung hình: lib/subject.ts (MainSubjectIsolator) chỉ giữ người chính — FaceDetector chọn
// mặt to và gần giữa nhất, pose + hand chạy trên ảnh crop quanh người đó, tay được khoá theo cổ tay pose chính.
// Isolator trả toạ độ chuẩn hoá theo video (đã lật nếu mirrorData); ở đây đổi tiếp sang toạ độ canvas
// FRAME_SIZE x FRAME_SIZE giống hệt src/features/frame.py:frame_padding (scale đều cho cạnh dài = 1280, đặt giữa,
// nền đen) nên x/y/z cùng tỉ lệ như lúc trích đặc trưng train.
// Model là đúng file của Python (scripts/copy-mediapipe.mjs chép vào public/mediapipe/), thêm blaze_face_short_range.
//
// Kết quả mỗi frame đúng định dạng src/features/landmarks.py: { pose, hands, hands_world }.

import { type Delegate, MainSubjectIsolator, type Point } from "./subject";

export type { Delegate };
export const FRAME_SIZE = 1280;
const ASSETS = "/mediapipe";
/** Giống LandmarkerLoader (Python): detection / presence / tracking confidence. */
const CONFIDENCE = 0.5;

export interface FrameLandmarks {
  /** 33 điểm [x, y, z, visibility]; null = không thấy người. */
  pose: number[][] | null;
  /** 0–2 tay, mỗi tay 21 điểm [x, y, z] (toạ độ ảnh chuẩn hoá). */
  hands: number[][][];
  /** Cùng số tay, toạ độ mét (hand_world_landmarks); thiếu thì server dùng landmark ảnh cho thủ hình. */
  hands_world?: number[][][];
}

/** 6 chữ số thập phân: sai lệch feature ~1e-4, nhỏ hơn nhiều nhiễu MediaPipe; JSON gọn hơn ~2 lần. */
const round = (v: number) => Math.round(v * 1e6) / 1e6;
const xyz = (points: { x: number; y: number; z: number }[]) => points.map((p) => [round(p.x), round(p.y), round(p.z)]);

export class LandmarkDetector {
  private lastTimestamp = -1;

  private constructor(private isolator: MainSubjectIsolator) {}

  get delegate(): Delegate {
    return this.isolator.delegate;
  }

  /** Tải WASM + 3 model (lần đầu ~17 MB). GPU lỗi (không có WebGL2…) thì tự chuyển CPU. */
  static async create(delegate: Delegate = "GPU", numPoses = 1): Promise<LandmarkDetector> {
    return new LandmarkDetector(await MainSubjectIsolator.create({ assets: ASSETS, delegate, numPoses, confidence: CONFIDENCE }));
  }

  /** Overlay vẽ khung crop (debug); null = tắt. */
  setDebugCanvas(canvas: HTMLCanvasElement | null): void {
    this.isolator.setDebugCanvas(canvas);
  }

  /** Detect người chính rồi đổi toạ độ về canvas pad 1280x1280; `timestamp` ms, tự đảm bảo tăng ngặt. */
  detect(video: HTMLVideoElement, timestamp: number, mirror: boolean): FrameLandmarks {
    const t = Math.max(Math.round(timestamp), this.lastTimestamp + 1);   // VIDEO mode: timestamp tăng ngặt
    this.lastTimestamp = t;
    const frame = this.isolator.process(video, t, mirror);
    if (!frame) return { pose: null, hands: [] };

    // frame_padding: scale = 1280 / cạnh dài, kích thước làm tròn, lề trên/trái = phần nguyên của nửa phần dư.
    // Lật gương (cv2.flip rồi pad) đã nằm sẵn trong toạ độ của isolator.
    const w = video.videoWidth;
    const h = video.videoHeight;
    const scale = FRAME_SIZE / Math.max(w, h);
    const nw = Math.max(1, Math.min(FRAME_SIZE, Math.round(w * scale)));
    const nh = Math.max(1, Math.min(FRAME_SIZE, Math.round(h * scale)));
    const ox = Math.floor((FRAME_SIZE - nw) / 2) / FRAME_SIZE;
    const oy = Math.floor((FRAME_SIZE - nh) / 2) / FRAME_SIZE;
    const kx = nw / FRAME_SIZE;
    const ky = nh / FRAME_SIZE;
    const toFrame = (p: Point) => [round(ox + p.x * kx), round(oy + p.y * ky), round(p.z * kx)];

    const hands: number[][][] = [];
    const world: number[][][] = [];
    let worldOk = true;
    for (const [hand, handWorld] of [[frame.leftHand, frame.leftWorld], [frame.rightHand, frame.rightWorld]] as const) {
      if (!hand) continue;
      hands.push(hand.map(toFrame));
      if (handWorld) world.push(xyz(handWorld));
      else worldOk = false;
    }
    return {
      pose: frame.pose ? frame.pose.map((p) => [...toFrame(p), round(p.visibility)]) : null,
      hands,
      ...(worldOk && hands.length ? { hands_world: world } : {}),
    };
  }

  close(): void {
    this.isolator.close();
  }
}
