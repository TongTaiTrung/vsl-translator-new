
import { type AssetUrls, type Delegate, MainSubjectIsolator, type Point } from "./subject";

export type { Delegate };
export const FRAME_SIZE = 1280;
const LOCAL_ASSETS = "/mediapipe";
const TASKS_VISION_VERSION = "1.1.1";
const MODELS = "https://storage.googleapis.com/mediapipe-models";
const CDN_ASSETS: AssetUrls = {
  wasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`,
  face: `${MODELS}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite`,
  pose: `${MODELS}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`,
  hand: `${MODELS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
};
const USE_CDN = process.env.NEXT_PUBLIC_MEDIAPIPE_CDN !== "0";
const CONFIDENCE = 0.5;

export interface FrameLandmarks {
  pose: number[][] | null;
  hands: number[][][];
  hands_world?: number[][][];
}

const round = (v: number) => Math.round(v * 1e6) / 1e6;
const xyz = (points: { x: number; y: number; z: number }[]) => points.map((p) => [round(p.x), round(p.y), round(p.z)]);

export class LandmarkDetector {
  private lastTimestamp = -1;

  private constructor(private isolator: MainSubjectIsolator) {}

  get delegate(): Delegate {
    return this.isolator.delegate;
  }

  static async create(delegate: Delegate = "GPU", numPoses = 1): Promise<LandmarkDetector> {
    const create = (assets: string | AssetUrls) =>
      MainSubjectIsolator.create({ assets, delegate, numPoses, confidence: CONFIDENCE });
    if (!USE_CDN) return new LandmarkDetector(await create(LOCAL_ASSETS));
    try {
      return new LandmarkDetector(await create(CDN_ASSETS));
    } catch (err) {
      console.warn("Không tải được MediaPipe từ CDN, dùng bản local:", err);
      return new LandmarkDetector(await create(LOCAL_ASSETS));
    }
  }

  setDebugCanvas(canvas: HTMLCanvasElement | null): void {
    this.isolator.setDebugCanvas(canvas);
  }

  detect(video: HTMLVideoElement, timestamp: number, mirror: boolean): FrameLandmarks {
    const t = Math.max(Math.round(timestamp), this.lastTimestamp + 1);
    this.lastTimestamp = t;
    const frame = this.isolator.process(video, t, mirror);
    if (!frame) return { pose: null, hands: [] };

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

  reset(): void {
    this.isolator.reset();
  }

  close(): void {
    this.isolator.close();
  }
}
