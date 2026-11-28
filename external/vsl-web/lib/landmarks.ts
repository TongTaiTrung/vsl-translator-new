/** Browser-side MediaPipe pose and hand landmarking, mapped to the 1280x1280 padded canvas. */

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

const round = (value: number) => Math.round(value * 1e6) / 1e6;
const xyz = (points: { x: number; y: number; z: number }[]) => points.map((point) => [round(point.x), round(point.y), round(point.z)]);

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
    const monotonicTimestamp = Math.max(Math.round(timestamp), this.lastTimestamp + 1);
    this.lastTimestamp = monotonicTimestamp;
    const frame = this.isolator.process(video, monotonicTimestamp, mirror);
    if (!frame) return { pose: null, hands: [] };

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    const scale = FRAME_SIZE / Math.max(videoWidth, videoHeight);
    const scaledWidth = Math.max(1, Math.min(FRAME_SIZE, Math.round(videoWidth * scale)));
    const scaledHeight = Math.max(1, Math.min(FRAME_SIZE, Math.round(videoHeight * scale)));
    const offsetX = Math.floor((FRAME_SIZE - scaledWidth) / 2) / FRAME_SIZE;
    const offsetY = Math.floor((FRAME_SIZE - scaledHeight) / 2) / FRAME_SIZE;
    const scaleX = scaledWidth / FRAME_SIZE;
    const scaleY = scaledHeight / FRAME_SIZE;
    const toFrame = (point: Point) => [round(offsetX + point.x * scaleX), round(offsetY + point.y * scaleY), round(point.z * scaleX)];

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
      pose: frame.pose ? frame.pose.map((point) => [...toFrame(point), round(point.visibility)]) : null,
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
