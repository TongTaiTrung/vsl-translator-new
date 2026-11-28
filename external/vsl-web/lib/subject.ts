/** Main-subject isolation: crop the camera frame around the primary signer. */

import type {
  Detection,
  FaceDetector,
  HandLandmarker,
  Landmark,
  NormalizedLandmark,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

export type Delegate = "GPU" | "CPU";

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Point {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface IsolatedFrame {
  pose: Point[] | null;
  leftHand: Point[] | null;
  rightHand: Point[] | null;
  leftWorld: Landmark[] | null;
  rightWorld: Landmark[] | null;
  crop: Box;
  faceLocked: boolean;
}

export interface AssetUrls {
  wasm: string;
  face: string;
  pose: string;
  hand: string;
}

export function assetDir(directory: string): AssetUrls {
  return {
    wasm: `${directory}/wasm`,
    face: `${directory}/blaze_face_short_range.tflite`,
    pose: `${directory}/pose_landmarker_full.task`,
    hand: `${directory}/hand_landmarker.task`,
  };
}

export interface IsolatorOptions {
  assets?: string | AssetUrls;
  delegate?: Delegate;
  numPoses?: number;
  numHands?: number;
  faceEvery?: number;
  faceHold?: number;
  confidence?: number;
}

const MIN_FACE_FRAC = 0.05;
const CENTER_WEIGHT = 0.6;
const MIN_CENTER_FACTOR = 0.1;
const CROP_WIDTH_FACES = 6.2;
const CROP_TOP_FACES = 0.9;
const CROP_BOTTOM_FACES = 4.8;
const EMA_ALPHA = 0.65;
const FALLBACK: Readonly<Box> = { x: 0.15, y: 0, w: 0.7, h: 1 };
const CROP_MAX_SIDE = 512;
const ASPECT_TOLERANCE = 0.02;
const WRIST_RATIO = 0.45;
const POSE_POINTS = 33;
const HAND_POINTS = 21;
const L_SHOULDER = 11, R_SHOULDER = 12, L_WRIST = 15, R_WRIST = 16;
const MIDDLE_MCP = 9;

const DASH = [8, 6];
const SOLID: number[] = [];

type Canvas = HTMLCanvasElement | OffscreenCanvas;
type Ctx = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

const makePoints = (count: number): Point[] => Array.from({ length: count }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));

export class MainSubjectIsolator {
  private readonly canvas: Canvas;
  private readonly ctx: Ctx;
  private canvasW = 0;
  private canvasH = 0;

  private readonly target: Box = { ...FALLBACK };
  private readonly smooth: Box = { ...FALLBACK };
  private readonly face: Box = { x: 0, y: 0, w: 0, h: 0 };
  private hasSmooth = false;
  private faceLocked = false;
  private faceMisses = 0;
  private frameIndex = 0;

  private readonly posePoints = makePoints(POSE_POINTS);
  private readonly leftPoints = makePoints(HAND_POINTS);
  private readonly rightPoints = makePoints(HAND_POINTS);
  private readonly out: IsolatedFrame = {
    pose: null, leftHand: null, rightHand: null, leftWorld: null, rightWorld: null,
    crop: { x: 0, y: 0, w: 0, h: 0 }, faceLocked: false,
  };

  private debugCanvas: HTMLCanvasElement | null = null;
  private debugCtx: CanvasRenderingContext2D | null = null;

  private constructor(
    private readonly faceDetector: FaceDetector,
    private readonly pose: PoseLandmarker,
    private readonly hand: HandLandmarker,
    readonly delegate: Delegate,
    private readonly faceEvery: number,
    private readonly faceHold: number,
  ) {
    this.canvas = typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(1, 1)
      : Object.assign(document.createElement("canvas"), { width: 1, height: 1 });
    this.ctx = this.canvas.getContext("2d", { alpha: false }) as Ctx;
  }

  static async create(options: IsolatorOptions = {}): Promise<MainSubjectIsolator> {
    const {
      assets = "/mediapipe", delegate = "GPU", numPoses = 1, numHands = 4,
      faceEvery = 4, faceHold = 3, confidence = 0.5,
    } = options;
    const urls = typeof assets === "string" ? assetDir(assets) : assets;
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(urls.wasm);
    const build = async (delegate: Delegate) => {
      const opened: { close(): void }[] = [];
      try {
        const face = await vision.FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: urls.face, delegate: delegate },
          runningMode: "VIDEO",
          minDetectionConfidence: confidence,
        });
        opened.push(face);
        const pose = await vision.PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: urls.pose, delegate: delegate },
          runningMode: "VIDEO",
          numPoses,
          minPoseDetectionConfidence: confidence,
          minPosePresenceConfidence: confidence,
          minTrackingConfidence: confidence,
        });
        opened.push(pose);
        const hand = await vision.HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: urls.hand, delegate: delegate },
          runningMode: "VIDEO",
          numHands,
          minHandDetectionConfidence: confidence,
          minHandPresenceConfidence: confidence,
          minTrackingConfidence: confidence,
        });
        return new MainSubjectIsolator(face, pose, hand, delegate, Math.max(1, faceEvery), Math.max(0, faceHold));
      } catch (err) {
        for (const task of opened) task.close();
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

  setDebugCanvas(canvas: HTMLCanvasElement | null): void {
    if (this.debugCanvas && this.debugCtx && this.debugCanvas !== canvas) {
      this.debugCtx.clearRect(0, 0, this.debugCanvas.width, this.debugCanvas.height);
    }
    this.debugCanvas = canvas;
    this.debugCtx = canvas ? canvas.getContext("2d") : null;
  }

  reset(): void {
    this.hasSmooth = false;
    this.faceLocked = false;
    this.faceMisses = 0;
    this.frameIndex = 0;
  }

  process(video: HTMLVideoElement, timestamp: number, mirror: boolean): IsolatedFrame | null {
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    if (!videoWidth || !videoHeight) return null;

    if (!this.faceLocked || this.frameIndex % this.faceEvery === 0) {
      this.locateFace(this.faceDetector.detectForVideo(video, timestamp).detections, videoWidth, videoHeight);
    }
    this.frameIndex += 1;
    const smoothed = this.smooth, targetCrop = this.target;
    if (!this.hasSmooth) {
      smoothed.x = targetCrop.x; smoothed.y = targetCrop.y; smoothed.w = targetCrop.w; smoothed.h = targetCrop.h;
      this.hasSmooth = true;
    } else {
      smoothed.x = EMA_ALPHA * smoothed.x + (1 - EMA_ALPHA) * targetCrop.x;
      smoothed.y = EMA_ALPHA * smoothed.y + (1 - EMA_ALPHA) * targetCrop.y;
      smoothed.w = EMA_ALPHA * smoothed.w + (1 - EMA_ALPHA) * targetCrop.w;
      smoothed.h = EMA_ALPHA * smoothed.h + (1 - EMA_ALPHA) * targetCrop.h;
    }

    let cropWidth = smoothed.w * videoWidth;
    let cropHeight = smoothed.h * videoHeight;
    if (cropWidth < 2 || cropHeight < 2) return null;
    const aspect = cropWidth / cropHeight;
    if (!this.canvasW || Math.abs(aspect / (this.canvasW / this.canvasH) - 1) > ASPECT_TOLERANCE) {
      this.canvasW = aspect >= 1 ? CROP_MAX_SIDE : Math.max(1, Math.round(CROP_MAX_SIDE * aspect));
      this.canvasH = aspect >= 1 ? Math.max(1, Math.round(CROP_MAX_SIDE / aspect)) : CROP_MAX_SIDE;
      this.canvas.width = this.canvasW;
      this.canvas.height = this.canvasH;
    }
    const canvasWidth = this.canvasW, canvasHeight = this.canvasH;
    const centerX = (smoothed.x + smoothed.w / 2) * videoWidth, centerY = (smoothed.y + smoothed.h / 2) * videoHeight;
    if (cropWidth / cropHeight < canvasWidth / canvasHeight) cropWidth = cropHeight * (canvasWidth / canvasHeight);
    else cropHeight = cropWidth / (canvasWidth / canvasHeight);
    const sourceX = centerX - cropWidth / 2, sourceY = centerY - cropHeight / 2;
    const visibleLeft = Math.max(0, sourceX), visibleTop = Math.max(0, sourceY);
    const visibleRight = Math.min(videoWidth, sourceX + cropWidth), visibleBottom = Math.min(videoHeight, sourceY + cropHeight);
    if (visibleRight - visibleLeft < 1 || visibleBottom - visibleTop < 1) return null;
    const scaleX = canvasWidth / cropWidth, scaleY = canvasHeight / cropHeight;
    const context = this.ctx;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvasWidth, canvasHeight);
    if (mirror) context.setTransform(-1, 0, 0, 1, canvasWidth, 0);
    context.drawImage(video, visibleLeft, visibleTop, visibleRight - visibleLeft, visibleBottom - visibleTop, (visibleLeft - sourceX) * scaleX, (visibleTop - sourceY) * scaleY, (visibleRight - visibleLeft) * scaleX, (visibleBottom - visibleTop) * scaleY);

    const crop = this.out.crop;
    crop.x = sourceX / videoWidth; crop.y = sourceY / videoHeight; crop.w = cropWidth / videoWidth; crop.h = cropHeight / videoHeight;

    const poses = this.pose.detectForVideo(this.canvas, timestamp).landmarks;
    const handResult = this.hand.detectForVideo(this.canvas, timestamp);
    const hands = handResult.landmarks;
    const worlds = handResult.worldLandmarks;

    let main: NormalizedLandmark[] | null = null;
    let widest = -1;
    for (let i = 0; i < poses.length; i++) {
      const personPose = poses[i];
      if (personPose.length < POSE_POINTS) continue;
      const shoulderWidth = Math.hypot((personPose[L_SHOULDER].x - personPose[R_SHOULDER].x) * canvasWidth, (personPose[L_SHOULDER].y - personPose[R_SHOULDER].y) * canvasHeight);
      if (shoulderWidth > widest) { widest = shoulderWidth; main = personPose; }
    }

    let left = -1, right = -1;
    if (main) {
      const maxDist = widest * WRIST_RATIO;
      const leftWrist = main[L_WRIST], rightWrist = main[R_WRIST];
      let best = maxDist, bestHand = -1, bestLeft = true;
      for (let i = 0; i < hands.length; i++) {
        const handWrist = hands[i][0];
        const distanceToLeft = Math.hypot((handWrist.x - leftWrist.x) * canvasWidth, (handWrist.y - leftWrist.y) * canvasHeight);
        const distanceToRight = Math.hypot((handWrist.x - rightWrist.x) * canvasWidth, (handWrist.y - rightWrist.y) * canvasHeight);
        if (distanceToLeft < best) { best = distanceToLeft; bestHand = i; bestLeft = true; }
        if (distanceToRight < best) { best = distanceToRight; bestHand = i; bestLeft = false; }
      }
      if (bestHand >= 0) {
        const other = bestLeft ? rightWrist : leftWrist;
        let second = -1, secondDist = maxDist;
        for (let i = 0; i < hands.length; i++) {
          if (i === bestHand) continue;
          const handWrist = hands[i][0];
          const wristDistance = Math.hypot((handWrist.x - other.x) * canvasWidth, (handWrist.y - other.y) * canvasHeight);
          if (wristDistance < secondDist) { secondDist = wristDistance; second = i; }
        }
        if (bestLeft) { left = bestHand; right = second; } else { right = bestHand; left = second; }
      }
    } else {
      let largestIndex = -1, secondIndex = -1, largestSize = -1, secondSize = -1;
      for (let i = 0; i < hands.length; i++) {
        const hand = hands[i];
        const size = Math.hypot((hand[MIDDLE_MCP].x - hand[0].x) * canvasWidth, (hand[MIDDLE_MCP].y - hand[0].y) * canvasHeight);
        if (size > largestSize) { secondIndex = largestIndex; secondSize = largestSize; largestIndex = i; largestSize = size; } else if (size > secondSize) { secondIndex = i; secondSize = size; }
      }
      if (largestIndex >= 0 && secondIndex >= 0 && hands[secondIndex][0].x < hands[largestIndex][0].x) { left = secondIndex; right = largestIndex; } else { left = largestIndex; right = secondIndex; }
    }

    const offsetX = mirror ? 1 - crop.x - crop.w : crop.x;
    const output = this.out;
    output.pose = main ? mapPoints(main, this.posePoints, offsetX, crop.y, crop.w, crop.h) : null;
    output.leftHand = left >= 0 ? mapPoints(hands[left], this.leftPoints, offsetX, crop.y, crop.w, crop.h) : null;
    output.rightHand = right >= 0 ? mapPoints(hands[right], this.rightPoints, offsetX, crop.y, crop.w, crop.h) : null;
    output.leftWorld = left >= 0 ? (worlds?.[left] ?? null) : null;
    output.rightWorld = right >= 0 ? (worlds?.[right] ?? null) : null;
    output.faceLocked = this.faceLocked;

    if (this.debugCtx) this.drawDebug(videoWidth, videoHeight);
    return output;
  }

  close(): void {
    this.setDebugCanvas(null);
    this.faceDetector.close();
    this.pose.close();
    this.hand.close();
  }

  private locateFace(detections: Detection[], videoWidth: number, videoHeight: number): void {
    const half = videoWidth / 2;
    let bestScore = -1;
    let faceX = 0, faceY = 0, faceWidth = 0, faceHeight = 0;
    for (let i = 0; i < detections.length; i++) {
      const box = detections[i].boundingBox;
      if (!box || box.width < videoWidth * MIN_FACE_FRAC || box.height <= 0) continue;
      const center = box.originX + box.width / 2;
      const score = box.width * box.height * Math.max(MIN_CENTER_FACTOR, 1 - CENTER_WEIGHT * Math.abs(center - half) / half);
      if (score > bestScore) {
        bestScore = score;
        faceX = box.originX; faceY = box.originY; faceWidth = box.width; faceHeight = box.height;
      }
    }
    const targetCrop = this.target;
    if (bestScore < 0) {
      this.faceMisses += 1;
      if (!this.faceLocked || this.faceMisses > this.faceHold) {
        this.faceLocked = false;
        targetCrop.x = FALLBACK.x; targetCrop.y = FALLBACK.y; targetCrop.w = FALLBACK.w; targetCrop.h = FALLBACK.h;
      }
      return;
    }
    this.faceMisses = 0;
    this.faceLocked = true;
    const face = this.face;
    face.x = faceX / videoWidth; face.y = faceY / videoHeight; face.w = faceWidth / videoWidth; face.h = faceHeight / videoHeight;

    const faceCenterX = faceX + faceWidth / 2;
    const cropLeft = Math.max(0, faceCenterX - (faceWidth * CROP_WIDTH_FACES) / 2);
    const cropRight = Math.min(videoWidth, faceCenterX + (faceWidth * CROP_WIDTH_FACES) / 2);
    const cropTop = Math.max(0, faceY - faceHeight * CROP_TOP_FACES);
    const cropBottom = Math.min(videoHeight, faceY + faceHeight * CROP_BOTTOM_FACES);
    if (cropRight - cropLeft < 2 || cropBottom - cropTop < 2) {
      targetCrop.x = FALLBACK.x; targetCrop.y = FALLBACK.y; targetCrop.w = FALLBACK.w; targetCrop.h = FALLBACK.h;
      return;
    }
    targetCrop.x = cropLeft / videoWidth; targetCrop.y = cropTop / videoHeight; targetCrop.w = (cropRight - cropLeft) / videoWidth; targetCrop.h = (cropBottom - cropTop) / videoHeight;
  }

  private drawDebug(videoWidth: number, videoHeight: number): void {
    const canvas = this.debugCanvas!, context = this.debugCtx!;
    const dpr = window.devicePixelRatio || 1;
    const canvasWidth = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const canvasHeight = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
    }
    context.clearRect(0, 0, canvasWidth, canvasHeight);
    const scale = Math.max(canvasWidth / videoWidth, canvasHeight / videoHeight);
    const offX = (canvasWidth - videoWidth * scale) / 2, offY = (canvasHeight - videoHeight * scale) / 2;
    const scaledWidth = videoWidth * scale, scaledHeight = videoHeight * scale;
    const color = this.faceLocked ? "#22c55e" : "#f59e0b";
    context.lineWidth = 2 * dpr;

    const crop = this.out.crop;
    context.strokeStyle = color;
    context.setLineDash(DASH);
    context.strokeRect(offX + crop.x * scaledWidth, offY + crop.y * scaledHeight, crop.w * scaledWidth, crop.h * scaledHeight);
    context.setLineDash(SOLID);
    const smoothed = this.smooth;
    context.strokeRect(offX + smoothed.x * scaledWidth, offY + smoothed.y * scaledHeight, smoothed.w * scaledWidth, smoothed.h * scaledHeight);
    if (this.faceLocked) {
      const face = this.face;
      context.strokeStyle = "#38bdf8";
      context.strokeRect(offX + face.x * scaledWidth, offY + face.y * scaledHeight, face.w * scaledWidth, face.h * scaledHeight);
    }
  }
}

function mapPoints(source: NormalizedLandmark[], into: Point[], offsetX: number, offsetY: number, boxWidth: number, boxHeight: number): Point[] {
  const count = Math.min(source.length, into.length);
  for (let i = 0; i < count; i++) {
    const sourcePoint = source[i], targetPoint = into[i];
    targetPoint.x = offsetX + sourcePoint.x * boxWidth;
    targetPoint.y = offsetY + sourcePoint.y * boxHeight;
    targetPoint.z = sourcePoint.z * boxWidth;
    targetPoint.visibility = sourcePoint.visibility ?? 1;
  }
  return into;
}
