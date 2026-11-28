
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

export function assetDir(dir: string): AssetUrls {
  return {
    wasm: `${dir}/wasm`,
    face: `${dir}/blaze_face_short_range.tflite`,
    pose: `${dir}/pose_landmarker_full.task`,
    hand: `${dir}/hand_landmarker.task`,
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

const makePoints = (n: number): Point[] => Array.from({ length: n }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));

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
    const build = async (d: Delegate) => {
      const opened: { close(): void }[] = [];
      try {
        const face = await vision.FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: urls.face, delegate: d },
          runningMode: "VIDEO",
          minDetectionConfidence: confidence,
        });
        opened.push(face);
        const pose = await vision.PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: urls.pose, delegate: d },
          runningMode: "VIDEO",
          numPoses,
          minPoseDetectionConfidence: confidence,
          minPosePresenceConfidence: confidence,
          minTrackingConfidence: confidence,
        });
        opened.push(pose);
        const hand = await vision.HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: urls.hand, delegate: d },
          runningMode: "VIDEO",
          numHands,
          minHandDetectionConfidence: confidence,
          minHandPresenceConfidence: confidence,
          minTrackingConfidence: confidence,
        });
        return new MainSubjectIsolator(face, pose, hand, d, Math.max(1, faceEvery), Math.max(0, faceHold));
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
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;

    if (!this.faceLocked || this.frameIndex % this.faceEvery === 0) {
      this.locateFace(this.faceDetector.detectForVideo(video, timestamp).detections, vw, vh);
    }
    this.frameIndex += 1;
    const s = this.smooth, tg = this.target;
    if (!this.hasSmooth) {
      s.x = tg.x; s.y = tg.y; s.w = tg.w; s.h = tg.h;
      this.hasSmooth = true;
    } else {
      s.x = EMA_ALPHA * s.x + (1 - EMA_ALPHA) * tg.x;
      s.y = EMA_ALPHA * s.y + (1 - EMA_ALPHA) * tg.y;
      s.w = EMA_ALPHA * s.w + (1 - EMA_ALPHA) * tg.w;
      s.h = EMA_ALPHA * s.h + (1 - EMA_ALPHA) * tg.h;
    }

    let sw = s.w * vw;
    let sh = s.h * vh;
    if (sw < 2 || sh < 2) return null;
    const aspect = sw / sh;
    if (!this.canvasW || Math.abs(aspect / (this.canvasW / this.canvasH) - 1) > ASPECT_TOLERANCE) {
      this.canvasW = aspect >= 1 ? CROP_MAX_SIDE : Math.max(1, Math.round(CROP_MAX_SIDE * aspect));
      this.canvasH = aspect >= 1 ? Math.max(1, Math.round(CROP_MAX_SIDE / aspect)) : CROP_MAX_SIDE;
      this.canvas.width = this.canvasW;
      this.canvas.height = this.canvasH;
    }
    const cw = this.canvasW, ch = this.canvasH;
    const cx = (s.x + s.w / 2) * vw, cy = (s.y + s.h / 2) * vh;
    if (sw / sh < cw / ch) sw = sh * (cw / ch);
    else sh = sw / (cw / ch);
    const sx = cx - sw / 2, sy = cy - sh / 2;
    const ix0 = Math.max(0, sx), iy0 = Math.max(0, sy);
    const ix1 = Math.min(vw, sx + sw), iy1 = Math.min(vh, sy + sh);
    if (ix1 - ix0 < 1 || iy1 - iy0 < 1) return null;
    const kx = cw / sw, ky = ch / sh;
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cw, ch);
    if (mirror) ctx.setTransform(-1, 0, 0, 1, cw, 0);
    ctx.drawImage(video, ix0, iy0, ix1 - ix0, iy1 - iy0, (ix0 - sx) * kx, (iy0 - sy) * ky, (ix1 - ix0) * kx, (iy1 - iy0) * ky);

    const crop = this.out.crop;
    crop.x = sx / vw; crop.y = sy / vh; crop.w = sw / vw; crop.h = sh / vh;

    const poses = this.pose.detectForVideo(this.canvas, timestamp).landmarks;
    const handResult = this.hand.detectForVideo(this.canvas, timestamp);
    const hands = handResult.landmarks;
    const worlds = handResult.worldLandmarks;

    let main: NormalizedLandmark[] | null = null;
    let widest = -1;
    for (let i = 0; i < poses.length; i++) {
      const p = poses[i];
      if (p.length < POSE_POINTS) continue;
      const d = Math.hypot((p[L_SHOULDER].x - p[R_SHOULDER].x) * cw, (p[L_SHOULDER].y - p[R_SHOULDER].y) * ch);
      if (d > widest) { widest = d; main = p; }
    }

    let left = -1, right = -1;
    if (main) {
      const maxDist = widest * WRIST_RATIO;
      const lw = main[L_WRIST], rw = main[R_WRIST];
      let best = maxDist, bestHand = -1, bestLeft = true;
      for (let i = 0; i < hands.length; i++) {
        const w = hands[i][0];
        const dl = Math.hypot((w.x - lw.x) * cw, (w.y - lw.y) * ch);
        const dr = Math.hypot((w.x - rw.x) * cw, (w.y - rw.y) * ch);
        if (dl < best) { best = dl; bestHand = i; bestLeft = true; }
        if (dr < best) { best = dr; bestHand = i; bestLeft = false; }
      }
      if (bestHand >= 0) {
        const other = bestLeft ? rw : lw;
        let second = -1, secondDist = maxDist;
        for (let i = 0; i < hands.length; i++) {
          if (i === bestHand) continue;
          const w = hands[i][0];
          const d = Math.hypot((w.x - other.x) * cw, (w.y - other.y) * ch);
          if (d < secondDist) { secondDist = d; second = i; }
        }
        if (bestLeft) { left = bestHand; right = second; } else { right = bestHand; left = second; }
      }
    } else {
      let a = -1, b = -1, sa = -1, sb = -1;
      for (let i = 0; i < hands.length; i++) {
        const h = hands[i];
        const size = Math.hypot((h[MIDDLE_MCP].x - h[0].x) * cw, (h[MIDDLE_MCP].y - h[0].y) * ch);
        if (size > sa) { b = a; sb = sa; a = i; sa = size; } else if (size > sb) { b = i; sb = size; }
      }
      if (a >= 0 && b >= 0 && hands[b][0].x < hands[a][0].x) { left = b; right = a; } else { left = a; right = b; }
    }

    const ox = mirror ? 1 - crop.x - crop.w : crop.x;
    const out = this.out;
    out.pose = main ? mapPoints(main, this.posePoints, ox, crop.y, crop.w, crop.h) : null;
    out.leftHand = left >= 0 ? mapPoints(hands[left], this.leftPoints, ox, crop.y, crop.w, crop.h) : null;
    out.rightHand = right >= 0 ? mapPoints(hands[right], this.rightPoints, ox, crop.y, crop.w, crop.h) : null;
    out.leftWorld = left >= 0 ? (worlds?.[left] ?? null) : null;
    out.rightWorld = right >= 0 ? (worlds?.[right] ?? null) : null;
    out.faceLocked = this.faceLocked;

    if (this.debugCtx) this.drawDebug(vw, vh);
    return out;
  }

  close(): void {
    this.setDebugCanvas(null);
    this.faceDetector.close();
    this.pose.close();
    this.hand.close();
  }

  private locateFace(detections: Detection[], vw: number, vh: number): void {
    const half = vw / 2;
    let bestScore = -1;
    let fx = 0, fy = 0, fw = 0, fh = 0;
    for (let i = 0; i < detections.length; i++) {
      const bb = detections[i].boundingBox;
      if (!bb || bb.width < vw * MIN_FACE_FRAC || bb.height <= 0) continue;
      const center = bb.originX + bb.width / 2;
      const score = bb.width * bb.height * Math.max(MIN_CENTER_FACTOR, 1 - CENTER_WEIGHT * Math.abs(center - half) / half);
      if (score > bestScore) {
        bestScore = score;
        fx = bb.originX; fy = bb.originY; fw = bb.width; fh = bb.height;
      }
    }
    const tg = this.target;
    if (bestScore < 0) {
      this.faceMisses += 1;
      if (!this.faceLocked || this.faceMisses > this.faceHold) {
        this.faceLocked = false;
        tg.x = FALLBACK.x; tg.y = FALLBACK.y; tg.w = FALLBACK.w; tg.h = FALLBACK.h;
      }
      return;
    }
    this.faceMisses = 0;
    this.faceLocked = true;
    const face = this.face;
    face.x = fx / vw; face.y = fy / vh; face.w = fw / vw; face.h = fh / vh;

    const cx = fx + fw / 2;
    const x1 = Math.max(0, cx - (fw * CROP_WIDTH_FACES) / 2);
    const x2 = Math.min(vw, cx + (fw * CROP_WIDTH_FACES) / 2);
    const y1 = Math.max(0, fy - fh * CROP_TOP_FACES);
    const y2 = Math.min(vh, fy + fh * CROP_BOTTOM_FACES);
    if (x2 - x1 < 2 || y2 - y1 < 2) {
      tg.x = FALLBACK.x; tg.y = FALLBACK.y; tg.w = FALLBACK.w; tg.h = FALLBACK.h;
      return;
    }
    tg.x = x1 / vw; tg.y = y1 / vh; tg.w = (x2 - x1) / vw; tg.h = (y2 - y1) / vh;
  }

  private drawDebug(vw: number, vh: number): void {
    const canvas = this.debugCanvas!, ctx = this.debugCtx!;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    const scale = Math.max(w / vw, h / vh);
    const offX = (w - vw * scale) / 2, offY = (h - vh * scale) / 2;
    const px = vw * scale, py = vh * scale;
    const color = this.faceLocked ? "#22c55e" : "#f59e0b";
    ctx.lineWidth = 2 * dpr;

    const c = this.out.crop;
    ctx.strokeStyle = color;
    ctx.setLineDash(DASH);
    ctx.strokeRect(offX + c.x * px, offY + c.y * py, c.w * px, c.h * py);
    ctx.setLineDash(SOLID);
    const s = this.smooth;
    ctx.strokeRect(offX + s.x * px, offY + s.y * py, s.w * px, s.h * py);
    if (this.faceLocked) {
      const f = this.face;
      ctx.strokeStyle = "#38bdf8";
      ctx.strokeRect(offX + f.x * px, offY + f.y * py, f.w * px, f.h * py);
    }
  }
}

function mapPoints(src: NormalizedLandmark[], into: Point[], ox: number, oy: number, bw: number, bh: number): Point[] {
  const n = Math.min(src.length, into.length);
  for (let i = 0; i < n; i++) {
    const p = src[i], q = into[i];
    q.x = ox + p.x * bw;
    q.y = oy + p.y * bh;
    q.z = p.z * bw;
    q.visibility = p.visibility ?? 1;
  }
  return into;
}
