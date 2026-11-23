// Mẫu ký cá nhân (bộ nhớ của mô hình, src/model/memory.py), lưu trong IndexedDB của trình duyệt này.
//
// - Ghi mẫu: client gửi landmark của một lần ký tới POST /api/memory/sample; server trích feature thô,
//   kiểm tra tay có xuất hiện đủ lâu, trả feature (float32 base64) để lưu ở đây. Server không giữ mẫu.
// - Khi mở phiên /api/stream, recognizer gửi kèm các mẫu (lib/recognizer.ts: setMemory); server chỉ giữ
//   chúng trong phiên đó. Xoá ở đây là xoá hẳn.
// - Lưu feature thô (không lưu embedding) nên vẫn dùng được khi đổi checkpoint, miễn cùng số cột feature
//   (`featureDim`, so với /api/health); lớp không còn trong checkpoint bị server bỏ qua.
// - Hai nguồn mẫu: "manual" (người dùng ghi / sửa nhãn) và "auto" (tự học từ câu người dùng giữ lại, không
//   xoá/undo). Mẫu tự động không bao giờ đẩy mẫu thủ công ra, và mỗi từ có tối đa MAX_AUTO_PER_LABEL mẫu tự động.

import type { FrameLandmarks } from "./landmarks";

export interface Sample {
  id: string;
  label: string;
  /** Feature thô float32 little-endian, base64; (frames, featureDim). */
  features: string;
  frames: number;
  featureDim: number;
  handFrames: number;
  createdAt: number;
  /** Mẫu cũ (trước khi có tự học) không có trường này: coi là "manual". */
  source?: SampleSource;
}

export type SampleSource = "manual" | "auto";

export interface MemoryPayload {
  strength: number;
  exemplars: { label: string; features: string; frames: number }[];
}

/** Số mẫu tối đa mỗi từ (giống MAX_PER_CLASS của server): thêm mẫu mới thì bỏ mẫu cũ nhất. */
export const MAX_PER_LABEL = 5;
/** Số mẫu tự động tối đa mỗi từ: luôn chừa chỗ cho mẫu thủ công, và tự học sai không chiếm hết bộ nhớ. */
export const MAX_AUTO_PER_LABEL = 3;

export const sourceOf = (s: Sample): SampleSource => s.source ?? "manual";

const DB_NAME = "vsl-web";
const STORE = "samples";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const store = request.result.createObjectStore(STORE, { keyPath: "id" });
      store.createIndex("label", "label");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function run<T>(mode: IDBTransactionMode, work: (store: IDBObjectStore) => IDBRequest<T> | void): Promise<T | undefined> {
  const db = await open();
  try {
    return await new Promise<T | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const request = work(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(request ? request.result : undefined);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

export async function listSamples(): Promise<Sample[]> {
  const all = (await run<Sample[]>("readonly", (store) => store.getAll())) ?? [];
  return all.sort((a, b) => a.createdAt - b.createdAt);
}

/** Những mẫu phải xoá để thêm `sample`; null = không thêm được (mẫu tự động khi từ đã đầy mẫu thủ công). */
export function evictionFor(existing: Sample[], sample: Sample): Sample[] | null {
  const same = existing.filter((s) => s.label === sample.label).sort((a, b) => a.createdAt - b.createdAt);
  const autos = same.filter((s) => sourceOf(s) === "auto");
  const manuals = same.filter((s) => sourceOf(s) === "manual");
  const drop: Sample[] = [];
  if (sourceOf(sample) === "auto") {
    if (manuals.length >= MAX_PER_LABEL) return null;
    const room = Math.min(MAX_AUTO_PER_LABEL, MAX_PER_LABEL - manuals.length);
    drop.push(...autos.slice(0, Math.max(0, autos.length - room + 1)));
  } else {
    // Mẫu thủ công: đẩy mẫu tự động cũ nhất ra trước, rồi mới tới mẫu thủ công cũ nhất.
    const over = same.length - MAX_PER_LABEL + 1;
    drop.push(...[...autos, ...manuals].slice(0, Math.max(0, over)));
  }
  return drop;
}

/** Thêm một mẫu theo luật của `evictionFor`; trả false nếu không thêm. */
export async function addSample(sample: Sample): Promise<boolean> {
  const drop = evictionFor(await listSamples(), sample);
  if (drop === null) return false;
  await run("readwrite", (store) => {
    drop.forEach((s) => store.delete(s.id));
    store.put(sample);
  });
  return true;
}

export async function deleteSample(id: string): Promise<void> {
  await run("readwrite", (store) => { store.delete(id); });
}

export async function deleteLabel(label: string): Promise<void> {
  const ids = (await listSamples()).filter((s) => s.label === label).map((s) => s.id);
  await run("readwrite", (store) => ids.forEach((id) => store.delete(id)));
}

export async function clearSamples(): Promise<void> {
  await run("readwrite", (store) => store.clear());
}

export function memoryPayload(samples: Sample[], strength: number, featureDim: number | undefined): MemoryPayload | null {
  const usable = samples.filter((s) => featureDim === undefined || s.featureDim === featureDim);
  if (usable.length === 0) return null;
  return { strength, exemplars: usable.map(({ label, features, frames }) => ({ label, features, frames })) };
}

export interface SampleResult {
  ok: boolean;
  reason: "few_hands" | null;
  warning: "model_disagrees" | null;
  label: string;
  features: string;
  frames: number;
  hand_frames: number;
  prediction: { top: { label: string; confidence: number }[] } | null;
}

/** Gửi landmark của một lần ký mẫu; server trả feature để lưu (chưa lưu gì ở đây). */
export async function submitSample(apiUrl: string, label: string, landmarks: FrameLandmarks[]): Promise<SampleResult> {
  const response = await fetch(`${apiUrl.replace(/\/+$/, "")}/api/memory/sample`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, landmarks }),
  });
  const body = await response.json().catch(() => ({}));
  if (response.status === 404 || response.status === 405) {
    throw new Error("Server chưa hỗ trợ cá nhân hoá: cập nhật và chạy lại src/app/server.py");
  }
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body as SampleResult;
}

export function newSampleId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
