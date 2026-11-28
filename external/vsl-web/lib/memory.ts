/** Personal sign memory stored in the browser and sent to the API. */

import type { FrameLandmarks } from "./landmarks";

export interface Sample {
  id: string;
  label: string;
  features: string;
  frames: number;
  featureDim: number;
  handFrames: number;
  createdAt: number;
  source?: SampleSource;
}

export type SampleSource = "manual" | "auto";

export interface MemoryPayload {
  strength: number;
  exemplars: { label: string; features: string; frames: number }[];
}

export const MAX_PER_LABEL = 5;
export const MAX_AUTO_PER_LABEL = 3;

export const sourceOf = (sample: Sample): SampleSource => sample.source ?? "manual";

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
  const database = await open();
  try {
    return await new Promise<T | undefined>((resolve, reject) => {
      const transaction = database.transaction(STORE, mode);
      const request = work(transaction.objectStore(STORE));
      transaction.oncomplete = () => resolve(request ? request.result : undefined);
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally {
    database.close();
  }
}

export async function listSamples(): Promise<Sample[]> {
  const all = (await run<Sample[]>("readonly", (store) => store.getAll())) ?? [];
  return all.sort((first, second) => first.createdAt - second.createdAt);
}

export function evictionFor(existing: Sample[], sample: Sample): Sample[] | null {
  const same = existing.filter((sample) => sample.label === sample.label).sort((first, second) => first.createdAt - second.createdAt);
  const autos = same.filter((sample) => sourceOf(sample) === "auto");
  const manuals = same.filter((sample) => sourceOf(sample) === "manual");
  const drop: Sample[] = [];
  if (sourceOf(sample) === "auto") {
    if (manuals.length >= MAX_PER_LABEL) return null;
    const room = Math.min(MAX_AUTO_PER_LABEL, MAX_PER_LABEL - manuals.length);
    drop.push(...autos.slice(0, Math.max(0, autos.length - room + 1)));
  } else {
    const over = same.length - MAX_PER_LABEL + 1;
    drop.push(...[...autos, ...manuals].slice(0, Math.max(0, over)));
  }
  return drop;
}

export async function addSample(sample: Sample): Promise<boolean> {
  const drop = evictionFor(await listSamples(), sample);
  if (drop === null) return false;
  await run("readwrite", (store) => {
    drop.forEach((sample) => store.delete(sample.id));
    store.put(sample);
  });
  return true;
}

export async function deleteSample(id: string): Promise<void> {
  await run("readwrite", (store) => { store.delete(id); });
}

export async function deleteLabel(label: string): Promise<void> {
  const ids = (await listSamples()).filter((sample) => sample.label === label).map((sample) => sample.id);
  await run("readwrite", (store) => ids.forEach((id) => store.delete(id)));
}

export async function clearSamples(): Promise<void> {
  await run("readwrite", (store) => store.clear());
}

export function memoryPayload(samples: Sample[], strength: number, featureDim: number | undefined): MemoryPayload | null {
  const usable = samples.filter((sample) => featureDim === undefined || sample.featureDim === featureDim);
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
