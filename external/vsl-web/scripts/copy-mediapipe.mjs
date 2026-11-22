// Chép WASM của @mediapipe/tasks-vision, 2 model .task dùng ở Python và blaze_face_short_range.tflite (cô lập người
// chính, lib/subject.ts) từ src/builtin/mediapipe_builtin/models
// vào public/mediapipe/ để trình duyệt chạy đúng cùng model với lúc trích đặc trưng train. Chạy tự động
// trước `npm run dev` / `npm run build` (predev / prebuild); public/mediapipe/ không commit.
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public", "mediapipe");
const wasmSrc = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const modelSrc = join(root, "..", "..", "src", "builtin", "mediapipe_builtin", "models");

function copy(from, to) {
  if (existsSync(to) && statSync(to).size === statSync(from).size && statSync(to).mtimeMs >= statSync(from).mtimeMs) return;
  mkdirSync(dirname(to), { recursive: true });
  copyFileSync(from, to);
  console.log(`mediapipe: ${to.slice(root.length + 1)}`);
}

for (const name of readdirSync(wasmSrc)) copy(join(wasmSrc, name), join(out, "wasm", name));
for (const name of ["pose_landmarker_full.task", "hand_landmarker.task", "blaze_face_short_range.tflite"]) {
  const from = join(modelSrc, name);
  if (!existsSync(from)) throw new Error(`Thiếu model ${from}`);
  copy(from, join(out, name));
}
