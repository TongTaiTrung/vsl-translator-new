# `external/vsl-web`

Web Next.js (React, không thư viện UI) nhận diện VSL từ webcam. Chi tiết vận hành: `external/vsl-web/README.md`.

| Đường dẫn | Vai trò |
|---|---|
| `app/page.tsx` | trang chính: camera, phụ đề, cài đặt, hội thoại |
| `app/api/compose/route.ts` | gọi Groq ghép câu (khoá do người dùng nhập trong Cài đặt) |
| `lib/landmarks.ts` | MediaPipe chạy trên trình duyệt (`@mediapipe/tasks-vision` 1.1.1), đổi toạ độ về canvas 1280×1280 |
| `lib/subject.ts` | `MainSubjectIsolator`: chỉ giữ người ký chính (FaceDetector + crop) |
| `lib/recognizer.ts` | mở phiên `/api/stream`, gửi landmark từng frame, `SPEED_PRESETS` |
| `lib/voter.ts` | bỏ phiếu, port từ `src/app/voting.py` |
| `lib/sentence.ts` | ghép câu, port từ `src/app/sentence.py` |
| `lib/memory.ts` | mẫu cá nhân lưu ở trình duyệt, gửi lên `/api/memory/sample` ([[Module-model]]) |
| `scripts/copy-mediapipe.mjs`, `scripts/dev-lan.mjs` | chép wasm/model; dev server HTTPS cho điện thoại cùng LAN |

Chạy: `npm install && npm run dev` trong thư mục này, sau khi chạy Flask API ([[Workflow-Web-API]]). Kiểm tra kiểu: `npm run typecheck`.
