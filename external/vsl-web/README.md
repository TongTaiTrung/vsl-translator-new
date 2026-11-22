# VSL Web

Web app Next.js (HTML/CSS thuần, không thư viện UI) để nhận diện ngôn ngữ ký hiệu Việt Nam từ webcam,
tự ghép các từ thành câu hoàn chỉnh bằng Groq với API key của chính người dùng.

## Chạy

1. Chạy Flask API dự đoán ở root dự án (cần checkpoint trong `runs/*/ensemble.pt`):

   ```bash
   .venv/bin/python -m src.app.server --port 8000
   ```

2. Chạy web app:

   ```bash
   cd external/vsl-web
   npm install
   npm run dev
   ```

3. Mở http://localhost:3000, bấm nút tròn **Bắt đầu phiên dịch** trong khung camera. Groq API key (không bắt
   buộc) nhập trong **Cài đặt** (biểu tượng bánh răng góc trên); không có key thì các từ được ghép thô.

Địa chỉ API mặc định là `http://127.0.0.1:8000`; đổi bằng `NEXT_PUBLIC_VSL_API_URL` (xem `.env.example`)
hoặc ngay trong phần Cài đặt.

## Cách hoạt động

- Webcam mở ở độ phân giải 16:9 lớn nhất (giống `src/features/camera.py`).
- **MediaPipe chạy trên trình duyệt** (`lib/landmarks.ts`, `@mediapipe/tasks-vision` cùng phiên bản 1.1.1 với
  MediaPipe Python, chế độ VIDEO, đúng 2 file model `.task` của `src/builtin/mediapipe_builtin/models`, thêm
  `blaze_face_short_range.tflite`).
  **Chỉ nhận diện người chính** (`lib/subject.ts`, `MainSubjectIsolator`): FaceDetector chạy mỗi 4 frame, chọn mặt
  to và gần giữa khung nhất (bỏ mặt rộng < 5% khung), nội suy khung crop đầu + thân trên + sải tay, làm mượt EMA;
  không thấy mặt thì dùng vùng giữa khung. Pose + hand chạy trên ảnh crop đó (cạnh dài ≤ 512 px); tay (tối đa 4)
  được khoá theo cổ tay của pose chính (trong 0.45 × bề rộng vai), các tay khác bị bỏ. `numPoses` mặc định 1
  (`RecognizerOptions.numPoses`, 2 để loại cả người đi ngang phía sau trong vùng crop). Bật "Hiện khung crop người
  chính (debug)" trong Cài đặt để xem khung: liền = khung đã làm mượt, đứt = vùng thực đưa vào model, xanh dương = mặt
  (xanh lá = đã khoá mặt, cam = vùng mặc định).
  Toạ độ crop được đổi về canvas **1280×1280** giống hệt `src/features/frame.py:frame_padding`
  (scale đều cho cạnh dài = 1280, đặt giữa, nền đen; lật gương trước nếu bật "Lật ảnh trước khi nhận diện")
  để toạ độ landmark chuẩn hoá cùng tỉ lệ như lúc trích đặc trưng train. Chỉ landmark (pose 33 điểm, 0–2 tay
  21 điểm + world landmark, ~2.7 KB JSON/frame) được gửi lên API, không gửi ảnh; server chỉ chạy `extract` và
  model. Mặc định dùng GPU (WebGL, ~34 ms/frame trên máy dev; CPU/WASM ~57 ms), GPU lỗi thì tự chuyển CPU.
- WASM và model được `scripts/copy-mediapipe.mjs` chép vào `public/mediapipe/` (không commit) tự động trước
  `npm run dev` / `npm run build`; chạy tay bằng `npm run mediapipe`. Lần mở camera đầu tiên tải ~17 MB model.
- Thu 15 fps, cửa sổ dự đoán 30 frame (2 giây). App mở một phiên `POST /api/stream` (phản hồi phải có
  `landmarks: true`) rồi gửi **từng frame ngay khi detect xong** tới `POST /api/stream/<id>` dạng JSON
  `{ landmarks, timestamps, predict, ... }`. Chỉ mỗi `strideFrames` frame mới gửi `predict: true`; các request
  còn lại chỉ thêm frame vào cửa sổ (`predicted: false`). Server giữ cửa sổ 30 frame; phiên nhận landmark không
  tạo landmarker MediaPipe trên server. Preflight CORS của JSON được cache 1 ngày (`Access-Control-Max-Age`),
  server bật HTTP/1.1 keep-alive. Định dạng landmark: `src/features/landmarks.py`.
- Độ trễ đo trên máy dev (từ lúc chụp frame tới lúc có kết quả, gồm MediaPipe trên trình duyệt): ~50 ms; server
  chỉ còn ~18 ms mỗi request dự đoán. Server mặc định chạy model trên CPU 1 luồng (`--device cpu
  --torch-threads 1`). Luôn chỉ một request đang chạy; HUD trên video hiện độ trễ (di chuột để xem thời gian
  MediaPipe mỗi frame).
- **Tốc độ nhận diện** (Cài đặt, `SPEED_PRESETS` trong `lib/recognizer.ts`): đánh đổi độ chính xác lấy tốc độ.
  Server dự đoán trên `window` frame gần nhất ngay khi có `min_frames` frame (`predict_raw` nội suy về 30).
  Thời gian từ lúc bắt đầu ký tới lúc chốt từ, đo trên trình duyệt:

  | Mức | Cửa sổ | Bắt đầu dự đoán | Gửi mỗi | Chốt từ sau |
  |---|---|---|---|---|
  | Chuẩn | 30 frame (2 s) | 30 frame | 5 frame | ~1.5 s |
  | Nhanh | 20 frame | 12 frame | 3 frame | ~1.0 s |
  | Rất nhanh | 15 frame | 8 frame | 2 frame | ~0.8 s |

  Cửa sổ ngắn hơn thì model thấy ít ký hiệu hơn, dễ nhầm hơn lúc đưa tay lên hoặc chuyển ký hiệu. Giảm
  `Số phiếu để chốt từ` xuống 1 còn nhanh hơn nữa.
- Mỗi phản hồi là một phiếu (`lib/voter.ts`, port từ `src/app/voting.py`): cùng nhãn đủ `Số phiếu để chốt từ`
  lần liên tiếp thì chốt từ. `hand_frames` trong phản hồi phân biệt nghỉ tay / đang vào ký hiệu / đang ký.
- Nghỉ tay liên tục 3 giây (chỉnh được) sau khi đã có từ thì kết thúc câu. Câu được ghép bằng Groq qua route
  `app/api/compose` (prompt port từ `src/app/sentence.py`); không có key hoặc Groq lỗi thì ghép thô.

## Groq API key

Key nằm trên trình duyệt người dùng và được gửi kèm từng yêu cầu ghép câu trong header `x-groq-key`.
Route `/api/compose` chỉ chuyển tiếp nó tới Groq, không lưu và không ghi log. Key chỉ được lưu vào
`localStorage` khi người dùng tích **Ghi nhớ key trên trình duyệt này**.

## Lệnh khác

```bash
npm run typecheck   # kiểm tra kiểu
npm run build       # build production
npm start           # chạy bản build
```
