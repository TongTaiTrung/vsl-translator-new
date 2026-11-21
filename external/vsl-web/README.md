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

- Webcam mở ở độ phân giải 16:9 lớn nhất (giống `src/features/camera.py`); frame được thu nhỏ về cạnh dài
  1280 rồi nén JPEG bất đồng bộ (`OffscreenCanvas.convertToBlob` / `canvas.toBlob`) để không chặn luồng
  giao diện.
- Thu 15 fps, cửa sổ dự đoán 30 frame (2 giây). App mở một phiên `POST /api/stream` (phản hồi có
  `binary: true`) rồi gửi **từng frame ngay khi chụp** tới `POST /api/stream/<id>` dạng `multipart/form-data`
  (mỗi frame một phần `frames` + chuỗi JSON `meta`), không base64. Chỉ mỗi `strideFrames` frame mới gửi
  `predict: true`; các request còn lại chỉ trích đặc trưng (`predicted: false` trong phản hồi). Server giữ
  tracking MediaPipe và cửa sổ 30 frame, mỗi frame chỉ trích một lần. Multipart là “simple request” nên không
  có preflight CORS; preflight của JSON được cache 1 ngày (`Access-Control-Max-Age`), và server bật HTTP/1.1
  keep-alive.
- Độ trễ đo trên máy dev (từ lúc frame được chụp tới lúc có kết quả, request có dự đoán): **~53 ms trung vị,
  p90 ~89 ms**, so với ~197 ms trước đây (gửi lô 5 frame base64 và chạy model trên MPS). Server mặc định chạy
  model trên CPU 1 luồng (`--device cpu --torch-threads 1`): ensemble nhỏ nên CPU 1 luồng (~13 ms) nhanh hơn
  MPS (~33–70 ms) và CPU nhiều luồng (~59 ms). Đổi bằng `--device mps` hoặc `--torch-threads 0` (mặc định
  PyTorch) nếu dùng model lớn hơn.
- Server cũ không có `/api/stream` thì app tự gửi cả cửa sổ tới `POST /api/predict`; server không trả
  `binary` thì app gửi JSON base64. Luôn chỉ một request đang chạy; HUD trên video hiện độ trễ.
- **Tốc độ nhận diện** (Cài đặt, `SPEED_PRESETS` trong `lib/recognizer.ts`): đánh đổi độ chính xác lấy tốc độ.
  Server dự đoán trên `window` frame gần nhất ngay khi có `min_frames` frame (`predict_raw` nội suy về 30).
  Thời gian từ lúc bắt đầu ký tới lúc chốt từ, đo trên trình duyệt:

  | Mức | Cửa sổ | Bắt đầu dự đoán | Gửi mỗi | Chốt từ sau |
  |---|---|---|---|---|
  | Chuẩn | 30 frame (2 s) | 30 frame | 5 frame | ~1.5 s |
  | Nhanh | 20 frame | 12 frame | 3 frame | ~1.0 s |
  | Rất nhanh | 15 frame | 8 frame | 2 frame | ~0.8 s |

  Cửa sổ ngắn hơn thì model thấy ít ký hiệu hơn, dễ nhầm hơn lúc đưa tay lên hoặc chuyển ký hiệu. Giảm
  `Số phiếu để chốt từ` xuống 1 còn nhanh hơn nữa. Chế độ `window` (server cũ) luôn dùng 30 frame.
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
