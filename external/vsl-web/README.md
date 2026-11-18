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

3. Mở http://localhost:3000, nhập Groq API key trong phần **Cài đặt** rồi bấm **Bắt đầu nhận diện**.

Địa chỉ API mặc định là `http://127.0.0.1:8000`; đổi bằng `NEXT_PUBLIC_VSL_API_URL` (xem `.env.example`)
hoặc ngay trong phần Cài đặt.

## Cách hoạt động

- Webcam mở ở độ phân giải 16:9 lớn nhất (giống `src/features/camera.py`); frame được thu nhỏ về cạnh dài
  1280 rồi gửi đi dạng JPEG.
- Thu 15 fps, cửa sổ dự đoán 30 frame (2 giây). Mặc định app mở một phiên `POST /api/stream` và mỗi
  5 frame chỉ gửi các frame MỚI tới `POST /api/stream/<id>`: server giữ tracking MediaPipe và cửa sổ 30
  frame nên mỗi frame chỉ trích một lần (~0.15 s/request trên CPU, so với ~1.5 s khi gửi cả 30 frame).
  Server cũ không có `/api/stream` thì app tự gửi cả cửa sổ tới `POST /api/predict`. Luôn chỉ một request
  đang chạy; HUD trên video hiện chế độ (`stream`/`window`) và thời gian mỗi request.
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
