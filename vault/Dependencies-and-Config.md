# Thư viện và cấu hình

## Python (`requirements.txt`, Python 3.12)

| Gói | Dùng ở |
|---|---|
| `mediapipe==1.1.1` | `src/builtin/mediapipe_builtin` |
| `opencv-python` | camera, đọc video, vẽ (`cv2`) |
| `numpy`, `scipy` | toàn bộ pipeline; `scipy` trong `extract.py` |
| `torch` | huấn luyện, dự đoán, ONNX |
| `scikit-learn` | fold, metric (`training.py`, `diagnostics.py`) |
| `matplotlib` | biểu đồ trong notebook train/finetune |
| `flask` | `src/app/server.py` |
| `requests` | gọi Groq (`src/app/sentence.py`) |
| `Pillow` | hiển thị chữ có dấu trong `realtime.py` (nhập khi cần) |

`pyproject.toml` khai báo gói `src*` (cài editable: `pip install -e .`).

## Web (`external/vsl-web/package.json`)

`next`, `react`, `react-dom`, `@mediapipe/tasks-vision` 1.1.1; dev: `typescript`, `@types/*`.

## Cấu hình

| Tên | Ở đâu | Ý nghĩa |
|---|---|---|
| `GROQ_API_KEY` | `.env` hoặc môi trường (`.env.example`) | ghép câu bằng Groq cho `src.app` |
| `VSL_API_ORIGIN` | `external/vsl-web/.env.example` | Next chuyển tiếp `/vsl-api/...` tới Flask (mặc định `http://127.0.0.1:8000`) |
| `NEXT_PUBLIC_VSL_API_URL` | tuỳ chọn | trình duyệt gọi thẳng API |
| `NEXT_PUBLIC_MEDIAPIPE_CDN=0` | lúc build | luôn dùng wasm/model local thay CDN |
| `.claude/launch.json` | gốc dự án | `vsl-api` (cổng 8000), `vsl-web` (3000), `vsl-web-lan` (HTTPS) |
| `runs/best/ensemble.pt` | | checkpoint mặc định của `VSLPredictor` |
