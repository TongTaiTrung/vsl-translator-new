# `src/app`

| File | Vai trò |
|---|---|
| `realtime.py` | `RealtimeTranslator`: webcam → dự đoán theo frame → bỏ phiếu → câu. Chạy `python -m src.app` |
| `fast.py` | `FastRealtimeTranslator`, `FastStreamSession`: sàng lọc bằng một model, xác nhận bằng cả ensemble trước khi chốt |
| `server.py` | HTTP API Flask cho web ([[Workflow-Web-API]]) |
| `voting.py` | `VoteConfig`, `WordVoter`: một nhãn thắng N frame liên tiếp mới vào chuỗi từ |
| `sentence.py` | `GroqClient`, `SentenceComposer`: ghép chuỗi nhãn thành câu tiếng Việt |
| `simulate.py`, `live_sim.py` | mô phỏng luồng ký, xem [[Experiment-Tools]] |
| `static/test_client.html` | trang thử API (webcam hoặc file video), phục vụ ở `GET /` |

## Realtime (`python -m src.app`)

Cờ: `--checkpoint`, `--camera`, `--resolution`, `--mirror-data`, `--no-mirror-preview`, `--min-consecutive`, `--min-confidence`, `--min-gap`, `--keep-window`, `--model`, `--no-llm`, `--display-size`, `--fast` (cùng `--fast-model`, `--early-window`, `--early-votes`, `--strong-confidence`, `--strong-gap`, `--escalate`).

## Nhánh nhanh (`fast.py`)

1. Cascade: mỗi frame chỉ chạy model toàn bộ dữ liệu; ensemble chỉ chạy lúc xác nhận.
2. Thoát sớm có điều kiện: từ `early_window_seconds` đã dự đoán nhưng chỉ phiếu "mạnh" mới tính trước `min_window_seconds`.
3. Xác nhận: ensemble chạy lại trên đúng cửa sổ và phải đồng ý nhãn.
`escalate` mặc định tắt (không cải thiện WER mà tốn ~3× số lần gọi ensemble).

## Ghép câu (`sentence.py`)

Gọi Groq (`GROQ_URL`, mặc định model `qwen/qwen3.8-27b`). Khoá lấy từ biến môi trường `GROQ_API_KEY` rồi đến `.env` ở gốc dự án (`load_api_key`). Có `naive_sentence` ghép thô làm phương án dự phòng khi LLM không dùng được.
