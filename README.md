# VSL translator

Dịch Ngôn ngữ ký hiệu Việt Nam: video BGR → pad 1280×1280 → MediaPipe pose + hand → 383 đặc trưng/frame → chuỗi `(30, 434)` → ensemble phân loại có từ chối → bỏ phiếu → câu tiếng Việt.

Tài liệu đầy đủ là vault Obsidian ở [`vault/`](vault/Home.md) (mở thư mục `vault/` bằng Obsidian).

## Cài đặt

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```

## Quy trình

Chạy các notebook trong `notebooks/` theo thứ tự: `data_collect` → `reextract_features` (sau khi đổi extractor) → `augment_features` (tuỳ chọn) → `train` → `realtime_predict`.

## Dự đoán bằng Python

```python
from src.model.predict import VSLPredictor

predictor = VSLPredictor()  # runs/best/ensemble.pt hoặc truyền đường dẫn
result = predictor.predict_raw(raw_features)  # (frames, 383) float32
result = predictor.predict_frames(frames_bgr, fps=15)
# {"label", "confidence", "accepted", "probability_gap", "rejection_reason", "top": [...]}
```

Luồng liên tục: `predictor.open_stream(StreamConfig(...))` trả `VSLStreamSession`; mỗi client/camera một phiên.

## Chạy ứng dụng

```bash
.venv/bin/python -m src.app            # webcam
.venv/bin/python -m src.app.server     # API cho external/vsl-web
```

## Test

```bash
.venv/bin/python -m unittest tests.test_predict tests.test_reextract tests.test_feature_augmentation tests.test_training_robustness tests.test_builtin_collection tests.test_training_p1 tests.test_evaluation tests.test_feature_v3 tests.test_app tests.test_fast tests.test_server tests.test_memory tests.test_camera tests.test_view tests.test_checkpoints tests.test_still_negatives
```
