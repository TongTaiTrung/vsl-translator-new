# Sử dụng module `src/model`

## 1. Huấn luyện từ file `.npz`

```bash
python -m src.model.training \
  --data dataset.npz \
  --out runs/vsl \
  --folds 5 \
  --epochs 200 \
  --head cosine
```

Kết quả trong `runs/vsl/`:

```text
ensemble.pt          # checkpoint đầy đủ
classes.json         # tên lớp theo thứ tự nhãn
metrics.json         # CV accuracy, threshold, temperature, config...
vsl_classifier.onnx  # nếu xuất ONNX thành công
```

Chạy thử pipeline bằng dữ liệu giả:

```bash
python -m src.model.training --demo --out runs/demo --folds 3 --epochs 60 --no-onnx
```

Chỉ chạy một vài fold để kiểm tra nhanh:

```bash
python -m src.model.training --data dataset.npz --out runs/smoke --folds-to-run 1 --no-onnx
```

Không huấn luyện model full-data:

```bash
python -m src.model.training --data dataset.npz --out runs/cv_only --no-full
```

## 2. Huấn luyện bằng Python API

```python
import numpy as np
from src.model.training import Config, run_training

X = ...          # (N, 30, 170) float32
y = ...          # (N,) int
classes = [...]  # list[str]
groups = [...]   # (N,) video/người/lần ghi

cfg = Config(
    max_epochs=200,
    min_epochs=30,
    patience=30,
    d_model=128,
    gru_hidden=96,
    head="cosine",
    batch_size=0,       # tự chọn
    seed=42,
)

result = run_training(
    X, y, classes,
    groups=groups,
    out_dir="runs/vsl",
    cfg=cfg,
    folds=5,
    add_full=True,
    export_onnx_model=True,
)

print(result["cv_acc_mean"], result["cv_acc_std"])
print(result["temperature"], result["reject_threshold"])
```

Loại mẫu augment khỏi validation (vẫn dùng để train):

```python
result = run_training(
    X, y, classes,
    groups=groups,
    validation_mask=~augmented,   # bool (N,)
    out_dir="runs/vsl_aug",
)
```

Khi `validation_mask` không phải toàn `True`, `groups` là bắt buộc và mỗi group phải còn ít nhất một mẫu gốc trong validation.

## 3. Chuẩn bị dataset đúng định dạng

```python
from src.data.data_reader import DataHandler
from src.features.extract import process_sequence

handler = DataHandler(processor=process_sequence)
handler.save_npz("dataset.npz", group_by="sample")
```

Nếu tự tạo `.npz`:

```python
np.savez_compressed(
    "dataset.npz",
    X=X.astype("float32"),           # (N, 30, 170)
    y=y.astype("int64"),             # (N,)
    classes=np.asarray(classes),     # (C,)
    groups=np.asarray(groups),       # (N,)
)
```

## 4. Dự đoán offline

### Từ clip BGR

```python
from src.model.predict import VSLPredictor

predictor = VSLPredictor()  # mới nhất runs/*/ensemble.pt

result = predictor.predict_frames(
    frames_bgr,
    mirror_data=False,   # phải khớp lúc thu thập dữ liệu
    top_k=3,
)
print(result)
```

### Từ raw features `(T, 131)`

```python
result = predictor.predict_raw(raw_features, top_k=3)
```

### Từ processed features `(30, 170)`

```python
result = predictor.predict_processed(sequence, top_k=3)
```

### Chọn checkpoint cụ thể

```python
predictor = VSLPredictor("runs/vsl/ensemble.pt", min_confidence=0.7)
print(predictor.classes)
print(predictor.target_len, predictor.feature_dim)
print(predictor.threshold)
```

### Đọc kết quả

```python
{
    "label": "xin chào" or None,
    "confidence": 0.93,
    "accepted": True,
    "top": [
        {"label": "xin chào", "confidence": 0.93},
        {"label": "cảm ơn", "confidence": 0.04},
    ],
}
```

Nếu `accepted = False`, `label = None`; `top` không chứa mục `"unknown"`, nhưng `confidence` là xác suất cao nhất.

## 5. Dự đoán realtime

```python
from src.model.predict import VSLPredictor, StreamConfig

predictor = VSLPredictor("runs/vsl/ensemble.pt", min_confidence=0.7)

config = StreamConfig(
    window_seconds=2.0,
    min_window_seconds=1.5,
    min_frames=8,
    predict_every=3,
    stability_frames=3,
    no_hand_timeout=0.6,
    mirror_data=False,
    top_k=3,
)

with predictor.open_stream(config) as session:
    for frame in camera_frames:
        state = session.update_frame(frame)
        print(state["status"], state["stable_label"])
        if state["stable_label"]:
            print("Chốt:", state["stable_label"])
```

Hoặc tự tạo session:

```python
session = predictor.open_stream(config)
try:
    state = session.update_frame(frame_bgr)
finally:
    session.close()
```

Reset cửa sổ khi bắt đầu câu mới:

```python
session.reset()
```

### Trạng thái stream

| `status` | Ý nghĩa |
|---|---|
| `idle` | Không có tay / hết `no_hand_timeout` |
| `collecting` | Đang thu frame, chưa đủ `min_frames` hoặc `min_window_seconds` |
| `ready` | Đủ dữ liệu nhưng chưa có nhãn ổn định |
| `predicting` | Đã dự đoán, có thể có `stable_label` |

Các trường khác: `prediction` (dict hoặc `None`), `stable_label`, `hand_present`, `frames`, `duration`.

## 6. Dùng checkpoint trong web server

```python
from flask import Flask, jsonify
from src.model.predict import VSLPredictor

app = Flask(__name__)
predictor = VSLPredictor("runs/vsl/ensemble.pt")  # tạo 1 lần
sessions = {}                                     # 1 session / client

@app.post("/stream/<client_id>")
def update(client_id):
    session = sessions.setdefault(client_id, predictor.open_stream())
    state = session.update_frame(get_frame_from_request())
    return jsonify(state)

@app.delete("/stream/<client_id>")
def disconnect(client_id):
    session = sessions.pop(client_id, None)
    if session:
        session.close()
    return jsonify(ok=True)
```

Quy tắc:

- `VSLPredictor` dùng chung được giữa các request.
- `VSLStreamSession` **không** dùng chung; mỗi client/camera một session.
- Luôn `close()` session khi client ngắt kết nối.
- `mirror_data` phải giống lúc thu thập dữ liệu (webcam thường lật gương, video upload thì không).

## 7. Tham chiếu nhanh

| API | Input / tham số | Output |
|---|---|---|
| `Config(...)` | Cấu hình kiến trúc/tối ưu | dataclass |
| `run_training(...)` | `X, y, classes, groups, out_dir, folds...` | dict metrics + đường dẫn |
| `train_one(...)` | tensor train/val | `(model, info)` |
| `load_dataset(path)` | `.npz` | `X, y, classes, groups` |
| `load_ensemble(path, device)` | checkpoint | `(VSLEnsemble, checkpoint)` |
| `predict_topk(ens, X, classes, k, thr)` | `(B,T,D)` | list top-k |
| `VSLPredictor(path, device, min_confidence)` | checkpoint | predictor |
| `predict_processed(seq, top_k)` | `(30, 170)` | dict JSON |
| `predict_raw(raw, top_k)` | `(T, 131)` | dict JSON |
| `predict_frames(frames, mirror_data, top_k)` | iterable BGR | dict JSON |
| `open_stream(config)` | `StreamConfig` | `VSLStreamSession` |
| `StreamConfig(...)` | cấu hình cửa sổ | dataclass đã validate |
| `VSLStreamSession.update_frame(frame, now=None)` | BGR frame | dict trạng thái |

## 8. Lỗi thường gặp

| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `FileNotFoundError: No ensemble.pt found under runs/` | Chưa train hoặc sai thư mục; truyền đường dẫn checkpoint |
| `ValueError: Checkpoint expects D features...` | Extractor hiện tại khác checkpoint; train lại |
| `ValueError: Expected processed features (30, 170)` | Truyền nhầm raw `(T,131)`; dùng `predict_raw` |
| `ValueError: Expected nonempty raw features (frames, 131)` | Clip rỗng hoặc sai chiều |
| `ValueError: Features cannot contain infinity` | Dữ liệu hỏng |
| `IndexError` khi dự đoán bị từ chối với `top_k=1` | `predict_processed` bỏ mục `"unknown"` khỏi danh sách; dùng `top_k >= 2` |
| `ValueError: Group ... has no original validation sample` | `validation_mask` loại hết mẫu gốc của một group |
| `ValueError: Class ... needs at least k independent source groups` | Thiếu mẫu gốc độc lập; giảm `folds` hoặc thêm dữ liệu |
| CV accuracy quá cao bất thường | Thiếu `groups` hoặc trùng lặp giữa train/val |
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| ONNX export thất bại | `pip install onnx`; checkpoint `.pt` vẫn dùng được |

## 9. Quy trình đầy đủ

```text
DataHandler(processor=process_sequence)
  → save_npz("dataset.npz", group_by="sample")
  → python -m src.model.training --data dataset.npz --out runs/vsl
  → VSLPredictor("runs/vsl/ensemble.pt")
  → predict_frames(...) hoặc open_stream(...)
```
