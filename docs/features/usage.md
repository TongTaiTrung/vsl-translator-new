# Sử dụng module `src/features`

## Yêu cầu

```bash
pip install numpy scipy opencv-python
```

## 1. Chuẩn hoá khung hình

```python
from src.features.frame import FRAME_SIZE, frame_padding

padded = frame_padding(bgr_frame)
assert padded.shape == (FRAME_SIZE, FRAME_SIZE, 3)
assert padded.dtype.name == "uint8"
```

`frame_padding` scale đều để cạnh dài bằng 1280 (thu nhỏ hoặc phóng to); ảnh được đặt giữa, phần thừa màu đen.

Lỗi thường gặp:

- `ValueError`: sai shape (không phải `(H, W, 3)`), chiều bằng 0.
- `TypeError`: dtype không phải `uint8`.

## 2. Trích xuất đặc trưng một frame

```python
from src.features.extract import extract

pose_result, hand_result, _ = landmarkers.detect(padded)
feature = extract(pose_result, hand_result)

print(feature.shape)  # (131,)
```

- Trả `float32 (131,)`, có thể chứa `NaN` cho phần không quan sát được.
- Nếu không có pose, toàn bộ vector là `NaN`.
- `extract` **không** nhận frame; nó chỉ dùng kết quả landmark. Vì vậy hãy đảm bảo frame đã qua `frame_padding` trước khi detect để toạ độ X/Y cùng tỉ lệ.

## 3. Xử lý cả clip

```python
import numpy as np
from src.features.extract import extract, process_sequence

raw_features = []
for frame in frames_bgr:
    pose, hand, _ = landmarkers.detect(frame_padding(frame))
    raw_features.append(extract(pose, hand))

raw = np.stack(raw_features)              # (T, 131)
sequence = process_sequence(raw, target_len=30)

print(sequence.shape)                     # (30, 170)
print(np.isfinite(sequence).all())        # True sau xử lý
```

Không thêm chuyển động:

```python
sequence_131 = process_sequence(raw, target_len=30, add_motion=False)
print(sequence_131.shape)                 # (30, 131)
```

Tuỳ chỉnh làm sạch:

```python
sequence = process_sequence(
    raw,
    target_len=30,
    spike_threshold=0.45,  # ngưỡng khử gai 1 frame
    max_gap=6,             # số frame NaN tối đa được nội suy
    add_motion=True,
)
```

## 4. Dùng trong dataset

`DataHandler` chỉ cần một `processor` nhận `(T, 131)` và trả `(30, 170)`:

```python
from src.data.data_reader import DataHandler
from src.features.extract import process_sequence

handler = DataHandler(processor=process_sequence)
sequence, label = handler[0]
print(sequence.shape)  # (30, 170)
```

Nếu không truyền `processor`, `DataHandler` chỉ đọc được mẫu có đúng 30 frame và trả `(30, 131)`.

## 5. Tham chiếu nhanh

| API | Input | Output | Ghi chú |
|---|---|---|---|
| `frame_padding(frame)` | `(H, W, 3) uint8` | `(1280, 1280, 3) uint8` | alias `pad_frame` |
| `extract(pose_res, hand_res)` | MediaPipe results | `(131,) float32` | có `NaN` |
| `process_sequence(raw, target_len=30)` | `(T, 131)` | `(30, 170) float32` | nội suy + khử gai + resample + motion |
| `process_sequence(..., add_motion=False)` | `(T, 131)` | `(30, 131) float32` | không thêm chuyển động |

## 6. Kiểm tra layout đặc trưng

```python
from src.features.extract import FEATURE_DIM, SEQ_FEATURE_DIM, MOTION_DIM

assert FEATURE_DIM == 131
assert MOTION_DIM == 39
assert SEQ_FEATURE_DIM == 170

# presence của tay trái/phải luôn là 2 cột cuối của vector raw
left_present, right_present = feature[-2], feature[-1]
```

## 7. Lưu ý khi debug

- `NaN` ở `handshape`/`orientation` là bình thường khi mất tay; `process_sequence` sẽ xử lý.
- Chuỗi rỗng được trả về ma trận 0 shape `(target_len, 170)` thay vì raise — kiểm tra `len(raw_features)` trước khi dự đoán nếu cần báo lỗi.
- Nếu toàn bộ `sequence` bằng 0, kiểm tra lại clip có pose hay không và `target_len`.
- `np.isinf(raw)` là dấu hiệu dữ liệu hỏng — pipeline không tạo ra `Inf`, nên hãy loại bỏ frame đó.
- Muốn đổi bố cục đặc trưng: cập nhật `_GROUPS`, `FEATURE_DIM`, `schema.FRAME_FEATURE_DIM`, tăng `FEATURE_VERSION`, rồi trích xuất lại toàn bộ dữ liệu và huấn luyện lại.
