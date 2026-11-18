# Sử dụng module `src/data`

## 1. Đọc dataset bằng `DataHandler`

```python
from src.data.data_reader import DataHandler
from src.features.extract import process_sequence

handler = DataHandler(
    processor=process_sequence,   # (T, 131) -> (30, 170)
    min_samples_per_class=1,      # số mẫu GỐC tối thiểu để lớp có nhãn
    strict=False,                 # True: mẫu hỏng -> raise
)

print(len(handler))
print(handler.class_names)          # mọi lớp có thư mục
print(handler.active_class_names)   # lớp đã được gán nhãn
print(handler.get_class_counts())
print(handler.get_origin_counts())  # số mẫu gốc độc lập mỗi lớp
print(handler.incomplete)           # mẫu dở dang bị bỏ qua
print(handler.excluded_classes)     # lớp chưa đủ mẫu

sequence, label = handler[0]
print(sequence.shape, label)        # (30, 170), int
```

Đọc raw không qua processor:

```python
raw = handler.load_raw(0)
print(raw.shape)  # (T, 131), có thể chứa NaN
```

Không có processor thì chỉ đọc được mẫu đúng 30 frame:

```python
handler = DataHandler()  # processor=None
seq, label = handler[0]  # (30, 131)
```

Kiểm tra chất lượng:

```python
rows = handler.quality_report()
for row in rows[:3]:
    print(row)
# {"index", "class", "sample_id", "session", "frames",
#  "pose_coverage", "left_hand", "right_hand"}

valid = handler.validate_dataset()
print(f"{valid} mẫu hợp lệ")
```

## 2. Ghi mẫu bằng `DataHandler`

```python
from src.data.data_reader import DataHandler
from src.features.extract import extract, process_sequence
from src.features.frame import frame_padding

handler = DataHandler(processor=process_sequence)
raw_features, frames = [], []

for frame in frames_bgr:
    padded = frame_padding(frame)
    pose, hand, _ = landmarkers.detect(padded)
    raw_features.append(extract(pose, hand))
    frames.append(padded)

sample_id = handler.write_sample(
    class_name="xin chào",
    features=raw_features,     # list các vector (131,)
    frames=frames,             # optional, lưu video gốc
    fps=30.0,
    session="session-2026-11-17",
    meta={"source": "webcam"},
    require_hand=True,
)
print(sample_id)
```

- `features` phải có tối thiểu 8 frame, dtype `float32`, shape `(131,)`.
- Nếu toàn bộ frame thiếu pose hoặc thiếu tay → `ValueError`.
- Ghi atomic: lỗi giữa chừng không để lại mẫu dở dang.
- Video được lưu vào `data/raw/<class>/<id>.mp4`; feature vào `data/processed/<class>/<id>/`.

Xoá mẫu:

```python
handler.remove_sample("xin chào", sample_id)
```

Quản lý lớp:

```python
handler.add_class("cảm ơn")
handler.remove_class("cảm ơn")
```

## 3. Xuất dataset để huấn luyện

```python
from src.data.data_reader import DataHandler
from src.features.extract import process_sequence

handler = DataHandler(processor=process_sequence)

data = handler.export_arrays(group_by="sample")
print(data["X"].shape)        # (N, 30, 170)
print(data["y"].shape)        # (N,)
print(data["classes"])        # list tên lớp theo thứ tự nhãn
print(data["groups"][:3])     # "class/origin_sample_id"
print(data["sessions"][:3])
print(data["augmented"][:3])  # bool

handler.save_npz("dataset.npz", group_by="sample")
```

`group_by="session"` nhóm theo phiên ghi thay vì mẫu gốc.

Lưu ý: nếu mẫu không cùng shape và không có `processor`, `export_arrays` raise. Luôn truyền `processor=process_sequence` cho dữ liệu dài tuỳ ý.

## 4. Thu thập dữ liệu bằng webcam (CLI)

```bash
python -m src.data.data_collect \
  --class_name xin_chao \
  --num_samples 10 \
  --group 0 \
  --root root_project \
  --fps 30 \
  --camera 0
```

Thao tác:

- `SPACE`: bắt đầu/kết thúc một mẫu.
- `Q` hoặc `ESC`: thoát.

Kết quả:

```text
root_project/raw/xin_chao/<id>.mp4
root_project/extracted/xin_chao/0/<id>.npz   # key "X", shape (30, 170)
```

Bố cục này khác `data/processed` của `DataHandler`; nếu muốn dùng cho training qua `DataHandler`, cần chuyển đổi hoặc dùng trực tiếp file `.npz` theo định dạng training.

## 5. Trích xuất lại từ video gốc

Khi đổi extractor/feature version, chạy lại từ `data/raw`:

```python
from pathlib import Path
from src.data.reextract import find_raw_samples, reextract_sample

raw_root = Path("data/raw")
processed_root = Path("data/processed")
backup_root = Path("data/processed_backup")

videos = find_raw_samples(raw_root, classes=["xin chào", "cảm ơn"])
for video in videos:
    result = reextract_sample(video, raw_root, processed_root, backup_root)
    print(result)
# {"class", "sample_id", "frames", "pose_coverage",
#  "left_hand", "right_hand", "backup"}
```

Đặc điểm:

- Không ghi đè backup đã tồn tại (`FileExistsError`).
- Nếu lỗi giữa chừng, mẫu cũ được khôi phục.
- `session` cũ được giữ nguyên; các trường chất lượng được cập nhật.
- Video phải có ít nhất 8 frame, detect được pose và ít nhất một tay.

Chỉ trích xuất feature, không ghi đĩa:

```python
from src.data.reextract import extract_video_features

raw_features, fps = extract_video_features("data/raw/xin_chao/0.mp4")
print(raw_features.shape, fps)  # (T, 131)
```

## 6. Tăng cường dữ liệu

### Lên kế hoạch

```python
from pathlib import Path
from src.data.augment import plan_augmentation

root = Path("data/processed")

# Mỗi lớp đạt tối thiểu 20 mẫu, mỗi mẫu gốc tối đa được nhân bản gần đều
jobs, counts = plan_augmentation(
    root,
    classes=None,             # None = mọi lớp
    minimum_samples=20,
    copies_per_sample=2,
)

# Hoặc nhân bản mọi mẫu gốc đúng 3 lần
jobs_all, counts_all = plan_augmentation(
    root, classes=None, minimum_samples=-1, copies_per_sample=3,
)
```

### Thực thi

```python
import numpy as np
from src.data.augment import augment_sample

rng = np.random.default_rng(42)
for source_dir, copies in jobs:
    created = augment_sample(
        source_dir,
        raw_root=Path("data/raw"),
        copies=copies,
        strength=1.0,          # (0, 3]
        rng=rng,
    )
    print(source_dir, "->", created)
```

Mẫu mới:

- Nằm cùng thư mục lớp, ID số mới.
- Meta có `origin_sample_id` và `augmentation = {"type": "feature_noise", ...}`.
- Không có video raw.
- Được tính vào tổng số mẫu nhưng **không** tính vào `min_samples_per_class` (vì không phải mẫu gốc độc lập).

### Chỉ tạo nhiễu trên mảng

```python
from src.data.augment import add_feature_noise

noisy = add_feature_noise(raw_features, rng, strength=1.0)
assert noisy.shape == raw_features.shape
```

## 7. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ValueError: Invalid feature shape ... expected (131,)` | Extractor/schema đã đổi; cập nhật `FRAME_FEATURE_DIM` hoặc trích xuất lại |
| `ValueError: Inf detected` | Dữ liệu hỏng; loại bỏ frame/file |
| `ValueError: Sample must contain at least 8 frames` | Ghi clip dài hơn |
| `ValueError: Không detect được tay...` | Kiểm tra ánh sáng/khung hình hoặc `require_hand=False` nếu chấp nhận |
| `RuntimeError: Incomplete sample` | Đang bật `strict=True`; sửa/xoá mẫu dở dang |
| `ValueError: Class ... needs at least k independent source groups` | Thêm mẫu gốc hoặc giảm số fold |
| `FileExistsError: Backup already exists` | Dùng `backup_root` khác hoặc dọn backup cũ |
| Lớp không xuất hiện trong dataset | Chưa đủ `min_samples_per_class` mẫu gốc; xem `excluded_classes` |

## 8. Quy trình gợi ý

```text
1. DataHandler.add_class(...) hoặc ghi mẫu trực tiếp
2. write_sample(...) cho từng lần ký
3. quality_report() + validate_dataset()
4. (tuỳ chọn) plan_augmentation + augment_sample
5. save_npz("dataset.npz", group_by="sample")
6. python -m src.model.training --data dataset.npz
```
