# Dữ liệu trên đĩa

`data/` nằm trong `.gitignore`; một số video `data/raw` đã được commit từ trước.

```
data/raw/<lớp>/<id>.mp4                    video gốc
data/processed/<lớp>/<id>/<frame>.npz      đặc trưng thô từng frame + meta
data/processed/<lớp>/<id>/landmarks.npz    landmark MediaPipe thô (chỉ mẫu gốc, do reextract ghi)
```

- Tên lớp là tiếng Việt chuẩn hoá NFC.
- `landmarks.npz`: pose `(T,33,4)`, tay và hand-world `(T,2,21,3)`, NaN khi không có; `DataHandler._frame_paths` bỏ qua vì tên không phải số.
- Mẫu hợp lệ có ≥ `MIN_FRAMES` = 8 frame; đọc ra được co giãn về `SEQUENCE_LENGTH` = 30.

## `DataHandler` (`src/data/data_reader.py`)

Quản lý bố cục trên. API công khai: `write_sample`, `load_raw`, `export_arrays`, `save_npz`, `add_class`, `remove_class`, `remove_sample`, `refresh_dataset`, `validate_dataset`, `quality_report`, `describe`, `get_class_counts`, `get_origin_counts`, `sample_groups`.

`augment.py` và `reextract.py` gọi trực tiếp các helper tĩnh private `_read_meta`, `_frame_paths`, `_load_feature` và hằng `FEATURE_DIM`, `PRESENCE_COLS`, `MIN_FRAMES`, `SEQUENCE_LENGTH`: đổi chữ ký sẽ làm hỏng cả hai module.

## Nhóm và rò rỉ

- Mẫu tăng cường ghi `origin_sample_id` trong meta và dùng chung nhóm với mẫu gốc.
- `make_folds` (StratifiedGroupKFold) giữ nguyên mỗi nhóm; validation/OOF chỉ tính trên mẫu gốc (`validation_mask`).
- **Không bao giờ tăng cường một mẫu đã tăng cường.**

## Lệch bối cảnh quay

Mọi lớp trừ `m` được quay trong một buổi; cùng một ký hiệu ở hai bối cảnh xa nhau hơn hai ký hiệu khác nhau trong cùng bối cảnh. CV theo mẫu gốc vì thế không đo được hiệu năng thật. Xem [[Experiment-Tools]] (`benchmark`) và `src/features/view.py` ([[Module-features]]).
