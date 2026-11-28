# `src/data`

| File | Vai trò |
|---|---|
| `data_reader.py` | `DataHandler`: bố cục dữ liệu, đọc/ghi mẫu, xuất `.npz` ([[Data-Layout]]) |
| `data_collect.py` | CLI quay clip: `python -m src.data.data_collect -c <lớp>` |
| `reextract.py` | dựng lại đặc trưng và `landmarks.npz` từ `data/raw/*.mp4` (`reextract_sample`) |
| `augment.py` | tạo biến thể từ mẫu gốc (`plan_augmentation`, `make_variant`, `augment_sample`) |

## Tăng cường (`augment.py`)

- Có `landmarks.npz`: `view.simulate_view` + `extract` (nhất quán mọi cột).
- Không có: nhiễu đặc trưng (`add_feature_noise`) + `rotate_view_pitch` (±`PITCH_AUG_DEG`).
- Biến thể ghi `origin_sample_id`, chia sẻ nhóm với mẫu gốc; không tăng cường biến thể ([[Data-Layout]]).

## Dùng qua notebook

`data_collect.ipynb` thu dữ liệu, `reextract_features.ipynb` gọi `reextract`, `augment_features.ipynb` gọi `augment`, đều qua `DataHandler`.
