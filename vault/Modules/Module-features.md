# `src/features`

| File | Vai trò |
|---|---|
| `frame.py` | `frame_padding`: scale đều cạnh dài = 1280, đặt giữa, nền đen (`FRAME_SIZE`) |
| `extract.py` | `extract` (383 cột/frame), `process_sequence`, `rotate_view_pitch`, `stroke_magnitude`, `still_versions`, `FEATURE_SLICES` ([[Feature-Contract]]) |
| `landmarks.py` | chuyển kết quả MediaPipe ↔ dict/mảng, `save_landmarks`/`load_landmarks` (`landmarks.npz`), `check_landmarks` |
| `view.py` | `simulate_view`: giả lập camera/bối cảnh khác trên landmark rồi tính lại bằng `extract` |
| `camera.py` | `open_camera`, `parse_resolution`: mở webcam ở độ phân giải 16:9 lớn nhất |
| `schema.py` | `EXPECTED_FRAMES = 30`, re-export `FRAME_FEATURE_DIM` |

## `view.py`

Biến đổi landmark (không phải đặc trưng): quỹ đạo camera quanh giữa hai vai (yaw/pitch/roll), khoảng cách/FOV bằng phối cảnh, cỡ tay, vị trí ký, lệch khớp ngón theo clip. `IDENTITY` tái tạo đúng đặc trưng đã lưu. `augment.make_variant` dùng nó khi có `landmarks.npz`; thiếu thì quay về nhiễu đặc trưng + `rotate_view_pitch` ([[Module-data]]).

## `camera.py`

OpenCV không tự chọn độ phân giải lớn nhất, nên mọi chỗ mở camera đi qua `open_camera`. Mặc định 16:9 vì dữ liệu `data/raw` quay 16:9.
