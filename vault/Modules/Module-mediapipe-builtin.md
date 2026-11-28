# `src/builtin/mediapipe_builtin`

Bọc MediaPipe Pose/Hand landmarker cho Python.

| File | Vai trò |
|---|---|
| `defs.py` | đường dẫn model (`PoseLandmarkerPath`, `HandLandmarkerPath`), alias kiểu MediaPipe, `HandConnections`, `PoseConnections` |
| `landmarkers_loader.py` | `LandmarkerLoader`: tạo landmarker, `detect`, `close`; xử lý chế độ IMAGE/VIDEO |
| `stream_process.py` | `StreamProcess`: giữ kết quả callback mới nhất |
| `models/` | `pose_landmarker_full.task`, `hand_landmarker.task`, `blaze_face_short_range.tflite` |

- Ở chế độ VIDEO, timestamp truyền vào `detect` phải **tăng nghiêm ngặt**.
- `blaze_face_short_range.tflite` chỉ dùng cho web ([[Module-web]]): `external/vsl-web/scripts/copy-mediapipe.mjs` chép nó cùng hai file `.task` vào `public/mediapipe/`.
- Phiên bản MediaPipe Python (`mediapipe==1.1.1`) khớp `@mediapipe/tasks-vision` 1.1.1 ở web.
