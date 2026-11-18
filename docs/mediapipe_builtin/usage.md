# Sử dụng module `src/builtin/mediapipe_builtin`

## 1. Detect một ảnh (IMAGE)

Phù hợp cho data collection, xử lý ảnh rời.

```python
import cv2
from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode

loader = LandmarkerLoader(running_mode=VisionRunningMode.IMAGE)
try:
    frame = cv2.imread("frame.jpg")
    pose, hand, _ = loader.detect(frame)
    print(len(pose.pose_landmarks), len(hand.hand_landmarks))
finally:
    loader.close()
```

`LandmarkerLoader` tự gọi `frame_padding`, nên ảnh đầu vào không nhất thiết vuông.

## 2. Detect video tuần tự (VIDEO)

Phù hợp cho file video hoặc batch clip. Timestamp tự tăng nếu không truyền.

```python
from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode

loader = LandmarkerLoader(running_mode=VisionRunningMode.VIDEO)
try:
    for frame in frames:
        pose, hand, ts = loader.detect(frame)
        # ts: timestamp ms đã dùng
finally:
    loader.close()
```

Tự kiểm soát timestamp (phải tăng nghiêm ngặt):

```python
pose, hand, ts = loader.detect(frame, timestamp_ms=0)
pose, hand, ts = loader.detect(frame, timestamp_ms=33)
```

Sai thứ tự → `ValueError("timestamp_ms must increase on every frame.")`.

## 3. Webcam realtime (LIVE_STREAM)

Không chờ kết quả (dùng kết quả mới nhất):

```python
import cv2
from src.builtin.mediapipe_builtin import LandmarkerLoader

loader = LandmarkerLoader()
cap = cv2.VideoCapture(0)
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pose, hand, ts = loader.detect(frame)  # async, có thể trả kết quả cũ/None
        cv2.imshow("VSL", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    loader.close()
    cv2.destroyAllWindows()
```

Chờ đồng bộ theo timestamp (tiện cho pipeline cần cặp pose/hand khớp nhau):

```python
pose, hand, ts = loader.detect(frame, wait_for_result=True, timeout_s=5.0)
```

Có thể raise:

- `TimeoutError`: không nhận đủ kết quả trong `timeout_s`.
- `RuntimeError`: MediaPipe bỏ qua timestamp (thường do gửi frame quá nhanh hoặc dùng sai timestamp).
- `ValueError`: `timeout_s <= 0` hoặc timestamp không tăng.

## 4. Đổi model hoặc delegate

```python
from src.builtin.mediapipe_builtin import Delegate, LandmarkerLoader
from src.builtin.mediapipe_builtin.defs import (
    HandLandmarkerPath, PoseLandmarkerPath,
)

loader = LandmarkerLoader(
    delegate=Delegate.GPU,
    pose_model_path=PoseLandmarkerPath,
    hand_model_path=HandLandmarkerPath,
    num_hands=1,
    detection_confidence=0.6,
    presence_confidence=0.6,
    tracking_confidence=0.5,
)
```

Có thể trỏ tới model `.task` khác, ví dụ `pose_landmarker_heavy.task`:

```python
from pathlib import Path
from src.builtin.mediapipe_builtin import LandmarkerLoader
from src.builtin.mediapipe_builtin.defs import CURRENT_DIR

loader = LandmarkerLoader(
    pose_model_path=str(CURRENT_DIR / "models" / "pose_landmarker_heavy.task")
)
```

## 5. Vẽ landmark

```python
from src.visualization.landmarks import draw

pose, hand, _ = loader.detect(frame)
frame = draw(frame, pose, point_color=(180, 180, 180), line_color=(100, 100, 100))
frame = draw(frame, hand, point_color=(0, 255, 0), secondary_color=(255, 200, 0))
```

## 6. Tham chiếu API

| API | Mô tả |
|---|---|
| `LandmarkerLoader(...)` | Tạo Pose + Hand landmarker |
| `detect(frame, wait_for_result=False, timeout_s=5.0, timestamp_ms=None)` | Trả `(pose_result, hand_result, timestamp_ms)` |
| `close()` | Giải phóng tài nguyên; luôn gọi trong `finally` |
| `VisionRunningMode.IMAGE / VIDEO / LIVE_STREAM` | Chế độ chạy |
| `Delegate.CPU / Delegate.GPU` | Thiết bị suy luận |
| `PoseLandmarkerPath`, `HandLandmarkerPath` | Đường dẫn model mặc định |
| `StreamProcess` | Hộp chứa kết quả callback async |

## 7. Lỗi thường gặp

| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `ValueError: timestamp_ms must increase...` | Dùng lại/giảm timestamp khi chạy VIDEO/LIVE_STREAM |
| `TimeoutError` | Chạy LIVE_STREAM + `wait_for_result=True` nhưng callback không kịp; tăng timeout hoặc giảm tải |
| `RuntimeError: callback skipped timestamp` | Gửi frame dồn dập hoặc timestamp nhảy cóc; dùng timestamp tự động |
| Không detect được gì | Ảnh quá tối/nhỏ, người ngoài khung, hoặc model path sai |
| Rò rỉ bộ nhớ | Quên `close()`, đặc biệt khi tạo loader mỗi request |

## 8. Quy tắc dùng an toàn

- Một `LandmarkerLoader` cho **một** camera/luồng/client; không chia sẻ giữa các luồng.
- Luôn `close()` trong `finally`.
- Với IMAGE, timestamp không cần và trả `None`.
- Với VIDEO, nên dùng timestamp tự động hoặc timestamp thực của video.
- Với LIVE_STREAM, nếu cần kết quả đồng bộ từng frame, dùng `wait_for_result=True`; nếu chỉ cần hiển thị, để `False`.
- Không trộn kết quả pose của frame này với hand của frame khác — kiểm tra `timestamp_ms` khi cần ghép cặp.
