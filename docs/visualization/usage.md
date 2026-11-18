# Sử dụng module `src/visualization`

## 1. Vẽ pose

```python
from src.visualization import draw

pose, hand, _ = loader.detect(frame)
frame = draw(
    frame,
    pose,
    point_color=(180, 180, 180),
    line_color=(100, 100, 100),
)
```

## 2. Vẽ bàn tay

```python
frame = draw(
    frame,
    hand,
    point_color=(0, 255, 0),        # tay phải
    line_color=(255, 255, 255),
    secondary_color=(255, 200, 0),   # tay trái (nếu có handedness)
)
```

## 3. Vẽ cả pose và tay

`draw` chỉ xử lý một loại kết quả mỗi lần, nên gọi hai lần:

```python
frame = draw(frame, pose, point_color=(180, 180, 180), line_color=(100, 100, 100))
frame = draw(frame, hand, point_color=(0, 255, 0), secondary_color=(255, 200, 0))
```

## 4. Vòng lặp webcam hoàn chỉnh

```python
import cv2
from src.builtin.mediapipe_builtin import LandmarkerLoader
from src.visualization import draw

loader = LandmarkerLoader()
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        pose, hand, _ = loader.detect(frame)
        frame = draw(frame, pose, point_color=(180, 180, 180), line_color=(100, 100, 100))
        frame = draw(frame, hand, point_color=(0, 255, 0), secondary_color=(255, 200, 0))

        cv2.imshow("VSL preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    loader.close()
    cv2.destroyAllWindows()
```

## 5. Kiểm tra kết quả rỗng

```python
frame = draw(frame, None)  # trả nguyên frame, không lỗi
```

Hàm tự bỏ qua khi `results` không có `hand_landmarks` lẫn `pose_landmarks`.

## 6. Tham chiếu API

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `frame` | — | Ảnh BGR `(H, W, 3)`, bị sửa trực tiếp |
| `results` | — | `PoseLandmarkerResult` hoặc `HandLandmarkerResult` |
| `point_color` | `(0, 0, 0)` | Màu BGR cho điểm landmark |
| `line_color` | `(255, 255, 255)` | Màu BGR cho đường nối |
| `secondary_color` | `(0, 0, 0)` | Màu cho tay trái (khi có nhãn `"Left"`) |

Giá trị trả về: chính `frame` sau khi vẽ.

## 7. Lưu ý

- `draw` vẽ **in-place**; nếu cần giữ ảnh gốc, hãy `frame.copy()` trước.
- Toạ độ landmark được nhân với chiều rộng/cao của frame; frame đã qua `frame_padding` sẽ cho vị trí đúng nhất quán.
- Với ảnh lật gương, nhãn `handedness` có thể đảo — module chỉ dùng nhãn để chọn màu, không ảnh hưởng pipeline đặc trưng (pipeline ghép tay theo hình học).
- Kết quả holistic không được hỗ trợ.
- Không có hàm vẽ HUD đặc trưng (`draw_visualize_features2frame`) trong module hiện tại.
