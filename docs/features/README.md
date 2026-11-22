# Module `src/features`

Chuyển khung hình BGR thành vector đặc trưng VSL và biến một clip có độ dài bất kỳ thành tensor cố định `(30, 170)`.

## Kiến trúc

```mermaid
flowchart LR
    A[BGR frame] -->|frame_padding| B[1280x1280 BGR]
    B --> C[MediaPipe results]
    C -->|extract| D[(T, 131) raw<br/>có NaN]
    D -->|process_sequence| E[(30, 170) processed]
```

### Các file

| File | Nội dung |
|---|---|
| `frame.py` | `FRAME_SIZE`, `frame_padding` / `pad_frame` |
| `schema.py` | Hằng số chia sẻ: `EXPECTED_FRAMES = 30`, `FRAME_FEATURE_DIM = 131` |
| `extract.py` | `extract`, `process_sequence`, layout đặc trưng, hằng số trích xuất |
| `__init__.py` | Rỗng — import trực tiếp từ module con |

## `frame.py` — chuẩn hoá khung hình

`frame_padding(frame)`:

1. Kiểm tra đầu vào là `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, kích thước dương.
2. Nếu đã là `1280 x 1280`, trả nguyên frame.
3. Scale **đều** bằng `1280 / max(H, W)` (thu nhỏ dùng `INTER_AREA`, phóng to dùng `INTER_LINEAR`), để toạ độ chuẩn hoá không phụ thuộc độ phân giải đầu vào (video cũ 720x720 và camera 1280x720 cho cùng hình học).
4. Đặt ảnh vào giữa canvas `1280 x 1280`, phần thừa tô đen `(0, 0, 0)`.

Kết quả luôn là `(1280, 1280, 3) uint8`, không méo tỉ lệ. `pad_frame` là alias.

## `extract.py` — 131 đặc trưng/frame

### Hệ quy chiếu

- Gốc: trung điểm hai vai (landmark pose 11, 12).
- Đơn vị: độ rộng vai 2D.
- `u_x`: vai trái → vai phải; `u_y`: phương thẳng đứng của ảnh, trực giao hoá với `u_x` (chỉ khử roll); `u_z = u_x × u_y`.
- Trục Z pose được giảm chấn (`Z_POSE = 0.5`) vì nhiễu lớn hơn X/Y; khi dựng `u_x` giảm thêm (`TORSO_Z = 0.3`).
- **Pitch (camera cao/thấp hơn người ký) không được khử trong `extract`** (từ v6). Đo trên 368 clip gốc: pitch thân suy
  từ z của pose (ảnh hay `pose_world`) lệch theo *lớp ký hiệu* trong cùng một camera (vd. session 22/11: `b` −20°, `o` −9°)
  nhiều hơn mức lệch giữa các session (~±5°), và xoay theo nó làm nhiễu vị trí tay tăng 2–4 lần; tín hiệu hình học đầu
  (tai/mắt/mũi) cũng phụ thuộc lớp như vậy. v5 dùng trục hông khi visibility > 0.5, nhưng hông hầu như ngoài khung nên
  nhánh đó chỉ chạy ở ~7% clip và đổi hệ trục giữa clip (pitch giả ~−4°); v6 bỏ nhánh này.
- Bất biến pitch được dạy bằng dữ liệu: `rotate_view_pitch(features, degrees)` mô phỏng camera thấp hơn (`degrees > 0`)
  trên feature thô: xoay `hand_pos`, `inter_hand_vec`, `face_vec` quanh trục x qua giữa vai trong thang độ sâu thật
  (z ảnh của pose ≈ `POSE_Z_GAIN = 3.35` × độ sâu, hồi quy với `pose_world`), và xoay `palm_dir`/`palm_normal`. Các nhóm
  còn lại bất biến (`body_contact` xấp xỉ). So với việc trích lại từ cảnh 3D đã xoay, nó giải thích ~97% thay đổi của y
  vị trí và ~92% của hướng lòng bàn tay. `augment_sample` gán mỗi bản sao một pitch ngẫu nhiên đều trong
  ±`PITCH_AUG_DEG` (12°), ghi vào `meta.augmentation.view_pitch_deg`.

### Ghép bàn tay với trái/phải

Không tin nhãn `handedness` (có thể bị đảo khi ảnh lật gương). Thay vào đó:

- So khoảng cách cổ tay hand-detect với cổ tay pose (landmark 15, 16) theo đơn vị độ rộng vai.
- Với 2 tay, chọn cặp gán thẳng hoặc chéo có tổng chi phí nhỏ hơn.
- Bỏ ghép nếu khoảng cách > `MAX_MATCH_DIST = 0.8`.
- `presence` là confidence mềm `exp(-(d / 0.35)^2)`, không phải cờ 0/1.

### Bố cục 131 chiều

| Nhóm | Slice | Kích thước | Nội dung |
|---|---|---|---|
| `hand_pos` | `0:18` | 18 | Mỗi tay 9: tâm lòng bàn tay, đầu ngón cái, đầu ngón trỏ (3 điểm × 3 trục) |
| `inter_hand_vec` | `18:21` | 3 | Tâm bàn tay trái − phải |
| `body_contact` | `21:43` | 22 | Mỗi tay 11: độ gần 11 vùng cơ thể |
| `hand_contact` | `43:47` | 4 | Lòng-lòng, cái-cái, trỏ-trỏ, cặp điểm gần nhất |
| `handshape` | `47:111` | 64 | Mỗi tay 32: 15 cos khớp, 5 độ duỗi, 4 độ xòe, 8 tiếp xúc đầu ngón |
| `orientation` | `111:129` | 18 | Mỗi tay 9: `palm_dir` 3, `palm_normal` 3, 3 cos góc cổ tay so với cẳng tay |
| `presence` | `129:131` | 2 | Confidence hiện diện tay trái/phải |

Bên trong mỗi tay, thứ tự luôn là **trái trước, phải sau**. Các mốc quan trọng:

- `hand_pos`: trái `0:9`, phải `9:18`
- `body_contact`: trái `21:32`, phải `32:43`
- `handshape`: trái `47:79`, phải `79:111`
- `orientation`: trái `111:120`, phải `120:129`
- `presence`: trái `129`, phải `130`

### 11 neo cơ thể cho `body_contact`

Thứ tự: trán, mũi, miệng, cằm, má trái, má phải, cổ, ngực, bụng, vai trái, vai phải. Mỗi neo có hệ số `gamma` riêng; độ gần dùng kernel `exp(-gamma * d²)` lấy **max** trên 6 điểm bàn tay (lòng bàn tay + 5 đầu ngón). Nhờ vậy thay đổi nhỏ khi chạm được phóng đại, jitter khi xa bị triệt.

### Giá trị thiếu

- Không có pose → trả toàn `NaN`.
- Không detect được một tay → nhóm hình dạng/hướng/vị trí của tay đó là `NaN`, `presence = 0`.
- Nếu visibility cổ tay/khuỷu tay pose thấp, góc cổ tay là `NaN`.
- `NaN` chỉ được xử lý ở `process_sequence`, không ở `extract`.

## `process_sequence` — chuỗi → `(30, 170)`

Đầu vào `(T, 131)`, đầu ra `(target_len, 170)`. Các bước theo thứ tự:

1. **Presence**: `NaN` → `0` (không nội suy cờ hiện diện).
2. **Nội suy khoảng trống**: mỗi cột, đoạn `NaN` dài ≤ `max_gap` (mặc định 6) được nội suy tuyến tính; đoạn dài hơn → `0.0`. Cột toàn `NaN` → `0.0`.
3. **Khử gai 1 frame**: nếu frame giữa lệch cùng chiều so với cả hai bên, biên độ vượt ngưỡng (`0.45` × scale), và hai frame lân cận gần nhau, thay bằng trung điểm. Scale là `1.0` cho `hand_pos`/`inter_hand_vec`, `2.0` cho nhóm còn lại, `inf` cho `presence` (không khử cờ).
4. **Co giãn thời gian**: nội suy tuyến tính trên trục thời gian về `target_len`; chuỗi 1 frame được lặp.
5. **Chuyển động** (`add_motion=True`): lấy các cột `hand_pos`, `inter_hand_vec`, `orientation` (39 cột), làm mượt nhẹ `[1,2,1]/4` theo thời gian, rồi sai phân trung tâm × `(target_len - 1)`. Ghép sau 131 chiều → 170.

`add_motion=False` trả `(30, 131)`.

Các cột chuyển động nằm ở `131:170`:

| Cột | Nội dung |
|---|---|
| `131:140` | Vận tốc vị trí tay trái |
| `140:149` | Vận tốc vị trí tay phải |
| `149:152` | Vận tốc vector giữa hai tay |
| `152:161` | Vận tốc hướng bàn tay trái |
| `161:170` | Vận tốc hướng bàn tay phải |

## Điểm thiết kế đáng lưu ý

- Đặc trưng bất biến với vị trí/khoảng cách người ký nhờ hệ trục thân và đơn vị độ rộng vai.
- Thủ hình/hướng bàn tay dùng `hand_world_landmarks` khi có (`USE_HAND_WORLD = True`) vì ổn định hơn toạ độ ảnh.
- Không làm mượt các cột `handshape` — giữ rung/búng ngón.
- Chuyển động tính **sau** khi co giãn thời gian, nên biên độ phụ thuộc `target_len` và độc lập tốc độ tuyệt đối của người ký.
- `FEATURE_VERSION = 2`: đổi bố cục đặc trưng phải tăng version và trích xuất lại dữ liệu.

## Phụ thuộc

- `numpy`, `scipy.interpolate.interp1d`.
- `opencv-python` chỉ trong `frame.py`.
- Kiểu kết quả MediaPipe từ `src.builtin.mediapipe_builtin.defs`.
