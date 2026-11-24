# Module `src/features`

Chuyển khung hình BGR thành vector đặc trưng VSL và biến một clip có độ dài bất kỳ thành tensor cố định `(30, 418)` (feature v7).

## Kiến trúc

```mermaid
flowchart LR
    A[BGR frame] -->|frame_padding| B[1280x1280 BGR]
    B --> C[MediaPipe results]
    C -->|extract| D[(T, 367) raw<br/>có NaN]
    D -->|process_sequence| E[(30, 418) processed]
```

### Các file

| File | Nội dung |
|---|---|
| `frame.py` | `FRAME_SIZE`, `frame_padding` / `pad_frame` |
| `schema.py` | Hằng số chia sẻ: `EXPECTED_FRAMES = 30`, `FRAME_FEATURE_DIM` (= `extract.FEATURE_DIM` = 367) |
| `extract.py` | `extract`, `process_sequence`, layout đặc trưng, hằng số trích xuất |
| `__init__.py` | Rỗng — import trực tiếp từ module con |

## `frame.py` — chuẩn hoá khung hình

`frame_padding(frame)`:

1. Kiểm tra đầu vào là `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, kích thước dương.
2. Nếu đã là `1280 x 1280`, trả nguyên frame.
3. Scale **đều** bằng `1280 / max(H, W)` (thu nhỏ dùng `INTER_AREA`, phóng to dùng `INTER_LINEAR`), để toạ độ chuẩn hoá không phụ thuộc độ phân giải đầu vào (video cũ 720x720 và camera 1280x720 cho cùng hình học).
4. Đặt ảnh vào giữa canvas `1280 x 1280`, phần thừa tô đen `(0, 0, 0)`.

Kết quả luôn là `(1280, 1280, 3) uint8`, không méo tỉ lệ. `pad_frame` là alias.

## `extract.py` — 367 đặc trưng/frame

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

### Bố cục 367 chiều (v7)

Mọi module khác lấy chỉ số cột từ `FEATURE_SLICES` / `SIDE_COLUMNS` / `LEFT_MIRROR`, không hard-code.

| Nhóm | Slice | Kích thước | Nội dung |
|---|---|---|---|
| `hand_pos` | `0:18` | 18 | Mỗi tay 9: tâm lòng bàn tay, đầu ngón cái, đầu ngón trỏ (3 điểm × 3 trục) |
| `inter_hand_vec` | `18:21` | 3 | Tâm bàn tay trái − phải |
| `body_contact` | `21:43` | 22 | Mỗi tay 11: độ gần 11 vùng cơ thể |
| `hand_contact` | `43:47` | 4 | Lòng-lòng, cái-cái, trỏ-trỏ, cặp điểm gần nhất |
| `face_vec` | `47:59` | 12 | Mỗi tay: tâm lòng bàn tay − mũi, đầu ngón trỏ − mũi |
| `hand_scale` | `59:61` | 2 | Cỡ lòng bàn tay trên ảnh / độ rộng vai |
| `handshape` | `61:137` | 76 | Mỗi tay 38: 15 cos khớp, 5 độ duỗi, 4 độ xòe, 8 tiếp xúc đầu ngón, 3 vị trí ngón cái, 3 thứ tự đầu ngón (từ `hand_world`) |
| `hand_local` | `137:257` | 120 | Mỗi tay 60: 20 điểm trong hệ trục lòng bàn tay (từ `hand_world`) |
| `finger_curl` | `257:347` | 90 | Mỗi tay 45 (v7): độ cong từng ngón, đo trên **cả** landmark ảnh và `hand_world`, xem dưới |
| `orientation` | `347:365` | 18 | Mỗi tay 9: `palm_dir` 3, `palm_normal` 3, 3 cos góc cổ tay so với cẳng tay |
| `presence` | `365:367` | 2 | Confidence hiện diện tay trái/phải |

`process_sequence` nối thêm 39 cột chuyển động (`367:406`) và 12 cột nét (`406:418`).

### `finger_curl` (v7)

Mỗi tay 45 số, thứ tự `[ảnh: góc 15 khớp (rad), tổng gập 5 ngón, gập PIP+DIP 5 ngón, độ thẳng 5 ngón |
world: tổng gập 5, gập PIP+DIP 5, độ thẳng 5]`. Ngón theo thứ tự cái, trỏ, giữa, nhẫn, út; khớp theo chuỗi
cổ tay → gốc ngón → … → đầu ngón (với ngón cái: CMC, MCP, IP).

- Góc khớp = `atan2(|a×b|, a·b)` giữa hai đốt liên tiếp (rad). Tuyến tính theo độ gập, khác `cos` ở `handshape` vốn
  nén phần gập nhẹ (cos 16° = 0.96, cos 32° = 0.85) và khác `arccos` (đạo hàm vô hạn quanh 0° → khuếch đại jitter).
- Tổng gập / gập PIP+DIP cộng 2–3 khớp của một ngón, nên lệch ngẫu nhiên từng khớp của MediaPipe triệt bớt nhau.
- Độ thẳng = |đầu ngón − gốc ngón| / tổng độ dài 3 đốt: 1 khi duỗi, nhỏ dần khi móc/cuộn.
- Mọi giá trị bất biến khi xoay, tịnh tiến, co giãn và soi gương (không cần đổi dấu trong `LEFT_MIRROR`,
  `rotate_view_pitch` giữ nguyên). Nhóm có stem riêng trong `training.default_groups`, nên group-drop buộc model học
  được cả từ độ cong ngón khi các nhóm khác bị tắt.

Lý do: các cặp hay nhầm khác nhau chủ yếu ở **độ móc ngón** — `u` (trỏ + giữa duỗi thẳng) với dấu â/ô/ê (trỏ + giữa
móc: PIP lệch ~16–37°, DIP ~23–29° trên `hand_world`), `i` (út duỗi) với dấu ơ/ư (út móc, ngón cái duỗi). Đo trên clip
gốc, `hand_world` làm phẳng độ móc (PIP+DIP ngón trỏ của dấu mũ: ~1.5 rad trên world, ~2.3 rad trên ảnh; của `u`:
0.8 và 0.2), nên độ tách giữa hai lớp so với độ lệch giữa hai session của lớp `m` lớn hơn 2–3 lần khi đo trên landmark
ảnh. z ảnh của tay cùng thang với x, y (frame vuông) nên góc 3D đo trực tiếp được. Giữ cả hai nguồn vì chúng là hai
đầu ra khác nhau của mạng.

### 11 neo cơ thể cho `body_contact`

Thứ tự: trán, mũi, miệng, cằm, má trái, má phải, cổ, ngực, bụng, vai trái, vai phải. Mỗi neo có hệ số `gamma` riêng; độ gần dùng kernel `exp(-gamma * d²)` lấy **max** trên 6 điểm bàn tay (lòng bàn tay + 5 đầu ngón). Nhờ vậy thay đổi nhỏ khi chạm được phóng đại, jitter khi xa bị triệt.

### Giá trị thiếu

- Không có pose → trả toàn `NaN`.
- Không detect được một tay → nhóm hình dạng/hướng/vị trí của tay đó là `NaN`, `presence = 0`.
- Nếu visibility cổ tay/khuỷu tay pose thấp, góc cổ tay là `NaN`.
- `NaN` chỉ được xử lý ở `process_sequence`, không ở `extract`.

## `process_sequence` — chuỗi → `(30, 418)`

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
- `handshape`, `hand_local`, hướng bàn tay dùng `hand_world_landmarks` khi có (`USE_HAND_WORLD = True`); `finger_curl` (v7)
  dùng cả landmark ảnh vì `hand_world` làm phẳng độ móc ngón.
- Không làm mượt các cột `handshape` — giữ rung/búng ngón.
- Chuyển động tính **sau** khi co giãn thời gian, nên biên độ phụ thuộc `target_len` và độc lập tốc độ tuyệt đối của người ký.
- `FEATURE_VERSION = 7`: đổi bố cục đặc trưng phải tăng version và trích xuất lại dữ liệu.

## Phụ thuộc

- `numpy`, `scipy.interpolate.interp1d`.
- `opencv-python` chỉ trong `frame.py`.
- Kiểu kết quả MediaPipe từ `src.builtin.mediapipe_builtin.defs`.
