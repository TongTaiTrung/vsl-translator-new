# Hợp đồng đặc trưng

Nguồn sự thật: `src/features/extract.py` (`FEATURE_VERSION = 8`).

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `FEATURE_DIM` | 383 | cột thô mỗi frame (`extract`) |
| `MOTION_DIM` | 39 | cột vận tốc thêm bởi `process_sequence` |
| `STROKE_DIM` | 12 | cột nét (`STROKE_SLICE`) thêm bởi `process_sequence` |
| `SEQ_FEATURE_DIM` | 434 | 383 + 39 + 12 |
| `EXPECTED_FRAMES` | 30 | số frame sau resample (`src/features/schema.py`) |
| `FRAME_SIZE` | 1280 | cạnh khung sau `frame_padding` |
| `REST_Y` | 0.9 | tay có median y của lòng bàn tay > 0.9 (sát mép dưới khung, y hướng xuống) coi là đang nghỉ, bị bỏ |

## Nhóm cột thô (`FEATURE_SLICES`)

| Nhóm | Cột | Nhóm | Cột |
|---|---|---|---|
| `hand_pos` | 0–18 | `hand_local` | 137–257 |
| `inter_hand_vec` | 18–21 | `finger_curl` | 257–347 |
| `body_contact` | 21–43 | `hand_closure` | 347–363 |
| `hand_contact` | 43–47 | `orientation` | 363–381 |
| `face_vec` | 47–59 | `presence` | 381–383 |
| `hand_scale` | 59–61 | | |
| `handshape` | 61–137 | | |

Truy cập qua `FEATURE_SLICES`, `PRESENCE_SLICE`, `SIDE_COLUMNS`, `LEFT_MIRROR`; `training.py`, `diagnostics.py`, `augment.py` đều suy ra chỉ số từ đây.

## Lịch sử phiên bản (tóm tắt)

- v4: `hand_local` (khung lòng bàn tay, đối xứng gương), `face_vec`.
- v5: `hand_scale` (độ sâu thay cho z của pose); z bị giảm trọng số trong `body_contact` (`CONTACT_Z`).
- v6: bỏ trục cột sống dựa trên hông; trục "lên" luôn là trục dọc ảnh.
- v7: `finger_curl` (góc khớp ngón, bất biến xoay/tỉ lệ/gương) tách `u` khỏi dấu â/ô/ê.
- v8: `hand_closure` (vòng ngón cái – trỏ) tách `o` khỏi `c`.

## `process_sequence`

Theo thứ tự: bỏ tay nghỉ (`_drop_resting_hands`, đặt NaN và `presence`=0), nội suy NaN (đoạn thiếu ≤ 6 frame), khử gai 1 frame, co giãn về 30 frame, thêm vận tốc (39) và nét (12). Cột thủ hình không bị làm mờ. Nét = độ dịch đầu ngón trỏ và lòng bàn tay trong khoảng `STROKE_LAG`=5 frame có dịch chuyển xy lớn nhất, lặp lại ở mọi frame; chỉ tính khoảng mà tay hợp lệ suốt. Giúp nhận dấu sắc/huyền/nặng chỉ chuyển động ~4/30 frame.

## Khi đổi extractor

1. Tăng `FEATURE_VERSION`.
2. Chạy lại trích đặc trưng ([[Module-data]], `src/data/reextract.py`).
3. Kiểm tra các slice trong `training.py`.
4. Chỉ đổi `process_sequence` thì không cần trích lại, nhưng phải train lại mọi checkpoint.

`rotate_view_pitch` mô phỏng đổi độ cao camera trên đặc trưng thô (`PITCH_AUG_DEG` = ±12°); pitch không được ước lượng trực tiếp.
