---
marp: true
title: VSL Translator — Tổng quan dự án
paginate: true
---

# VSL Translator
## Dịch Ngôn ngữ Ký hiệu Việt Nam từ webcam

- Nhận dạng ký hiệu VSL theo thời gian thực
- Ghép các từ nhận được thành câu tiếng Việt bằng LLM
- Python (MediaPipe + PyTorch) · Flask API · Web app Next.js

*Cập nhật: 20/11/2026 · nhánh `feat/vsl-web`*

---

## 1. Bài toán

- **Đầu vào:** video webcam của người ký
- **Đầu ra:** chuỗi từ → câu tiếng Việt hoàn chỉnh
  - Ví dụ: `xin chào | toi | ten_la | T-R-U-N-G` → *"Xin chào, tôi tên là Trung."*
- **Khó khăn:**
  - Dữ liệu ít, tự thu thập
  - Các ký hiệu gần giống nhau (`cảm ơn` / `xin chào`, `h` / `k`, `toi` / `m`)
  - Phải chạy realtime và biết **từ chối** khi không chắc chắn

---

## 2. Kiến trúc tổng thể

```text
Webcam (BGR, ~15 fps)
  → frame_padding 1280×1280
  → MediaPipe Pose + Hand landmarker
  → extract(): 277 đặc trưng / frame
  → process_sequence(): chuỗi (30, 316)
  → Ensemble 6 model (5 fold + 1 full-data) + từ chối
  → WordVoter (bỏ phiếu chốt từ)
  → LLM Groq ghép câu
```

- Một luồng xử lý **dùng chung cho train và serve**, nên không lệch đặc trưng giữa hai bên

---

## 3. Cấu trúc mã nguồn

| Thư mục | Vai trò |
|---|---|
| `src/builtin/mediapipe_builtin` | Bọc MediaPipe Pose/Hand (`LandmarkerLoader`), model `.task` đóng gói sẵn |
| `src/features` | Chuẩn hoá khung hình, trích đặc trưng, mở camera |
| `src/data` | Thu thập, đọc/ghi dataset, trích lại, augment |
| `src/model` | Huấn luyện, dự đoán, đánh giá, chẩn đoán |
| `src/app` | Realtime CLI, nhánh nhanh, bỏ phiếu, ghép câu, Flask server, mô phỏng |
| `external/vsl-web` | Web app Next.js |
| `notebooks/` | Quy trình làm việc từng bước |
| `tests/` | 12 module, 105 test `unittest` |

---

## 4. Dữ liệu

- **16 lớp:** `age`, `cảm ơn`, `dấu nặng`, `g`, `h`, `i`, `k`, `m`, `n`, `r`, `t`, `ten_la`, `toi`, `u`, `xin chào`, `địa chỉ`
  - Từ vựng: chào hỏi, giới thiệu bản thân
  - Chữ cái đánh vần ngón tay và dấu thanh
- **15 video gốc / lớp** (`h`, `m`: 30) → mỗi lớp được bổ sung bản augment cho đủ 100 mẫu (tổng 1.600)
- Lưu trữ (`DataHandler`):
  - `data/raw/<lớp>/<id>.mp4`: video gốc
  - `data/processed/<lớp>/<id>/<frame>.npz` + `meta.json`: đặc trưng từng frame
- Tên lớp chuẩn hoá Unicode NFC

---

## 5. Đặc trưng — 4 thành tố của ký hiệu

| Thành tố | Nhóm | Số chiều |
|---|---|---:|
| **Vị trí** | `hand_pos`, `inter_hand_vec`, `body_contact`, `hand_contact`, `face_vec`, `hand_scale` | 61 |
| **Thủ hình** | `handshape` (góc khớp, độ duỗi, độ xoè, tiếp xúc ngón), `hand_local` (20 điểm trong hệ lòng bàn tay) | 196 |
| **Hướng bàn tay** | `orientation` (palm dir/normal, góc cổ tay) | 18 |
| **Hiện diện** | `presence` trái/phải | 2 |
| | **Tổng / frame** | **277** |
| **Chuyển động** | đạo hàm vị trí + hướng (thêm trong `process_sequence`) | +39 → **316** |

- Toạ độ trong **hệ trục thân**: tâm giữa hai vai, đơn vị = độ rộng vai
- Hiện chưa dùng thành tố phi thủ công (biểu cảm mặt, miệng)

---

## 6. Tiến hoá bộ đặc trưng

| Phiên bản | Thay đổi chính |
|---|---|
| v2 | 131 chiều/frame, chuỗi (30, 170) |
| v3 | Bỏ qua tay đang nghỉ (`REST_Y`), tách cột trái/phải/chung |
| v4 | `hand_local` (bất biến khi lật gương), `face_vec` (tay − mũi) |
| **v5** | `hand_scale` (độ sâu tay ổn định hơn z của pose); giảm trọng số z khi đo tiếp xúc thân (`CONTACT_Z`) → tách được `toi` và `m` |

- Mọi module suy ra chỉ số cột từ `FEATURE_SLICES` / `SIDE_COLUMNS`, không hard-code
- Đổi extractor → tăng `FEATURE_VERSION` → trích lại toàn bộ dataset (`reextract.py`)

---

## 7. Xử lý chuỗi `process_sequence`

1. **Bỏ tay nghỉ:** tay có lòng bàn tay thấp hơn bụng (`REST_Y`) bị coi là vắng mặt
2. **Lấp khoảng trống:** nội suy các đoạn mất tay ≤ 6 frame; dài hơn → 0 + cờ presence
3. **Khử gai:** sửa lỗi detect kéo dài đúng 1 frame, giữ cử động thật
4. **Co giãn thời gian** về đúng 30 frame
5. **Thêm chuyển động:** vận tốc vị trí tay và hướng bàn tay

→ Cùng một hàm cho dữ liệu huấn luyện, API và luồng realtime

---

## 8. Mô hình — 3 kiến trúc

| Kiến trúc | Ý tưởng |
|---|---|
| **VSLNet** (đang dùng) | `GroupStem` theo nhóm đặc trưng → 2 khối Conv đa tỉ lệ (kernel 3/5/9) → BiGRU → attention + mean + max pooling → head linear/cosine |
| **HandPairNet** | Một stem dùng chung cho hai tay, token học được cho tay vắng mặt |
| **SignFormer** | Encoder attention trên 20 điểm `hand_local` + Conv + 2 lớp Transformer pre-LN (chưa kiểm chứng trên ladder) |

- VSLNet: ~260k tham số/model
- Mọi kiến trúc dùng chung một bước chuẩn hoá: `presence_norm` chỉ tính thống kê trên các frame có tay

---

## 9. Huấn luyện

- **Cross-validation 5 fold** `StratifiedGroupKFold`, nhóm theo mẫu gốc
- **Hàm mất mát:** CE + label smoothing + **hard-negative margin** (phạt lớp đối thủ mạnh nhất)
- **Tối ưu:** AdamW, warmup + cosine, EMA, early stopping
- **Augment online:** time-warp, nhiễu, che frame, **chèn tay gây nhiễu** vào clip một tay
- `AdaptiveController` tự chỉnh dropout và cường độ augment theo khoảng cách train/val
- `select_after_warmup`: không chọn checkpoint khi model còn trong giai đoạn warmup
- Sau CV: temperature scaling + ngưỡng từ chối trên OOF → train thêm model full-data
- Kết quả: `runs/<tên>/ensemble.pt`, `metrics.json`, `folds.json`

---

## 10. Chống rò rỉ dữ liệu

- Bản augment ghi `origin_sample_id` → **luôn cùng fold** với mẫu gốc
- Validation và OOF **chỉ chấm mẫu gốc** (`validation_mask`)
- Không bao giờ augment một bản augment
- Fine-tune (`vsl_model_finetuner`) giữ đúng fold cũ theo `folds.json`
- Rà soát độc lập (18/11): **không phát hiện rò rỉ**

---

## 11. Suy luận (`src/model/predict.py`)

- `VSLPredictor`: nạp checkpoint một lần
  - `predict_raw`, `predict_processed`, `predict_frames`
- Kết quả dạng JSON: `label`, `confidence`, `probability_gap`, `accepted`, `rejection_reason`, `top`
- **Từ chối** khi:
  - `low_confidence`: xác suất < 0.65
  - `ambiguous`: chênh lệch giữa hai lớp đứng đầu < 0.08
- `VSLStreamSession` cho video liên tục
  - Mỗi client có landmarker và lịch sử frame riêng
  - Cửa sổ 2 s, giới hạn 15 fps

---

## 12. Từ nhãn đến câu

- **`WordVoter`** (`voting.py`): một nhãn phải thắng N phiếu liên tiếp mới được chốt
  - Từ vừa chốt bị chặn cho tới khi tay rời khung → không lặp từ khi giữ nguyên tay
- **Nhánh nhanh** (`fast.py`, `--fast`):
  - Mỗi frame chỉ chạy 1 model (~1/6 chi phí)
  - Phiếu "mạnh" (≥ 0.9, gap ≥ 0.5) được chốt sớm sau 1 s
  - Cả ensemble xác nhận lại ngay trước khi chốt từ
- **`SentenceComposer`** (`sentence.py`): LLM trên Groq ghép nhãn thô thành câu
  - Gom chữ cái đánh vần thành một từ, thêm hư từ, không bịa thông tin
  - Không có key hoặc lỗi mạng → ghép thô

---

## 13. Flask API (`src/app/server.py`)

| Endpoint | Chức năng |
|---|---|
| `GET /` | Trang test (webcam hoặc file video) |
| `GET /api/health` | Trạng thái, checkpoint, danh sách lớp |
| `POST /api/predict` | Gửi đủ 30 frame (hoặc feature) → kết quả |
| `POST /api/stream` | Mở phiên, server giữ MediaPipe và cửa sổ 30 frame |
| `POST /api/stream/<id>` | Chỉ gửi frame **mới** → ~0.15 s/request (so với ~1.5 s khi gửi cả cửa sổ) |
| `DELETE /api/stream/<id>` | Đóng phiên (tự đóng khi rảnh quá lâu) |

---

## 14. Web app (`external/vsl-web`)

- Next.js 16 + React 19, HTML/CSS thuần
- Thu webcam 15 fps, gửi JPEG tới `/api/stream`
- **3 mức tốc độ:**

| Mức | Cửa sổ | Chốt từ sau |
|---|---|---|
| Chuẩn | 30 frame | ~1.5 s |
| Nhanh | 20 frame | ~1.0 s |
| Rất nhanh | 15 frame | ~0.8 s |

- Nghỉ tay 3 s → kết thúc câu → `/api/compose` gọi Groq
- Người dùng tự nhập Groq API key; key chỉ được chuyển tiếp, không lưu trên server

---

## 15. Quy trình làm việc (notebooks)

1. `data_collect`: quay clip có nhãn + trích đặc trưng
   - `alphabet_data_collector`: chữ cái tĩnh, quay 1 mẫu rồi tự sinh bản nhiễu
2. `reextract_features`: trích lại sau khi đổi extractor
3. `augment_features`: tạo biến thể đặc trưng (tuỳ chọn)
4. `train`: CV + ensemble
   - `vsl_model_finetuner`: warm-start khi chỉ thêm dữ liệu
5. `realtime_predict`: chạy webcam trực tiếp

CLI: `python -m src.app.realtime [--fast]` · `python -m src.app.server` · `python -m src.app.simulate`

---

## 16. Kết quả

| Run | Lớp | Mẫu | CV accuracy | Nhầm lẫn |
|---|---:|---:|---:|---|
| `vsl_v3_s6` | 12 | 540 | 99.4% | cảm ơn → xin chào (1) |
| `vsl_v4_s6` | 14 | 630 | 100% | không có |
| **`vsl_v5_20261119`** | **16** | **1.600** | **99.6%** (macro-F1 99.6%) | cảm ơn → xin chào (1) |

- Cặp gần nhau nhất: `cảm ơn`/`xin chào`, `h`/`k`
- Thời gian từ frame cuối đến khi có dự đoán: p50 ≈ 23 ms, p95 ≈ 42 ms
- MediaPipe chiếm >95% chi phí (~34 ms/frame trên M3)

---

## 17. Thực nghiệm có kiểm soát (ladder)

- Cùng fold, 3 seed, mỗi bước thêm một thay đổi (`runs/experiments/p2`, `p3`)
- Accuracy đã bão hoà, nên đo thêm các chỉ số khác:
  - `distractor_flip_rate`: chèn tay lạ vào clip một tay, đếm tỉ lệ đổi nhãn
  - Open-set: bỏ hẳn một lớp, xem model có nhận nhầm không
  - Độ tách các cặp khó, độ bền với nhiễu
- **Kết quả chính:** `presence_norm` + `distractor_geometry` giảm flip rate **0.115 → 0.026** (3.3σ)
- Bộ cấu hình S6 hiện là mặc định của notebook

---

## 18. Hạn chế và rủi ro

- **CV đang đánh giá quá cao:** mỗi lớp gần như chỉ quay trong **một buổi** với một người ký
  - Session trùng với lớp, nên model có thể học bối cảnh quay thay vì ký hiệu
  - Cặp `cảm ơn`/`xin chào` quay chung buổi cũng chính là cặp khó tách nhất
- Augment chỉ là nhiễu đặc trưng, gần như nhân bản mẫu gốc
- Chưa có dữ liệu âm ("không ký"), nên ngưỡng từ chối chưa được hiệu chỉnh thật
- Temperature chạm biên 1.0 → xác suất chưa được hiệu chỉnh
- Từ vựng nhỏ (16 lớp), chưa có đặc trưng phi thủ công, chưa nhận dạng câu liên tục thật sự

---

## 19. Hướng phát triển

**Ưu tiên cao**
- Thu thêm dữ liệu ở nhiều buổi, nhiều người ký; tách tập test theo session
- Thu dữ liệu âm để hiệu chỉnh ngưỡng từ chối

**Trung hạn**
- Kiểm chứng SignFormer và cosine head trên ladder v4 (F2, F3)
- Mở rộng từ vựng, thêm đặc trưng mặt và miệng

**Dài hạn**
- Mô hình causal cho streaming, phân đoạn câu liên tục
- Triển khai ONNX / on-device

---

# Tóm tắt

- Pipeline hoàn chỉnh: **webcam → landmark → 316 đặc trưng → ensemble → bỏ phiếu → LLM → câu**
- Đặc trưng bám theo 4 thành tố ngôn ngữ ký hiệu, bất biến khi lật gương, có kiểm soát phiên bản
- Huấn luyện có chống rò rỉ, từ chối khi không chắc, và có thực nghiệm đo độ bền
- Đã chạy được trên CLI, Flask API và web app
- **Bước tiếp theo quan trọng nhất:** dữ liệu đa dạng hơn để đo đúng khả năng tổng quát hoá
