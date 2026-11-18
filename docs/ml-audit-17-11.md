# Rà soát toàn diện ML pipeline VSL — 17/11/2026

Phạm vi: Phase 1–3 (audit, chẩn đoán, lộ trình). **Chưa sửa mã nguồn nào.** Phase 4–5 (triển khai, đánh giá) chờ phê duyệt.

Thực hiện bởi 3 agent chỉ đọc, phiên chính tổng hợp:

- **Agent 1 — Dữ liệu và đặc trưng.** Kịch bản và số liệu nằm trong `/tmp/vsl_audit/agent1/`.
- **Agent 2 — Mô hình và huấn luyện.** Đã chạy lại baseline và 8 biến thể. Bản baseline tái lập *bit-exact* run `vsl_20261117_205525`. Kịch bản nằm trong `/tmp/vsl_audit/agent2/`.
- **Agent 3 — Đánh giá, suy luận và latency, đồng thời phản biện độc lập Agent 1 và 2.** Kịch bản nằm trong `/tmp/vsl_audit/agent3/`.

Nhãn dùng cho mỗi phát hiện:

- **[BUG]**: đã kiểm chứng bằng mã hoặc số liệu.
- **[NGHI]**: nghi vấn có cơ sở.
- **[GT]**: giả thuyết nghiên cứu.

`docs/tue-17-11.md` là đánh giá trước đó. Các nhận định của nó được kiểm tra lại ở đây; tài liệu này không lặp lại chúng.

---

## 0. Kết luận chính

1. **CV = 1.000 không chứng minh mô hình nhận dạng được ký hiệu.**
   - Mỗi lớp được quay trong đúng **một session**. Ngoại lệ: `m` có 2 session cách nhau 50 giây, còn `cảm ơn` và `xin chào` dùng chung một session. Vì vậy session trùng với lớp.
   - Riêng `fps` đã phân loại được 16.1% (mức ngẫu nhiên là 8.3%).
   - Bằng chứng mạnh nhất: `cảm ơn`/`xin chào` là **cặp duy nhất cùng session**, và cũng là cặp **kém tách nhất** trong embedding. Tỉ lệ tách của nó là 36, trong khi mọi cặp khác nằm trong khoảng 67–419. Agent 1 và Agent 2 đi đến kết quả này bằng hai phương pháp độc lập. Khi không còn tín hiệu session, nội dung ký hiệu khó tách hơn nhiều so với con số CV.
2. **Không thể xếp hạng cải tiến bằng accuracy trên dữ liệu hiện tại.**
   - n = 180 mẫu gốc, 0 lỗi; khoảng tin cậy Wilson 95% là [0.98, 1.00].
   - Mọi biến thể kiến trúc đều chênh nhau không quá 1/180.
   - Cần một **tập test session mới** (§6) và các metric thay thế (§5.3).
3. **Có bốn lỗi thật trong huấn luyện và hiệu chuẩn, sửa được ngay.** Chúng không phụ thuộc dữ liệu mới:
   - Temperature bị kẹp ở biên 0.2, nên nó *làm sắc* xác suất thay vì hiệu chuẩn.
   - Mô hình full-data chỉ được huấn luyện trong giai đoạn warmup.
   - Checkpoint được chọn khi mô hình còn trong warmup.
   - CLI huấn luyện bỏ qua `validation_mask`.
4. **Augmentation hiện tại gần như nhân bản mẫu gốc.**
   - Nhiễu chỉ dịch chuyển mẫu khoảng 4–13% độ phân tán thật trong lớp.
   - Số này đo trên đặc trưng đã lấy trung bình theo thời gian, nên có thể đánh giá thấp nhiễu thực tế ở từng frame.
5. **Về latency:**
   - Thời gian từ frame cuối đến khi có dự đoán là p50 ≈ 23 ms, p95 ≈ 42 ms, max ≈ 66 ms, **đạt yêu cầu ≤ 1 s**.
   - Tuy nhiên `min_window_seconds = 1.5` khiến dự đoán **đầu tiên** chỉ có sau khoảng 1.6–1.7 s kể từ khi bắt đầu ký.
   - Đây là mâu thuẫn cấu hình, không phải vấn đề hiệu năng (§8).

---

## 1. Kiến trúc pipeline hiện tại

```text
OpenCV frame (BGR, ~15 fps lúc thu; 7.8–10.9 fps ở session cảm ơn/xin chào)
  │  [mirror_data tuỳ chọn — không lưu vào meta]
  ▼
frame_padding 720×720                        src/features/frame.py
  ▼
MediaPipe Pose + Hand (VIDEO mode)           src/builtin/mediapipe_builtin/landmarkers_loader.py
  │  ~34 ms/frame trên M3 (>95% chi phí)
  ▼
extract() → 131 đặc trưng/frame              src/features/extract.py
  │  - Tâm: giữa hai vai. Đơn vị: độ rộng vai. Xoay: chỉ trong mặt phẳng.
  │  - Gán tay trái/phải theo cổ tay gần nhất.
  │  - NaN và cờ presence khi mất tay. _fill_gaps lấp các khoảng hở ≤ 6 frame.
  ▼
DataHandler: data/processed/<lớp>/<id>/*.npz + meta.json     src/data/data_reader.py
  │  augment.py: nhiễu Gaussian i.i.d. theo frame → 30 bản/mẫu gốc (origin_sample_id)
  ▼
process_sequence → (30, 170)                 extract.py:358 (dùng chung train/serve)
  │  Nội suy theo SỐ frame, không theo thời gian thực.
  │  Thêm khối motion (gradient) cho hand_pos + inter_hand + orientation.
  ▼
VSLNet 259k tham số                          src/model/training.py
  │  GroupStem → 2×MultiScaleBlock → BiGRU → attention pooling → linear head
  │  CE + label smoothing 0.1 + hard-negative margin + mixup + EMA
  │  5 fold (StratifiedGroupKFold theo origin) + 1 model full-data = ensemble 6
  │  fit_temperature và choose_reject_threshold trên OOF mẫu gốc
  ▼
VSLPredictor / VSLStreamSession              src/model/predict.py
     Cửa sổ 2.0 s (tối thiểu 1.5 s), dự đoán mỗi 3 frame, ổn định 3 lần.
     Chấp nhận khi p ≥ max(0.2, 0.65) và gap ≥ 0.08.
```

---

## 2. Dữ liệu

| Mục | Giá trị |
|---|---|
| Lớp có dữ liệu | 12 (`i`, `y` rỗng, bị loại tự động) |
| Mẫu gốc | 15 mỗi lớp, tổng 180 |
| Mẫu augment | 30 mỗi lớp, tổng 360 |
| Session | 1 mỗi lớp. `m` có 2; `cảm ơn` và `xin chào` chung 1 |
| fps | 14.9–15.6. Riêng session cảm ơn/xin chào là 7.8–10.9 |
| Số frame | 16–32, tương quan với lớp |

### Phát hiện

- **[BUG] Session trùng với lớp**, nên CV theo từng clip không đo được khả năng tổng quát hoá.
  - `train.ipynb` có tuỳ chọn `GROUP_BY="auto"`. Vì mỗi lớp chỉ có 1 session, nó âm thầm lùi về nhóm theo từng mẫu và chỉ in cảnh báo.
- **[BUG] fps không cố định và tương quan với lớp.** Chỉ riêng fps đạt 16.1% (ngẫu nhiên 8.3%).
  - `process_sequence` nội suy theo số frame, nên khối motion là "đơn vị trên frame đã resample", không phải trên giây (extract.py:388-395).
  - [GT] Điều này khuếch đại dấu vân tay của session.
- **[NGHI] Bộ đặc trưng "nuisance" đạt 66.1%** (fps, số frame, presence, body_contact, hand_contact, khoảng cách hai tay).
  - *Phản biện của Agent 3:* body_contact, hand_contact và inter_hand là tham số ngôn ngữ thật (vị trí, tiếp xúc), không chỉ là nhiễu thu hình. Vì vậy con số 66% **phóng đại** phần confound.
  - Bằng chứng confound sạch là fps 16.1% và cặp cảm ơn/xin chào.
- **[NGHI] Tay trái gần như vắng mặt ở các chữ cái**, nhưng có mặt 36–95% ở `age`, `ten_la`, `địa chỉ`.
  - Có thể đây là tín hiệu một tay / hai tay thật, nhưng hiện trùng với session.

---

## 3. Đặc trưng

Phân tích trên mẫu gốc: đặc trưng 170 chiều, lấy trung bình theo thời gian, logistic regression 5-fold cho từng nhóm riêng lẻ.

| Nhóm | Số chiều | Fisher | Acc riêng nhóm | Đánh giá |
|---|---|---|---|---|
| handshape | 64 | 5.09 | **0.983** | Mang tín hiệu thật mạnh nhất |
| orientation | 18 | 5.07 | 0.956 | Mạnh. [GT] Có thể lẫn tư thế nghiêng giữa các session (z chỉ được giảm, không chuẩn hoá) |
| hand_pos | 18 | 4.90 | 0.939 | Mạnh, nhưng nghi lẫn session |
| motion | 39 | 0.68 | 0.517 | Yếu; phụ thuộc fps |
| body_contact | 22 | 0.51 | 0.400 | Fisher thấp nhất |
| presence | 2 | 4.77 | 0.400 | |
| inter_hand_vec | 3 | 4.23 | 0.350 | |
| hand_contact | 4 | 10.11 | 0.300 | |
| Toàn bộ | 170 | — | 0.994 | |

Nhận xét chung về extractor:

- Bất biến tịnh tiến và tỉ lệ được làm đúng (extract.py:200-213).
- Gán tay theo cổ tay gần nhất (extract.py:148-155) là thiết kế đúng: không phụ thuộc nhãn handedness của MediaPipe, vốn thay đổi khi lật ảnh.
- Cờ presence không bao giờ bị nội suy. Đây là đúng.

Một điểm còn yếu: [GT] khối motion không chứa động học ngón tay (handshape được loại khỏi motion), và chỉ xoay trong mặt phẳng.

---

## 4. Augmentation

- **[BUG, thiết kế] `add_feature_noise` (augment.py:103-116) quá yếu.**
  - Với sigma 0.01–0.015 × strength 1.0, bản augment dịch chuyển khoảng 4–13% khoảng cách tới mẫu thật gần nhất cùng lớp.
  - Dịch chuyển này chỉ bằng khoảng 1–3% khoảng cách tới tâm lớp đối thủ.
  - Phép đo dùng trung bình theo thời gian, nên đánh giá thấp nhiễu i.i.d. ở từng frame.
- Augmentation đúng khi chỉ chạm vào hình học (0:129) và không chạm presence. Cần giữ bất biến này.
- **[BUG] Augmentation "tay thứ hai" trong `training.py`** kiểm tra `presence > 0` trên một giá trị presence mềm (training.py:255, :470). Kết quả là trạng thái một tay / hai tay bị lật ở 13.9% clip.
  - Nguồn tay trái chỉ lấy từ `age`, `ten_la`, `địa chỉ`.
- **[NGHI] Time-warp áp lên cột vận tốc** nhưng không đổi tỉ lệ vận tốc theo.

---

## 5. Mô hình và huấn luyện

### 5.1 Lỗi đã kiểm chứng (Agent 2, Agent 3 xác nhận)

| ID | Vị trí | Lỗi | Hệ quả đo được |
|---|---|---|---|
| F1 | training.py:638-663, gọi ở :876 | `fit_temperature` chạy tới biên `t_min = 0.2` vì OOF không có lỗi. Không có biên thì T = 0.04. Nguyên nhân gốc: label smoothing 0.1 giới hạn khoảng cách logit ở 4.69, rồi T kéo giãn lại. | Xác suất bị làm sắc ×5. Xem chi tiết bên dưới bảng. |
| F2 | :666-680 | Ngưỡng reject luôn bằng mức thấp nhất của lưới (0.2). Không có dữ liệu âm. Predictor ghi đè bằng 0.65, cũng chưa hiệu chuẩn (predict.py:48,71). | Ngưỡng vô nghĩa |
| F3 | :900, :63, :520-524 | `ep_full = int(4·1.15) + 1 = 5`, bằng đúng `warmup_epochs = 5`. Model full-data không bao giờ vào pha cosine. | 1/6 ensemble huấn luyện dở dang, không được đánh giá |
| F4 | :580 | Chọn checkpoint theo `(val_acc, −val_loss)`. Khi acc bão hoà, val_loss tăng dần tới sàn của label smoothing, nên chọn epoch 3–4, lúc còn warmup và EMA ≈ 0.88. Cùng tập mẫu gốc này lại dùng để fit T và ngưỡng. | Bias chọn lựa |
| F5 | :418-428, :576, :582-588 | `AdaptiveController` không hoạt động (strength = 0.5 ở cả 40 fold) nhưng có đọc `va_acc`. | Code thừa; rò rỉ tiềm ẩn |
| F6 | :447-453 | Hard-negative margin 0.8, nhỏ hơn nhiều so với khoảng cách 4.69, nên không bao giờ kích hoạt. | Loss không có tác dụng |
| CLI | :939-964 | `main()` không truyền `validation_mask`, nên validate cả trên bản augment. | Accuracy phồng nếu dùng CLI |

Hệ quả chi tiết của F1:

- Tỉ lệ chấp nhận nhầm lớp chưa thấy (hold-one-class-out) đổi như sau khi T ≈ 0.2 so với T = 1:

  | Lớp bị giữ lại | T ≈ 0.2 | T = 1 |
  |---|---|---|
  | `toi` | 87% | 7% |
  | `cảm ơn` | 80% | 13% |
  | `địa chỉ` | 100% | 80% |
  | `m` (đều bị đoán là `n`) | 100% | 100% |

- Nhiễu Gaussian thuần bị chấp nhận 64% ở T ≈ 0.2, so với 0% ở T = 1.

Phần còn lại đã kiểm tra và **đúng**: shape và mask của loss, mixup, EMA (kể cả các buffer BN), chế độ eval, deepcopy trạng thái tốt nhất, công thức time-warp. Huấn luyện tái lập bit-exact trên MPS với seed 42. Lưu ý: các fold dùng chung một luồng RNG, nên thay đổi `folds_to_run` sẽ đổi kết quả của các fold sau.

### 5.2 Kiến trúc so với dữ liệu

- Model có 259k tham số; ensemble khoảng 1.56M, cho 180 clip độc lập.
- Đảo ngược thời gian cho acc 1.000, xáo trộn frame cho 0.994, chỉ dùng frame giữa cho 0.972. Các lớp giảm khi chỉ dùng frame giữa là `cảm ơn` (0.80) và `age` (0.87).
- Bag-of-frames với 16.5k tham số đạt 0.994. GRU-only (46k) và TCN nhỏ (26k) cũng đạt 0.994.
- *Phản biện của Agent 3:* điều này phù hợp với một bộ từ vựng phần lớn là chữ cái tĩnh, cộng với confound. Nó **chưa** là lý do để thu nhỏ model. Chỉ quyết định khi có dữ liệu nhiều session.

### 5.3 Metric thay thế để xếp hạng biến thể khi acc bão hoà

Chạy ≥ 3 seed với fold cố định:

1. **Tỉ lệ chấp nhận nhầm khi giữ lại một lớp, kèm tỉ lệ chấp nhận nhầm với input không phải ký hiệu** (nhiễu, không có tay). Đây là metric chính.
2. Phần trăm khoảng cách logit trung vị còn giữ được khi thêm nhiễu σ = 1. Kết quả hiện tại:

   | Biến thể | Giữ được |
   |---|---|
   | Không label smoothing | 0.92 |
   | Baseline | 0.83–0.85 |
   | Cosine head | 0.73 |
   | Bag-of-frames | 0.56 |

3. Độ tách theo cặp khó (khoảng cách giữa tâm lớp / độ phân tán trong lớp) và p10 của biên centroid.
4. Kiểm tra độ nhạy thứ tự trên các lớp động.

---

## 6. Giao thức đánh giá

- **Đúng:**
  - Fold giữ trọn nhóm origin; 0/180 nhóm bị chia ra nhiều fold.
  - Theo đường notebook, OOF chỉ tính trên mẫu gốc.
  - Đường phục vụ khớp với đặc trưng đã lưu: 60/60 nhãn trùng, chênh lệch confidence tối đa 0.0034.
- **Thiếu:**
  - Tập test giữ riêng.
  - Macro-F1, P/R/F1 theo lớp, ma trận nhầm lẫn đầy đủ (hiện chỉ có `confusion_pairs`).
  - Khoảng cách embedding, acc/AUC theo cặp khó, ECE, đường coverage–accuracy.
- **Thống kê:** n = 3 mẫu/lớp/fold cho CI [0.44, 1.00]; n = 180 cho CI [0.98, 1.00]. Không thể phát hiện chênh lệch ±5–10 điểm phần trăm.
- **Đề xuất giao thức:**
  - **Tập test session mới.** Mỗi lớp ≥ 8 clip, lý tưởng là 20, quay vào ngày khác với ánh sáng và quần áo khác. Chỉ dùng **một lần**, sau khi đã cố định mọi tinh chỉnh.
  - **Dữ liệu huấn luyện.** Mỗi lớp ≥ 2 session, ví dụ 8 + 7 clip, để chạy được `GROUP_BY="session"`.
  - **Lớp âm "không ký".** Dùng để hiệu chuẩn ngưỡng reject.

---

## 7. Cặp khó

| Cặp | Khoảng cách tâm (đặc trưng chuẩn hoá) | Độ tách embedding | Cùng session? | Nhận định |
|---|---|---|---|---|
| m – n | 6.22 (gần nhất) | 151 | Khác | Khác nhau chủ yếu ở orientation và vị trí tay. Hold-out `m` thì 100% bị đoán là `n` → khó thật về đặc trưng |
| r – u | 6.91 | 419 | Khác | Khác nhau ở orientation |
| g – n | 6.82 | — | Khác | |
| g – t | xa (handshape 9.16) | 267 | Khác | **Không phải cặp khó** |
| cảm ơn – xin chào | — | **36 (thấp nhất)** | **Cùng** | Cặp duy nhất không có tín hiệu session → khó nhất |
| age – ten_la | — | 67 | Khác | |

Ma trận nhầm lẫn OOF hiện tại: 179/180 nằm trên đường chéo. Lỗi duy nhất là một clip `cảm ơn`. Với dữ liệu này, ma trận nhầm lẫn **không cung cấp thông tin** về cặp khó. Phải dựa vào hình học đặc trưng và embedding như bảng trên, và cuối cùng là tập test session mới.

---

## 8. Suy luận và latency

Đo trên Apple M3: MediaPipe chạy CPU, model chạy MPS. Frame được giải mã từ `data/raw/*.mp4` vì webcam chưa đo. Không tính khởi tạo và warm-up.

| Giai đoạn | p50 | p95 | max |
|---|---|---|---|
| Giải mã frame | 0.25 ms | 0.43 | 2.07 |
| frame_padding | ~0 | 0.07 | 1.23 |
| MediaPipe pose + hand | **33.7 ms** | 42.6 | 170.5 |
| extract | 0.31 ms | 0.41 | 1.95 |
| process_sequence | 0.56 ms | | |
| Forward ensemble 6 model | 8.2 ms | 9.9 | 10.9 |
| `predict_processed` (gồm hậu xử lý) | 22.8 ms | 42.5 | 65.8 |
| **Frame cuối → dự đoán** | **≈ 23 ms** | **≈ 42 ms** | **≈ 66 ms** |
| **Bắt đầu ký → dự đoán đầu tiên** | **≈ 1.6–1.7 s** | | |

Nhận xét:

- Mỗi frame tốn khoảng 34 ms, nên trần khoảng 29 fps. Thu 30 fps thì gần như không còn dư.
- **[BUG, cấu hình]** `StreamConfig.min_window_seconds = 1.5` (predict.py:24, chặn ở :190) mâu thuẫn với yêu cầu ≤ 1 s nếu tính từ lúc bắt đầu ký.
  - Có thể giảm cửa sổ xuống ≤ 1.0 s, nhưng phải kiểm chứng: `cảm ơn` và `age` mất 13–20 điểm khi chỉ còn một frame.
- **[NGHI]** Timestamp trong `predict_frames` lấy theo đồng hồ thật, còn `reextract.py:61` lấy theo fps. Đo thực tế thì không có ảnh hưởng.
- **[NGHI]** `mirror_data` không được lưu vào meta hay checkpoint. `feature_version` được lưu nhưng không được kiểm tra (predict.py:70).
- **[GT]** `stability_frames = 3` với `predict_every = 3` khiến các cửa sổ chung nhau khoảng 95% frame, nên cơ chế ổn định gần như không độc lập.

---

## 9. Lộ trình ưu tiên

Nguyên tắc chung:

- Mỗi thay đổi là một thí nghiệm có đối chứng: cùng fold, ≥ 3 seed, dùng metric ở §5.3.
- Giữ nguyên baseline `runs/vsl_20261117_213118`.
- Không gộp nhiều thay đổi vào một lần chạy.

### P0 — Dữ liệu (người dùng thực hiện; điều kiện để chứng minh mọi cải tiến)

| # | Việc | Metric chứng minh |
|---|---|---|
| D1 | Quay tập **test session mới**: 12 lớp × ≥ 8 clip, cất riêng, không dùng để tinh chỉnh | Acc và macro-F1 cross-session so với CV |
| D2 | Thêm ≥ 1 session huấn luyện cho mỗi lớp; hai session cân bằng số clip | CV theo `GROUP_BY="session"`; acc chỉ dùng fps giảm về ~8% |
| D3 | Cố định fps khi thu và ghi timestamp từng frame | Acc chỉ dùng fps giảm về ~8% |
| D4 | Quay lớp âm (không ký, tay buông, gãi đầu…) | Tỉ lệ chấp nhận nhầm input không phải ký hiệu |

### P1 — Sửa lỗi nhanh (không cần dữ liệu mới)

| # | Chủ sở hữu | Thay đổi | Metric |
|---|---|---|---|
| Q1 | Agent 2, `training.py` | Temperature: không fit khi OOF không có lỗi (giữ T = 1), hoặc ràng buộc T ≥ 1; cảnh báo khi T chạm biên | Chấp nhận nhầm khi giữ lại một lớp, chấp nhận nhầm nhiễu |
| Q2 | Agent 2, `training.py` | Model full-data: số epoch cố định vượt warmup (ví dụ max(median·1.15, warmup + 20)) | Độ bền với nhiễu, chấp nhận nhầm của model full so với model fold |
| Q3 | Agent 2, `training.py` | Chỉ chọn checkpoint sau warmup + 5 epoch | Độ bền với nhiễu (3 seed) |
| Q4 | Agent 2, `training.py` | Sửa kiểm tra presence của augmentation tay thứ hai (dùng ngưỡng 0.5) | % clip bị lật một tay / hai tay → 0 |
| Q5 | Agent 2, `training.py` | CLI: nạp và truyền `validation_mask`; từ chối chạy nếu có bản augment mà không có mask | Unit test |
| Q6 | Agent 3, `evaluation.py` mới + `run_training` | Ma trận nhầm lẫn, P/R/F1 theo lớp, macro-F1, top-k, acc theo cặp khó, CI bootstrap theo nhóm; ghi vào `metrics.json` | Có trong `metrics.json` |
| Q7 | Agent 3, `src/model/` | Script chẩn đoán: chấp nhận nhầm khi giữ lại một lớp, chấp nhận nhầm nhiễu, độ bền σ, độ tách cặp khó (chuyển từ `/tmp/vsl_audit/agent2/common.py`) | Tái lập được số liệu baseline |
| Q8 | Agent 3, `predict.py` | Báo lỗi khi `feature_version` không khớp | Unit test |

### P2 — Thí nghiệm có đối chứng trên dữ liệu hiện tại (sau P1)

| # | Chủ sở hữu | Thí nghiệm | Ghi chú |
|---|---|---|---|
| E1 | Agent 2 | Chỉ giữ label smoothing **hoặc** margin loss, không cả hai (bỏ LS: độ bền 0.92 so với 0.83) | Làm sau Q1 |
| E2 | Agent 2 | Bỏ `AdaptiveController`, dùng cosine LR thuần | Kỳ vọng kết quả không đổi, ít mã hơn |
| E3 | Agent 1 | Quét strength của augmentation (1 → 3) và thử time-warp / frame-drop | Không được làm bản augment vượt sang tâm lớp đối thủ |
| E4 | Agent 3 | Cửa sổ 1.0 s và 1.2 s: cắt clip theo thật (không phải mask), dùng các probe framedrop / first_half | Quyết định đổi `min_window_seconds`, cần D1 để chốt |

### P3 — Kiến trúc (chỉ sau khi có D1–D2)

- Thu nhỏ model xuống 20–50k tham số (GRU-only hoặc TCN nhân quả).
- Reject theo khoảng cách tới prototype của lớp.
- SupCon/triplet với hard-negative mining. Chỉ thử khi đã có ≥ 2 session mỗi lớp; nếu không, phương pháp này học tín hiệu session.
- Không đề xuất center loss, và chưa đề xuất temporal attention hay motion stream.

### Đã quyết định giữa các ý kiến khác nhau

| Vấn đề | Quyết định | Căn cứ |
|---|---|---|
| "Nuisance 66%" so với "tham số ngôn ngữ" | Dùng fps 16.1% và cặp cảm ơn/xin chào làm bằng chứng confound; 66% chỉ là cận trên | Phản biện của Agent 3 |
| Thu nhỏ model ngay (Agent 2) | Hoãn đến P3 | Bất biến thứ tự có thể do từ vựng tĩnh và confound |
| g/t có phải cặp khó? | Không | Hai phương pháp độc lập khớp nhau |
| Rút cửa sổ ≤ 1 s | Thử nghiệm E4, chưa đổi mặc định | Lớp động mất 13–20 điểm khi chỉ còn một frame |

---

## 10. Giới hạn và hướng nghiên cứu

- Mọi số liệu ở đây là **trong-session**. Không kết luận được khả năng tổng quát hoá, kiến trúc tốt nhất, hay giá trị của metric learning cho m/n và r/u, cho đến khi có D1–D2.
- Chỉ có một người ký, nên tổng quát hoá sang người ký khác chưa được đo.
- Latency chưa đo với webcam thật (jitter, backpressure).
- Hướng tiếp theo:
  - Chuẩn hoá xoay 3D.
  - Đặc trưng động học ngón tay.
  - Resample theo thời gian thực thay vì theo số frame.
  - Lớp âm và nhận dạng ký hiệu liên tục (phân đoạn).

## 11. Kết quả P2 (18-11): tay thứ 2 lọt khung + cặp gần giống

**Thiết lập.** Đặc trưng v3 có 143 cột thô và chuỗi (30, 182). Thang thí nghiệm cộng dồn, mỗi bước 3 seed (42, 7, 1), giữ nguyên fold. Mọi metric chỉ tính trên mẫu gốc OOF (`runs/experiments/p2/`, gộp bằng `aggregate.py`). Bước nào chạy quá 40 phút thì bị tắt.

**Metric.**
- Đổi nhãn: lấy tay trái của một clip hai tay cùng fold, ghép vào clip một tay rồi xem dự đoán có đổi không (`distractor_flip_rate`). Có hai kiểu ghép: tay ghép giữ nguyên độ cao, hoặc tay ghép đứng yên và hạ xuống y=1.1.
- Open-set FA: lần lượt giữ lại m và địa chỉ khi train, đo tỉ lệ clip của lớp đó bị chấp nhận nhầm.
- Độ tách cặp khó: khoảng cách cosine giữa centroid hai lớp chia cho độ phân tán trong lớp.
- Độ bền với nhiễu: tỉ lệ logit gap giữ được khi thêm nhiễu σ=1.

Bảng là trung bình ± std qua 3 seed:

| Bước | Thay đổi | Đổi nhãn (ngang) | Đổi nhãn (hạ thấp) | Open-set FA | Bền nhiễu | m/n | r/u | g/n |
|---|---|---|---|---|---|---|---|---|
| S0 | baseline (code P1 + v2) | 7.3±3.9% | 13.0±2.9% | .78±.16 | .885 | 118 | 280 | 166 |
| S1 | + đặc trưng v3 (thứ tự ngón, lọc tay nghỉ) | 11.5±1.8% | **0%** | .67±.10 | .886 | 122 | 282 | 163 |
| S2 | + presence_norm | 3.6±2.2% | 0% | .76±.30 | .717 | 95 | 220 | 110 |
| S3 | + HandPairNet | 1.0±0.4% | 0% | .76±.19 | .748 | 83 | 249 | 125 |
| **S4b** | S2 + distractor_geometry (VSLNet) | **2.6±1.9%** | 0% | **.52±.11** | .726 | 121 | 245 | 118 |
| S4c | S1 + distractor_geometry | 2.6±1.3% | 0% | .53±.31 | .902 | 85 | 182 | 94 |

So sánh từng cặp seed với S1 (* = cùng dấu trên cả 3 seed và |Δ| lớn hơn độ dao động giữa các seed):

| Bước | Thay đổi so với S1 |
|---|---|
| S4b | đổi nhãn −8.9 điểm*, chấp nhận nhầm −4.4*, open-set FA −0.14*, bền nhiễu −0.16* (xấu đi), r/u và g/n giảm nhưng không có ý nghĩa |
| S4c | đổi nhãn −8.9*, bền nhiễu không đổi, r/u −100*, g/n −69* (xấu đi) |

**Quyết định.**
- **Chọn S4b**: VSLNet + `presence_norm` + `distractor_geometry`. Không bước nào qua được luật chặt, vì bước nào cũng có một metric xấu đi rõ.
  - S4b đánh đổi độ bền với nhiễu Gauss chung chung.
  - S4c đánh đổi đúng các cặp khó mà người dùng cần sửa.
- **Không giữ HandPairNet (S3)**: không hơn S2 một cách có ý nghĩa.
- **Loại LS=0**: open-set FA lên 1.0 (S5b, 1 seed).
- **S4, S5, S5c bị dừng giữa chừng.** S4 và S5 xây trên S3, mà S3 không được giữ. S5c xây trên S4c, mà S4c đã bị loại.

**Giới hạn cần nhớ.**
- Đổi nhãn khi tay hạ thấp bằng 0% là kết quả của cách dựng metric: y=1.1 lớn hơn `REST_Y`=0.9 nên bộ lọc tay nghỉ luôn bỏ tay đó. Con số này chỉ chứng minh bộ lọc chạy đúng, không chứng minh hiệu quả trên tay nghỉ thật.
- Metric đổi nhãn ghép tay theo cùng công thức với augmentation `distractor_geometry`. Vì vậy mức cải thiện của S4b/S4c trên metric này có thể bị thổi phồng.
- Mọi kết quả vẫn chỉ là trong-session. Cách duy nhất để chứng minh lỗi 1 đã hết là quay clip thật một tay, có tay kia trong khung, và chỉ dùng để đánh giá.

**Lệch train/serve.** Đặc trưng từ `predict_frames(fps=...)` khớp đặc trưng đã lưu trên 60 clip gốc, max|Δ|=0 (`skew_check.py`).

**Agent 3 duyệt (18-11, `runs/experiments/p2/review/`).**
- Không thấy rò rỉ: fold theo origin, metric chỉ tính trên mẫu gốc, T và ngưỡng fit trên OOF.
- Đổi nhãn khi tay thứ 2 lọt khung: VERIFIED, kèm lưu ý metric và augmentation dùng chung công thức ghép tay.
- Đổi nhãn bằng 0% khi tay hạ thấp: đúng theo cách dựng metric, chưa phải bằng chứng về tay nghỉ thật.
- Open-set FA: PLAUSIBLE.
- Cải thiện cặp khó: NOT SUPPORTED, vì 3 seed không đủ để thấy một hiệu ứng nhỏ hơn 1σ.
- Độ bền với nhiễu giảm khi bật presence_norm: Agent 3 cho là artefact của thang nhiễu. Đo tỉ lệ std toàn cục / std chỉ trên frame có tay cho cột tay trái được trung vị 0.61 và p90 2.5, tức lệch theo cả hai chiều. Vì vậy câu hỏi này **chưa ngã ngũ**. Cách kiểm tra đúng là đo lại với nhiễu tỉ lệ theo std chỉ trên frame có tay, cho mọi bước.
- Checkpoint `runs/vsl_v3_s4b` nạp đúng presence_norm (`presence_aware=True`).

### 11b. Khôi phục Q3 (S6 = S4b + chọn checkpoint từ epoch warmup + 5)

Lý do: các fold của S4b chọn epoch [3, 25, 6, 5, 2], tức phần lớn rơi vào pha warmup, khi EMA chưa ổn định. `Config.select_after_warmup=5` chỉ cho phép chọn checkpoint từ epoch 10 trở đi. Mặc định của tham số là `None`, giữ hành vi cũ.

Bảng dưới so từng cặp S6 với S4b trên cùng fold, cùng seed (42, 7, 1). Δ ghi theo thứ tự seed. Dấu \* nghĩa là cả 3 seed cùng chiều và |Δ trung bình| lớn hơn std của Δ.

| Metric | S4b | S6 | Δ theo seed |
|---|---|---|---|
| Độ tách m/n | 121 ± 47 | **263 ± 19** | +101 / +163 / +163 \* |
| Độ tách r/u | 245 ± 91 | **579 ± 51** | +274 / +337 / +391 \* |
| Độ tách t/n | 125 ± 65 | **270 ± 22** | \* |
| Độ tách g/n | 118 ± 53 | **329 ± 36** | \* |
| Độ tách cảm ơn/xin chào | 46 ± 10 | **63 ± 5** | \* |
| Độ bền với nhiễu σ=1 | 0.726 | **0.917** | +0.19 \* (mức giảm khi bật presence_norm biến mất) |
| Đổi nhãn khi tay thứ 2 lọt khung | 2.6% | 0.5% | 0 / −3.9 / −2.3 điểm (không đạt \*) |
| Chấp nhận nhầm lớp chưa học | 0.52 | **0.81** (xấu đi) | +0.63 / −0.10 / +0.33 (không đạt \*) |
| Acc OOF | 1.000 | 0.996 | −0.004 |

Quyết định: **giữ S6**.
- Cả 5 cặp khó đều tách tốt hơn và đạt \*.
- Độ bền với nhiễu hồi phục. Vậy mức giảm ở S2/S4b nhiều khả năng do checkpoint được chọn trong pha warmup, không phải do presence_norm.
- Chấp nhận nhầm lớp chưa học tăng nhưng không đều giữa các seed: lớp m bị chấp nhận nhầm 100% ở 5/6 lần chạy của cả hai bước. Đây là rủi ro đã biết, phải kiểm lại khi có lớp âm (D4).
- Lưu ý: độ tách tăng có thể một phần chỉ vì model được train lâu hơn, khiến cụm của mọi lớp co lại. Đây không phải bằng chứng về khả năng tách cặp trên session mới.

Checkpoint cuối: `runs/vsl_v3_s6`.

Checkpoint `runs/vsl_v3_s6` gồm 5 fold và model full-data 25 epoch.
- Các fold chọn epoch [10, 10, 12, 13, 10], đúng với Q3.
- T = 1.0, ngưỡng reject 0.2.
- Acc OOF 1.00 ở mọi lớp trừ cảm ơn (0.93, một clip nhầm sang xin chào).
- Predictor tự nạp checkpoint này với presence_aware=True. Input toàn số 0 bị từ chối.
