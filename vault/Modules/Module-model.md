# `src/model`

| File | Vai trò |
|---|---|
| `training.py` | `Config`, `Normalizer`, `Augmenter`, `AdaptiveController`, `VSLNet`/`HandPairNet`/`SignFormer`, EMA, `run_training`, `export_onnx`, CLI |
| `predict.py` | `VSLPredictor`, `VSLStreamSession`, `StreamConfig` |
| `memory.py` | `MemoryEnsemble`: mẫu cá nhân cộng điểm vào logit, không train lại |
| `checkpoints.py` | `publish_run`, `list_runs`, `latest_checkpoint` |
| `evaluation.py` | `oof_report`: chỉ số bổ sung trên logit OOF (chỉ numpy) |
| `diagnostics.py`, `benchmark.py` | công cụ đo, xem [[Experiment-Tools]] |

## Huấn luyện

- `Config.arch`: `vslnet` (mặc định: `GroupStem` → `MultiScaleBlock` → head tuyến tính hoặc cosine), `handpair` (stem dùng chung cho từng tay, cần `presence_norm=True`), `former` (`SignFormer`, cần `presence_norm`; đã có test nhưng chưa được kiểm chứng trên ladder).
- `run_training`: k fold + tuỳ chọn model toàn bộ dữ liệu, rồi temperature scaling và ngưỡng từ chối trên OOF logit. Ghi `ensemble.pt`, `metrics.json` (`confusion_pairs`, `closest_rivals`), tuỳ chọn ONNX.
- `AdaptiveController` chỉnh dropout/mixup trực tuyến; `select_after_warmup=N` chỉ chọn checkpoint tốt nhất từ epoch `warmup_epochs+N` (notebook dùng 5).
- Mặc định: `max_epochs=60`, `min_epochs=15`, `patience=10`, `batch_size=0` (tự chọn).
- Giữ **hình dạng batch cố định** trong mỗi lần forward (MPS biên dịch lại GRU nếu đổi, chậm ~6×): khe tay nhiễu và mẫu "still negative" có số lượng cố định.
- Clip tăng cường không có tay hợp lệ nào bị bỏ; mẫu gốc luôn giữ.
- **Still negative** (`still_negative_weight`=0.5): với lớp có nét (`stroke_magnitude` ≥ `dynamic_min_stroke`=0.25), `still_versions` dựng bản chỉ giữ tư thế, train về phân phối đều để model không nhận lớp từ một frame tĩnh.
- CLI: `python -m src.model.training --demo --folds-to-run 1 --epochs 5` (dữ liệu giả) hoặc `--data <file.npz> --out runs/vsl`. Cờ khác: `--folds`, `--no-full`, `--head`, `--lr`, `--batch-size`, `--seed`, `--arch`, `--no-onnx`.

## Dự đoán (`predict.py`)

- `VSLPredictor()` mặc định đọc `runs/best/ensemble.pt`; `predict_raw((frames, 383))`, `predict_processed((30, 434))`, `predict_frames(bgr_frames, fps=None)`, `open_stream(StreamConfig)`.
- Kết quả JSON: `label`, `confidence`, `accepted`, `probability_gap`, `rejection_reason` (`ambiguous`/`low_confidence`), `top`. `min_probability_gap` mặc định 0.08.
- `VSLStreamSession`: `update_frame`, `update_features`, `reset`, `close`. Mỗi client một phiên; không chia sẻ.
- `StreamConfig` mặc định: cửa sổ 2.0 s, tối thiểu 1.5 s, `min_frames`=8, dự đoán mỗi 3 frame, `stability_frames`=3, `no_hand_timeout`=0.6 s, `target_fps`=15 (bỏ frame đến nhanh hơn; `None` tắt).
- `mirror_data` phải khớp lúc thu dữ liệu.

## Bộ nhớ cá nhân (`memory.py`)

Mỗi mẫu đi qua từng model lấy embedding. Khi dự đoán: `bonus_k[c] = alpha * max_i exp(-sharpness * (1 - cos))` cộng vào logit lớp `c`; `alpha = strength * MAX_BONUS` (`MAX_BONUS` = 10). Lấy `max` theo lớp để lớp nhiều mẫu không thắng chỉ vì số lượng. Dữ liệu mẫu nằm ở trình duyệt; server chỉ giữ trong phiên stream ([[Workflow-Web-API]]).

## Quản lý run (`checkpoints.py`)

`publish_run` đổi tên thư mục run mới thành `runs/best`, chuyển best cũ về tên theo thời điểm của nó (`.run_name`), xoá run thừa, giữ `KEEP_RUNS` = 5. Không đụng `runs/experiments/`.
