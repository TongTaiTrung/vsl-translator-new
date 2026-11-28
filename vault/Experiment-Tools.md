# Công cụ đánh giá và thí nghiệm

Không nằm trong luồng chạy chính; dùng để đo những gì CV theo mẫu gốc không đo được ([[Data-Layout]]).

| Công cụ | Lệnh | Đo gì |
|---|---|---|
| `src/model/diagnostics.py` | `python -m src.model.diagnostics --data <npz> --out <dir> --seeds 42 7 1` | `distractor_flip_rate` (tay của người khác chen vào clip một tay), open-set hold-out, tách cặp khó, độ bền nhiễu |
| `src/model/benchmark.py` | `python -m src.model.benchmark {invariance,build,train,eval}` | giữ hẳn `lớp@session` (hoặc `@session`) ra ngoài để chấm khả năng tổng quát hoá giữa bối cảnh quay; `build --mode landmark\|feature\|none` |
| `src/app/simulate.py` | `python -m src.app.simulate [--streams N]` | so nhánh thường và nhánh nhanh trên luồng nhiều từ ghép từ clip gốc, mặc định hold-out theo fold |
| `src/app/live_sim.py` | `python -m src.app.live_sim [--view hard] [--still] [--speed ...]` | phát lại clip gốc qua luồng web với tay nâng/hạ; `--still` giữ tư thế không vẫy, nên lệnh chốt từ là chấp nhận sai |

## `runs/experiments/`

Kết quả đã lưu (không bị `publish_run` xoá): `bench/`, `p1/`, `p2/` (ladder 3 seed, `run_step.sh` giới hạn 40 phút/bước), `p3/` (ladder v4: F1 = v4 + VSLNet, F2 = + SignFormer, F3 = + cosine head; `compare.py BASE STEP` in chênh lệch ghép cặp; `v3_snapshot/` giữ `src/` trước v4 để quay lại), `still_neg/` (so sánh still negative), `train_speed/`.

Giữ output thí nghiệm ở đây để `VSLPredictor` không nhặt nhầm `runs/*/ensemble.pt` mới nhất.
