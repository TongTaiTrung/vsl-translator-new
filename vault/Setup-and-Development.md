# Cài đặt và phát triển

## Cài đặt

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```

Web (tuỳ chọn): `cd external/vsl-web && npm install`.

## Chạy

| Việc | Lệnh |
|---|---|
| Dịch webcam | `.venv/bin/python -m src.app [--no-llm] [--fast]` |
| API cho web | `.venv/bin/python -m src.app.server --port 8000` |
| Web | `cd external/vsl-web && npm run dev` |
| Train thử nhanh | `.venv/bin/python -m src.model.training --demo --folds-to-run 1 --epochs 5 --out <thư mục tạm>` |
| Train thật | `.venv/bin/python -m src.model.training --data <file.npz> --out runs/vsl` |
| Quay clip | `.venv/bin/python -m src.data.data_collect -c <lớp>` |

Notebook chạy từ gốc dự án hoặc từ `notebooks/`, theo thứ tự ở [[Workflow-Data-to-Model]].

## Test

Dùng `unittest` (không có pytest). `tests/` không có `__init__.py` nên `discover` không chạy; gọi từng module theo tên:

```bash
.venv/bin/python -m unittest tests.test_predict tests.test_reextract tests.test_feature_augmentation tests.test_training_robustness tests.test_builtin_collection tests.test_training_p1 tests.test_evaluation tests.test_feature_v3 tests.test_app tests.test_fast tests.test_server tests.test_memory tests.test_camera tests.test_view tests.test_checkpoints tests.test_still_negatives
```

Một test: `.venv/bin/python -m unittest tests.test_predict.PredictorTests.test_checkpoint_prediction_is_json_ready_and_validates_shape`.

| Nhóm | Module test |
|---|---|
| Đặc trưng | `test_feature_v3`, `test_view`, `test_camera`, `test_feature_augmentation`, `test_reextract`, `test_builtin_collection` |
| Mô hình | `test_training_p1`, `test_training_robustness`, `test_still_negatives`, `test_evaluation`, `test_checkpoints`, `test_predict`, `test_memory` |
| Ứng dụng | `test_app`, `test_fast`, `test_server` |

Web: `cd external/vsl-web && npm run typecheck`.

## Quy tắc khi sửa

- Đổi extractor: tăng `FEATURE_VERSION`, trích lại, kiểm tra slice ([[Feature-Contract]]).
- Không hard-code chỉ số cột; dùng `FEATURE_SLICES`.
- Giữ chữ ký `DataHandler._read_meta`/`_frame_paths`/`_load_feature` ([[Data-Layout]]).
- Mỗi forward huấn luyện giữ hình dạng batch cố định ([[Module-model]]).
- Không đưa output thí nghiệm vào `runs/` ngoài `runs/experiments/`.
