# Cây thư mục

```
vsl-translator-new/
├── src/
│   ├── builtin/mediapipe_builtin/   defs, LandmarkerLoader, StreamProcess, models/*.task|tflite
│   ├── features/                    frame, extract, landmarks, view, camera, schema
│   ├── data/                        data_reader, data_collect, reextract, augment
│   ├── model/                       training, predict, memory, checkpoints, evaluation, diagnostics, benchmark
│   ├── app/                         realtime, fast, server, voting, sentence, simulate, live_sim, static/
│   └── visualization/               landmarks (vẽ lên khung BGR)
├── tests/                           unittest, 16 module test_*.py
├── notebooks/                       data_collect, alphabet_data_collector, reextract_features,
│                                    augment_features, train, vsl_model_finetuner, realtime_predict
├── external/vsl-web/                web Next.js ([[Module-web]])
├── runs/                            best/ + các run gần nhất + experiments/
├── data/                            raw/ và processed/ ([[Data-Layout]])
├── docs/                            tài liệu cũ theo module và các báo cáo đánh giá
├── vault/                           vault Obsidian này
└── requirements.txt, pyproject.toml, .env.example, .claude/launch.json
```

## `runs/`

- `runs/best/`: run mới nhất, `VSLPredictor` đọc `runs/best/ensemble.pt` mặc định. File: `ensemble.pt`, `classes.json`, `folds.json`, `metrics.json`, `samples.json`, `.run_name`.
- `runs/<timestamp>/`: các run cũ, giữ `KEEP_RUNS` = 5 gần nhất (`checkpoints.publish_run`).
- `runs/experiments/`: kết quả thí nghiệm, không bị `publish_run` đụng tới ([[Experiment-Tools]]).

## Notebook theo thứ tự dùng

`data_collect` → `reextract_features` (sau khi đổi extractor) → `augment_features` (tuỳ chọn) → `train` (hoặc `vsl_model_finetuner` để tinh chỉnh) → `realtime_predict`. `alphabet_data_collector` quay chữ cái một lần kèm nhiễu.
