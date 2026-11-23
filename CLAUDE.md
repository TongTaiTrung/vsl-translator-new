# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Vietnamese Sign Language (VSL) translator. The pipeline is: BGR video → pad every frame to 1280×1280 (`src/features/frame.py:frame_padding`, scales up or down so the long side fills the canvas; raw videos recorded before this are 720×720) → MediaPipe pose and hand landmarkers → **277 raw features per frame** (`src/features/extract.py:extract`) → `process_sequence` resamples and expands each clip to a **`(30, 328)` sequence** (277 raw + 39 velocity + 12 stroke columns) → an ensemble classifier (`VSLNet`, `HandPairNet` or `SignFormer`) with rejection.

Comments, docstrings and many CLI messages are written in Vietnamese. Match that style when editing nearby code.

## Commands

Use the project virtualenv at `.venv/` (Python 3.12). Install with `pip install -r requirements.txt && pip install -e .` (the `src*` package is defined in `pyproject.toml`).

Tests use `unittest`, not pytest, which isn't installed. `tests/` has no `__init__.py`, so `discover` fails. Run modules by dotted name from the repo root:

```bash
.venv/bin/python -m unittest tests.test_predict tests.test_reextract tests.test_feature_augmentation tests.test_training_robustness tests.test_builtin_collection tests.test_training_p1 tests.test_evaluation tests.test_feature_v3 tests.test_app tests.test_fast tests.test_server tests.test_memory tests.test_camera tests.test_view tests.test_checkpoints
.venv/bin/python -m unittest tests.test_predict.PredictorTests.test_checkpoint_prediction_is_json_ready_and_validates_shape
```

Training CLI (in `src/model/training.py`), using a `.npz` file containing `X, y [, classes, groups]`:

```bash
.venv/bin/python -m src.model.training --demo --folds-to-run 1 --epochs 5   # smoke test with synthetic data
.venv/bin/python -m src.model.training --data <file.npz> --out runs/vsl
```

The normal workflow runs through the notebooks in `notebooks/`, from the project root or from `notebooks/`, in this order: `data_collect` → `reextract_features` (after changing the extractor) → `augment_features` (optional) → `train` → `realtime_predict`.

## Architecture

**Feature contract.** `src/features/schema.py` (`EXPECTED_FRAMES=30`; `FRAME_FEATURE_DIM` is re-exported from `extract.FEATURE_DIM`) and `FEATURE_VERSION` (currently 6) in `extract.py` define the raw format. v4 adds `hand_local` (20 hand points per hand in a mirror-equivariant palm frame, scaled by palm size) and `face_vec` (palm and index tip minus nose). v5 adds `hand_scale` (mean 3D palm-bone length in the image / shoulder width, a depth proxy that is steadier than pose z) and down-weights z by `CONTACT_Z` in `body_contact`, because noisy pose z otherwise kept a hand resting on the chest from registering contact (this is what separates `toi` from `m`). v6 drops the hip-based spine axis (hips are almost always out of frame, so it ran in ~7% of clips and switched the torso frame mid-clip): the up axis is always image vertical. Pitch (camera higher or lower than the signer) is deliberately not estimated, because every pose-z or head cue varied more with the sign than with the camera. Instead `rotate_view_pitch` simulates a camera-height change on raw features (positions are rotated in true depth, image pose z ≈ `POSE_Z_GAIN`=3.35 × depth), and `augment_sample` gives each copy a random pitch within ±`PITCH_AUG_DEG`. Column groups are exposed as `FEATURE_SLICES` / `PRESENCE_SLICE`, and `SIDE_COLUMNS` / `LEFT_MIRROR` split the sequence into left, right and shared columns; `training.py`, `diagnostics.py` and `augment.py` derive every index from these, so never hard-code column numbers. `process_sequence` drops a hand whose median palm y is below `REST_Y` (a resting hand), and stream presence uses `active_presence`, so the same rule applies at train and serve time. `training.py:default_groups` and `hand_feature_slices` rely on that layout for the per-group stem and hand augmentation. `process_sequence` also appends `STROKE_SLICE` (12 columns, its own stem group): for each hand, the index-tip and palm displacement over the `STROKE_LAG`-frame span with the largest xy index-tip displacement, repeated on every frame. Short flicks (dấu sắc/huyền/nặng move in ~4 of 30 frames) otherwise barely register in per-frame velocity, and the model learned those classes from the static hold alone. Only spans where the hand is valid throughout (present, above `REST_Y`, one frame away from gaps because of smoothing) count, so a hand entering the frame is not a stroke. Changing only `process_sequence` leaves raw features alone (no re-extraction, no `FEATURE_VERSION` bump), but every checkpoint must be retrained. If you change the extractor, bump `FEATURE_VERSION`, re-run re-extraction (`src/data/reextract.py`), and check the slices in `training.py`.

**Data storage (`src/data/data_reader.py:DataHandler`).** `DataHandler` owns the dataset layout:
- `data/raw/<class>/<id>.mp4`: original recordings
- `data/processed/<class>/<id>/<frame>.npz` plus meta: raw per-frame features
- `data/processed/<class>/<id>/landmarks.npz` (original samples only, written by `reextract`): raw MediaPipe pose `(T,33,4)`, hands and hand-world `(T,2,21,3)`, NaN when absent (`src/features/landmarks.py`). `_frame_paths` ignores it because its stem is not numeric.

Class names are NFC-normalized Vietnamese. `augment.py` and `reextract.py` call its *private static* helpers directly (`_read_meta`, `_frame_paths`, `_load_feature`) and its constants (`FEATURE_DIM`, `PRESENCE_COLS`, `MIN_FRAMES`, `SEQUENCE_LENGTH`). Changing those signatures breaks both modules.

**Setup shift and landmark augmentation.** Every class except `m` was recorded in one session, and the same sign in two setups (`m` on 16/11 vs 18/11) is farther apart in feature space than two different signs in one setup, so CV (grouped by origin sample) does not measure real use. `src/features/view.py` simulates another recording setup on landmarks (camera yaw/pitch/roll orbit around mid-shoulder, distance/FOV via perspective, hand size, signing position, per-clip finger-joint bias, because rigid rotation leaves handshape unchanged but MediaPipe's joint estimates drift between setups) and recomputes features with `extract`; `IDENTITY` reproduces the stored features. `augment.make_variant` uses it whenever `landmarks.npz` exists and falls back to feature noise plus `rotate_view_pitch` otherwise. `src/model/benchmark.py` holds out `class@session` (or `@session`) entirely: `invariance` (no training), `build` (in-memory augmentation, `--mode landmark|feature|none`), `train`, `eval`. Keep its runs under `runs/experiments/` so `VSLPredictor` does not pick them as the newest `runs/*/ensemble.pt`.

**Grouping and augmentation.** Augmented samples record `origin_sample_id` in their meta and share their source's group. Cross-validation (`make_folds`, StratifiedGroupKFold) keeps each origin group together, and validation/OOF metrics use original samples only (`validation_mask`). Never augment an augmented sample. Keep these invariants: they are the project's only guard against leakage.

**Training (`src/model/training.py`).** This single module contains `Config`, `Normalizer`, `Augmenter` and `AdaptiveController` (which adjusts dropout and mixup online), `VSLNet` (`GroupStem` → `MultiScaleBlock`s → linear or cosine head) or `HandPairNet` (`Config.arch="handpair"`, CLI `--arch`; a weight-shared per-hand stem with a learned absent-hand token, which requires `presence_norm=True`) or `SignFormer` (`arch="former"`, also needs `presence_norm`: a point-attention encoder over `hand_local`, the same absent token, a conv block plus 2 pre-LN Transformer layers with stochastic depth `drop_path`; implemented and unit-tested but not yet validated on the ladder), EMA, and a hard-negative margin loss. `presence_norm` computes per-hand normalization stats only on frames where that hand is present and zeroes absent hands; `distractor_geometry` recomputes the inter-hand columns when the augmenter injects a distractor hand. `select_after_warmup=N` only lets epochs from `warmup_epochs+N` on be selected as the best checkpoint (otherwise the EMA model is often picked mid-warmup); the notebook uses 5. Old checkpoints without `arch` load as `VSLNet`. `run_training` trains k fold models, plus an optional full-data model. It then fits temperature scaling and a reject threshold on OOF logits and writes `runs/<name>/ensemble.pt`, `metrics.json` (`confusion_pairs`, `closest_rivals`) and optionally an ONNX export. Then `src/model/checkpoints.py:publish_run` (also called by `vsl_model_finetuner.ipynb`) points the `runs/best` symlink at the new run and deletes all but the `KEEP_RUNS`=5 newest run directories directly under `runs/` (ordered by the timestamp in the name, because mtime is unreliable after a git checkout); `runs/experiments/` and the `--demo` CLI run are never touched. `runs/` is committed to git, so pruned runs remain in history.

**Inference (`src/model/predict.py`).**
- `VSLPredictor` loads one checkpoint (by default `runs/best/ensemble.pt`, see below) and exposes `predict_raw((frames,143))`, `predict_processed((30,182))` and `predict_frames(bgr_frames, fps=None)`; pass the video fps so MediaPipe gets the same timestamps as `reextract.py`.
- Results are JSON-compatible and include `accepted`, `probability_gap` and `rejection_reason` (`ambiguous` or `low_confidence`). `min_probability_gap` defaults to 0.08.
- For live video, use `predictor.open_stream(StreamConfig(...))` to get a `VSLStreamSession`. Each session owns its MediaPipe landmarkers and frame history, so create one per client or camera and close it when done. Never share a session between clients. `StreamConfig.target_fps` (default 15, the collection rate; `None` disables it) skips frames that arrive faster, because gap filling and spike removal count frames.
- `mirror_data` must match the setting used during collection.

**MediaPipe wrapper.** `src/builtin/mediapipe_builtin/` provides `LandmarkerLoader`, which wraps the bundled pose and hand `.task` models (paths are in `defs.py`) and handles IMAGE/VIDEO running modes. Timestamps passed to `detect` must strictly increase in VIDEO mode.

Per-module docs, partly in Vietnamese, are in `docs/<module>/README.md` and `usage.md`. `docs/tue-17-11.md` is a recent architecture and data review: it notes that per-clip CV overstates accuracy when a class is recorded in a single session.

**Experiments.** `src/model/diagnostics.py` measures what CV accuracy hides: `distractor_flip_rate` (a donor's left hand is injected into one-hand clips), open-set hold-out, hard-pair separation and noise robustness. `runs/experiments/p2/ladder.py` runs the controlled ladder (3 seeds, same folds); `run_step.sh` caps each step at 40 minutes. `runs/experiments/p3/` is the v4 ladder (F1 = v4 + VSLNet, F2 = + SignFormer, F3 = + cosine head; restricted to the 12 classes of `dataset_v3.npz` so it pairs with S6), `compare.py BASE STEP` prints paired deltas, and `v3_snapshot/` holds the pre-v4 `src/` for rollback.
