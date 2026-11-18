# VSL translator

The pipeline records BGR video, pads every frame to 1280 x 1280, extracts 131 raw
features per frame with the bundled MediaPipe landmarker models, and processes a
clip into a `(30, 170)` sequence for classification.

## Notebooks

Use the project environment and run these notebooks from the project root or
`notebooks/`:

1. `notebooks/data_collect.ipynb`: record labeled clips and raw features.
2. `notebooks/reextract_features.ipynb`: rebuild features after changing the extractor.
3. `notebooks/augment_features.ipynb`: optionally add feature-only variants.
4. `notebooks/train.ipynb`: grouped cross-validation and ensemble training.
5. `notebooks/realtime_predict.ipynb`: live webcam prediction.

Original recordings are in `data/raw/<class>/<id>.mp4`. Raw per-frame features
are in `data/processed/<class>/<id>/<frame>.npz`. Augmented samples share their
original sample's group; validation uses original samples only.

## Python Prediction

`VSLPredictor` loads a checkpoint once and predicts from a complete sequence:

```python
from src.model.predict import VSLPredictor

predictor = VSLPredictor()  # newest runs/*/ensemble.pt, or pass a path
result = predictor.predict_raw(raw_features)  # (frames, 131) float32
# Or send a complete clip of BGR frames (for example, decoded from an upload):
result = predictor.predict_frames(frames_bgr, mirror_data=False)
# result: {"label": str | None, "confidence": float,
#          "accepted": bool, "top": [{"label": str, "confidence": float}, ...]}
```

The predictor also reports `probability_gap` between the two leading classes
and `rejection_reason` (`ambiguous` or `low_confidence`). The default
`min_probability_gap=0.08` rejects close decisions. Set it explicitly for your
application's tolerance for uncertain results.

New training runs use a margin loss against each sample's strongest competing
class. `runs/*/metrics.json` records both actual
OOF mistakes (`confusion_pairs`) and the nearest rival for each class
(`closest_rivals`). These are computed from original validation samples.
The training notebook also prints the source clip IDs of wrong OOF predictions.
Record these signs again in different sessions and with different signers:
cross-validation by original clip cannot measure generalization when nearly
all recordings of a class come from one session. The margin objective and
probability gap rejection reduce some ambiguous decisions, but cannot correct
a confidently wrong prediction without representative training examples.

For live BGR frames, create a separate session for each camera or client:

```python
from src.model.predict import StreamConfig

with predictor.open_stream(StreamConfig(mirror_data=False)) as session:
    result = session.update_frame(frame_bgr)
    # result is JSON-compatible: status, prediction, stable_label,
    # hand_present, frames, duration
```

`VSLStreamSession` owns its MediaPipe landmarker and frame history. A Flask
application can share one `VSLPredictor` across requests and maintain one
session per client. Close a session when the client disconnects. The `mirror_data`
setting must match the collection pipeline. The `last_frame`,
`last_pose_result`, and `last_hand_result` attributes are available for local
preview rendering; they are not included in the JSON result.
