"""Lệch train/serve: đặc trưng từ đường predict_frames (fps nhịp đúng) vs đặc trưng đã lưu, trên 60 clip gốc."""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.data_reader import DataHandler  # noqa: E402
from src.features.extract import process_sequence  # noqa: E402
from src.model.predict import VSLPredictor  # noqa: E402

h = DataHandler(processor=process_sequence)
h.refresh_dataset()
orig = [i for i in range(len(h)) if not h.describe(i)["augmented"]]
rng = np.random.default_rng(0)
diffs, labels = [], []
for i in sorted(rng.choice(orig, size=min(60, len(orig)), replace=False)):
    d = h.describe(int(i))
    video = ROOT / "data/raw" / d["class"] / f"{d['sample_id']}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    captured = {}
    predictor = VSLPredictor.__new__(VSLPredictor)      # đúng đường predict_frames, bỏ qua model
    predictor.predict_raw = lambda raw, **_: captured.setdefault("raw", raw)
    predictor.predict_frames(frames, fps=fps)
    feats = captured["raw"]
    live = process_sequence(feats)
    stored = process_sequence(h.load_raw(int(i)))
    diffs.append(float(np.abs(live - stored).max()))
    labels.append(d["class"])
diffs = np.array(diffs)
print(f"n={len(diffs)} | max|Δ| median={np.median(diffs):.2e} max={diffs.max():.2e} | "
      f"clip lệch >1e-4: {int((diffs > 1e-4).sum())}")
for lab, dv in sorted(zip(labels, diffs), key=lambda t: -t[1])[:5]:
    print(f"  {lab}: {dv:.2e}")
