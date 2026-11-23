"""Re-extract v4 -> cất mẫu tăng cường cũ (v2) -> tăng cường lại 2 bản/mẫu -> dataset_v4.npz + raw_v4.npz."""
import json, shutil, sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from src.data import augment
from src.data.data_reader import DataHandler
from src.data.reextract import find_raw_samples, reextract_sample
from src.features.extract import FEATURE_VERSION, process_sequence

RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = PROC / f".reextract_backup_{run_id}"
videos = find_raw_samples(RAW)
for i, v in enumerate(videos, 1):
    r = reextract_sample(v, RAW, PROC, backup)
    print(f"[{i}/{len(videos)}] {r['class']}/{r['sample_id']} {r['frames']}f", flush=True)

stale = PROC / f".stale_augmented_{run_id}"
moved = 0
for meta_path in PROC.glob("*/*/meta.json"):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "augmentation" in meta and meta.get("feature_version") != FEATURE_VERSION:
        dst = stale / meta_path.parent.parent.name / meta_path.parent.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(meta_path.parent), dst); moved += 1
print("moved stale augmented:", moved, "->", stale)

rng = np.random.default_rng(0)
plan, _ = augment.plan_augmentation(PROC, None, -1, 2)
for src, copies in plan:
    augment.augment_sample(src, RAW, copies, 1.0, rng)
print("augmented sources:", len(plan))

h = DataHandler(processor=process_sequence, min_samples_per_class=5)
h.refresh_dataset()
d = h.export_arrays(group_by="sample")
aug = np.asarray(d["augmented"])
np.savez_compressed(ROOT / "data/cache/dataset_v4.npz", X=d["X"], y=d["y"], classes=np.array(d["classes"]),
                    groups=np.asarray(d["groups"]), sessions=np.asarray(d["sessions"]), augmented=aug)
raw = np.empty(len(h), dtype=object)
for i in range(len(h)):
    raw[i] = h.load_raw(i)
np.savez_compressed(ROOT / "data/cache/raw_v4.npz", raw=raw, groups=np.asarray(d["groups"]), y=d["y"],
                    classes=np.array(d["classes"]), augmented=aug)
print("dataset_v4:", d["X"].shape, "originals:", int((~aug).sum()))
