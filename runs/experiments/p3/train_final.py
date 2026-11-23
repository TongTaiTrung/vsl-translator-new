"""Checkpoint v4 (14 lớp): cấu hình S6 (VSLNet + presence_norm + distractor_geometry + Q3) trên đặc trưng v4."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.model import diagnostics as dg, training as tr  # noqa: E402

X, y, classes, groups, vmask = dg.load_dataset_npz(ROOT / "data/cache/dataset_v4.npz")
cfg = dg._notebook_like_config(42, presence_norm=True, distractor_geometry=True, select_after_warmup=5)
res = tr.run_training(X, y, classes, groups, out_dir=str(ROOT / "runs/vsl_v4_s6"), cfg=cfg,
                      folds=5, add_full=True, export_onnx_model=False, validation_mask=vmask)
print("OUT", res["out_dir"])
