"""Thang thí nghiệm P2: mỗi bước thêm 1 thay đổi, 3 seed, cùng fold. Chỉ đo trên mẫu gốc (OOF).

python runs/experiments/p2/ladder.py S3 --seeds 42 7 1 [--overrides '{"presence_norm": true}']
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.model import diagnostics as dg  # noqa: E402

P1 = Path("/tmp/vsl_p2/p1")
STEPS = {  # cộng dồn; bước bị loại -> chạy lại bước sau bằng --overrides
    "S0": {},
    "S1": {},
    "S2": {"presence_norm": True},
    "S3": {"presence_norm": True, "arch": "handpair"},
    "S4": {"presence_norm": True, "arch": "handpair", "distractor_geometry": True},
    "S5": {"presence_norm": True, "arch": "handpair", "distractor_geometry": True, "label_smoothing": 0.0},
    "S6": {"presence_norm": True, "distractor_geometry": True, "select_after_warmup": 5},  # S4b + Q3
}
HARD_PAIRS = ["m/n", "r/u", "t/n", "g/n", "cảm ơn/xin chào"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1])
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    overrides = json.loads(args.overrides) if args.overrides else STEPS[args.step]

    if args.step == "S0":                       # baseline: code P1 đóng băng + dữ liệu v2
        dg.tr = _load("p1_training", P1 / "training.py")
        fx = _load("extract_v2", P1 / "extract_v2.py")
        data, raw_path = ROOT / "data/cache/dataset_v2.npz", ROOT / "data/cache/raw_v2.npz"
    else:
        from src.features import extract as fx
        data, raw_path = ROOT / "data/cache/dataset_v3.npz", ROOT / "data/cache/raw_v3.npz"
    X, y, classes, groups, vmask = dg.load_dataset_npz(data)
    raw = np.load(raw_path, allow_pickle=True)["raw"]
    assert len(raw) == len(X)
    device = dg.tr.get_device()
    out = ROOT / "runs/experiments/p2" / (args.tag or args.step)
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        t0 = time.time()
        cfg = dg._notebook_like_config(seed, **overrides)
        res = dg.run_cv(X, y, classes, groups, vmask, cfg, device=device)
        rep = dg.diagnose(X, y, classes, vmask, res, device, seed=seed)
        pairs = rep.pop("hard_pair_separation_ratio")
        rep["hard_pairs"] = {p: pairs.get(p, pairs.get("/".join(p.split("/")[::-1]))) for p in HARD_PAIRS}
        rep["closest_pairs"] = dict(list(pairs.items())[:5])
        rep["distractor"] = dg.distractor_flip_rate(raw, y, classes, vmask, res, device, fx)
        rep["open_set"] = dg.open_set_hold_one_class_out(X, y, classes, groups, vmask, device, seed,
                                                         held_classes=["m", "địa chỉ"], cfg=cfg)
        rep.update(seed=seed, overrides=overrides, minutes=(time.time() - t0) / 60,
                   val_accs=[i["val_acc"] for i in res["infos"]])
        (out / f"seed{seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(json.dumps({k: rep[k] for k in ("seed", "minutes", "oof_acc", "oof_macro_f1", "distractor",
                                              "hard_pairs")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
