"""Thang thí nghiệm P3 (đặc trưng v4, so với S6 của P2): mỗi bước thêm 1 thay đổi, 3 seed, cùng fold. Chỉ đo trên mẫu gốc (OOF).

python runs/experiments/p3/ladder.py F1 --seeds 42 7 1 [--overrides '{"presence_norm": true}']
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.model import diagnostics as dg  # noqa: E402

S6 = {"presence_norm": True, "distractor_geometry": True, "select_after_warmup": 5}
STEPS = {  # cộng dồn trên S6
    "F1": dict(S6),                                   # đặc trưng v4 + VSLNet
    "F2": dict(S6, arch="former"),                    # + SignFormer
    "F3": dict(S6, arch="former", head="cosine"),     # + head cosine
}
HARD_PAIRS = ["m/n", "r/u", "t/n", "g/n", "cảm ơn/xin chào"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1])
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    overrides = json.loads(args.overrides) if args.overrides else STEPS[args.step]

    from src.features import extract as fx
    data, raw_path = ROOT / "data/cache/dataset_v4.npz", ROOT / "data/cache/raw_v4.npz"
    X, y, classes, groups, vmask = dg.load_dataset_npz(data)
    raw = np.load(raw_path, allow_pickle=True)["raw"]
    assert len(raw) == len(X)
    # So sánh công bằng với S6: chỉ 12 lớp có trong dataset_v3 (h, k thêm sau), giữ thứ tự mẫu.
    old = [str(c) for c in np.load(ROOT / "data/cache/dataset_v3.npz", allow_pickle=True)["classes"]]
    keep = np.isin(np.asarray(classes)[y], old)
    remap = {c: i for i, c in enumerate(old)}
    y = np.array([remap[classes[k]] for k in y[keep]])
    X, groups, vmask, raw, classes = X[keep], groups[keep], vmask[keep], raw[keep], old
    v3g = np.load(ROOT / "data/cache/dataset_v3.npz", allow_pickle=True)["groups"]
    assert list(map(str, groups)) == list(map(str, v3g)), "thứ tự mẫu khác v3 -> fold sẽ khác S6"
    device = dg.tr.get_device()
    out = ROOT / "runs/experiments/p3" / (args.tag or args.step)
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
