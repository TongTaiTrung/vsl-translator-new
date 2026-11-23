"""P1 controlled experiments (Agent 2). Usage: exp.py <seed> <variant> [<variant> ...]
variants: baseline (snapshot), q3, q4, p1 (current src). Q1 is post-hoc on every run (T_old vs T_new).
Outputs: out/<variant>_s<seed>.json
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/tmp/vsl_audit/agent2")
import common  # noqa: E402  (audit harness: run_cv, diagnose, logits_and_emb, gaps, softmax)

PATHS = {"baseline": REPO / "runs/experiments/p1/baseline_src/model/training.py",
         "q3": HERE / "code/training_q3.py", "q4": HERE / "code/training_q4.py",
         "p1": REPO / "src/model/training.py", "p1q3": HERE / "code/training_p1_q3.py"}


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"vsl_training_{name}", PATHS[name])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


OLD = load_module("baseline")      # fit_temperature cũ (t_min=0.2)
NEW = load_module("p1")            # calibrate_temperature mới
OPEN_SET = ["m", "cảm ơn", "toi", "địa chỉ"]
MIN_CONF, MIN_GAP = 0.65, 0.08


def accept(p, thr):
    s = np.sort(p, 1)
    return (s[:, -1] >= max(thr, MIN_CONF)) & (s[:, -1] - s[:, -2] >= MIN_GAP)


def temps(lo, yo):
    lt, yt = torch.from_numpy(lo), torch.from_numpy(yo)
    t_old = float(OLD.fit_temperature(lt, yt))
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        t_new, status = NEW.calibrate_temperature(lt, yt)
    return {"old": t_old, "new": float(t_new), "new_status": status}


def thr_for(lo, yo, T):
    return float(OLD.choose_reject_threshold(common.softmax(lo / T), yo)[0])


def run(variant, seed, dev):
    tr = load_module(variant)
    common.tr = tr
    X, y, classes, groups, sessions, vmask = common.load()
    cfg = common.notebook_cfg(seed=seed)
    ci = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    t0 = time.time()
    res = common.run_cv(X, y, classes, groups, vmask, cfg, device=dev, tag=f"{variant}/s{seed}")
    d, oof, mask = common.diagnose(X, y, classes, vmask, res, dev,
                                   perturbs=["clean", "noise_0.6", "noise_1.0"], seed=0)
    lo, yo = oof[mask], y[mask]
    T = temps(lo, yo)
    out = {"variant": variant, "seed": seed, "best_epochs": d["best_epochs"], "epochs_run": d["epochs_run"],
           "oof_acc": d["oof_acc"], "oof_nll_T1": d["oof_nll_T1"], "logit_gap_median": d["logit_gap"]["median"],
           "gap_kept_noise1": d["stress_gap_median"]["noise_1.0"] / d["stress_gap_median"]["clean"],
           "acc_noise1": d["stress_acc"]["noise_1.0"],
           "hard_pair_sep": {k: d["hard_pair_sep_ratio"][k] for k in ("m/n", "r/u", "cảm ơn/xin chào")},
           "T": T, "thr": {k: thr_for(lo, yo, T[k]) for k in ("old", "new")}}

    # ---- nhiễu Gaussian (không gian đã chuẩn hoá) qua ensemble fold
    Z = np.random.default_rng(0).normal(size=(200, 30, 170)).astype(np.float32)
    zl = [common.logits_and_emb(m, Z, dev)[0] for m in res["models"]]
    out["gauss_fa"] = {k: float(accept(np.mean([common.softmax(l / T[k]) for l in zl], 0), out["thr"][k]).mean())
                       for k in ("old", "new")}

    # ---- open-set: bỏ hẳn 1 lớp, train trên fold 0 còn lại, đo tỉ lệ mẫu gốc của lớp đó bị chấp nhận
    fold_of = common.fold_assignment(y, groups, vmask, 5, seed)
    L = tr.default_groups(170)
    osr = {}
    for h in OPEN_SET:
        tr.set_seed(seed)
        keep = y != ci[h]
        remap = {o: i for i, o in enumerate([c for c in range(C) if c != ci[h]])}
        yy = np.array([remap.get(v, 0) for v in y])
        va = np.where((fold_of == 0) & vmask & keep)[0]
        trn = np.where((fold_of != 0) & keep)[0]
        norm = tr.Normalizer(X[trn]); yt = torch.from_numpy(yy).to(dev)
        m, info = tr.train_one(torch.from_numpy(norm(X[trn])).to(dev), yt[trn],
                               torch.from_numpy(norm(X[va])).to(dev), yt[va], C - 1, L, cfg, dev, log_every=0,
                               hand_presence=torch.from_numpy(X[trn, :, 129:131].copy()).to(dev))
        lv = common.logits_and_emb(m, norm(X[va]), dev)[0]
        test = np.where(~keep & vmask)[0]
        lt = common.logits_and_emb(m, norm(X[test]), dev)[0]
        Th = temps(lv, yy[va])
        osr[h] = {"best_epoch": info["best_epoch"], "id_val_acc": float((lv.argmax(1) == yy[va]).mean()), "T": Th}
        for k in ("old", "new"):
            thr = thr_for(lv, yy[va], Th[k])
            osr[h][f"fa_{k}"] = float(accept(common.softmax(lt / Th[k]), thr).mean())
            osr[h][f"id_accept_{k}"] = float(accept(common.softmax(lv / Th[k]), thr).mean())
    out["open_set"] = osr
    for k in ("old", "new"):
        out[f"open_set_fa_{k}"] = float(np.mean([osr[h][f"fa_{k}"] for h in OPEN_SET]))
        out[f"open_set_id_accept_{k}"] = float(np.mean([osr[h][f"id_accept_{k}"] for h in OPEN_SET]))

    # ---- Q2: model toàn bộ dữ liệu (chỉ baseline và p1)
    if variant in ("baseline", "p1", "p1q3"):
        tr.set_seed(seed)
        be = [i["best_epoch"] for i in res["infos"]]
        ep_full = tr.full_model_epochs(be, cfg) if hasattr(tr, "full_model_epochs") else int(np.median(be) * 1.15) + 1
        s_full = float(np.mean([i["strength"] for i in res["infos"]]))
        norm_all = tr.Normalizer(X)
        yt = torch.from_numpy(y).to(dev)
        full, finfo = tr.train_one(torch.from_numpy(norm_all(X)).to(dev), yt, None, None, C, L, cfg, dev,
                                   fixed_epochs=ep_full, init_strength=s_full, freeze_strength=True, log_every=0,
                                   hand_presence=torch.from_numpy(X[:, :, 129:131].copy()).to(dev))
        n = len(X); bs = min(int(np.clip(n // 20, 16, 64)), n); spe = max(1, n // bs)
        tot, warm, last = ep_full * spe, cfg.warmup_epochs * spe, ep_full * spe - 1
        lr_last = (cfg.lr * (last + 1) / warm if last < warm else
                   cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, (last - warm) / max(1, tot - warm)))))
        common.GLOBAL_STD = X.reshape(-1, X.shape[-1]).std(0).astype(np.float32)
        orig = np.where(vmask)[0]

        def gap_kept(model, nm, idx):
            rng = np.random.default_rng(0)
            g0 = common.gaps(common.logits_and_emb(model, nm(X[idx]), dev)[0], y[idx])[0]
            g1 = common.gaps(common.logits_and_emb(model, nm(common.perturb(X[idx], "noise_1.0", rng)), dev)[0], y[idx])[0]
            return float(np.median(g1) / np.median(g0)), float(np.median(g0))

        # cùng điều kiện "trong mẫu": mỗi model đo trên mẫu gốc nó đã train
        fold_in = [gap_kept(m, nm, np.setdiff1d(orig, va)) for m, nm, va in zip(res["models"], res["norms"], res["va"])]
        full_in = gap_kept(full, norm_all, orig)
        fl = common.logits_and_emb(full, Z, dev)[0]
        out["full_model"] = {"epochs": ep_full, "warmup": cfg.warmup_epochs,
                             "final_lr": float(finfo.get("final_lr", lr_last)), "final_lr_analytic": lr_last,
                             "reached_cosine": ep_full > cfg.warmup_epochs, "peak_lr": cfg.lr,
                             "gap_kept_noise1_insample": full_in[0], "gap_median_insample": full_in[1],
                             "fold_gap_kept_noise1_insample_mean": float(np.mean([f[0] for f in fold_in])),
                             "fold_gap_median_insample_mean": float(np.mean([f[1] for f in fold_in])),
                             "gauss_fa": {k: float(accept(common.softmax(fl / T[k]), out["thr"][k]).mean())
                                          for k in ("old", "new")}}
    out["time_s"] = time.time() - t0
    (HERE / "out" / f"{variant}_s{seed}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("variant", "seed", "best_epochs", "oof_acc", "gap_kept_noise1",
                                          "open_set_fa_old", "open_set_fa_new", "gauss_fa", "time_s")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    for v in sys.argv[2:]:
        run(v, seed, dev)
