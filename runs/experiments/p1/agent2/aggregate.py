"""Gộp out/<variant>_s<seed>.json -> bảng mean ± std (3 seed) + so sánh với baseline."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
SEEDS = [42, 7, 1]
R = {v: [json.loads((OUT / f"{v}_s{s}.json").read_text()) for s in SEEDS if (OUT / f"{v}_s{s}.json").exists()]
     for v in ("baseline", "q3", "q4", "p1", "p1q3")}

# (tên hàng, variant, temperature dùng)
ROWS = [("baseline", "baseline", "old"), ("Q1 alone (post-hoc)", "baseline", "new"), ("Q3 alone", "q3", "old"),
        ("Q4 alone", "q4", "old"), ("P1 as in repo (Q1+Q2+Q4, no Q3)", "p1", "new"),
        ("P1 full (Q1+Q2+Q3+Q4)", "p1q3", "new")]
METRICS = [
    ("open-set FA", lambda r, t: r[f"open_set_fa_{t}"]),
    ("open-set ID accept", lambda r, t: r[f"open_set_id_accept_{t}"]),
    ("gauss FA", lambda r, t: r["gauss_fa"][t]),
    ("gap kept σ=1", lambda r, t: r["gap_kept_noise1"]),
    ("m/n sep", lambda r, t: r["hard_pair_sep"]["m/n"]),
    ("r/u sep", lambda r, t: r["hard_pair_sep"]["r/u"]),
    ("cảm ơn/xin chào sep", lambda r, t: r["hard_pair_sep"]["cảm ơn/xin chào"]),
    ("OOF acc", lambda r, t: r["oof_acc"]),
    ("median best epoch", lambda r, t: float(np.median(r["best_epochs"]))),
    ("T", lambda r, t: r["T"][t]),
]


def vals(variant, t, f):
    return np.array([f(r, t) for r in R[variant]])


summary = {}
print("| metric | " + " | ".join(n for n, _, _ in ROWS) + " |")
print("|---" * (len(ROWS) + 1) + "|")
for mname, f in METRICS:
    cells = []
    base = vals("baseline", "old", f)
    for rname, v, t in ROWS:
        a = vals(v, t, f)
        if len(a) == 0:
            cells.append("n/a"); continue
        d = a - base[:len(a)]                 # chênh lệch ghép cặp theo seed
        spread = max(a.std(), base.std())
        flag = ""
        if rname != "baseline":
            # vượt độ dao động seed: |mean diff| > 2*max(std) và cùng dấu ở mọi seed
            sig = abs(d.mean()) > 2 * spread and (np.all(d > 0) or np.all(d < 0))
            flag = " *" if sig else ""
        cells.append(f"{a.mean():.3f}±{a.std():.3f}{flag}")
        summary.setdefault(rname, {})[mname] = {"mean": float(a.mean()), "std": float(a.std()),
                                                "per_seed": a.tolist(), "paired_diff": d.tolist()}
    print(f"| {mname} | " + " | ".join(cells) + " |")

print("\nOpen-set FA per held-out class (mean over seeds):")
for rname, v, t in ROWS:
    print(f"  {rname:22s}", {h: round(float(np.mean([r['open_set'][h][f'fa_{t}'] for r in R[v]])), 3)
                             for h in R[v][0]["open_set"]} if R[v] else "n/a")
print("\nT status (new) per seed:", {v: [r["T"]["new_status"] for r in R[v]] for v in R})
print("Best epochs:", {v: [r["best_epochs"] for r in R[v]] for v in R})
print("\nQ2 full model:")
for v in ("baseline", "p1", "p1q3"):
    for r in R[v]:
        fm = r.get("full_model")
        if fm:
            print(f"  {v} s{r['seed']}: epochs={fm['epochs']} warmup={fm['warmup']} final_lr={fm['final_lr']:.2e} "
                  f"(peak {fm['peak_lr']:.0e}) | gap kept σ=1 in-sample full={fm['gap_kept_noise1_insample']:.3f} "
                  f"vs folds={fm['fold_gap_kept_noise1_insample_mean']:.3f} | gap median full={fm['gap_median_insample']:.2f} "
                  f"folds={fm['fold_gap_median_insample_mean']:.2f} | gauss FA old/new={fm['gauss_fa']}")
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
