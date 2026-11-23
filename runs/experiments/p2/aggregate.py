"""Gộp kết quả thang: mean ± std qua seed cho mỗi bước. python aggregate.py S0 S1 ..."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def metrics(r):
    d = r["distractor"]
    out = {"oof_acc": r["oof_acc"], "oof_f1": r["oof_macro_f1"], "noise_ret": r["noise_sigma1_gap_retention"]}
    for v in ("same_height", "lowered_static"):
        out[f"flip_{v}"] = d[v]["flip_rate"]
        out[f"wrongacc_{v}"] = d[v]["wrong_accept_rate"]
    out["openset_fa"] = float(np.mean([o["false_accept_rate_T_fit"] for o in r["open_set"]]))
    out["openset_maxp"] = float(np.mean([o["median_maxprob_T_fit"] for o in r["open_set"]]))
    out.update({f"sep_{k}": v for k, v in r["hard_pairs"].items()})
    return out


rows = {}
for step in sys.argv[1:]:
    runs = {p.stem: metrics(json.loads(p.read_text())) for p in sorted((HERE / step).glob("seed*.json"))}
    if runs:
        rows[step] = runs
keys = list(next(iter(next(iter(rows.values())).values())))
print(f"{'metric':<24}" + "".join(f"{s + ' (n=' + str(len(r)) + ')':>22}" for s, r in rows.items()))
for k in keys:
    cells = []
    for r in rows.values():
        v = np.array([m[k] for m in r.values() if m.get(k) is not None], dtype=float)
        cells.append(f"{v.mean():>12.3f} ± {v.std():<7.3f}")
    print(f"{k:<24}" + "".join(cells))
