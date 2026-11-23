"""Δ ghép cặp theo seed: python compare.py BASE STEP (BASE = S6 lấy từ p2, còn lại từ p3).
* = cùng dấu ở mọi seed và |mean Δ| > std Δ. Dấu ↑/↓ cho biết hướng tốt của metric."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
P2 = HERE.parent / "p2"
src = (P2 / "aggregate.py").read_text()
exec(src[:src.index("rows = {}")])                       # lấy hàm metrics() dùng chung với P2
LOWER_BETTER = ("flip_", "wrongacc_", "openset_")


def load(step):
    d = (P2 if (P2 / step).is_dir() and not (HERE / step).is_dir() else HERE) / step
    return {p.stem: metrics(json.loads(p.read_text())) for p in sorted(d.glob("seed*.json"))}


base, new = load(sys.argv[1]), load(sys.argv[2])
seeds = sorted(set(base) & set(new))
print(f"{sys.argv[1]} -> {sys.argv[2]}  seeds={seeds}")
for k in base[seeds[0]]:
    a = np.array([base[s][k] for s in seeds], float)
    b = np.array([new[s][k] for s in seeds], float)
    d = b - a
    star = "*" if (np.all(d > 0) or np.all(d < 0)) and abs(d.mean()) > d.std() else " "
    good = "↓" if k.startswith(LOWER_BETTER) else "↑"
    print(f"{k:<24}{good} {a.mean():>9.3f} -> {b.mean():>9.3f}  Δ={d.mean():>+9.3f} ±{d.std():<7.3f}{star}"
          f"  per-seed {np.round(d, 3).tolist()}")
