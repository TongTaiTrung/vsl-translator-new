"""Quản lý các run trong ``runs/``: ``runs/best`` trỏ tới run mới nhất, chỉ giữ ``KEEP_RUNS`` run gần nhất.

Một "run" là thư mục con trực tiếp của ``runs/`` có ``ensemble.pt`` (``train.ipynb`` tạo ``vsl_<thời gian>``,
``vsl_model_finetuner.ipynb`` tạo ``vsl_ft_<thời gian>``). ``runs/experiments/`` và các thư mục sâu hơn
không bao giờ bị đụng tới.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
BEST_NAME = "best"
KEEP_RUNS = 5
PROTECTED = {BEST_NAME, "experiments"}
_STAMP = re.compile(r"(\d{8}_\d{6})")


def run_time(run_dir: Path) -> datetime:
    """Thời điểm tạo run: lấy từ tên (``vsl_20261123_204353``); không có thì dùng mtime của ensemble.pt.

    mtime không tin được (copy/khôi phục thư mục làm mọi file cùng mtime) nên tên được ưu tiên.
    """
    match = _STAMP.search(run_dir.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return datetime.fromtimestamp((run_dir / "ensemble.pt").stat().st_mtime)


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Các run trực tiếp trong runs_dir (bỏ symlink ``best`` và ``experiments``), mới nhất trước."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    runs = [p for p in runs_dir.iterdir()
            if p.name not in PROTECTED and not p.is_symlink() and (p / "ensemble.pt").is_file()]
    return sorted(runs, key=run_time, reverse=True)


def latest_checkpoint(runs_dir: Path = RUNS_DIR) -> Path:
    """``runs/best/ensemble.pt`` nếu có; không thì ensemble.pt của run mới nhất."""
    runs_dir = Path(runs_dir)
    best = runs_dir / BEST_NAME / "ensemble.pt"
    if best.is_file():
        return best
    runs = list_runs(runs_dir)
    if not runs:
        raise FileNotFoundError(f"Không có ensemble.pt nào trong {runs_dir}; hãy train model trước")
    return runs[0] / "ensemble.pt"


def publish_run(run_dir: str | Path, runs_dir: Path = RUNS_DIR, keep: int = KEEP_RUNS) -> list[Path]:
    """Đặt run vừa lưu làm ``runs/best`` rồi xoá các run cũ, chỉ giữ ``keep`` run gần nhất.

    Chỉ áp dụng cho run nằm trực tiếp trong runs_dir; run ở nơi khác (vd. ``runs/experiments/...``, thư mục tạm
    của test) được bỏ qua. Run vừa lưu luôn được giữ. Trả về danh sách thư mục đã xoá.
    """
    if keep < 1:
        raise ValueError("keep phải >= 1")
    run_dir, runs_dir = Path(run_dir).resolve(), Path(runs_dir).resolve()
    if run_dir.parent != runs_dir or run_dir.name in PROTECTED or not (run_dir / "ensemble.pt").is_file():
        return []

    best = runs_dir / BEST_NAME
    if best.is_symlink() or best.is_file():
        best.unlink()
    elif best.exists():
        raise FileExistsError(f"{best} là thư mục thật, không phải symlink; hãy đổi tên nó rồi chạy lại")
    best.symlink_to(run_dir.name, target_is_directory=True)   # đường dẫn tương đối: chuyển cả runs/ vẫn đúng

    others = [p for p in list_runs(runs_dir) if p != run_dir]
    removed = others[keep - 1:]
    for path in removed:
        shutil.rmtree(path)
    print(f"runs/{BEST_NAME} -> {run_dir.name}" + (f" | đã xoá {len(removed)} run cũ: "
                                                   + ", ".join(p.name for p in removed) if removed else ""))
    return removed
