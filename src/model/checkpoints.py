"""Manage training runs in runs/: runs/best is the newest run and only KEEP_RUNS runs are kept."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
BEST_NAME = "best"
KEEP_RUNS = 5
PROTECTED = {BEST_NAME, "experiments"}
RUN_NAME_FILE = ".run_name"
_STAMP = re.compile(r"(\d{8}_\d{6})")


def run_time(run_dir: Path) -> datetime:
    match = _STAMP.search(run_dir.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return datetime.fromtimestamp((run_dir / "ensemble.pt").stat().st_mtime)


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    runs = [p for p in runs_dir.iterdir()
            if p.name not in PROTECTED and not p.is_symlink() and (p / "ensemble.pt").is_file()]
    return sorted(runs, key=run_time, reverse=True)


def latest_checkpoint(runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir = Path(runs_dir)
    best = runs_dir / BEST_NAME / "ensemble.pt"
    if best.is_file():
        return best
    runs = list_runs(runs_dir)
    if not runs:
        raise FileNotFoundError(f"Không có ensemble.pt nào trong {runs_dir}; hãy train model trước")
    return runs[0] / "ensemble.pt"


def _archive_name(best: Path) -> str:
    marker = best / RUN_NAME_FILE
    if marker.is_file() and marker.read_text(encoding="utf-8").strip():
        return marker.read_text(encoding="utf-8").strip()
    return datetime.fromtimestamp((best / "ensemble.pt").stat().st_mtime).strftime("%Y%m%d_%H%M%S")


def publish_run(run_dir: str | Path, runs_dir: Path = RUNS_DIR, keep: int = KEEP_RUNS) -> Path | None:
    if keep < 1:
        raise ValueError("keep phải >= 1")
    run_dir, runs_dir = Path(run_dir).resolve(), Path(runs_dir).resolve()
    best = runs_dir / BEST_NAME
    if run_dir == best.resolve() and best.is_dir() and not best.is_symlink():
        return best
    if run_dir.parent != runs_dir or run_dir.name in PROTECTED or not (run_dir / "ensemble.pt").is_file():
        return None

    if best.is_symlink() or best.is_file():
        best.unlink()
    elif best.is_dir():
        archive = runs_dir / _archive_name(best)
        n = 1
        while archive.exists():
            archive = runs_dir / f"{_archive_name(best)}_{n}"
            n += 1
        best.rename(archive)
    (run_dir / RUN_NAME_FILE).write_text(run_dir.name, encoding="utf-8")
    name = run_dir.name
    run_dir.rename(best)

    removed = list_runs(runs_dir)[keep - 1:]
    for path in removed:
        shutil.rmtree(path)
    print(f"Model mới nhất: runs/{BEST_NAME} (train lúc {name})" + (f" | đã xoá {len(removed)} run cũ: "
                                                                   + ", ".join(p.name for p in removed) if removed else ""))
    return best
