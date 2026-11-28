"""DataHandler: store and read VSL samples and export the training dataset."""

from __future__ import annotations

import json
import os
import shutil
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.features.schema import EXPECTED_FRAMES, FRAME_FEATURE_DIM
from src.features.frame import FRAME_SIZE, frame_padding

_INVALID_NAME_CHARS = set('<>:"/\\|?*')


def get_project_root_dir(current_path: Path | None = None) -> Path:
    current_path = Path(current_path or __file__).resolve()
    if current_path.is_file():
        current_path = current_path.parent
    for parent in [current_path, *current_path.parents]:
        if (parent / "src").is_dir():
            return parent
    raise FileNotFoundError("Project invalid: 'src' directory not found.")


def get_data_path() -> Path:
    return get_project_root_dir() / "data"


def _normalize_nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name.strip())


def _check_class_name(name: str) -> str:
    name = _normalize_nfc(name)
    if (
        not name
        or name in {".", ".."}
        or name.startswith(".")
        or name.endswith(".")
        or any(char in _INVALID_NAME_CHARS or ord(char) < 32 for char in name)
    ):
        raise ValueError(
            "class_name must be non-empty, must not start/end with '.', and must not "
            f"contain any of {''.join(sorted(_INVALID_NAME_CHARS))} or control characters"
        )
    return name


@dataclass(frozen=True)
class SampleRef:
    frame_paths: List[Path]
    label: int
    class_name: str
    sample_id: str
    session: str
    origin_id: str
    augmented: bool


class DataHandler:

    SEQUENCE_LENGTH = EXPECTED_FRAMES
    FEATURE_DIM = FRAME_FEATURE_DIM
    MIN_FRAMES = 8
    PRESENCE_COLS = slice(-2, None)

    def __init__(
        self,
        processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        min_samples_per_class: int = 1,
        strict: bool = False,
    ) -> None:
        self.project_dir = get_project_root_dir()
        self.data_dir = self.project_dir / "data"
        self.processed = self.data_dir / "processed"
        self.raw = self.data_dir / "raw"
        self.processed.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)

        self.processor = processor
        self.min_samples_per_class = max(1, int(min_samples_per_class))
        self.strict = strict

        self.incomplete: List[Tuple[str, str]] = []
        self.excluded_classes: Dict[str, int] = {}
        self._samples: Optional[List[SampleRef]] = None
        self._class_to_idx: Optional[Dict[str, int]] = None
        self._counts: Dict[str, int] = {}
        self._origin_counts: Dict[str, int] = {}

    @staticmethod
    def _child_dirs(root: Path) -> Dict[str, Path]:
        if not root.is_dir():
            return {}
        return {
            _normalize_nfc(directory.name): directory
            for directory in sorted(root.iterdir())
            if directory.is_dir() and not directory.name.startswith(".")
        }

    def _class_dir(self, root: Path, class_name: str) -> Path:
        name = _normalize_nfc(class_name)
        return self._child_dirs(root).get(name, root / name)

    @property
    def class_names(self) -> List[str]:
        return sorted(self._child_dirs(self.processed))

    @property
    def active_class_names(self) -> List[str]:
        self._ensure_dataset_index()
        return list(self._class_to_idx or {})

    @property
    def class_to_idx(self) -> Dict[str, int]:
        self._ensure_dataset_index()
        return dict(self._class_to_idx or {})

    def get_class_counts(self) -> Dict[str, int]:
        self._ensure_dataset_index()
        return {name: self._counts.get(name, 0) for name in self.class_names}

    def get_origin_counts(self) -> Dict[str, int]:
        self._ensure_dataset_index()
        return {name: self._origin_counts.get(name, 0) for name in self.class_names}

    def add_class(self, class_name: str) -> str:
        name = _check_class_name(class_name)
        self._class_dir(self.processed, name).mkdir(parents=True, exist_ok=True)
        self._class_dir(self.raw, name).mkdir(parents=True, exist_ok=True)
        self._invalidate_dataset_index()
        return name

    def remove_class(self, class_name: str) -> None:
        name = _normalize_nfc(class_name)
        pdir, rdir = self._class_dir(self.processed, name), self._class_dir(self.raw, name)
        if not pdir.exists() and not rdir.exists():
            raise ValueError(f"Class not found: {class_name}")
        for directory in (pdir, rdir):
            if directory.is_dir():
                shutil.rmtree(directory)
        self._invalidate_dataset_index()

    def _invalidate_dataset_index(self) -> None:
        self._samples = None
        self._class_to_idx = None

    @staticmethod
    def _frame_paths(sample_dir: Path) -> Tuple[List[Path], Optional[str]]:
        stems = sorted(int(path.stem) for path in sample_dir.glob("*.npz") if path.stem.isdigit())
        if not stems:
            return [], "no frame files"
        if stems != list(range(len(stems))):
            return [], f"non-contiguous frame files ({len(stems)} files, max index {stems[-1]})"
        if len(stems) < DataHandler.MIN_FRAMES:
            return [], f"only {len(stems)} frames (< {DataHandler.MIN_FRAMES})"
        return [sample_dir / f"{frame_index}.npz" for frame_index in stems], None

    @staticmethod
    def _read_meta(sample_dir: Path) -> dict:
        try:
            return json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _scan_dataset(self) -> Tuple[List[SampleRef], Dict[str, int]]:
        self.incomplete = []
        pending: List[Tuple[str, str, List[Path], str, str, bool]] = []
        counts: Counter = Counter()
        origins: Dict[str, set[str]] = defaultdict(set)

        for class_name, class_dir in self._child_dirs(self.processed).items():
            sample_dirs = sorted(
                (entry for entry in class_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
                key=lambda entry: (0 if entry.name.isdigit() else 1, int(entry.name) if entry.name.isdigit() else entry.name),
            )
            for sample_dir in sample_dirs:
                frame_paths, problem = self._frame_paths(sample_dir)
                if problem:
                    if self.strict:
                        raise RuntimeError(f"Incomplete sample: {sample_dir} ({problem})")
                    self.incomplete.append((str(sample_dir), problem))
                    continue
                meta = self._read_meta(sample_dir)
                session = str(meta.get("session", "legacy"))
                augmented = "augmentation" in meta
                origin_id = str(meta.get("origin_sample_id", sample_dir.name))
                pending.append((class_name, sample_dir.name, frame_paths, session, origin_id, augmented))
                counts[class_name] += 1
                if not augmented:
                    origins[class_name].add(origin_id)

        self._counts = dict(counts)
        self._origin_counts = {class_key: len(origins[class_key]) for class_key in counts}
        active = [class_key for class_key in sorted(counts) if self._origin_counts[class_key] >= self.min_samples_per_class]
        self.excluded_classes = {
            class_key: self._origin_counts[class_key] for class_key in counts if self._origin_counts[class_key] < self.min_samples_per_class
        }
        class_to_idx = {class_key: class_index for class_index, class_key in enumerate(active)}
        samples = [
            SampleRef(paths, class_to_idx[class_key], class_key, sample_id, sess, origin_id, augmented)
            for class_key, sample_id, paths, sess, origin_id, augmented in pending
            if class_key in class_to_idx
        ]
        return samples, class_to_idx

    def _ensure_dataset_index(self) -> None:
        if self._samples is None or self._class_to_idx is None:
            self._samples, self._class_to_idx = self._scan_dataset()

    def refresh_dataset(self) -> int:
        self._invalidate_dataset_index()
        self._ensure_dataset_index()
        return len(self._samples or [])

    def sample_groups(self, by: str = "sample") -> List[str]:
        self._ensure_dataset_index()
        if by == "session":
            return [sample.session for sample in self._samples or []]
        if by == "sample":
            return [f"{sample.class_name}/{sample.origin_id}" for sample in self._samples or []]
        raise ValueError("by must be 'sample' or 'session'")

    def describe(self, index: int) -> dict:
        self._ensure_dataset_index()
        sample = (self._samples or [])[index]
        return {"class": sample.class_name, "sample_id": sample.sample_id, "label": sample.label,
                "session": sample.session, "origin_sample_id": sample.origin_id, "augmented": sample.augmented}

    @classmethod
    def _load_feature(cls, path: Path, allow_nan: bool = True) -> np.ndarray:
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != 1:
                raise ValueError(f"{path} contains {len(data.files)} arrays: {data.files}. Expected exactly one.")
            feature = data[data.files[0]]
        if feature.shape != (cls.FEATURE_DIM,):
            raise ValueError(
                f"Invalid feature shape in {path}: {feature.shape}, expected ({cls.FEATURE_DIM},). "
                "Nếu vừa đổi extractor, cập nhật FRAME_FEATURE_DIM trong src/features/schema.py "
                "và trích xuất lại dữ liệu cũ."
            )
        if feature.dtype != np.float32:
            raise TypeError(f"Invalid dtype in {path}: {feature.dtype}, expected float32")
        if np.isinf(feature).any():
            raise ValueError(f"Inf detected in {path}")
        if not allow_nan and np.isnan(feature).any():
            raise ValueError(f"NaN detected in {path}")
        return feature

    def validate_dataset(self) -> int:
        self.refresh_dataset()
        for sample in self._samples or []:
            for frame_path in sample.frame_paths:
                self._load_feature(frame_path)
        return len(self._samples or [])

    def __len__(self) -> int:
        self._ensure_dataset_index()
        return len(self._samples or [])

    def load_raw(self, index: int) -> np.ndarray:
        self._ensure_dataset_index()
        samples = self._samples or []
        index = int(index)
        if index < 0:
            index += len(samples)
        if index < 0 or index >= len(samples):
            raise IndexError(f"Dataset index out of range: {index}")
        return np.stack([self._load_feature(frame_path) for frame_path in samples[index].frame_paths], axis=0)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        raw = self.load_raw(index)
        label = (self._samples or [])[index].label
        if self.processor is not None:
            sequence = np.asarray(self.processor(raw), dtype=np.float32)
        elif raw.shape[0] == self.SEQUENCE_LENGTH:
            sequence = raw
        else:
            raise ValueError(
                f"Sample {index} has {raw.shape[0]} frames but SEQUENCE_LENGTH={self.SEQUENCE_LENGTH}. "
                "Truyền processor=process_sequence để co giãn thời gian."
            )
        return sequence, label

    def _next_sample_idx(self, proc_cls: Path, raw_cls: Path) -> int:
        used = [int(directory.name) for directory in proc_cls.iterdir() if directory.is_dir() and directory.name.isdigit()] if proc_cls.is_dir() else []
        if raw_cls.is_dir():
            used += [int(path.stem) for path in raw_cls.glob("*.mp4") if path.stem.isdigit()]
        return (max(used) + 1) if used else 0

    @staticmethod
    def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (FRAME_SIZE, FRAME_SIZE)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {path}")
        try:
            for frame in frames:
                writer.write(frame_padding(frame))
        finally:
            writer.release()

    def write_sample(
        self,
        class_name: str,
        features: Union[List[np.ndarray], np.ndarray],
        frames: Optional[Sequence[np.ndarray]] = None,
        fps: float = 30.0,
        session: Optional[str] = None,
        meta: Optional[dict] = None,
        require_hand: bool = True,
    ) -> int:
        feats = [np.asarray(frame_feature) for frame_feature in features]
        frame_count = len(feats)
        if frame_count < self.MIN_FRAMES:
            raise ValueError(f"Sample must contain at least {self.MIN_FRAMES} frames, got {frame_count}.")
        for frame_index, frame_feature in enumerate(feats):
            if frame_feature.shape != (self.FEATURE_DIM,):
                raise ValueError(f"Invalid feature shape at frame {frame_index}: {frame_feature.shape}, expected ({self.FEATURE_DIM},)")
            if frame_feature.dtype != np.float32:
                raise TypeError(f"Invalid feature dtype at frame {frame_index}: {frame_feature.dtype}, expected float32")
            if np.isinf(frame_feature).any():
                raise ValueError(f"Inf detected at frame {frame_index}.")
        stack = np.stack(feats)
        if np.isnan(stack).all(axis=1).all():
            raise ValueError("Không detect được pose ở frame nào -> không lưu.")
        if require_hand and np.nan_to_num(stack[:, self.PRESENCE_COLS]).sum() == 0:
            raise ValueError("Không detect được tay ở frame nào -> không lưu (require_hand=False để bỏ qua).")
        if frames is not None:
            if len(frames) != frame_count:
                raise ValueError("Quantity of features and frames must be equal.")
            if any(frame.dtype != np.uint8 or frame.ndim != 3 for frame in frames):
                raise TypeError("frames must be uint8 BGR images with shape (H, W, 3).")

        name = self.add_class(class_name)
        proc_cls, raw_cls = self._class_dir(self.processed, name), self._class_dir(self.raw, name)
        sample_idx = self._next_sample_idx(proc_cls, raw_cls)

        token = uuid.uuid4().hex[:8]
        tmp_dir, tmp_video = proc_cls / f".tmp_{token}", raw_cls / f".tmp_{token}.mp4"
        final_dir, final_video = proc_cls / str(sample_idx), raw_cls / f"{sample_idx}.mp4"
        committed_dir = False
        try:
            tmp_dir.mkdir()
            for frame_index, feat in enumerate(feats):
                np.savez_compressed(tmp_dir / f"{frame_index}.npz", feature=feat)
            info = {
                "session": session or "unspecified",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_frames": frame_count,
                "fps": float(fps),
                "feature_dim": self.FEATURE_DIM,
                **(meta or {}),
            }
            (tmp_dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
            if frames is not None:
                self._write_video(tmp_video, frames, fps)
            os.replace(tmp_dir, final_dir)
            committed_dir = True
            if frames is not None:
                os.replace(tmp_video, final_video)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if tmp_video.exists():
                tmp_video.unlink()
            if committed_dir:
                shutil.rmtree(final_dir, ignore_errors=True)
            raise
        finally:
            self._invalidate_dataset_index()
        return sample_idx

    def remove_sample(self, class_name: str, sample_idx: int) -> None:
        pdir = self._class_dir(self.processed, class_name) / str(sample_idx)
        video = self._class_dir(self.raw, class_name) / f"{sample_idx}.mp4"
        if not pdir.exists() and not video.exists():
            raise ValueError(f"Sample not found: {class_name}/{sample_idx}")
        if pdir.is_dir():
            shutil.rmtree(pdir)
        if video.exists():
            video.unlink()
        self._invalidate_dataset_index()

    def quality_report(self) -> List[dict]:
        self._ensure_dataset_index()
        rows = []
        for index, sample in enumerate(self._samples or []):
            raw = self.load_raw(index)
            presence = np.nan_to_num(raw[:, self.PRESENCE_COLS])
            rows.append({
                "index": index, "class": sample.class_name, "sample_id": sample.sample_id, "session": sample.session,
                "frames": int(raw.shape[0]),
                "pose_coverage": float((~np.isnan(raw).all(axis=1)).mean()),
                "left_hand": float(presence[:, 0].mean()), "right_hand": float(presence[:, 1].mean()),
            })
        return rows

    def export_arrays(self, group_by: str = "sample", progress: bool = True) -> dict:
        self._ensure_dataset_index()
        sample_count = len(self)
        if sample_count == 0:
            raise RuntimeError("Dataset trống (chưa có mẫu hoàn chỉnh).")
        sequences, labels = [], []
        for index in range(sample_count):
            sequence, label = self[index]
            sequences.append(sequence)
            labels.append(label)
            if progress and (index + 1) % 100 == 0:
                print(f"  đã đọc {index + 1}/{sample_count} mẫu", flush=True)
        shapes = {array.shape for array in sequences}
        if len(shapes) != 1:
            raise ValueError(f"Các mẫu có shape khác nhau: {sorted(shapes)} -> cần processor để co giãn.")
        return {
            "X": np.stack(sequences).astype(np.float32),
            "y": np.asarray(labels, dtype=np.int64),
            "classes": self.active_class_names,
            "groups": np.asarray(self.sample_groups(group_by)),
            "sessions": np.asarray(self.sample_groups("session")),
            "augmented": np.asarray([sample.augmented for sample in self._samples or []], dtype=bool),
        }

    def save_npz(self, path: Union[str, Path], group_by: str = "sample") -> Path:
        data = self.export_arrays(group_by=group_by)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{**data, "classes": np.asarray(data["classes"])})
        return path
