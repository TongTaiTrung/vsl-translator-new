"""Create augmented variants of original samples while keeping the source group for validation."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from heapq import heappop, heappush
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.data_reader import DataHandler
from src.features.extract import FEATURE_SLICES, FEATURE_VERSION, PITCH_AUG_DEG, rotate_view_pitch
from src.features.landmarks import LANDMARKS_FILE, load_landmarks
from src.features.view import features_from_landmarks, random_view, simulate_view


def available_sources(processed_root: Path, classes: list[str] | None = None) -> list[Path]:
    processed_root = Path(processed_root)
    if not processed_root.is_dir():
        raise FileNotFoundError(processed_root)
    wanted = set(classes) if classes is not None else None
    existing = {path.name for path in processed_root.iterdir() if path.is_dir() and not path.name.startswith(".")}
    if wanted is not None and wanted - existing:
        raise ValueError(f"Unknown classes: {sorted(wanted - existing)}")
    sources = []
    for class_dir in sorted(processed_root.iterdir()):
        if class_dir.name.startswith(".") or not class_dir.is_dir():
            continue
        if wanted is not None and class_dir.name not in wanted:
            continue
        for sample_dir in sorted(class_dir.iterdir(), key=lambda entry: (not entry.name.isdigit(), entry.name)):
            if not sample_dir.is_dir() or not sample_dir.name.isdigit():
                continue
            if "augmentation" in DataHandler._read_meta(sample_dir):
                continue
            paths, problem = DataHandler._frame_paths(sample_dir)
            if problem is None and paths:
                sources.append(sample_dir)
    return sources


def plan_augmentation(
    processed_root: Path, classes: list[str] | None,
    minimum_samples: int, copies_per_sample: int,
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    if minimum_samples < -1:
        raise ValueError("minimum_samples must be -1 or non-negative")
    if copies_per_sample < 1:
        raise ValueError("copies_per_sample must be positive")

    root = Path(processed_root)
    sources = available_sources(root, classes)
    selected = sorted({entry.name for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")
                       and (classes is None or entry.name in classes)})
    by_class: dict[str, list[Path]] = {name: [] for name in selected}
    for source in sources:
        by_class[source.parent.name].append(source)

    counts = {}
    jobs = []
    for class_name in selected:
        class_dir = root / class_name
        per_origin = {source.name: 0 for source in by_class[class_name]}
        total = 0
        for sample_dir in class_dir.iterdir():
            if not sample_dir.is_dir() or not sample_dir.name.isdigit():
                continue
            paths, problem = DataHandler._frame_paths(sample_dir)
            if problem is not None or not paths:
                continue
            total += 1
            origin = str(DataHandler._read_meta(sample_dir).get("origin_sample_id", sample_dir.name))
            if origin in per_origin:
                per_origin[origin] += 1
        counts[class_name] = total

        if minimum_samples == -1:
            jobs.extend((source, copies_per_sample) for source in by_class[class_name])
            continue
        deficit = max(0, minimum_samples - total)
        if deficit == 0:
            continue
        if not by_class[class_name]:
            raise ValueError(f"Class {class_name} has no complete original sample to augment")
        queue = [(per_origin[source.name], int(source.name), source) for source in by_class[class_name]]
        queue.sort()
        assigned = {source: 0 for source in by_class[class_name]}
        for _ in range(deficit):
            count, sample_id, source = heappop(queue)
            assigned[source] += 1
            heappush(queue, (count + 1, sample_id, source))
        jobs.extend((source, number) for source, number in assigned.items() if number)
    return jobs, counts


def add_feature_noise(features: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    if not np.isfinite(strength) or strength <= 0 or strength > 3:
        raise ValueError("strength must be in (0, 3]")
    source = np.asarray(features, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != DataHandler.FEATURE_DIM:
        raise ValueError(f"Expected (frames, {DataHandler.FEATURE_DIM}), got {source.shape}")
    out = source.copy()
    slices = FEATURE_SLICES
    position = slice(0, slices["body_contact"].start)
    contact = slice(slices["body_contact"].start, slices["face_vec"].start)
    scales = ((position, 0.015), (contact, 0.01), (slices["face_vec"], 0.015), (slices["hand_scale"], 0.01),
              (slices["handshape"], 0.012), (slices["hand_local"], 0.012), (slices["finger_curl"], 0.012),
              (slices["hand_closure"], 0.012), (slices["orientation"], 0.01))
    for cols, sigma in scales:
        block = out[:, cols]
        perturbation = rng.normal(0, sigma * strength, size=block.shape).astype(np.float32)
        np.add(block, perturbation, out=block, where=np.isfinite(block))
    np.clip(out[:, contact], 0, 1, out=out[:, contact])
    np.clip(out[:, slices["orientation"]], -1, 1, out=out[:, slices["orientation"]])
    return out


def load_source_landmarks(source_dir: Path, n_frames: int):
    path = Path(source_dir) / LANDMARKS_FILE
    if not path.is_file():
        return None
    landmarks = load_landmarks(path)
    return landmarks if len(landmarks[0]) == n_frames else None


def make_variant(features: np.ndarray, landmarks, rng: np.random.Generator, strength: float,
                 pitch_degrees: float = PITCH_AUG_DEG, view_kwargs: dict | None = None) -> tuple[np.ndarray, dict]:
    if landmarks is not None:
        view = random_view(rng, pitch=pitch_degrees, **(view_kwargs or {}))
        viewed = features_from_landmarks(*simulate_view(*landmarks, view))
        out = add_feature_noise(viewed, rng, strength)
        return out, {"type": "landmark_view", "strength": float(strength),
                     "view_pitch_deg": round(view.pitch, 3), "view": view.to_dict()}
    out = add_feature_noise(features, rng, strength)
    view_pitch = float(rng.uniform(-pitch_degrees, pitch_degrees)) if pitch_degrees > 0 else 0.0
    if view_pitch:
        out = rotate_view_pitch(out, view_pitch)
    return out, {"type": "feature_noise", "strength": float(strength), "view_pitch_deg": round(view_pitch, 3)}


def augment_sample(source_dir: Path, raw_root: Path, copies: int, strength: float,
                   rng: np.random.Generator, pitch_degrees: float = PITCH_AUG_DEG) -> list[Path]:
    source_dir = Path(source_dir)
    raw_root = Path(raw_root)
    if copies < 1:
        raise ValueError("copies must be positive")
    if not np.isfinite(pitch_degrees) or not 0 <= pitch_degrees <= 45:
        raise ValueError("pitch_degrees must be in [0, 45]")
    meta = DataHandler._read_meta(source_dir)
    if "augmentation" in meta:
        raise ValueError(f"Cannot augment a derived sample: {source_dir}")
    if meta.get("feature_version", 1) != FEATURE_VERSION:
        raise ValueError(f"Source has stale features: {source_dir}")
    class_dir = source_dir.parent
    source_id = source_dir.name
    if not source_id.isdigit():
        raise ValueError(f"Source ID must be numeric: {source_dir}")
    paths, problem = DataHandler._frame_paths(source_dir)
    if problem:
        raise ValueError(f"Incomplete source {source_dir}: {problem}")
    features = np.stack([DataHandler._load_feature(path) for path in paths])
    if np.isnan(features).all(axis=1).all():
        raise ValueError(f"Source has no detected pose: {source_dir}")
    landmarks = load_source_landmarks(source_dir, len(features))
    used = {int(entry.name) for entry in class_dir.iterdir() if entry.is_dir() and entry.name.isdigit()}
    raw_class_dir = raw_root / class_dir.name
    if raw_class_dir.is_dir():
        used.update(int(entry.stem) for entry in raw_class_dir.glob("*.mp4") if entry.stem.isdigit())
    next_id = max(used, default=-1) + 1
    created = []
    for copy_number in range(copies):
        target = class_dir / str(next_id + copy_number)
        stage = class_dir / f".augment_{uuid.uuid4().hex}"
        try:
            augmented, aug_info = make_variant(features, landmarks, rng, strength, pitch_degrees)
            stage.mkdir()
            for index, feature in enumerate(augmented):
                np.savez_compressed(stage / f"{index}.npz", feature=feature)
            info = {
                **meta,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_frames": len(augmented),
                "feature_dim": DataHandler.FEATURE_DIM,
                "feature_version": FEATURE_VERSION,
                "origin_sample_id": source_id,
                "augmentation": {"source_sample_id": source_id, **aug_info},
            }
            (stage / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            os.rename(stage, target)
            created.append(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return created
