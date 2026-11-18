"""Rebuild per-frame VSL features from the original recorded videos."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.builtin.mediapipe_builtin.defs import HandLandmarkerPath, PoseLandmarkerPath
from src.data.data_reader import DataHandler
from src.features.extract import FEATURE_DIM, FEATURE_VERSION, SEQ_FEATURE_DIM, extract, process_sequence
from src.features.frame import FRAME_SIZE, frame_padding


def find_raw_samples(raw_root: Path, classes: list[str] | None = None) -> list[Path]:
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw video directory not found: {raw_root}")
    wanted = set(classes) if classes is not None else None
    videos = []
    for class_dir in sorted(raw_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        if wanted is not None and class_dir.name not in wanted:
            continue
        videos.extend(
            sorted(
                (p for p in class_dir.glob("*.mp4") if p.stem.isdigit()),
                key=lambda p: int(p.stem),
            )
        )
    if wanted is not None:
        missing = wanted - {p.name for p in raw_root.iterdir() if p.is_dir()}
        if missing:
            raise ValueError(f"Classes not found in raw videos: {sorted(missing)}")
    return videos


def extract_video_features(video_path: Path) -> tuple[np.ndarray, float]:
    """Decode every frame, pad it to the FRAME_SIZE square, and run the builtin landmarks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    features = []
    loader = None
    last_ms = -1
    try:
        loader = LandmarkerLoader(running_mode=VisionRunningMode.VIDEO)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = frame_padding(frame)
            timestamp_ms = max(round(len(features) * 1000 / fps), last_ms + 1)
            pose, hand, _ = loader.detect(frame, timestamp_ms=timestamp_ms)
            features.append(extract(pose, hand))
            last_ms = timestamp_ms
    finally:
        cap.release()
        if loader is not None:
            loader.close()

    if len(features) < DataHandler.MIN_FRAMES:
        raise ValueError(f"Only {len(features)} frames decoded from {video_path}")
    result = np.stack(features)
    if result.shape[1] != FEATURE_DIM or result.dtype != np.float32 or np.isinf(result).any():
        raise ValueError(f"Invalid extracted features from {video_path}: {result.shape}, {result.dtype}")
    if np.isnan(result).all(axis=1).all():
        raise ValueError(f"No pose detected in {video_path}")
    if np.nan_to_num(result[:, DataHandler.PRESENCE_COLS]).sum() == 0:
        raise ValueError(f"No hands detected in {video_path}")
    sequence = process_sequence(result)
    if sequence.shape != (DataHandler.SEQUENCE_LENGTH, SEQ_FEATURE_DIM) or not np.isfinite(sequence).all():
        raise ValueError(f"Invalid processed sequence from {video_path}: {sequence.shape}")
    return result, fps


def reextract_sample(
    video_path: Path,
    raw_root: Path,
    processed_root: Path,
    backup_root: Path,
) -> dict:
    """Replace one processed sample after validation, retaining its old directory."""
    video_path = Path(video_path)
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    backup_root = Path(backup_root)
    if video_path.parent.parent.resolve() != raw_root.resolve() or not video_path.stem.isdigit():
        raise ValueError(f"Expected raw/<class>/<numeric id>.mp4: {video_path}")
    if processed_root.resolve() == backup_root.resolve():
        raise ValueError("Backup and processed roots must differ")

    class_name = video_path.parent.name
    target = processed_root / class_name / video_path.stem
    backup = backup_root / class_name / video_path.stem
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Processed sample is not a directory: {target}")

    old_meta = DataHandler._read_meta(target) if target.is_dir() else {}
    features, fps = extract_video_features(video_path)
    pose_coverage = float((~np.isnan(features).all(axis=1)).mean())
    hands = np.nan_to_num(features[:, DataHandler.PRESENCE_COLS])
    meta = {
        **old_meta,
        "session": old_meta.get("session", "legacy"),
        "n_frames": len(features),
        "fps": fps,
        "feature_dim": FEATURE_DIM,
        "feature_version": FEATURE_VERSION,
        "frame_size": FRAME_SIZE,
        "pose_coverage": pose_coverage,
        "left_hand": float(hands[:, 0].mean()),
        "right_hand": float(hands[:, 1].mean()),
        "reextracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_video": video_path.relative_to(raw_root).as_posix(),
        "pose_model": Path(PoseLandmarkerPath).name,
        "hand_model": Path(HandLandmarkerPath).name,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".reextract_{video_path.stem}_{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        for index, feature in enumerate(features):
            np.savez_compressed(stage / f"{index}.npz", feature=feature)
        (stage / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for index in range(len(features)):
            DataHandler._load_feature(stage / f"{index}.npz")

        if target.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.rename(backup)
        try:
            stage.rename(target)
        except BaseException:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return {
        "class": class_name,
        "sample_id": video_path.stem,
        "frames": len(features),
        "pose_coverage": pose_coverage,
        "left_hand": meta["left_hand"],
        "right_hand": meta["right_hand"],
        "backup": str(backup) if backup.exists() else None,
    }
