"""Reusable VSL prediction from feature sequences or a stream of BGR frames."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.features.extract import FEATURE_DIM, FEATURE_VERSION, SEQ_FEATURE_DIM, extract, process_sequence
from src.features.frame import frame_padding
from src.model import training


@dataclass(frozen=True)
class StreamConfig:
    window_seconds: float = 2.0
    min_window_seconds: float = 1.5
    min_frames: int = 8
    predict_every: int = 3
    stability_frames: int = 3
    no_hand_timeout: float = 0.6
    mirror_data: bool = False
    top_k: int = 3

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or not 0 < self.min_window_seconds <= self.window_seconds:
            raise ValueError("Window durations must be positive and min_window_seconds <= window_seconds")
        if self.min_frames < 1 or self.predict_every < 1 or self.stability_frames < 1 or self.top_k < 1:
            raise ValueError("Frame counts and top_k must be positive")
        if self.no_hand_timeout <= 0:
            raise ValueError("no_hand_timeout must be positive")


class VSLPredictor:
    """Load one checkpoint and predict independently from complete feature sequences.

    Share this object across requests. Use a separate ``VSLStreamSession`` for
    each live camera/client because its frame history and MediaPipe state vary.
    """

    def __init__(self, checkpoint_path: str | Path | None = None, *, device=None,
                 min_confidence: float = 0.65, min_probability_gap: float = 0.08) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0 <= min_probability_gap <= 1:
            raise ValueError("min_probability_gap must be in [0, 1]")
        if checkpoint_path is None:
            checkpoints = list((Path(__file__).resolve().parents[2] / "runs").glob("*/ensemble.pt"))
            if not checkpoints:
                raise FileNotFoundError("No ensemble.pt found under runs/; train the model first")
            checkpoint_path = max(checkpoints, key=lambda path: path.stat().st_mtime)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)

        self.device = device or training.get_device()
        self.ensemble, self.checkpoint = training.load_ensemble(self.checkpoint_path, device=self.device)
        self.classes = tuple(self.checkpoint["classes"])
        self.target_len, self.feature_dim = map(int, self.checkpoint["in_shape"])
        if self.feature_dim != SEQ_FEATURE_DIM:
            raise ValueError(
                f"Checkpoint expects {self.feature_dim} features; current extractor outputs {SEQ_FEATURE_DIM}"
            )
        self.feature_version = int(self.checkpoint.get("feature_version", 1))
        self.threshold = max(float(self.checkpoint["reject_threshold"]), min_confidence)
        self.min_probability_gap = min_probability_gap

    def predict_processed(self, sequence: np.ndarray, *, top_k: int = 3) -> dict:
        """Predict a processed ``(T, D)`` sequence; return JSON-compatible data."""
        features = np.asarray(sequence, dtype=np.float32)
        if features.shape != (self.target_len, self.feature_dim):
            raise ValueError(f"Expected processed features {(self.target_len, self.feature_dim)}, got {features.shape}")
        if np.isinf(features).any():
            raise ValueError("Features cannot contain infinity")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        ranked = training.predict_topk(
            self.ensemble, features, self.classes,
            k=min(max(top_k, 2), len(self.classes)), reject_threshold=self.threshold,
        )[0]
        confident = ranked[0][0] != "unknown"
        candidates = ranked if confident else ranked[1:]
        confidence = float(candidates[0][1])
        gap = confidence - float(candidates[1][1]) if len(candidates) > 1 else 1.0
        accepted = confident and gap >= self.min_probability_gap
        return {
            "label": str(candidates[0][0]) if accepted else None,
            "confidence": confidence,
            "probability_gap": gap,
            "accepted": accepted,
            "rejection_reason": None if accepted else "ambiguous" if confident else "low_confidence",
            "top": [{"label": str(label), "confidence": float(score)} for label, score in candidates[:top_k]],
        }

    def predict_raw(self, raw_features: np.ndarray, *, top_k: int = 3) -> dict:
        """Process raw ``(frames, 131)`` features and predict a complete clip."""
        features = np.asarray(raw_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected nonempty raw features (frames, {FEATURE_DIM}), got {features.shape}")
        if np.isinf(features).any():
            raise ValueError("Features cannot contain infinity")
        sequence = process_sequence(features, target_len=self.target_len)
        return self.predict_processed(sequence, top_k=top_k)

    def predict_frames(self, frames_bgr: Iterable[np.ndarray], *, mirror_data: bool = False,
                       top_k: int = 3) -> dict:
        """Predict one complete clip of BGR frames without persistent stream state."""
        features = []
        landmarkers = LandmarkerLoader(running_mode=VisionRunningMode.VIDEO)
        try:
            for frame in frames_bgr:
                if mirror_data:
                    frame = cv2.flip(frame, 1)
                frame = frame_padding(frame)
                pose_result, hand_result, _ = landmarkers.detect(frame)
                features.append(extract(pose_result, hand_result))
        finally:
            landmarkers.close()
        if not features:
            raise ValueError("At least one BGR frame is required")
        return self.predict_raw(np.stack(features), top_k=top_k)

    def open_stream(self, config: StreamConfig | None = None) -> VSLStreamSession:
        return VSLStreamSession(self, config=config)


class VSLStreamSession:
    """Stateful prediction for one video stream; never share between clients."""

    def __init__(self, predictor: VSLPredictor, config: StreamConfig | None = None,
                 landmarkers=None) -> None:
        self.predictor = predictor
        self.config = config or StreamConfig()
        self.landmarkers = landmarkers if landmarkers is not None else LandmarkerLoader(
            running_mode=VisionRunningMode.VIDEO
        )
        self._closed = False
        self.reset()

    def reset(self) -> None:
        """Start a fresh signing window without reloading the model or MediaPipe."""
        self.history: deque[tuple[float, np.ndarray]] = deque()
        self.recent_labels: deque[str | None] = deque(maxlen=self.config.stability_frames)
        self.prediction: dict | None = None
        self.stable_label: str | None = None
        self.last_hand_time: float | None = None
        self.frame_count = 0
        self.last_frame: np.ndarray | None = None
        self.last_pose_result = None
        self.last_hand_result = None

    def update_frame(self, frame_bgr: np.ndarray, *, now: float | None = None) -> dict:
        """Consume a BGR frame and return JSON-compatible prediction state."""
        if self._closed:
            raise RuntimeError("Stream session is closed")
        frame = cv2.flip(frame_bgr, 1) if self.config.mirror_data else frame_bgr
        frame = frame_padding(frame)
        pose_result, hand_result, _ = self.landmarkers.detect(frame)
        feature = extract(pose_result, hand_result)
        now = time.monotonic() if now is None else float(now)
        hand_present = bool(np.nan_to_num(feature[-2:], nan=0.0).sum() > 0)

        self.last_frame = frame
        self.last_pose_result = pose_result
        self.last_hand_result = hand_result
        if hand_present:
            self.last_hand_time = now
        if self.last_hand_time is None or now - self.last_hand_time > self.config.no_hand_timeout:
            self.history.clear()
            self.recent_labels.clear()
            self.prediction = None
            self.stable_label = None
            self.last_hand_time = None
        else:
            self.history.append((now, feature))
            while self.history and now - self.history[0][0] > self.config.window_seconds:
                self.history.popleft()
            if not hand_present:
                self.stable_label = None
                self.recent_labels.clear()

        duration = now - self.history[0][0] if self.history else 0.0
        ready = hand_present and len(self.history) >= self.config.min_frames \
            and duration >= self.config.min_window_seconds
        self.frame_count += 1
        if ready and self.frame_count % self.config.predict_every == 0:
            raw = np.stack([item[1] for item in self.history])
            self.prediction = self.predictor.predict_raw(raw, top_k=self.config.top_k)
            label = self.prediction["label"]
            self.recent_labels.append(label)
            self.stable_label = label if label and len(self.recent_labels) == self.config.stability_frames \
                and all(candidate == label for candidate in self.recent_labels) else None

        status = "idle" if self.last_hand_time is None else "collecting" if not ready \
            else "ready" if self.stable_label else "predicting"
        return {
            "status": status,
            "prediction": self.prediction,
            "stable_label": self.stable_label,
            "hand_present": hand_present,
            "frames": len(self.history),
            "duration": float(duration),
        }

    def close(self) -> None:
        if not self._closed:
            self.landmarkers.close()
            self._closed = True

    def __enter__(self) -> VSLStreamSession:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
