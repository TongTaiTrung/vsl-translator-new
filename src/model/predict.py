"""Reusable VSL prediction from feature sequences or a stream of BGR frames."""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.features.extract import (FEATURE_DIM, FEATURE_VERSION, SEQ_FEATURE_DIM, active_presence, extract,
                                  process_sequence)
from src.features.frame import frame_padding
from src.model import training
from src.model.checkpoints import latest_checkpoint


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
    target_fps: float | None = 15.0   # dữ liệu thu ~15 fps; lấp gap/khử gai tính theo số frame. None = nhận mọi frame

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or not 0 < self.min_window_seconds <= self.window_seconds:
            raise ValueError("Window durations must be positive and min_window_seconds <= window_seconds")
        if self.min_frames < 1 or self.predict_every < 1 or self.stability_frames < 1 or self.top_k < 1:
            raise ValueError("Frame counts and top_k must be positive")
        if self.no_hand_timeout <= 0:
            raise ValueError("no_hand_timeout must be positive")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be positive or None")


class VSLPredictor:
    """Load one checkpoint and predict independently from complete feature sequences.

    Share this object across requests. Use a separate ``VSLStreamSession`` for
    each live camera/client because its frame history and MediaPipe state vary.
    """

    def __init__(self, checkpoint_path: str | Path | None = None, *, device=None,
                 min_confidence: float = 0.65, min_probability_gap: float = 0.08,
                 allow_feature_version_mismatch: bool = False) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0 <= min_probability_gap <= 1:
            raise ValueError("min_probability_gap must be in [0, 1]")
        if checkpoint_path is None:
            checkpoint_path = latest_checkpoint()          # runs/best, không có thì run mới nhất theo tên
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
        if self.feature_version != FEATURE_VERSION and not allow_feature_version_mismatch:
            raise ValueError(
                f"Checkpoint có feature_version={self.feature_version}, "
                f"nhưng extractor hiện tại là FEATURE_VERSION={FEATURE_VERSION}. "
                "Hãy chạy lại reextract_features rồi train lại checkpoint, hoặc "
                "truyền allow_feature_version_mismatch=True nếu chắc chắn muốn bỏ qua."
            )
        self.threshold = max(float(self.checkpoint["reject_threshold"]), min_confidence)
        self.min_probability_gap = min_probability_gap

    def predict_processed(self, sequence: np.ndarray, *, top_k: int = 3,
                          ensemble: training.VSLEnsemble | None = None) -> dict:
        """Predict a processed ``(T, D)`` sequence; return JSON-compatible data.

        ensemble: chạy một ensemble khác (vd. một model con để sàng lọc nhanh, hoặc
        ``memory.MemoryEnsemble`` có bộ nhớ mẫu cá nhân) với cùng ngưỡng từ chối; mặc định là
        ensemble của checkpoint. Với ``MemoryEnsemble`` kết quả có thêm ``memory`` (lớp có mẫu giống nhất).
        """
        features = np.asarray(sequence, dtype=np.float32)
        if features.shape != (self.target_len, self.feature_dim):
            raise ValueError(f"Expected processed features {(self.target_len, self.feature_dim)}, got {features.shape}")
        if np.isinf(features).any():
            raise ValueError("Features cannot contain infinity")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        model = self.ensemble if ensemble is None else ensemble
        ranked = training.predict_topk(
            model, features, self.classes,
            k=min(max(top_k, 2), len(self.classes)), reject_threshold=self.threshold,
        )[0]
        confident = ranked[0][0] != "unknown"
        candidates = ranked if confident else ranked[1:]
        confidence = float(candidates[0][1])
        gap = confidence - float(candidates[1][1]) if len(candidates) > 1 else 1.0
        accepted = confident and gap >= self.min_probability_gap
        result = {
            "label": str(candidates[0][0]) if accepted else None,
            "confidence": confidence,
            "probability_gap": gap,
            "accepted": accepted,
            "rejection_reason": None if accepted else "ambiguous" if confident else "low_confidence",
            "top": [{"label": str(label), "confidence": float(score)} for label, score in candidates[:top_k]],
        }
        match = getattr(model, "last_match", None)
        if match is not None:
            result["memory"] = dict(match)
        return result

    def predict_raw(self, raw_features: np.ndarray, *, top_k: int = 3, ensemble=None) -> dict:
        """Process raw ``(frames, FEATURE_DIM)`` features and predict a complete clip."""
        features = np.asarray(raw_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected nonempty raw features (frames, {FEATURE_DIM}), got {features.shape}")
        if np.isinf(features).any():
            raise ValueError("Features cannot contain infinity")
        sequence = process_sequence(features, target_len=self.target_len)
        return self.predict_processed(sequence, top_k=top_k, ensemble=ensemble)

    def predict_frames(self, frames_bgr: Iterable[np.ndarray], *, mirror_data: bool = False,
                       top_k: int = 3, fps: float | None = None) -> dict:
        """Predict one complete clip of BGR frames without persistent stream state.

        fps: nếu biết fps của video, timestamp gửi MediaPipe theo đúng nhịp (giống reextract).
        """
        features = []
        landmarkers = LandmarkerLoader(running_mode=VisionRunningMode.VIDEO)
        try:
            for i, frame in enumerate(frames_bgr):
                if mirror_data:
                    frame = cv2.flip(frame, 1)
                frame = frame_padding(frame)
                timestamp = None if fps is None else round(i * 1000 / fps)   # giống reextract
                pose_result, hand_result, _ = landmarkers.detect(frame, timestamp_ms=timestamp)
                features.append(extract(pose_result, hand_result))
        finally:
            landmarkers.close()
        if not features:
            raise ValueError("At least one BGR frame is required")
        return self.predict_raw(np.stack(features), top_k=top_k)

    def open_stream(self, config: StreamConfig | None = None, *, memory=None) -> VSLStreamSession:
        session = VSLStreamSession(self, config=config)
        session.memory = memory
        return session

    def build_memory(self, exemplars, *, strength: float = 0.5, **kwargs):
        """Bộ nhớ mẫu cá nhân (``memory.MemoryEnsemble``) từ list ``memory.Exemplar``; truyền vào
        ``predict_raw(..., ensemble=)`` hoặc ``open_stream(memory=)``."""
        from src.model.memory import MemoryEnsemble
        return MemoryEnsemble(self.ensemble, self.classes, list(exemplars), target_len=self.target_len,
                              strength=strength, **kwargs)


class VSLStreamSession:
    """Stateful prediction for one video stream; never share between clients."""

    def __init__(self, predictor: VSLPredictor, config: StreamConfig | None = None,
                 landmarkers=None) -> None:
        self.predictor = predictor
        self.config = config or StreamConfig()
        checkpoint_mirror = getattr(predictor, "checkpoint", {}).get("mirror_data")
        if checkpoint_mirror is not None and bool(checkpoint_mirror) != self.config.mirror_data:
            warnings.warn(
                f"StreamConfig.mirror_data={self.config.mirror_data} khác với "
                f"mirror_data={checkpoint_mirror} lúc thu thập dữ liệu huấn luyện; "
                "có thể khiến tay trái/phải bị đảo.",
                stacklevel=2,
            )
        self.landmarkers = landmarkers if landmarkers is not None else LandmarkerLoader(
            running_mode=VisionRunningMode.VIDEO
        )
        self.memory = None          # memory.MemoryEnsemble của người dùng, None = mô hình gốc
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
        self.last_processed: float | None = None
        self.last_state: dict | None = None

    def update_frame(self, frame_bgr: np.ndarray, *, now: float | None = None) -> dict:
        """Consume a BGR frame and return JSON-compatible prediction state."""
        if self._closed:
            raise RuntimeError("Stream session is closed")
        now = time.monotonic() if now is None else float(now)
        fps = self.config.target_fps
        # bỏ frame đến sớm hơn nhịp thu dữ liệu: không chạy MediaPipe, trả trạng thái cũ
        if fps and self.last_state is not None and now - self.last_processed < 1.0 / fps - 1e-3:
            return self.last_state
        self.last_processed = now
        frame = cv2.flip(frame_bgr, 1) if self.config.mirror_data else frame_bgr
        frame = frame_padding(frame)
        pose_result, hand_result, _ = self.landmarkers.detect(frame)
        feature = extract(pose_result, hand_result)

        self.last_frame = frame
        self.last_pose_result = pose_result
        self.last_hand_result = hand_result
        return self.update_features(feature, now=now)

    def update_features(self, feature: np.ndarray, *, now: float) -> dict:
        """Cập nhật cửa sổ bằng đặc trưng thô ``(FEATURE_DIM,)`` của một frame đã trích xuất.

        ``update_frame`` gọi hàm này sau MediaPipe; có thể gọi trực tiếp để phát lại
        đặc trưng đã lưu (đánh giá offline) mà không cần camera.
        """
        if self._closed:
            raise RuntimeError("Stream session is closed")
        now = float(now)
        hand_present = bool(active_presence(feature).any())   # tay đang buông (dưới bụng) không tính
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
            and duration >= self._min_ready_seconds()
        self.frame_count += 1
        if ready and self.frame_count % self.config.predict_every == 0:
            raw = np.stack([item[1] for item in self.history])
            self.prediction = self._predict(raw, duration)
            label = self.prediction["label"]
            self.recent_labels.append(label)
            self.stable_label = label if label and len(self.recent_labels) == self.config.stability_frames \
                and all(candidate == label for candidate in self.recent_labels) else None

        status = "idle" if self.last_hand_time is None else "collecting" if not ready \
            else "ready" if self.stable_label else "predicting"
        self.last_state = {
            "status": status,
            "prediction": self.prediction,
            "stable_label": self.stable_label,
            "hand_present": hand_present,
            "frames": len(self.history),
            "duration": float(duration),
        }
        return self.last_state

    def _min_ready_seconds(self) -> float:
        """Độ dài cửa sổ tối thiểu để bắt đầu dự đoán (lớp con có thể dự đoán sớm hơn)."""
        return self.config.min_window_seconds

    def _predict(self, raw: np.ndarray, duration: float) -> dict:
        """Dự đoán cửa sổ thô hiện tại ``(frames, FEATURE_DIM)``."""
        if self.memory is None:
            return self.predictor.predict_raw(raw, top_k=self.config.top_k)
        return self.predictor.predict_raw(raw, top_k=self.config.top_k, ensemble=self.memory)

    def close(self) -> None:
        if not self._closed:
            self.landmarkers.close()
            self._closed = True

    def __enter__(self) -> VSLStreamSession:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
