"""HTTP API dự đoán VSL: frontend gửi 30 frame của một clip, server trả kết quả của mô hình.

Endpoint:

- ``GET  /``            → trang HTML test API (``static/test_client.html``: webcam hoặc file video).
- ``GET  /api/health``  → trạng thái server, checkpoint và danh sách lớp.
- ``POST /api/predict`` → JSON body::

      {
        "frames": ["data:image/jpeg;base64,...", ...],   # đúng 30 ảnh (JPEG/PNG, base64, có/không tiền tố data URL)
        "fps": 15,             # tuỳ chọn; nhịp thu frame, mặc định 15 (giống lúc thu dữ liệu)
        "mirror_data": false,  # tuỳ chọn; phải giống MIRROR_DATA lúc thu dữ liệu
        "top_k": 3             # tuỳ chọn
      }

  Thay cho ``frames`` có thể gửi ``features``: mảng ``(frames, FEATURE_DIM)`` feature thô đã trích.

  Trả về dict của ``VSLPredictor.predict_raw`` (``label``, ``confidence``, ``probability_gap``,
  ``accepted``, ``rejection_reason``, ``top``) cộng ``frames``, ``hand_frames`` (số frame có tay
  đang ký, không tính tay buông — 0 nghĩa là người dùng đang nghỉ) và ``elapsed_ms``.
  Lỗi đầu vào trả 400 ``{"error": ...}``.

Luồng liên tục (nhanh hơn nhiều): mỗi client mở một phiên giữ landmarker MediaPipe và cửa sổ feature
``expected_frames`` frame trên server, mỗi request chỉ gửi các frame MỚI nên chỉ phải trích chừng đó frame:

- ``POST   /api/stream``        → ``{"session_id", "window"}`` (201). 503 khi đã đủ số phiên.
- ``POST   /api/stream/<id>``   → JSON body::

      {
        "frames": ["data:image/jpeg;base64,...", ...],  # 1–60 frame mới, theo thứ tự thời gian
        "timestamps": [1234.5, ...],  # tuỳ chọn; ms lúc thu từng frame (tăng dần), mặc định theo fps
        "reset": false,               # tuỳ chọn; xoá cửa sổ trước khi thêm (vd. sau khi chốt một từ)
        "fps": 15, "mirror_data": false, "top_k": 3
      }

  Trả ``ready`` (cửa sổ đã đủ frame), ``buffered``, ``window``, ``hand_frames`` (trong cửa sổ),
  ``elapsed_ms``; khi ``ready`` thêm kết quả dự đoán như ``/api/predict``. Phiên không tồn tại/hết hạn → 404.
- ``DELETE /api/stream/<id>``   → đóng phiên (204). Phiên không dùng quá ``stream_idle_s`` tự đóng.

Chạy::

    .venv/bin/python -m src.app.server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import binascii
import secrets
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.features.extract import FEATURE_DIM, active_presence, extract
from src.features.frame import frame_padding
from src.features.schema import EXPECTED_FRAMES
from src.model.predict import VSLPredictor

DEFAULT_FPS = 15.0
MAX_TOP_K = 10
MAX_STREAM_FRAMES = 60
STATIC_DIR = Path(__file__).resolve().parent / "static"


class BadRequest(ValueError):
    """Lỗi do dữ liệu client gửi lên (trả HTTP 400)."""


def decode_frame(data: str) -> np.ndarray:
    """Giải mã một ảnh base64 (JPEG/PNG, cho phép tiền tố ``data:image/...;base64,``) thành frame BGR."""
    if not isinstance(data, str) or not data:
        raise BadRequest("Mỗi frame phải là chuỗi base64 không rỗng")
    if data.startswith("data:"):
        data = data.partition(",")[2]
    try:
        buffer = np.frombuffer(base64.b64decode(data, validate=True), dtype=np.uint8)
    except (binascii.Error, ValueError) as exc:
        raise BadRequest(f"Frame không phải base64 hợp lệ: {exc}") from None
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if frame is None:
        raise BadRequest("Không giải mã được ảnh (cần JPEG hoặc PNG)")
    return frame


def _close_later(closable) -> None:
    """Đóng ở luồng nền: ``LandmarkerLoader.close`` mất ~300 ms (có lúc rất lâu khi MediaPipe gửi telemetry)."""
    threading.Thread(target=closable.close, name="landmarker-close", daemon=True).start()


class FrameFeatureExtractor:
    """Một landmarker VIDEO (pose ‖ hand song song) → feature thô ``(FEATURE_DIM,)`` cho từng frame BGR.

    Giữ trạng thái tracking giữa các frame nên mỗi luồng video/phiên cần một đối tượng riêng.
    """

    def __init__(self) -> None:
        self._landmarkers = LandmarkerLoader(running_mode=VisionRunningMode.VIDEO, parallel=True)

    def extract(self, frame_bgr: np.ndarray, *, timestamp_ms: int, mirror_data: bool) -> np.ndarray:
        if mirror_data:
            frame_bgr = cv2.flip(frame_bgr, 1)
        pose_result, hand_result, _ = self._landmarkers.detect(frame_padding(frame_bgr), timestamp_ms=timestamp_ms)
        return extract(pose_result, hand_result)

    def close(self) -> None:
        self._landmarkers.close()


def extract_features(frames_bgr: list[np.ndarray], *, fps: float, mirror_data: bool) -> np.ndarray:
    """Trích feature thô ``(len(frames), FEATURE_DIM)``; landmarker riêng cho mỗi request (giống predict_frames)."""
    extractor = FrameFeatureExtractor()
    try:
        return np.stack([extractor.extract(frame, timestamp_ms=round(i * 1000 / fps), mirror_data=mirror_data)
                         for i, frame in enumerate(frames_bgr)])
    finally:
        _close_later(extractor)


class StreamSession:
    """Phiên của một client: extractor giữ tracking MediaPipe + cửa sổ feature trượt."""

    def __init__(self, extractor, window: int) -> None:
        self.extractor = extractor
        self.features: deque[np.ndarray] = deque(maxlen=window)
        self.last_timestamp = -1
        self.lock = threading.Lock()          # mỗi phiên xử lý tuần tự (tracking phụ thuộc thứ tự frame)
        self.touched = time.monotonic()

    def next_timestamps(self, timestamps, count: int, fps: float) -> list[int]:
        """Timestamp ms tăng ngặt cho ``count`` frame mới (VIDEO mode bắt buộc)."""
        if timestamps is None:
            step = 1000 / fps                     # frame đầu tiên ở 0 ms, giống reextract (i * 1000 / fps)
            start = 0.0 if self.last_timestamp < 0 else self.last_timestamp + step
            raw = [start + step * i for i in range(count)]
        else:
            raw = timestamps
        result = []
        for t in raw:
            self.last_timestamp = max(self.last_timestamp + 1, round(t))
            result.append(self.last_timestamp)
        return result


class StreamStore:
    """Các phiên đang mở; phiên không dùng quá ``idle_s`` giây bị đóng khi có request kế tiếp."""

    def __init__(self, factory, *, window: int, max_sessions: int, idle_s: float) -> None:
        self.factory, self.window, self.max_sessions, self.idle_s = factory, window, max_sessions, idle_s
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._sessions)

    def _sweep(self) -> None:
        now = time.monotonic()
        for sid, session in list(self._sessions.items()):
            if now - session.touched > self.idle_s and not session.lock.locked():
                _close_later(self._sessions.pop(sid).extractor)

    def open(self) -> str | None:
        with self._lock:
            self._sweep()
            if len(self._sessions) >= self.max_sessions:
                return None
            sid = secrets.token_urlsafe(12)
            self._sessions[sid] = StreamSession(self.factory(), self.window)
            return sid

    def get(self, sid: str) -> StreamSession | None:
        with self._lock:
            self._sweep()
            session = self._sessions.get(sid)
            if session is not None:
                session.touched = time.monotonic()
            return session

    def close(self, sid: str) -> bool:
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is not None:
            _close_later(session.extractor)
        return session is not None


def _parse_options(body: dict, default_mirror_data: bool) -> tuple[float, bool, int]:
    fps = body.get("fps", DEFAULT_FPS)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps <= 240:
        raise BadRequest("fps phải là số trong (0, 240]")
    mirror_data = body.get("mirror_data", default_mirror_data)
    if not isinstance(mirror_data, bool):
        raise BadRequest("mirror_data phải là true/false")
    top_k = body.get("top_k", 3)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise BadRequest(f"top_k phải là số nguyên trong [1, {MAX_TOP_K}]")
    return float(fps), mirror_data, top_k


def create_app(predictor: VSLPredictor, *, expected_frames: int = EXPECTED_FRAMES,
               default_mirror_data: bool = False, feature_extractor=extract_features,
               stream_extractor_factory=FrameFeatureExtractor, max_streams: int = 8,
               stream_idle_s: float = 60.0) -> Flask:
    """Tạo Flask app dùng chung một ``predictor`` cho mọi request.

    expected_frames: số frame mỗi request phải gửi (mặc định 30); ``None`` nhận mọi số frame > 0
        (khi đó cửa sổ của luồng liên tục là ``EXPECTED_FRAMES``).
    feature_extractor: ``(frames, fps=, mirror_data=) -> (N, FEATURE_DIM)``; thay được khi test.
    stream_extractor_factory: ``() -> obj`` có ``extract(frame, timestamp_ms=, mirror_data=)`` và ``close()``.
    """
    app = Flask(__name__, static_folder=None)
    app.json.ensure_ascii = False          # trả nhãn tiếng Việt nguyên dạng
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    model_lock = threading.Lock()          # torch model dùng chung; MediaPipe chạy song song theo request
    streams = StreamStore(stream_extractor_factory, window=expected_frames or EXPECTED_FRAMES,
                          max_sessions=max_streams, idle_s=stream_idle_s)

    def run_model(features: np.ndarray, top_k: int) -> dict:
        try:
            with model_lock, torch.inference_mode():
                return predictor.predict_raw(features, top_k=top_k)
        except ValueError as exc:          # vd. feature chứa vô cực
            raise BadRequest(str(exc)) from None

    def count_hand_frames(features) -> int:
        return int(sum(bool(active_presence(frame).any()) for frame in features))

    def json_body() -> dict:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise BadRequest("Body phải là JSON object")
        return body

    def check_count(n: int, name: str) -> None:
        if n == 0:
            raise BadRequest(f"'{name}' không được rỗng")
        if expected_frames is not None and n != expected_frames:
            raise BadRequest(f"Cần đúng {expected_frames} frame, nhận {n}")

    @app.after_request
    def allow_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        return response

    @app.errorhandler(BadRequest)
    def bad_request(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify(error="Request quá lớn (tối đa 64 MB)"), 413

    @app.get("/")
    def test_client():
        return send_from_directory(STATIC_DIR, "test_client.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", checkpoint=str(predictor.checkpoint_path), classes=list(predictor.classes),
                       expected_frames=expected_frames, feature_dim=FEATURE_DIM,
                       stream={"window": streams.window, "active": len(streams), "max": streams.max_sessions})

    @app.route("/api/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return "", 204
        started = time.perf_counter()
        body = json_body()
        fps, mirror_data, top_k = _parse_options(body, default_mirror_data)

        if "frames" in body:
            frames = body["frames"]
            if not isinstance(frames, list):
                raise BadRequest("'frames' phải là mảng ảnh base64")
            check_count(len(frames), "frames")
            features = feature_extractor([decode_frame(f) for f in frames], fps=fps, mirror_data=mirror_data)
        elif "features" in body:
            try:
                features = np.asarray(body["features"], dtype=np.float32)
            except (TypeError, ValueError):
                raise BadRequest("'features' phải là mảng số 2 chiều") from None
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                raise BadRequest(f"'features' phải có shape (frames, {FEATURE_DIM}), nhận {features.shape}")
            check_count(features.shape[0], "features")
        else:
            raise BadRequest("Thiếu 'frames' (hoặc 'features')")

        result = run_model(features, top_k)
        result.update(frames=int(features.shape[0]), hand_frames=count_hand_frames(features),
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return jsonify(result)

    @app.route("/api/stream", methods=["POST", "OPTIONS"])
    def stream_open():
        if request.method == "OPTIONS":
            return "", 204
        sid = streams.open()
        if sid is None:
            return jsonify(error=f"Server đã đủ {streams.max_sessions} phiên, thử lại sau"), 503
        return jsonify(session_id=sid, window=streams.window), 201

    @app.route("/api/stream/<sid>", methods=["POST", "DELETE", "OPTIONS"])
    def stream_frames(sid: str):
        if request.method == "OPTIONS":
            return "", 204
        if request.method == "DELETE":
            streams.close(sid)
            return "", 204
        started = time.perf_counter()
        body = json_body()
        fps, mirror_data, top_k = _parse_options(body, default_mirror_data)
        frames = body.get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_STREAM_FRAMES:
            raise BadRequest(f"'frames' phải là mảng 1–{MAX_STREAM_FRAMES} ảnh base64")
        timestamps = body.get("timestamps")
        if timestamps is not None and (
            not isinstance(timestamps, list) or len(timestamps) != len(frames)
            or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and np.isfinite(t) for t in timestamps)
        ):
            raise BadRequest("'timestamps' phải là mảng số, cùng độ dài với 'frames'")
        reset = body.get("reset", False)
        if not isinstance(reset, bool):
            raise BadRequest("reset phải là true/false")
        decoded = [decode_frame(f) for f in frames]       # lỗi ảnh → 400 trước khi đụng vào phiên

        session = streams.get(sid)
        if session is None:
            return jsonify(error="Phiên không tồn tại hoặc đã hết hạn"), 404
        with session.lock:
            if reset:
                session.features.clear()
            for frame, ts in zip(decoded, session.next_timestamps(timestamps, len(decoded), fps)):
                session.features.append(session.extractor.extract(frame, timestamp_ms=ts, mirror_data=mirror_data))
            session.touched = time.monotonic()
            window = np.stack(session.features)

        ready = window.shape[0] == streams.window
        result = run_model(window, top_k) if ready else {}
        result.update(ready=ready, buffered=int(window.shape[0]), window=streams.window,
                      frames=int(window.shape[0]), hand_frames=count_hand_frames(window),
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return jsonify(result)

    return app


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API server dự đoán VSL từ 30 frame")
    parser.add_argument("--checkpoint", type=Path, default=None, help="mặc định: runs/*/ensemble.pt mới nhất")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frames", type=int, default=EXPECTED_FRAMES, help="số frame mỗi request; 0 = không giới hạn")
    parser.add_argument("--mirror-data", action="store_true", help="mặc định cho request không gửi mirror_data")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-gap", type=float, default=0.08)
    parser.add_argument("--max-streams", type=int, default=8, help="số phiên /api/stream mở cùng lúc")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    predictor = VSLPredictor(args.checkpoint, min_confidence=args.min_confidence, min_probability_gap=args.min_gap)
    print(f"Model: {predictor.checkpoint_path} | {len(predictor.classes)} lớp: {', '.join(predictor.classes)}")
    app = create_app(predictor, expected_frames=args.frames or None, default_mirror_data=args.mirror_data,
                     max_streams=args.max_streams)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
