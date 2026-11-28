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

  Thay cho ``frames`` có thể gửi ``landmarks`` (landmark MediaPipe client đã chạy trên frame pad 1280x1280,
  định dạng ở ``src/features/landmarks.py``; ``mirror_data`` khi đó bị bỏ qua vì client tự lật ảnh) hoặc
  ``features``: mảng ``(frames, FEATURE_DIM)`` feature thô đã trích.

  Trả về dict của ``VSLPredictor.predict_raw`` (``label``, ``confidence``, ``probability_gap``,
  ``accepted``, ``rejection_reason``, ``top``) cộng ``frames``, ``hand_frames`` (số frame có tay
  đang ký, không tính tay buông — 0 nghĩa là người dùng đang nghỉ) và ``elapsed_ms``.
  Lỗi đầu vào trả 400 ``{"error": ...}``.

Luồng liên tục (nhanh hơn nhiều): mỗi client mở một phiên giữ landmarker MediaPipe và cửa sổ feature
``expected_frames`` frame trên server, mỗi request chỉ gửi các frame MỚI nên chỉ phải trích chừng đó frame:

- ``POST   /api/stream``        → ``{"session_id", "window", "binary": true, "landmarks": true}`` (201). 503 khi
  đã đủ số phiên. ``binary`` báo server nhận multipart và cờ ``predict`` bên dưới; ``landmarks`` báo server nhận
  landmark thay cho ảnh.
- ``POST   /api/stream/<id>``   → JSON body::

      {
        "frames": ["data:image/jpeg;base64,...", ...],  # 1–60 frame mới, theo thứ tự thời gian
        "timestamps": [1234.5, ...],  # tuỳ chọn; ms lúc thu từng frame (tăng dần), mặc định theo fps
        "reset": false,               # tuỳ chọn; xoá cửa sổ trước khi thêm (vd. sau khi chốt một từ)
        "predict": true,              # tuỳ chọn; false = chỉ trích feature các frame mới, không chạy model
        "window": 30,                 # tuỳ chọn; dự đoán trên N nhịp (frame ở ``fps``) gần nhất (1–cửa sổ phiên)
        "min_frames": 30,             # tuỳ chọn; dự đoán khi đã có >= N nhịp (1–window), mặc định = window
        "fill": "interp",             # tuỳ chọn; cửa sổ ít hơn 30 frame: "interp" (nội suy) | "repeat" (lặp frame)
        "fps": 15, "mirror_data": false, "top_k": 3
      }

  Hoặc ``multipart/form-data`` (nhanh hơn: ảnh nhị phân, không base64, trình duyệt không phải gửi
  preflight CORS): mỗi frame là một phần ``frames`` (JPEG/PNG, đúng thứ tự), các tuỳ chọn còn lại nằm
  trong phần ``meta`` dạng chuỗi JSON như trên (không có ``frames``).

  Hoặc JSON có ``landmarks`` thay cho ``frames`` (1–60 frame landmark, như ``/api/predict``): server không chạy
  MediaPipe (phiên chỉ tạo landmarker khi nhận ảnh) nên mỗi request chỉ còn ``extract`` + model.

  Client nhanh nhất gửi từng frame ngay khi thu (``predict: false``) để server trích feature dần, và chỉ
  bật ``predict`` mỗi vài frame: request dự đoán khi đó chỉ còn phải trích 1 frame rồi chạy model.

  ``window``/``min_frames`` nhỏ hơn cho kết quả sớm hơn (model thấy ít ký hiệu hơn nên kém chính xác hơn);
  ``predict_raw`` tự nội suy số frame đó về 30. ``fill: "repeat"`` thay vào đó chép lặp từng frame thật cho
  đủ 30 (``repeat_frames``): cửa sổ 1–3 frame (siêu tốc x20–x40 trên web) thành một tư thế giữ yên, vận tốc ≈ 0.

  Cửa sổ tính theo THỜI GIAN (timestamp của client, thiếu thì suy từ ``fps``): ``window`` = 30 ở ``fps`` 15 là
  2 giây gần nhất, dù client thu 7 fps (điện thoại, ~15 frame) hay 30 fps (~60 frame). Cửa sổ theo số frame
  từng làm máy chậm đưa cho model 4 giây cử chỉ (lẫn lúc nghỉ / ký hiệu trước) và ngưỡng ``hand_frames``
  không bao giờ đạt.
  Trả ``ready`` (đã đủ ``min_frames``), ``predicted``, ``buffered`` (thời lượng đang giữ, đơn vị nhịp ``fps``),
  ``window``, ``frames`` (số frame thật đem dự đoán), ``hand_frames`` (thời lượng có tay trong cửa sổ đó, đơn
  vị nhịp ``fps``), ``capture_fps`` (nhịp thu thật của client), ``elapsed_ms``; khi ``predicted`` (``ready`` và ``predict``) thêm kết quả dự
  đoán như ``/api/predict``.
  Phiên không tồn tại/hết hạn → 404.
- ``DELETE /api/stream/<id>``   → đóng phiên (204). Phiên không dùng quá ``stream_idle_s`` tự đóng.

Cá nhân hoá (bộ nhớ mẫu, ``src/model/memory.py``): mẫu được lưu ở client, server chỉ giữ trong phiên.

- ``POST /api/memory/sample`` → JSON ``{"label": "toi", "landmarks": [...]}`` (hoặc ``features``) của một lần
  ký mẫu. Trả ``{"ok", "reason", "warning", "features", "frames", "hand_frames", "prediction"}``: ``features``
  là feature thô float32 little-endian mã hoá base64 để client lưu; ``ok: false`` (``reason: "few_hands"``)
  khi tay không xuất hiện đủ lâu; ``warning: "model_disagrees"`` khi mô hình gốc rất chắc đó là từ khác.
- ``POST /api/stream`` nhận thêm body tuỳ chọn ``{"memory": {"strength": 0.5, "exemplars": [{"label",
  "features", "frames"}, ...]}}``: mọi dự đoán của phiên dùng bộ nhớ đó, kết quả có thêm ``memory``
  (lớp có mẫu giống nhất và độ giống). Phản hồi mở phiên có ``memory`` (số mẫu theo lớp, lớp bị bỏ qua).
- ``POST /api/stream/<id>/memory`` → body ``{"memory": {...}}`` như trên (``null`` hoặc không có mẫu = tắt):
  thay bộ mẫu của phiên đang mở mà KHÔNG xoá cửa sổ frame (dùng khi tự học theo thời gian thực).
  Trả ``{"memory": tóm tắt | null}``; phiên không tồn tại → 404.

Chạy::

    .venv/bin/python -m src.app.server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import secrets
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.features.extract import FEATURE_DIM, active_presence, extract
from src.features.frame import frame_padding
from src.features.landmarks import dict_to_results
from src.features.schema import EXPECTED_FRAMES
from src.model.memory import MAX_EXEMPLARS, MAX_FRAMES, Exemplar, check_exemplar_features
from src.model.predict import VSLPredictor

DEFAULT_FPS = 15.0
MAX_TOP_K = 10
MAX_STREAM_FRAMES = 60
MIN_SAMPLE_HAND_FRAMES = 8
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
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadRequest(f"Frame không phải base64 hợp lệ: {exc}") from None
    return decode_image(raw)


def decode_image(raw: bytes) -> np.ndarray:
    """Giải mã bytes JPEG/PNG thành frame BGR."""
    buffer = np.frombuffer(raw, dtype=np.uint8)
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


def landmarks_to_features(frames) -> np.ndarray:
    """Landmark JSON của từng frame (MediaPipe chạy ở client) → feature thô ``(len(frames), FEATURE_DIM)``."""
    try:
        return np.stack([extract(*dict_to_results(frame)) for frame in frames])
    except ValueError as exc:
        raise BadRequest(f"Landmark không hợp lệ: {exc}") from None


class StreamSession:
    """Phiên của một client: extractor giữ tracking MediaPipe + cửa sổ feature trượt.

    Extractor (landmarker MediaPipe, nặng) chỉ được tạo khi phiên nhận ảnh; phiên gửi landmark không cần nó.
    """

    def __init__(self, factory, window: int) -> None:
        self._factory = factory
        self._extractor = None
        self.window = window
        self.features: deque[np.ndarray] = deque(maxlen=4 * window)
        self.times: deque[int] = deque(maxlen=4 * window)
        self.last_timestamp = -1
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        self.memory = None

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = self._factory()
        return self._extractor

    def close(self) -> None:
        if self._extractor is not None:
            _close_later(self._extractor)
            self._extractor = None

    def next_timestamps(self, timestamps, count: int, fps: float) -> list[int]:
        """Timestamp ms tăng ngặt cho ``count`` frame mới (VIDEO mode bắt buộc)."""
        if timestamps is None:
            step = 1000 / fps
            start = 0.0 if self.last_timestamp < 0 else self.last_timestamp + step
            raw = [start + step * i for i in range(count)]
        else:
            raw = timestamps
        result = []
        for t in raw:
            self.last_timestamp = max(self.last_timestamp + 1, round(t))
            result.append(self.last_timestamp)
        return result

    def clear(self) -> None:
        self.features.clear()
        self.times.clear()

    def _interval(self, step: float) -> float:
        """Khoảng cách thật giữa hai frame của client (trung vị), mặc định ``step`` khi chưa đủ frame."""
        return float(np.median(np.diff(self.times))) if len(self.times) > 1 else step

    def append(self, features, times, fps: float) -> None:
        """Thêm frame mới rồi bỏ frame đã ra ngoài cửa sổ ``window`` frame ở nhịp ``fps``."""
        self.features.extend(features)
        self.times.extend(times)
        step = 1000 / fps
        limit = self.window * step - self._interval(step) / 2
        while len(self.times) > 1 and self.times[-1] - self.times[0] > limit:
            self.times.popleft()
            self.features.popleft()

    def recent(self, window: int, fps: float) -> tuple[np.ndarray, int, int]:
        """Feature của ``window`` frame (quy về nhịp ``fps``) gần nhất theo thời gian.

        Trả ``(features, covered, buffered)``: ``covered``/``buffered`` là thời lượng của cửa sổ này / của cả
        bộ đệm, đổi ra số frame ở nhịp ``fps`` (thời lượng = khoảng từ frame đầu tới frame cuối + một khoảng
        thu của client).
        """
        step = 1000 / fps
        interval = self._interval(step)
        times = np.asarray(self.times, dtype=np.float64)
        first = int(np.searchsorted(times, times[-1] - (window * step - interval / 2)))
        frames = lambda span: int((span + interval) / step + 0.5)
        covered = min(window, frames(times[-1] - times[first]))
        buffered = min(self.window, frames(times[-1] - times[0]))
        return np.stack(list(self.features)[first:]), covered, buffered


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
                self._sessions.pop(sid).close()

    def open(self, memory=None) -> str | None:
        with self._lock:
            self._sweep()
            if len(self._sessions) >= self.max_sessions:
                return None
            sid = secrets.token_urlsafe(12)
            self._sessions[sid] = StreamSession(self.factory, self.window)
            self._sessions[sid].memory = memory
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
            session.close()
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


def encode_features(features: np.ndarray) -> str:
    """Feature thô -> base64 của float32 little-endian (gọn hơn JSON số ~3 lần, giữ nguyên NaN)."""
    return base64.b64encode(np.ascontiguousarray(features, dtype="<f4").tobytes()).decode("ascii")


def decode_features(data, frames) -> np.ndarray:
    """Ngược với ``encode_features``: ``(frames, FEATURE_DIM)``; dữ liệu sai -> BadRequest."""
    if not isinstance(data, str) or isinstance(frames, bool) or not isinstance(frames, int):
        raise BadRequest("Mỗi mẫu cần 'features' (chuỗi base64) và 'frames' (số nguyên)")
    if not 1 <= frames <= MAX_FRAMES:
        raise BadRequest(f"'frames' của mẫu phải trong [1, {MAX_FRAMES}]")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise BadRequest("'features' của mẫu không phải base64 hợp lệ") from None
    if len(raw) != frames * FEATURE_DIM * 4:
        raise BadRequest(f"'features' của mẫu phải có {frames} x {FEATURE_DIM} số float32")
    try:
        return check_exemplar_features(np.frombuffer(raw, dtype="<f4").reshape(frames, FEATURE_DIM))
    except ValueError as exc:
        raise BadRequest(str(exc)) from None


def parse_memory(body) -> tuple[list[Exemplar], float] | None:
    """``{"strength", "exemplars": [{"label", "features", "frames"}]}`` -> (mẫu, strength); không có -> None."""
    if body is None:
        return None
    if not isinstance(body, dict):
        raise BadRequest("'memory' phải là JSON object")
    strength = body.get("strength", 0.5)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
        raise BadRequest("'memory.strength' phải là số trong [0, 1]")
    items = body.get("exemplars", [])
    if not isinstance(items, list) or len(items) > MAX_EXEMPLARS:
        raise BadRequest(f"'memory.exemplars' phải là mảng tối đa {MAX_EXEMPLARS} mẫu")
    exemplars = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise BadRequest("Mỗi mẫu cần 'label' là chuỗi")
        exemplars.append(Exemplar(item["label"], decode_features(item.get("features"), item.get("frames"))))
    return exemplars, float(strength)


FILL_MODES = ("interp", "repeat")


def repeat_frames(features: np.ndarray, target_len: int) -> np.ndarray:
    """Chép lặp đều từng frame cho đủ ``target_len`` (giữ thứ tự, không tạo giá trị mới như nội suy).

    Chỉ dùng khi cửa sổ ngắn hơn ``target_len``; dài hơn thì giữ nguyên để ``process_sequence`` co lại.
    """
    n = len(features)
    if n == 0 or n >= target_len:
        return features
    return features[(np.arange(target_len) * n) // target_len]


def _int_option(body: dict, key: str, default: int, low: int, high: int) -> int:
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise BadRequest(f"{key} phải là số nguyên trong [{low}, {high}]")
    return value


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
    app.json.ensure_ascii = False
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    model_lock = threading.Lock()
    streams = StreamStore(stream_extractor_factory, window=expected_frames or EXPECTED_FRAMES,
                          max_sessions=max_streams, idle_s=stream_idle_s)

    def run_model(features: np.ndarray, top_k: int, memory=None) -> dict:
        try:
            with model_lock, torch.inference_mode():
                return predictor.predict_raw(features, top_k=top_k, ensemble=memory)
        except ValueError as exc:
            raise BadRequest(str(exc)) from None

    def build_memory(parsed):
        """Kết quả ``parse_memory`` -> ``MemoryEnsemble`` (None khi không có mẫu)."""
        if parsed is None or not parsed[0]:
            return None
        try:
            with model_lock, torch.inference_mode():
                return predictor.build_memory(parsed[0], strength=parsed[1])
        except ValueError as exc:
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

    def stream_body() -> tuple[dict, list[np.ndarray] | None, np.ndarray | None]:
        """Body của /api/stream/<id>: ``(body, ảnh, None)`` cho JSON (frame base64) hoặc multipart (frame nhị
        phân + ``meta`` JSON); ``(body, None, feature thô)`` cho JSON ``landmarks``."""
        if request.mimetype == "multipart/form-data":
            try:
                body = json.loads(request.form.get("meta") or "{}")
            except ValueError:
                raise BadRequest("'meta' phải là chuỗi JSON") from None
            if not isinstance(body, dict):
                raise BadRequest("'meta' phải là JSON object")
            files = request.files.getlist("frames")
            if not 1 <= len(files) <= MAX_STREAM_FRAMES:
                raise BadRequest(f"'frames' phải là mảng 1–{MAX_STREAM_FRAMES} ảnh")
            return body, [decode_image(f.read()) for f in files], None
        body = json_body()
        if "landmarks" in body:
            landmarks = body["landmarks"]
            if not isinstance(landmarks, list) or not 1 <= len(landmarks) <= MAX_STREAM_FRAMES:
                raise BadRequest(f"'landmarks' phải là mảng 1–{MAX_STREAM_FRAMES} frame")
            return body, None, landmarks_to_features(landmarks)
        frames = body.get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_STREAM_FRAMES:
            raise BadRequest(f"'frames' phải là mảng 1–{MAX_STREAM_FRAMES} ảnh base64")
        return body, [decode_frame(f) for f in frames], None

    @app.after_request
    def allow_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Max-Age"] = "86400"
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
        elif "landmarks" in body:
            landmarks = body["landmarks"]
            if not isinstance(landmarks, list):
                raise BadRequest("'landmarks' phải là mảng frame landmark")
            check_count(len(landmarks), "landmarks")
            features = landmarks_to_features(landmarks)
        elif "features" in body:
            try:
                features = np.asarray(body["features"], dtype=np.float32)
            except (TypeError, ValueError):
                raise BadRequest("'features' phải là mảng số 2 chiều") from None
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                raise BadRequest(f"'features' phải có shape (frames, {FEATURE_DIM}), nhận {features.shape}")
            check_count(features.shape[0], "features")
        else:
            raise BadRequest("Thiếu 'frames' (hoặc 'landmarks' / 'features')")

        result = run_model(features, top_k)
        result.update(frames=int(features.shape[0]), hand_frames=count_hand_frames(features),
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return jsonify(result)

    @app.route("/api/stream", methods=["POST", "OPTIONS"])
    def stream_open():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) if request.content_length else None
        memory = build_memory(parse_memory(body.get("memory")) if isinstance(body, dict) else None)
        sid = streams.open(memory)
        if sid is None:
            return jsonify(error=f"Server đã đủ {streams.max_sessions} phiên, thử lại sau"), 503
        return jsonify(session_id=sid, window=streams.window, binary=True, landmarks=True,
                       memory=memory.summary() if memory is not None else None), 201

    @app.route("/api/stream/<sid>/memory", methods=["POST", "OPTIONS"])
    def stream_memory(sid: str):
        if request.method == "OPTIONS":
            return "", 204
        body = json_body()
        parsed = parse_memory(body.get("memory"))
        memory = build_memory(parsed)
        session = streams.get(sid)
        if session is None:
            return jsonify(error="Phiên không tồn tại hoặc đã hết hạn"), 404
        session.memory = memory
        return jsonify(memory=memory.summary() if memory is not None else None)

    @app.route("/api/memory/sample", methods=["POST", "OPTIONS"])
    def memory_sample():
        if request.method == "OPTIONS":
            return "", 204
        body = json_body()
        label = body.get("label")
        if not isinstance(label, str) or label not in predictor.classes:
            raise BadRequest("'label' phải là một lớp của checkpoint (xem /api/health)")
        if "landmarks" in body:
            landmarks = body["landmarks"]
            if not isinstance(landmarks, list) or not 1 <= len(landmarks) <= MAX_FRAMES:
                raise BadRequest(f"'landmarks' phải là mảng 1–{MAX_FRAMES} frame")
            features = landmarks_to_features(landmarks)
        else:
            features = decode_features(body.get("features"), body.get("frames"))
        try:
            features = check_exemplar_features(features)
        except ValueError as exc:
            raise BadRequest(str(exc)) from None
        hand_frames = count_hand_frames(features)
        ok = hand_frames >= max(MIN_SAMPLE_HAND_FRAMES, len(features) // 2)
        result = run_model(features, 3) if ok else None
        warning = None
        if result is not None and result["top"][0]["label"] != label and result["top"][0]["confidence"] >= 0.9:
            warning = "model_disagrees"
        return jsonify(ok=ok, reason=None if ok else "few_hands", warning=warning, label=label,
                       features=encode_features(features), frames=int(len(features)), hand_frames=hand_frames,
                       prediction=result)

    @app.route("/api/stream/<sid>", methods=["POST", "DELETE", "OPTIONS"])
    def stream_frames(sid: str):
        if request.method == "OPTIONS":
            return "", 204
        if request.method == "DELETE":
            streams.close(sid)
            return "", 204
        started = time.perf_counter()
        body, decoded, given = stream_body()
        count = len(decoded) if decoded is not None else len(given)
        fps, mirror_data, top_k = _parse_options(body, default_mirror_data)
        timestamps = body.get("timestamps")
        if timestamps is not None and (
            not isinstance(timestamps, list) or len(timestamps) != count
            or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and np.isfinite(t) for t in timestamps)
        ):
            raise BadRequest("'timestamps' phải là mảng số, cùng độ dài với 'frames'/'landmarks'")
        reset, predict_now = body.get("reset", False), body.get("predict", True)
        if not isinstance(reset, bool) or not isinstance(predict_now, bool):
            raise BadRequest("reset/predict phải là true/false")
        window_size = _int_option(body, "window", streams.window, 1, streams.window)
        min_frames = _int_option(body, "min_frames", window_size, 1, window_size)
        fill = body.get("fill", "interp")
        if fill not in FILL_MODES:
            raise BadRequest(f"fill phải là một trong {list(FILL_MODES)}")

        session = streams.get(sid)
        if session is None:
            return jsonify(error="Phiên không tồn tại hoặc đã hết hạn"), 404
        with session.lock:
            if reset:
                session.clear()
            times = session.next_timestamps(timestamps, count, fps)
            if given is None:
                given = [session.extractor.extract(frame, timestamp_ms=ts, mirror_data=mirror_data)
                         for frame, ts in zip(decoded, times)]
            session.append(given, times, fps)
            session.touched = time.monotonic()
            recent, covered, buffered = session.recent(window_size, fps)

        ready = buffered >= min_frames
        predicted = ready and predict_now
        model_input = repeat_frames(recent, expected_frames) if fill == "repeat" else recent
        result = run_model(model_input, top_k, session.memory) if predicted else {}
        hand_frames = round(count_hand_frames(recent) * covered / len(recent))
        capture_fps = round(1000 / session._interval(1000 / fps), 1)
        result.update(ready=ready, predicted=predicted, buffered=buffered, window=window_size,
                      frames=int(recent.shape[0]), hand_frames=hand_frames, capture_fps=capture_fps,
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return jsonify(result)

    return app


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API server dự đoán VSL từ 30 frame")
    parser.add_argument("--checkpoint", type=Path, default=None, help="mặc định: runs/best/ensemble.pt (run mới nhất)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frames", type=int, default=EXPECTED_FRAMES, help="số frame mỗi request; 0 = không giới hạn")
    parser.add_argument("--mirror-data", action="store_true", help="mặc định cho request không gửi mirror_data")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-gap", type=float, default=0.08)
    parser.add_argument("--max-streams", type=int, default=8, help="số phiên /api/stream mở cùng lúc")
    parser.add_argument("--device", default="cpu", help="thiết bị chạy model (cpu, mps, cuda); mặc định cpu")
    parser.add_argument("--torch-threads", type=int, default=1, help="số luồng torch cho model; mặc định 1")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    predictor = VSLPredictor(args.checkpoint, device=args.device, min_confidence=args.min_confidence,
                             min_probability_gap=args.min_gap)
    print(f"Model: {predictor.checkpoint_path} ({predictor.device}, {torch.get_num_threads()} luồng) | "
          f"{len(predictor.classes)} lớp: {', '.join(predictor.classes)}")
    app = create_app(predictor, expected_frames=args.frames or None, default_mirror_data=args.mirror_data,
                     max_streams=args.max_streams)
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
