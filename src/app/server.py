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
  ``accepted``, ``rejection_reason``, ``top``) cộng ``frames`` và ``elapsed_ms``.
  Lỗi đầu vào trả 400 ``{"error": ...}``.

Chạy::

    .venv/bin/python -m src.app.server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import binascii
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.features.extract import FEATURE_DIM, extract
from src.features.frame import frame_padding
from src.features.schema import EXPECTED_FRAMES
from src.model.predict import VSLPredictor

DEFAULT_FPS = 15.0
MAX_TOP_K = 10
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


def extract_features(frames_bgr: list[np.ndarray], *, fps: float, mirror_data: bool) -> np.ndarray:
    """Trích feature thô ``(len(frames), FEATURE_DIM)``; landmarker riêng cho mỗi request (giống predict_frames)."""
    landmarkers = LandmarkerLoader(running_mode=VisionRunningMode.VIDEO)
    try:
        features = []
        for i, frame in enumerate(frames_bgr):
            if mirror_data:
                frame = cv2.flip(frame, 1)
            pose_result, hand_result, _ = landmarkers.detect(frame_padding(frame), timestamp_ms=round(i * 1000 / fps))
            features.append(extract(pose_result, hand_result))
    finally:
        landmarkers.close()
    return np.stack(features)


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
               default_mirror_data: bool = False, feature_extractor=extract_features) -> Flask:
    """Tạo Flask app dùng chung một ``predictor`` cho mọi request.

    expected_frames: số frame mỗi request phải gửi (mặc định 30); ``None`` nhận mọi số frame > 0.
    feature_extractor: ``(frames, fps=, mirror_data=) -> (N, FEATURE_DIM)``; thay được khi test.
    """
    app = Flask(__name__, static_folder=None)
    app.json.ensure_ascii = False          # trả nhãn tiếng Việt nguyên dạng
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    model_lock = threading.Lock()          # torch model dùng chung; MediaPipe chạy song song theo request

    def check_count(n: int, name: str) -> None:
        if n == 0:
            raise BadRequest(f"'{name}' không được rỗng")
        if expected_frames is not None and n != expected_frames:
            raise BadRequest(f"Cần đúng {expected_frames} frame, nhận {n}")

    @app.after_request
    def allow_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
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
                       expected_frames=expected_frames, feature_dim=FEATURE_DIM)

    @app.route("/api/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return "", 204
        started = time.perf_counter()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise BadRequest("Body phải là JSON object")
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

        try:
            with model_lock, torch.inference_mode():
                result = predictor.predict_raw(features, top_k=top_k)
        except ValueError as exc:          # vd. feature chứa vô cực
            raise BadRequest(str(exc)) from None
        result.update(frames=int(features.shape[0]), elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
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
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    predictor = VSLPredictor(args.checkpoint, min_confidence=args.min_confidence, min_probability_gap=args.min_gap)
    print(f"Model: {predictor.checkpoint_path} | {len(predictor.classes)} lớp: {', '.join(predictor.classes)}")
    app = create_app(predictor, expected_frames=args.frames or None, default_mirror_data=args.mirror_data)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
