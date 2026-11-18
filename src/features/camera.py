"""Mở webcam ở độ phân giải gốc lớn nhất thay vì mặc định của backend.

OpenCV không tự chọn độ phân giải tối đa: backend AVFoundation (macOS) mở session với
``AVCaptureSessionPresetMedium``, còn V4L2/DirectShow/MSMF thường mặc định 640x480. Vì vậy
mọi chỗ mở camera phải đi qua ``open_camera``.

Cách chọn: thử các độ phân giải ứng viên từ lớn xuống nhỏ (cùng tỉ lệ khung hình), đọc thử một
frame và giữ ứng viên đầu tiên mà camera trả về đúng kích thước. Trên macOS, ``set`` width/height
làm AVFoundation tự đổi sang một định dạng *gốc* của thiết bị (không phóng to giả), còn driver
khác thì giữ nguyên/kẹp kích thước khi không hỗ trợ — nên kích thước frame đọc được là thật.

Mặc định giữ tỉ lệ 16:9 vì dữ liệu ``data/raw`` được quay ở 16:9; đổi tỉ lệ (4:3, vuông) làm
thay đổi vùng nhìn so với lúc train.
"""

from __future__ import annotations

import sys

import cv2

# Ứng viên 16:9, lớn → nhỏ. Camera trả về đúng kích thước đầu tiên nó hỗ trợ.
RESOLUTIONS_16_9 = (
    (3840, 2160), (2560, 1440), (1920, 1080), (1600, 900),
    (1280, 720), (960, 540), (848, 480), (640, 360),
)

__all__ = ["RESOLUTIONS_16_9", "open_camera", "parse_resolution"]


def parse_resolution(text: str | None) -> tuple[int, int] | None:
    """``"1920x1080"`` → ``(1920, 1080)``; ``None``/``""``/``"max"`` → ``None`` (tự chọn lớn nhất)."""
    if text is None or text.strip().lower() in ("", "max", "auto"):
        return None
    try:
        width, height = (int(v) for v in text.lower().replace("*", "x").split("x"))
    except ValueError:
        raise ValueError(f"Độ phân giải phải có dạng RỘNGxCAO (vd. 1920x1080), nhận {text!r}") from None
    if width <= 0 or height <= 0:
        raise ValueError(f"Độ phân giải phải dương, nhận {text!r}")
    return width, height


def _read_size(cap: cv2.VideoCapture, attempts: int = 3) -> tuple[int, int] | None:
    """Kích thước (w, h) của frame thật đọc được — tin cậy hơn ``cap.get`` trên một số backend."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame.shape[1], frame.shape[0]
    return None


def _request(cap: cv2.VideoCapture, size: tuple[int, int]) -> tuple[int, int] | None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return _read_size(cap)


def open_camera(index: int = 0, resolution: tuple[int, int] | None = None, *,
                candidates=RESOLUTIONS_16_9, fps: float | None = None,
                api: int = cv2.CAP_ANY, verbose: bool = True) -> cv2.VideoCapture:
    """Mở camera ``index`` ở độ phân giải gốc lớn nhất (hoặc đúng ``resolution`` nếu truyền vào).

    Trả về ``cv2.VideoCapture`` đã mở; ném ``RuntimeError`` nếu không mở/đọc được camera.
    ``fps`` (tuỳ chọn) đặt sau khi chọn độ phân giải vì dải fps phụ thuộc định dạng.
    """
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không mở được camera {index}")
    try:
        default = _read_size(cap)
        if default is None:
            raise RuntimeError(f"Camera {index} không trả về frame nào")
        if cap.getBackendName() != "AVFOUNDATION":
            # V4L2/DirectShow/MSMF: YUYV ở độ phân giải cao chỉ được vài fps, MJPG thì đủ 30 fps.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        wanted = [tuple(resolution)] if resolution is not None else [tuple(c) for c in candidates]
        best = default
        chosen = None
        for size in wanted:
            actual = _request(cap, size)
            if actual is None:
                continue
            if actual == size:
                chosen = actual
                break
            if actual[0] * actual[1] > best[0] * best[1]:
                best = actual
        if chosen is None:
            # Không ứng viên nào khớp đúng: quay lại kích thước thật lớn nhất đã thấy.
            chosen = _request(cap, best) or best
            if resolution is not None and verbose:
                print(f"Camera {index} không hỗ trợ {resolution[0]}x{resolution[1]}, dùng {chosen[0]}x{chosen[1]}",
                      file=sys.stderr)
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, fps)
            if (_read_size(cap) or chosen) != chosen:     # vài driver đổi định dạng khi đặt fps
                _request(cap, chosen)
    except Exception:
        cap.release()
        raise

    if verbose:
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        fps_text = f" @ {actual_fps:.0f} fps" if actual_fps and actual_fps > 0 else ""
        print(f"Camera {index}: {chosen[0]}x{chosen[1]}{fps_text} "
              f"(mặc định {default[0]}x{default[1]}, backend {cap.getBackendName()})")
    return cap
