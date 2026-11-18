import cv2
import numpy as np

FRAME_SIZE = 1280


def frame_padding(frame: np.ndarray) -> np.ndarray:
    """Fit a BGR frame into a centered FRAME_SIZE x FRAME_SIZE canvas without distortion.

    Ảnh luôn được scale đều sao cho cạnh dài bằng FRAME_SIZE (thu nhỏ hoặc phóng to),
    nên toạ độ landmark chuẩn hoá [0, 1] giống nhau bất kể độ phân giải đầu vào
    (vd. video cũ 720x720 và camera 1280x720 cho cùng hình học).
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected a BGR frame with shape (H, W, 3), got {getattr(frame, 'shape', None)}.")
    if frame.dtype != np.uint8:
        raise TypeError(f"Expected uint8 frame, got {frame.dtype}.")

    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Frame height and width must be positive.")
    if height == FRAME_SIZE and width == FRAME_SIZE:
        return frame

    scale = FRAME_SIZE / max(height, width)
    if scale != 1.0:
        new_width = max(1, min(FRAME_SIZE, round(width * scale)))
        new_height = max(1, min(FRAME_SIZE, round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        content = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    else:
        new_height, new_width = height, width
        content = frame

    top = (FRAME_SIZE - new_height) // 2
    bottom = FRAME_SIZE - new_height - top
    left = (FRAME_SIZE - new_width) // 2
    right = FRAME_SIZE - new_width - left

    padded = cv2.copyMakeBorder(
        content,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded

pad_frame = frame_padding
__all__ = ["FRAME_SIZE", "frame_padding", "pad_frame"]
