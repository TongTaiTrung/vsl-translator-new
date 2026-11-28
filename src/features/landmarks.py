"""Landmark MediaPipe dạng JSON ↔ đối tượng kết quả mà ``extract`` đọc.

Client (web) chạy MediaPipe trên frame đã pad về ``FRAME_SIZE`` x ``FRAME_SIZE`` (như ``frame_padding``,
lật gương trước nếu ``mirror_data``) rồi gửi landmark thay cho ảnh. Mỗi frame là một dict::

    {
      "pose": [[x, y, z, visibility], ...] | null,   # 33 điểm của người đầu tiên; null = không có người
      "hands": [[[x, y, z], ...], ...],              # 0–2 tay, mỗi tay 21 điểm (toạ độ ảnh chuẩn hoá)
      "hands_world": [[[x, y, z], ...], ...]         # tuỳ chọn; cùng số tay, toạ độ mét (hand_world_landmarks)
    }

``extract`` chỉ dùng các trường trên (không dùng handedness / pose_world_landmarks).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

POSE_POINTS = 33
HAND_POINTS = 21
MAX_HANDS = 2


class _Point:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, z: float, visibility: float | None = None) -> None:
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' phải là mảng số") from None
    if arr.shape != shape:
        raise ValueError(f"'{name}' phải có shape {shape}, nhận {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"'{name}' chứa NaN/vô cực")
    return arr


def _points(arr: np.ndarray) -> list[_Point]:
    return [_Point(*map(float, row)) for row in arr]


def dict_to_results(frame: dict) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Một frame landmark JSON → ``(pose_res, hand_res)`` cho ``extract``. Sai định dạng → ``ValueError``."""
    if not isinstance(frame, dict):
        raise ValueError("Mỗi frame landmark phải là object")
    pose = frame.get("pose")
    pose_landmarks = [] if pose is None else [_points(_array(pose, (POSE_POINTS, 4), "pose"))]

    hands = frame.get("hands") or []
    if not isinstance(hands, list) or len(hands) > MAX_HANDS:
        raise ValueError(f"'hands' phải là mảng 0–{MAX_HANDS} tay")
    world = frame.get("hands_world")
    if world is not None and (not isinstance(world, list) or len(world) != len(hands)):
        raise ValueError("'hands_world' phải cùng số tay với 'hands'")
    hand_landmarks = [_points(_array(h, (HAND_POINTS, 3), "hands")) for h in hands]
    hand_world = None if world is None else [_points(_array(h, (HAND_POINTS, 3), "hands_world")) for h in world]
    return (SimpleNamespace(pose_landmarks=pose_landmarks),
            SimpleNamespace(hand_landmarks=hand_landmarks, hand_world_landmarks=hand_world))


def results_to_dict(pose_res, hand_res) -> dict:
    """Ngược lại ``dict_to_results`` (kết quả MediaPipe Python → JSON), dùng để test / ghi log."""
    def rows(landmarks, with_vis: bool = False) -> list[list[float]]:
        return [[lm.x, lm.y, lm.z] + ([1.0 if lm.visibility is None else lm.visibility] if with_vis else [])
                for lm in landmarks]

    pose = rows(pose_res.pose_landmarks[0], True) if pose_res and pose_res.pose_landmarks else None
    hands = list(hand_res.hand_landmarks[:MAX_HANDS]) if hand_res and hand_res.hand_landmarks else []
    world = getattr(hand_res, "hand_world_landmarks", None) if hand_res else None
    out = {"pose": pose, "hands": [rows(h) for h in hands]}
    if world is not None and len(world) >= len(hands):
        out["hands_world"] = [rows(h) for h in world[:len(hands)]]
    return out


LANDMARKS_FILE = "landmarks.npz"


def results_to_arrays(pose_res, hand_res) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kết quả MediaPipe của 1 frame → ``pose (33, 4)``, ``hands (2, 21, 3)``, ``world (2, 21, 3)`` float32.

    Phần không có (không người, < 2 tay, thiếu world) là NaN. Thứ tự tay giữ nguyên như MediaPipe trả về
    (``extract`` tự ghép trái/phải theo cổ tay pose nên thứ tự không quan trọng).
    """
    frame = results_to_dict(pose_res, hand_res)
    pose = np.full((POSE_POINTS, 4), np.nan, dtype=np.float32)
    hands = np.full((MAX_HANDS, HAND_POINTS, 3), np.nan, dtype=np.float32)
    world = np.full((MAX_HANDS, HAND_POINTS, 3), np.nan, dtype=np.float32)
    if frame["pose"] is not None:
        pose[:] = frame["pose"]
    for k, h in enumerate(frame["hands"]):
        hands[k] = h
    for k, h in enumerate(frame.get("hands_world") or []):
        world[k] = h
    return pose, hands, world


def arrays_to_results(pose: np.ndarray, hands: np.ndarray, world: np.ndarray) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Ngược lại ``results_to_arrays`` cho 1 frame: hàng/tay có NaN coi như không detect được."""
    frame = {"pose": pose.tolist() if np.isfinite(pose).all() else None, "hands": []}
    kept_world = []
    for h, w in zip(hands, world):
        if np.isfinite(h).all():
            frame["hands"].append(h.tolist())
            kept_world.append(w.tolist() if np.isfinite(w).all() else None)
    if frame["hands"] and all(w is not None for w in kept_world):
        frame["hands_world"] = kept_world
    return dict_to_results(frame)


def save_landmarks(path, pose: np.ndarray, hands: np.ndarray, world: np.ndarray) -> None:
    """Ghi landmark của cả clip: pose (T, 33, 4), hands / world (T, 2, 21, 3)."""
    pose, hands, world = check_landmarks(pose, hands, world)
    np.savez_compressed(path, pose=pose, hands=hands, world=world)


def load_landmarks(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return check_landmarks(data["pose"], data["hands"], data["world"])


def check_landmarks(pose, hands, world) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose, hands, world = (np.asarray(a, dtype=np.float32) for a in (pose, hands, world))
    T = len(pose)
    if (pose.shape != (T, POSE_POINTS, 4) or hands.shape != (T, MAX_HANDS, HAND_POINTS, 3)
            or world.shape != hands.shape):
        raise ValueError(f"Landmark sai shape: pose {pose.shape}, hands {hands.shape}, world {world.shape}")
    if any(np.isinf(a).any() for a in (pose, hands, world)):
        raise ValueError("Landmark chứa vô cực")
    return pose, hands, world
