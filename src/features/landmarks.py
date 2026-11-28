"""Convert MediaPipe landmarks between JSON/array form and result objects read by `extract`."""

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
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' phải là mảng số") from None
    if array.shape != shape:
        raise ValueError(f"'{name}' phải có shape {shape}, nhận {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"'{name}' chứa NaN/vô cực")
    return array


def _points(array: np.ndarray) -> list[_Point]:
    return [_Point(*map(float, row)) for row in array]


def dict_to_results(frame: dict) -> tuple[SimpleNamespace, SimpleNamespace]:
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
    hand_landmarks = [_points(_array(hand_array, (HAND_POINTS, 3), "hands")) for hand_array in hands]
    hand_world = None if world is None else [_points(_array(hand_array, (HAND_POINTS, 3), "hands_world")) for hand_array in world]
    return (SimpleNamespace(pose_landmarks=pose_landmarks),
            SimpleNamespace(hand_landmarks=hand_landmarks, hand_world_landmarks=hand_world))


def results_to_dict(pose_res, hand_res) -> dict:
    def rows(landmarks, with_vis: bool = False) -> list[list[float]]:
        return [[landmark.x, landmark.y, landmark.z] + ([1.0 if landmark.visibility is None else landmark.visibility] if with_vis else [])
                for landmark in landmarks]

    pose = rows(pose_res.pose_landmarks[0], True) if pose_res and pose_res.pose_landmarks else None
    hands = list(hand_res.hand_landmarks[:MAX_HANDS]) if hand_res and hand_res.hand_landmarks else []
    world = getattr(hand_res, "hand_world_landmarks", None) if hand_res else None
    out = {"pose": pose, "hands": [rows(hand) for hand in hands]}
    if world is not None and len(world) >= len(hands):
        out["hands_world"] = [rows(hand) for hand in world[:len(hands)]]
    return out


LANDMARKS_FILE = "landmarks.npz"


def results_to_arrays(pose_res, hand_res) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = results_to_dict(pose_res, hand_res)
    pose = np.full((POSE_POINTS, 4), np.nan, dtype=np.float32)
    hands = np.full((MAX_HANDS, HAND_POINTS, 3), np.nan, dtype=np.float32)
    world = np.full((MAX_HANDS, HAND_POINTS, 3), np.nan, dtype=np.float32)
    if frame["pose"] is not None:
        pose[:] = frame["pose"]
    for hand_index, hand in enumerate(frame["hands"]):
        hands[hand_index] = hand
    for hand_index, hand in enumerate(frame.get("hands_world") or []):
        world[hand_index] = hand
    return pose, hands, world


def arrays_to_results(pose: np.ndarray, hands: np.ndarray, world: np.ndarray) -> tuple[SimpleNamespace, SimpleNamespace]:
    frame = {"pose": pose.tolist() if np.isfinite(pose).all() else None, "hands": []}
    kept_world = []
    for hand, hand_world in zip(hands, world):
        if np.isfinite(hand).all():
            frame["hands"].append(hand.tolist())
            kept_world.append(hand_world.tolist() if np.isfinite(hand_world).all() else None)
    if frame["hands"] and all(hand_world is not None for hand_world in kept_world):
        frame["hands_world"] = kept_world
    return dict_to_results(frame)


def save_landmarks(path, pose: np.ndarray, hands: np.ndarray, world: np.ndarray) -> None:
    pose, hands, world = check_landmarks(pose, hands, world)
    np.savez_compressed(path, pose=pose, hands=hands, world=world)


def load_landmarks(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return check_landmarks(data["pose"], data["hands"], data["world"])


def check_landmarks(pose, hands, world) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose, hands, world = (np.asarray(array, dtype=np.float32) for array in (pose, hands, world))
    frame_count = len(pose)
    if (pose.shape != (frame_count, POSE_POINTS, 4) or hands.shape != (frame_count, MAX_HANDS, HAND_POINTS, 3)
            or world.shape != hands.shape):
        raise ValueError(f"Landmark sai shape: pose {pose.shape}, hands {hands.shape}, world {world.shape}")
    if any(np.isinf(array).any() for array in (pose, hands, world)):
        raise ValueError("Landmark chứa vô cực")
    return pose, hands, world
