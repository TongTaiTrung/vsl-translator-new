"""Simulate another recording setup (camera angle, distance, FOV, hand size) on raw landmarks and recompute features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.features.extract import FEATURE_DIM, PITCH_AUG_DEG, POSE_Z_GAIN, extract
from src.features.landmarks import arrays_to_results

FOCAL = 0.8
DEPTH_LIMITS = (-1.2, 0.5)
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_POSE_WRISTS = (15, 16)
_POSE_HAND_POINTS = {15: (17, 19, 21), 16: (18, 20, 22)}


@dataclass(frozen=True)
class ViewParams:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    distance: float = 1.0
    hand_size: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    finger_offsets: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        fields = {field: round(float(value), 4) for field, value in asdict(self).items() if field != "finger_offsets"}
        if self.finger_offsets:
            fields["finger_offset_rms"] = round(float(np.sqrt(np.mean(np.square(self.finger_offsets)))), 3)
        return fields


IDENTITY = ViewParams()


def random_view(rng: np.random.Generator, *, yaw: float = 20.0, pitch: float = PITCH_AUG_DEG, roll: float = 5.0,
                distance: tuple[float, float] = (0.7, 1.6), hand_size: tuple[float, float] = (0.9, 1.1),
                shift: float = 0.08, finger_sigma: float = 8.0) -> ViewParams:
    log_near, log_far = np.log(distance)
    fingers = np.clip(rng.normal(0.0, finger_sigma, 30), -2.5 * finger_sigma, 2.5 * finger_sigma) if finger_sigma > 0 else ()
    return ViewParams(
        finger_offsets=tuple(float(offset) for offset in fingers),
        yaw=float(rng.uniform(-yaw, yaw)), pitch=float(rng.uniform(-pitch, pitch)),
        roll=float(rng.uniform(-roll, roll)), distance=float(np.exp(rng.uniform(log_near, log_far))),
        hand_size=float(rng.uniform(*hand_size)),
        shift_x=float(rng.uniform(-shift, shift)), shift_y=float(rng.uniform(-shift, shift)),
    )


def rotation(view: ViewParams) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad([view.yaw, view.pitch, view.roll])
    yaw_matrix = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    pitch_matrix = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    roll_matrix = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return roll_matrix @ pitch_matrix @ yaw_matrix


_FINGER_CHAINS = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])


def _rotate_about(points: np.ndarray, center: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    unit_axis = axis / (np.linalg.norm(axis) + 1e-9)
    offsets = points - center
    return center + offsets * np.cos(angle) + np.cross(unit_axis, offsets) * np.sin(angle) + np.outer(offsets @ unit_axis, unit_axis) * (1 - np.cos(angle))


def bend_fingers(hand: np.ndarray, offsets: np.ndarray, is_left: bool) -> np.ndarray:
    bent = np.array(hand, dtype=np.float64)
    across = bent[17] - bent[5]
    along = bent[9] - bent[0]
    sign = -1.0 if is_left else 1.0
    for finger_index, chain in enumerate(_FINGER_CHAINS):
        axis = sign * (along if finger_index == 0 else across)
        for joint_index in range(3):
            degrees = offsets[3 * finger_index + joint_index]
            if degrees:
                tail = chain[joint_index + 1:]
                bent[tail] = _rotate_about(bent[tail], bent[chain[joint_index]], axis, degrees)
    return bent


def _match_pose_wrist(hand: np.ndarray, pose: np.ndarray) -> int:
    distances = [np.linalg.norm(hand[0, :2] - pose[wrist_index, :2]) for wrist_index in _POSE_WRISTS]
    return _POSE_WRISTS[int(np.argmin(distances))]


def _smooth_depth(depth: np.ndarray, window_size: int = 5) -> np.ndarray:
    smoothed = depth.copy()
    half = window_size // 2
    for frame_index in range(len(depth)):
        window = depth[max(0, frame_index - half): frame_index + half + 1]
        window = window[np.isfinite(window)]
        if window.size:
            smoothed[frame_index] = np.median(window)
    return smoothed


def simulate_view(pose: np.ndarray, hands: np.ndarray, world: np.ndarray, view: ViewParams,
                  focal: float = FOCAL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose, hands, world = (np.array(array, dtype=np.float64) for array in (pose, hands, world))
    frame_count = len(pose)
    rotation_matrix = rotation(view)
    focal_length, camera_distance = float(focal), float(view.distance)
    has_pose = np.isfinite(pose[:, :, :3]).all(axis=(1, 2))
    if view.finger_offsets:
        if len(view.finger_offsets) != 30:
            raise ValueError("finger_offsets phải có 30 số (2 tay x 15 khớp)")
        fingers = np.reshape(view.finger_offsets, (2, 15))
    else:
        fingers = (None, None)

    match = np.full((frame_count, 2), -1, dtype=int)
    wrist_depth = np.full((frame_count, 2), np.nan)
    for frame_index in np.flatnonzero(has_pose):
        shoulder_width = np.linalg.norm(pose[frame_index, _LEFT_SHOULDER, :2] - pose[frame_index, _RIGHT_SHOULDER, :2])
        shoulder_depth = 0.5 * (pose[frame_index, _LEFT_SHOULDER, 2] + pose[frame_index, _RIGHT_SHOULDER, 2])
        for hand_index in range(2):
            if np.isfinite(hands[frame_index, hand_index]).all():
                wrist_index = _match_pose_wrist(hands[frame_index, hand_index], pose[frame_index])
                match[frame_index, hand_index] = wrist_index
                depth = (pose[frame_index, wrist_index, 2] - shoulder_depth) / POSE_Z_GAIN
                wrist_depth[frame_index, hand_index] = np.clip(depth, DEPTH_LIMITS[0] * shoulder_width, DEPTH_LIMITS[1] * shoulder_width)
    for hand_index in range(2):
        wrist_depth[:, hand_index] = _smooth_depth(wrist_depth[:, hand_index])

    for frame_index in np.flatnonzero(has_pose):
        center_x, center_y = 0.5 * (pose[frame_index, _LEFT_SHOULDER, :2] + pose[frame_index, _RIGHT_SHOULDER, :2])
        shoulder_depth = 0.5 * (pose[frame_index, _LEFT_SHOULDER, 2] + pose[frame_index, _RIGHT_SHOULDER, 2])
        shoulder_width = np.linalg.norm(pose[frame_index, _LEFT_SHOULDER, :2] - pose[frame_index, _RIGHT_SHOULDER, :2])
        shift = np.array([view.shift_x, view.shift_y]) * shoulder_width

        def transform(xy: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            depth = np.maximum(depth, -0.8 * focal_length)
            rotated = np.column_stack([xy * ((focal_length + depth) / focal_length)[:, None], depth]) @ rotation_matrix.T
            camera_z = np.maximum(rotated[:, 2], -0.8 * focal_length * camera_distance)
            proj = focal_length / (focal_length * camera_distance + camera_z)
            return rotated[:, :2] * proj[:, None], camera_z - depth, focal_length / (focal_length + depth), proj

        pose_points = pose[frame_index, :, :3].copy()
        for hand_index in range(2):
            wrist_index = match[frame_index, hand_index]
            if wrist_index >= 0:
                pose_points[[wrist_index, *_POSE_HAND_POINTS[wrist_index]], :2] += shift
        xy, depth_shift, _, _ = transform(pose_points[:, :2] - [center_x, center_y], (pose_points[:, 2] - shoulder_depth) / POSE_Z_GAIN)
        pose[frame_index, :, :2] = xy + [center_x, center_y]
        pose[frame_index, :, 2] = shoulder_depth + (pose_points[:, 2] - shoulder_depth + depth_shift * POSE_Z_GAIN) / camera_distance

        for hand_index in range(2):
            if match[frame_index, hand_index] < 0:
                continue
            source_hand = hands[frame_index, hand_index]
            hand_points = source_hand[0] + (source_hand - source_hand[0]) * view.hand_size
            is_left = match[frame_index, hand_index] == _POSE_WRISTS[0]
            offsets = fingers[0 if is_left else 1]
            if offsets is not None:
                hand_points = bend_fingers(hand_points, offsets, is_left)
            hand_points[:, :2] += shift
            proj_old = focal_length / (focal_length + max(wrist_depth[frame_index, hand_index], -0.8 * focal_length))
            depth = wrist_depth[frame_index, hand_index] + (hand_points[:, 2] - hand_points[0, 2]) / proj_old
            xy, depth_shift, _, proj = transform(hand_points[:, :2] - [center_x, center_y], depth)
            moved_hand = np.empty_like(hand_points)
            moved_hand[:, :2] = xy + [center_x, center_y]
            z_true = depth + depth_shift
            moved_hand[:, 2] = source_hand[0, 2] + (z_true - z_true[0]) * proj[0]
            inside = lambda array: ((array[:, :2] >= -0.02) & (array[:, :2] <= 1.02)).all()
            if inside(source_hand) and not inside(moved_hand):
                hands[frame_index, hand_index] = np.nan
                world[frame_index, hand_index] = np.nan
                continue
            hands[frame_index, hand_index] = moved_hand
            if np.isfinite(world[frame_index, hand_index]).all():
                bent_world = world[frame_index, hand_index] if offsets is None else bend_fingers(world[frame_index, hand_index], offsets, is_left)
                world[frame_index, hand_index] = bent_world @ rotation_matrix.T

    return pose.astype(np.float32), hands.astype(np.float32), world.astype(np.float32)


def features_from_landmarks(pose: np.ndarray, hands: np.ndarray, world: np.ndarray) -> np.ndarray:
    features = np.empty((len(pose), FEATURE_DIM), dtype=np.float32)
    for frame_index in range(len(pose)):
        features[frame_index] = extract(*arrays_to_results(pose[frame_index], hands[frame_index], world[frame_index]))
    return features
