"""Feature extraction: per-frame features (FEATURE_DIM) and 30-frame sequences (SEQ_FEATURE_DIM)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

from src.builtin.mediapipe_builtin.defs import HandLandmarkerResult, PoseLandmarkerResult

FEATURE_VERSION = 8

Z_POSE = 0.5
TORSO_Z = 0.3
VIS_MIN = 0.3
PITCH_AUG_DEG = 12.0
POSE_Z_GAIN = 3.35
MAX_MATCH_DIST = 0.8
PRESENCE_TEMPERATURE = 0.35
USE_HAND_WORLD = True
REST_Y = 0.9
CONTACT_Z = 0.25
STROKE_LAG = 5

_GROUPS = (
    ("hand_pos", 18),
    ("inter_hand_vec", 3),
    ("body_contact", 22),
    ("hand_contact", 4),
    ("face_vec", 12),
    ("hand_scale", 2),
    ("handshape", 76),
    ("hand_local", 120),
    ("finger_curl", 90),
    ("hand_closure", 16),
    ("orientation", 18),
    ("presence", 2),
)
FEATURE_SLICES: dict[str, slice] = {}
_offset = 0
for _name, _size in _GROUPS:
    FEATURE_SLICES[_name] = slice(_offset, _offset + _size)
    _offset += _size
FEATURE_DIM = _offset

_COLUMN_INDEX = np.arange(FEATURE_DIM)
_MOTION_COLS = np.concatenate([_COLUMN_INDEX[FEATURE_SLICES[n]] for n in ("hand_pos", "inter_hand_vec", "orientation")])
MOTION_DIM = len(_MOTION_COLS)

_STROKE_SRC = [(FEATURE_SLICES["hand_pos"].start + 9 * side + 6, FEATURE_SLICES["hand_pos"].start + 9 * side) for side in (0, 1)]
STROKE_DIM = 12
SEQ_FEATURE_DIM = FEATURE_DIM + MOTION_DIM + STROKE_DIM

PRESENCE_SLICE = FEATURE_SLICES["presence"]
MOTION_SLICE = slice(FEATURE_DIM, FEATURE_DIM + MOTION_DIM)
STROKE_SLICE = slice(MOTION_SLICE.stop, SEQ_FEATURE_DIM)


def _side_raw_cols(side: int) -> np.ndarray:
    cols = []
    for name in ("hand_pos", "body_contact", "face_vec", "hand_scale", "handshape", "hand_local", "finger_curl",
                 "hand_closure", "orientation", "presence"):
        feature_slice = FEATURE_SLICES[name]
        half = (feature_slice.stop - feature_slice.start) // 2
        cols.append(_COLUMN_INDEX[feature_slice.start + side * half: feature_slice.start + (side + 1) * half])
    return np.concatenate(cols)


_SHARED_RAW = np.concatenate([_COLUMN_INDEX[FEATURE_SLICES["inter_hand_vec"]], _COLUMN_INDEX[FEATURE_SLICES["hand_contact"]]])
_PALM_Y = (FEATURE_SLICES["hand_pos"].start + 1, FEATURE_SLICES["hand_pos"].start + 10)


def _with_motion(raw_cols: np.ndarray, side: int | None = None) -> np.ndarray:
    motion = FEATURE_DIM + np.flatnonzero(np.isin(_MOTION_COLS, raw_cols))
    half = STROKE_DIM // 2
    stroke = np.arange(0) if side is None else np.arange(STROKE_SLICE.start + side * half,
                                                         STROKE_SLICE.start + (side + 1) * half)
    return np.concatenate([raw_cols, motion, stroke])


SIDE_COLUMNS: dict[str, np.ndarray] = {
    "left": _with_motion(_side_raw_cols(0), 0),
    "right": _with_motion(_side_raw_cols(1), 1),
    "shared": _with_motion(_SHARED_RAW),
}


def _left_mirror() -> tuple[np.ndarray, np.ndarray]:
    cols = SIDE_COLUMNS["left"]
    column_position = {int(column): left_position for left_position, column in enumerate(cols)}
    hand_pos_start, body_contact_start, orientation_start = FEATURE_SLICES["hand_pos"].start, FEATURE_SLICES["body_contact"].start, FEATURE_SLICES["orientation"].start
    flip = [hand_pos_start, hand_pos_start + 3, hand_pos_start + 6, orientation_start, orientation_start + 3, orientation_start + 8]
    flip += [FEATURE_DIM + int(np.flatnonzero(_MOTION_COLS == column)[0]) for column in flip]
    face_vec_start = FEATURE_SLICES["face_vec"].start
    flip += [face_vec_start, face_vec_start + 3]
    flip += [STROKE_SLICE.start, STROKE_SLICE.start + 3]
    sign = np.ones(len(cols), dtype=np.float32)
    sign[[column_position[column] for column in flip]] = -1.0
    perm = np.arange(len(cols))
    for left_offset, right_offset in ((4, 5), (9, 10)):
        left_position, right_position = column_position[body_contact_start + left_offset], column_position[body_contact_start + right_offset]
        perm[left_position], perm[right_position] = right_position, left_position
    return perm, sign


LEFT_MIRROR = _left_mirror()

_PITCH_POS = np.concatenate([FEATURE_SLICES["hand_pos"].start + 3 * np.arange(6), [FEATURE_SLICES["inter_hand_vec"].start],
                             FEATURE_SLICES["face_vec"].start + 3 * np.arange(4)])
_PITCH_DIR = FEATURE_SLICES["orientation"].start + np.array([0, 3, 9, 12])


def rotate_view_pitch(features: np.ndarray, degrees: float) -> np.ndarray:
    rotated = np.array(features, dtype=np.float32, copy=True)
    if rotated.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Expected (..., {FEATURE_DIM}), got {rotated.shape}")
    angle = np.radians(float(degrees))
    cosine, sine = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    for starts, z_scale in ((_PITCH_POS, Z_POSE * POSE_Z_GAIN), (_PITCH_DIR, 1.0)):
        y_values = rotated[..., starts + 1]
        z_values = rotated[..., starts + 2] / z_scale
        rotated[..., starts + 1] = y_values * cosine + z_values * sine
        rotated[..., starts + 2] = (z_values * cosine - y_values * sine) * z_scale
    return rotated

_FINGERTIPS = np.array([4, 8, 12, 16, 20])
_FINGER_BASES = np.array([1, 5, 9, 13, 17])
_PALM_INDICES = np.array([0, 5, 9, 13, 17])
_PARENT_JOINTS = np.array([0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 13, 14, 0, 17, 18])
_JOINTS = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
_CHILD_JOINTS = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])
_PINCH_FROM = np.array([4, 4, 4, 4, 8, 12, 16, 4])
_PINCH_TO = np.array([8, 12, 16, 20, 12, 16, 20, 6])
_PALM_BONES = np.array([[0, 5], [0, 9], [0, 13], [0, 17], [5, 17]])

_ANCHOR_GAMMA = np.array([18, 18, 18, 18, 18, 18, 10, 6, 4, 8, 8], dtype=np.float32)


def _unit_vector(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / (norm + eps)


def _contact_kernel(dist: np.ndarray, gamma) -> np.ndarray:
    return np.exp(-gamma * (dist ** 2))


def _landmark_visibility(landmark) -> float:
    visibility = getattr(landmark, "visibility", None)
    return 1.0 if visibility is None else float(visibility)


def _body_anchors(torso_pose: np.ndarray) -> np.ndarray:
    nose = torso_pose[0]
    eye_mid = 0.5 * (torso_pose[2] + torso_pose[5])
    mouth = 0.5 * (torso_pose[9] + torso_pose[10])
    forehead = eye_mid + 0.8 * (eye_mid - mouth)
    chin = mouth + 1.2 * (mouth - nose)
    neck = 0.5 * chin
    cheek_l = 0.5 * (torso_pose[7] + mouth)
    cheek_r = 0.5 * (torso_pose[8] + mouth)
    chest = np.array([0.0, 0.35, 0.0])
    belly = np.array([0.0, 0.9, 0.0])
    sh_l = np.array([0.5, 0.0, 0.0])
    sh_r = np.array([-0.5, 0.0, 0.0])
    return np.stack([forehead, nose, mouth, chin, cheek_l, cheek_r,
                     neck, chest, belly, sh_l, sh_r]).astype(np.float32)


def _body_contacts(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    diff = anchors[None, :, :] - points[:, None, :]
    diff[..., 2] *= CONTACT_Z
    distances = np.linalg.norm(diff, axis=-1)
    return _contact_kernel(distances, _ANCHOR_GAMMA[None, :]).max(axis=0)


def _hand_scale(image_points: np.ndarray, shoulder_width: float) -> float:
    bones = np.linalg.norm(image_points[_PALM_BONES[:, 0]] - image_points[_PALM_BONES[:, 1]], axis=-1)
    return float(bones.mean() / max(shoulder_width, 1e-6))


_FINGER_CHAINS = np.array([[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
                           [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]])
CURL_DIM = 45


def _curl_stats(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bones = np.diff(points[_FINGER_CHAINS], axis=1)
    lengths = np.linalg.norm(bones, axis=-1)
    incoming, outgoing = bones[:, :-1].astype(np.float64), bones[:, 1:].astype(np.float64)
    angles = np.arctan2(np.linalg.norm(np.cross(incoming, outgoing), axis=-1), np.sum(incoming * outgoing, axis=-1))
    total = angles.sum(axis=1)
    hook = angles[:, 1:].sum(axis=1)
    chord = np.linalg.norm(points[_FINGER_CHAINS[:, 4]] - points[_FINGER_CHAINS[:, 1]], axis=-1)
    straight = chord / (lengths[:, 1:].sum(axis=1) + 1e-9)
    return angles.ravel(), total, hook, straight


def _finger_curl(image_points: np.ndarray, world: np.ndarray) -> np.ndarray:
    image_angles, image_total, image_hook, image_straight = _curl_stats(image_points)
    _, world_total, world_hook, world_straight = _curl_stats(world)
    return np.concatenate([image_angles, image_total, image_hook, image_straight, world_total, world_hook, world_straight]).astype(np.float32)


_CLOSURE_SEGMENT_START = np.array([5, 6, 7, 9, 10, 11])
_CLOSURE_SEGMENT_END = np.array([6, 7, 8, 10, 11, 12])
_THUMB_LOOP = np.array([4, 3, 2, 5, 6, 7, 8])
CLOSURE_DIM = 8


def _closure_stats(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float64)
    palm = np.linalg.norm(points[_PALM_BONES[:, 0]] - points[_PALM_BONES[:, 1]], axis=-1).mean() + 1e-9
    thumb_tip = points[4]
    segment_start, segment_end = points[_CLOSURE_SEGMENT_START], points[_CLOSURE_SEGMENT_END]
    segment_vector = segment_end - segment_start
    segment_fraction = np.clip(np.sum((thumb_tip - segment_start) * segment_vector, axis=-1) / (np.sum(segment_vector * segment_vector, axis=-1) + 1e-12), 0.0, 1.0)
    segment_distance = np.linalg.norm(thumb_tip - (segment_start + segment_fraction[:, None] * segment_vector), axis=-1).min()
    gap_i = np.linalg.norm(thumb_tip - points[8])
    gap_m = np.linalg.norm(thumb_tip - points[12])
    loop = np.linalg.norm(np.diff(points[_THUMB_LOOP], axis=0), axis=-1).sum()
    return np.array([gap_i / palm, gap_m / palm, segment_distance / palm, gap_i / (gap_i + loop + 1e-9)])


def _hand_closure(image_points: np.ndarray, world: np.ndarray) -> np.ndarray:
    return np.concatenate([_closure_stats(image_points), _closure_stats(world)]).astype(np.float32)


def _hand_hand_contacts(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    distances = np.array([
        np.linalg.norm(left_points[0] - right_points[0]),
        np.linalg.norm(left_points[1] - right_points[1]),
        np.linalg.norm(left_points[2] - right_points[2]),
        cdist(left_points, right_points).min(),
    ], dtype=np.float32)
    return _contact_kernel(distances, 8.0)


def _assign_hands(hand_wrists: np.ndarray, pose_wrists: np.ndarray, shoulder_width: float) -> list:
    assign = [None, None]
    if len(hand_wrists) == 0:
        return assign
    cost = cdist(hand_wrists, pose_wrists) / shoulder_width
    for hand_index, side in zip(*linear_sum_assignment(cost)):
        if cost[hand_index, side] <= MAX_MATCH_DIST:
            assign[side] = int(hand_index)
    return assign


def _hand_match_confidence(hand_wrist: np.ndarray, pose_wrist: np.ndarray, shoulder_width: float) -> float:
    distance = float(np.linalg.norm(hand_wrist - pose_wrist) / max(shoulder_width, 1e-6))
    if distance > MAX_MATCH_DIST:
        return 0.0
    return float(np.exp(-((distance / PRESENCE_TEMPERATURE) ** 2)))


def _hand_shape_and_orientation(
    local: np.ndarray,
    is_left: bool,
    forearm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    palm_size = np.linalg.norm(local[9] - local[0]) + 1e-6

    palm_dir = _unit_vector(local[9] - local[0])
    palm_normal = _unit_vector(np.cross(local[5] - local[0], local[17] - local[0]))
    if is_left:
        palm_normal = -palm_normal
    palm_side = _unit_vector(np.cross(palm_dir, palm_normal))

    incoming_bones = _unit_vector(local[_JOINTS] - local[_PARENT_JOINTS])
    outgoing_bones = _unit_vector(local[_CHILD_JOINTS] - local[_JOINTS])
    joint_cos = np.sum(incoming_bones * outgoing_bones, axis=-1)

    tip_ext = np.linalg.norm(local[_FINGERTIPS] - local[0], axis=-1) / palm_size

    f_dir = _unit_vector(local[_FINGERTIPS] - local[_FINGER_BASES])
    spread = np.sum(f_dir[:-1] * f_dir[1:], axis=-1)

    pair_d = np.linalg.norm(local[_PINCH_FROM] - local[_PINCH_TO], axis=-1) / palm_size
    intra = _contact_kernel(pair_d, 3.0)

    across = local[17] - local[5]
    across_len = np.linalg.norm(across) + 1e-6
    u_across = across / across_len
    palm_center = local[_PALM_INDICES].mean(axis=0)
    thumb = np.array([
        np.dot(local[4] - local[5], u_across) / across_len,
        np.dot(local[4] - palm_center, palm_normal) / palm_size,
        np.dot(local[4] - local[0], palm_dir) / palm_size,
    ])
    tip_order = (local[[12, 16, 20]] - local[[8, 12, 16]]) @ u_across / across_len

    handshape = np.concatenate([joint_cos, tip_ext, spread, intra, thumb, tip_order])

    across_axis = u_across
    forward_axis = _unit_vector(palm_dir - np.dot(palm_dir, across_axis) * across_axis - np.dot(palm_dir, palm_normal) * palm_normal)
    hand_local = ((local[1:] - local[0]) @ np.stack([forward_axis, across_axis, palm_normal], axis=1) / palm_size).ravel()

    if forearm is None:
        wrist = np.full(3, np.nan)
    else:
        wrist = np.array([np.dot(forearm, palm_dir),
                          np.dot(forearm, palm_normal),
                          np.dot(forearm, palm_side)])
    orientation = np.concatenate([palm_dir, palm_normal, wrist])

    return handshape.astype(np.float32), orientation.astype(np.float32), hand_local.astype(np.float32)


def extract(pose_res: PoseLandmarkerResult, hand_res: HandLandmarkerResult) -> np.ndarray:
    nan_features = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    if not pose_res or not pose_res.pose_landmarks:
        return nan_features

    pose_landmarks = pose_res.pose_landmarks[0]
    pose_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks], dtype=np.float32)
    pose_points[:, 2] *= Z_POSE

    left_shoulder, right_shoulder = pose_points[11], pose_points[12]
    mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if shoulder_width < 0.02:
        return nan_features

    torso_scale = np.array([1.0, 1.0, TORSO_Z], dtype=np.float32)
    torso_x = _unit_vector((left_shoulder - right_shoulder) * torso_scale)

    v_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    torso_y = _unit_vector(v_up - np.dot(v_up, torso_x) * torso_x)
    torso_z = _unit_vector(np.cross(torso_x, torso_y))
    torso_rotation = np.stack([torso_x, torso_y, torso_z], axis=1).astype(np.float32)

    def to_torso(points: np.ndarray) -> np.ndarray:
        return ((points - mid_shoulder) @ torso_rotation) / shoulder_width

    torso_pose = to_torso(pose_points)
    anchors = _body_anchors(torso_pose)

    hand_landmark_lists = list(hand_res.hand_landmarks[:2]) if (hand_res and hand_res.hand_landmarks) else []
    hand_world = getattr(hand_res, "hand_world_landmarks", None) if hand_res else None
    hand_wrists = np.array([[hand[0].x, hand[0].y] for hand in hand_landmark_lists], dtype=np.float32).reshape(-1, 2)
    assign = _assign_hands(hand_wrists, pose_points[[15, 16], :2], shoulder_width)

    sides = []
    for side in (0, 1):
        is_left = side == 0
        wrist_index, elbow_index = 15 + side, 13 + side
        wrist_visible = _landmark_visibility(pose_landmarks[wrist_index]) >= VIS_MIN

        forearm = None
        if wrist_visible and _landmark_visibility(pose_landmarks[elbow_index]) >= VIS_MIN:
            forearm = _unit_vector(torso_pose[wrist_index] - torso_pose[elbow_index])

        hand_index = assign[side]
        if hand_index is not None:
            image_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmark_lists[hand_index]], dtype=np.float32)

            global_points = image_points.copy()
            global_points[:, 2] = pose_points[wrist_index, 2] + image_points[:, 2] * Z_POSE
            torso_points = to_torso(global_points)
            palm_center = torso_points[_PALM_INDICES].mean(axis=0)
            hand_points = np.vstack([palm_center, torso_points[_FINGERTIPS]])

            if USE_HAND_WORLD and hand_world and len(hand_world) > hand_index:
                shape_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_world[hand_index]], dtype=np.float32)
            else:
                shape_points = image_points
            local = shape_points @ torso_rotation

            shape, orient, hand_local_points = _hand_shape_and_orientation(local, is_left, forearm)
            curl = _finger_curl(image_points, shape_points)
            closure = _hand_closure(image_points, shape_points)
            present = _hand_match_confidence(hand_wrists[hand_index], pose_points[wrist_index, :2], shoulder_width)
            scale = _hand_scale(image_points, shoulder_width)
        else:
            hand_points = np.full((6, 3), np.nan, dtype=np.float32)
            shape = np.full((FEATURE_SLICES["handshape"].stop - FEATURE_SLICES["handshape"].start) // 2, np.nan, dtype=np.float32)
            orient = np.full(9, np.nan, dtype=np.float32)
            hand_local_points = np.full(60, np.nan, dtype=np.float32)
            curl = np.full(CURL_DIM, np.nan, dtype=np.float32)
            closure = np.full(CLOSURE_DIM, np.nan, dtype=np.float32)
            present = 0.0
            scale = np.nan
        sides.append((hand_points, shape, orient, hand_local_points, curl, closure, present, scale))

    ((left_points, left_shape, left_orient, left_local, left_curl, left_closure, left_presence, left_scale),
     (right_points, right_shape, right_orient, right_local, right_curl, right_closure, right_presence, right_scale)) = sides
    nose = torso_pose[0]

    hand_pos = np.concatenate([left_points[:3].ravel(), right_points[:3].ravel()])
    inter_hand_vec = left_points[0] - right_points[0]
    body_contact = np.concatenate([_body_contacts(left_points, anchors),
                                   _body_contacts(right_points, anchors)])
    hand_contact = _hand_hand_contacts(left_points, right_points)
    face_vec = np.concatenate([(left_points[[0, 2]] - nose).ravel(),
                               (right_points[[0, 2]] - nose).ravel()])
    hand_scale = np.array([left_scale, right_scale], dtype=np.float32)

    handshape = np.concatenate([left_shape, right_shape])
    hand_local = np.concatenate([left_local, right_local])
    finger_curl = np.concatenate([left_curl, right_curl])
    hand_closure = np.concatenate([left_closure, right_closure])
    orientation = np.concatenate([left_orient, right_orient])
    presence = np.array([left_presence, right_presence], dtype=np.float32)

    features = np.concatenate([hand_pos, inter_hand_vec, body_contact, hand_contact, face_vec, hand_scale,
                          handshape, hand_local, finger_curl, hand_closure, orientation, presence]).astype(np.float32)
    return features


def _fill_gaps(column: np.ndarray, max_gap: int) -> np.ndarray:
    missing = np.isnan(column)
    if not missing.any():
        return column
    if missing.all():
        return np.zeros_like(column)
    known = np.flatnonzero(~missing)
    filled = column.copy()
    filled[missing] = np.interp(np.flatnonzero(missing), known, column[known])
    runs, _ = ndimage.label(missing)
    for gap in ndimage.find_objects(runs):
        if gap[0].stop - gap[0].start > max_gap:
            filled[gap] = 0.0
    return filled


def _spike_scales() -> np.ndarray:
    scales = np.full(FEATURE_DIM, 2.0, dtype=np.float32)
    scales[FEATURE_SLICES["hand_pos"]] = 1.0
    scales[FEATURE_SLICES["inter_hand_vec"]] = 1.0
    scales[FEATURE_SLICES["face_vec"]] = 1.0
    scales[FEATURE_SLICES["hand_scale"]] = 0.5
    scales[FEATURE_SLICES["presence"]] = np.inf
    return scales


def _drop_resting_hands(sequence: np.ndarray) -> None:
    for side in (0, 1):
        palm_y = sequence[:, _PALM_Y[side]]
        palm_y = palm_y[~np.isnan(palm_y)]
        if palm_y.size and np.median(palm_y) > REST_Y:
            sequence[:, _side_raw_cols(side)] = np.nan
            sequence[:, _SHARED_RAW] = np.nan
            sequence[:, PRESENCE_SLICE.start + side] = 0.0


def active_presence(feature: np.ndarray) -> np.ndarray:
    palm_y = np.asarray(feature, dtype=np.float32)[list(_PALM_Y)]
    return palm_y <= REST_Y


def _hand_valid(sequence: np.ndarray) -> np.ndarray:
    hand_pos_start = FEATURE_SLICES["hand_pos"].start
    return np.stack([(np.abs(sequence[:, hand_pos_start + 9 * side: hand_pos_start + 9 * (side + 1)]).sum(1) > 0) & (sequence[:, _PALM_Y[side]] <= REST_Y)
                     for side in (0, 1)], axis=1)


def _stroke(positions: np.ndarray, valid: np.ndarray, lag: int = STROKE_LAG) -> np.ndarray:
    frame_count = len(positions)
    lag = min(lag, frame_count - 1)
    stroke = np.zeros(STROKE_DIM, dtype=np.float32)
    if lag < 1:
        return stroke
    half = STROKE_DIM // 2
    for side, (tip_column, palm_column) in enumerate(_STROKE_SRC):
        window_valid = np.lib.stride_tricks.sliding_window_view(valid[:, side], lag + 1).all(axis=1)
        if not window_valid.any():
            continue
        d_tip = positions[lag:, tip_column:tip_column + 3] - positions[:-lag, tip_column:tip_column + 3]
        score = np.where(window_valid, np.linalg.norm(d_tip * np.array([1.0, 1.0, CONTACT_Z]), axis=1), -1.0)
        start_frame = int(np.argmax(score))
        stroke[side * half: side * half + 3] = d_tip[start_frame]
        stroke[side * half + 3: (side + 1) * half] = positions[start_frame + lag, palm_column:palm_column + 3] - positions[start_frame, palm_column:palm_column + 3]
    return stroke


def process_sequence(
    raw_sequence: list[np.ndarray] | np.ndarray,
    target_len: int = 30,
    spike_threshold: float = 0.45,
    max_gap: int = 6,
    add_motion: bool = True,
) -> np.ndarray:
    sequence = np.array(raw_sequence, dtype=np.float32)
    out_dim = SEQ_FEATURE_DIM if add_motion else FEATURE_DIM
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        return np.zeros((target_len, out_dim), dtype=np.float32)
    num_frames = sequence.shape[0]

    presence = sequence[:, FEATURE_SLICES["presence"]]
    sequence[:, FEATURE_SLICES["presence"]] = np.nan_to_num(presence, nan=0.0)
    _drop_resting_hands(sequence)

    for column_index in range(FEATURE_DIM):
        sequence[:, column_index] = _fill_gaps(sequence[:, column_index], max_gap)

    if num_frames >= 3:
        thresholds = spike_threshold * _spike_scales()
        prev_diff = sequence[1:-1] - sequence[:-2]
        next_diff = sequence[1:-1] - sequence[2:]
        neighbor_gap = np.abs(sequence[2:] - sequence[:-2])
        magnitude = np.minimum(np.abs(prev_diff), np.abs(next_diff))
        is_spike = (prev_diff * next_diff > 0) & (magnitude > thresholds) & (neighbor_gap < 0.5 * magnitude)
        midpoint = 0.5 * (sequence[:-2] + sequence[2:])
        sequence[1:-1] = np.where(is_spike, midpoint, sequence[1:-1])

    valid = _hand_valid(sequence).astype(np.float32)
    if num_frames != target_len:
        if num_frames == 1:
            sequence = np.repeat(sequence, target_len, axis=0)
            valid = np.repeat(valid, target_len, axis=0)
        else:
            old_time = np.linspace(0.0, 1.0, num_frames)
            new_time = np.linspace(0.0, 1.0, target_len)
            sequence = interp1d(old_time, sequence, axis=0, kind="linear")(new_time).astype(np.float32)
            valid = interp1d(old_time, valid, axis=0, kind="linear")(new_time)
    valid = valid > 1.0 - 1e-6

    if not add_motion:
        return sequence.astype(np.float32)

    motion_source = sequence[:, _MOTION_COLS]
    if target_len >= 3:
        padded = np.pad(motion_source, ((1, 1), (0, 0)), mode="edge")
        motion_source = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    motion = (np.gradient(motion_source, axis=0) * (target_len - 1)).astype(np.float32)

    positions = sequence.copy()
    positions[:, _MOTION_COLS] = motion_source
    if target_len >= 3:
        padded_valid = np.pad(valid, ((1, 1), (0, 0)), mode="edge")
        valid = padded_valid[:-2] & padded_valid[1:-1] & padded_valid[2:]
    stroke = np.broadcast_to(_stroke(positions, valid), (target_len, STROKE_DIM))

    return np.concatenate([sequence, motion, stroke], axis=1).astype(np.float32)


STILL_MIN_HOLD = 4
STILL_MAX_STROKE = 0.15


def stroke_magnitude(sequence: np.ndarray) -> float:
    stroke = np.asarray(sequence, dtype=np.float32)[0, STROKE_SLICE]
    half = STROKE_DIM // 2
    return float(max(np.linalg.norm(stroke[side * half: side * half + 2]) for side in (0, 1)))


def still_versions(sequence: np.ndarray, target_len: int | None = None) -> list[np.ndarray]:
    raw_features = np.asarray(sequence, dtype=np.float32)[:, :FEATURE_DIM]
    frame_count = len(raw_features)
    target_len = target_len or frame_count
    valid = _hand_valid(raw_features)
    if frame_count < 2 or not valid.any():
        return []
    speeds = []
    for side, (tip_column, _palm_column) in enumerate(_STROKE_SRC):
        step = np.linalg.norm(np.diff(raw_features[:, tip_column:tip_column + 2], axis=0), axis=1)
        step_valid = valid[1:, side] & valid[:-1, side]
        speeds.append(np.where(step_valid, step, 0.0))
    speed = np.max(speeds, axis=0)
    full = np.asarray(sequence, dtype=np.float32) if np.shape(sequence)[1] == SEQ_FEATURE_DIM else process_sequence(raw_features)
    if speed.max() <= 0 or stroke_magnitude(full) <= STILL_MAX_STROKE:
        return []
    fast = np.flatnonzero(speed >= 0.3 * speed.max())
    start, stop = max(0, fast[0] - 1), min(frame_count - 1, fast[-1] + 2)
    moving = np.zeros(frame_count, dtype=bool)
    moving[start:stop + 1] = True
    any_valid = valid.any(axis=1)
    limit = min(0.5 * stroke_magnitude(full), STILL_MAX_STROKE)
    versions = []
    for segment in (np.arange(0, start), np.arange(stop + 1, frame_count)):
        segment = segment[any_valid[segment]]
        if len(segment) >= STILL_MIN_HOLD:
            held = process_sequence(raw_features[segment[0]:segment[-1] + 1], target_len=target_len)
            if stroke_magnitude(held) <= limit:
                versions.append(held)
    still = np.where(any_valid & ~moving)[0]
    if still.size == 0:
        still = np.where(any_valid)[0]
    frame_speed = np.concatenate([[np.inf], speed]) + np.concatenate([speed, [np.inf]])
    calmest = int(still[np.argmin(frame_speed[still])])
    versions.append(process_sequence(raw_features[calmest:calmest + 1], target_len=target_len))
    return versions
