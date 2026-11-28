"""Giả lập bối cảnh quay khác (góc máy, khoảng cách, FOV, cỡ tay) trên LANDMARK thô rồi tính lại feature.

Đo trên dữ liệu (lớp ``m`` quay ở hai bối cảnh): cùng một ký hiệu ở hai bối cảnh cách nhau trong không gian
feature xa hơn hai ký hiệu khác nhau quay cùng bối cảnh, và lệch nhiều nhất ở các kênh phụ thuộc góc nhìn
(z, ``hand_scale``, thành phần pháp tuyến của ``hand_local``, cả cos khớp 3D). Augment trên feature
(``rotate_view_pitch``) chỉ xấp xỉ được pitch; ở đây ta biến đổi chính landmark rồi chạy lại ``extract`` nên
MỌI cột (kể cả thủ hình từ ``hand_world``) đổi nhất quán như khi quay thật bằng camera khác.

Mô hình hình học (đơn vị = bề rộng canvas FRAME_SIZE, trục như ảnh: x phải, y xuống, z ra xa camera):
  - Gốc xoay = giữa hai vai của từng frame. Độ sâu tương đối d = (z_pose - z_vai) / POSE_Z_GAIN.
  - Tay: d = d(cổ tay pose ghép với tay đó, làm mượt theo thời gian, kẹp biên) + z tương đối của tay.
  - Camera hiện tại cách gốc ``FOCAL`` (đơn vị canvas đã quy về mặt phẳng vai): bỏ phối cảnh
    X = u (f + d) / f, xoay quanh gốc, chiếu lại với khoảng cách f·r: u' = X' f / (f r + Z').
    r > 1: camera xa hơn / FOV hẹp hơn (phối cảnh yếu đi, tay trước ngực bớt "to"); r < 1 ngược lại.
  - ``hand_world`` (mét, trục theo camera) xoay cùng ma trận.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.features.extract import FEATURE_DIM, PITCH_AUG_DEG, POSE_Z_GAIN, extract
from src.features.landmarks import arrays_to_results

FOCAL = 0.8
DEPTH_LIMITS = (-1.2, 0.5)
_L_SH, _R_SH = 11, 12
_POSE_WRISTS = (15, 16)
_POSE_HAND_PTS = {15: (17, 19, 21), 16: (18, 20, 22)}


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
        out = {k: round(float(v), 4) for k, v in asdict(self).items() if k != "finger_offsets"}
        if self.finger_offsets:
            out["finger_offset_rms"] = round(float(np.sqrt(np.mean(np.square(self.finger_offsets)))), 3)
        return out


IDENTITY = ViewParams()


def random_view(rng: np.random.Generator, *, yaw: float = 20.0, pitch: float = PITCH_AUG_DEG, roll: float = 5.0,
                distance: tuple[float, float] = (0.7, 1.6), hand_size: tuple[float, float] = (0.9, 1.1),
                shift: float = 0.08, finger_sigma: float = 8.0) -> ViewParams:
    """Bối cảnh ngẫu nhiên: góc đều trong ±biên, khoảng cách log-đều, cỡ tay đều, dịch tay đều ±shift vai,
    độ gập từng khớp lệch N(0, finger_sigma) kẹp ±2.5 sigma."""
    lo, hi = np.log(distance)
    fingers = np.clip(rng.normal(0.0, finger_sigma, 30), -2.5 * finger_sigma, 2.5 * finger_sigma) if finger_sigma > 0 else ()
    return ViewParams(
        finger_offsets=tuple(float(v) for v in fingers),
        yaw=float(rng.uniform(-yaw, yaw)), pitch=float(rng.uniform(-pitch, pitch)),
        roll=float(rng.uniform(-roll, roll)), distance=float(np.exp(rng.uniform(lo, hi))),
        hand_size=float(rng.uniform(*hand_size)),
        shift_x=float(rng.uniform(-shift, shift)), shift_y=float(rng.uniform(-shift, shift)),
    )


def rotation(view: ViewParams) -> np.ndarray:
    """R = Rz(roll) · Rx(pitch) · Ry(yaw), trục ảnh (x phải, y xuống, z ra xa)."""
    y, p, r = np.deg2rad([view.yaw, view.pitch, view.roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


_FINGER_CHAINS = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])


def _rotate_about(points: np.ndarray, center: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues: xoay points (N,3) quanh trục qua center."""
    a = np.deg2rad(degrees)
    k = axis / (np.linalg.norm(axis) + 1e-9)
    v = points - center
    return center + v * np.cos(a) + np.cross(k, v) * np.sin(a) + np.outer(v @ k, k) * (1 - np.cos(a))


def bend_fingers(hand: np.ndarray, offsets: np.ndarray, is_left: bool) -> np.ndarray:
    """hand (21,3) → bản sao với khớp thứ j (thứ tự extract._JOINTS) gập thêm offsets[j] độ.

    Ngón 2-5 gập quanh trục ngang lòng bàn tay (gốc trỏ → gốc út), ngón cái quanh trục dọc bàn tay. Tay trái
    đổi dấu trục để cùng một offset là cùng chiều gập ở hai tay (đối xứng gương như extract).
    """
    out = np.array(hand, dtype=np.float64)
    across = out[17] - out[5]
    along = out[9] - out[0]
    sign = -1.0 if is_left else 1.0
    for f, chain in enumerate(_FINGER_CHAINS):
        axis = sign * (along if f == 0 else across)
        for j in range(3):
            deg = offsets[3 * f + j]
            if deg:
                tail = chain[j + 1:]
                out[tail] = _rotate_about(out[tail], out[chain[j]], axis, deg)
    return out


def _match_pose_wrist(hand: np.ndarray, pose: np.ndarray) -> int:
    d = [np.linalg.norm(hand[0, :2] - pose[w, :2]) for w in _POSE_WRISTS]
    return _POSE_WRISTS[int(np.argmin(d))]


def _smooth_depth(depth: np.ndarray, k: int = 5) -> np.ndarray:
    """Trung vị trượt (bỏ qua NaN) theo thời gian: z cổ tay pose nhảy từng frame."""
    out = depth.copy()
    half = k // 2
    for t in range(len(depth)):
        win = depth[max(0, t - half): t + half + 1]
        win = win[np.isfinite(win)]
        if win.size:
            out[t] = np.median(win)
    return out


def simulate_view(pose: np.ndarray, hands: np.ndarray, world: np.ndarray, view: ViewParams,
                  focal: float = FOCAL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Landmark của cả clip (T, …) nhìn từ bối cảnh ``view``. Không sửa mảng vào.

    Frame không có pose giữ nguyên (``extract`` trả NaN cho frame đó dù sao). Tay bị chiếu ra ngoài
    canvas thì coi như mất (NaN), như detector thật.
    """
    pose, hands, world = (np.array(a, dtype=np.float64) for a in (pose, hands, world))
    T = len(pose)
    R = rotation(view)
    f, r = float(focal), float(view.distance)
    has_pose = np.isfinite(pose[:, :, :3]).all(axis=(1, 2))
    if view.finger_offsets:
        if len(view.finger_offsets) != 30:
            raise ValueError("finger_offsets phải có 30 số (2 tay x 15 khớp)")
        fingers = np.reshape(view.finger_offsets, (2, 15))
    else:
        fingers = (None, None)

    match = np.full((T, 2), -1, dtype=int)
    wrist_d = np.full((T, 2), np.nan)
    for t in np.flatnonzero(has_pose):
        sw = np.linalg.norm(pose[t, _L_SH, :2] - pose[t, _R_SH, :2])
        pz = 0.5 * (pose[t, _L_SH, 2] + pose[t, _R_SH, 2])
        for k in range(2):
            if np.isfinite(hands[t, k]).all():
                w = _match_pose_wrist(hands[t, k], pose[t])
                match[t, k] = w
                d = (pose[t, w, 2] - pz) / POSE_Z_GAIN
                wrist_d[t, k] = np.clip(d, DEPTH_LIMITS[0] * sw, DEPTH_LIMITS[1] * sw)
    for k in range(2):
        wrist_d[:, k] = _smooth_depth(wrist_d[:, k])

    for t in np.flatnonzero(has_pose):
        px, py = 0.5 * (pose[t, _L_SH, :2] + pose[t, _R_SH, :2])
        pz = 0.5 * (pose[t, _L_SH, 2] + pose[t, _R_SH, 2])
        sw = np.linalg.norm(pose[t, _L_SH, :2] - pose[t, _R_SH, :2])
        shift = np.array([view.shift_x, view.shift_y]) * sw

        def transform(u: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """u (N,2) toạ độ ảnh trừ gốc, d (N,) độ sâu → (u', Z' - d, hệ số chiếu cũ, hệ số chiếu mới).

            Với IDENTITY: u' = u và Z' = d chính xác, nên landmark ra trùng landmark vào.
            """
            d = np.maximum(d, -0.8 * f)
            P = np.column_stack([u * ((f + d) / f)[:, None], d]) @ R.T
            z = np.maximum(P[:, 2], -0.8 * f * r)
            proj = f / (f * r + z)
            return P[:, :2] * proj[:, None], z - d, f / (f + d), proj

        pts = pose[t, :, :3].copy()
        for k in range(2):
            w = match[t, k]
            if w >= 0:
                pts[[w, *_POSE_HAND_PTS[w]], :2] += shift
        u, dz, _, _ = transform(pts[:, :2] - [px, py], (pts[:, 2] - pz) / POSE_Z_GAIN)
        pose[t, :, :2] = u + [px, py]
        pose[t, :, 2] = pz + (pts[:, 2] - pz + dz * POSE_Z_GAIN) / r

        for k in range(2):
            if match[t, k] < 0:
                continue
            src = hands[t, k]
            h = src[0] + (src - src[0]) * view.hand_size
            is_left = match[t, k] == _POSE_WRISTS[0]
            offsets = fingers[0 if is_left else 1]
            if offsets is not None:
                h = bend_fingers(h, offsets, is_left)
            h[:, :2] += shift
            proj_old = f / (f + max(wrist_d[t, k], -0.8 * f))
            d = wrist_d[t, k] + (h[:, 2] - h[0, 2]) / proj_old
            u, dz, _, proj = transform(h[:, :2] - [px, py], d)
            new = np.empty_like(h)
            new[:, :2] = u + [px, py]
            z_true = d + dz
            new[:, 2] = src[0, 2] + (z_true - z_true[0]) * proj[0]
            inside = lambda a: ((a[:, :2] >= -0.02) & (a[:, :2] <= 1.02)).all()
            if inside(src) and not inside(new):
                hands[t, k] = np.nan
                world[t, k] = np.nan
                continue
            hands[t, k] = new
            if np.isfinite(world[t, k]).all():
                w = world[t, k] if offsets is None else bend_fingers(world[t, k], offsets, is_left)
                world[t, k] = w @ R.T

    return pose.astype(np.float32), hands.astype(np.float32), world.astype(np.float32)


def features_from_landmarks(pose: np.ndarray, hands: np.ndarray, world: np.ndarray) -> np.ndarray:
    """(T, FEATURE_DIM) feature thô từ landmark của cả clip, đúng như ``extract`` trên kết quả MediaPipe."""
    out = np.empty((len(pose), FEATURE_DIM), dtype=np.float32)
    for t in range(len(pose)):
        out[t] = extract(*arrays_to_results(pose[t], hands[t], world[t]))
    return out
