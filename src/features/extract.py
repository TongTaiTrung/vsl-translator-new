"""
Trích xuất đặc trưng VSL cho bài toán phân loại cửa sổ 30 frame.

Tập trung 4 thành tố: Thủ hình (handshape), Vị trí (location),
Hướng bàn tay (palm orientation), Chuyển động (movement).
Không dùng thành tố phi thủ công.

Hàm vào/ra giữ nguyên:
    extract(pose_res, hand_res)            -> (FEATURE_DIM,)          mỗi frame
    process_sequence(raw_sequence, ...)    -> (target_len, SEQ_FEATURE_DIM)

Giả định đầu vào: frame đã qua frame_padding thành 720 x 720 nên toạ độ
chuẩn hoá x, y của MediaPipe cùng một tỉ lệ, không cần W/H.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from src.builtin.mediapipe_builtin.defs import HandLandmarkerResult, PoseLandmarkerResult

FEATURE_VERSION = 4

# =============================================================================
# CẤU HÌNH
# =============================================================================
Z_POSE = 0.5          # giảm chấn trục Z (nhiễu gấp ~4-5 lần X, Y)
TORSO_Z = 0.3         # giảm chấn thêm khi dựng hệ trục thân (tránh lắc yaw/pitch giả)
VIS_MIN = 0.3         # visibility tối thiểu để tin cổ tay / khuỷu tay của pose
HIP_VIS_MIN = 0.5     # visibility tối thiểu để dùng hông dựng trục sống lưng
MAX_MATCH_DIST = 0.8  # (đơn vị vai) cổ tay tay-detect cách cổ tay pose xa hơn -> bỏ tay đó
PRESENCE_TEMPERATURE = 0.35  # soft confidence cho ghép hand <-> pose wrist
USE_HAND_WORLD = True # thủ hình/hướng bàn tay lấy từ hand_world_landmarks (mét, chuẩn hơn).
                      # Nếu kết quả kém hơn khi A/B, đặt False để dùng landmark ảnh.
REST_Y = 0.9          # (đơn vị vai) tâm lòng bàn tay thấp hơn neo bụng -> tay đang nghỉ, bỏ qua

# =============================================================================
# BỐ CỤC VECTOR ĐẶC TRƯNG (theo thành tố)
# =============================================================================
_GROUPS = (
    # --- VỊ TRÍ (47) ---
    ("hand_pos", 18),        # mỗi tay: tâm lòng bàn tay, đầu ngón cái, đầu ngón trỏ (3 điểm x 3)
    ("inter_hand_vec", 3),   # tâm bàn tay trái - phải
    ("body_contact", 22),    # mỗi tay: độ gần 11 vùng cơ thể
    ("hand_contact", 4),     # tay-tay: lòng-lòng, cái-cái, trỏ-trỏ, cặp điểm gần nhất
    ("face_vec", 12),        # mỗi tay: tâm lòng bàn tay - mũi, đầu ngón trỏ - mũi (không phụ thuộc chiều cao)
    # --- THỦ HÌNH (196) ---
    ("handshape", 76),       # mỗi tay 38: 15 cos khớp, 5 độ duỗi, 4 độ xòe, 8 tiếp xúc đầu ngón,
                             #             3 vị trí ngón cái, 3 thứ tự đầu ngón
    ("hand_local", 120),     # mỗi tay 60: 20 điểm (bỏ cổ tay) trong hệ trục lòng bàn tay, đơn vị palm_size
    # --- HƯỚNG BÀN TAY (18) ---
    ("orientation", 18),     # mỗi tay 9: palm_dir, palm_normal, 3 góc cổ tay so với cẳng tay
    # --- CỜ HIỆN DIỆN (2) ---
    ("presence", 2),         # tay trái/phải có được detect hay không
)
_SL: dict[str, slice] = {}
_off = 0
for _name, _size in _GROUPS:
    _SL[_name] = slice(_off, _off + _size)
    _off += _size
FEATURE_DIM = _off  # 275

# Chuyển động (tính trong process_sequence): đạo hàm của vị trí tay + hướng bàn tay
_IDX = np.arange(FEATURE_DIM)
_MOTION_COLS = np.concatenate([_IDX[_SL[n]] for n in ("hand_pos", "inter_hand_vec", "orientation")])
MOTION_DIM = len(_MOTION_COLS)               # 39
SEQ_FEATURE_DIM = FEATURE_DIM + MOTION_DIM   # 314

# Bố cục công khai: mọi module khác suy ra chỉ số cột từ đây, không hard-code.
FEATURE_SLICES: dict[str, slice] = dict(_SL)
PRESENCE_SLICE = _SL["presence"]


def _side_raw_cols(side: int) -> np.ndarray:
    """Cột thô (trong FEATURE_DIM) chỉ thuộc tay trái (0) hoặc phải (1)."""
    cols = []
    for name in ("hand_pos", "body_contact", "face_vec", "handshape", "hand_local", "orientation", "presence"):
        sl = _SL[name]
        half = (sl.stop - sl.start) // 2
        cols.append(_IDX[sl.start + side * half: sl.start + (side + 1) * half])
    return np.concatenate(cols)


_SHARED_RAW = np.concatenate([_IDX[_SL["inter_hand_vec"]], _IDX[_SL["hand_contact"]]])
_PALM_Y = (_SL["hand_pos"].start + 1, _SL["hand_pos"].start + 10)   # y tâm lòng bàn tay trái/phải


def _with_motion(raw_cols: np.ndarray) -> np.ndarray:
    """Thêm cột chuyển động (trong SEQ_FEATURE_DIM) sinh ra từ các cột thô đã cho."""
    motion = FEATURE_DIM + np.flatnonzero(np.isin(_MOTION_COLS, raw_cols))
    return np.concatenate([raw_cols, motion])


# Chỉ số cột trong chuỗi SEQ_FEATURE_DIM: riêng tay trái, riêng tay phải, dùng chung hai tay.
SIDE_COLUMNS: dict[str, np.ndarray] = {
    "left": _with_motion(_side_raw_cols(0)),
    "right": _with_motion(_side_raw_cols(1)),
    "shared": _with_motion(_SHARED_RAW),
}


def _left_mirror() -> tuple[np.ndarray, np.ndarray]:
    """(perm, sign) đưa cột tay trái (thứ tự SIDE_COLUMNS["left"]) về hệ của tay phải qua gương trục x:
    đảo dấu toạ độ x, palm_dir.x, palm_normal.x (normal trái đã lật), góc cổ tay ngang, face_vec.x;
    đổi chỗ neo má trái/phải và vai trái/phải. hand_local tự đối xứng gương nên giữ nguyên."""
    cols = SIDE_COLUMNS["left"]
    pos = {int(c): i for i, c in enumerate(cols)}
    hp, bc, ori = _SL["hand_pos"].start, _SL["body_contact"].start, _SL["orientation"].start
    flip = [hp, hp + 3, hp + 6, ori, ori + 3, ori + 8]
    flip += [FEATURE_DIM + int(np.flatnonzero(_MOTION_COLS == c)[0]) for c in flip]
    fv = _SL["face_vec"].start
    flip += [fv, fv + 3]                        # face_vec không có cột chuyển động
    sign = np.ones(len(cols), dtype=np.float32)
    sign[[pos[c] for c in flip]] = -1.0
    perm = np.arange(len(cols))
    for a, b in ((4, 5), (9, 10)):              # má trái/phải, vai trái/phải trong _body_anchors
        i, j = pos[bc + a], pos[bc + b]
        perm[i], perm[j] = j, i
    return perm, sign


LEFT_MIRROR = _left_mirror()

# =============================================================================
# HẰNG SỐ CẤU TRÚC BÀN TAY (MediaPipe 21 điểm)
# =============================================================================
_TIPS = np.array([4, 8, 12, 16, 20])
_MCPS = np.array([1, 5, 9, 13, 17])  # thumb dùng CMC (1)
_PALM_IDX = np.array([0, 5, 9, 13, 17])
_PARENTS = np.array([0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 13, 14, 0, 17, 18])
_JOINTS = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
_CHILDS = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])
_PAIR_A = np.array([4, 4, 4, 4, 8, 12, 16, 4])
_PAIR_B = np.array([8, 12, 16, 20, 12, 16, 20, 6])

# Neo cơ thể (hệ quy chiếu thân, đơn vị = độ rộng vai). Thứ tự:
# trán, mũi, miệng, cằm, má trái, má phải, cổ, ngực, bụng, vai trái, vai phải
_ANCHOR_GAMMA = np.array([18, 18, 18, 18, 18, 18, 10, 6, 4, 8, 8], dtype=np.float32)


# =============================================================================
# TIỆN ÍCH
# =============================================================================
def _unit_vector(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + eps)


def _contact_kernel(dist: np.ndarray, gamma) -> np.ndarray:
    """exp(-gamma d^2): ~1 khi chạm, ~0 khi xa -> phóng đại thay đổi nhỏ ở gần, triệt jitter ở xa."""
    return np.exp(-gamma * (dist ** 2))


def _vis(lm) -> float:
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else float(v)


def _body_anchors(P: np.ndarray) -> np.ndarray:
    """11 neo cơ thể trong hệ quy chiếu thân (P = pose đã chuẩn hoá, gốc = giữa hai vai)."""
    nose = P[0]
    eye_mid = 0.5 * (P[2] + P[5])
    mouth = 0.5 * (P[9] + P[10])
    forehead = eye_mid + 0.8 * (eye_mid - mouth)
    chin = mouth + 1.2 * (mouth - nose)
    neck = 0.5 * chin                       # trung điểm cằm và gốc (giữa vai)
    cheek_l = 0.5 * (P[7] + mouth)
    cheek_r = 0.5 * (P[8] + mouth)
    chest = np.array([0.0, 0.35, 0.0])
    belly = np.array([0.0, 0.9, 0.0])
    sh_l = np.array([0.5, 0.0, 0.0])        # u_x hướng từ vai phải sang vai trái
    sh_r = np.array([-0.5, 0.0, 0.0])
    return np.stack([forehead, nose, mouth, chin, cheek_l, cheek_r,
                     neck, chest, belly, sh_l, sh_r]).astype(np.float32)


def _body_contacts(pts: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """pts (6,3): tâm lòng bàn tay + 5 đầu ngón. Trả (11,): độ gần lớn nhất của bất kỳ điểm nào."""
    d = np.linalg.norm(anchors[None, :, :] - pts[:, None, :], axis=-1)  # (6, 11)
    return _contact_kernel(d, _ANCHOR_GAMMA[None, :]).max(axis=0)


def _hand_hand_contacts(lp: np.ndarray, rp: np.ndarray) -> np.ndarray:
    """lp, rp (6,3) theo thứ tự [palm, thumb, index, middle, ring, pinky]."""
    d = np.array([
        np.linalg.norm(lp[0] - rp[0]),
        np.linalg.norm(lp[1] - rp[1]),
        np.linalg.norm(lp[2] - rp[2]),
        np.linalg.norm(lp[:, None, :] - rp[None, :, :], axis=-1).min(),
    ], dtype=np.float32)
    return _contact_kernel(d, 8.0)


def _assign_hands(hand_wrists: np.ndarray, pose_wrists: np.ndarray, sw: float) -> list:
    """
    Ghép tay detect được với cổ tay trái/phải của POSE bằng hình học (không tin nhãn handedness,
    vì nhãn phụ thuộc ảnh có lật gương hay không). Trả [idx_tay_cho_trái, idx_tay_cho_phải].
    """
    assign = [None, None]
    n = len(hand_wrists)
    if n == 0:
        return assign
    cost = np.linalg.norm(hand_wrists[:, None, :] - pose_wrists[None, :, :], axis=-1) / sw  # (n, 2)
    if n == 1:
        pairs = [(0, int(np.argmin(cost[0])))]
    else:
        straight = cost[0, 0] + cost[1, 1]
        crossed = cost[0, 1] + cost[1, 0]
        pairs = [(0, 0), (1, 1)] if straight <= crossed else [(0, 1), (1, 0)]
    for h, side in pairs:
        if cost[h, side] <= MAX_MATCH_DIST:
            assign[side] = h
    return assign


def _hand_match_confidence(hand_wrist: np.ndarray, pose_wrist: np.ndarray, shoulder_width: float) -> float:
    """Confidence of a hand/pose wrist match, used instead of a hard 0/1 flag."""
    distance = float(np.linalg.norm(hand_wrist - pose_wrist) / max(shoulder_width, 1e-6))
    if distance > MAX_MATCH_DIST:
        return 0.0
    return float(np.exp(-((distance / PRESENCE_TEMPERATURE) ** 2)))


def _hand_shape_and_orientation(
    local: np.ndarray,
    is_left: bool,
    forearm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    local: (21,3) bàn tay đã xoay vào hệ trục thân (chỉ dùng hiệu vector, bất biến tịnh tiến).
    Trả: handshape (38), orientation (9), hand_local (60).
    """
    palm_size = np.linalg.norm(local[9] - local[0]) + 1e-6

    # --- Hướng bàn tay (tính trước vì thứ tự ngón cần palm_dir/palm_normal) ---
    palm_dir = _unit_vector(local[9] - local[0])
    palm_normal = _unit_vector(np.cross(local[5] - local[0], local[17] - local[0]))
    if is_left:                      # đối xứng gương giữa hai tay
        palm_normal = -palm_normal
    palm_side = _unit_vector(np.cross(palm_dir, palm_normal))

    # --- Thủ hình ---
    v1 = _unit_vector(local[_JOINTS] - local[_PARENTS])
    v2 = _unit_vector(local[_CHILDS] - local[_JOINTS])
    joint_cos = np.sum(v1 * v2, axis=-1)                                   # 15

    tip_ext = np.linalg.norm(local[_TIPS] - local[0], axis=-1) / palm_size  # 5 (độ duỗi)

    f_dir = _unit_vector(local[_TIPS] - local[_MCPS])                       # hướng 5 ngón
    spread = np.sum(f_dir[:-1] * f_dir[1:], axis=-1)                        # 4 (độ xòe)

    pair_d = np.linalg.norm(local[_PAIR_A] - local[_PAIR_B], axis=-1) / palm_size
    intra = _contact_kernel(pair_d, 3.0)                                    # 8 (tiếp xúc đầu ngón)

    # Thứ tự ngón: trục ngang gốc trỏ (5) -> gốc út (17) tự đối xứng giữa hai tay.
    across = local[17] - local[5]
    across_len = np.linalg.norm(across) + 1e-6
    u_across = across / across_len
    palm_center = local[_PALM_IDX].mean(axis=0)
    thumb = np.array([
        np.dot(local[4] - local[5], u_across) / across_len,       # ngón cái nằm dưới ngón nào (t/n/m)
        np.dot(local[4] - palm_center, palm_normal) / palm_size,  # ngón cái trước/sau lòng bàn tay
        np.dot(local[4] - local[0], palm_dir) / palm_size,        # ngón cái cao/thấp dọc bàn tay
    ])
    tip_order = (local[[12, 16, 20]] - local[[8, 12, 16]]) @ u_across / across_len  # âm = bắt chéo (r)

    handshape = np.concatenate([joint_cos, tip_ext, spread, intra, thumb, tip_order])  # 38

    # Hệ trục lòng bàn tay trực chuẩn, đẳng biến gương (normal trái đã lật; không dùng tích có hướng
    # cho trục thứ 3 vì nó đổi dấu qua gương): e3 = normal, e2 = trục ngang (đã vuông góc normal),
    # e1 = palm_dir trừ hình chiếu lên e2, e3.
    e2 = u_across
    e1 = _unit_vector(palm_dir - np.dot(palm_dir, e2) * e2 - np.dot(palm_dir, palm_normal) * palm_normal)
    hand_local = ((local[1:] - local[0]) @ np.stack([e1, e2, palm_normal], axis=1) / palm_size).ravel()  # 60

    if forearm is None:
        wrist = np.full(3, np.nan)
    else:
        wrist = np.array([np.dot(forearm, palm_dir),      # thẳng/gập cổ tay
                          np.dot(forearm, palm_normal),   # gập/duỗi
                          np.dot(forearm, palm_side)])    # lắc ngang
    orientation = np.concatenate([palm_dir, palm_normal, wrist])            # 9

    return handshape.astype(np.float32), orientation.astype(np.float32), hand_local.astype(np.float32)


# =============================================================================
# TRÍCH XUẤT 1 FRAME
# =============================================================================
def extract(pose_res: PoseLandmarkerResult, hand_res: HandLandmarkerResult) -> np.ndarray:
    """
    Trích xuất FEATURE_DIM (=275) đặc trưng của 4 thành tố VSL trên 1 frame.
    Giá trị không xác định được trả NaN (process_sequence sẽ xử lý).
    """
    nan_out = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    if not pose_res or not pose_res.pose_landmarks:
        return nan_out

    pose_lms = pose_res.pose_landmarks[0]
    pose_pts = np.array([[lm.x, lm.y, lm.z] for lm in pose_lms], dtype=np.float32)  # (33, 3)
    pose_pts[:, 2] *= Z_POSE

    # ---- Hệ quy chiếu thân: gốc = giữa hai vai, đơn vị = độ rộng vai (2D) ----
    l_sh, r_sh = pose_pts[11], pose_pts[12]
    mid_sh = 0.5 * (l_sh + r_sh)
    shoulder_width = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    if shoulder_width < 0.02:  # pose hỏng / người quá nhỏ
        return nan_out

    tz = np.array([1.0, 1.0, TORSO_Z], dtype=np.float32)
    u_x = _unit_vector((l_sh - r_sh) * tz)

    mid_hip = 0.5 * (pose_pts[23] + pose_pts[24])
    hip_ok = min(_vis(pose_lms[23]), _vis(pose_lms[24])) > HIP_VIS_MIN
    if hip_ok and np.linalg.norm((mid_hip - mid_sh)[:2]) > 0.1 * shoulder_width:
        v_spine = (mid_hip - mid_sh) * tz
    else:
        v_spine = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    u_y = _unit_vector(v_spine - np.dot(v_spine, u_x) * u_x)
    u_z = _unit_vector(np.cross(u_x, u_y))
    rot = np.stack([u_x, u_y, u_z], axis=1).astype(np.float32)  # (3,3), det=+1

    def to_torso(pts: np.ndarray) -> np.ndarray:
        return ((pts - mid_sh) @ rot) / shoulder_width

    norm_pose = to_torso(pose_pts)
    anchors = _body_anchors(norm_pose)

    # ---- Ghép tay <-> trái/phải theo hình học với cổ tay pose ----
    hand_lms = list(hand_res.hand_landmarks[:2]) if (hand_res and hand_res.hand_landmarks) else []
    hand_world = getattr(hand_res, "hand_world_landmarks", None) if hand_res else None
    hand_wrists = np.array([[h[0].x, h[0].y] for h in hand_lms], dtype=np.float32).reshape(-1, 2)
    assign = _assign_hands(hand_wrists, pose_pts[[15, 16], :2], shoulder_width)

    sides = []  # mỗi phần tử: (pts(6,3), handshape(38), orient(9), present)
    for side in (0, 1):
        is_left = side == 0
        w_idx, e_idx = 15 + side, 13 + side
        wrist_ok = _vis(pose_lms[w_idx]) >= VIS_MIN

        forearm = None
        if wrist_ok and _vis(pose_lms[e_idx]) >= VIS_MIN:
            forearm = _unit_vector(norm_pose[w_idx] - norm_pose[e_idx])

        h_idx = assign[side]
        if h_idx is not None:
            img = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms[h_idx]], dtype=np.float32)

            # Vị trí toàn cục: x, y từ hand model (chính xác hơn), z = z cổ tay pose + z tương đối của tay
            glob = img.copy()
            glob[:, 2] = pose_pts[w_idx, 2] + img[:, 2] * Z_POSE
            loc = to_torso(glob)
            palm_center = loc[_PALM_IDX].mean(axis=0)
            pts = np.vstack([palm_center, loc[_TIPS]])  # [palm, thumb, index, middle, ring, pinky]

            # Hình dạng cục bộ: chỉ cần hướng -> xoay vào hệ thân, không cần tịnh tiến/tỉ lệ
            if USE_HAND_WORLD and hand_world and len(hand_world) > h_idx:
                shape_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_world[h_idx]], dtype=np.float32)
            else:
                shape_pts = img
            local = shape_pts @ rot

            shape, orient, hloc = _hand_shape_and_orientation(local, is_left, forearm)
            present = _hand_match_confidence(hand_wrists[h_idx], pose_pts[w_idx, :2], shoulder_width)
        else:
            # Pose wrist is not a detected hand; keep its geometry unknown.
            pts = np.full((6, 3), np.nan, dtype=np.float32)
            shape = np.full((_SL["handshape"].stop - _SL["handshape"].start) // 2, np.nan, dtype=np.float32)
            orient = np.full(9, np.nan, dtype=np.float32)
            hloc = np.full(60, np.nan, dtype=np.float32)
            present = 0.0
        sides.append((pts, shape, orient, hloc, present))

    (l_pts, l_shape, l_orient, l_loc, l_pres), (r_pts, r_shape, r_orient, r_loc, r_pres) = sides
    nose = norm_pose[0]

    # ---- VỊ TRÍ ----
    hand_pos = np.concatenate([l_pts[:3].ravel(), r_pts[:3].ravel()])          # 18
    inter_hand_vec = l_pts[0] - r_pts[0]                                       # 3
    body_contact = np.concatenate([_body_contacts(l_pts, anchors),
                                   _body_contacts(r_pts, anchors)])            # 22
    hand_contact = _hand_hand_contacts(l_pts, r_pts)                           # 4
    face_vec = np.concatenate([(l_pts[[0, 2]] - nose).ravel(),
                               (r_pts[[0, 2]] - nose).ravel()])                # 12

    # ---- THỦ HÌNH / HƯỚNG ----
    handshape = np.concatenate([l_shape, r_shape])                             # 76
    hand_local = np.concatenate([l_loc, r_loc])                                # 120
    orientation = np.concatenate([l_orient, r_orient])                         # 18
    presence = np.array([l_pres, r_pres], dtype=np.float32)                    # 2

    out = np.concatenate([hand_pos, inter_hand_vec, body_contact, hand_contact, face_vec,
                          handshape, hand_local, orientation, presence]).astype(np.float32)
    return out


# =============================================================================
# XỬ LÝ CHUỖI
# =============================================================================
def _fill_gaps(col: np.ndarray, max_gap: int) -> np.ndarray:
    """Nội suy các đoạn NaN ngắn (<= max_gap). Đoạn dài hơn -> 0.0 (đã có cờ presence báo vắng)."""
    nan = np.isnan(col)
    if not nan.any():
        return col
    if nan.all():
        return np.zeros_like(col)
    out = col.copy()
    n = len(col)
    i = 0
    while i < n:
        if not nan[i]:
            i += 1
            continue
        j = i
        while j < n and nan[j]:
            j += 1
        left = i - 1 if i > 0 else None
        right = j if j < n else None
        if j - i <= max_gap:
            if left is not None and right is not None:
                out[i:j] = np.interp(np.arange(i, j), [left, right], [col[left], col[right]])
            elif left is not None:
                out[i:j] = col[left]
            else:
                out[i:j] = col[right]
        else:
            out[i:j] = 0.0
        i = j
    return out


def _spike_scales() -> np.ndarray:
    s = np.full(FEATURE_DIM, 2.0, dtype=np.float32)   # góc/tiếp xúc/hướng: ngưỡng rộng hơn
    s[_SL["hand_pos"]] = 1.0
    s[_SL["inter_hand_vec"]] = 1.0
    s[_SL["face_vec"]] = 1.0
    s[_SL["presence"]] = np.inf                       # không khử gai cờ
    return s


def _drop_resting_hands(seq: np.ndarray) -> None:
    """Bỏ (in-place) tay có trung vị tâm lòng bàn tay thấp hơn REST_Y: tay buông/để bàn lọt khung.

    Đặt NaN cho mọi cột của bên đó (kể cả cột tay-tay dùng chung) và presence = 0,
    để model thấy đúng một clip một tay. Chạy chung cho train, reextract và predict.
    """
    for side in (0, 1):
        # Dùng mọi frame có hình học tay (presence mềm có thể ~0.02 với tay thật, vd. cảm ơn).
        palm_y = seq[:, _PALM_Y[side]]
        palm_y = palm_y[~np.isnan(palm_y)]
        if palm_y.size and np.median(palm_y) > REST_Y:
            seq[:, _side_raw_cols(side)] = np.nan
            seq[:, _SHARED_RAW] = np.nan
            seq[:, PRESENCE_SLICE.start + side] = 0.0


def active_presence(feature: np.ndarray) -> np.ndarray:
    """(2,) bool cho 1 frame thô: tay được detect (có hình học) và không ở vị trí nghỉ."""
    palm_y = np.asarray(feature, dtype=np.float32)[list(_PALM_Y)]
    return palm_y <= REST_Y          # NaN (không có tay) -> False


def process_sequence(
    raw_sequence: list[np.ndarray] | np.ndarray,
    target_len: int = 30,
    spike_threshold: float = 0.45,
    max_gap: int = 6,
    add_motion: bool = True,
) -> np.ndarray:
    """
    Đóng gói chuỗi (T, FEATURE_DIM) về (target_len, SEQ_FEATURE_DIM):
      1. Nội suy NaN (chỉ đoạn thiếu <= max_gap frame; dài hơn -> 0 + cờ presence).
      2. Khử gai lỗi 1-frame (chỉ khi 2 frame lân cận khớp nhau, giữ nguyên cử động thật).
      3. Co giãn thời gian về target_len.
      4. Thêm đặc trưng CHUYỂN ĐỘNG (vận tốc vị trí tay + hướng bàn tay) nếu add_motion.
    Các cột thủ hình giữ nguyên không làm mờ (để không mất rung/búng ngón).
    """
    seq = np.array(raw_sequence, dtype=np.float32)
    out_dim = SEQ_FEATURE_DIM if add_motion else FEATURE_DIM
    if seq.ndim != 2 or seq.shape[0] == 0:
        return np.zeros((target_len, out_dim), dtype=np.float32)
    num_frames = seq.shape[0]

    # Cờ presence: NaN (mất cả pose) -> 0, không nội suy
    pres = seq[:, _SL["presence"]]
    seq[:, _SL["presence"]] = np.nan_to_num(pres, nan=0.0)
    _drop_resting_hands(seq)

    # 1. Nội suy các đoạn NaN ngắn
    for col in range(FEATURE_DIM):
        seq[:, col] = _fill_gaps(seq[:, col], max_gap)

    # 2. Khử gai 1-frame
    if num_frames >= 3:
        thr = spike_threshold * _spike_scales()
        prev_diff = seq[1:-1] - seq[:-2]
        next_diff = seq[1:-1] - seq[2:]
        neighbor_gap = np.abs(seq[2:] - seq[:-2])
        mag = np.minimum(np.abs(prev_diff), np.abs(next_diff))
        is_spike = (prev_diff * next_diff > 0) & (mag > thr) & (neighbor_gap < 0.5 * mag)
        midpoint = 0.5 * (seq[:-2] + seq[2:])
        seq[1:-1] = np.where(is_spike, midpoint, seq[1:-1])

    # 3. Co giãn thời gian
    if num_frames != target_len:
        if num_frames == 1:
            seq = np.repeat(seq, target_len, axis=0)
        else:
            old_time = np.linspace(0.0, 1.0, num_frames)
            new_time = np.linspace(0.0, 1.0, target_len)
            seq = interp1d(old_time, seq, axis=0, kind="linear")(new_time).astype(np.float32)

    if not add_motion:
        return seq.astype(np.float32)

    # 4. Chuyển động: làm mượt nhẹ [1,2,1]/4 CHỈ trên cột nguồn, rồi sai phân trung tâm.
    #    Đơn vị: độ rộng vai trên toàn cửa sổ (độc lập tốc độ tuyệt đối của người ký).
    src = seq[:, _MOTION_COLS]
    if target_len >= 3:
        pad = np.pad(src, ((1, 1), (0, 0)), mode="edge")
        src = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    motion = (np.gradient(src, axis=0) * (target_len - 1)).astype(np.float32)

    return np.concatenate([seq, motion], axis=1).astype(np.float32)
