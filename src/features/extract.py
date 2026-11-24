"""
Trích xuất đặc trưng VSL cho bài toán phân loại cửa sổ 30 frame.

Tập trung 4 thành tố: Thủ hình (handshape), Vị trí (location),
Hướng bàn tay (palm orientation), Chuyển động (movement).
Không dùng thành tố phi thủ công.

Hàm vào/ra giữ nguyên:
    extract(pose_res, hand_res)            -> (FEATURE_DIM,)          mỗi frame
    process_sequence(raw_sequence, ...)    -> (target_len, SEQ_FEATURE_DIM)
    rotate_view_pitch(features, degrees)   -> cùng shape, mô phỏng camera cao/thấp hơn (dùng khi augment)

Giả định đầu vào: frame đã qua frame_padding thành FRAME_SIZE x FRAME_SIZE (1280) nên toạ độ
chuẩn hoá x, y của MediaPipe cùng một tỉ lệ, không cần W/H.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from src.builtin.mediapipe_builtin.defs import HandLandmarkerResult, PoseLandmarkerResult

FEATURE_VERSION = 7

# =============================================================================
# CẤU HÌNH
# =============================================================================
Z_POSE = 0.5          # giảm chấn trục Z (nhiễu gấp ~4-5 lần X, Y)
TORSO_Z = 0.3         # giảm chấn thêm z của đường vai khi dựng trục x của thân (tránh lắc yaw giả)
VIS_MIN = 0.3         # visibility tối thiểu để tin cổ tay / khuỷu tay của pose
# Pitch (camera đặt cao/thấp hơn người ký) KHÔNG được ước lượng từ landmark: đo trên 368 clip gốc, pitch thân
# suy từ z của pose (ảnh hay world) lệch theo lớp ký hiệu trong cùng một camera (vd. 22/11: "b" -20°, "o" -9°)
# nhiều hơn mức lệch giữa các session (~±5°), còn xoay theo nó làm nhiễu vị trí tay tăng 2-4 lần. Vì vậy trục dọc
# luôn là phương thẳng đứng của ảnh, và bất biến pitch được dạy qua augmentation (rotate_view_pitch).
PITCH_AUG_DEG = 12.0  # biên độ pitch ngẫu nhiên mặc định khi tăng cường dữ liệu (src/data/augment.py)
POSE_Z_GAIN = 3.35    # z ảnh của pose ≈ 3.35 x độ sâu thật (cùng đơn vị x): hồi quy với pose_world trên clip gốc,
                      # tương quan 0.95 (y khớp 1.04). rotate_view_pitch xoay trong thang độ sâu thật.
MAX_MATCH_DIST = 0.8  # (đơn vị vai) cổ tay tay-detect cách cổ tay pose xa hơn -> bỏ tay đó
PRESENCE_TEMPERATURE = 0.35  # soft confidence cho ghép hand <-> pose wrist
USE_HAND_WORLD = True # thủ hình/hướng bàn tay lấy từ hand_world_landmarks (mét, chuẩn hơn).
                      # Nếu kết quả kém hơn khi A/B, đặt False để dùng landmark ảnh.
REST_Y = 0.9          # (đơn vị vai) tâm lòng bàn tay thấp hơn neo bụng -> tay đang nghỉ, bỏ qua
CONTACT_Z = 0.25      # trọng số trục z khi đo tiếp xúc cơ thể: z pose của tay lệch ~1-2 vai và nhiễu
                      # gấp 4-5 lần x, y, nên nếu giữ nguyên thì tay đặt trên ngực vẫn tính là "xa"
STROKE_LAG = 5        # (frame của chuỗi đã co giãn, ~0.33 s ở cửa sổ 2 s) độ dài "nét" khi tìm cú di chuyển mạnh nhất.
                      # Dấu sắc/huyền/nặng là một cú vẩy chỉ ~4 frame ở 15 fps; đo trên clip gốc, phần còn lại là
                      # giữ tay tĩnh, nên vận tốc từng frame gần như không đóng góp và model học tắt theo tư thế.

# =============================================================================
# BỐ CỤC VECTOR ĐẶC TRƯNG (theo thành tố)
# =============================================================================
_GROUPS = (
    # --- VỊ TRÍ (49) ---
    ("hand_pos", 18),        # mỗi tay: tâm lòng bàn tay, đầu ngón cái, đầu ngón trỏ (3 điểm x 3)
    ("inter_hand_vec", 3),   # tâm bàn tay trái - phải
    ("body_contact", 22),    # mỗi tay: độ gần 11 vùng cơ thể (z giảm trọng số CONTACT_Z)
    ("hand_contact", 4),     # tay-tay: lòng-lòng, cái-cái, trỏ-trỏ, cặp điểm gần nhất
    ("face_vec", 12),        # mỗi tay: tâm lòng bàn tay - mũi, đầu ngón trỏ - mũi (không phụ thuộc chiều cao)
    ("hand_scale", 2),       # mỗi tay: cỡ lòng bàn tay trên ảnh / độ rộng vai -> độ sâu tay so với thân
                             # (ổn định hơn z của pose; bất biến khi xoay bàn tay và lật gương)
    # --- THỦ HÌNH (286) ---
    ("handshape", 76),       # mỗi tay 38: 15 cos khớp, 5 độ duỗi, 4 độ xòe, 8 tiếp xúc đầu ngón,
                             #             3 vị trí ngón cái, 3 thứ tự đầu ngón
    ("hand_local", 120),     # mỗi tay 60: 20 điểm (bỏ cổ tay) trong hệ trục lòng bàn tay, đơn vị palm_size
    ("finger_curl", 90),     # mỗi tay 45: độ cong từng ngón (v7), xem _finger_curl
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
FEATURE_DIM = _off  # 367

# Chuyển động (tính trong process_sequence): đạo hàm của vị trí tay + hướng bàn tay
_IDX = np.arange(FEATURE_DIM)
_MOTION_COLS = np.concatenate([_IDX[_SL[n]] for n in ("hand_pos", "inter_hand_vec", "orientation")])
MOTION_DIM = len(_MOTION_COLS)               # 39

# Nét (tính trong process_sequence): mỗi tay, độ dời của đầu ngón trỏ và tâm lòng bàn tay trên đoạn STROKE_LAG
# frame có độ dời đầu ngón trỏ lớn nhất trong cửa sổ, lặp lại ở mọi frame -> không phụ thuộc cú vẩy rơi vào frame nào.
_STROKE_SRC = [(_SL["hand_pos"].start + 9 * side + 6, _SL["hand_pos"].start + 9 * side) for side in (0, 1)]
STROKE_DIM = 12                              # 2 tay x (trỏ 3 + lòng bàn tay 3)
SEQ_FEATURE_DIM = FEATURE_DIM + MOTION_DIM + STROKE_DIM   # 418

# Bố cục công khai: mọi module khác suy ra chỉ số cột từ đây, không hard-code.
FEATURE_SLICES: dict[str, slice] = dict(_SL)
PRESENCE_SLICE = _SL["presence"]
MOTION_SLICE = slice(FEATURE_DIM, FEATURE_DIM + MOTION_DIM)
STROKE_SLICE = slice(MOTION_SLICE.stop, SEQ_FEATURE_DIM)


def _side_raw_cols(side: int) -> np.ndarray:
    """Cột thô (trong FEATURE_DIM) chỉ thuộc tay trái (0) hoặc phải (1)."""
    cols = []
    for name in ("hand_pos", "body_contact", "face_vec", "hand_scale", "handshape", "hand_local", "finger_curl",
                 "orientation", "presence"):
        sl = _SL[name]
        half = (sl.stop - sl.start) // 2
        cols.append(_IDX[sl.start + side * half: sl.start + (side + 1) * half])
    return np.concatenate(cols)


_SHARED_RAW = np.concatenate([_IDX[_SL["inter_hand_vec"]], _IDX[_SL["hand_contact"]]])
_PALM_Y = (_SL["hand_pos"].start + 1, _SL["hand_pos"].start + 10)   # y tâm lòng bàn tay trái/phải


def _with_motion(raw_cols: np.ndarray, side: int | None = None) -> np.ndarray:
    """Thêm cột chuyển động (trong SEQ_FEATURE_DIM) sinh ra từ các cột thô đã cho, và cột nét của tay `side`."""
    motion = FEATURE_DIM + np.flatnonzero(np.isin(_MOTION_COLS, raw_cols))
    half = STROKE_DIM // 2
    stroke = np.arange(0) if side is None else np.arange(STROKE_SLICE.start + side * half,
                                                         STROKE_SLICE.start + (side + 1) * half)
    return np.concatenate([raw_cols, motion, stroke])


# Chỉ số cột trong chuỗi SEQ_FEATURE_DIM: riêng tay trái, riêng tay phải, dùng chung hai tay.
# Thứ tự luôn là cột thô -> chuyển động -> nét (HandPairNet/SignFormer cắt [:n] cho input FEATURE_DIM).
SIDE_COLUMNS: dict[str, np.ndarray] = {
    "left": _with_motion(_side_raw_cols(0), 0),
    "right": _with_motion(_side_raw_cols(1), 1),
    "shared": _with_motion(_SHARED_RAW),
}


def _left_mirror() -> tuple[np.ndarray, np.ndarray]:
    """(perm, sign) đưa cột tay trái (thứ tự SIDE_COLUMNS["left"]) về hệ của tay phải qua gương trục x:
    đảo dấu toạ độ x, palm_dir.x, palm_normal.x (normal trái đã lật), góc cổ tay ngang, face_vec.x;
    đổi chỗ neo má trái/phải và vai trái/phải. hand_local, hand_scale, finger_curl (góc, tỉ số độ dài) tự đối xứng
    gương nên giữ nguyên."""
    cols = SIDE_COLUMNS["left"]
    pos = {int(c): i for i, c in enumerate(cols)}
    hp, bc, ori = _SL["hand_pos"].start, _SL["body_contact"].start, _SL["orientation"].start
    flip = [hp, hp + 3, hp + 6, ori, ori + 3, ori + 8]
    flip += [FEATURE_DIM + int(np.flatnonzero(_MOTION_COLS == c)[0]) for c in flip]
    fv = _SL["face_vec"].start
    flip += [fv, fv + 3]                        # face_vec không có cột chuyển động
    flip += [STROKE_SLICE.start, STROKE_SLICE.start + 3]   # nét trỏ.x, nét lòng bàn tay.x
    sign = np.ones(len(cols), dtype=np.float32)
    sign[[pos[c] for c in flip]] = -1.0
    perm = np.arange(len(cols))
    for a, b in ((4, 5), (9, 10)):              # má trái/phải, vai trái/phải trong _body_anchors
        i, j = pos[bc + a], pos[bc + b]
        perm[i], perm[j] = j, i
    return perm, sign


LEFT_MIRROR = _left_mirror()

# Cột đầu (x) của các vector 3D đổi theo góc nhìn. Vị trí: z đã nhân Z_POSE; hướng: vector đơn vị đẳng hướng.
_PITCH_POS = np.concatenate([_SL["hand_pos"].start + 3 * np.arange(6), [_SL["inter_hand_vec"].start],
                             _SL["face_vec"].start + 3 * np.arange(4)])
_PITCH_DIR = _SL["orientation"].start + np.array([0, 3, 9, 12])   # palm_dir, palm_normal mỗi tay


def rotate_view_pitch(features: np.ndarray, degrees: float) -> np.ndarray:
    """Mô phỏng camera đặt thấp hơn (degrees > 0, nhìn hất lên) hoặc cao hơn (< 0) trên feature THÔ (..., FEATURE_DIM).

    Xoay quanh trục x của thân (qua giữa hai vai): y' = y cos + z sin, z' = -y sin + z cos, với z hướng ra xa camera,
    nên điểm ở trước thân hiện cao hơn khi nhìn từ dưới. Vị trí (hand_pos, inter_hand_vec, face_vec) xoay trong thang
    độ sâu thật (z lưu = Z_POSE * POSE_Z_GAIN * độ sâu thật); palm_dir, palm_normal xoay trực tiếp. Bất biến, giữ nguyên: thủ hình,
    hand_local, finger_curl, hand_scale, góc cổ tay (tích vô hướng với cẳng tay cùng xoay), tiếp xúc, presence. body_contact chỉ xấp xỉ
    bất biến (neo ngực/bụng cố định, z giảm trọng số) nên cũng giữ nguyên. NaN giữ NaN.
    """
    out = np.array(features, dtype=np.float32, copy=True)
    if out.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Expected (..., {FEATURE_DIM}), got {out.shape}")
    a = np.radians(float(degrees))
    c, s = np.float32(np.cos(a)), np.float32(np.sin(a))
    for starts, z_scale in ((_PITCH_POS, Z_POSE * POSE_Z_GAIN), (_PITCH_DIR, 1.0)):
        y = out[..., starts + 1]
        z = out[..., starts + 2] / z_scale
        out[..., starts + 1] = y * c + z * s
        out[..., starts + 2] = (z * c - y * s) * z_scale
    return out

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
_PALM_BONES = np.array([[0, 5], [0, 9], [0, 13], [0, 17], [5, 17]])   # xương lòng bàn tay (gần như cứng)

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
    diff = anchors[None, :, :] - pts[:, None, :]
    diff[..., 2] *= CONTACT_Z
    d = np.linalg.norm(diff, axis=-1)  # (6, 11)
    return _contact_kernel(d, _ANCHOR_GAMMA[None, :]).max(axis=0)


def _hand_scale(img: np.ndarray, shoulder_width: float) -> float:
    """img (21,3): landmark ảnh của tay (z cùng thang với x). Trả cỡ lòng bàn tay / độ rộng vai.

    Lấy trung bình độ dài 3D 5 xương lòng bàn tay nên ít đổi khi xoay/gập bàn tay; tỉ số với vai
    tăng khi tay đưa lại gần camera hơn thân (vd. "m" giơ ra trước, "tôi" áp vào ngực).
    """
    bones = np.linalg.norm(img[_PALM_BONES[:, 0]] - img[_PALM_BONES[:, 1]], axis=-1)
    return float(bones.mean() / max(shoulder_width, 1e-6))


_FINGER_CHAINS = np.array([[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
                           [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]])   # cổ tay -> gốc -> ... -> đầu ngón
CURL_DIM = 45   # mỗi tay


def _curl_stats(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """pts (21,3) -> (góc gập 15 khớp [rad], tổng gập mỗi ngón 5, gập PIP+DIP mỗi ngón 5, độ thẳng mỗi ngón 5).

    Góc (rad) tuyến tính theo độ gập, khác cos (cos 16° = 0.96, cos 32° = 0.85: phần gập nhẹ bị nén).
    Tổng theo ngón cộng 2-3 khớp nên lệch ngẫu nhiên từng khớp của MediaPipe triệt bớt nhau.
    Độ thẳng = |đầu ngón - gốc ngón| / tổng độ dài 3 đốt: 1 = duỗi thẳng, nhỏ dần khi móc/cuộn.
    Mọi giá trị bất biến khi xoay, tịnh tiến, co giãn và soi gương.
    """
    bones = np.diff(pts[_FINGER_CHAINS], axis=1)                          # (5, 4, 3)
    lengths = np.linalg.norm(bones, axis=-1)                              # (5, 4)
    a, b = bones[:, :-1].astype(np.float64), bones[:, 1:].astype(np.float64)
    # atan2(|a x b|, a.b) thay arccos(cos): arccos có đạo hàm vô hạn quanh 0° nên khuếch đại jitter của ngón duỗi.
    angles = np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))   # (5, 3)
    total = angles.sum(axis=1)
    hook = angles[:, 1:].sum(axis=1)
    chord = np.linalg.norm(pts[_FINGER_CHAINS[:, 4]] - pts[_FINGER_CHAINS[:, 1]], axis=-1)
    straight = chord / (lengths[:, 1:].sum(axis=1) + 1e-9)
    return angles.ravel(), total, hook, straight


def _finger_curl(img: np.ndarray, world: np.ndarray) -> np.ndarray:
    """(CURL_DIM,) độ cong ngón của một tay, đo trên CẢ landmark ảnh lẫn hand_world.

    [ảnh: góc 15 khớp, tổng gập 5, gập PIP+DIP 5, độ thẳng 5 | world: tổng gập 5, gập PIP+DIP 5, độ thẳng 5]

    Thủ hình cũ (cos khớp, hand_local) chỉ lấy từ hand_world, nhưng đo trên dữ liệu thật hand_world làm phẳng
    độ móc của ngón: "u" và dấu â/ô/ê (trỏ + giữa duỗi thẳng / móc xuống, PIP+DIP lệch ~40-90°), "i" và
    dấu ơ/ư (út duỗi / móc, ngón cái duỗi) tách xa gấp 2-3 lần khi đo trên landmark ảnh, kể cả khi giả lập
    đổi bối cảnh quay (src/features/view.py). Hai nguồn là hai đầu ra khác nhau của mạng nên giữ cả hai.
    z ảnh của tay cùng thang với x, y (frame vuông) nên góc 3D đo trực tiếp được.
    """
    i_ang, i_total, i_hook, i_straight = _curl_stats(img)
    _, w_total, w_hook, w_straight = _curl_stats(world)
    return np.concatenate([i_ang, i_total, i_hook, i_straight, w_total, w_hook, w_straight]).astype(np.float32)


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
    Trích xuất FEATURE_DIM (=367) đặc trưng của 4 thành tố VSL trên 1 frame.
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

    # Trục dọc = phương thẳng đứng của ảnh (chỉ khử roll theo đường vai). v5 dùng hông khi visibility > 0.5,
    # nhưng hông hầu như ngoài khung: nhánh đó chỉ chạy ở ~7% clip (26% frame của một session) và đổi định nghĩa
    # hệ trục giữa chừng clip, đưa vào một pitch giả ~-4° phụ thuộc session.
    v_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u_y = _unit_vector(v_up - np.dot(v_up, u_x) * u_x)
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

    sides = []  # mỗi phần tử: (pts(6,3), handshape(38), orient(9), hand_local(60), curl(45), present, scale)
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
            curl = _finger_curl(img, shape_pts)
            present = _hand_match_confidence(hand_wrists[h_idx], pose_pts[w_idx, :2], shoulder_width)
            scale = _hand_scale(img, shoulder_width)
        else:
            # Pose wrist is not a detected hand; keep its geometry unknown.
            pts = np.full((6, 3), np.nan, dtype=np.float32)
            shape = np.full((_SL["handshape"].stop - _SL["handshape"].start) // 2, np.nan, dtype=np.float32)
            orient = np.full(9, np.nan, dtype=np.float32)
            hloc = np.full(60, np.nan, dtype=np.float32)
            curl = np.full(CURL_DIM, np.nan, dtype=np.float32)
            present = 0.0
            scale = np.nan
        sides.append((pts, shape, orient, hloc, curl, present, scale))

    ((l_pts, l_shape, l_orient, l_loc, l_curl, l_pres, l_scale),
     (r_pts, r_shape, r_orient, r_loc, r_curl, r_pres, r_scale)) = sides
    nose = norm_pose[0]

    # ---- VỊ TRÍ ----
    hand_pos = np.concatenate([l_pts[:3].ravel(), r_pts[:3].ravel()])          # 18
    inter_hand_vec = l_pts[0] - r_pts[0]                                       # 3
    body_contact = np.concatenate([_body_contacts(l_pts, anchors),
                                   _body_contacts(r_pts, anchors)])            # 22
    hand_contact = _hand_hand_contacts(l_pts, r_pts)                           # 4
    face_vec = np.concatenate([(l_pts[[0, 2]] - nose).ravel(),
                               (r_pts[[0, 2]] - nose).ravel()])                # 12
    hand_scale = np.array([l_scale, r_scale], dtype=np.float32)                # 2

    # ---- THỦ HÌNH / HƯỚNG ----
    handshape = np.concatenate([l_shape, r_shape])                             # 76
    hand_local = np.concatenate([l_loc, r_loc])                                # 120
    finger_curl = np.concatenate([l_curl, r_curl])                             # 90
    orientation = np.concatenate([l_orient, r_orient])                         # 18
    presence = np.array([l_pres, r_pres], dtype=np.float32)                    # 2

    out = np.concatenate([hand_pos, inter_hand_vec, body_contact, hand_contact, face_vec, hand_scale,
                          handshape, hand_local, finger_curl, orientation, presence]).astype(np.float32)
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
    s[_SL["hand_scale"]] = 0.5                         # giá trị ~0.3-0.6
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


def _hand_valid(seq: np.ndarray) -> np.ndarray:
    """(T, 2) bool trên chuỗi đã lấp gap: tay có hình học (hand_pos khác 0) và không ở vị trí nghỉ."""
    hp = _SL["hand_pos"].start
    return np.stack([(np.abs(seq[:, hp + 9 * s: hp + 9 * (s + 1)]).sum(1) > 0) & (seq[:, _PALM_Y[s]] <= REST_Y)
                     for s in (0, 1)], axis=1)


def _stroke(pos: np.ndarray, valid: np.ndarray, lag: int = STROKE_LAG) -> np.ndarray:
    """(STROKE_DIM,) nét chính của mỗi tay: [Δ đầu trỏ (3), Δ tâm lòng bàn tay (3)] trên đoạn `lag` frame
    có |Δ đầu trỏ| lớn nhất mà cả đoạn tay đều hợp lệ. Không có đoạn nào -> 0.

    pos: (T, FEATURE_DIM) đã làm mượt; valid: (T, 2). Chỉ xét đoạn tay hợp lệ suốt đoạn nên tay vừa lọt khung
    (từ 0 nhảy lên vị trí thật) hay tay đang nâng từ dưới bụng lên không bị tính là một nét.
    """
    T = len(pos)
    lag = min(lag, T - 1)
    out = np.zeros(STROKE_DIM, dtype=np.float32)
    if lag < 1:
        return out
    half = STROKE_DIM // 2
    for side, (tip, palm) in enumerate(_STROKE_SRC):
        ok = np.lib.stride_tricks.sliding_window_view(valid[:, side], lag + 1).all(axis=1)   # (T - lag,)
        if not ok.any():
            continue
        d_tip = pos[lag:, tip:tip + 3] - pos[:-lag, tip:tip + 3]
        # Chọn đoạn theo xy: z pose của tay nhiễu (ký hiệu tĩnh |Δz| ~0.3 vai, |Δxy| chỉ ~0.05)
        score = np.where(ok, np.linalg.norm(d_tip * np.array([1.0, 1.0, CONTACT_Z]), axis=1), -1.0)
        t = int(np.argmax(score))
        out[side * half: side * half + 3] = d_tip[t]
        out[side * half + 3: (side + 1) * half] = pos[t + lag, palm:palm + 3] - pos[t, palm:palm + 3]
    return out


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
      4. Thêm đặc trưng CHUYỂN ĐỘNG (vận tốc vị trí tay + hướng bàn tay) và NÉT (cú di chuyển mạnh nhất
         của mỗi tay trong cửa sổ, lặp lại ở mọi frame) nếu add_motion.
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

    # 3. Co giãn thời gian (cờ hợp lệ của tay co giãn cùng: frame nội suy chạm frame không hợp lệ -> không hợp lệ)
    valid = _hand_valid(seq).astype(np.float32)
    if num_frames != target_len:
        if num_frames == 1:
            seq = np.repeat(seq, target_len, axis=0)
            valid = np.repeat(valid, target_len, axis=0)
        else:
            old_time = np.linspace(0.0, 1.0, num_frames)
            new_time = np.linspace(0.0, 1.0, target_len)
            seq = interp1d(old_time, seq, axis=0, kind="linear")(new_time).astype(np.float32)
            valid = interp1d(old_time, valid, axis=0, kind="linear")(new_time)
    valid = valid > 1.0 - 1e-6

    if not add_motion:
        return seq.astype(np.float32)

    # 4. Chuyển động: làm mượt nhẹ [1,2,1]/4 CHỈ trên cột nguồn, rồi sai phân trung tâm.
    #    Đơn vị: độ rộng vai trên toàn cửa sổ (độc lập tốc độ tuyệt đối của người ký).
    src = seq[:, _MOTION_COLS]
    if target_len >= 3:
        pad = np.pad(src, ((1, 1), (0, 0)), mode="edge")
        src = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    motion = (np.gradient(src, axis=0) * (target_len - 1)).astype(np.float32)

    # 5. Nét: trên vị trí đã làm mượt như bước 4 (cùng cột nguồn hand_pos), đơn vị độ rộng vai.
    #    Bộ làm mượt trộn frame kề nên frame sát đoạn vắng (giá trị 0) cũng không hợp lệ.
    pos = seq.copy()
    pos[:, _MOTION_COLS] = src
    if target_len >= 3:
        pv = np.pad(valid, ((1, 1), (0, 0)), mode="edge")
        valid = pv[:-2] & pv[1:-1] & pv[2:]
    stroke = np.broadcast_to(_stroke(pos, valid), (target_len, STROKE_DIM))

    return np.concatenate([seq, motion, stroke], axis=1).astype(np.float32)
