"""
Xây dựng & huấn luyện bộ phân loại VSL từ chuỗi đặc trưng (30, feature_dim).

Thiết kế cho dữ liệu ÍT (vd. 100 lớp x ~15 mẫu):
  - Mô hình nhỏ, nhận biết nhóm đặc trưng (vị trí / thủ hình / hướng / hiện diện / chuyển động),
    conv đa tỉ lệ + BiGRU + attention pooling.
  - Huấn luyện tự điều chỉnh:
      * AdaptiveController: tăng/giảm độ mạnh augmentation, mixup, dropout theo khoảng cách
        train-val (overfit -> tăng, underfit -> giảm).
      * LR: warmup + cosine, tự giảm khi val loss đứng yên (plateau), AdamW tách weight decay.
      * EMA trọng số, early stopping, tự chọn batch size, tự cân trọng số lớp khi lệch.
  - Đánh giá trung thực bằng k-fold phân tầng (có hỗ trợ `groups` để tránh rò rỉ giữa các mẫu
    cùng 1 lần ghi), gộp thành ensemble + 1 model huấn luyện trên toàn bộ dữ liệu.
  - Hiệu chỉnh xác suất (temperature scaling từ OOF) + ngưỡng từ chối "không phải ký hiệu".
  - Xuất 1 file ONNX (đã gồm chuẩn hoá + ensemble) để chạy ONNX Runtime Web.

Dữ liệu vào: file .npz với
    X       (N, 30, D) float   -- đầu ra của process_sequence
    y       (N,)       int|str -- nhãn
    classes (C,)       tuỳ chọn -- tên lớp
    groups  (N,)       tuỳ chọn -- id nhóm (cùng video/lần ghi/người) để chia fold không rò rỉ

Ví dụ:
    python train_vsl.py --data dataset.npz --out runs/vsl
    python train_vsl.py --demo --out runs/demo --folds 3 --epochs 60      # chạy thử nhanh
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.extract import (FEATURE_DIM, FEATURE_SLICES, FEATURE_VERSION, LEFT_MIRROR, PRESENCE_SLICE,
                                  SEQ_FEATURE_DIM, SIDE_COLUMNS, STROKE_SLICE)


# =============================================================================
# CẤU HÌNH
# =============================================================================
@dataclass
class Config:
    # kiến trúc
    arch: str = "vslnet"            # "vslnet" | "handpair" | "former"
    drop_path: float = 0.1          # stochastic depth cho các lớp Transformer của "former"
    d_model: int = 128
    gru_hidden: int = 96
    kernels: tuple = (3, 5, 9)
    n_blocks: int = 2
    head: str = "linear"            # "linear" | "cosine"
    base_dropout: float = 0.2
    # tối ưu
    max_epochs: int = 200
    min_epochs: int = 30
    patience: int = 30              # early stopping (epoch)
    lr: float = 2e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    select_after_warmup: int | None = None  # Q3: chỉ chọn checkpoint từ epoch warmup+N (EMA còn chưa ổn trong warmup)
    lr_patience: int = 8            # plateau -> giảm LR
    label_smoothing: float = 0.1
    ema_decay: float = 0.99
    grad_clip: float = 1.0
    batch_size: int = 0             # 0 = tự chọn
    mixup_alpha: float = 0.0         # blending different signs can erase small distinctions
    hard_negative_margin: float = 0.8
    hard_negative_weight: float = 0.25
    # Mix an unrelated hand into a one-hand clip while keeping its label.
    extra_hand_probability: float = 0.4
    extra_hand_loss: float = 0.5
    hand_consistency: float = 0.12
    presence_scale: float = 0.25
    # Chuẩn hoá theo tay: thống kê chỉ trên frame có tay, tay vắng -> 0, cột presence -> cờ có hình học 0/1.
    presence_norm: bool = False
    # Tay phụ chèn vào: tính lại cột tay-tay từ vị trí thật thay vì gán 0.
    distractor_geometry: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.arch not in ("vslnet", "handpair", "former"):
            raise ValueError(f"arch phải là 'vslnet', 'handpair' hoặc 'former', nhận {self.arch!r}")
        if self.arch in ("handpair", "former") and not self.presence_norm:
            raise ValueError(f"arch={self.arch!r} cần presence_norm=True (cổng tay đọc cờ có hình học)")


# Presence là độ tin cậy mềm exp(-(d/0.35)^2) (0 = không có tay), không phải cờ 0/1.
# Coi một tay là "có" khi độ tin cậy >= 0.5 (khớp wrist trong ~0.29 vai).
PRESENCE_THRESHOLD = 0.5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DỮ LIỆU
# =============================================================================
def default_groups(d: int) -> list[tuple[int, int]]:
    """Nhóm cột theo bố cục FEATURE_SLICES (FEATURE_DIM hoặc SEQ_FEATURE_DIM). Khác -> 1 nhóm."""
    if d in (FEATURE_DIM, SEQ_FEATURE_DIM):
        S = FEATURE_SLICES   # vị trí | thủ hình | điểm bàn tay | độ cong ngón | hướng | hiện diện | chuyển động | nét
        b = [0, S["handshape"].start, S["hand_local"].start, S["finger_curl"].start, S["orientation"].start,
             S["presence"].start, FEATURE_DIM]
        if d == SEQ_FEATURE_DIM:
            b += [STROKE_SLICE.start, SEQ_FEATURE_DIM]   # nhóm riêng: group-drop vị trí buộc model dùng nét
    else:
        b = [0, d]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def load_dataset(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y_raw = d["y"]
    if X.ndim != 3:
        raise ValueError(f"X phải có shape (N, T, D), nhận {X.shape}")
    if np.isnan(X).any() or np.isinf(X).any():
        print("[cảnh báo] X chứa NaN/Inf -> thay bằng 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if "classes" in d.files:
        classes = [str(c) for c in d["classes"]]
        if y_raw.dtype.kind in "USO":
            lut = {c: i for i, c in enumerate(classes)}
            y = np.array([lut[str(v)] for v in y_raw], dtype=np.int64)
        else:
            y = y_raw.astype(np.int64)
    else:
        if y_raw.dtype.kind in "USO":
            classes, y = np.unique(y_raw.astype(str), return_inverse=True)
            classes = [str(c) for c in classes]
            y = y.astype(np.int64)
        else:
            y = y_raw.astype(np.int64)
            classes = [str(i) for i in range(int(y.max()) + 1)]
    groups = d["groups"] if "groups" in d.files else None
    return X, y, classes, groups


def make_demo_dataset(n_classes=20, per_class=15, T=30, D=SEQ_FEATURE_DIM, seed=0):
    """Dữ liệu giả để kiểm tra pipeline: mỗi lớp = 1 quỹ đạo đặc trưng riêng + nhiễu + lệch thời gian."""
    rng = np.random.default_rng(seed)
    knots = rng.normal(size=(n_classes, 6, D)).astype(np.float32)
    t_src = np.linspace(0, 1, 6)
    X, y = [], []
    for c in range(n_classes):
        for _ in range(per_class):
            shift = rng.uniform(-0.08, 0.08)
            t = np.clip(np.linspace(0, 1, T) + shift, 0, 1)
            seq = np.stack([np.interp(t, t_src, knots[c, :, j]) for j in range(D)], axis=1)
            seq = seq * rng.uniform(0.9, 1.1) + rng.normal(0, 0.6, size=seq.shape)
            X.append(seq.astype(np.float32))
            y.append(c)
    return np.stack(X), np.array(y, dtype=np.int64), [f"lop_{i}" for i in range(n_classes)], None


def make_folds(y: np.ndarray, k: int, groups, seed: int) -> np.ndarray:
    """Stratify while assigning every sample of a group to one fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y)
    g_all = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(g_all) != len(y):
        raise ValueError("groups must match the number of samples")
    fold_of = np.full(len(y), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(y)), y, g_all)):
        fold_of[validation] = fold
    return fold_of


def hand_feature_columns(feature_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(trái, phải, dùng chung) chỉ số cột cho bố cục FEATURE_DIM/SEQ_FEATURE_DIM; khác -> None.

    Cột dùng chung (vector/tiếp xúc tay-tay) chỉ có nghĩa khi cả hai tay có mặt.
    """
    if feature_dim not in (FEATURE_DIM, SEQ_FEATURE_DIM):
        return None
    return tuple(SIDE_COLUMNS[k][SIDE_COLUMNS[k] < feature_dim] for k in ("left", "right", "shared"))


_HAND_POS = FEATURE_SLICES["hand_pos"]


def hand_validity(x: torch.Tensor) -> torch.Tensor:
    """(…, T, D) đặc trưng THÔ -> (…, T, 2) cờ 0/1: tay trái/phải có hình học ở frame đó.

    process_sequence điền 0 cho tay vắng dài, nên hand_pos toàn 0 = vắng. Không dùng presence mềm
    (tay thật vẫn có thể ~0.02, vd. cảm ơn).
    """
    hp = torch.nan_to_num(x[..., _HAND_POS], nan=0.0).abs()
    half = hp.shape[-1] // 2
    return torch.stack([hp[..., :half].sum(-1) > 0, hp[..., half:].sum(-1) > 0], dim=-1).to(x.dtype)


def _membership(d: int, device=None) -> torch.Tensor | None:
    cols = hand_feature_columns(d)
    if cols is None:
        return None
    m = torch.zeros(3, d, device=device)
    for k, c in enumerate(cols):
        m[k, torch.as_tensor(c, device=device)] = 1.0
    return m


def normalize_features(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                       presence_aware: bool = False) -> torch.Tensor:
    """Đặc trưng thô -> z-score; dùng chung cho Normalizer, VSLEnsemble (predict) và ONNX.

    presence_aware: cột của tay vắng (và cột tay-tay khi thiếu một tay) -> 0; cột presence -> cờ 0/1.
    """
    z = torch.nan_to_num(((x - mean) / std).clamp(-8.0, 8.0), nan=0.0)   # NaN -> giá trị trung bình
    m = _membership(x.shape[-1], x.device) if presence_aware else None
    if m is None:
        return z
    v = hand_validity(x)
    vl, vr = v[..., :1], v[..., 1:]
    keep = (1.0 - m.sum(0)) + vl * m[0] + vr * m[1] + vl * vr * m[2]
    z = z * keep
    return torch.cat([z[..., :PRESENCE_SLICE.start], v, z[..., PRESENCE_SLICE.stop:]], dim=-1)


class Normalizer:
    """z-score theo cột đặc trưng. Luôn fit trên đúng tập train của từng model (không nhìn thấy val).

    presence_aware: mean/std cột của mỗi tay chỉ tính trên frame tay đó có mặt (không bị số 0 của
    tay vắng kéo lệch -> tay trái không còn bị khuếch đại).
    """

    def __init__(self, X: np.ndarray, presence_aware: bool = False):
        flat = X.reshape(-1, X.shape[-1])
        self.presence_aware = bool(presence_aware)
        self.mean = flat.mean(0).astype(np.float32)
        self.std = np.maximum(flat.std(0), 1e-3).astype(np.float32)
        cols = hand_feature_columns(X.shape[-1]) if presence_aware else None
        if cols is not None:
            v = hand_validity(torch.from_numpy(flat)).numpy().astype(bool)
            for rows, c in ((v[:, 0], cols[0]), (v[:, 1], cols[1]), (v[:, 0] & v[:, 1], cols[2])):
                if rows.sum() >= 2:
                    self.mean[c] = flat[rows][:, c].mean(0)
                    self.std[c] = np.maximum(flat[rows][:, c].std(0), 1e-3)
            self.mean[PRESENCE_SLICE], self.std[PRESENCE_SLICE] = 0.0, 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        z = normalize_features(torch.from_numpy(np.asarray(X, dtype=np.float32)),
                               torch.from_numpy(self.mean), torch.from_numpy(self.std), self.presence_aware)
        return z.numpy().astype(np.float32)


def _motion(src: torch.Tensor) -> torch.Tensor:
    """Giống bước 4 của process_sequence: làm mượt [1,2,1]/4 (biên lặp) rồi np.gradient * (T-1)."""
    T = src.shape[1]
    if T >= 3:
        pad = torch.cat([src[:, :1], src, src[:, -1:]], dim=1)
        src = 0.25 * pad[:, :-2] + 0.5 * pad[:, 1:-1] + 0.25 * pad[:, 2:]
    if T < 2:
        return torch.zeros_like(src)
    grad = torch.cat([src[:, 1:2] - src[:, :1], 0.5 * (src[:, 2:] - src[:, :-2]), src[:, -1:] - src[:, -2:-1]], 1)
    return grad * (T - 1)


def _inter_hand_raw(raw: torch.Tensor) -> torch.Tensor:
    """(B,T,D) thô -> (B,T,10) cột dùng chung [inter_hand_vec(3), hand_contact(4), motion inter(3)].

    shortcut: tiếp xúc "cặp gần nhất" chỉ xét 3x3 điểm (palm/cái/trỏ) có trong hand_pos thay vì 6x6
    như extract(); đủ cho tay phụ ở xa, cần lưu đủ 6 điểm nếu muốn khớp tuyệt đối khi hai tay chạm.
    """
    hp = raw[..., _HAND_POS]
    lp, rp = hp[..., :9].unflatten(-1, (3, 3)), hp[..., 9:].unflatten(-1, (3, 3))
    inter = lp[..., 0, :] - rp[..., 0, :]
    d = torch.stack([(lp[..., i, :] - rp[..., i, :]).norm(dim=-1) for i in range(3)]
                    + [(lp[..., :, None, :] - rp[..., None, :, :]).norm(dim=-1).flatten(-2).amin(-1)], -1)
    return torch.cat([inter, torch.exp(-8.0 * d ** 2), _motion(inter)], dim=-1)


def add_distractor_hand_features(
    x: torch.Tensor, donor: torch.Tensor, side: torch.Tensor | int, norm_stats=None,
) -> torch.Tensor:
    """Chèn tay bên `side` của clip donor vào clip x (đều đã chuẩn hoá).

    norm_stats=(mean, std, presence_aware): tính lại cột tay-tay trong không gian thô từ vị trí hai tay
    (giống lúc chạy thật khi tay thứ 2 lọt khung). Không có norm_stats -> gán 0 (hành vi cũ).
    """
    layout = hand_feature_columns(x.shape[-1])
    if layout is None:
        return x
    shared = torch.as_tensor(layout[2], device=x.device)
    out = x.clone()
    if isinstance(side, int):
        side = torch.full((x.shape[0],), side, dtype=torch.long, device=x.device)
    for batch_index, selected in enumerate(side.tolist()):
        cols = torch.as_tensor(layout[selected], device=x.device)
        out[batch_index, :, cols] = donor[batch_index, :, cols]
        if norm_stats is None:
            out[batch_index, :, shared] = 0.0
    if norm_stats is None:
        return out
    mean, std, presence_aware = norm_stats
    if x.shape[-1] != SEQ_FEATURE_DIM:
        raise ValueError("Tính lại cột tay-tay cần đủ cột chuyển động (SEQ_FEATURE_DIM)")
    raw = out * std + mean
    if presence_aware:
        both = out[..., PRESENCE_SLICE].clamp(0, 1).round().prod(-1, keepdim=True)
    else:
        both = hand_validity(raw).prod(-1, keepdim=True)
    shared_raw = _inter_hand_raw(raw) * both          # thiếu một tay -> 0 như process_sequence
    z = ((shared_raw - mean[shared]) / std[shared]).clamp(-8.0, 8.0)
    out[..., shared] = z * both if presence_aware else z
    return out


# =============================================================================
# AUGMENTATION (trên tensor đã chuẩn hoá, (B, T, D))
# =============================================================================
class Augmenter:
    def __call__(self, x: torch.Tensor, s: float, extra_hand_probability: float = 0.0,
                 hand_presence: torch.Tensor | None = None,
                 donor_pool: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
                 norm_stats=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if s <= 0:
            return x, None
        B, T, D = x.shape
        dev = x.device
        # 1) Time-warp + crop ngẫu nhiên: mô phỏng người ký nhanh/chậm, cắt lệch đầu/cuối
        inc = (1.0 + 0.5 * s * torch.randn(B, T, device=dev)).clamp(min=0.2)
        cum = torch.cumsum(inc, dim=1)
        cum = (cum - cum[:, :1]) / (cum[:, -1:] - cum[:, :1] + 1e-6)          # [0,1] đơn điệu
        lo = torch.rand(B, 1, device=dev) * 0.12 * s
        hi = 1.0 - torch.rand(B, 1, device=dev) * 0.12 * s
        t_new = lo + (hi - lo) * cum
        grid = torch.stack([t_new * 2 - 1, torch.zeros_like(t_new)], dim=-1).unsqueeze(1)  # (B,1,T,2)
        img = x.permute(0, 2, 1).unsqueeze(2)                                  # (B,D,1,T)
        x = F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                          align_corners=True).squeeze(2).permute(0, 2, 1)
        # 2) Nhiễu Gaussian + co giãn biên độ theo cột (jitter landmark, khác người ký)
        x = x + torch.randn_like(x) * (0.08 * s)
        x = x * (1.0 + 0.08 * s * torch.randn(B, 1, D, device=dev))
        # 3) Che ngẫu nhiên vài frame (mất detect thoáng qua)
        mask = (torch.rand(B, T, 1, device=dev) < 0.06 * s).float()
        x = x * (1.0 - mask)

        augmented = None
        if extra_hand_probability > 0 and hand_presence is not None and donor_pool is not None:
            present = (hand_presence >= PRESENCE_THRESHOLD).float().mean(dim=1) > 0.5
            one_hand = present.sum(dim=1) == 1
            chosen = one_hand & (torch.rand(B, device=dev) < extra_hand_probability)
            for side, pool in enumerate(donor_pool):
                if pool is None or len(pool) == 0:
                    continue
                eligible = chosen & ~present[:, side]
                if not eligible.any():
                    continue
                selected = torch.where(eligible)[0]
                donors = pool[torch.randint(len(pool), (len(selected),), device=dev)]
                if augmented is None:
                    augmented = x.clone()
                augmented[selected] = add_distractor_hand_features(x[selected], donors, side, norm_stats)
        return x, augmented


# =============================================================================
# MÔ HÌNH
# =============================================================================
class GroupStem(nn.Module):
    """Mã hoá riêng từng nhóm đặc trưng rồi trộn; group-dropout khi train -> chịu được thiếu tay."""

    def __init__(self, groups, d_model, dropout, presence_scale=0.25):
        super().__init__()
        self.groups = list(groups)
        self.presence_scale = float(presence_scale)
        self.group_drop = 0.0
        self.enc = nn.ModuleList()
        total = 0
        for a, b in self.groups:
            h = int(min(64, max(8, round(0.5 * (b - a)))))
            self.enc.append(nn.Sequential(nn.Linear(b - a, h), nn.LayerNorm(h), nn.GELU()))
            total += h
        self.proj = nn.Sequential(nn.Linear(total, d_model), nn.LayerNorm(d_model),
                                  nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        outs = []
        for (a, b), encoder in zip(self.groups, self.enc):
            encoded = encoder(x[..., a:b])
            # Presence is useful for missing-data handling, but should not act
            # as a class identity shortcut (one-hand vs two-hand).
            if (a, b) == (PRESENCE_SLICE.start, PRESENCE_SLICE.stop):
                encoded = encoded * self.presence_scale
            outs.append(encoded)
        if self.training and self.group_drop > 0 and len(outs) > 1:
            keep = (torch.rand(x.shape[0], len(outs), device=x.device) >= self.group_drop).float()
            none = keep.sum(1) == 0
            n_none = int(none.sum())
            if n_none:
                keep[none, torch.randint(len(outs), (n_none,), device=x.device)] = 1.0
            outs = [o * keep[:, i, None, None] for i, o in enumerate(outs)]
        return self.proj(torch.cat(outs, dim=-1))


class MultiScaleBlock(nn.Module):
    """Conv depthwise đa kích thước kernel (bắt cử động nhanh + chậm) + pointwise + residual."""

    def __init__(self, d, kernels, dropout):
        super().__init__()
        self.dw = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2, groups=d) for k in kernels])
        self.pw = nn.Conv1d(d * len(kernels), d, 1)
        self.bn = nn.BatchNorm1d(d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, d, T)
        y = torch.cat([c(x) for c in self.dw], dim=1)
        return x + self.drop(self.act(self.bn(self.pw(y))))


class CosineHead(nn.Module):
    """Classifier cosine: ổn định hơn khi nhiều lớp, ít mẫu."""

    def __init__(self, d_in, n_classes, emb=128, scale=12.0):
        super().__init__()
        self.emb = nn.Linear(d_in, emb)
        self.W = nn.Parameter(torch.randn(n_classes, emb) * 0.02)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, z):
        return self.scale * F.normalize(self.emb(z), dim=-1) @ F.normalize(self.W, dim=-1).t()


class VSLNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, groups, cfg: Config):
        super().__init__()
        d, h = cfg.d_model, cfg.gru_hidden
        self.stem = GroupStem(groups, d, cfg.base_dropout, cfg.presence_scale)
        self.blocks = nn.Sequential(*[MultiScaleBlock(d, cfg.kernels, cfg.base_dropout)
                                      for _ in range(cfg.n_blocks)])
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * h, 1)
        self.norm = nn.LayerNorm(6 * h)
        self.drop = nn.Dropout(cfg.base_dropout)
        self.head = CosineHead(6 * h, n_classes) if cfg.head == "cosine" else nn.Linear(6 * h, n_classes)

    def pool(self, x):  # (B, T, D) -> vector gộp theo thời gian (B, 6h), trước norm/head
        h = self.stem(x)
        h = self.blocks(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        w = torch.softmax(self.attn(h), dim=1)
        return torch.cat([(w * h).sum(1), h.mean(1), h.amax(1)], dim=-1)

    def forward(self, x):  # (B, T, D) -> logits (B, C); chạy được với mọi T
        return self.head(self.drop(self.norm(self.pool(x))))


class HandPairNet(nn.Module):
    """Mã hoá hai tay bằng CÙNG một encoder (tay trái được soi gương về hệ tay phải).

    Tay vắng -> token học được (không phải "tay ở giữa vai"); nhánh tay-tay nhân với p_trái * p_phải,
    nên có/không có tay thứ 2 không còn là một cột mà model bám vào được. Cần presence_norm=True.
    """

    def __init__(self, in_dim: int, n_classes: int, cfg: Config, d_hand: int = 64, d_inter: int = 16,
                 d_model: int = 64, gru_hidden: int = 32):
        super().__init__()
        cols = hand_feature_columns(in_dim)
        if cols is None:
            raise ValueError(f"HandPairNet cần {FEATURE_DIM} hoặc {SEQ_FEATURE_DIM} cột, nhận {in_dim}")
        left, right, shared = cols
        n = len(left)                       # cột thô đứng trước cột chuyển động -> cắt [:n] vẫn hợp lệ
        perm, sign = LEFT_MIRROR[0][:n], LEFT_MIRROR[1][:n]
        left, sign = left[perm], sign[perm]
        keep_l, keep_r = left != PRESENCE_SLICE.start, right != PRESENCE_SLICE.start + 1
        self.register_buffer("left_idx", torch.as_tensor(left[keep_l]), persistent=False)
        self.register_buffer("left_sign", torch.as_tensor(sign[keep_l]), persistent=False)
        self.register_buffer("right_idx", torch.as_tensor(right[keep_r]), persistent=False)
        self.register_buffer("shared_idx", torch.as_tensor(shared), persistent=False)
        self.hand = nn.Linear(int(keep_r.sum()), d_hand, bias=False)
        self.side = nn.Parameter(torch.zeros(2, d_hand))
        self.hand_norm = nn.Sequential(nn.LayerNorm(d_hand), nn.GELU())
        self.absent = nn.Parameter(torch.zeros(2, d_hand))
        self.inter = nn.Sequential(nn.Linear(len(shared), d_inter), nn.LayerNorm(d_inter), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(2 * d_hand + d_inter, d_model), nn.LayerNorm(d_model),
                                  nn.GELU(), nn.Dropout(cfg.base_dropout))
        self.block = MultiScaleBlock(d_model, cfg.kernels, cfg.base_dropout)
        self.gru = nn.GRU(d_model, gru_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * gru_hidden, 1)
        self.norm = nn.LayerNorm(6 * gru_hidden)
        self.drop = nn.Dropout(cfg.base_dropout)
        self.head = (CosineHead(6 * gru_hidden, n_classes) if cfg.head == "cosine"
                     else nn.Linear(6 * gru_hidden, n_classes))

    def forward(self, x):  # (B, T, D) đã chuẩn hoá presence-aware -> logits (B, C)
        return self.head(self.drop(self.norm(self.pool(x))))

    def pool(self, x):
        p = x[..., PRESENCE_SLICE].clamp(0.0, 1.0)
        pl, pr = p[..., :1], p[..., 1:]
        hl = self.hand_norm(self.hand(x[..., self.left_idx] * self.left_sign) + self.side[0])
        hr = self.hand_norm(self.hand(x[..., self.right_idx]) + self.side[1])
        hl = pl * hl + (1 - pl) * self.absent[0]
        hr = pr * hr + (1 - pr) * self.absent[1]
        hi = self.inter(x[..., self.shared_idx]) * (pl * pr)
        h = self.proj(torch.cat([hl, hr, hi], dim=-1))
        h = self.block(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        w = torch.softmax(self.attn(h), dim=1)
        return torch.cat([(w * h).sum(1), h.mean(1), h.amax(1)], dim=-1)


class PreLNBlock(nn.Module):
    """Lớp Transformer pre-LN + stochastic depth (bỏ cả nhánh residual theo mẫu khi train)."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float, drop_path: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.drop_path = float(drop_path)

    def _path(self, y):
        if not self.training or self.drop_path <= 0:
            return y
        keep = (torch.rand(y.shape[0], *([1] * (y.dim() - 1)), device=y.device) >= self.drop_path)
        return y * keep.to(y.dtype) / (1.0 - self.drop_path)

    def forward(self, x):
        h = self.n1(x)
        x = x + self._path(self.drop(self.attn(h, h, h, need_weights=False)[0]))
        return x + self._path(self.drop(self.ff(self.n2(x))))


def _sinusoid(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-np.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe


class PointShapeEncoder(nn.Module):
    """20 điểm bàn tay (hệ trục lòng bàn tay) -> vector thủ hình: mỗi điểm là 1 token,
    attention giữa các điểm tự học quan hệ ngón (cái nằm dưới ngón nào, trỏ-giữa bắt chéo...)."""

    def __init__(self, d: int, dropout: float, n_points: int = 20):
        super().__init__()
        self.inp = nn.Linear(3, d)
        self.joint = nn.Parameter(torch.randn(n_points, d) * 0.02)
        self.block = PreLNBlock(d, 4, 2 * d, dropout)
        self.norm = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)

    def forward(self, pts):  # (N, 20, 3) -> (N, 2d)
        h = self.norm(self.block(self.inp(pts) + self.joint))
        w = torch.softmax(self.score(h), dim=1)
        return torch.cat([(w * h).sum(1), h.mean(1)], dim=-1)


class SignFormer(nn.Module):
    """Hai tay qua CÙNG encoder (tay trái soi gương), thủ hình đọc từ 20 điểm hand_local bằng attention,
    tay vắng -> token học được, nhánh tay-tay nhân p_trái*p_phải; thời gian: conv đa kernel + 2 lớp
    Transformer pre-LN (vị trí sinusoid -> chạy được với mọi T). Cần presence_norm=True."""

    def __init__(self, in_dim: int, n_classes: int, cfg: Config, d_point: int = 32, d_hand: int = 128,
                 d_inter: int = 16, d_model: int = 128, n_layers: int = 2, heads: int = 4):
        super().__init__()
        cols = hand_feature_columns(in_dim)
        if cols is None or "hand_local" not in FEATURE_SLICES:
            raise ValueError(f"SignFormer cần {FEATURE_DIM} hoặc {SEQ_FEATURE_DIM} cột có hand_local, nhận {in_dim}")
        left, right, shared = cols
        n = len(left)
        perm, sign = LEFT_MIRROR[0][:n], LEFT_MIRROR[1][:n]
        left, sign = left[perm], sign[perm]
        hl = FEATURE_SLICES["hand_local"]
        half = (hl.stop - hl.start) // 2
        local_l, local_r = np.arange(hl.start, hl.start + half), np.arange(hl.start + half, hl.stop)
        keep_l = (left != PRESENCE_SLICE.start) & ~np.isin(left, local_l)
        keep_r = (right != PRESENCE_SLICE.start + 1) & ~np.isin(right, local_r)
        for name, v in (("left_idx", left[keep_l]), ("left_sign", sign[keep_l]), ("right_idx", right[keep_r]),
                        ("local_l", local_l), ("local_r", local_r), ("shared_idx", shared)):
            self.register_buffer(name, torch.as_tensor(v), persistent=False)
        self.points = PointShapeEncoder(d_point, cfg.base_dropout)
        self.other = nn.Sequential(nn.Linear(int(keep_r.sum()), d_hand), nn.LayerNorm(d_hand), nn.GELU())
        self.fuse = nn.Linear(2 * d_point + d_hand, d_hand)
        self.side = nn.Parameter(torch.zeros(2, d_hand))
        self.hand_norm = nn.Sequential(nn.LayerNorm(d_hand), nn.GELU())
        self.absent = nn.Parameter(torch.zeros(2, d_hand))
        self.inter = nn.Sequential(nn.Linear(len(shared), d_inter), nn.LayerNorm(d_inter), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(2 * d_hand + d_inter, d_model), nn.LayerNorm(d_model),
                                  nn.GELU(), nn.Dropout(cfg.base_dropout))
        self.conv = MultiScaleBlock(d_model, cfg.kernels, cfg.base_dropout)
        self.layers = nn.ModuleList([PreLNBlock(d_model, heads, 2 * d_model, cfg.base_dropout,
                                                cfg.drop_path * (i + 1) / n_layers) for i in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(3 * d_model)
        self.drop = nn.Dropout(cfg.base_dropout)
        self.head = (CosineHead(3 * d_model, n_classes) if cfg.head == "cosine"
                     else nn.Linear(3 * d_model, n_classes))

    def forward(self, x):  # (B, T, D) đã chuẩn hoá presence-aware -> logits (B, C)
        return self.head(self.drop(self.norm(self.pool(x))))

    def pool(self, x):
        B, T, _ = x.shape
        p = x[..., PRESENCE_SLICE].clamp(0.0, 1.0).unsqueeze(-1)                 # (B,T,2,1)
        pts = torch.stack([x[..., self.local_l], x[..., self.local_r]], dim=2)   # (B,T,2,60)
        shape = self.points(pts.reshape(B * T * 2, -1, 3)).reshape(B, T, 2, -1)
        other = torch.stack([x[..., self.left_idx] * self.left_sign, x[..., self.right_idx]], dim=2)
        h = self.hand_norm(self.fuse(torch.cat([shape, self.other(other)], dim=-1)) + self.side)
        h = p * h + (1 - p) * self.absent                                        # (B,T,2,d_hand)
        hi = self.inter(x[..., self.shared_idx]) * (p[:, :, 0] * p[:, :, 1])
        z = self.proj(torch.cat([h[:, :, 0], h[:, :, 1], hi], dim=-1))
        z = self.conv(z.transpose(1, 2)).transpose(1, 2) + _sinusoid(T, z.shape[-1], x.device)
        for layer in self.layers:
            z = layer(z)
        z = self.out_norm(z)
        w = torch.softmax(self.attn(z), dim=1)
        return torch.cat([(w * z).sum(1), z.mean(1), z.amax(1)], dim=-1)


def build_model(in_dim: int, n_classes: int, groups, cfg: Config) -> nn.Module:
    if cfg.arch == "former":
        return SignFormer(in_dim, n_classes, cfg)
    if cfg.arch == "handpair":
        return HandPairNet(in_dim, n_classes, cfg)
    return VSLNet(in_dim, n_classes, groups, cfg)


def set_dropout(model: nn.Module, p: float):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)


# =============================================================================
# EMA + BỘ ĐIỀU KHIỂN THÍCH ỨNG
# =============================================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        d = min(self.decay, (1 + step) / (10 + step))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


class AdaptiveController:
    """
    Tự điều chỉnh mức điều hoà (regularization) bằng HAI tín hiệu:
      - gap  = acc(train sạch) - acc(val): đo overfit
      - aug_acc = acc trên batch train ĐÃ augment/dropout/mixup: đo "còn sức học không"
    Luật: aug_acc thấp (<0.75)         -> giảm s (điều hoà đã quá tay, model không fit nổi)
          gap lớn VÀ aug_acc còn >0.85 -> tăng s (overfit nhưng vẫn còn dư sức chịu thêm)
    Nhờ phanh aug_acc, s không thể bị đẩy lên cực đại chỉ vì train sạch ~100%.
    `s` thuộc [0.1, 1] điều khiển: biên độ augmentation, xác suất mixup, dropout, group-dropout.
    """

    def __init__(self, cfg: Config, init: float = 0.5, frozen: bool = False, period: int = 3):
        self.cfg, self.s, self.frozen, self.period = cfg, init, frozen, period
        self.hist: list[float] = []
        self.aug_hist: list[float] = []

    @property
    def dropout(self):
        return float(np.clip(self.cfg.base_dropout * (0.5 + self.s), 0.05, 0.5))

    @property
    def mixup_p(self):
        return 0.6 * self.s if self.cfg.mixup_alpha > 0 else 0.0

    @property
    def group_drop(self):
        return 0.12 * self.s

    def update(self, epoch: int, aug_acc: float, gap: float):
        self.hist.append(self.s)
        self.aug_hist.append(aug_acc)
        if self.frozen or epoch < 5 or (epoch + 1) % self.period:
            return
        recent = float(np.mean(self.aug_hist[-self.period:]))
        if recent < 0.75:
            self.s = max(0.1, self.s - 0.1)
        elif gap > 0.12 and recent > 0.85:
            self.s = min(1.0, self.s + 0.1)


# =============================================================================
# HUẤN LUYỆN
# =============================================================================
@torch.no_grad()
def logits_of(model: nn.Module, X: torch.Tensor, bs: int = 512) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def eval_metrics(logits: torch.Tensor, y: torch.Tensor):
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    k = min(5, logits.shape[1])
    top5 = (logits.topk(k, dim=1).indices == y[:, None]).any(1).float().mean().item()
    return loss, acc, top5


def hard_negative_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    """Push the true class beyond each sample's strongest competing class."""
    if logits.ndim != 2 or logits.shape[1] < 2 or target.shape != (logits.shape[0],):
        raise ValueError("Expected logits (batch, classes>=2) and target (batch,)")
    true_logits = logits.gather(1, target[:, None]).squeeze(1)
    competing = logits.masked_fill(F.one_hot(target, logits.shape[1]).bool(), -torch.inf).max(dim=1).values
    return F.relu(margin + competing - true_logits).mean()


class ReplaySampler:
    """Chọn mẫu train cho từng epoch khi fine-tune nhanh (chỉ vài lớp có mẫu mới).

    Mỗi epoch: MỌI mẫu `always` (mẫu mới) + tối đa `quota[c]` mẫu khác của lớp c, rút ngẫu nhiên lại ở mỗi
    epoch (qua nhiều epoch vẫn ôn được phần lớn dữ liệu cũ). Số mẫu mỗi epoch cố định -> lịch LR ổn định.
    `train_one` bù trọng số lớp theo tỉ lệ (số mẫu lớp đó trong Xtr / số mẫu được rút) nên kỳ vọng loss vẫn
    bằng loss trên toàn bộ Xtr: lớp được ôn nhiều hơn không vì thế mà được model "thiên vị".
    """

    def __init__(self, y, quota, always=None, seed: int = 0):
        y = np.asarray(y)
        always = np.zeros(len(y), dtype=bool) if always is None else np.asarray(always, dtype=bool)
        if len(always) != len(y):
            raise ValueError("always phải cùng độ dài với y")
        self.always = np.where(always)[0]
        self.pools = [np.where((y == c) & ~always)[0] for c in range(len(quota))]
        self.take = [min(max(int(q), 0), len(pool)) for q, pool in zip(quota, self.pools)]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.always) + sum(self.take)

    def __call__(self, epoch: int) -> torch.Tensor:
        parts = [self.always] + [self.rng.choice(pool, k, replace=False)
                                 for pool, k in zip(self.pools, self.take) if k]
        idx = np.concatenate(parts)
        self.rng.shuffle(idx)
        return torch.from_numpy(idx.astype(np.int64))


def train_one(Xtr, ytr, Xva, yva, n_classes, groups, cfg: Config, device,
              fixed_epochs: int | None = None, init_strength: float = 0.5,
              freeze_strength: bool = False, tag: str = "", log_every: int = 5,
              hand_presence: torch.Tensor | None = None, norm_stats=None, init_state: dict | None = None,
              sampler=None):
    """Huấn luyện 1 model. Có (Xva,yva): early stopping + tự thích ứng. Không có: chạy `fixed_epochs`.

    norm_stats=(mean, std, presence_aware) tensor trên device: để augmentation tay phụ tính lại cột tay-tay.
    init_state: state_dict để fine-tune (warm-start) thay vì khởi tạo ngẫu nhiên. Khi có val, model ban đầu
    được chấm trước epoch 1 và là ứng viên "best" (best_epoch=0) -> fine-tune không thể trả về model tệ hơn
    trên val so với lúc bắt đầu.
    sampler: callable epoch -> chỉ số vào Xtr (số lượng cố định, vd. `ReplaySampler`): mỗi epoch chỉ train trên
    tập con đó thay vì toàn bộ Xtr (fine-tune nhanh). Batch size vẫn tính theo len(Xtr) như khi train đầy đủ.
    """
    has_val = Xva is not None
    model = build_model(Xtr.shape[-1], n_classes, groups, cfg).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    ema = EMA(model, cfg.ema_decay)
    aug = Augmenter()
    donor_pool = None
    if hand_presence is not None:
        donor_pool = tuple(
            Xtr[(hand_presence[:, :, side] >= PRESENCE_THRESHOLD).float().mean(dim=1) > 0.5]
            for side in (0, 1)
        )
    ctrl = AdaptiveController(cfg, init=init_strength, frozen=freeze_strength or not has_val)

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)

    n = len(Xtr)
    bs = cfg.batch_size or int(np.clip(n // 20, 16, 64))
    bs = min(bs, n)
    first = sampler(0).to(device) if sampler is not None else None
    n_epoch = len(first) if first is not None else n
    if n_epoch == 0:
        raise ValueError("sampler trả về tập rỗng")
    bs = min(bs, n_epoch)
    steps_per_epoch = max(1, n_epoch // bs)
    total_epochs = fixed_epochs or cfg.max_epochs
    total_steps = total_epochs * steps_per_epoch
    warm_steps = cfg.warmup_epochs * steps_per_epoch

    counts = torch.bincount(ytr, minlength=n_classes).float().clamp(min=1)
    cw = None
    if (counts.max() / counts.min()) > 1.5:       # lệch lớp -> trọng số căn bậc hai nghịch đảo
        cw = ((counts.sum() / (n_classes * counts)) ** 0.5).to(device)
    if first is not None:     # sampler rút lớp lệch so với Xtr -> bù để kỳ vọng loss = loss trên toàn bộ Xtr
        seen = torch.bincount(ytr[first], minlength=n_classes).float()
        iw = torch.where(seen > 0, counts / seen.clamp(min=1), torch.zeros_like(seen))
        iw = (iw / iw[seen > 0].mean()).to(device)
        cw = iw if cw is None else cw * iw

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"key": (-1.0, 1e9), "state": None, "epoch": 0, "m": None, "logits": None}
    best_loss, plateau, step = 1e9, 0, 0
    history: list[dict] = []
    lr_scale, lr_target = 1.0, 1.0      # lr_scale tiến mượt về lr_target (không "vách đứng")
    sub = torch.randperm(n, device=device)[:min(n, 512)]   # tập con cố định để đo train-acc sạch (rẻ)
    Xsub, ysub = Xtr[sub], ytr[sub]
    if init_state is not None and has_val:                  # điểm xuất phát của fine-tune = mốc để so
        va_logits = logits_of(ema.module, Xva)
        m0 = eval_metrics(va_logits, yva)
        best.update(key=(m0[1], m0[0]), state=copy.deepcopy(ema.module.state_dict()), epoch=-1, m=m0,
                    logits=va_logits.cpu())
        best_loss = m0[0]
        if log_every:
            print(f"  {tag} ep   0 | (model ban đầu) val_loss {m0[0]:.3f} acc {m0[1]:.3f} top5 {m0[2]:.3f}",
                  flush=True)
    t0 = time.time()

    for ep in range(total_epochs):
        model.train()
        set_dropout(model, ctrl.dropout)
        if hasattr(model, "stem"):
            model.stem.group_drop = ctrl.group_drop
        if sampler is None:
            perm = torch.randperm(n, device=device)
        else:
            perm = first if ep == 0 else sampler(ep).to(device)
        run_correct, run_total = torch.zeros((), device=device), 0   # acc ngay trong lúc train (miễn phí)

        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            x, y = Xtr[idx], ytr[idx]
            batch_presence = hand_presence[idx] if hand_presence is not None else None
            x, x_extra_hand = aug(
                x, ctrl.s, cfg.extra_hand_probability, batch_presence, donor_pool, norm_stats
            )

            y2, lam = None, 1.0
            if x_extra_hand is None and random.random() < ctrl.mixup_p:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                j = torch.randperm(len(x), device=device)
                x = lam * x + (1 - lam) * x[j]
                y2 = y[j]

            lr_scale += 0.05 * (lr_target - lr_scale)
            if step < warm_steps:
                lr = cfg.lr * (step + 1) / warm_steps
            else:
                prog = (step - warm_steps) / max(1, total_steps - warm_steps)
                lr = (cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))) * lr_scale
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                extra_out = model(x_extra_hand) if x_extra_hand is not None else None
            out = out.float()
            if extra_out is not None:
                extra_out = extra_out.float()
            with torch.no_grad():                    # đo acc theo nhãn trội (mixup: lam>=0.5 -> y, ngược lại y2)
                tgt = y if (y2 is None or lam >= 0.5) else y2
                run_correct = run_correct + (out.argmax(1) == tgt).sum()
                run_total += len(x)
            loss = F.cross_entropy(out, y, weight=cw, label_smoothing=cfg.label_smoothing)
            if y2 is not None:
                loss = lam * loss + (1 - lam) * F.cross_entropy(
                    out, y2, weight=cw, label_smoothing=cfg.label_smoothing)
            elif cfg.hard_negative_weight > 0:
                loss = loss + cfg.hard_negative_weight * hard_negative_margin_loss(
                    out, y, cfg.hard_negative_margin
                )
            if extra_out is not None:
                loss = loss + cfg.extra_hand_loss * F.cross_entropy(
                    extra_out, y, weight=cw, label_smoothing=cfg.label_smoothing
                )
            if extra_out is not None and cfg.hand_consistency > 0:
                teacher = torch.softmax(out.detach(), dim=-1)
                loss = loss + cfg.hand_consistency * F.kl_div(
                    F.log_softmax(extra_out, dim=-1), teacher, reduction="batchmean"
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model, step)
            step += 1

        # ---- đánh giá bằng EMA (sạch, không augmentation) ----
        aug_acc = (run_correct / max(1, run_total)).item()
        if has_val:
            tr_clean = eval_metrics(logits_of(ema.module, Xsub), ysub)[1]
            va_logits = logits_of(ema.module, Xva)
            va_loss, va_acc, va_top5 = eval_metrics(va_logits, yva)
            ctrl.update(ep, aug_acc, tr_clean - va_acc)
            history.append({"epoch": ep + 1, "train_aug_acc": aug_acc, "train_clean_acc": tr_clean,
                            "val_loss": va_loss, "val_acc": va_acc, "strength": ctrl.s})
            selectable = (cfg.select_after_warmup is None
                          or ep + 1 >= cfg.warmup_epochs + cfg.select_after_warmup)
            if selectable and (va_acc, -va_loss) > (best["key"][0], -best["key"][1]):   # acc cao hơn; hoà thì loss thấp hơn
                best.update(key=(va_acc, va_loss), state=copy.deepcopy(ema.module.state_dict()),
                            epoch=ep, m=(va_loss, va_acc, va_top5), logits=va_logits.cpu())
            if va_loss < best_loss - 1e-4:
                best_loss, plateau = va_loss, 0
            else:
                plateau += 1
                if plateau >= cfg.lr_patience:
                    lr_target, plateau = max(0.1, lr_target * 0.5), 0
            if log_every and ((ep + 1) % log_every == 0 or ep == 0):
                print(f"  {tag} ep {ep+1:3d} | train aug {aug_acc:.3f} clean {tr_clean:.3f} | val_loss {va_loss:.3f} "
                      f"acc {va_acc:.3f} top5 {va_top5:.3f} | s={ctrl.s:.1f} lr_x={lr_scale:.2f} "
                      f"| best {best['m'][1] if best['m'] else float('nan'):.3f}@{best['epoch']+1} "
                      f"| {time.time()-t0:.0f}s", flush=True)
            if (best["state"] is not None and ep + 1 >= cfg.min_epochs
                    and ep - best["epoch"] >= cfg.patience):
                print(f"  {tag} early stop @ {ep+1}")
                break
        else:
            ctrl.hist.append(ctrl.s)
            if log_every and ((ep + 1) % (log_every * 4) == 0):
                print(f"  {tag} ep {ep+1:3d} | train aug {aug_acc:.3f} | s={ctrl.s:.1f}", flush=True)

    if has_val:
        ema.module.load_state_dict(best["state"])
        hist = ctrl.hist[:best["epoch"] + 1] or [ctrl.s]      # best_epoch=0 (giữ model ban đầu) -> s hiện tại
        info = {"best_epoch": best["epoch"] + 1, "val_loss": best["m"][0], "val_acc": best["m"][1],
                "val_top5": best["m"][2], "strength": float(np.mean(hist[-10:])), "val_logits": best["logits"],
                "history": history}
    else:
        # final_lr ~ min_lr nghĩa là đã đi hết pha cosine (không dừng giữa warmup)
        info = {"best_epoch": total_epochs, "strength": ctrl.s, "final_lr": lr}
    return ema.module.eval(), info


# =============================================================================
# ENSEMBLE, HIỆU CHỈNH, XUẤT MODEL
# =============================================================================
def logits_and_embedding(model: nn.Module, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Một lượt chạy model -> (logits (B, C), embedding chuẩn hoá L2 (B, E)).

    Embedding là không gian head dùng để phân lớp: head cosine -> vector ``emb`` của nó (so với tâm lớp
    bằng cosine), head tuyến tính -> vector sau LayerNorm. Dùng cho bộ nhớ mẫu cá nhân (src/model/memory.py).
    """
    h = model.norm(model.pool(z))
    logits = model.head(model.drop(h))
    emb = model.head.emb(h) if isinstance(model.head, CosineHead) else h
    return logits, F.normalize(emb, dim=-1)


class VSLEnsemble(nn.Module):
    """
    Nhận đặc trưng THÔ (B, T, D) -> xác suất (B, C). Đã gồm chuẩn hoá + xử lý NaN + ensemble.
    Mỗi model giữ mean/std RIÊNG (tính chỉ trên tập train của fold đó) -> không rò rỉ khi CV.
    means, stds: (K, D) với K = số model.
    """

    def __init__(self, models, means, stds, temperature=1.0, presence_aware: bool = False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.presence_aware = bool(presence_aware)
        self.register_buffer("means", torch.as_tensor(np.asarray(means), dtype=torch.float32))
        self.register_buffer("stds", torch.as_tensor(np.asarray(stds), dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x):
        probs = []
        for k, m in enumerate(self.models):
            z = normalize_features(x, self.means[k], self.stds[k], self.presence_aware)
            probs.append(torch.softmax(m(z) / self.temperature, dim=-1))
        return torch.stack(probs).mean(0)

    def logits_and_embeddings(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """x thô (B, T, D) -> (logits đã chia nhiệt độ (K, B, C), embedding (K, B, E)) của từng model."""
        logits, embs = [], []
        for k, m in enumerate(self.models):
            z = normalize_features(x, self.means[k], self.stds[k], self.presence_aware)
            lg, emb = logits_and_embedding(m, z)
            logits.append(lg / self.temperature)
            embs.append(emb)
        return torch.stack(logits), torch.stack(embs)


def full_model_epochs(best_epochs, cfg: Config) -> int:
    """Số epoch model toàn bộ dữ liệu: theo CV nhưng luôn >= warmup + 20 để thật sự đi qua pha cosine."""
    return max(int(np.median(best_epochs) * 1.15) + 1, cfg.warmup_epochs + 20)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, t_min: float = 1.0, t_max: float = 10.0) -> float:
    """
    Fit T bằng LBFGS với T luôn nằm trong [t_min, t_max] ngay trong lúc tối ưu
    (tham số hoá qua sigmoid, không có exp không giới hạn) + line search; lỗi/NaN -> T=1.
    Mặc định t_min=1: không bao giờ làm sắc xác suất (T<1 phá ngưỡng từ chối).
    """
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    lo, hi = math.log(t_min), math.log(t_max)
    u = torch.zeros(1, requires_grad=True)                     # sigmoid(0)=0.5 -> T ≈ 1.41

    def temp():
        return torch.exp(lo + (hi - lo) * torch.sigmoid(u))

    opt = torch.optim.LBFGS([u], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / temp(), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:  # noqa: BLE001
        return 1.0
    t = float(temp().detach())
    return t if math.isfinite(t) else 1.0


def calibrate_temperature(logits: torch.Tensor, y: torch.Tensor, t_min: float = 1.0,
                          t_max: float = 10.0) -> tuple[float, str]:
    """
    Temperature từ OOF logits -> (T, status), status thuộc fitted | at_bound | skipped_no_errors.
    OOF không có lỗi top-1 thì NLL giảm đơn điệu khi T -> 0 (không có tín hiệu hiệu chỉnh) -> giữ T=1.
    """
    if bool((logits.argmax(1) == y).all()):
        return 1.0, "skipped_no_errors"
    t = fit_temperature(logits, y, t_min, t_max)
    if t <= t_min * 1.01 or t >= t_max / 1.01:
        print(f"[cảnh báo] Temperature={t:.3f} chạm biên [{t_min}, {t_max}]: hiệu chỉnh bị cắt, "
              "cần dữ liệu âm/hiệu chỉnh riêng để tin được.")
        return t, "at_bound"
    return t, "fitted"


def choose_reject_threshold(probs: np.ndarray, y: np.ndarray, target_acc: float = 0.95):
    conf, pred = probs.max(1), probs.argmax(1)
    ok = pred == y
    table = []
    for thr in np.arange(0.2, 0.96, 0.05):
        sel = conf >= thr
        cov = float(sel.mean())
        acc = float(ok[sel].mean()) if sel.any() else 0.0
        table.append((round(float(thr), 2), cov, acc))
    chosen = 0.5
    for thr, cov, acc in table:
        if acc >= target_acc and cov >= 0.5:
            chosen = thr
            break
    return chosen, table


def save_ensemble(path, models, norms: list, cfg: Config, groups, classes, temperature, reject_thr, in_shape):
    torch.save({"cfg": asdict(cfg), "feature_version": FEATURE_VERSION, "groups": groups, "classes": classes,
                "means": np.stack([n.mean for n in norms]), "stds": np.stack([n.std for n in norms]),
                "temperature": temperature, "reject_threshold": reject_thr, "in_shape": in_shape,
                "state_dicts": [m.cpu().state_dict() for m in models]}, path)


def load_ensemble(path, device="cpu") -> tuple[VSLEnsemble, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = {**ck["cfg"], "kernels": tuple(ck["cfg"]["kernels"])}
    saved_cfg.setdefault("presence_scale", 1.0)
    cfg = Config(**saved_cfg)
    models = []
    for sd in ck["state_dicts"]:
        m = build_model(ck["in_shape"][-1], len(ck["classes"]), ck["groups"], cfg)   # thiếu "arch" -> vslnet
        m.load_state_dict(sd)
        models.append(m.eval())
    ens = VSLEnsemble(models, ck["means"], ck["stds"], ck["temperature"],
                      presence_aware=cfg.presence_norm).to(device).eval()
    return ens, ck


@torch.no_grad()
def predict_topk(ens: VSLEnsemble, X: np.ndarray, classes, k=5, reject_threshold=0.0):
    """X: (B,30,D) hoặc (30,D) -> list[list[(tên_lớp, xác_suất)]]; dưới ngưỡng -> ('unknown', conf) đứng đầu."""
    dev = next(ens.buffers()).device
    x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
    if x.ndim == 2:
        x = x[None]
    p = ens(x).cpu().numpy()
    out = []
    for row in p:
        top = np.argsort(-row)[:k]
        res = [(classes[i], float(row[i])) for i in top]
        if row[top[0]] < reject_threshold:
            res.insert(0, ("unknown", float(row[top[0]])))
        out.append(res)
    return out


def export_onnx(ens: VSLEnsemble, path: str, in_shape):
    ens = ens.cpu().eval()
    dummy = torch.zeros(1, *in_shape)
    kw = dict(input_names=["features"], output_names=["probs"], opset_version=17,
              dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}})
    try:
        torch.onnx.export(ens, dummy, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(ens, dummy, path, **kw)


# =============================================================================
# PHÂN TÍCH LỖI
# =============================================================================
def report_errors(oof_logits: np.ndarray, y: np.ndarray, classes, mask: np.ndarray, top=8):
    pred = oof_logits.argmax(1)
    per = []
    for c in range(len(classes)):
        m = mask & (y == c)
        if m.any():
            per.append((float((pred[m] == c).mean()), classes[c]))
    per.sort()
    print("  Lớp khó nhất (OOF acc):", ", ".join(f"{n}={a:.2f}" for a, n in per[:top]))
    conf: dict = {}
    for t, p in zip(y[mask], pred[mask]):
        if t != p:
            conf[(t, p)] = conf.get((t, p), 0) + 1
    worst = sorted(conf.items(), key=lambda kv: -kv[1])[:top]
    if worst:
        print("  Nhầm lẫn nhiều nhất:", ", ".join(f"{classes[t]}→{classes[p]} ({c})" for (t, p), c in worst))


def confusion_pairs(oof_logits: np.ndarray, y: np.ndarray, classes, mask: np.ndarray) -> list[dict]:
    """Count directed mistakes on original, held-out samples only."""
    pred = oof_logits[mask].argmax(axis=1)
    truth = y[mask]
    counts: dict[tuple[int, int], int] = {}
    for actual, predicted in zip(truth, pred):
        if actual != predicted:
            key = (int(actual), int(predicted))
            counts[key] = counts.get(key, 0) + 1
    return [
        {"actual": classes[actual], "predicted": classes[predicted], "count": count}
        for (actual, predicted), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def closest_rivals(probs: np.ndarray, y: np.ndarray, classes) -> list[dict]:
    """Find the strongest rival for each class on original OOF examples."""
    rivals = []
    for class_index, class_name in enumerate(classes):
        rows = probs[y == class_index]
        if len(rows) == 0:
            continue
        mean_scores = rows.mean(axis=0).copy()
        mean_scores[class_index] = -np.inf
        rival_index = int(np.argmax(mean_scores))
        rivals.append({
            "actual": class_name,
            "rival": classes[rival_index],
            "mean_rival_probability": float(mean_scores[rival_index]),
            "samples": int(len(rows)),
        })
    return sorted(rivals, key=lambda item: -item["mean_rival_probability"])


# =============================================================================
# CHẠY HUẤN LUYỆN (gọi từ notebook hoặc CLI)
# =============================================================================
def run_training(X, y, classes, groups=None, out_dir="runs/vsl", cfg: Config | None = None,
                 folds: int = 5, folds_to_run: int = 0, add_full: bool = True,
                 export_onnx_model: bool = True, device=None,
                 validation_mask: np.ndarray | None = None, publish: bool = True) -> dict:
    """
    Huấn luyện trên MỌI lớp trong (X, y, classes): k-fold -> hiệu chỉnh -> (tuỳ chọn) model toàn bộ
    dữ liệu -> lưu ensemble.pt / classes.json / metrics.json / vsl_classifier.onnx.
    publish: nếu out_dir nằm trực tiếp trong runs/ thì nó được đổi tên thành runs/best (out_dir trả về là đường dẫn mới) và chỉ giữ 5 run gần nhất
    (``checkpoints.publish_run``).
    Trả về dict kết quả (CV, lịch sử từng fold, OOF logits, đường dẫn).
    """
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = device or get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    classes = [str(c) for c in classes]
    N, T, D = X.shape
    C = len(classes)
    validation_mask = np.ones(N, dtype=bool) if validation_mask is None else np.asarray(validation_mask, dtype=bool)
    if validation_mask.shape != (N,):
        raise ValueError("validation_mask must have shape (N,)")
    if groups is None and not validation_mask.all():
        raise ValueError("groups are required when augmented samples are excluded from validation")
    if groups is not None:
        groups_array = np.asarray(groups)
        for group in np.unique(groups_array):
            members = groups_array == group
            if not validation_mask[members].any():
                raise ValueError(f"Group {group} has no original validation sample")
    if C < 2:
        raise ValueError("Cần ít nhất 2 lớp để huấn luyện.")
    layout = default_groups(D)
    per_class = np.bincount(y, minlength=C)
    print(f"Dữ liệu: N={N}, T={T}, D={D}, lớp={C}, mẫu/lớp: min {per_class.min()} max {per_class.max()} "
          f"| nhóm cột: {layout} | thiết bị: {device}")
    if per_class.min() < folds:
        print(f"[cảnh báo] có lớp chỉ {per_class.min()} mẫu < {folds} fold: lớp đó sẽ vắng trong tập train của một số fold.")

    yt = torch.from_numpy(y).to(device)

    def to_dev(norm: Normalizer, idx):
        return torch.from_numpy(norm(X[idx])).to(device)

    def presence_of(idx):
        if hand_feature_columns(D) is None:
            return None
        if cfg.distractor_geometry:              # cờ có hình học: presence mềm bỏ sót tay thật (cảm ơn)
            return hand_validity(torch.from_numpy(X[idx])).to(device)
        return torch.from_numpy(X[idx, :, PRESENCE_SLICE].copy()).to(device)

    def stats_of(norm: Normalizer):
        if not cfg.distractor_geometry:
            return None
        return (torch.from_numpy(norm.mean).to(device), torch.from_numpy(norm.std).to(device),
                norm.presence_aware)

    # ---------------- k-fold phân tầng ----------------
    k = max(2, folds)
    group_ids = np.arange(N) if groups is None else np.asarray(groups)
    for class_index in range(C):
        originals = (y == class_index) & validation_mask
        if len(np.unique(group_ids[originals])) < k:
            raise ValueError(f"Class {classes[class_index]} needs at least {k} independent source groups")
    original_folds = make_folds(y[validation_mask], k, group_ids[validation_mask], cfg.seed)
    fold_by_group = dict(zip(group_ids[validation_mask], original_folds))
    fold_of = np.asarray([fold_by_group[group] for group in group_ids], dtype=np.int64)
    run_folds = list(range(k))[: (folds_to_run or k)]
    oof = np.zeros((N, C), dtype=np.float32)
    oof_mask = np.zeros(N, dtype=bool)
    models, norms, infos = [], [], []
    for f in run_folds:
        va = np.where((fold_of == f) & validation_mask)[0]
        tr = np.where(fold_of != f)[0]
        if len(va) == 0:
            raise ValueError(f"Fold {f + 1} has no original validation samples")
        if len(np.unique(y[va])) != C or len(np.unique(y[tr])) != C:
            raise ValueError(f"Fold {f + 1} does not contain every class in train and validation")
        print(f"\n=== Fold {f+1}/{k}: train {len(tr)} | val {len(va)} ===")
        norm_f = Normalizer(X[tr], cfg.presence_norm)  # thống kê CHỈ từ tập train của fold này
        model, info = train_one(to_dev(norm_f, tr), yt[tr], to_dev(norm_f, va), yt[va],
                                C, layout, cfg, device, tag=f"[f{f+1}]",
                                hand_presence=presence_of(tr), norm_stats=stats_of(norm_f))
        print(f"  -> best ep {info['best_epoch']} | val acc {info['val_acc']:.3f} "
              f"top5 {info['val_top5']:.3f} loss {info['val_loss']:.3f} | s≈{info['strength']:.2f}")
        oof[va] = info["val_logits"].numpy()
        oof_mask[va] = True
        models.append(model)
        norms.append(norm_f)
        infos.append(info)

    accs = [i["val_acc"] for i in infos]
    print(f"\nCV acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
          f"top5: {np.mean([i['val_top5'] for i in infos]):.4f}")

    # ---------------- hiệu chỉnh + ngưỡng từ chối ----------------
    lo = torch.from_numpy(oof[oof_mask])
    yo = torch.from_numpy(y[oof_mask])
    temp, temp_status = calibrate_temperature(lo, yo)
    probs = torch.softmax(lo / temp, 1).numpy()
    thr, table = choose_reject_threshold(probs, y[oof_mask])
    print(f"Temperature={temp:.3f} ({temp_status}) | ngưỡng từ chối={thr:.2f}")
    print("  ngưỡng | coverage | acc trên phần giữ lại")
    for t, cov, acc in table[::3]:
        print(f"   {t:.2f}  |  {cov:.3f}   |  {acc:.3f}")
    report_errors(oof, y, classes, oof_mask)
    confused = confusion_pairs(oof, y, classes, oof_mask)
    rivals = closest_rivals(probs, y[oof_mask], classes)
    evaluation = None
    try:
        from src.model.evaluation import oof_report
    except ImportError:
        print("[cảnh báo] Chưa có src.model.evaluation -> bỏ qua metrics['evaluation']")
    else:
        try:                                            # OOF chỉ mẫu gốc, logits thô (T=1)
            evaluation = oof_report(oof[oof_mask], y[oof_mask], classes,
                                    None if groups is None else np.asarray(groups)[oof_mask])
        except Exception as e:  # noqa: BLE001
            print(f"[cảnh báo] oof_report lỗi: {e}")
            evaluation = {"error": str(e)}
    if rivals:
        print("  Đối thủ gần nhất:", ", ".join(
            f"{item['actual']}→{item['rival']} ({item['mean_rival_probability']:.1%})"
            for item in rivals[:8]
        ))

    # ---------------- model toàn bộ dữ liệu (số epoch & độ mạnh điều hoà lấy từ CV) ----------------
    if add_full:
        ep_full = full_model_epochs([i["best_epoch"] for i in infos], cfg)
        s_full = float(np.mean([i["strength"] for i in infos]))
        print(f"\n=== Model toàn bộ dữ liệu: {ep_full} epoch, s={s_full:.2f} ===")
        norm_all = Normalizer(X, cfg.presence_norm)    # model cuối: không có val nên không có gì để rò rỉ
        full, full_info = train_one(to_dev(norm_all, np.arange(N)), yt, None, None, C, layout, cfg, device,
                                    fixed_epochs=ep_full, init_strength=s_full, freeze_strength=True,
                                    tag="[full]", hand_presence=presence_of(np.arange(N)),
                                    norm_stats=stats_of(norm_all))
        models.append(full)
        norms.append(norm_all)

    # ---------------- lưu ----------------
    save_ensemble(out_dir / "ensemble.pt", models, norms, cfg, layout, classes, temp, thr, (T, D))
    (out_dir / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")
    # Group gốc -> fold: fine-tune sau này phải giữ đúng phép chia này (model fold f chưa từng thấy val fold f).
    (out_dir / "folds.json").write_text(json.dumps(
        {"k": k, "seed": cfg.seed, "fold_by_group": {str(g): int(f) for g, f in fold_by_group.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    metrics = {
        "cv_acc_mean": float(np.mean(accs)), "cv_acc_std": float(np.std(accs)),
        "fold_acc": accs, "best_epochs": [i["best_epoch"] for i in infos],
        "temperature": temp, "temperature_status": temp_status, "reject_threshold": thr,
        "n_models": len(models),
        "n_samples": int(N), "n_classes": int(C), "confusion_pairs": confused,
        "closest_rivals": rivals, "config": asdict(cfg)}
    if add_full:
        metrics["full_model"] = {"epochs": ep_full, "final_lr": float(full_info["final_lr"])}
    if evaluation is not None:
        metrics["evaluation"] = evaluation
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"\nĐã lưu: {out_dir/'ensemble.pt'}")

    onnx_path = None
    if export_onnx_model:
        try:
            ens = VSLEnsemble([m.cpu() for m in models], np.stack([n.mean for n in norms]),
                              np.stack([n.std for n in norms]), temp, presence_aware=cfg.presence_norm)
            onnx_path = out_dir / "vsl_classifier.onnx"
            export_onnx(ens, str(onnx_path), (T, D))
            print(f"Đã xuất ONNX: {onnx_path} (input 'features' (B,{T},{D}) thô -> 'probs')")
        except Exception as e:  # noqa: BLE001
            onnx_path = None
            print(f"[cảnh báo] Xuất ONNX thất bại: {e}\n  (pip install onnx; model .pt vẫn dùng được)")

    if publish:
        from src.model.checkpoints import publish_run
        out_dir = publish_run(out_dir) or out_dir          # thư mục run đã được đổi thành runs/best

    return {**metrics, "out_dir": str(out_dir), "onnx_path": str(onnx_path) if onnx_path else None,
            "fold_histories": [i["history"] for i in infos], "oof_logits": oof, "oof_mask": oof_mask,
            "classes": classes, "y": y}


def load_validation_mask(path: str, n: int, groups) -> np.ndarray | None:
    """
    validation_mask = ~augmented nếu .npz có cột 'augmented' (mẫu tăng cường không bao giờ dùng để
    validate/OOF). Có mẫu tăng cường mà thiếu groups (origin) -> từ chối: CV sẽ rò rỉ giữa bản gốc và bản sao.
    """
    d = np.load(path, allow_pickle=True)
    if "augmented" not in d.files:
        return None
    aug = np.asarray(d["augmented"]).astype(bool)
    if aug.shape != (n,):
        raise ValueError(f"'augmented' phải có shape ({n},), nhận {aug.shape} -> không dựng được validation_mask")
    if aug.any() and groups is None:
        raise ValueError(f"{int(aug.sum())} mẫu tăng cường nhưng .npz thiếu 'groups' (origin_sample_id) "
                         "-> từ chối huấn luyện để tránh rò rỉ CV")
    if aug.all():
        raise ValueError("mọi mẫu đều là tăng cường -> không còn mẫu gốc để validate")
    return ~aug


def main():
    ap = argparse.ArgumentParser(description="Train VSL classifier (30, feature_dim) -> class")
    ap.add_argument("--data", type=str, help=".npz gồm X, y [, classes, groups]")
    ap.add_argument("--demo", action="store_true", help="dùng dữ liệu giả để thử pipeline")
    ap.add_argument("--out", type=str, default="runs/vsl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds-to-run", type=int, default=0, help="chỉ chạy N fold đầu (0 = tất cả)")
    ap.add_argument("--no-full", action="store_true", help="không huấn luyện model trên toàn bộ dữ liệu")
    ap.add_argument("--epochs", type=int, default=Config.max_epochs)
    ap.add_argument("--head", choices=["linear", "cosine"], default=Config.head)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--arch", choices=["vslnet", "handpair", "former"], default=Config.arch,
                    help="handpair tự bật chuẩn hoá theo presence")
    args = ap.parse_args()

    cfg = Config(max_epochs=args.epochs, head=args.head, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
                 arch=args.arch, presence_norm=args.arch in ("handpair", "former"))
    if args.demo:
        X, y, classes, groups = make_demo_dataset()
    elif args.data:
        X, y, classes, groups = load_dataset(args.data)
    else:
        ap.error("cần --data hoặc --demo")
    validation_mask = load_validation_mask(args.data, len(y), groups) if args.data and not args.demo else None
    run_training(X, y, classes, groups, out_dir=args.out, cfg=cfg, folds=args.folds,
                 folds_to_run=args.folds_to_run, add_full=not args.no_full,
                 export_onnx_model=not args.no_onnx, validation_mask=validation_mask,
                 publish=not args.demo)   # dữ liệu giả không bao giờ thành runs/best


if __name__ == "__main__":
    main()
