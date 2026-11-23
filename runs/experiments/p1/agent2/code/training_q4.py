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


# =============================================================================
# CẤU HÌNH
# =============================================================================
@dataclass
class Config:
    # kiến trúc
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
    seed: int = 42


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
    """Nhóm cột theo bố cục của vsl_feature_extractor (170 hoặc 131 chiều). Khác -> 1 nhóm."""
    if d == 170:
        b = [0, 47, 111, 129, 131, 170]   # vị trí | thủ hình | hướng | hiện diện | chuyển động
    elif d == 131:
        b = [0, 47, 111, 129, 131]
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


def make_demo_dataset(n_classes=20, per_class=15, T=30, D=170, seed=0):
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


class Normalizer:
    """z-score theo cột đặc trưng. Luôn fit trên đúng tập train của từng model (không nhìn thấy val)."""

    def __init__(self, X: np.ndarray):
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(0).astype(np.float32)
        self.std = np.maximum(flat.std(0), 1e-3).astype(np.float32)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) / self.std, -8, 8).astype(np.float32)


def hand_feature_slices(feature_dim: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]] | None:
    """Return side-specific feature ranges for the 131/170 feature layouts.

    The shared inter-hand/contact values are masked with the selected side too,
    since they are undefined when that hand is absent.
    """
    if feature_dim == 170:
        motion_offset = 131
    elif feature_dim == 131:
        motion_offset = None
    else:
        return None
    left = [(0, 9), (21, 32), (47, 79), (111, 120), (129, 130)]
    right = [(9, 18), (32, 43), (79, 111), (120, 129), (130, 131)]
    shared = [(18, 21), (43, 47)]
    if motion_offset is not None:
        left.extend(((motion_offset, motion_offset + 9), (motion_offset + 21, motion_offset + 30)))
        right.extend(((motion_offset + 9, motion_offset + 18), (motion_offset + 30, motion_offset + 39)))
        shared.append((motion_offset + 18, motion_offset + 21))
    return (tuple(left), tuple(right), tuple(shared)), (129, 131)


def add_distractor_hand_features(
    x: torch.Tensor, donor: torch.Tensor, side: torch.Tensor | int
) -> torch.Tensor:
    """Insert the opposite hand from a donor clip into normalized features."""
    layout = hand_feature_slices(x.shape[-1])
    if layout is None:
        return x
    side_ranges, _ = layout
    shared = side_ranges[2]
    out = x.clone()
    if isinstance(side, int):
        side = torch.full((x.shape[0],), side, dtype=torch.long, device=x.device)
    for batch_index, selected in enumerate(side.tolist()):
        for start, end in side_ranges[selected]:
            out[batch_index, :, start:end] = donor[batch_index, :, start:end]
        # Cross-hand geometry is not reconstructable from the feature vector;
        # leave it neutral rather than inventing a false interaction.
        for start, end in shared:
            out[batch_index, :, start:end] = 0.0
    return out


# =============================================================================
# AUGMENTATION (trên tensor đã chuẩn hoá, (B, T, D))
# =============================================================================
class Augmenter:
    def __call__(self, x: torch.Tensor, s: float, extra_hand_probability: float = 0.0,
                 hand_presence: torch.Tensor | None = None,
                 donor_pool: tuple[torch.Tensor | None, torch.Tensor | None] | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            present = (hand_presence >= 0.5).float().mean(dim=1) > 0.5
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
                augmented[selected] = add_distractor_hand_features(x[selected], donors, side)
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
            if (a, b) == (129, 131):
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

    def forward(self, x):  # (B, T, D) -> logits (B, C); chạy được với mọi T
        h = self.stem(x)
        h = self.blocks(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        w = torch.softmax(self.attn(h), dim=1)
        pooled = torch.cat([(w * h).sum(1), h.mean(1), h.amax(1)], dim=-1)
        return self.head(self.drop(self.norm(pooled)))


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


def train_one(Xtr, ytr, Xva, yva, n_classes, groups, cfg: Config, device,
              fixed_epochs: int | None = None, init_strength: float = 0.5,
              freeze_strength: bool = False, tag: str = "", log_every: int = 5,
              hand_presence: torch.Tensor | None = None):
    """Huấn luyện 1 model. Có (Xva,yva): early stopping + tự thích ứng. Không có: chạy `fixed_epochs`."""
    has_val = Xva is not None
    model = VSLNet(Xtr.shape[-1], n_classes, groups, cfg).to(device)
    ema = EMA(model, cfg.ema_decay)
    aug = Augmenter()
    donor_pool = None
    if hand_presence is not None:
        donor_pool = tuple(
            Xtr[(hand_presence[:, :, side] >= 0.5).float().mean(dim=1) > 0.5]
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
    steps_per_epoch = max(1, n // bs)
    total_epochs = fixed_epochs or cfg.max_epochs
    total_steps = total_epochs * steps_per_epoch
    warm_steps = cfg.warmup_epochs * steps_per_epoch

    counts = torch.bincount(ytr, minlength=n_classes).float().clamp(min=1)
    cw = None
    if (counts.max() / counts.min()) > 1.5:       # lệch lớp -> trọng số căn bậc hai nghịch đảo
        cw = ((counts.sum() / (n_classes * counts)) ** 0.5).to(device)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"key": (-1.0, 1e9), "state": None, "epoch": 0, "m": None, "logits": None}
    best_loss, plateau, step = 1e9, 0, 0
    history: list[dict] = []
    lr_scale, lr_target = 1.0, 1.0      # lr_scale tiến mượt về lr_target (không "vách đứng")
    sub = torch.randperm(n, device=device)[:min(n, 512)]   # tập con cố định để đo train-acc sạch (rẻ)
    Xsub, ysub = Xtr[sub], ytr[sub]
    t0 = time.time()

    for ep in range(total_epochs):
        model.train()
        set_dropout(model, ctrl.dropout)
        model.stem.group_drop = ctrl.group_drop
        perm = torch.randperm(n, device=device)
        run_correct, run_total = torch.zeros((), device=device), 0   # acc ngay trong lúc train (miễn phí)

        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            x, y = Xtr[idx], ytr[idx]
            batch_presence = hand_presence[idx] if hand_presence is not None else None
            x, x_extra_hand = aug(
                x, ctrl.s, cfg.extra_hand_probability, batch_presence, donor_pool
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

            if (va_acc, -va_loss) > (best["key"][0], -best["key"][1]):   # acc cao hơn; hoà thì loss thấp hơn
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
                      f"| best {best['m'][1]:.3f}@{best['epoch']+1} | {time.time()-t0:.0f}s", flush=True)
            if ep + 1 >= cfg.min_epochs and ep - best["epoch"] >= cfg.patience:
                print(f"  {tag} early stop @ {ep+1}")
                break
        else:
            ctrl.hist.append(ctrl.s)
            if log_every and ((ep + 1) % (log_every * 4) == 0):
                print(f"  {tag} ep {ep+1:3d} | train aug {aug_acc:.3f} | s={ctrl.s:.1f}", flush=True)

    if has_val:
        ema.module.load_state_dict(best["state"])
        hist = ctrl.hist[:best["epoch"] + 1]
        info = {"best_epoch": best["epoch"] + 1, "val_loss": best["m"][0], "val_acc": best["m"][1],
                "val_top5": best["m"][2], "strength": float(np.mean(hist[-10:])), "val_logits": best["logits"],
                "history": history}
    else:
        info = {"best_epoch": total_epochs, "strength": ctrl.s}
    return ema.module.eval(), info


# =============================================================================
# ENSEMBLE, HIỆU CHỈNH, XUẤT MODEL
# =============================================================================
class VSLEnsemble(nn.Module):
    """
    Nhận đặc trưng THÔ (B, T, D) -> xác suất (B, C). Đã gồm chuẩn hoá + xử lý NaN + ensemble.
    Mỗi model giữ mean/std RIÊNG (tính chỉ trên tập train của fold đó) -> không rò rỉ khi CV.
    means, stds: (K, D) với K = số model.
    """

    def __init__(self, models, means, stds, temperature=1.0):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("means", torch.as_tensor(np.asarray(means), dtype=torch.float32))
        self.register_buffer("stds", torch.as_tensor(np.asarray(stds), dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x):
        probs = []
        for k, m in enumerate(self.models):
            z = ((x - self.means[k]) / self.stds[k]).clamp(-8.0, 8.0)
            z = torch.where(torch.isnan(z), torch.zeros_like(z), z)   # NaN -> giá trị trung bình
            probs.append(torch.softmax(m(z) / self.temperature, dim=-1))
        return torch.stack(probs).mean(0)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, t_min: float = 0.2, t_max: float = 10.0) -> float:
    """
    Fit T bằng LBFGS với T luôn nằm trong [t_min, t_max] ngay trong lúc tối ưu
    (tham số hoá qua sigmoid, không có exp không giới hạn) + line search; lỗi/NaN -> T=1.
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
    torch.save({"cfg": asdict(cfg), "feature_version": 2, "groups": groups, "classes": classes,
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
        m = VSLNet(ck["in_shape"][-1], len(ck["classes"]), ck["groups"], cfg)
        m.load_state_dict(sd)
        models.append(m.eval())
    ens = VSLEnsemble(models, ck["means"], ck["stds"], ck["temperature"]).to(device).eval()
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
                 validation_mask: np.ndarray | None = None) -> dict:
    """
    Huấn luyện trên MỌI lớp trong (X, y, classes): k-fold -> hiệu chỉnh -> (tuỳ chọn) model toàn bộ
    dữ liệu -> lưu ensemble.pt / classes.json / metrics.json / vsl_classifier.onnx.
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
        if D not in (131, 170):
            return None
        return torch.from_numpy(X[idx, :, 129:131].copy()).to(device)

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
        norm_f = Normalizer(X[tr])                     # thống kê CHỈ từ tập train của fold này
        model, info = train_one(to_dev(norm_f, tr), yt[tr], to_dev(norm_f, va), yt[va],
                                C, layout, cfg, device, tag=f"[f{f+1}]",
                                hand_presence=presence_of(tr))
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
    temp = fit_temperature(lo, yo)
    probs = torch.softmax(lo / temp, 1).numpy()
    thr, table = choose_reject_threshold(probs, y[oof_mask])
    print(f"Temperature={temp:.3f} | ngưỡng từ chối={thr:.2f}")
    print("  ngưỡng | coverage | acc trên phần giữ lại")
    for t, cov, acc in table[::3]:
        print(f"   {t:.2f}  |  {cov:.3f}   |  {acc:.3f}")
    report_errors(oof, y, classes, oof_mask)
    confused = confusion_pairs(oof, y, classes, oof_mask)
    rivals = closest_rivals(probs, y[oof_mask], classes)
    if rivals:
        print("  Đối thủ gần nhất:", ", ".join(
            f"{item['actual']}→{item['rival']} ({item['mean_rival_probability']:.1%})"
            for item in rivals[:8]
        ))

    # ---------------- model toàn bộ dữ liệu (số epoch & độ mạnh điều hoà lấy từ CV) ----------------
    if add_full:
        ep_full = int(np.median([i["best_epoch"] for i in infos]) * 1.15) + 1
        s_full = float(np.mean([i["strength"] for i in infos]))
        print(f"\n=== Model toàn bộ dữ liệu: {ep_full} epoch, s={s_full:.2f} ===")
        norm_all = Normalizer(X)                       # model cuối: không có val nên không có gì để rò rỉ
        full, _ = train_one(to_dev(norm_all, np.arange(N)), yt, None, None, C, layout, cfg, device,
                            fixed_epochs=ep_full, init_strength=s_full, freeze_strength=True, tag="[full]",
                            hand_presence=presence_of(np.arange(N)))
        models.append(full)
        norms.append(norm_all)

    # ---------------- lưu ----------------
    save_ensemble(out_dir / "ensemble.pt", models, norms, cfg, layout, classes, temp, thr, (T, D))
    (out_dir / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")
    metrics = {
        "cv_acc_mean": float(np.mean(accs)), "cv_acc_std": float(np.std(accs)),
        "fold_acc": accs, "best_epochs": [i["best_epoch"] for i in infos],
        "temperature": temp, "reject_threshold": thr, "n_models": len(models),
        "n_samples": int(N), "n_classes": int(C), "confusion_pairs": confused,
        "closest_rivals": rivals, "config": asdict(cfg)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"\nĐã lưu: {out_dir/'ensemble.pt'}")

    onnx_path = None
    if export_onnx_model:
        try:
            ens = VSLEnsemble([m.cpu() for m in models], np.stack([n.mean for n in norms]),
                              np.stack([n.std for n in norms]), temp)
            onnx_path = out_dir / "vsl_classifier.onnx"
            export_onnx(ens, str(onnx_path), (T, D))
            print(f"Đã xuất ONNX: {onnx_path} (input 'features' (B,{T},{D}) thô -> 'probs')")
        except Exception as e:  # noqa: BLE001
            onnx_path = None
            print(f"[cảnh báo] Xuất ONNX thất bại: {e}\n  (pip install onnx; model .pt vẫn dùng được)")

    return {**metrics, "out_dir": str(out_dir), "onnx_path": str(onnx_path) if onnx_path else None,
            "fold_histories": [i["history"] for i in infos], "oof_logits": oof, "oof_mask": oof_mask,
            "classes": classes, "y": y}


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
    args = ap.parse_args()

    cfg = Config(max_epochs=args.epochs, head=args.head, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    if args.demo:
        X, y, classes, groups = make_demo_dataset()
    elif args.data:
        X, y, classes, groups = load_dataset(args.data)
    else:
        ap.error("cần --data hoặc --demo")
    run_training(X, y, classes, groups, out_dir=args.out, cfg=cfg, folds=args.folds,
                 folds_to_run=args.folds_to_run, add_full=not args.no_full,
                 export_onnx_model=not args.no_onnx)


if __name__ == "__main__":
    main()
