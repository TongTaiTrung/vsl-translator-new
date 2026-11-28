"""Model definitions and training for the VSL classifier (k-fold ensemble, calibration, rejection).

Usage: python -m src.model.training --data <file.npz> --out runs/vsl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.extract import (FEATURE_DIM, FEATURE_SLICES, FEATURE_VERSION, LEFT_MIRROR, PRESENCE_SLICE,
                                  SEQ_FEATURE_DIM, SIDE_COLUMNS, STROKE_SLICE, still_versions, stroke_magnitude)


@dataclass
class Config:
    arch: str = "vslnet"
    drop_path: float = 0.1
    d_model: int = 128
    gru_hidden: int = 96
    kernels: tuple = (3, 5, 9)
    n_blocks: int = 2
    head: str = "linear"
    base_dropout: float = 0.2
    max_epochs: int = 60
    min_epochs: int = 15
    patience: int = 10
    lr: float = 2e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    select_after_warmup: int | None = None
    lr_patience: int = 8
    label_smoothing: float = 0.1
    ema_decay: float = 0.99
    grad_clip: float = 1.0
    batch_size: int = 0
    mixup_alpha: float = 0.0
    hard_negative_margin: float = 0.8
    hard_negative_weight: float = 0.25
    extra_hand_probability: float = 0.4
    extra_hand_loss: float = 0.5
    hand_consistency: float = 0.12
    presence_scale: float = 0.25
    presence_norm: bool = False
    distractor_geometry: bool = False
    still_negative_weight: float = 0.5
    still_negative_fraction: float = 0.25
    dynamic_min_stroke: float = 0.25
    seed: int = 42

    def __post_init__(self):
        if self.arch not in ("vslnet", "handpair", "former"):
            raise ValueError(f"arch phải là 'vslnet', 'handpair' hoặc 'former', nhận {self.arch!r}")
        if self.arch in ("handpair", "former") and not self.presence_norm:
            raise ValueError(f"arch={self.arch!r} cần presence_norm=True (cổng tay đọc cờ có hình học)")


PRESENCE_THRESHOLD = 0.5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_groups(feature_dim: int) -> list[tuple[int, int]]:
    if feature_dim in (FEATURE_DIM, SEQ_FEATURE_DIM):
        slices = FEATURE_SLICES
        boundaries = [0, slices["handshape"].start, slices["hand_local"].start, slices["finger_curl"].start, slices["hand_closure"].start,
             slices["orientation"].start, slices["presence"].start, FEATURE_DIM]
        if feature_dim == SEQ_FEATURE_DIM:
            boundaries += [STROKE_SLICE.start, SEQ_FEATURE_DIM]
    else:
        boundaries = [0, feature_dim]
    return list(pairwise(boundaries))


def load_dataset(path: str):
    archive = np.load(path, allow_pickle=True)
    sequences = archive["X"].astype(np.float32)
    y_raw = archive["y"]
    if sequences.ndim != 3:
        raise ValueError(f"X phải có shape (N, T, D), nhận {sequences.shape}")
    if np.isnan(sequences).any() or np.isinf(sequences).any():
        print("[cảnh báo] X chứa NaN/Inf -> thay bằng 0")
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    if "classes" in archive.files:
        classes = [str(class_name) for class_name in archive["classes"]]
        if y_raw.dtype.kind in "USO":
            class_to_index = {class_name: class_index for class_index, class_name in enumerate(classes)}
            labels = np.array([class_to_index[str(value)] for value in y_raw], dtype=np.int64)
        else:
            labels = y_raw.astype(np.int64)
    else:
        if y_raw.dtype.kind in "USO":
            classes, labels = np.unique(y_raw.astype(str), return_inverse=True)
            classes = [str(class_name) for class_name in classes]
            labels = labels.astype(np.int64)
        else:
            labels = y_raw.astype(np.int64)
            classes = [str(class_index) for class_index in range(int(labels.max()) + 1)]
    groups = archive["groups"] if "groups" in archive.files else None
    return sequences, labels, classes, groups


def make_demo_dataset(n_classes=20, per_class=15, frame_count=30, feature_dim=SEQ_FEATURE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    knots = rng.normal(size=(n_classes, 6, feature_dim)).astype(np.float32)
    t_src = np.linspace(0, 1, 6)
    sequences, labels = [], []
    for class_index in range(n_classes):
        for _ in range(per_class):
            shift = rng.uniform(-0.08, 0.08)
            time_grid = np.clip(np.linspace(0, 1, frame_count) + shift, 0, 1)
            sequence = np.stack([np.interp(time_grid, t_src, knots[class_index, :, feature_index]) for feature_index in range(feature_dim)], axis=1)
            sequence = sequence * rng.uniform(0.9, 1.1) + rng.normal(0, 0.6, size=sequence.shape)
            sequences.append(sequence.astype(np.float32))
            labels.append(class_index)
    return np.stack(sequences), np.array(labels, dtype=np.int64), [f"lop_{name_index}" for name_index in range(n_classes)], None


def make_folds(labels: np.ndarray, fold_count: int, groups, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels)
    g_all = np.arange(len(labels)) if groups is None else np.asarray(groups)
    if len(g_all) != len(labels):
        raise ValueError("groups must match the number of samples")
    fold_of = np.full(len(labels), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(labels)), labels, g_all)):
        fold_of[validation] = fold
    return fold_of


def hand_feature_columns(feature_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if feature_dim not in (FEATURE_DIM, SEQ_FEATURE_DIM):
        return None
    return tuple(SIDE_COLUMNS[side][SIDE_COLUMNS[side] < feature_dim] for side in ("left", "right", "shared"))


_HAND_POS = FEATURE_SLICES["hand_pos"]


def hand_validity(features: torch.Tensor) -> torch.Tensor:
    hand_pos_values = torch.nan_to_num(features[..., _HAND_POS], nan=0.0).abs()
    half = hand_pos_values.shape[-1] // 2
    return torch.stack([hand_pos_values[..., :half].sum(-1) > 0, hand_pos_values[..., half:].sum(-1) > 0], dim=-1).to(features.dtype)


def _membership(feature_dim: int, device=None) -> torch.Tensor | None:
    cols = hand_feature_columns(feature_dim)
    if cols is None:
        return None
    membership = torch.zeros(3, feature_dim, device=device)
    for side_index, side_columns in enumerate(cols):
        membership[side_index, torch.as_tensor(side_columns, device=device)] = 1.0
    return membership


def normalize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                       presence_aware: bool = False) -> torch.Tensor:
    normalized = torch.nan_to_num(((features - mean) / std).clamp(-8.0, 8.0), nan=0.0)
    membership = _membership(features.shape[-1], features.device) if presence_aware else None
    if membership is None:
        return normalized
    validity = hand_validity(features)
    left_valid, right_valid = validity[..., :1], validity[..., 1:]
    keep = (1.0 - membership.sum(0)) + left_valid * membership[0] + right_valid * membership[1] + left_valid * right_valid * membership[2]
    normalized = normalized * keep
    return torch.cat([normalized[..., :PRESENCE_SLICE.start], validity, normalized[..., PRESENCE_SLICE.stop:]], dim=-1)


class Normalizer:

    def __init__(self, sequences: np.ndarray, presence_aware: bool = False):
        flat = sequences.reshape(-1, sequences.shape[-1])
        self.presence_aware = bool(presence_aware)
        self.mean = flat.mean(0).astype(np.float32)
        self.std = np.maximum(flat.std(0), 1e-3).astype(np.float32)
        cols = hand_feature_columns(sequences.shape[-1]) if presence_aware else None
        if cols is not None:
            validity = hand_validity(torch.from_numpy(flat)).numpy().astype(bool)
            for rows, side_columns in ((validity[:, 0], cols[0]), (validity[:, 1], cols[1]), (validity[:, 0] & validity[:, 1], cols[2])):
                if rows.sum() >= 2:
                    self.mean[side_columns] = flat[rows][:, side_columns].mean(0)
                    self.std[side_columns] = np.maximum(flat[rows][:, side_columns].std(0), 1e-3)
            self.mean[PRESENCE_SLICE], self.std[PRESENCE_SLICE] = 0.0, 1.0

    def __call__(self, sequences: np.ndarray) -> np.ndarray:
        normalized = normalize_features(torch.from_numpy(np.asarray(sequences, dtype=np.float32)),
                               torch.from_numpy(self.mean), torch.from_numpy(self.std), self.presence_aware)
        return normalized.numpy().astype(np.float32)


def _motion(values: torch.Tensor) -> torch.Tensor:
    frame_count = values.shape[1]
    if frame_count >= 3:
        padded = torch.cat([values[:, :1], values, values[:, -1:]], dim=1)
        values = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    if frame_count < 2:
        return torch.zeros_like(values)
    grad = torch.cat([values[:, 1:2] - values[:, :1], 0.5 * (values[:, 2:] - values[:, :-2]), values[:, -1:] - values[:, -2:-1]], 1)
    return grad * (frame_count - 1)


def _inter_hand_raw(raw: torch.Tensor) -> torch.Tensor:
    hand_pos_values = raw[..., _HAND_POS]
    left_points, right_points = hand_pos_values[..., :9].unflatten(-1, (3, 3)), hand_pos_values[..., 9:].unflatten(-1, (3, 3))
    inter = left_points[..., 0, :] - right_points[..., 0, :]
    distances = torch.stack([(left_points[..., pair_index, :] - right_points[..., pair_index, :]).norm(dim=-1) for pair_index in range(3)]
                    + [(left_points[..., :, None, :] - right_points[..., None, :, :]).norm(dim=-1).flatten(-2).amin(-1)], -1)
    return torch.cat([inter, torch.exp(-8.0 * distances ** 2), _motion(inter)], dim=-1)


def add_distractor_hand_features(
    features: torch.Tensor, donor: torch.Tensor, side: torch.Tensor | int, norm_stats=None,
) -> torch.Tensor:
    layout = hand_feature_columns(features.shape[-1])
    if layout is None:
        return features
    shared = torch.as_tensor(layout[2], device=features.device)
    if isinstance(side, int):
        side = torch.full((features.shape[0],), side, dtype=torch.long, device=features.device)
    side = torch.as_tensor(side, device=features.device)
    side_columns = torch.zeros(2, features.shape[-1], dtype=torch.bool, device=features.device)
    for side_index in (0, 1):
        side_columns[side_index, torch.as_tensor(layout[side_index], device=features.device)] = True
    result = torch.where(side_columns[side][:, None, :], donor, features)
    if norm_stats is None:
        result[..., shared] = 0.0
        return result
    mean, std, presence_aware = norm_stats
    if features.shape[-1] != SEQ_FEATURE_DIM:
        raise ValueError("Tính lại cột tay-tay cần đủ cột chuyển động (SEQ_FEATURE_DIM)")
    raw = result * std + mean
    if presence_aware:
        both = result[..., PRESENCE_SLICE].clamp(0, 1).round().prod(-1, keepdim=True)
    else:
        both = hand_validity(raw).prod(-1, keepdim=True)
    shared_raw = _inter_hand_raw(raw) * both
    normalized = ((shared_raw - mean[shared]) / std[shared]).clamp(-8.0, 8.0)
    result[..., shared] = normalized * both if presence_aware else normalized
    return result


def dynamic_classes(sequences: np.ndarray, labels: np.ndarray, n_classes: int, min_stroke: float) -> np.ndarray:
    is_dynamic = np.zeros(n_classes, dtype=bool)
    if sequences.ndim != 3 or sequences.shape[-1] != SEQ_FEATURE_DIM:
        return is_dynamic
    magnitude = np.array([stroke_magnitude(sequence) for sequence in sequences])
    for class_index in range(n_classes):
        if (labels == class_index).any():
            is_dynamic[class_index] = np.median(magnitude[labels == class_index]) >= min_stroke
    return is_dynamic


def still_negative_pool(sequences: np.ndarray, max_count: int = 1024, seed: int = 0) -> np.ndarray:
    pool = [version for sequence in sequences for version in still_versions(sequence)]
    if not pool:
        return np.zeros((0, *sequences.shape[1:]), dtype=np.float32)
    pool = np.stack(pool).astype(np.float32)
    if len(pool) > max_count:
        pool = pool[np.random.default_rng(seed).choice(len(pool), max_count, replace=False)]
    return pool


def still_negatives_for(sequences: np.ndarray, labels: np.ndarray, indices: np.ndarray, originals: np.ndarray, n_classes: int,
                        cfg: Config, norm: Normalizer, device) -> torch.Tensor | None:
    if cfg.still_negative_weight <= 0:
        return None
    indices = np.asarray(indices)
    source_originals = indices[np.asarray(originals)[indices]]
    dynamic = dynamic_classes(sequences[source_originals], labels[source_originals], n_classes, cfg.dynamic_min_stroke)
    pool = still_negative_pool(sequences[indices[dynamic[labels[indices]]]], seed=cfg.seed)
    if len(pool) == 0:
        return None
    return torch.from_numpy(norm(pool)).to(device)


def uniform_target_loss(logits: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, dim=-1).mean(dim=-1).mean() - math.log(logits.shape[-1])


class Augmenter:
    def __call__(self, sequences: torch.Tensor, strength: float, extra_hand_probability: float = 0.0,
                 hand_presence: torch.Tensor | None = None,
                 donor_pool: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
                 norm_stats=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if strength <= 0:
            return sequences, None
        batch_size, frame_count, feature_dim = sequences.shape
        device = sequences.device
        time_steps = (1.0 + 0.5 * strength * torch.randn(batch_size, frame_count, device=device)).clamp(min=0.2)
        cumulative_time = torch.cumsum(time_steps, dim=1)
        cumulative_time = (cumulative_time - cumulative_time[:, :1]) / (cumulative_time[:, -1:] - cumulative_time[:, :1] + 1e-6)
        start = torch.rand(batch_size, 1, device=device) * 0.12 * strength
        end = 1.0 - torch.rand(batch_size, 1, device=device) * 0.12 * strength
        t_new = start + (end - start) * cumulative_time
        grid = torch.stack([t_new * 2 - 1, torch.zeros_like(t_new)], dim=-1).unsqueeze(1)
        image = sequences.permute(0, 2, 1).unsqueeze(2)
        sequences = F.grid_sample(image, grid, mode="bilinear", padding_mode="border",
                          align_corners=True).squeeze(2).permute(0, 2, 1)
        sequences = sequences + torch.randn_like(sequences) * (0.08 * strength)
        sequences = sequences * (1.0 + 0.08 * strength * torch.randn(batch_size, 1, feature_dim, device=device))
        mask = (torch.rand(batch_size, frame_count, 1, device=device) < 0.06 * strength).float()
        sequences = sequences * (1.0 - mask)

        augmented = None
        if extra_hand_probability > 0 and hand_presence is not None and donor_pool is not None:
            present = (hand_presence >= PRESENCE_THRESHOLD).float().mean(dim=1) > 0.5
            one_hand = present.sum(dim=1) == 1
            chosen = one_hand & (torch.rand(batch_size, device=device) < extra_hand_probability)
            for side, pool in enumerate(donor_pool):
                if pool is None or len(pool) == 0:
                    continue
                eligible = chosen & ~present[:, side]
                if not eligible.any():
                    continue
                selected = torch.where(eligible)[0]
                donors = pool[torch.randint(len(pool), (len(selected),), device=device)]
                if augmented is None:
                    augmented = sequences.clone()
                augmented[selected] = add_distractor_hand_features(sequences[selected], donors, side, norm_stats)
        return sequences, augmented


class GroupStem(nn.Module):

    def __init__(self, groups, d_model, dropout, presence_scale=0.25):
        super().__init__()
        self.groups = list(groups)
        self.presence_scale = float(presence_scale)
        self.group_drop = 0.0
        self.enc = nn.ModuleList()
        total = 0
        for start, stop in self.groups:
            hidden = int(min(64, max(8, round(0.5 * (stop - start)))))
            self.enc.append(nn.Sequential(nn.Linear(stop - start, hidden), nn.LayerNorm(hidden), nn.GELU()))
            total += hidden
        self.proj = nn.Sequential(nn.Linear(total, d_model), nn.LayerNorm(d_model),
                                  nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        outs = []
        for (start, stop), encoder in zip(self.groups, self.enc):
            encoded = encoder(x[..., start:stop])
            if (start, stop) == (PRESENCE_SLICE.start, PRESENCE_SLICE.stop):
                encoded = encoded * self.presence_scale
            outs.append(encoded)
        if self.training and self.group_drop > 0 and len(outs) > 1:
            keep = (torch.rand(x.shape[0], len(outs), device=x.device) >= self.group_drop).float()
            none = keep.sum(1) == 0
            n_none = int(none.sum())
            if n_none:
                keep[none, torch.randint(len(outs), (n_none,), device=x.device)] = 1.0
            outs = [group_output * keep[:, group_index, None, None] for group_index, group_output in enumerate(outs)]
        return self.proj(torch.cat(outs, dim=-1))


class MultiScaleBlock(nn.Module):

    def __init__(self, channels, kernels, dropout):
        super().__init__()
        self.dw = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels) for kernel_size in kernels])
        self.pw = nn.Conv1d(channels * len(kernels), channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        branches = torch.cat([conv(x) for conv in self.dw], dim=1)
        return x + self.drop(self.act(self.bn(self.pw(branches))))


class CosineHead(nn.Module):

    def __init__(self, d_in, n_classes, embedding_dim=128, scale=12.0):
        super().__init__()
        self.emb = nn.Linear(d_in, embedding_dim)
        self.W = nn.Parameter(torch.randn(n_classes, embedding_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, features):
        return self.scale * F.normalize(self.emb(features), dim=-1) @ F.normalize(self.W, dim=-1).t()


class VSLNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, groups, cfg: Config):
        super().__init__()
        model_dim, hidden_size = cfg.d_model, cfg.gru_hidden
        self.stem = GroupStem(groups, model_dim, cfg.base_dropout, cfg.presence_scale)
        self.blocks = nn.Sequential(*[MultiScaleBlock(model_dim, cfg.kernels, cfg.base_dropout)
                                      for _ in range(cfg.n_blocks)])
        self.gru = nn.GRU(model_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.norm = nn.LayerNorm(6 * hidden_size)
        self.drop = nn.Dropout(cfg.base_dropout)
        self.head = CosineHead(6 * hidden_size, n_classes) if cfg.head == "cosine" else nn.Linear(6 * hidden_size, n_classes)

    def pool(self, x):
        hidden = self.stem(x)
        hidden = self.blocks(hidden.transpose(1, 2)).transpose(1, 2)
        hidden, _ = self.gru(hidden)
        attention_weights = torch.softmax(self.attn(hidden), dim=1)
        return torch.cat([(attention_weights * hidden).sum(1), hidden.mean(1), hidden.amax(1)], dim=-1)

    def forward(self, x):
        return self.head(self.drop(self.norm(self.pool(x))))


class HandPairNet(nn.Module):

    def __init__(self, in_dim: int, n_classes: int, cfg: Config, d_hand: int = 64, d_inter: int = 16,
                 d_model: int = 64, gru_hidden: int = 32):
        super().__init__()
        cols = hand_feature_columns(in_dim)
        if cols is None:
            raise ValueError(f"HandPairNet cần {FEATURE_DIM} hoặc {SEQ_FEATURE_DIM} cột, nhận {in_dim}")
        left, right, shared = cols
        column_count = len(left)
        perm, sign = LEFT_MIRROR[0][:column_count], LEFT_MIRROR[1][:column_count]
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

    def forward(self, x):
        return self.head(self.drop(self.norm(self.pool(x))))

    def pool(self, x):
        presence = x[..., PRESENCE_SLICE].clamp(0.0, 1.0)
        left_presence, right_presence = presence[..., :1], presence[..., 1:]
        left_hand = self.hand_norm(self.hand(x[..., self.left_idx] * self.left_sign) + self.side[0])
        right_hand = self.hand_norm(self.hand(x[..., self.right_idx]) + self.side[1])
        left_hand = left_presence * left_hand + (1 - left_presence) * self.absent[0]
        right_hand = right_presence * right_hand + (1 - right_presence) * self.absent[1]
        inter_hand = self.inter(x[..., self.shared_idx]) * (left_presence * right_presence)
        hidden = self.proj(torch.cat([left_hand, right_hand, inter_hand], dim=-1))
        hidden = self.block(hidden.transpose(1, 2)).transpose(1, 2)
        hidden, _ = self.gru(hidden)
        attention_weights = torch.softmax(self.attn(hidden), dim=1)
        return torch.cat([(attention_weights * hidden).sum(1), hidden.mean(1), hidden.amax(1)], dim=-1)


class PreLNBlock(nn.Module):

    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float, drop_path: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.drop = nn.Dropout(dropout)
        self.drop_path = float(drop_path)

    def _path(self, values):
        if not self.training or self.drop_path <= 0:
            return values
        keep = (torch.rand(values.shape[0], *([1] * (values.dim() - 1)), device=values.device) >= self.drop_path)
        return values * keep.to(values.dtype) / (1.0 - self.drop_path)

    def forward(self, x):
        normed = self.n1(x)
        x = x + self._path(self.drop(self.attn(normed, normed, normed, need_weights=False)[0]))
        return x + self._path(self.drop(self.ff(self.n2(x))))


def _sinusoid(frame_count: int, dim: int, device) -> torch.Tensor:
    positions = torch.arange(frame_count, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-np.log(10000.0) / dim))
    encoding = torch.zeros(frame_count, dim, device=device)
    encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions * freq), torch.cos(positions * freq)
    return encoding


class PointShapeEncoder(nn.Module):

    def __init__(self, dim: int, dropout: float, n_points: int = 20):
        super().__init__()
        self.inp = nn.Linear(3, dim)
        self.joint = nn.Parameter(torch.randn(n_points, dim) * 0.02)
        self.block = PreLNBlock(dim, 4, 2 * dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, points):
        hidden = self.norm(self.block(self.inp(points) + self.joint))
        attention_weights = torch.softmax(self.score(hidden), dim=1)
        return torch.cat([(attention_weights * hidden).sum(1), hidden.mean(1)], dim=-1)


class SignFormer(nn.Module):

    def __init__(self, in_dim: int, n_classes: int, cfg: Config, d_point: int = 32, d_hand: int = 128,
                 d_inter: int = 16, d_model: int = 128, n_layers: int = 2, heads: int = 4):
        super().__init__()
        cols = hand_feature_columns(in_dim)
        if cols is None or "hand_local" not in FEATURE_SLICES:
            raise ValueError(f"SignFormer cần {FEATURE_DIM} hoặc {SEQ_FEATURE_DIM} cột có hand_local, nhận {in_dim}")
        left, right, shared = cols
        column_count = len(left)
        perm, sign = LEFT_MIRROR[0][:column_count], LEFT_MIRROR[1][:column_count]
        left, sign = left[perm], sign[perm]
        hand_local_slice = FEATURE_SLICES["hand_local"]
        half = (hand_local_slice.stop - hand_local_slice.start) // 2
        local_l, local_r = np.arange(hand_local_slice.start, hand_local_slice.start + half), np.arange(hand_local_slice.start + half, hand_local_slice.stop)
        keep_l = (left != PRESENCE_SLICE.start) & ~np.isin(left, local_l)
        keep_r = (right != PRESENCE_SLICE.start + 1) & ~np.isin(right, local_r)
        for name, values in (("left_idx", left[keep_l]), ("left_sign", sign[keep_l]), ("right_idx", right[keep_r]),
                        ("local_l", local_l), ("local_r", local_r), ("shared_idx", shared)):
            self.register_buffer(name, torch.as_tensor(values), persistent=False)
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
                                                cfg.drop_path * (layer_index + 1) / n_layers) for layer_index in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(3 * d_model)
        self.drop = nn.Dropout(cfg.base_dropout)
        self.head = (CosineHead(3 * d_model, n_classes) if cfg.head == "cosine"
                     else nn.Linear(3 * d_model, n_classes))

    def forward(self, x):
        return self.head(self.drop(self.norm(self.pool(x))))

    def pool(self, x):
        batch_size, frame_count, _ = x.shape
        presence = x[..., PRESENCE_SLICE].clamp(0.0, 1.0).unsqueeze(-1)
        points = torch.stack([x[..., self.local_l], x[..., self.local_r]], dim=2)
        shape = self.points(points.reshape(batch_size * frame_count * 2, -1, 3)).reshape(batch_size, frame_count, 2, -1)
        other = torch.stack([x[..., self.left_idx] * self.left_sign, x[..., self.right_idx]], dim=2)
        hands = self.hand_norm(self.fuse(torch.cat([shape, self.other(other)], dim=-1)) + self.side)
        hands = presence * hands + (1 - presence) * self.absent
        inter_hand = self.inter(x[..., self.shared_idx]) * (presence[:, :, 0] * presence[:, :, 1])
        encoded = self.proj(torch.cat([hands[:, :, 0], hands[:, :, 1], inter_hand], dim=-1))
        encoded = self.conv(encoded.transpose(1, 2)).transpose(1, 2) + _sinusoid(frame_count, encoded.shape[-1], x.device)
        for layer in self.layers:
            encoded = layer(encoded)
        encoded = self.out_norm(encoded)
        attention_weights = torch.softmax(self.attn(encoded), dim=1)
        return torch.cat([(attention_weights * encoded).sum(1), encoded.mean(1), encoded.amax(1)], dim=-1)


def build_model(in_dim: int, n_classes: int, groups, cfg: Config) -> nn.Module:
    if cfg.arch == "former":
        return SignFormer(in_dim, n_classes, cfg)
    if cfg.arch == "handpair":
        return HandPairNet(in_dim, n_classes, cfg)
    return VSLNet(in_dim, n_classes, groups, cfg)


def set_dropout(model: nn.Module, probability: float):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(probability)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        pairs = list(zip(self.module.state_dict().values(), model.state_dict().values()))
        self._float = ([ema_tensor for ema_tensor, model_tensor in pairs if ema_tensor.dtype.is_floating_point],
                       [model_tensor for ema_tensor, model_tensor in pairs if ema_tensor.dtype.is_floating_point])
        self._other = [(ema_tensor, model_tensor) for ema_tensor, model_tensor in pairs if not ema_tensor.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        decay = min(self.decay, (1 + step) / (10 + step))
        ema_tensors, model_tensors = self._float
        torch._foreach_mul_(ema_tensors, decay)
        torch._foreach_add_(ema_tensors, model_tensors, alpha=1 - decay)
        for ema_tensor, model_tensor in self._other:
            ema_tensor.copy_(model_tensor)


class AdaptiveController:

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

    def update(self, epoch: int, aug_acc: float, train_val_gap: float):
        self.hist.append(self.s)
        self.aug_hist.append(aug_acc)
        if self.frozen or epoch < 5 or (epoch + 1) % self.period:
            return
        recent = float(np.mean(self.aug_hist[-self.period:]))
        if recent < 0.75:
            self.s = max(0.1, self.s - 0.1)
        elif train_val_gap > 0.12 and recent > 0.85:
            self.s = min(1.0, self.s + 0.1)


@torch.no_grad()
def logits_of(model: nn.Module, sequences: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    return torch.cat([model(sequences[start:start + batch_size]) for start in range(0, len(sequences), batch_size)])


def eval_metrics(logits: torch.Tensor, labels: torch.Tensor):
    loss = F.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(1) == labels).float().mean().item()
    top_k = min(5, logits.shape[1])
    top5 = (logits.topk(top_k, dim=1).indices == labels[:, None]).any(1).float().mean().item()
    return loss, accuracy, top5


def hard_negative_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2 or target.shape != (logits.shape[0],):
        raise ValueError("Expected logits (batch, classes>=2) and target (batch,)")
    true_logits = logits.gather(1, target[:, None]).squeeze(1)
    competing = logits.masked_fill(F.one_hot(target, logits.shape[1]).bool(), -torch.inf).max(dim=1).values
    return F.relu(margin + competing - true_logits).mean()


class ReplaySampler:

    def __init__(self, labels, quota, always=None, seed: int = 0):
        labels = np.asarray(labels)
        always = np.zeros(len(labels), dtype=bool) if always is None else np.asarray(always, dtype=bool)
        if len(always) != len(labels):
            raise ValueError("always phải cùng độ dài với y")
        self.always = np.where(always)[0]
        self.pools = [np.where((labels == class_index) & ~always)[0] for class_index in range(len(quota))]
        self.take = [min(max(int(class_quota), 0), len(pool)) for class_quota, pool in zip(quota, self.pools)]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.always) + sum(self.take)

    def __call__(self, epoch: int) -> torch.Tensor:
        parts = [self.always] + [self.rng.choice(pool, take_count, replace=False)
                                 for pool, take_count in zip(self.pools, self.take) if take_count]
        indices = np.concatenate(parts)
        self.rng.shuffle(indices)
        return torch.from_numpy(indices.astype(np.int64))


def train_one(train_sequences, train_labels, val_sequences, val_labels, n_classes, groups, cfg: Config, device,
              fixed_epochs: int | None = None, init_strength: float = 0.5,
              freeze_strength: bool = False, tag: str = "", log_every: int = 5,
              hand_presence: torch.Tensor | None = None, norm_stats=None, init_state: dict | None = None,
              sampler=None, negatives: torch.Tensor | None = None):
    has_val = val_sequences is not None
    model = build_model(train_sequences.shape[-1], n_classes, groups, cfg).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    ema = EMA(model, cfg.ema_decay)
    augmenter = Augmenter()
    donor_pool = None
    if hand_presence is not None:
        donor_pool = tuple(
            train_sequences[(hand_presence[:, :, side] >= PRESENCE_THRESHOLD).float().mean(dim=1) > 0.5]
            for side in (0, 1)
        )
    controller = AdaptiveController(cfg, init=init_strength, frozen=freeze_strength or not has_val)

    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr,
                            fused=device.type in ("cuda", "mps"))

    train_count = len(train_sequences)
    batch_size = cfg.batch_size or int(np.clip(train_count // 20, 16, 64))
    batch_size = min(batch_size, train_count)
    first_epoch_indices = sampler(0).to(device) if sampler is not None else None
    samples_per_epoch = len(first_epoch_indices) if first_epoch_indices is not None else train_count
    if samples_per_epoch == 0:
        raise ValueError("sampler trả về tập rỗng")
    batch_size = min(batch_size, samples_per_epoch)
    steps_per_epoch = max(1, samples_per_epoch // batch_size)
    extra_slots = max(1, math.ceil(cfg.extra_hand_probability * batch_size))
    use_negatives = negatives is not None and len(negatives) > 0 and cfg.still_negative_weight > 0
    neg_slots = max(1, math.ceil(cfg.still_negative_fraction * batch_size)) if use_negatives else 0
    total_epochs = fixed_epochs or cfg.max_epochs
    total_steps = total_epochs * steps_per_epoch
    warm_steps = cfg.warmup_epochs * steps_per_epoch

    counts = torch.bincount(train_labels, minlength=n_classes).float().clamp(min=1)
    class_weights = None
    if (counts.max() / counts.min()) > 1.5:
        class_weights = ((counts.sum() / (n_classes * counts)) ** 0.5).to(device)
    if first_epoch_indices is not None:
        seen = torch.bincount(train_labels[first_epoch_indices], minlength=n_classes).float()
        inverse_sampling_weights = torch.where(seen > 0, counts / seen.clamp(min=1), torch.zeros_like(seen))
        inverse_sampling_weights = (inverse_sampling_weights / inverse_sampling_weights[seen > 0].mean()).to(device)
        class_weights = inverse_sampling_weights if class_weights is None else class_weights * inverse_sampling_weights

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"key": (-1.0, 1e9), "state": None, "epoch": 0, "m": None, "logits": None}
    best_loss, plateau, step = 1e9, 0, 0
    history: list[dict] = []
    lr_scale, lr_target = 1.0, 1.0
    subset_indices = torch.randperm(train_count, device=device)[:min(train_count, 512)]
    subset_sequences, subset_labels = train_sequences[subset_indices], train_labels[subset_indices]
    if init_state is not None and has_val:
        val_logits = logits_of(ema.module, val_sequences)
        initial_metrics = eval_metrics(val_logits, val_labels)
        best.update(key=(initial_metrics[1], initial_metrics[0]), state=copy.deepcopy(ema.module.state_dict()), epoch=-1, m=initial_metrics,
                    logits=val_logits.cpu())
        best_loss = initial_metrics[0]
        if log_every:
            print(f"  {tag} ep   0 | (model ban đầu) val_loss {initial_metrics[0]:.3f} acc {initial_metrics[1]:.3f} top5 {initial_metrics[2]:.3f}",
                  flush=True)
    start_time = time.time()

    for epoch in range(total_epochs):
        model.train()
        set_dropout(model, controller.dropout)
        if hasattr(model, "stem"):
            model.stem.group_drop = controller.group_drop
        if sampler is None:
            perm = torch.randperm(train_count, device=device)
        else:
            perm = first_epoch_indices if epoch == 0 else sampler(epoch).to(device)
        run_correct, run_total = torch.zeros((), device=device), 0

        for step_in_epoch in range(steps_per_epoch):
            batch_indices = perm[step_in_epoch * batch_size:(step_in_epoch + 1) * batch_size]
            batch_inputs, batch_labels = train_sequences[batch_indices], train_labels[batch_indices]
            batch_presence = hand_presence[batch_indices] if hand_presence is not None else None
            batch_inputs, x_extra_hand = augmenter(
                batch_inputs, controller.s, cfg.extra_hand_probability, batch_presence, donor_pool, norm_stats
            )

            mixed_labels, mix_ratio = None, 1.0
            if x_extra_hand is None and random.random() < controller.mixup_p:
                mix_ratio = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                shuffle_indices = torch.randperm(len(batch_inputs), device=device)
                batch_inputs = mix_ratio * batch_inputs + (1 - mix_ratio) * batch_inputs[shuffle_indices]
                mixed_labels = batch_labels[shuffle_indices]

            lr_scale += 0.05 * (lr_target - lr_scale)
            if step < warm_steps:
                lr = cfg.lr * (step + 1) / warm_steps
            else:
                progress = (step - warm_steps) / max(1, total_steps - warm_steps)
                lr = (cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))) * lr_scale
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            changed = None
            if x_extra_hand is not None:
                changed = torch.where((x_extra_hand != batch_inputs).flatten(1).any(1))[0]
                changed = changed[torch.randperm(len(changed), device=device)[:extra_slots]]
                pad_indices = torch.randint(len(batch_inputs), (extra_slots - len(changed),), device=device)
                model_input = torch.cat([batch_inputs, x_extra_hand[changed], batch_inputs[pad_indices]])
            else:
                model_input = batch_inputs
            if neg_slots:
                negative_inputs, _ = augmenter(negatives[torch.randint(len(negatives), (neg_slots,), device=device)], controller.s)
                model_input = torch.cat([model_input, negative_inputs])
            with torch.autocast(device_type=device.type, enabled=use_amp):
                all_outputs = model(model_input).float()
            outputs = all_outputs[:len(batch_inputs)]
            extra_outputs = None
            if changed is not None:
                extra_outputs = outputs.index_copy(0, changed, all_outputs[len(batch_inputs):len(batch_inputs) + len(changed)]) if len(changed) else outputs
            with torch.no_grad():
                target = batch_labels if (mixed_labels is None or mix_ratio >= 0.5) else mixed_labels
                run_correct = run_correct + (outputs.argmax(1) == target).sum()
                run_total += len(batch_inputs)
            loss = F.cross_entropy(outputs, batch_labels, weight=class_weights, label_smoothing=cfg.label_smoothing)
            if mixed_labels is not None:
                loss = mix_ratio * loss + (1 - mix_ratio) * F.cross_entropy(
                    outputs, mixed_labels, weight=class_weights, label_smoothing=cfg.label_smoothing)
            elif cfg.hard_negative_weight > 0:
                loss = loss + cfg.hard_negative_weight * hard_negative_margin_loss(
                    outputs, batch_labels, cfg.hard_negative_margin
                )
            if extra_outputs is not None:
                loss = loss + cfg.extra_hand_loss * F.cross_entropy(
                    extra_outputs, batch_labels, weight=class_weights, label_smoothing=cfg.label_smoothing
                )
            if extra_outputs is not None and cfg.hand_consistency > 0:
                teacher = torch.softmax(outputs.detach(), dim=-1)
                loss = loss + cfg.hand_consistency * F.kl_div(
                    F.log_softmax(extra_outputs, dim=-1), teacher, reduction="batchmean"
                )
            if neg_slots:
                loss = loss + cfg.still_negative_weight * uniform_target_loss(all_outputs[-neg_slots:])

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model, step)
            step += 1

        aug_acc = (run_correct / max(1, run_total)).item()
        if has_val:
            train_clean_acc = eval_metrics(logits_of(ema.module, subset_sequences), subset_labels)[1]
            val_logits = logits_of(ema.module, val_sequences)
            val_loss, val_acc, val_top5 = eval_metrics(val_logits, val_labels)
            controller.update(epoch, aug_acc, train_clean_acc - val_acc)
            history.append({"epoch": epoch + 1, "train_aug_acc": aug_acc, "train_clean_acc": train_clean_acc,
                            "val_loss": val_loss, "val_acc": val_acc, "strength": controller.s})
            selectable = (cfg.select_after_warmup is None
                          or epoch + 1 >= cfg.warmup_epochs + cfg.select_after_warmup)
            if selectable and (val_acc, -val_loss) > (best["key"][0], -best["key"][1]):
                best.update(key=(val_acc, val_loss), state=copy.deepcopy(ema.module.state_dict()),
                            epoch=epoch, m=(val_loss, val_acc, val_top5), logits=val_logits.cpu())
            if val_loss < best_loss - 1e-4:
                best_loss, plateau = val_loss, 0
            else:
                plateau += 1
                if plateau >= cfg.lr_patience:
                    lr_target, plateau = max(0.1, lr_target * 0.5), 0
            if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
                print(f"  {tag} ep {epoch+1:3d} | train aug {aug_acc:.3f} clean {train_clean_acc:.3f} | val_loss {val_loss:.3f} "
                      f"acc {val_acc:.3f} top5 {val_top5:.3f} | s={controller.s:.1f} lr_x={lr_scale:.2f} "
                      f"| best {best['m'][1] if best['m'] else float('nan'):.3f}@{best['epoch']+1} "
                      f"| {time.time()-start_time:.0f}s", flush=True)
            if (best["state"] is not None and epoch + 1 >= cfg.min_epochs
                    and epoch - best["epoch"] >= cfg.patience):
                print(f"  {tag} early stop @ {epoch+1}")
                break
        else:
            controller.hist.append(controller.s)
            if log_every and ((epoch + 1) % (log_every * 4) == 0):
                print(f"  {tag} ep {epoch+1:3d} | train aug {aug_acc:.3f} | s={controller.s:.1f}", flush=True)

    if has_val:
        ema.module.load_state_dict(best["state"])
        hist = controller.hist[:best["epoch"] + 1] or [controller.s]
        info = {"best_epoch": best["epoch"] + 1, "val_loss": best["m"][0], "val_acc": best["m"][1],
                "val_top5": best["m"][2], "strength": float(np.mean(hist[-10:])), "val_logits": best["logits"],
                "history": history}
    else:
        info = {"best_epoch": total_epochs, "strength": controller.s, "final_lr": lr}
    return ema.module.eval(), info


def logits_and_embedding(model: nn.Module, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.norm(model.pool(features))
    logits = model.head(model.drop(hidden))
    embedding = model.head.emb(hidden) if isinstance(model.head, CosineHead) else hidden
    return logits, F.normalize(embedding, dim=-1)


class VSLEnsemble(nn.Module):

    def __init__(self, models, means, stds, temperature=1.0, presence_aware: bool = False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.presence_aware = bool(presence_aware)
        self.register_buffer("means", torch.as_tensor(np.asarray(means), dtype=torch.float32))
        self.register_buffer("stds", torch.as_tensor(np.asarray(stds), dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x):
        probs = []
        for model_index, model in enumerate(self.models):
            normalized = normalize_features(x, self.means[model_index], self.stds[model_index], self.presence_aware)
            probs.append(torch.softmax(model(normalized) / self.temperature, dim=-1))
        return torch.stack(probs).mean(0)

    def logits_and_embeddings(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        logits, embs = [], []
        for model_index, model in enumerate(self.models):
            normalized = normalize_features(x, self.means[model_index], self.stds[model_index], self.presence_aware)
            model_logits, embedding = logits_and_embedding(model, normalized)
            logits.append(model_logits / self.temperature)
            embs.append(embedding)
        return torch.stack(logits), torch.stack(embs)


def full_model_epochs(best_epochs, cfg: Config) -> int:
    return max(int(np.median(best_epochs) * 1.15) + 1, cfg.warmup_epochs + 20)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, t_min: float = 1.0, t_max: float = 10.0) -> float:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    log_min, log_max = math.log(t_min), math.log(t_max)
    raw_temperature = torch.zeros(1, requires_grad=True)

    def temp():
        return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(raw_temperature))

    optimizer = torch.optim.LBFGS([raw_temperature], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temp(), labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:  # noqa: BLE001
        return 1.0
    temperature = float(temp().detach())
    return temperature if math.isfinite(temperature) else 1.0


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor, t_min: float = 1.0,
                          t_max: float = 10.0) -> tuple[float, str]:
    if bool((logits.argmax(1) == labels).all()):
        return 1.0, "skipped_no_errors"
    temperature = fit_temperature(logits, labels, t_min, t_max)
    if temperature <= t_min * 1.01 or temperature >= t_max / 1.01:
        print(f"[cảnh báo] Temperature={temperature:.3f} chạm biên [{t_min}, {t_max}]: hiệu chỉnh bị cắt, "
              "cần dữ liệu âm/hiệu chỉnh riêng để tin được.")
        return temperature, "at_bound"
    return temperature, "fitted"


def choose_reject_threshold(probs: np.ndarray, labels: np.ndarray, target_acc: float = 0.95):
    conf, pred = probs.max(1), probs.argmax(1)
    is_correct = pred == labels
    table = []
    for threshold in np.arange(0.2, 0.96, 0.05):
        kept = conf >= threshold
        coverage = float(kept.mean())
        accuracy = float(is_correct[kept].mean()) if kept.any() else 0.0
        table.append((round(float(threshold), 2), coverage, accuracy))
    chosen = 0.5
    for threshold, coverage, accuracy in table:
        if accuracy >= target_acc and coverage >= 0.5:
            chosen = threshold
            break
    return chosen, table


def save_ensemble(path, models, norms: list, cfg: Config, groups, classes, temperature, reject_thr, in_shape):
    torch.save({"cfg": asdict(cfg), "feature_version": FEATURE_VERSION, "groups": groups, "classes": classes,
                "means": np.stack([normalizer.mean for normalizer in norms]), "stds": np.stack([normalizer.std for normalizer in norms]),
                "temperature": temperature, "reject_threshold": reject_thr, "in_shape": in_shape,
                "state_dicts": [model.cpu().state_dict() for model in models]}, path)


def load_ensemble(path, device="cpu") -> tuple[VSLEnsemble, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = {**checkpoint["cfg"], "kernels": tuple(checkpoint["cfg"]["kernels"])}
    saved_cfg.setdefault("presence_scale", 1.0)
    cfg = Config(**saved_cfg)
    models = []
    for state_dict in checkpoint["state_dicts"]:
        model = build_model(checkpoint["in_shape"][-1], len(checkpoint["classes"]), checkpoint["groups"], cfg)
        model.load_state_dict(state_dict)
        models.append(model.eval())
    ensemble = VSLEnsemble(models, checkpoint["means"], checkpoint["stds"], checkpoint["temperature"],
                      presence_aware=cfg.presence_norm).to(device).eval()
    return ensemble, checkpoint


@torch.no_grad()
def predict_topk(ensemble: VSLEnsemble, sequences: np.ndarray, classes, k=5, reject_threshold=0.0):
    device = next(ensemble.buffers()).device
    batch = torch.as_tensor(np.asarray(sequences, dtype=np.float32), device=device)
    if batch.ndim == 2:
        batch = batch[None]
    probabilities = ensemble(batch).cpu().numpy()
    results = []
    for row in probabilities:
        top_indices = np.argsort(-row)[:k]
        ranked = [(classes[class_index], float(row[class_index])) for class_index in top_indices]
        if row[top_indices[0]] < reject_threshold:
            ranked.insert(0, ("unknown", float(row[top_indices[0]])))
        results.append(ranked)
    return results


def export_onnx(ensemble: VSLEnsemble, path: str, in_shape):
    ensemble = ensemble.cpu().eval()
    dummy = torch.zeros(1, *in_shape)
    export_options = dict(input_names=["features"], output_names=["probs"], opset_version=17,
              dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}})
    try:
        torch.onnx.export(ensemble, dummy, path, dynamo=False, **export_options)
    except TypeError:
        torch.onnx.export(ensemble, dummy, path, **export_options)


def report_errors(oof_logits: np.ndarray, labels: np.ndarray, classes, mask: np.ndarray, top=8):
    pred = oof_logits.argmax(1)
    per_class = []
    for class_index in range(len(classes)):
        class_mask = mask & (labels == class_index)
        if class_mask.any():
            per_class.append((float((pred[class_mask] == class_index).mean()), classes[class_index]))
    per_class.sort()
    print("  Lớp khó nhất (OOF acc):", ", ".join(f"{name}={accuracy:.2f}" for accuracy, name in per_class[:top]))
    wrong = mask & (pred != labels)
    worst = Counter(zip(labels[wrong], pred[wrong])).most_common(top)
    if worst:
        print("  Nhầm lẫn nhiều nhất:", ", ".join(f"{classes[true_index]}→{classes[predicted_index]} ({count})" for (true_index, predicted_index), count in worst))


def confusion_pairs(oof_logits: np.ndarray, labels: np.ndarray, classes, mask: np.ndarray) -> list[dict]:
    pred = oof_logits[mask].argmax(axis=1)
    truth = labels[mask]
    wrong = truth != pred
    counts = Counter(zip(truth[wrong].tolist(), pred[wrong].tolist()))
    return [
        {"actual": classes[actual], "predicted": classes[predicted], "count": count}
        for (actual, predicted), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def closest_rivals(probs: np.ndarray, labels: np.ndarray, classes) -> list[dict]:
    rivals = []
    for class_index, class_name in enumerate(classes):
        rows = probs[labels == class_index]
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


def run_training(sequences, labels, classes, groups=None, out_dir="runs/vsl", cfg: Config | None = None,
                 folds: int = 5, folds_to_run: int = 0, add_full: bool = True,
                 export_onnx_model: bool = True, device=None,
                 validation_mask: np.ndarray | None = None, publish: bool = True) -> dict:
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = device or get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    classes = [str(class_name) for class_name in classes]
    sample_count, frame_count, feature_dim = sequences.shape
    class_count = len(classes)
    validation_mask = np.ones(sample_count, dtype=bool) if validation_mask is None else np.asarray(validation_mask, dtype=bool)
    if validation_mask.shape != (sample_count,):
        raise ValueError("validation_mask must have shape (N,)")
    if groups is None and not validation_mask.all():
        raise ValueError("groups are required when augmented samples are excluded from validation")
    if groups is not None:
        groups_array = np.asarray(groups)
        for group in np.unique(groups_array):
            members = groups_array == group
            if not validation_mask[members].any():
                raise ValueError(f"Group {group} has no original validation sample")
    if class_count < 2:
        raise ValueError("Cần ít nhất 2 lớp để huấn luyện.")
    layout = default_groups(feature_dim)
    per_class = np.bincount(labels, minlength=class_count)
    print(f"Dữ liệu: N={sample_count}, T={frame_count}, D={feature_dim}, lớp={class_count}, mẫu/lớp: min {per_class.min()} max {per_class.max()} "
          f"| nhóm cột: {layout} | thiết bị: {device}")
    if per_class.min() < folds:
        print(f"[cảnh báo] có lớp chỉ {per_class.min()} mẫu < {folds} fold: lớp đó sẽ vắng trong tập train của một số fold.")

    trainable = np.ones(sample_count, dtype=bool)
    if hand_feature_columns(feature_dim) is not None:
        no_hand = hand_validity(torch.from_numpy(sequences)).sum(dim=(1, 2)).numpy() == 0
        trainable = validation_mask | ~no_hand
        if (~trainable).any():
            print(f"Bỏ {int((~trainable).sum())} bản tăng cường không còn tay nào khỏi tập train")

    labels_tensor = torch.from_numpy(labels).to(device)

    def to_dev(norm: Normalizer, indices):
        return torch.from_numpy(norm(sequences[indices])).to(device)

    def presence_of(indices):
        if hand_feature_columns(feature_dim) is None:
            return None
        if cfg.distractor_geometry:
            return hand_validity(torch.from_numpy(sequences[indices])).to(device)
        return torch.from_numpy(sequences[indices, :, PRESENCE_SLICE].copy()).to(device)

    def stats_of(norm: Normalizer):
        if not cfg.distractor_geometry:
            return None
        return (torch.from_numpy(norm.mean).to(device), torch.from_numpy(norm.std).to(device),
                norm.presence_aware)

    def negatives_of(norm: Normalizer, indices):
        return still_negatives_for(sequences, labels, indices, validation_mask, class_count, cfg, norm, device)

    dynamic_all = dynamic_classes(sequences[validation_mask], labels[validation_mask], class_count, cfg.dynamic_min_stroke)
    if cfg.still_negative_weight > 0:
        print("Lớp có nét (train thêm mẫu âm đứng yên):", ", ".join(np.asarray(classes)[dynamic_all]) or "không có")

    fold_count = max(2, folds)
    group_ids = np.arange(sample_count) if groups is None else np.asarray(groups)
    for class_index in range(class_count):
        originals = (labels == class_index) & validation_mask
        if len(np.unique(group_ids[originals])) < fold_count:
            raise ValueError(f"Class {classes[class_index]} needs at least {fold_count} independent source groups")
    original_folds = make_folds(labels[validation_mask], fold_count, group_ids[validation_mask], cfg.seed)
    fold_by_group = dict(zip(group_ids[validation_mask], original_folds))
    fold_of = np.asarray([fold_by_group[group] for group in group_ids], dtype=np.int64)
    run_folds = list(range(fold_count))[: (folds_to_run or fold_count)]
    oof = np.zeros((sample_count, class_count), dtype=np.float32)
    oof_mask = np.zeros(sample_count, dtype=bool)
    models, norms, infos = [], [], []
    for fold in run_folds:
        val_indices = np.where((fold_of == fold) & validation_mask)[0]
        train_indices = np.where((fold_of != fold) & trainable)[0]
        if len(val_indices) == 0:
            raise ValueError(f"Fold {fold + 1} has no original validation samples")
        if len(np.unique(labels[val_indices])) != class_count or len(np.unique(labels[train_indices])) != class_count:
            raise ValueError(f"Fold {fold + 1} does not contain every class in train and validation")
        print(f"\n=== Fold {fold+1}/{fold_count}: train {len(train_indices)} | val {len(val_indices)} ===")
        fold_normalizer = Normalizer(sequences[train_indices], cfg.presence_norm)
        model, info = train_one(to_dev(fold_normalizer, train_indices), labels_tensor[train_indices], to_dev(fold_normalizer, val_indices), labels_tensor[val_indices],
                                class_count, layout, cfg, device, tag=f"[f{fold+1}]",
                                hand_presence=presence_of(train_indices), norm_stats=stats_of(fold_normalizer),
                                negatives=negatives_of(fold_normalizer, train_indices))
        print(f"  -> best ep {info['best_epoch']} | val acc {info['val_acc']:.3f} "
              f"top5 {info['val_top5']:.3f} loss {info['val_loss']:.3f} | s≈{info['strength']:.2f}")
        oof[val_indices] = info["val_logits"].numpy()
        oof_mask[val_indices] = True
        models.append(model)
        norms.append(fold_normalizer)
        infos.append(info)

    accs = [fold_info["val_acc"] for fold_info in infos]
    print(f"\nCV acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
          f"top5: {np.mean([fold_info['val_top5'] for fold_info in infos]):.4f}")

    masked_logits = torch.from_numpy(oof[oof_mask])
    masked_labels = torch.from_numpy(labels[oof_mask])
    temp, temp_status = calibrate_temperature(masked_logits, masked_labels)
    probs = torch.softmax(masked_logits / temp, 1).numpy()
    reject_threshold, table = choose_reject_threshold(probs, labels[oof_mask])
    print(f"Temperature={temp:.3f} ({temp_status}) | ngưỡng từ chối={reject_threshold:.2f}")
    print("  ngưỡng | coverage | acc trên phần giữ lại")
    for candidate_threshold, coverage, accuracy in table[::3]:
        print(f"   {candidate_threshold:.2f}  |  {coverage:.3f}   |  {accuracy:.3f}")
    report_errors(oof, labels, classes, oof_mask)
    confused = confusion_pairs(oof, labels, classes, oof_mask)
    rivals = closest_rivals(probs, labels[oof_mask], classes)
    evaluation = None
    try:
        from src.model.evaluation import oof_report
    except ImportError:
        print("[cảnh báo] Chưa có src.model.evaluation -> bỏ qua metrics['evaluation']")
    else:
        try:
            evaluation = oof_report(oof[oof_mask], labels[oof_mask], classes,
                                    None if groups is None else np.asarray(groups)[oof_mask])
        except Exception as e:  # noqa: BLE001
            print(f"[cảnh báo] oof_report lỗi: {e}")
            evaluation = {"error": str(e)}
    if rivals:
        print("  Đối thủ gần nhất:", ", ".join(
            f"{item['actual']}→{item['rival']} ({item['mean_rival_probability']:.1%})"
            for item in rivals[:8]
        ))

    if add_full:
        full_epochs = full_model_epochs([fold_info["best_epoch"] for fold_info in infos], cfg)
        full_strength = float(np.mean([fold_info["strength"] for fold_info in infos]))
        print(f"\n=== Model toàn bộ dữ liệu: {full_epochs} epoch, s={full_strength:.2f} ===")
        all_indices = np.where(trainable)[0]
        norm_all = Normalizer(sequences[all_indices], cfg.presence_norm)
        full, full_info = train_one(to_dev(norm_all, all_indices), labels_tensor[all_indices], None, None, class_count, layout, cfg, device,
                                    fixed_epochs=full_epochs, init_strength=full_strength, freeze_strength=True,
                                    tag="[full]", hand_presence=presence_of(all_indices),
                                    norm_stats=stats_of(norm_all), negatives=negatives_of(norm_all, all_indices))
        models.append(full)
        norms.append(norm_all)

    save_ensemble(out_dir / "ensemble.pt", models, norms, cfg, layout, classes, temp, reject_threshold, (frame_count, feature_dim))
    (out_dir / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "folds.json").write_text(json.dumps(
        {"k": fold_count, "seed": cfg.seed, "fold_by_group": {str(group_key): int(fold) for group_key, fold in fold_by_group.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    metrics = {
        "cv_acc_mean": float(np.mean(accs)), "cv_acc_std": float(np.std(accs)),
        "fold_acc": accs, "best_epochs": [fold_info["best_epoch"] for fold_info in infos],
        "temperature": temp, "temperature_status": temp_status, "reject_threshold": reject_threshold,
        "n_models": len(models),
        "n_samples": int(sample_count), "n_classes": int(class_count), "confusion_pairs": confused,
        "closest_rivals": rivals, "config": asdict(cfg),
        "dynamic_classes": [class_name for class_name, is_dynamic in zip(classes, dynamic_all) if is_dynamic]}
    if add_full:
        metrics["full_model"] = {"epochs": full_epochs, "final_lr": float(full_info["final_lr"])}
    if evaluation is not None:
        metrics["evaluation"] = evaluation
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"\nĐã lưu: {out_dir/'ensemble.pt'}")

    onnx_path = None
    if export_onnx_model:
        try:
            ensemble = VSLEnsemble([member_model.cpu() for member_model in models], np.stack([normalizer.mean for normalizer in norms]),
                              np.stack([normalizer.std for normalizer in norms]), temp, presence_aware=cfg.presence_norm)
            onnx_path = out_dir / "vsl_classifier.onnx"
            export_onnx(ensemble, str(onnx_path), (frame_count, feature_dim))
            print(f"Đã xuất ONNX: {onnx_path} (input 'features' (B,{frame_count},{feature_dim}) thô -> 'probs')")
        except Exception as e:  # noqa: BLE001
            onnx_path = None
            print(f"[cảnh báo] Xuất ONNX thất bại: {e}\n  (pip install onnx; model .pt vẫn dùng được)")

    if publish:
        from src.model.checkpoints import publish_run
        out_dir = publish_run(out_dir) or out_dir

    return {**metrics, "out_dir": str(out_dir), "onnx_path": str(onnx_path) if onnx_path else None,
            "fold_histories": [fold_info["history"] for fold_info in infos], "oof_logits": oof, "oof_mask": oof_mask,
            "classes": classes, "y": labels}


def load_validation_mask(path: str, sample_count: int, groups) -> np.ndarray | None:
    archive = np.load(path, allow_pickle=True)
    if "augmented" not in archive.files:
        return None
    is_augmented = np.asarray(archive["augmented"]).astype(bool)
    if is_augmented.shape != (sample_count,):
        raise ValueError(f"'augmented' phải có shape ({sample_count},), nhận {is_augmented.shape} -> không dựng được validation_mask")
    if is_augmented.any() and groups is None:
        raise ValueError(f"{int(is_augmented.sum())} mẫu tăng cường nhưng .npz thiếu 'groups' (origin_sample_id) "
                         "-> từ chối huấn luyện để tránh rò rỉ CV")
    if is_augmented.all():
        raise ValueError("mọi mẫu đều là tăng cường -> không còn mẫu gốc để validate")
    return ~is_augmented


def main():
    parser = argparse.ArgumentParser(description="Train VSL classifier (30, feature_dim) -> class")
    parser.add_argument("--data", type=str, help=".npz gồm X, y [, classes, groups]")
    parser.add_argument("--demo", action="store_true", help="dùng dữ liệu giả để thử pipeline")
    parser.add_argument("--out", type=str, default="runs/vsl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--folds-to-run", type=int, default=0, help="chỉ chạy N fold đầu (0 = tất cả)")
    parser.add_argument("--no-full", action="store_true", help="không huấn luyện model trên toàn bộ dữ liệu")
    parser.add_argument("--epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--head", choices=["linear", "cosine"], default=Config.head)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--arch", choices=["vslnet", "handpair", "former"], default=Config.arch,
                    help="handpair tự bật chuẩn hoá theo presence")
    args = parser.parse_args()

    cfg = Config(max_epochs=args.epochs, head=args.head, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
                 arch=args.arch, presence_norm=args.arch in ("handpair", "former"))
    if args.demo:
        sequences, labels, classes, groups = make_demo_dataset()
    elif args.data:
        sequences, labels, classes, groups = load_dataset(args.data)
    else:
        parser.error("cần --data hoặc --demo")
    validation_mask = load_validation_mask(args.data, len(labels), groups) if args.data and not args.demo else None
    run_training(sequences, labels, classes, groups, out_dir=args.out, cfg=cfg, folds=args.folds,
                 folds_to_run=args.folds_to_run, add_full=not args.no_full,
                 export_onnx_model=not args.no_onnx, validation_mask=validation_mask,
                 publish=not args.demo)


if __name__ == "__main__":
    main()
