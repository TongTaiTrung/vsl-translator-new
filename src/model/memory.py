"""Bộ nhớ mẫu cá nhân: người dùng ký mẫu vài từ, mô hình ưu tiên các từ có mẫu giống clip đang ký.

Không train lại. Mỗi mẫu (feature thô của một lần ký) được đưa qua từng model của ensemble để lấy
embedding (``training.logits_and_embedding``). Khi dự đoán, với mỗi lớp có mẫu, cộng vào logit của model k::

    bonus_k[c] = alpha * max_i exp(-sharpness * (1 - cos(emb_k(clip), emb_k(mẫu_i))))   (mẫu_i thuộc lớp c)

- ``max`` theo lớp (không cộng dồn) để lớp có 5 mẫu không thắng lớp có 1 mẫu chỉ vì nhiều mẫu hơn.
- Mẫu ít giống (cos thấp) gần như không cộng gì, nên lớp không có mẫu vẫn dự đoán như mô hình gốc.
- ``strength`` (0–1) đổi sang ``alpha = strength * MAX_BONUS`` (đơn vị logit sau nhiệt độ); 0 = tắt hẳn.

Dữ liệu mẫu nằm ở phía client (trình duyệt); server chỉ giữ bộ nhớ trong phiên stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.features.extract import FEATURE_DIM, process_sequence

MAX_BONUS = 10.0          # logit cộng thêm tối đa (strength = 1, mẫu giống hệt)
DEFAULT_SHARPNESS = 6.0   # cos 0.9 -> 0.55, cos 0.5 -> 0.05 của bonus tối đa
MAX_PER_CLASS = 5
MAX_EXEMPLARS = 240
MIN_FRAMES, MAX_FRAMES = 8, 120


@dataclass(frozen=True)
class Exemplar:
    label: str
    features: np.ndarray      # feature thô (frames, FEATURE_DIM)


def check_exemplar_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Mẫu phải có shape (frames, {FEATURE_DIM}), nhận {features.shape}")
    if not MIN_FRAMES <= features.shape[0] <= MAX_FRAMES:
        raise ValueError(f"Mẫu phải có {MIN_FRAMES}–{MAX_FRAMES} frame, nhận {features.shape[0]}")
    if np.isinf(features).any():
        raise ValueError("Mẫu không được chứa vô cực")
    return features


class MemoryEnsemble(nn.Module):
    """Bọc ``VSLEnsemble``: cùng giao diện (x thô (B, T, D) -> xác suất (B, C)) cộng bonus từ bộ nhớ mẫu.

    Dùng được ở mọi chỗ nhận ensemble (``VSLPredictor.predict_processed(ensemble=...)``). Sau mỗi lần
    gọi, ``last_match`` giữ lớp có mẫu giống clip cuối cùng nhất và độ giống (cos, trung bình các model).
    """

    def __init__(self, ensemble, classes, exemplars: list[Exemplar], *, target_len: int,
                 strength: float = 0.5, sharpness: float = DEFAULT_SHARPNESS,
                 max_per_class: int = MAX_PER_CLASS) -> None:
        super().__init__()
        if not 0 <= strength <= 1:
            raise ValueError("strength phải trong [0, 1]")
        if sharpness <= 0:
            raise ValueError("sharpness phải dương")
        self.ensemble = ensemble
        self.classes = tuple(classes)
        self.alpha = float(strength) * MAX_BONUS
        self.sharpness = float(sharpness)
        index = {name: i for i, name in enumerate(self.classes)}
        kept: list[Exemplar] = []
        per_class: dict[str, int] = {}
        self.ignored: list[str] = []
        for ex in reversed(exemplars):                     # giữ mẫu MỚI nhất khi một lớp quá nhiều mẫu
            if ex.label not in index:
                self.ignored.append(ex.label)               # lớp không có trong checkpoint này
                continue
            if per_class.get(ex.label, 0) >= max_per_class:
                continue
            per_class[ex.label] = per_class.get(ex.label, 0) + 1
            kept.append(ex)
        kept.reverse()
        self.counts = {name: per_class[name] for name in self.classes if name in per_class}
        device = next(ensemble.buffers()).device
        n_classes = len(self.classes)
        self.register_buffer("labels", torch.as_tensor([index[ex.label] for ex in kept], dtype=torch.long,
                                                       device=device))
        if kept:
            x = torch.as_tensor(np.stack([process_sequence(check_exemplar_features(ex.features), target_len=target_len)
                                          for ex in kept]),
                                device=device)
            with torch.inference_mode():
                _, keys = ensemble.logits_and_embeddings(x)          # (K, N, E)
            keys = keys.clone()
        else:
            keys = torch.zeros(0)
        self.register_buffer("keys", keys)
        self.register_buffer("onehot", nn.functional.one_hot(self.labels, n_classes).bool()
                             if kept else torch.zeros(0, n_classes, dtype=torch.bool, device=device))
        self.last_match: dict | None = None

    def __len__(self) -> int:
        return int(self.labels.numel())

    def class_similarity(self, emb: torch.Tensor) -> torch.Tensor:
        """emb (K, B, E) -> cos lớn nhất với mẫu của từng lớp (K, B, C); lớp không có mẫu = -1."""
        sim = torch.einsum("kbe,kne->kbn", emb, self.keys)                         # (K, B, N)
        masked = sim.unsqueeze(-1).masked_fill(~self.onehot, -1.0)                 # (K, B, N, C)
        return masked.amax(2)

    def forward(self, x):
        logits, emb = self.ensemble.logits_and_embeddings(x)                        # (K, B, C), (K, B, E)
        if len(self) == 0 or self.alpha == 0:
            self.last_match = None
            return torch.softmax(logits, -1).mean(0)
        sim = self.class_similarity(emb)
        bonus = torch.where(sim > -1, torch.exp(-self.sharpness * (1 - sim)), torch.zeros_like(sim))
        probs = torch.softmax(logits + self.alpha * bonus, -1).mean(0)
        mean_sim = sim.mean(0)[0]                                                    # clip đầu tiên trong batch
        best = int(mean_sim.argmax())
        self.last_match = {"label": self.classes[best], "similarity": float(mean_sim[best])}
        return probs

    def summary(self) -> dict:
        return {"exemplars": len(self), "classes": self.counts, "strength": self.alpha / MAX_BONUS,
                "ignored": sorted(set(self.ignored))}
