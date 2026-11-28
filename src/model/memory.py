"""Personal exemplar memory: user-signed samples bias the ensemble logits without retraining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.features.extract import FEATURE_DIM, process_sequence

MAX_BONUS = 10.0
DEFAULT_SHARPNESS = 6.0
MAX_PER_CLASS = 5
MAX_EXEMPLARS = 240
MIN_FRAMES, MAX_FRAMES = 8, 120


@dataclass(frozen=True)
class Exemplar:
    label: str
    features: np.ndarray


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
        for ex in reversed(exemplars):
            if ex.label not in index:
                self.ignored.append(ex.label)
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
                _, keys = ensemble.logits_and_embeddings(x)
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
        sim = torch.einsum("kbe,kne->kbn", emb, self.keys)
        masked = sim.unsqueeze(-1).masked_fill(~self.onehot, -1.0)
        return masked.amax(2)

    def forward(self, x):
        logits, emb = self.ensemble.logits_and_embeddings(x)
        if len(self) == 0 or self.alpha == 0:
            self.last_match = None
            return torch.softmax(logits, -1).mean(0)
        sim = self.class_similarity(emb)
        bonus = torch.where(sim > -1, torch.exp(-self.sharpness * (1 - sim)), torch.zeros_like(sim))
        probs = torch.softmax(logits + self.alpha * bonus, -1).mean(0)
        mean_sim = sim.mean(0)[0]
        best = int(mean_sim.argmax())
        self.last_match = {"label": self.classes[best], "similarity": float(mean_sim[best])}
        return probs

    def summary(self) -> dict:
        return {"exemplars": len(self), "classes": self.counts, "strength": self.alpha / MAX_BONUS,
                "ignored": sorted(set(self.ignored))}
