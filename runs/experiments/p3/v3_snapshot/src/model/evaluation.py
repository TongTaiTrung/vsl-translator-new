"""Các chỉ số đánh giá bổ sung cho OOF logits (originals-only, T=1).

Module này KHÔNG import ``src.model.training`` để tránh phụ thuộc vòng (training.py
sẽ gọi ``oof_report`` sau khi tính xong ``oof``/``oof_mask``). Chỉ dùng numpy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["oof_report"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n), where=denom > 0)
    return precision, recall, f1, support


def _macro_f1(cm: np.ndarray) -> float:
    return float(_precision_recall_f1(cm)[2].mean())


def _topk_accuracy(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    top = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def _mean_prob_by_true_class(probs: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """mean_probs[c] = trung bình xác suất dự đoán (mọi lớp) trên các mẫu có nhãn thật = c."""
    mean_probs = np.full((n_classes, n_classes), np.nan)
    for c in range(n_classes):
        rows = probs[y == c]
        if len(rows):
            mean_probs[c] = rows.mean(axis=0)
    return mean_probs


def _closest_pairs(probs: np.ndarray, y: np.ndarray, n_classes: int, n_pairs: int) -> list[tuple[int, int, float]]:
    """Xếp hạng mọi cặp lớp theo mean softmax rival probability (đối xứng, lấy max 2 chiều)."""
    mean_probs = _mean_prob_by_true_class(probs, y, n_classes)
    scored = []
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            sab = mean_probs[a, b] if np.isfinite(mean_probs[a, b]) else -np.inf
            sba = mean_probs[b, a] if np.isfinite(mean_probs[b, a]) else -np.inf
            score = max(sab, sba)
            if np.isfinite(score):
                scored.append((score, a, b))
    scored.sort(key=lambda item: -item[0])
    return [(a, b, score) for score, a, b in scored[:n_pairs]]


def _pairwise_binary_accuracy(logits: np.ndarray, y: np.ndarray, a: int, b: int) -> tuple[float, int] | None:
    sel = (y == a) | (y == b)
    if not sel.any():
        return None
    two_way = logits[sel][:, (a, b)]
    pred = np.where(two_way[:, 0] >= two_way[:, 1], a, b)
    return float((pred == y[sel]).mean()), int(sel.sum())


def _bootstrap_ci(y: np.ndarray, pred: np.ndarray, n_classes: int, groups: np.ndarray | None,
                   n_resamples: int, seed: int) -> dict:
    """CI 95% cho accuracy/macro_f1 bằng bootstrap; resample theo nhóm nếu có groups."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        index_by_group = {g: np.where(groups == g)[0] for g in unique_groups}

        def sample_indices():
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            return np.concatenate([index_by_group[g] for g in chosen])
    else:
        def sample_indices():
            return rng.integers(0, n, n)

    accs = np.empty(n_resamples)
    f1s = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = sample_indices()
        by, bp = y[idx], pred[idx]
        accs[i] = (by == bp).mean()
        f1s[i] = _macro_f1(_confusion_matrix(by, bp, n_classes))
    return {
        "accuracy": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def oof_report(logits: np.ndarray, y: np.ndarray, classes: list[str],
               groups: np.ndarray | None = None) -> dict:
    """Báo cáo JSON-ready từ OOF logits (chỉ mẫu gốc, T=1).

    Tham số:
      logits: (N, C) logits OOF đã pool qua các fold.
      y: (N,) nhãn thật (chỉ số lớp).
      classes: tên C lớp, theo đúng thứ tự chỉ số.
      groups: (N,) id nhóm gốc (để bootstrap resample theo nhóm, tránh đếm hai lần
        các bản augmentation); None -> resample theo từng mẫu.

    Trả về dict gồm: accuracy, macro_f1, per_class (list precision/recall/f1/support),
    confusion_matrix + labels, top_k (k<=C trong {1,3,5}), hard_pairs (5 cặp lớp gần
    nhau nhất theo mean softmax rival probability, kèm accuracy nhị phân trên riêng
    2 lớp đó), bootstrap_ci_95 (accuracy, macro_f1).
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_classes = len(classes)
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError(f"Expected logits (N, {n_classes}), got {logits.shape}")
    if logits.shape[0] != len(y):
        raise ValueError("logits và y phải cùng số mẫu")
    if groups is not None and len(groups) != len(y):
        raise ValueError("groups phải cùng số mẫu với y")

    pred = logits.argmax(axis=1)
    probs = _softmax(logits)
    cm = _confusion_matrix(y, pred, n_classes)
    precision, recall, f1, support = _precision_recall_f1(cm)

    per_class = [
        {"class": classes[c], "precision": float(precision[c]), "recall": float(recall[c]),
         "f1": float(f1[c]), "support": int(support[c])}
        for c in range(n_classes)
    ]

    top_k = {str(k): _topk_accuracy(logits, y, k) for k in (1, 3, 5) if k <= n_classes}

    hard_pairs = []
    for a, b, score in _closest_pairs(probs, y, n_classes, n_pairs=5):
        result = _pairwise_binary_accuracy(logits, y, a, b)
        if result is None:
            continue
        pair_acc, n_samples = result
        hard_pairs.append({
            "class_a": classes[a], "class_b": classes[b],
            "mean_rival_probability": float(score),
            "pairwise_accuracy": pair_acc, "n_samples": n_samples,
        })

    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": _macro_f1(cm),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": list(classes),
        "top_k": top_k,
        "hard_pairs": hard_pairs,
        "bootstrap_ci_95": _bootstrap_ci(y, pred, n_classes, groups, n_resamples=1000, seed=0),
    }
