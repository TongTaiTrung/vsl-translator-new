"""Extra metrics on out-of-fold logits (original samples only)."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

__all__ = ["oof_report"]


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    return precision_recall_fscore_support(y_true, y_pred, labels=np.arange(n_classes), zero_division=0)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0))


def _topk_accuracy(logits: np.ndarray, labels: np.ndarray, top_k: int) -> float:
    top_indices = np.argsort(-logits, axis=1)[:, :top_k]
    return float((top_indices == labels[:, None]).any(axis=1).mean())


def _mean_prob_by_true_class(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    mean_probs = np.full((n_classes, n_classes), np.nan)
    for class_index in range(n_classes):
        rows = probs[labels == class_index]
        if len(rows):
            mean_probs[class_index] = rows.mean(axis=0)
    return mean_probs


def _closest_pairs(probs: np.ndarray, labels: np.ndarray, n_classes: int, n_pairs: int) -> list[tuple[int, int, float]]:
    mean_probs = _mean_prob_by_true_class(probs, labels, n_classes)
    scored = []
    for class_a in range(n_classes):
        for class_b in range(class_a + 1, n_classes):
            a_to_b = mean_probs[class_a, class_b] if np.isfinite(mean_probs[class_a, class_b]) else -np.inf
            b_to_a = mean_probs[class_b, class_a] if np.isfinite(mean_probs[class_b, class_a]) else -np.inf
            score = max(a_to_b, b_to_a)
            if np.isfinite(score):
                scored.append((score, class_a, class_b))
    scored.sort(key=lambda item: -item[0])
    return [(class_a, class_b, score) for score, class_a, class_b in scored[:n_pairs]]


def _pairwise_binary_accuracy(logits: np.ndarray, labels: np.ndarray, class_a: int, class_b: int) -> tuple[float, int] | None:
    selected = (labels == class_a) | (labels == class_b)
    if not selected.any():
        return None
    two_way = logits[selected][:, (class_a, class_b)]
    pred = np.where(two_way[:, 0] >= two_way[:, 1], class_a, class_b)
    return float((pred == labels[selected]).mean()), int(selected.sum())


def _bootstrap_ci(labels: np.ndarray, pred: np.ndarray, n_classes: int, groups: np.ndarray | None,
                   n_resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sample_count = len(labels)
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        index_by_group = {group: np.where(groups == group)[0] for group in unique_groups}

        def sample_indices():
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            return np.concatenate([index_by_group[group] for group in chosen])
    else:
        def sample_indices():
            return rng.integers(0, sample_count, sample_count)

    accs = np.empty(n_resamples)
    macro_f1s = np.empty(n_resamples)
    for resample_index in range(n_resamples):
        indices = sample_indices()
        resampled_labels, resampled_pred = labels[indices], pred[indices]
        accs[resample_index] = (resampled_labels == resampled_pred).mean()
        macro_f1s[resample_index] = _macro_f1(resampled_labels, resampled_pred, n_classes)
    return {
        "accuracy": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1": [float(np.percentile(macro_f1s, 2.5)), float(np.percentile(macro_f1s, 97.5))],
    }


def oof_report(logits: np.ndarray, labels: np.ndarray, classes: list[str],
               groups: np.ndarray | None = None) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = len(classes)
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError(f"Expected logits (N, {n_classes}), got {logits.shape}")
    if logits.shape[0] != len(labels):
        raise ValueError("logits và y phải cùng số mẫu")
    if groups is not None and len(groups) != len(labels):
        raise ValueError("groups phải cùng số mẫu với y")

    pred = logits.argmax(axis=1)
    probs = softmax(logits, axis=1)
    confusion = _confusion_matrix(labels, pred, n_classes)
    precision, recall, f1, support = _precision_recall_f1(labels, pred, n_classes)

    per_class = [
        {"class": classes[class_index], "precision": float(precision[class_index]), "recall": float(recall[class_index]),
         "f1": float(f1[class_index]), "support": int(support[class_index])}
        for class_index in range(n_classes)
    ]

    top_k = {str(cutoff): _topk_accuracy(logits, labels, cutoff) for cutoff in (1, 3, 5) if cutoff <= n_classes}

    hard_pairs = []
    for class_a, class_b, score in _closest_pairs(probs, labels, n_classes, n_pairs=5):
        result = _pairwise_binary_accuracy(logits, labels, class_a, class_b)
        if result is None:
            continue
        pair_acc, n_samples = result
        hard_pairs.append({
            "class_a": classes[class_a], "class_b": classes[class_b],
            "mean_rival_probability": float(score),
            "pairwise_accuracy": pair_acc, "n_samples": n_samples,
        })

    return {
        "accuracy": float((pred == labels).mean()),
        "macro_f1": _macro_f1(labels, pred, n_classes),
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
        "labels": list(classes),
        "top_k": top_k,
        "hard_pairs": hard_pairs,
        "bootstrap_ci_95": _bootstrap_ci(labels, pred, n_classes, groups, n_resamples=1000, seed=0),
    }
