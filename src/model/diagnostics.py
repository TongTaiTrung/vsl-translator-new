"""Standalone ensemble diagnostics: open-set, noise robustness, distractor hand and embedding geometry.

Usage: python -m src.model.diagnostics --data <npz> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from src.model import training as tr

__all__ = [
    "load_dataset_npz", "run_cv", "diagnose", "open_set_hold_one_class_out",
    "gaussian_noise_false_acceptance", "distractor_flip_rate",
]


def load_dataset_npz(path):
    archive = np.load(path, allow_pickle=True)
    sequences = archive["X"].astype(np.float32)
    labels = archive["y"].astype(np.int64)
    classes = [str(class_name) for class_name in archive["classes"]] if "classes" in archive.files \
        else [str(class_index) for class_index in range(int(labels.max()) + 1)]
    groups = np.asarray(archive["groups"]) if "groups" in archive.files else np.arange(len(labels))
    validation_mask = ~np.asarray(archive["augmented"]) if "augmented" in archive.files \
        else np.ones(len(labels), dtype=bool)
    return sequences, labels, classes, groups, validation_mask


def _fit_fold(sequences, train_indices, validation_indices, y_used, n_classes, layout, config, device):
    presence_norm = getattr(config, "presence_norm", False)
    norm = tr.Normalizer(sequences[train_indices], presence_norm) if presence_norm else tr.Normalizer(sequences[train_indices])
    kwargs = {}
    if getattr(config, "distractor_geometry", False):
        kwargs["hand_presence"] = tr.hand_validity(torch.from_numpy(sequences[train_indices])).to(device)
        kwargs["norm_stats"] = (torch.from_numpy(norm.mean).to(device), torch.from_numpy(norm.std).to(device),
                                norm.presence_aware)
    else:
        presence = getattr(tr, "PRESENCE_SLICE", slice(129, 131))
        kwargs["hand_presence"] = torch.from_numpy(sequences[train_indices][..., presence].copy()).to(device)
    target_labels = torch.from_numpy(y_used).to(device)
    model, info = tr.train_one(
        torch.from_numpy(norm(sequences[train_indices])).to(device), target_labels[train_indices],
        torch.from_numpy(norm(sequences[validation_indices])).to(device), target_labels[validation_indices],
        n_classes, layout, config, device, log_every=0, **kwargs,
    )
    return model.eval(), norm, info


def _fold_assignment(labels, groups, validation_mask, fold_count, seed):
    fold_of_originals = tr.make_folds(labels[validation_mask], fold_count, groups[validation_mask], seed)
    fold_by_group = dict(zip(groups[validation_mask], fold_of_originals))
    return np.asarray([fold_by_group.get(group, -1) for group in groups], dtype=np.int64)


def _notebook_like_config(seed: int, **overrides) -> tr.Config:
    base = dict(max_epochs=200, head="linear", seed=seed, hard_negative_margin=0.8,
                hard_negative_weight=0.25, mixup_alpha=0.0)
    base.update(overrides)
    return tr.Config(**base)


@torch.no_grad()
def _logits_and_embedding(model, normalized: np.ndarray, device):
    captured = []
    hook = model.head.register_forward_pre_hook(lambda _mod, inputs: captured.append(inputs[0].detach()))
    model.eval()
    logits = model(torch.from_numpy(normalized).to(device))
    hook.remove()
    return logits.float().cpu().numpy(), captured[0].float().cpu().numpy()


def run_cv(sequences, labels, classes, groups, validation_mask, config: tr.Config, device=None, fold_count: int = 5,
           folds_to_run: int | None = None):
    device = device or tr.get_device()
    tr.set_seed(config.seed)
    layout = tr.default_groups(sequences.shape[-1])
    fold_of = _fold_assignment(labels, groups, validation_mask, fold_count, config.seed)
    result = {"models": [], "norms": [], "va": [], "infos": []}
    for fold in range(fold_count)[: folds_to_run or fold_count]:
        validation_indices = np.where((fold_of == fold) & validation_mask)[0]
        train_indices = np.where(fold_of != fold)[0]
        model, norm, info = _fit_fold(sequences, train_indices, validation_indices, labels, len(classes), layout, config, device)
        result["models"].append(model)
        result["norms"].append(norm)
        result["va"].append(validation_indices)
        result["infos"].append({key: value for key, value in info.items() if key != "val_logits"})
    return result


def diagnose(sequences, labels, classes, validation_mask, cv_result, device, noise_sigma: float = 1.0, seed: int = 0) -> dict:
    class_count = len(classes)
    sample_count = len(labels)
    global_std = sequences.reshape(-1, sequences.shape[-1]).std(0).astype(np.float32)
    rng = np.random.default_rng(seed)

    oof_logits = np.zeros((sample_count, class_count), np.float32)
    mask = np.zeros(sample_count, bool)
    pair_scores: dict[tuple[int, int], list[float]] = {}
    clean_gap, noisy_gap = [], []

    for model, norm, validation_indices in zip(cv_result["models"], cv_result["norms"], cv_result["va"]):
        trn_orig = np.setdiff1d(np.where(validation_mask)[0], validation_indices)
        logits, validation_embeddings = _logits_and_embedding(model, norm(sequences[validation_indices]), device)
        oof_logits[validation_indices] = logits
        mask[validation_indices] = True
        _, train_embeddings = _logits_and_embedding(model, norm(sequences[trn_orig]), device)

        normalized_validation = normalize(validation_embeddings)
        normalized_train = normalize(train_embeddings)
        centroids = normalize(np.stack([normalized_train[labels[trn_orig] == class_index].mean(0) for class_index in range(class_count)]))
        spread = np.array([np.mean(1 - normalized_train[labels[trn_orig] == class_index] @ centroids[class_index]) for class_index in range(class_count)])
        for class_a, class_b in combinations(range(class_count), 2):
            separation = float((1 - centroids[class_a] @ centroids[class_b]) / max((spread[class_a] + spread[class_b]) / 2, 1e-9))
            pair_scores.setdefault((class_a, class_b), []).append(separation)

        true_logit = logits[np.arange(len(validation_indices)), labels[validation_indices]]
        rival = logits.copy()
        rival[np.arange(len(validation_indices)), labels[validation_indices]] = -np.inf
        clean_gap.extend((true_logit - rival.max(1)).tolist())

        live = (sequences[validation_indices] != 0).any(axis=1, keepdims=True)
        noisy_X = sequences[validation_indices] + rng.normal(0, 1, sequences[validation_indices].shape).astype(np.float32) * noise_sigma * global_std * live
        noisy_logits, _ = _logits_and_embedding(model, norm(noisy_X), device)
        true_logit_n = noisy_logits[np.arange(len(validation_indices)), labels[validation_indices]]
        rival_n = noisy_logits.copy()
        rival_n[np.arange(len(validation_indices)), labels[validation_indices]] = -np.inf
        noisy_gap.extend((true_logit_n - rival_n.max(1)).tolist())

    oof_labels, lo_oof = labels[mask], oof_logits[mask]
    hard_pair_sep = {
        f"{classes[class_a]}/{classes[class_b]}": float(np.mean(scores))
        for (class_a, class_b), scores in sorted(pair_scores.items(), key=lambda pair_item: np.mean(pair_item[1]))
    }
    clean_gap, noisy_gap = np.array(clean_gap), np.array(noisy_gap)
    return {
        "oof_acc": float((lo_oof.argmax(1) == oof_labels).mean()),
        "oof_macro_f1": float(f1_score(oof_labels, lo_oof.argmax(1), average="macro")),
        "logit_gap_median_clean": float(np.median(clean_gap)),
        "logit_gap_median_noise_sigma1": float(np.median(noisy_gap)),
        "noise_sigma1_gap_retention": float(np.median(noisy_gap) / np.median(clean_gap))
        if np.median(clean_gap) else float("nan"),
        "hard_pair_separation_ratio": hard_pair_sep,
    }


def open_set_hold_one_class_out(sequences, labels, classes, groups, validation_mask, device, seed, held_classes=None,
                                 fold: int = 0, fold_count: int = 5, min_confidence: float = 0.65,
                                 min_gap: float = 0.08, config: tr.Config | None = None) -> list[dict]:
    tr.set_seed(seed)
    config = config or _notebook_like_config(seed)
    fold_of = _fold_assignment(labels, groups, validation_mask, fold_count, seed)
    layout = tr.default_groups(sequences.shape[-1])
    held_classes = list(held_classes) if held_classes else list(classes)
    results = []
    for held in held_classes:
        held_index = classes.index(held)
        keep = labels != held_index
        remaining = sorted({int(label_index) for label_index in labels[keep].tolist()})
        remap = {old_index: new_index for new_index, old_index in enumerate(remaining)}
        y_remap = np.array([remap.get(int(label_index), 0) for label_index in labels], dtype=np.int64)

        validation_indices = np.where((fold_of == fold) & validation_mask & keep)[0]
        train_indices = np.where((fold_of != fold) & keep)[0]
        model, norm, info = _fit_fold(sequences, train_indices, validation_indices, y_remap, len(remaining), layout, config, device)
        validation_logits, _ = _logits_and_embedding(model, norm(sequences[validation_indices]), device)
        fitted_temperature = tr.fit_temperature(torch.from_numpy(validation_logits), torch.from_numpy(y_remap[validation_indices]))

        test_idx = np.where((~keep) & validation_mask)[0]
        logits, _ = _logits_and_embedding(model, norm(sequences[test_idx]), device)
        row = {"held_class": held, "n_test": int(len(test_idx)), "fitted_T": fitted_temperature,
               "val_acc": info["val_acc"]}
        for temperature, label in ((1.0, "T1"), (fitted_temperature, "T_fit")):
            probabilities = softmax(logits / temperature, axis=1)
            sorted_probabilities = np.sort(probabilities, axis=1)
            probability_gap = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
            accepted = (probabilities.max(1) >= min_confidence) & (probability_gap >= min_gap)
            row[f"false_accept_rate_{label}"] = float(accepted.mean())
            row[f"median_maxprob_{label}"] = float(np.median(probabilities.max(1)))
        results.append(row)
    return results


def gaussian_noise_false_acceptance(cv_result, classes, device, n_samples: int = 50, seed: int = 0,
                                     min_confidence: float = 0.65, min_gap: float = 0.08,
                                     temperatures=(1.0, 0.2)) -> dict:
    rng = np.random.default_rng(seed)
    in_dim = cv_result["norms"][0].mean.shape[-1]
    noise_sequences = rng.normal(size=(n_samples, 30, in_dim)).astype(np.float32)
    per_model_logits = []
    for model in cv_result["models"]:
        with torch.no_grad():
            logits = model(torch.from_numpy(noise_sequences).to(device)).float().cpu().numpy()
        per_model_logits.append(logits)
    report = {}
    for temperature in temperatures:
        probs = np.mean([softmax(logits / temperature, axis=1) for logits in per_model_logits], axis=0)
        sorted_probabilities = np.sort(probs, axis=1)
        probability_gap = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        accepted = (probs.max(1) >= min_confidence) & (probability_gap >= min_gap)
        pred_counts = {classes[class_index]: int(count) for class_index, count in
                       zip(*np.unique(probs.argmax(1), return_counts=True))}
        report[f"T={temperature}"] = {
            "median_maxprob": float(np.median(probs.max(1))),
            "false_accept_rate": float(accepted.mean()),
            "predicted_as": pred_counts,
        }
    return report


def _composite_left_hand(target: np.ndarray, donor: np.ndarray, slices: dict, lowered_y: float | None):
    frame_count = len(target)
    left = np.concatenate([np.arange(slices[group_name].start, slices[group_name].start + (slices[group_name].stop - slices[group_name].start) // 2)
                           for group_name in ("hand_pos", "body_contact", "hand_scale", "handshape", "orientation",
                                     "presence")])
    donor_frames = donor[np.round(np.linspace(0, len(donor) - 1, frame_count)).astype(int)]
    if lowered_y is not None:
        valid = np.flatnonzero(np.isfinite(donor[:, slices["hand_pos"].start]))
        donor_frames = np.repeat(donor[valid[len(valid) // 2]][None], frame_count, axis=0)
        hand_pos_start = slices["hand_pos"].start
        donor_frames[:, [hand_pos_start + 1, hand_pos_start + 4, hand_pos_start + 7]] += lowered_y - donor_frames[:, [hand_pos_start + 1]]
    composite = target.copy()
    composite[:, left] = donor_frames[:, left]
    hand_pos = composite[:, slices["hand_pos"]]
    left_points, right_points = hand_pos[:, :9].reshape(frame_count, 3, 3), hand_pos[:, 9:].reshape(frame_count, 3, 3)
    contact_distances = np.concatenate([np.linalg.norm(left_points - right_points, axis=-1),
                        np.linalg.norm(left_points[:, :, None] - right_points[:, None], axis=-1).reshape(frame_count, -1).min(1, keepdims=True)], 1)
    composite[:, slices["inter_hand_vec"]] = left_points[:, 0] - right_points[:, 0]
    composite[:, slices["hand_contact"]] = np.exp(-8.0 * contact_distances ** 2)
    return composite


def distractor_flip_rate(raw, labels, classes, validation_mask, cv_result, device, extract_module, lowered_y: float = 1.1,
                         min_confidence: float = 0.65, min_gap: float = 0.08) -> dict:
    slices, hand_pos_start = extract_module.FEATURE_SLICES, extract_module.FEATURE_SLICES["hand_pos"].start
    left_frac = np.array([np.isfinite(row_features[:, hand_pos_start]).mean() if is_validation else np.nan for row_features, is_validation in zip(raw, validation_mask)])
    report = {}
    for name, lowered in (("same_height", None), ("lowered_static", lowered_y)):
        flips, wrong_acc, to_twohand, total = 0, 0, 0, 0
        for model, norm, validation_indices in zip(cv_result["models"], cv_result["norms"], cv_result["va"]):
            targets = validation_indices[left_frac[validation_indices] < 0.1]
            donors = validation_indices[left_frac[validation_indices] >= 0.5]
            if not len(donors) or not len(targets):
                continue
            twohand = set(labels[donors].tolist())
            clean = np.stack([extract_module.process_sequence(raw[sample_index], target_len=X_LEN) for sample_index in targets])
            comp = np.stack([extract_module.process_sequence(_composite_left_hand(raw[sample_index], raw[donors[target_position % len(donors)]], slices, lowered),
                                                 target_len=X_LEN) for target_position, sample_index in enumerate(targets)])
            clean_predictions = _logits_and_embedding(model, norm(clean.astype(np.float32)), device)[0].argmax(1)
            probabilities = softmax(_logits_and_embedding(model, norm(comp.astype(np.float32)), device)[0], axis=1)
            pred, sorted_probabilities = probabilities.argmax(1), np.sort(probabilities, axis=1)
            accepted = (sorted_probabilities[:, -1] >= min_confidence) & (sorted_probabilities[:, -1] - sorted_probabilities[:, -2] >= min_gap)
            flips += int((pred != clean_predictions).sum())
            wrong_acc += int((accepted & (pred != labels[targets])).sum())
            to_twohand += int(np.isin(pred, list(twohand)).sum())
            total += len(targets)
        report[name] = {"n": total, "flip_rate": flips / max(total, 1), "wrong_accept_rate": wrong_acc / max(total, 1),
                     "to_two_hand_class_rate": to_twohand / max(total, 1)}
    return report


X_LEN = 30


def _run_one_seed(sequences, labels, classes, groups, validation_mask, seed, fold_count, held_classes, out_dir, device):
    config = _notebook_like_config(seed)
    cv_result = run_cv(sequences, labels, classes, groups, validation_mask, config, device=device, fold_count=fold_count)
    report = diagnose(sequences, labels, classes, validation_mask, cv_result, device, seed=seed)
    report["open_set_hold_one_class_out"] = open_set_hold_one_class_out(
        sequences, labels, classes, groups, validation_mask, device, seed, held_classes=held_classes, fold_count=fold_count,
    )
    report["gaussian_noise"] = gaussian_noise_false_acceptance(cv_result, classes, device, seed=seed)
    report["seed"] = seed
    path = Path(out_dir) / f"diagnostics_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chẩn đoán open-set / nhiễu / hình học embedding")
    parser.add_argument("--data", required=True, help="File .npz (X, y, classes[, groups, augmented])")
    parser.add_argument("--out", required=True, help="Thư mục ghi kết quả JSON")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--k", type=int, default=5, help="Số fold")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Chỉ chạy hold-one-class-out trên các lớp này (mặc định: tất cả)")
    args = parser.parse_args()

    sequences, labels, classes, groups, validation_mask = load_dataset_npz(args.data)
    device = tr.get_device()
    held_classes = args.classes if args.classes else classes
    for seed in args.seeds:
        report = _run_one_seed(sequences, labels, classes, groups, validation_mask, seed, args.k, held_classes, args.out, device)
        print(f"seed={seed} oof_acc={report['oof_acc']:.4f} "
              f"gap_retention_sigma1={report['noise_sigma1_gap_retention']:.3f}")


if __name__ == "__main__":
    main()
