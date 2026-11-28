"""Cross-session benchmark: hold out whole class@session groups to measure generalization to unseen setups.

Usage: python -m src.model.benchmark {invariance,build,train,eval}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist

from src.data.augment import load_source_landmarks, make_variant
from src.data.data_reader import DataHandler, get_data_path
from src.features.extract import PITCH_AUG_DEG, process_sequence


@dataclass
class Clip:
    cls: str
    session: str
    sample_id: str
    features: np.ndarray
    landmarks: tuple | None


def parse_holdout(specs: list[str] | None) -> list[tuple[str | None, str]]:
    holdouts = []
    for spec in specs or []:
        class_name, separator, session = spec.partition("@")
        if not separator or not session:
            raise ValueError(f"Holdout phải dạng lớp@session hoặc @session: {spec!r}")
        holdouts.append((class_name or None, session))
    return holdouts


def is_held_out(clip: Clip, holdout: list[tuple[str | None, str]]) -> bool:
    return any((class_name is None or class_name == clip.cls) and clip.session.startswith(session) for class_name, session in holdout)


def load_originals(processed_root: Path | None = None, with_landmarks: bool = True) -> list[Clip]:
    root = Path(processed_root) if processed_root else get_data_path() / "processed"
    clips = []
    for class_dir in sorted(entry for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")):
        for sample_dir in sorted((entry for entry in class_dir.iterdir() if entry.is_dir() and entry.name.isdigit()),
                                 key=lambda entry: int(entry.name)):
            meta = DataHandler._read_meta(sample_dir)
            if "augmentation" in meta:
                continue
            paths, problem = DataHandler._frame_paths(sample_dir)
            if problem:
                continue
            features = np.stack([DataHandler._load_feature(entry) for entry in paths])
            landmarks = load_source_landmarks(sample_dir, len(features)) if with_landmarks else None
            clips.append(Clip(class_dir.name, str(meta.get("session", "legacy")), sample_dir.name, features, landmarks))
    return clips


def _summarize_sequence(sequence: np.ndarray) -> np.ndarray:
    return np.concatenate([sequence.mean(0), sequence.std(0)])


def session_invariance(clips: list[Clip], day_chars: int = 8) -> dict:
    pooled = np.stack([_summarize_sequence(process_sequence(clip.features)) for clip in clips])
    pooled = (pooled - pooled.mean(0)) / (pooled.std(0) + 1e-6)
    days = np.array([clip.session[:day_chars] for clip in clips])
    labels = np.array([clip.cls for clip in clips])
    report = {}
    for class_name in sorted(set(labels)):
        cls_days = sorted(set(days[labels == class_name]))
        if len(cls_days) < 2:
            continue
        hits = total = 0
        for day in cls_days:
            query = (labels == class_name) & (days == day)
            reference_mask = ~query & ((labels != class_name) | (days != day))
            nearest = cdist(pooled[query], pooled[reference_mask]).argmin(axis=1)
            hits += int((labels[reference_mask][nearest] == class_name).sum())
            total += int(query.sum())
        centers = np.stack([pooled[(labels == class_name) & (days == other_day)].mean(0) for other_day in cls_days])
        others = np.stack([pooled[labels == other_class].mean(0) for other_class in sorted(set(labels)) if other_class != class_name])
        within = pdist(centers).mean()
        between = cdist(centers, others).min(axis=1).mean()
        report[class_name] = {"sessions": cls_days, "cross_session_nn": round(hits / total, 3),
                       "ratio": round(float(within / between), 3)}
    return report


def build_dataset(clips: list[Clip], holdout: list[tuple[str | None, str]], mode: str, per_class: int,
                  rng: np.random.Generator, strength: float = 1.0, pitch_degrees: float = PITCH_AUG_DEG) -> dict:
    if mode not in ("landmark", "feature", "none"):
        raise ValueError("mode phải là landmark, feature hoặc none")
    train = [source_clip for source_clip in clips if not is_held_out(source_clip, holdout)]
    by_class: dict[str, list[Clip]] = defaultdict(list)
    for source_clip in train:
        by_class[source_clip.cls].append(source_clip)
    classes = sorted(by_class)
    sequences, labels, groups, sessions, augmented = [], [], [], [], []

    def add(clip: Clip, raw_features: np.ndarray, is_aug: bool) -> None:
        sequences.append(process_sequence(raw_features))
        labels.append(classes.index(clip.cls))
        groups.append(f"{clip.cls}/{clip.sample_id}")
        sessions.append(clip.session)
        augmented.append(is_aug)

    for class_name in classes:
        members = by_class[class_name]
        for clip in members:
            add(clip, clip.features, False)
        if mode == "none":
            continue
        for variant_index in range(max(0, per_class - len(members))):
            clip = members[variant_index % len(members)]
            if mode == "landmark" and clip.landmarks is None:
                raise ValueError(f"{class_name}/{clip.sample_id} chưa có landmarks.npz: chạy reextract trước")
            landmarks = clip.landmarks if mode == "landmark" else None
            raw_features, _ = make_variant(clip.features, landmarks, rng, strength, pitch_degrees)
            add(clip, raw_features, True)
    return {"X": np.stack(sequences).astype(np.float32), "y": np.asarray(labels, dtype=np.int64), "classes": np.asarray(classes),
            "groups": np.asarray(groups), "sessions": np.asarray(sessions), "augmented": np.asarray(augmented)}


def evaluate_holdout(predictor, clips: list[Clip], holdout: list[tuple[str | None, str]]) -> dict:
    test = [held_out_clip for held_out_clip in clips if is_held_out(held_out_clip, holdout)]
    if not test:
        raise ValueError("Không có mẫu gốc nào khớp holdout")
    per_class: dict[str, Counter] = defaultdict(Counter)
    top1 = accepted_right = 0
    known = set(predictor.classes)
    for clip in test:
        result = predictor.predict_raw(clip.features)
        label = result["top"][0]["label"]
        per_class[clip.cls][label] += 1
        top1 += label == clip.cls
        accepted_right += bool(result["accepted"]) and result["label"] == clip.cls
    return {"n": len(test), "top1": round(top1 / len(test), 3), "accepted_correct": round(accepted_right / len(test), 3),
            "unseen_classes": sorted({held_out_clip.cls for held_out_clip in test} - known),
            "predictions": {class_name: dict(counter.most_common()) for class_name, counter in sorted(per_class.items())}}


def _train(args) -> None:
    from src.model import training
    cfg_kwargs = {}
    if args.config_from:
        cfg_kwargs = json.loads(Path(args.config_from).read_text(encoding="utf-8"))["config"]
        cfg_kwargs = {option: tuple(value) if isinstance(value, list) else value for option, value in cfg_kwargs.items()}
    if args.epochs:
        cfg_kwargs["max_epochs"] = args.epochs
    if args.seed is not None:
        cfg_kwargs["seed"] = args.seed
    config = training.Config(**cfg_kwargs)
    sequences, labels, classes, groups = training.load_dataset(args.data)
    mask = training.load_validation_mask(args.data, len(labels), groups)
    training.run_training(sequences, labels, classes, groups, out_dir=args.out, cfg=config, folds=args.folds,
                          folds_to_run=args.folds_to_run, add_full=True, export_onnx_model=False,
                          validation_mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark xuyên bối cảnh quay")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("invariance", help="đo bất biến bối cảnh của feature (không train)")
    build_parser = subparsers.add_parser("build", help="dựng .npz train bỏ holdout")
    build_parser.add_argument("--mode", choices=["landmark", "feature", "none"], default="landmark")
    build_parser.add_argument("--holdout", nargs="*", default=[])
    build_parser.add_argument("--per-class", type=int, default=130)
    build_parser.add_argument("--strength", type=float, default=1.0)
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument("--out", required=True)
    train_parser = subparsers.add_parser("train", help="train trên .npz đã dựng")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--config-from", help="metrics.json của một run để dùng lại Config")
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--folds-to-run", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=None, help="ghi đè seed của Config")
    eval_parser = subparsers.add_parser("eval", help="chấm checkpoint trên holdout")
    eval_parser.add_argument("--ckpt", required=True)
    eval_parser.add_argument("--holdout", nargs="+", required=True)
    args = parser.parse_args()

    if args.cmd == "invariance":
        print(json.dumps(session_invariance(load_originals(with_landmarks=False)), ensure_ascii=False, indent=1))
    elif args.cmd == "build":
        data = build_dataset(load_originals(), parse_holdout(args.holdout), args.mode, args.per_class,
                             np.random.default_rng(args.seed), args.strength)
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **data)
        print(f"{output_path}: {len(data['y'])} mẫu ({int(data['augmented'].sum())} augment), {len(data['classes'])} lớp")
    elif args.cmd == "train":
        _train(args)
    elif args.cmd == "eval":
        from src.model.predict import VSLPredictor
        report = evaluate_holdout(VSLPredictor(args.ckpt), load_originals(with_landmarks=False),
                                  parse_holdout(args.holdout))
        print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
