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
    out = []
    for spec in specs or []:
        cls, sep, session = spec.partition("@")
        if not sep or not session:
            raise ValueError(f"Holdout phải dạng lớp@session hoặc @session: {spec!r}")
        out.append((cls or None, session))
    return out


def is_held_out(clip: Clip, holdout: list[tuple[str | None, str]]) -> bool:
    return any((cls is None or cls == clip.cls) and clip.session.startswith(session) for cls, session in holdout)


def load_originals(processed_root: Path | None = None, with_landmarks: bool = True) -> list[Clip]:
    root = Path(processed_root) if processed_root else get_data_path() / "processed"
    clips = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for sample_dir in sorted((p for p in class_dir.iterdir() if p.is_dir() and p.name.isdigit()),
                                 key=lambda p: int(p.name)):
            meta = DataHandler._read_meta(sample_dir)
            if "augmentation" in meta:
                continue
            paths, problem = DataHandler._frame_paths(sample_dir)
            if problem:
                continue
            features = np.stack([DataHandler._load_feature(p) for p in paths])
            landmarks = load_source_landmarks(sample_dir, len(features)) if with_landmarks else None
            clips.append(Clip(class_dir.name, str(meta.get("session", "legacy")), sample_dir.name, features, landmarks))
    return clips


def _pool(sequence: np.ndarray) -> np.ndarray:
    return np.concatenate([sequence.mean(0), sequence.std(0)])


def session_invariance(clips: list[Clip], day_chars: int = 8) -> dict:
    pooled = np.stack([_pool(process_sequence(c.features)) for c in clips])
    pooled = (pooled - pooled.mean(0)) / (pooled.std(0) + 1e-6)
    days = np.array([c.session[:day_chars] for c in clips])
    labels = np.array([c.cls for c in clips])
    report = {}
    for cls in sorted(set(labels)):
        cls_days = sorted(set(days[labels == cls]))
        if len(cls_days) < 2:
            continue
        hits = total = 0
        for day in cls_days:
            query = (labels == cls) & (days == day)
            ref = ~query & ((labels != cls) | (days != day))
            for v in pooled[query]:
                nearest = np.argmin(np.linalg.norm(pooled[ref] - v, axis=1))
                hits += labels[ref][nearest] == cls
                total += 1
        centers = {d: pooled[(labels == cls) & (days == d)].mean(0) for d in cls_days}
        others = [pooled[labels == o].mean(0) for o in sorted(set(labels)) if o != cls]
        within = np.mean([np.linalg.norm(centers[a] - centers[b])
                          for i, a in enumerate(cls_days) for b in cls_days[i + 1:]])
        between = np.mean([min(np.linalg.norm(c - o) for o in others) for c in centers.values()])
        report[cls] = {"sessions": cls_days, "cross_session_nn": round(hits / total, 3),
                       "ratio": round(float(within / between), 3)}
    return report


def build_dataset(clips: list[Clip], holdout: list[tuple[str | None, str]], mode: str, per_class: int,
                  rng: np.random.Generator, strength: float = 1.0, pitch_degrees: float = PITCH_AUG_DEG) -> dict:
    if mode not in ("landmark", "feature", "none"):
        raise ValueError("mode phải là landmark, feature hoặc none")
    train = [c for c in clips if not is_held_out(c, holdout)]
    by_class: dict[str, list[Clip]] = defaultdict(list)
    for c in train:
        by_class[c.cls].append(c)
    classes = sorted(by_class)
    X, y, groups, sessions, augmented = [], [], [], [], []

    def add(clip: Clip, raw: np.ndarray, is_aug: bool) -> None:
        X.append(process_sequence(raw))
        y.append(classes.index(clip.cls))
        groups.append(f"{clip.cls}/{clip.sample_id}")
        sessions.append(clip.session)
        augmented.append(is_aug)

    for cls in classes:
        members = by_class[cls]
        for clip in members:
            add(clip, clip.features, False)
        if mode == "none":
            continue
        for i in range(max(0, per_class - len(members))):
            clip = members[i % len(members)]
            if mode == "landmark" and clip.landmarks is None:
                raise ValueError(f"{cls}/{clip.sample_id} chưa có landmarks.npz: chạy reextract trước")
            landmarks = clip.landmarks if mode == "landmark" else None
            raw, _ = make_variant(clip.features, landmarks, rng, strength, pitch_degrees)
            add(clip, raw, True)
    return {"X": np.stack(X).astype(np.float32), "y": np.asarray(y, dtype=np.int64), "classes": np.asarray(classes),
            "groups": np.asarray(groups), "sessions": np.asarray(sessions), "augmented": np.asarray(augmented)}


def evaluate_holdout(predictor, clips: list[Clip], holdout: list[tuple[str | None, str]]) -> dict:
    test = [c for c in clips if is_held_out(c, holdout)]
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
            "unseen_classes": sorted({c.cls for c in test} - known),
            "predictions": {cls: dict(counter.most_common()) for cls, counter in sorted(per_class.items())}}


def _train(args) -> None:
    from src.model import training
    cfg_kwargs = {}
    if args.config_from:
        cfg_kwargs = json.loads(Path(args.config_from).read_text(encoding="utf-8"))["config"]
        cfg_kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg_kwargs.items()}
    if args.epochs:
        cfg_kwargs["max_epochs"] = args.epochs
    if args.seed is not None:
        cfg_kwargs["seed"] = args.seed
    cfg = training.Config(**cfg_kwargs)
    X, y, classes, groups = training.load_dataset(args.data)
    mask = training.load_validation_mask(args.data, len(y), groups)
    training.run_training(X, y, classes, groups, out_dir=args.out, cfg=cfg, folds=args.folds,
                          folds_to_run=args.folds_to_run, add_full=True, export_onnx_model=False,
                          validation_mask=mask)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark xuyên bối cảnh quay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("invariance", help="đo bất biến bối cảnh của feature (không train)")
    b = sub.add_parser("build", help="dựng .npz train bỏ holdout")
    b.add_argument("--mode", choices=["landmark", "feature", "none"], default="landmark")
    b.add_argument("--holdout", nargs="*", default=[])
    b.add_argument("--per-class", type=int, default=130)
    b.add_argument("--strength", type=float, default=1.0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    t = sub.add_parser("train", help="train trên .npz đã dựng")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config-from", help="metrics.json của một run để dùng lại Config")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--folds-to-run", type=int, default=1)
    t.add_argument("--epochs", type=int, default=0)
    t.add_argument("--seed", type=int, default=None, help="ghi đè seed của Config")
    e = sub.add_parser("eval", help="chấm checkpoint trên holdout")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--holdout", nargs="+", required=True)
    args = ap.parse_args()

    if args.cmd == "invariance":
        print(json.dumps(session_invariance(load_originals(with_landmarks=False)), ensure_ascii=False, indent=1))
    elif args.cmd == "build":
        data = build_dataset(load_originals(), parse_holdout(args.holdout), args.mode, args.per_class,
                             np.random.default_rng(args.seed), args.strength)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **data)
        print(f"{out}: {len(data['y'])} mẫu ({int(data['augmented'].sum())} augment), {len(data['classes'])} lớp")
    elif args.cmd == "train":
        _train(args)
    elif args.cmd == "eval":
        from src.model.predict import VSLPredictor
        report = evaluate_holdout(VSLPredictor(args.ckpt), load_originals(with_landmarks=False),
                                  parse_holdout(args.holdout))
        print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
