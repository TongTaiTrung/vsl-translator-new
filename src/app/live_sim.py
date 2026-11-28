"""Replay recorded clips through the web streaming flow with simulated hand raise/lower.

Usage: python -m src.app.live_sim [--checkpoint runs/best/ensemble.pt] [--view hard] [--still]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.app.fast import make_fast_ensemble
from src.data.data_reader import DataHandler
from src.features.extract import active_presence
from src.features.landmarks import load_landmarks
from src.features.view import features_from_landmarks, random_view, simulate_view
from src.model.predict import VSLPredictor

PRESETS = {
    "accurate": (30, 30, 5, 12),
    "fast": (20, 12, 3, 8),
    "fastest": (15, 8, 2, 6),
}
VIEWS = {
    "train": {},
    "hard": dict(yaw=30.0, pitch=18.0, roll=8.0, distance=(0.6, 1.8), hand_size=(0.85, 1.15), shift=0.2,
                 finger_sigma=12.0),
}
IDLE_HAND_FRAMES = 3
MIN_CONSECUTIVE = 2
_POSE_WRISTS = (15, 16)
_POSE_HAND_POINTS = {15: (17, 19, 21), 16: (18, 20, 22)}
_POSE_ELBOWS = {15: 13, 16: 14}


def _ease(steps: int) -> np.ndarray:
    progress = np.arange(1, steps + 1) / (steps + 1)
    return 1.0 - (3 * progress ** 2 - 2 * progress ** 3)


def _lift(pose_t, hands_t, world_t, y_offsets: np.ndarray):
    step_count = len(y_offsets)
    pose = np.repeat(pose_t[None], step_count, 0).copy()
    hands = np.repeat(hands_t[None], step_count, 0).copy()
    world = np.repeat(world_t[None], step_count, 0).copy()
    for hand_index in range(2):
        if not np.isfinite(hands_t[hand_index]).all():
            continue
        wrist = _POSE_WRISTS[int(np.argmin([np.linalg.norm(hands_t[hand_index, 0, :2] - pose_t[step, :2]) for step in _POSE_WRISTS]))]
        for step in range(step_count):
            hands[step, hand_index, :, 1] += y_offsets[step]
            pose[step, [wrist, *_POSE_HAND_POINTS[wrist]], 1] += y_offsets[step]
            pose[step, _POSE_ELBOWS[wrist], 1] += 0.5 * y_offsets[step]
            if (hands[step, hand_index, :, 1] > 1.0).mean() > 0.5:
                hands[step, hand_index] = np.nan
                world[step, hand_index] = np.nan
    return pose, hands, world


def retime(pose, hands, world, *, speed: float = 1.0, pre_hold: int = 0, post_hold: int | None = None,
           still: bool = False):
    frame_count = len(pose)
    if still:
        tip = hands[:, :, 8, :2]
        step = np.linalg.norm(np.diff(tip, axis=0), axis=2)
        step = np.where(np.isfinite(step), step, np.inf).min(1)
        calm = np.concatenate([[np.inf], step]) + np.concatenate([step, [np.inf]])
        calm_frame = int(np.argmin(calm))
        return tuple(np.repeat(array[calm_frame:calm_frame + 1], frame_count, 0) for array in (pose, hands, world))
    idx = np.round(np.arange(0, frame_count - 1e-9, speed)).astype(int).clip(0, frame_count - 1)
    pose, hands, world = pose[idx], hands[idx], world[idx]
    if post_hold is not None:
        tip = hands[:, :, 8, :2]
        speed_t = np.nan_to_num(np.linalg.norm(np.diff(tip, axis=0), axis=2)).max(1)
        if len(speed_t):
            fast = np.flatnonzero(speed_t >= 0.3 * speed_t.max())
            end = int(fast[-1]) + 1 + post_hold
            pose, hands, world = pose[:end + 1], hands[:end + 1], world[:end + 1]
    if pre_hold:
        pose, hands, world = (np.concatenate([np.repeat(array[:1], pre_hold, 0), array]) for array in (pose, hands, world))
    return pose, hands, world


def build_live(pose, hands, world, *, entry: int, exit_: int, pre: int, post: int):
    has = np.isfinite(hands[:, :, 0, 0]).any(1) & np.isfinite(pose[:, 0, 0])
    if not has.any():
        return None
    first, last = int(np.flatnonzero(has)[0]), int(np.flatnonzero(has)[-1])
    parts = []

    def no_hands(frame_index, frame_count):
        return (np.repeat(pose[frame_index][None], frame_count, 0), np.full((frame_count, 2, 21, 3), np.nan, np.float32),
                np.full((frame_count, 2, 21, 3), np.nan, np.float32))

    def drop(frame_index):
        top_y = np.nanmin(hands[frame_index, :, 0, 1])
        return (1.15 - top_y)

    parts.append(no_hands(first, pre))
    if entry:
        parts.append(_lift(pose[first], hands[first], world[first], drop(first) * _ease(entry)))
    parts.append((pose, hands, world))
    if exit_:
        parts.append(_lift(pose[last], hands[last], world[last], drop(last) * _ease(exit_)[::-1]))
    parts.append(no_hands(last, post))
    return tuple(np.concatenate([part[component_index] for part in parts]).astype(np.float32) for component_index in range(3))


def replay(predictor, ensemble, features, preset) -> list[str]:
    predict_frames, min_frames, stride, min_hand = preset
    buffer: list[np.ndarray] = []
    since = None
    candidate, count, blocked = None, 0, None
    committed = []
    for feature in features:
        buffer.append(feature)
        buffer = buffer[-30:]
        since = 0 if since is None else since + 1
        if since % stride != stride - 1 or len(buffer) < min_frames:
            continue
        window = np.stack(buffer[-predict_frames:])
        hands = int(sum(active_presence(frame).any() for frame in window))
        if hands <= IDLE_HAND_FRAMES:
            candidate, count, blocked = None, 0, None
            continue
        if hands < min_hand:
            candidate, count = None, 0
            continue
        result = predictor.predict_raw(window, ensemble=ensemble)
        label = result["label"] if result["accepted"] else None
        if label is None:
            candidate, count = None, 0
            continue
        count = count + 1 if label == candidate else 1
        candidate = label
        if label != blocked and count >= MIN_CONSECUTIVE:
            committed.append(label)
            blocked, candidate, count = label, None, 0
            buffer, since = [], None
    return committed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mô phỏng luồng web có động tác nâng/hạ tay")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="accurate")
    parser.add_argument("--entry", type=int, default=6, help="số frame nâng tay (0 = tay xuất hiện ngay)")
    parser.add_argument("--exit", type=int, default=6, help="số frame hạ tay")
    parser.add_argument("--speed", type=float, default=1.0, help="ký nhanh hơn chừng này lần")
    parser.add_argument("--pre-hold", type=int, default=0, help="giữ tư thế đầu thêm chừng này frame")
    parser.add_argument("--post-hold", type=int, default=None, help="chỉ giữ chừng này frame sau nét ký")
    parser.add_argument("--still", action="store_true", help="giữ tay yên ở tư thế của clip, không vẩy")
    parser.add_argument("--view", choices=sorted(VIEWS), default=None, help="mô phỏng bối cảnh quay khác")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="*", default=None, help="chỉ các lớp này")
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    predictor = VSLPredictor(args.checkpoint, device="cpu")
    folds_path = predictor.checkpoint_path.parent / "folds.json"
    folds = None if args.no_holdout else json.loads(folds_path.read_text(encoding="utf-8"))["fold_by_group"]
    ensembles = {}
    stats: dict[str, Counter] = defaultdict(Counter)
    wrong: dict[str, Counter] = defaultdict(Counter)
    for name in args.classes or predictor.classes:
        class_dir = args.data / name
        if not class_dir.is_dir():
            continue
        for sample_dir in sorted(entry for entry in class_dir.iterdir() if entry.is_dir()):
            meta = DataHandler._read_meta(sample_dir)
            lm_path = sample_dir / "landmarks.npz"
            if "augmentation" in meta or not lm_path.is_file():
                continue
            key = f"{name}/{sample_dir.name}"
            if folds is not None and key not in folds:
                continue
            fold = -1 if folds is None else int(folds[key])
            if fold not in ensembles:
                ensembles[fold] = predictor.ensemble if fold < 0 else make_fast_ensemble(predictor, fold)
            clip_landmarks = load_landmarks(lm_path)
            if args.view:
                rng = np.random.default_rng([args.seed, *map(ord, key)])
                clip_landmarks = simulate_view(*clip_landmarks, random_view(rng, **VIEWS[args.view]))
            clip_landmarks = retime(*clip_landmarks, speed=args.speed, pre_hold=args.pre_hold,
                        post_hold=args.post_hold, still=args.still)
            live = build_live(*clip_landmarks, entry=args.entry, exit_=args.exit, pre=8, post=12)
            if live is None:
                continue
            words = replay(predictor, ensembles[fold], features_from_landmarks(*live), PRESETS[args.preset])
            first = words[0] if words else None
            stats[name]["n"] += 1
            stats[name]["first_ok" if first == name else "none" if first is None else "first_wrong"] += 1
            stats[name]["any_ok"] += name in words
            stats[name]["extra"] += sum(word != name for word in words)
            if first not in (None, name):
                wrong[name][first] += 1

    total = Counter()
    print(f"{'lớp':12s}  n  đúng  sai  bỏ sót  thừa  nhầm thành")
    for name, class_stats in stats.items():
        total.update(class_stats)
        print(f"{name:12s} {class_stats['n']:2d} {class_stats['first_ok'] / class_stats['n']:5.0%} {class_stats['first_wrong']:4d} {class_stats['none']:6d} "
              f"{class_stats['extra']:5d}  {dict(wrong[name].most_common(3))}")
    print(f"TỔNG: {total['first_ok']}/{total['n']} đúng ({total['first_ok'] / max(total['n'], 1):.1%}), "
          f"sai {total['first_wrong']}, bỏ sót {total['none']}, từ thừa {total['extra']}")
    if args.json:
        args.json.write_text(json.dumps({class_label: dict(counter) for class_label, counter in stats.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
