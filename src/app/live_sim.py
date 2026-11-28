"""Mô phỏng luồng web (lib/recognizer.ts + /api/stream) trên clip gốc, CÓ động tác đưa tay lên và hạ tay.

Clip thu dữ liệu bắt đầu khi tay đã ở tư thế ký (bấm SPACE lúc tay đã giơ lên), còn trên camera thật cửa sổ
2 giây dính cả cú nâng/hạ tay, và người ký vẩy sớm/muộn, giữ tay ngắn/dài tuỳ lúc. Mỗi clip gốc (có
``landmarks.npz``) được ghép thành một luồng: vài frame không có tay → tay nâng từ mép dưới khung lên tư thế đầu
clip → clip → tay hạ xuống → không có tay, rồi phát lại ở 15 fps qua cùng luật cửa sổ / đếm tay / bỏ phiếu như web.
Đặc trưng tính lại bằng ``extract`` từ landmark nên giống hệt pipeline thật. Mỗi clip chỉ do model của fold chứa
nó dự đoán (chưa thấy clip đó lúc train).

- ``--speed`` / ``--pre-hold`` / ``--post-hold``: ký nhanh hơn, vẩy muộn hơn, giữ tay ngắn sau nét.
- ``--view train|hard``: bối cảnh quay khác (góc camera, khoảng cách, cỡ tay, lệch khớp ngón; ``view.simulate_view``).
- ``--still``: giơ tay đúng tư thế của clip nhưng KHÔNG vẩy. Với lớp có nét, "đúng" ở chế độ này là nhận nhầm
  (model nhận theo tư thế chứ không theo nét); kết quả mong muốn là "bỏ sót".

Chạy: ``.venv/bin/python -m src.app.live_sim [--checkpoint runs/.../ensemble.pt] [--preset accurate]``
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
_POSE_HAND_PTS = {15: (17, 19, 21), 16: (18, 20, 22)}
_POSE_ELBOWS = {15: 13, 16: 14}


def _ease(n: int) -> np.ndarray:
    """n hệ số 1 → 0 (smoothstep), không gồm 0: frame cuối của đoạn nâng tay sát tư thế đầu clip."""
    s = np.arange(1, n + 1) / (n + 1)
    return 1.0 - (3 * s ** 2 - 2 * s ** 3)


def _lift(pose_t, hands_t, world_t, dy: np.ndarray):
    """Các frame của tư thế (pose_t, hands_t) với tay dịch xuống dy (đơn vị canvas) mỗi frame.

    Cổ tay + điểm ngón của pose dịch theo tay, khuỷu dịch nửa đoạn. Tay ra ngoài khung → mất (NaN).
    """
    n = len(dy)
    pose = np.repeat(pose_t[None], n, 0).copy()
    hands = np.repeat(hands_t[None], n, 0).copy()
    world = np.repeat(world_t[None], n, 0).copy()
    for k in range(2):
        if not np.isfinite(hands_t[k]).all():
            continue
        w = _POSE_WRISTS[int(np.argmin([np.linalg.norm(hands_t[k, 0, :2] - pose_t[i, :2]) for i in _POSE_WRISTS]))]
        for i in range(n):
            hands[i, k, :, 1] += dy[i]
            pose[i, [w, *_POSE_HAND_PTS[w]], 1] += dy[i]
            pose[i, _POSE_ELBOWS[w], 1] += 0.5 * dy[i]
            if (hands[i, k, :, 1] > 1.0).mean() > 0.5:
                hands[i, k] = np.nan
                world[i, k] = np.nan
    return pose, hands, world


def retime(pose, hands, world, *, speed: float = 1.0, pre_hold: int = 0, post_hold: int | None = None,
           still: bool = False):
    """Đổi nhịp clip: ký nhanh hơn ``speed`` lần (bỏ frame), giữ tư thế đầu thêm ``pre_hold`` frame,
    và chỉ giữ ``post_hold`` frame sau cú di chuyển mạnh nhất của đầu ngón trỏ (None = giữ hết).
    ``still``: thay cả clip bằng frame yên nhất lặp lại (giơ tay đúng tư thế nhưng KHÔNG vẩy) -> đo nhận nhầm."""
    T = len(pose)
    if still:
        tip = hands[:, :, 8, :2]
        step = np.linalg.norm(np.diff(tip, axis=0), axis=2)
        step = np.where(np.isfinite(step), step, np.inf).min(1)
        calm = np.concatenate([[np.inf], step]) + np.concatenate([step, [np.inf]])
        t = int(np.argmin(calm))
        return tuple(np.repeat(a[t:t + 1], T, 0) for a in (pose, hands, world))
    idx = np.round(np.arange(0, T - 1e-9, speed)).astype(int).clip(0, T - 1)
    pose, hands, world = pose[idx], hands[idx], world[idx]
    if post_hold is not None:
        tip = hands[:, :, 8, :2]
        speed_t = np.nan_to_num(np.linalg.norm(np.diff(tip, axis=0), axis=2)).max(1)
        if len(speed_t):
            fast = np.flatnonzero(speed_t >= 0.3 * speed_t.max())
            end = int(fast[-1]) + 1 + post_hold
            pose, hands, world = pose[:end + 1], hands[:end + 1], world[:end + 1]
    if pre_hold:
        pose, hands, world = (np.concatenate([np.repeat(a[:1], pre_hold, 0), a]) for a in (pose, hands, world))
    return pose, hands, world


def build_live(pose, hands, world, *, entry: int, exit_: int, pre: int, post: int):
    """Landmark của luồng: pre frame không tay, nâng tay, clip, hạ tay, post frame không tay."""
    has = np.isfinite(hands[:, :, 0, 0]).any(1) & np.isfinite(pose[:, 0, 0])
    if not has.any():
        return None
    first, last = int(np.flatnonzero(has)[0]), int(np.flatnonzero(has)[-1])
    parts = []

    def no_hands(t, n):
        return (np.repeat(pose[t][None], n, 0), np.full((n, 2, 21, 3), np.nan, np.float32),
                np.full((n, 2, 21, 3), np.nan, np.float32))

    def drop(t):
        y = np.nanmin(hands[t, :, 0, 1])
        return (1.15 - y)

    parts.append(no_hands(first, pre))
    if entry:
        parts.append(_lift(pose[first], hands[first], world[first], drop(first) * _ease(entry)))
    parts.append((pose, hands, world))
    if exit_:
        parts.append(_lift(pose[last], hands[last], world[last], drop(last) * _ease(exit_)[::-1]))
    parts.append(no_hands(last, post))
    return tuple(np.concatenate([p[i] for p in parts]).astype(np.float32) for i in range(3))


def replay(predictor, ensemble, features, preset) -> list[str]:
    """Các từ được chốt khi phát ``features`` (N, FEATURE_DIM) ở 15 fps qua luồng web."""
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
        hands = int(sum(active_presence(f).any() for f in window))
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
        for sample_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
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
            lm = load_landmarks(lm_path)
            if args.view:
                rng = np.random.default_rng([args.seed, *map(ord, key)])
                lm = simulate_view(*lm, random_view(rng, **VIEWS[args.view]))
            lm = retime(*lm, speed=args.speed, pre_hold=args.pre_hold,
                        post_hold=args.post_hold, still=args.still)
            live = build_live(*lm, entry=args.entry, exit_=args.exit, pre=8, post=12)
            if live is None:
                continue
            words = replay(predictor, ensembles[fold], features_from_landmarks(*live), PRESETS[args.preset])
            first = words[0] if words else None
            stats[name]["n"] += 1
            stats[name]["first_ok" if first == name else "none" if first is None else "first_wrong"] += 1
            stats[name]["any_ok"] += name in words
            stats[name]["extra"] += sum(w != name for w in words)
            if first not in (None, name):
                wrong[name][first] += 1

    total = Counter()
    print(f"{'lớp':12s}  n  đúng  sai  bỏ sót  thừa  nhầm thành")
    for name, s in stats.items():
        total.update(s)
        print(f"{name:12s} {s['n']:2d} {s['first_ok'] / s['n']:5.0%} {s['first_wrong']:4d} {s['none']:6d} "
              f"{s['extra']:5d}  {dict(wrong[name].most_common(3))}")
    print(f"TỔNG: {total['first_ok']}/{total['n']} đúng ({total['first_ok'] / max(total['n'], 1):.1%}), "
          f"sai {total['first_wrong']}, bỏ sót {total['none']}, từ thừa {total['extra']}")
    if args.json:
        args.json.write_text(json.dumps({k: dict(v) for k, v in stats.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
