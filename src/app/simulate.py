"""So sánh nhánh thường và nhánh nhanh trên luồng ký mô phỏng (không cần camera/MediaPipe).

Ghép các clip gốc (không augment) trong ``data/processed`` thành chuỗi nhiều từ, xen giữa là
0..``max_gap`` frame không có tay (gap ngắn hơn ``no_hand_timeout`` -> cửa sổ dính đoạn chuyển
tiếp giữa hai từ), rồi phát lại ở 15 fps qua ``RealtimeTranslator`` / ``FastRealtimeTranslator``.

``--holdout`` (mặc định khi checkpoint có ``folds.json``): với mỗi fold k chỉ dùng model fold k
và chỉ phát các clip thuộc fold k -> mọi con số đều trên dữ liệu model CHƯA thấy. Khi đó model
nhanh và "ensemble" xác nhận là cùng một model, nên chỉ đo được tác dụng của luật thoát sớm;
lớp xác nhận bằng ensemble thật chỉ làm nhánh nhanh an toàn hơn.

Chạy: ``.venv/bin/python -m src.app.simulate [--checkpoint runs/.../ensemble.pt] [--streams 20]``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.app.fast import FastConfig, FastRealtimeTranslator, FastStreamSession, make_fast_ensemble
from src.app.realtime import RealtimeTranslator
from src.app.sentence import SentenceComposer
from src.app.voting import VoteConfig
from src.data.data_reader import DataHandler
from src.features.extract import FEATURE_DIM
from src.model.predict import StreamConfig, VSLPredictor, VSLStreamSession

FPS = 15.0


class _NoLandmarkers:
    def close(self) -> None:
        pass


class _Replay:
    """Nhận đặc trưng thô thay cho frame BGR (``RealtimeTranslator.process`` gọi ``update_frame``)."""

    def update_frame(self, feature, *, now=None):
        return self.update_features(feature, now=now)


class ReplaySession(_Replay, VSLStreamSession):
    pass


class FastReplaySession(_Replay, FastStreamSession):
    pass


def load_original_samples(data_dir: Path, classes, folds: dict | None) -> list[tuple[str, int, np.ndarray]]:
    """[(lớp, fold, đặc trưng thô (frames, FEATURE_DIM))] của các mẫu gốc."""
    samples = []
    for name in classes:
        class_dir = data_dir / name
        if not class_dir.is_dir():
            continue
        for sample_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            meta = DataHandler._read_meta(sample_dir)
            if "augmentation" in meta:
                continue
            paths, problem = DataHandler._frame_paths(sample_dir)
            if problem:
                continue
            key = f"{name}/{meta.get('origin_sample_id', sample_dir.name)}"
            if folds is not None and key not in folds:
                continue
            raw = np.stack([DataHandler._load_feature(p) for p in paths])
            samples.append((name, -1 if folds is None else int(folds[key]), raw))
    return samples


def build_stream(samples, rng, words: int, max_gap: int):
    """Trả về (đặc trưng (N, D), [(nhãn, frame bắt đầu)]) — không có hai từ liền nhau trùng nhãn."""
    rest = np.full((1, FEATURE_DIM), np.nan, dtype=np.float32)
    parts, spans, cursor, previous = [np.repeat(rest, 12, 0)], [], 12, None
    for _ in range(words):
        choices = [i for i, s in enumerate(samples) if s[0] != previous]
        label, _fold, raw = samples[rng.choice(choices)]
        spans.append((label, cursor))
        gap = int(rng.integers(0, max_gap + 1))
        parts += [raw, np.repeat(rest, gap, 0)]
        cursor += len(raw) + gap
        previous = label
    parts.append(np.repeat(rest, 12, 0))
    return np.concatenate(parts), spans


def score(commits, spans) -> dict:
    """Căn chuỗi từ đã chốt với chuỗi đúng (Levenshtein, như WER).

    commits: [(frame, nhãn)]; spans: [(nhãn, frame bắt đầu)]. Độ trễ của một từ khớp = frame
    chốt - frame bắt đầu của từ đó.
    """
    hyp, ref = [label for _f, label in commits], [label for label, _s in spans]
    n, m = len(ref), len(hyp)
    cost = np.zeros((n + 1, m + 1), dtype=np.int64)
    cost[:, 0], cost[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j] + 1, cost[i, j - 1] + 1,
                             cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, counts, latencies = n, m, {"sub": 0, "del": 0, "ins": 0}, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                latencies.append((commits[j - 1][0] - spans[i - 1][1]) / FPS)
            else:
                counts["sub"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            counts["del"] += 1
            i -= 1
        else:
            counts["ins"] += 1
            j -= 1
    return {"words": n, "commits": m, "correct": len(latencies), **counts, "latencies": latencies}


def run_stream(translator, features) -> tuple[list, float]:
    commits, spent = [], 0.0
    for t, feature in enumerate(features):
        start = time.perf_counter()
        result = translator.process(feature, now=t / FPS)
        spent += time.perf_counter() - start
        if result["committed"]:
            commits.append((t, result["committed"]))
    translator.close()
    return commits, spent / len(features)


def summarize(name: str, rows: list[dict], ms: list[float], stats: dict | None = None) -> dict:
    total = {key: sum(r[key] for r in rows) for key in ("words", "commits", "correct", "sub", "del", "ins")}
    lat = np.array([x for r in rows for x in r["latencies"]])
    out = {
        "branch": name,
        "wer": (total["sub"] + total["del"] + total["ins"]) / max(total["words"], 1),
        "precision": total["correct"] / max(total["commits"], 1),
        "recall": total["correct"] / max(total["words"], 1),
        **{k: total[k] for k in ("sub", "del", "ins")},
        "latency_mean_s": float(lat.mean()) if lat.size else float("nan"),
        "latency_median_s": float(np.median(lat)) if lat.size else float("nan"),
        "model_ms_per_frame": float(np.mean(ms) * 1e3),
    }
    if stats:
        out["calls"] = stats
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="So sánh nhánh thường / nhánh nhanh trên luồng mô phỏng")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--streams", type=int, default=20, help="số luồng mỗi fold")
    parser.add_argument("--words", type=int, default=5, help="số từ mỗi luồng")
    parser.add_argument("--max-gap", type=int, default=12, help="số frame buông tay tối đa giữa hai từ")
    parser.add_argument("--no-holdout", action="store_true", help="dùng cả ensemble (đã thấy dữ liệu)")
    parser.add_argument("--min-consecutive", type=int, default=5)
    parser.add_argument("--early-window", type=float, default=FastConfig.early_window_seconds)
    parser.add_argument("--early-votes", type=int, default=FastConfig.early_votes)
    parser.add_argument("--strong-confidence", type=float, default=FastConfig.strong_confidence)
    parser.add_argument("--strong-gap", type=float, default=FastConfig.strong_gap)
    parser.add_argument("--escalate", action="store_true", help="chạy ensemble mỗi khi model nhanh từ chối")
    parser.add_argument("--no-confirm", action="store_true", help="không xác nhận bằng ensemble trước khi chốt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    predictor = VSLPredictor(args.checkpoint, device=args.device)
    full_ensemble = predictor.ensemble
    folds_path = predictor.checkpoint_path.parent / "folds.json"
    holdout = not args.no_holdout and folds_path.is_file()
    folds = json.loads(folds_path.read_text(encoding="utf-8"))["fold_by_group"] if holdout else None
    samples = load_original_samples(args.data, predictor.classes, folds)
    if not samples:
        raise SystemExit(f"Không có mẫu gốc nào trong {args.data}")
    print(f"{predictor.checkpoint_path} | {len(samples)} mẫu gốc | holdout={holdout}")

    stream_config = StreamConfig(predict_every=1)
    vote_config = VoteConfig(min_consecutive=args.min_consecutive)
    fast_config = FastConfig(early_window_seconds=args.early_window, early_votes=args.early_votes,
                             strong_confidence=args.strong_confidence, strong_gap=args.strong_gap,
                             escalate=args.escalate, confirm=not args.no_confirm)
    rng = np.random.default_rng(args.seed)
    groups = sorted({s[1] for s in samples})
    rows = {"base": [], "fast": []}
    ms = {"base": [], "fast": []}
    stats = {"fast": 0, "ensemble": 0, "confirm": 0, "confirm_rejected": 0}
    for fold in groups:
        fold_samples = [s for s in samples if s[1] == fold]
        if holdout:
            predictor.ensemble = full_ensemble
            predictor.ensemble = make_fast_ensemble(predictor, fold)
        fast_ensemble = make_fast_ensemble(predictor, 0 if holdout else fast_config.fast_model)
        for _ in range(args.streams):
            features, spans = build_stream(fold_samples, rng, args.words, args.max_gap)
            base = RealtimeTranslator(
                predictor, vote_config=vote_config, composer=SentenceComposer(None, background=False),
                session=ReplaySession(predictor, stream_config, landmarkers=_NoLandmarkers()))
            session = FastReplaySession(predictor, stream_config, fast_config, landmarkers=_NoLandmarkers(),
                                        fast_ensemble=fast_ensemble)
            fast = FastRealtimeTranslator(predictor, fast_config=fast_config, vote_config=vote_config,
                                          composer=SentenceComposer(None, background=False), session=session)
            for name, translator in (("base", base), ("fast", fast)):
                commits, spent = run_stream(translator, features)
                rows[name].append(score(commits, spans))
                ms[name].append(spent)
            for key, value in session.stats.items():
                stats[key] += value
    predictor.ensemble = full_ensemble

    results = [summarize("base", rows["base"], ms["base"]), summarize("fast", rows["fast"], ms["fast"], stats)]
    for r in results:
        print(f'{r["branch"]:5s} WER={r["wer"]:.3f} precision={r["precision"]:.3f} recall={r["recall"]:.3f} '
              f'(sub={r["sub"]} del={r["del"]} ins={r["ins"]}) latency mean={r["latency_mean_s"]:.2f}s '
              f'median={r["latency_median_s"]:.2f}s model={r["model_ms_per_frame"]:.2f}ms/frame')
    print("fast calls:", stats)


if __name__ == "__main__":
    main()
