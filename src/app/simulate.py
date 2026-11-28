"""Compare the regular and fast branches on simulated multi-word signing streams.

Usage: python -m src.app.simulate [--checkpoint runs/best/ensemble.pt] [--streams 20]
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

FRAMES_PER_SECOND = 15.0


class _NoLandmarkers:
    def close(self) -> None:
        pass


class _Replay:

    def update_frame(self, feature, *, now=None):
        return self.update_features(feature, now=now)


class ReplaySession(_Replay, VSLStreamSession):
    pass


class FastReplaySession(_Replay, FastStreamSession):
    pass


def load_original_samples(data_dir: Path, classes, folds: dict | None) -> list[tuple[str, int, np.ndarray]]:
    samples = []
    for name in classes:
        class_dir = data_dir / name
        if not class_dir.is_dir():
            continue
        for sample_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            meta = DataHandler._read_meta(sample_dir)
            if "augmentation" in meta:
                continue
            paths, problem = DataHandler._frame_paths(sample_dir)
            if problem:
                continue
            key = f"{name}/{meta.get('origin_sample_id', sample_dir.name)}"
            if folds is not None and key not in folds:
                continue
            raw = np.stack([DataHandler._load_feature(path) for path in paths])
            samples.append((name, -1 if folds is None else int(folds[key]), raw))
    return samples


def build_stream(samples, rng, words: int, max_gap: int):
    rest = np.full((1, FEATURE_DIM), np.nan, dtype=np.float32)
    parts, spans, cursor, previous = [np.repeat(rest, 12, 0)], [], 12, None
    for _ in range(words):
        choices = [index for index, sample in enumerate(samples) if sample[0] != previous]
        label, _fold, raw = samples[rng.choice(choices)]
        spans.append((label, cursor))
        gap = int(rng.integers(0, max_gap + 1))
        parts += [raw, np.repeat(rest, gap, 0)]
        cursor += len(raw) + gap
        previous = label
    parts.append(np.repeat(rest, 12, 0))
    return np.concatenate(parts), spans


def score(commits, spans) -> dict:
    hyp, ref = [label for _f, label in commits], [label for label, _s in spans]
    ref_len, hyp_len = len(ref), len(hyp)
    cost = np.zeros((ref_len + 1, hyp_len + 1), dtype=np.int64)
    cost[:, 0], cost[0, :] = np.arange(ref_len + 1), np.arange(hyp_len + 1)
    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            cost[ref_index, hyp_index] = min(cost[ref_index - 1, hyp_index] + 1, cost[ref_index, hyp_index - 1] + 1,
                             cost[ref_index - 1, hyp_index - 1] + (ref[ref_index - 1] != hyp[hyp_index - 1]))
    ref_index, hyp_index, counts, latencies = ref_len, hyp_len, {"sub": 0, "del": 0, "ins": 0}, []
    while ref_index > 0 or hyp_index > 0:
        if ref_index > 0 and hyp_index > 0 and cost[ref_index, hyp_index] == cost[ref_index - 1, hyp_index - 1] + (ref[ref_index - 1] != hyp[hyp_index - 1]):
            if ref[ref_index - 1] == hyp[hyp_index - 1]:
                latencies.append((commits[hyp_index - 1][0] - spans[ref_index - 1][1]) / FRAMES_PER_SECOND)
            else:
                counts["sub"] += 1
            ref_index, hyp_index = ref_index - 1, hyp_index - 1
        elif ref_index > 0 and cost[ref_index, hyp_index] == cost[ref_index - 1, hyp_index] + 1:
            counts["del"] += 1
            ref_index -= 1
        else:
            counts["ins"] += 1
            hyp_index -= 1
    return {"words": ref_len, "commits": hyp_len, "correct": len(latencies), **counts, "latencies": latencies}


def run_stream(translator, features) -> tuple[list, float]:
    commits, spent = [], 0.0
    for frame_index, feature in enumerate(features):
        start = time.perf_counter()
        result = translator.process(feature, now=frame_index / FRAMES_PER_SECOND)
        spent += time.perf_counter() - start
        if result["committed"]:
            commits.append((frame_index, result["committed"]))
    translator.close()
    return commits, spent / len(features)


def summarize(name: str, rows: list[dict], model_seconds: list[float], stats: dict | None = None) -> dict:
    total = {key: sum(row[key] for row in rows) for key in ("words", "commits", "correct", "sub", "del", "ins")}
    latencies = np.array([latency for row in rows for latency in row["latencies"]])
    summary = {
        "branch": name,
        "wer": (total["sub"] + total["del"] + total["ins"]) / max(total["words"], 1),
        "precision": total["correct"] / max(total["commits"], 1),
        "recall": total["correct"] / max(total["words"], 1),
        **{metric: total[metric] for metric in ("sub", "del", "ins")},
        "latency_mean_s": float(latencies.mean()) if latencies.size else float("nan"),
        "latency_median_s": float(np.median(latencies)) if latencies.size else float("nan"),
        "model_ms_per_frame": float(np.mean(model_seconds) * 1e3),
    }
    if stats:
        summary["calls"] = stats
    return summary


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
    groups = sorted({sample[1] for sample in samples})
    rows = {"base": [], "fast": []}
    model_seconds = {"base": [], "fast": []}
    stats = {"fast": 0, "ensemble": 0, "confirm": 0, "confirm_rejected": 0}
    for fold in groups:
        fold_samples = [sample for sample in samples if sample[1] == fold]
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
                model_seconds[name].append(spent)
            for key, value in session.stats.items():
                stats[key] += value
    predictor.ensemble = full_ensemble

    results = [summarize("base", rows["base"], model_seconds["base"]), summarize("fast", rows["fast"], model_seconds["fast"], stats)]
    for result in results:
        print(f'{result["branch"]:5s} WER={result["wer"]:.3f} precision={result["precision"]:.3f} recall={result["recall"]:.3f} '
              f'(sub={result["sub"]} del={result["del"]} ins={result["ins"]}) latency mean={result["latency_mean_s"]:.2f}s '
              f'median={result["latency_median_s"]:.2f}s model={result["model_ms_per_frame"]:.2f}ms/frame')
    print("fast calls:", stats)


if __name__ == "__main__":
    main()
