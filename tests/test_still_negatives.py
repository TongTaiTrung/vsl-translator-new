"""Tests for still-negative training copies."""

import unittest

import numpy as np
import torch

from src.features import extract as fx
from src.model import training as tr


def _flick(start: int = 8, y_offset: float = -0.6, frame_count: int = 30, side: int = 1, frames: int = 4) -> np.ndarray:
    raw = np.full((frame_count, fx.FEATURE_DIM), np.nan, dtype=np.float32)
    raw[:, fx.FEATURE_SLICES["presence"]] = 0.0
    hand_pos_start = fx.FEATURE_SLICES["hand_pos"].start + 9 * side
    x = 0.3 if side else -0.3
    rng = np.random.default_rng(start)
    labels = 0.2 + y_offset * np.clip((np.arange(frame_count) - start) / frames, 0.0, 1.0) + rng.normal(0, 0.002, frame_count)
    for k, off in enumerate((0.0, 0.1, -0.2)):
        raw[:, hand_pos_start + 3 * k], raw[:, hand_pos_start + 3 * k + 1], raw[:, hand_pos_start + 3 * k + 2] = x, labels + off, 0.0
    raw[:, fx.PRESENCE_SLICE.start + side] = 1.0
    return raw


class StillVersionTests(unittest.TestCase):
    def test_flick_gives_holds_without_the_stroke(self):
        seq = fx.process_sequence(_flick())
        self.assertGreater(fx.stroke_magnitude(seq), 0.5)
        versions = fx.still_versions(seq)
        self.assertEqual(len(versions), 3)
        for v in versions:
            self.assertEqual(v.shape, (30, fx.SEQ_FEATURE_DIM))
            self.assertLess(fx.stroke_magnitude(v), fx.STILL_MAX_STROKE)
            self.assertLess(np.abs(v[:, fx.MOTION_SLICE]).max(), 0.2)
        tip_y = fx.FEATURE_SLICES["hand_pos"].start + 9 + 7
        before, after = versions[0][0, tip_y], versions[1][0, tip_y]
        self.assertAlmostEqual(before, seq[0, tip_y], delta=0.02)
        self.assertAlmostEqual(after, seq[-1, tip_y], delta=0.02)

    def test_flick_at_the_start_keeps_only_the_hold_after_it(self):
        versions = fx.still_versions(fx.process_sequence(_flick(start=0)))
        self.assertEqual(len(versions), 2)

    def test_static_or_missing_hand_gives_nothing(self):
        self.assertEqual(fx.still_versions(fx.process_sequence(_flick(y_offset=0.0))), [])
        empty = np.full((30, fx.FEATURE_DIM), np.nan, dtype=np.float32)
        empty[:, fx.PRESENCE_SLICE] = 0.0
        self.assertEqual(fx.still_versions(fx.process_sequence(empty)), [])

    def test_accepts_raw_columns_only(self):
        self.assertEqual(len(fx.still_versions(_flick())), 3)


class DynamicClassTests(unittest.TestCase):
    def test_classes_split_by_median_stroke(self):
        sequences = np.stack([fx.process_sequence(_flick(start=s, y_offset=y_offset)) for y_offset in (-0.6, 0.0) for s in (5, 10, 15)])
        labels = np.repeat([0, 1], 3)
        np.testing.assert_array_equal(tr.dynamic_classes(sequences, labels, 3, 0.25), [True, False, False])
        self.assertFalse(tr.dynamic_classes(sequences[..., :fx.FEATURE_DIM], labels, 3, 0.25).any())

    def test_pool_is_capped_and_empty_for_static_clips(self):
        sequences = np.stack([fx.process_sequence(_flick(start=s)) for s in (5, 10, 15)])
        self.assertEqual(tr.still_negative_pool(sequences).shape, (9, 30, fx.SEQ_FEATURE_DIM))
        self.assertEqual(len(tr.still_negative_pool(sequences, max_count=4)), 4)
        static = np.stack([fx.process_sequence(_flick(y_offset=0.0))])
        self.assertEqual(tr.still_negative_pool(static).shape, (0, 30, fx.SEQ_FEATURE_DIM))


class UniformLossTests(unittest.TestCase):
    def test_zero_for_flat_logits_and_positive_for_confident_ones(self):
        self.assertAlmostEqual(tr.uniform_target_loss(torch.zeros(4, 5)).item(), 0.0, places=6)
        self.assertGreater(tr.uniform_target_loss(torch.tensor([[6.0, 0, 0, 0, 0]])).item(), 1.0)

    def test_negatives_push_the_model_towards_uniform_output(self):
        rng = np.random.default_rng(0)
        labels = np.arange(24) % 3
        sequences = (rng.normal(size=(24, 30, fx.SEQ_FEATURE_DIM)) + labels[:, None, None]).astype(np.float32)
        neg = (rng.normal(size=(16, 30, fx.SEQ_FEATURE_DIM)) + 1.0).astype(np.float32)
        sequences, labels, neg = torch.from_numpy(sequences), torch.from_numpy(labels.astype(np.int64)), torch.from_numpy(neg)
        cfg = dict(d_model=16, gru_hidden=8, n_blocks=1, warmup_epochs=1, batch_size=8,
                   extra_hand_probability=0.0, seed=0)
        entropy = []
        for weight in (0.0, 2.0):
            tr.set_seed(0)
            model, _ = tr.train_one(sequences, labels, None, None, 3, tr.default_groups(fx.SEQ_FEATURE_DIM),
                                    tr.Config(**cfg, still_negative_weight=weight), torch.device("cpu"),
                                    fixed_epochs=8, log_every=0, negatives=neg)
            with torch.no_grad():
                p = torch.softmax(model(neg), -1)
            entropy.append(float(-(p * p.clamp_min(1e-9).log()).sum(-1).mean()))
        self.assertGreater(entropy[1], entropy[0])


if __name__ == "__main__":
    unittest.main()
