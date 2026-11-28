"""Tests for the fast prediction branch."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.app.fast import FastConfig, FastRealtimeTranslator, FastStreamSession, make_fast_ensemble
from src.app.sentence import SentenceComposer
from src.app.voting import VoteConfig, WordVoter
from src.features.extract import _PALM_Y, FEATURE_DIM, SEQ_FEATURE_DIM
from src.model.predict import StreamConfig, VSLPredictor
from src.model.training import Config, Normalizer, VSLNet, default_groups, save_ensemble


class NoLandmarkers:
    def close(self):
        pass


def hand_feature():
    feature = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    feature[_PALM_Y[1]] = 0.4
    return feature


def result(label, confidence, gap):
    return {"label": label, "confidence": confidence, "probability_gap": gap, "accepted": label is not None,
            "rejection_reason": None if label else "low_confidence",
            "top": [{"label": label or "x", "confidence": confidence}]}


class ScriptedPredictor:

    target_len = 30

    def __init__(self, fast, full):
        self.fast, self.full = fast, full
        self.fast_ensemble = object()
        self.calls = []

    def predict_processed(self, sequence, *, top_k=3, ensemble=None):
        kind = "fast" if ensemble is self.fast_ensemble else "full"
        self.calls.append(kind)
        return dict(self.fast if kind == "fast" else self.full)


def open_session(predictor, fast_config=None, **stream):
    config = StreamConfig(**{"predict_every": 1, "min_frames": 2, "min_window_seconds": 1.5,
                             "window_seconds": 2.0, "target_fps": None, **stream})
    return FastStreamSession(predictor, config, fast_config or FastConfig(), landmarkers=NoLandmarkers(),
                             fast_ensemble=predictor.fast_ensemble)


class FastVoterTests(unittest.TestCase):
    def test_all_strong_streak_commits_early(self):
        voter = WordVoter(VoteConfig(min_consecutive=5, early_votes=3))
        self.assertEqual([voter.vote("toi", strong=True) for _ in range(3)], [None, None, "toi"])

    def test_one_weak_vote_requires_full_streak(self):
        voter = WordVoter(VoteConfig(min_consecutive=5, early_votes=3))
        votes = [voter.vote("toi", strong=s) for s in (True, False, True, True, True)]
        self.assertEqual(votes, [None, None, None, None, "toi"])

    def test_failed_confirmation_drops_streak(self):
        voter = WordVoter(VoteConfig(min_consecutive=2))
        seen = []
        confirm = lambda label, early: seen.append((label, early)) or False   # noqa: E731
        self.assertIsNone(voter.vote("a", confirm=confirm))
        self.assertIsNone(voter.vote("a", confirm=confirm))
        self.assertEqual(seen, [("a", False)])
        self.assertEqual((voter.sequence, voter.count), ([], 0))

    def test_early_votes_must_not_exceed_min_consecutive(self):
        with self.assertRaises(ValueError):
            VoteConfig(min_consecutive=3, early_votes=4)


class FastStreamSessionTests(unittest.TestCase):
    def feed(self, session, seconds, fps=15):
        return [session.update_features(hand_feature(), now=i / fps) for i in range(int(seconds * fps) + 1)]

    def test_strong_fast_prediction_starts_at_early_window_with_one_model(self):
        predictor = ScriptedPredictor(result("toi", 0.95, 0.9), result("toi", 0.9, 0.8))
        states = self.feed(open_session(predictor), 1.0)
        self.assertEqual(states[-2]["status"], "collecting")
        prediction = states[-1]["prediction"]
        self.assertEqual((prediction["label"], prediction["source"]), ("toi", "fast"))
        self.assertTrue(prediction["strong"] and prediction["early"])
        self.assertEqual(predictor.calls, ["fast"])

    def test_weak_prediction_is_ignored_before_min_window(self):
        predictor = ScriptedPredictor(result("toi", 0.7, 0.2), result("toi", 0.9, 0.8))
        session = open_session(predictor)
        early = self.feed(session, 1.2)[-1]["prediction"]
        self.assertIsNone(early["label"])
        self.assertEqual(early["rejection_reason"], "early_uncertain")
        session.reset()
        late = self.feed(session, 1.6)[-1]["prediction"]
        self.assertEqual((late["label"], late["strong"], late["early"]), ("toi", False, False))
        self.assertNotIn("full", predictor.calls)

    def test_escalation_runs_ensemble_only_when_enabled(self):
        predictor = ScriptedPredictor(result(None, 0.4, 0.1), result("toi", 0.8, 0.5))
        late = self.feed(open_session(predictor, FastConfig(escalate=True)), 1.6)[-1]["prediction"]
        self.assertEqual((late["label"], late["source"]), ("toi", "ensemble"))
        predictor.calls.clear()
        late = self.feed(open_session(predictor), 1.6)[-1]["prediction"]
        self.assertIsNone(late["label"])
        self.assertNotIn("full", predictor.calls)

    def test_confirm_uses_strict_threshold_for_early_commits(self):
        predictor = ScriptedPredictor(result("toi", 0.95, 0.9), result("toi", 0.8, 0.5))
        session = open_session(predictor, FastConfig(strict_confidence=0.85))
        self.feed(session, 1.0)
        self.assertFalse(session.confirm("toi", early=True))
        self.feed(session, 1.6)
        self.assertTrue(session.confirm("toi", early=False))
        self.assertFalse(session.confirm("cảm ơn"))
        self.assertEqual(session.stats["confirm"], 3)
        self.assertEqual(session.stats["confirm_rejected"], 2)

    def test_early_window_cannot_exceed_min_window(self):
        with self.assertRaises(ValueError):
            open_session(ScriptedPredictor({}, {}), FastConfig(early_window_seconds=1.8))


class FastTranslatorTests(unittest.TestCase):
    def test_commits_after_early_votes_and_ensemble_confirmation(self):
        predictor = ScriptedPredictor(result("toi", 0.95, 0.9), result("toi", 0.92, 0.85))
        session = open_session(predictor)
        session.update_frame = lambda feature, now=None: session.update_features(feature, now=now)
        translator = FastRealtimeTranslator(None, session=session, vote_config=VoteConfig(min_consecutive=5),
                                            composer=SentenceComposer(None, background=False))
        committed = [translator.process(hand_feature(), now=i / 15)["committed"] for i in range(20)]
        first = committed.index("toi")
        self.assertEqual(first, 17)
        self.assertEqual(translator.words, ["toi"])
        self.assertEqual(predictor.calls.count("full"), 1)
        translator.close()


class FastEnsembleTests(unittest.TestCase):
    def test_fast_ensemble_wraps_one_model_with_its_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            config = Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1)
            models = [VSLNet(SEQ_FEATURE_DIM, 2, default_groups(SEQ_FEATURE_DIM), config).eval() for _ in range(2)]
            rng = np.random.default_rng(0)
            norms = [Normalizer(rng.normal(size=(2, 30, SEQ_FEATURE_DIM)).astype(np.float32)) for _ in range(2)]
            save_ensemble(path, models, norms, config, default_groups(SEQ_FEATURE_DIM), ["A", "B"], 1.3, 0.0,
                          (30, SEQ_FEATURE_DIM))
            predictor = VSLPredictor(path, device="cpu", min_confidence=0.0, min_probability_gap=0.0)
            fast = make_fast_ensemble(predictor, -1)
            self.assertEqual(len(fast.models), 1)
            self.assertIs(fast.models[0], predictor.ensemble.models[1])
            np.testing.assert_allclose(fast.means.numpy()[0], predictor.ensemble.means.numpy()[1])
            self.assertAlmostEqual(float(fast.temperature), 1.3, places=5)
            sequence = np.zeros((30, SEQ_FEATURE_DIM), dtype=np.float32)
            self.assertIn(predictor.predict_processed(sequence, ensemble=fast)["label"], ("A", "B"))
            with self.assertRaises(IndexError):
                make_fast_ensemble(predictor, 2)


if __name__ == "__main__":
    unittest.main()
