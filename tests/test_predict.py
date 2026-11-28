import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.features.extract import _PALM_Y, FEATURE_DIM, SEQ_FEATURE_DIM
from src.features.frame import FRAME_SIZE
from src.model.predict import StreamConfig, VSLPredictor, VSLStreamSession
from src.model.training import Config, Normalizer, VSLNet, default_groups, save_ensemble


class PredictorTests(unittest.TestCase):
    def test_checkpoint_prediction_is_json_ready_and_validates_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            config = Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1)
            model = VSLNet(SEQ_FEATURE_DIM, 2, default_groups(SEQ_FEATURE_DIM), config).eval()
            norm = Normalizer(np.zeros((2, 30, SEQ_FEATURE_DIM), dtype=np.float32))
            save_ensemble(path, [model], [norm], config, default_groups(SEQ_FEATURE_DIM),
                          ["xin chào", "cảm ơn"], 1.0, 0.0, (30, SEQ_FEATURE_DIM))

            predictor = VSLPredictor(path, device="cpu", min_confidence=0.0, min_probability_gap=0.0)
            result = predictor.predict_raw(np.zeros((12, FEATURE_DIM), dtype=np.float32), top_k=2)
            json.dumps(result, ensure_ascii=False)
            self.assertEqual(len(result["top"]), 2)
            self.assertTrue(result["accepted"])
            self.assertIn(result["label"], predictor.classes)
            with self.assertRaisesRegex(ValueError, "Expected processed features"):
                predictor.predict_processed(np.zeros((30, FEATURE_DIM), dtype=np.float32))

    def test_ambiguous_top_two_is_rejected_but_keeps_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            config = Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1)
            model = VSLNet(SEQ_FEATURE_DIM, 2, default_groups(SEQ_FEATURE_DIM), config).eval()
            norm = Normalizer(np.zeros((2, 30, SEQ_FEATURE_DIM), dtype=np.float32))
            save_ensemble(path, [model], [norm], config, default_groups(SEQ_FEATURE_DIM),
                          ["A", "B"], 1.0, 0.0, (30, SEQ_FEATURE_DIM))
            predictor = VSLPredictor(path, device="cpu", min_confidence=0.0, min_probability_gap=0.1)
            with patch("src.model.predict.training.predict_topk", return_value=[[('A', 0.52), ('B', 0.48)]]):
                result = predictor.predict_processed(np.zeros((30, SEQ_FEATURE_DIM), dtype=np.float32), top_k=1)
            self.assertIsNone(result["label"])
            self.assertEqual(result["rejection_reason"], "ambiguous")
            self.assertAlmostEqual(result["probability_gap"], 0.04)
            self.assertEqual(len(result["top"]), 1)

    def test_stream_owns_window_and_resets_after_hand_disappears(self):
        class FakeLandmarkers:
            def __init__(self):
                self.calls = 0
                self.closed = False

            def detect(self, frame):
                self.calls += 1
                assert frame.shape == (FRAME_SIZE, FRAME_SIZE, 3)
                return None, self.calls <= 3, self.calls

            def close(self):
                self.closed = True

        class FakePredictor:
            def predict_raw(self, features, *, top_k):
                assert features.shape[1] == FEATURE_DIM
                return {"label": "xin chào", "confidence": 0.9, "accepted": True,
                        "top": [{"label": "xin chào", "confidence": 0.9}]}

        def fake_extract(_pose, hand):
            feature = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
            if hand:
                feature[_PALM_Y[1]] = 0.4
            return feature

        landmarkers = FakeLandmarkers()
        config = StreamConfig(window_seconds=1.0, min_window_seconds=0.1, min_frames=2,
                              predict_every=1, stability_frames=2, no_hand_timeout=0.3,
                              target_fps=None)
        with patch("src.model.predict.extract", side_effect=fake_extract):
            with VSLStreamSession(FakePredictor(), config, landmarkers) as session:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                self.assertEqual(session.update_frame(frame, now=0.0)["status"], "collecting")
                session.update_frame(frame, now=0.1)
                ready = session.update_frame(frame, now=0.2)
                self.assertEqual(ready["stable_label"], "xin chào")
                json.dumps(ready, ensure_ascii=False)
                self.assertIsNone(session.update_frame(frame, now=0.25)["stable_label"])
                self.assertEqual(session.update_frame(frame, now=0.7)["status"], "idle")
                session.reset()
                self.assertEqual(len(session.history), 0)
        self.assertTrue(landmarkers.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.update_frame(frame)

    def test_feature_version_mismatch_raises_unless_opted_out(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            config = Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1)
            model = VSLNet(SEQ_FEATURE_DIM, 2, default_groups(SEQ_FEATURE_DIM), config).eval()
            norm = Normalizer(np.zeros((2, 30, SEQ_FEATURE_DIM), dtype=np.float32))
            save_ensemble(path, [model], [norm], config, default_groups(SEQ_FEATURE_DIM),
                          ["A", "B"], 1.0, 0.0, (30, SEQ_FEATURE_DIM))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["feature_version"] = 999
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "feature_version"):
                VSLPredictor(path, device="cpu")

            predictor = VSLPredictor(path, device="cpu", allow_feature_version_mismatch=True)
            self.assertEqual(predictor.feature_version, 999)

    def test_stream_warns_on_mirror_data_mismatch_with_checkpoint(self):
        class FakePredictor:
            checkpoint = {"mirror_data": True}

        landmarkers = type("L", (), {"close": lambda self: None})()
        with self.assertWarnsRegex(UserWarning, "mirror_data"):
            session = VSLStreamSession(FakePredictor(), StreamConfig(mirror_data=False), landmarkers)
        session.close()

    def test_complete_clip_accepts_bgr_frames_and_closes_landmarkers(self):
        class FakeLandmarkers:
            closed_count = 0
            timestamps = []

            def __init__(self, *, running_mode):
                self.running_mode = running_mode

            def detect(self, frame, timestamp_ms=None):
                assert frame.shape == (FRAME_SIZE, FRAME_SIZE, 3)
                type(self).timestamps.append(timestamp_ms)
                return None, None, None

            def close(self):
                type(self).closed_count += 1

        fake = FakeLandmarkers
        predictor = VSLPredictor.__new__(VSLPredictor)
        predictor.predict_raw = lambda features, **_kwargs: {"frames": len(features)}
        with patch("src.model.predict.LandmarkerLoader", fake), \
             patch("src.model.predict.extract", return_value=np.zeros(FEATURE_DIM, dtype=np.float32)):
            result = predictor.predict_frames([np.zeros((360, 640, 3), dtype=np.uint8)] * 2)
            predictor.predict_frames([np.zeros((360, 640, 3), dtype=np.uint8)] * 3, fps=15)
        self.assertEqual(result, {"frames": 2})
        self.assertEqual(fake.closed_count, 2)
        self.assertEqual(fake.timestamps, [None, None, 0, 67, 133])

    def test_stream_skips_frames_faster_than_target_fps_and_ignores_resting_hand(self):
        class FakeLandmarkers:
            calls = 0

            def detect(self, frame):
                type(self).calls += 1
                return None, None, None

            def close(self):
                pass

        resting = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
        resting[_PALM_Y[0]] = 1.2
        config = StreamConfig(target_fps=10)
        with patch("src.model.predict.extract", return_value=resting), \
             VSLStreamSession(object(), config, FakeLandmarkers()) as session:
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            states = [session.update_frame(frame, now=t) for t in (0.0, 0.05, 0.1, 0.12, 0.2)]
        self.assertEqual(FakeLandmarkers.calls, 3)
        self.assertIs(states[1], states[0])
        self.assertFalse(states[-1]["hand_present"])


if __name__ == "__main__":
    unittest.main()
