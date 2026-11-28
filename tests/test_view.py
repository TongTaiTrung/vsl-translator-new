import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.augment import augment_sample
from src.data.data_reader import DataHandler
from src.features.extract import FEATURE_DIM, FEATURE_SLICES, FEATURE_VERSION, extract
from src.features.landmarks import (LANDMARKS_FILE, arrays_to_results, load_landmarks, results_to_arrays,
                                    save_landmarks)
from src.features.view import IDENTITY, ViewParams, features_from_landmarks, random_view, simulate_view
from src.model.benchmark import Clip, build_dataset, is_held_out, parse_holdout, session_invariance
from tests.test_feature_v3 import _results, _scene

S = FEATURE_SLICES
RIGHT_SHAPE = slice(S["handshape"].start + 38, S["handshape"].stop)
RIGHT_SCALE = S["hand_scale"].start + 1


def _clip(T: int = 10, seed: int = 0):
    """Landmark (pose, hands, world) của một clip: cảnh của test_feature_v3, tay phải trôi nhẹ theo thời gian."""
    rng = np.random.default_rng(seed)
    pose0, image0, world0 = _scene()
    marks = []
    for t in range(T):
        pose, image = pose0.copy(), image0.copy()
        drift = np.array([0.003 * t, -0.002 * t, 0.0]) + rng.normal(0, 0.001, 3)
        pose[16] += drift
        image += drift
        marks.append(results_to_arrays(*_results(pose, image, world0)))
    return tuple(np.stack(parts) for parts in zip(*marks))


class LandmarkStorageTests(unittest.TestCase):
    def test_arrays_round_trip_gives_identical_features(self):
        pose, image, world = _scene()
        direct = extract(*_results(pose, image, world))
        arrays = results_to_arrays(*_results(pose, image, world))
        self.assertTrue(np.isnan(arrays[1][1]).all())
        np.testing.assert_array_equal(extract(*arrays_to_results(*arrays)), direct)

    def test_save_and_load(self):
        marks = _clip(5)
        with tempfile.TemporaryDirectory() as d:
            save_landmarks(Path(d) / LANDMARKS_FILE, *marks)
            for a, b in zip(marks, load_landmarks(Path(d) / LANDMARKS_FILE)):
                np.testing.assert_array_equal(a, b)


class ViewSimulationTests(unittest.TestCase):
    def setUp(self):
        self.marks = _clip()
        self.base = features_from_landmarks(*self.marks)

    def viewed(self, view):
        return features_from_landmarks(*simulate_view(*self.marks, view))

    def test_identity_view_keeps_features(self):
        out = self.viewed(IDENTITY)
        np.testing.assert_array_equal(np.isnan(out), np.isnan(self.base))
        np.testing.assert_allclose(out, self.base, atol=1e-5, equal_nan=True)

    def test_camera_orbit_moves_position_but_keeps_handshape(self):
        out = self.viewed(ViewParams(yaw=20, pitch=10))
        self.assertGreater(np.nanmax(np.abs(out[:, S["hand_pos"]] - self.base[:, S["hand_pos"]])), 0.05)
        np.testing.assert_allclose(out[:, RIGHT_SHAPE], self.base[:, RIGHT_SHAPE], atol=1e-4)

    def test_farther_camera_shrinks_hand_held_in_front(self):
        far, near = self.viewed(ViewParams(distance=1.5)), self.viewed(ViewParams(distance=0.7))
        self.assertLess(far[:, RIGHT_SCALE].mean(), self.base[:, RIGHT_SCALE].mean())
        self.assertGreater(near[:, RIGHT_SCALE].mean(), self.base[:, RIGHT_SCALE].mean())

    def test_finger_offsets_change_joint_angles_of_that_hand_only(self):
        offsets = tuple([0.0] * 15 + [12.0] * 15)
        out = self.viewed(ViewParams(finger_offsets=offsets))
        self.assertGreater(np.abs(out[:, RIGHT_SHAPE] - self.base[:, RIGHT_SHAPE]).max(), 0.05)
        np.testing.assert_allclose(out[:, S["hand_pos"]][:, 9:12], self.base[:, S["hand_pos"]][:, 9:12], atol=1e-5)

    def test_hand_pushed_out_of_frame_is_dropped(self):
        pose, hands, world = (a.copy() for a in self.marks)
        out = simulate_view(pose, hands, world, ViewParams(shift_x=-40.0))
        self.assertTrue(np.isnan(out[1][:, 0]).all())
        self.assertTrue(np.isfinite(hands[:, 0]).all())

    def test_random_view_is_reproducible(self):
        a, b = random_view(np.random.default_rng(3)), random_view(np.random.default_rng(3))
        self.assertEqual(a, b)
        self.assertEqual(len(a.finger_offsets), 30)


class LandmarkAugmentTests(unittest.TestCase):
    def test_augment_uses_landmarks_when_present(self):
        marks = _clip()
        features = features_from_landmarks(*marks)
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw"
            source = Path(d) / "processed" / "m" / "0"
            source.mkdir(parents=True)
            for i, f in enumerate(features):
                np.savez_compressed(source / f"{i}.npz", feature=f)
            (source / "meta.json").write_text(json.dumps({"session": "s1", "feature_version": FEATURE_VERSION}))
            save_landmarks(source / LANDMARKS_FILE, *marks)
            self.assertEqual(len(DataHandler._frame_paths(source)[0]), len(features))

            created = augment_sample(source, raw, 2, 0.5, np.random.default_rng(0))
            for target in created:
                meta = json.loads((target / "meta.json").read_text())
                self.assertEqual(meta["augmentation"]["type"], "landmark_view")
                self.assertEqual(meta["origin_sample_id"], "0")
                self.assertIn("yaw", meta["augmentation"]["view"])
                self.assertFalse((target / LANDMARKS_FILE).exists())
                out = np.stack([DataHandler._load_feature(p) for p in DataHandler._frame_paths(target)[0]])
                self.assertEqual(out.shape, (len(features), FEATURE_DIM))


class BenchmarkTests(unittest.TestCase):
    def test_holdout_spec(self):
        holdout = parse_holdout(["m@20261118", "@20261201"])
        clip = lambda cls, ses: Clip(cls, ses, "0", np.zeros((1, FEATURE_DIM)), None)
        self.assertTrue(is_held_out(clip("m", "20261118-101010"), holdout))
        self.assertFalse(is_held_out(clip("n", "20261118-101010"), holdout))
        self.assertTrue(is_held_out(clip("n", "20261201-090000"), holdout))
        with self.assertRaises(ValueError):
            parse_holdout(["m"])

    def test_build_excludes_holdout_and_keeps_groups(self):
        marks = _clip()
        feats = features_from_landmarks(*marks)
        clips = [Clip("m", "day1", str(i), feats, marks) for i in range(2)]
        clips += [Clip("m", "day2", str(i + 2), feats, marks) for i in range(2)]
        clips += [Clip("n", "day1", str(i), feats, marks) for i in range(2)]
        data = build_dataset(clips, parse_holdout(["m@day2"]), "landmark", 5, np.random.default_rng(0))
        self.assertEqual(list(data["classes"]), ["m", "n"])
        self.assertNotIn("day2", set(data["sessions"]))
        self.assertEqual(np.bincount(data["y"]).tolist(), [5, 5])
        self.assertEqual(int((~data["augmented"]).sum()), 4)
        originals = set(data["groups"][~data["augmented"]])
        self.assertTrue(set(data["groups"][data["augmented"]]) <= originals)

    def test_invariance_scores_classes_with_two_sessions(self):
        rng = np.random.default_rng(0)
        base = features_from_landmarks(*_clip())
        clips = []
        for cls, offset in (("m", 0.0), ("n", 3.0)):
            for day in ("day1", "day2"):
                for i in range(3):
                    f = base.copy()
                    f[:, S["hand_pos"]] += offset + rng.normal(0, 0.01, f[:, S["hand_pos"]].shape)
                    clips.append(Clip(cls, day, str(i), f, None))
        report = session_invariance(clips, day_chars=4)
        self.assertEqual(report["m"]["cross_session_nn"], 1.0)
        self.assertLess(report["m"]["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
