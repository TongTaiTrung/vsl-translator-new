"""Tests for feature augmentation and grouping."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.augment import add_feature_noise, augment_sample, available_sources, plan_augmentation
from src.data.data_reader import DataHandler
from src.features.extract import FEATURE_DIM, FEATURE_VERSION, PRESENCE_SLICE
from src.model.training import make_folds


class FeatureAugmentationTests(unittest.TestCase):
    def test_noise_keeps_missing_values_and_presence(self):
        source = np.full((8, FEATURE_DIM), 0.5, dtype=np.float32)
        source[:, PRESENCE_SLICE] = (0.0, 0.8)
        source[2, 4] = np.nan
        output = add_feature_noise(source, np.random.default_rng(4))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isnan(output[2, 4]))
        np.testing.assert_array_equal(output[:, PRESENCE_SLICE], source[:, PRESENCE_SLICE])
        self.assertFalse(np.allclose(output[:, :4], source[:, :4]))
        np.testing.assert_allclose(source[:, PRESENCE_SLICE], np.tile((0.0, 0.8), (8, 1)))

    def _source(self, root: Path) -> Path:
        source = root / "processed" / "hello" / "0"
        source.mkdir(parents=True)
        rng = np.random.default_rng(1)
        for index in range(8):
            feature = rng.normal(0, 0.3, FEATURE_DIM).astype(np.float32)
            feature[PRESENCE_SLICE] = (0.0, 0.8)
            np.savez_compressed(source / f"{index}.npz", feature=feature)
        (source / "meta.json").write_text(json.dumps({"feature_version": FEATURE_VERSION}))
        return source

    def test_copies_get_random_view_pitch_recorded_in_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._source(Path(tmp))
            created = augment_sample(source, Path(tmp) / "raw", 3, 1.0, np.random.default_rng(2), pitch_degrees=10.0)
            pitches = [json.loads((p / "meta.json").read_text())["augmentation"]["view_pitch_deg"] for p in created]
            self.assertTrue(all(-10.0 <= p <= 10.0 for p in pitches))
            self.assertGreater(len(set(pitches)), 1)
            flat = augment_sample(source, Path(tmp) / "raw", 1, 1.0, np.random.default_rng(3), pitch_degrees=0.0)[0]
            features = np.stack([DataHandler._load_feature(source / f"{i}.npz") for i in range(8)])
            expected = add_feature_noise(features, np.random.default_rng(3), 1.0)
            got = np.stack([DataHandler._load_feature(flat / f"{i}.npz") for i in range(8)])
            np.testing.assert_allclose(got, expected, atol=1e-6)
            self.assertEqual(json.loads((flat / "meta.json").read_text())["augmentation"]["view_pitch_deg"], 0.0)
            with self.assertRaises(ValueError):
                augment_sample(source, Path(tmp) / "raw", 1, 1.0, np.random.default_rng(4), pitch_degrees=90.0)

    def test_copies_share_origin_group_and_leave_raw_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "data" / "processed" / "hello"
            raw = root / "data" / "raw" / "hello"
            source = processed / "0"
            source.mkdir(parents=True)
            raw.mkdir(parents=True)
            (raw / "0.mp4").write_bytes(b"original")
            for index in range(8):
                feature = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
                feature[PRESENCE_SLICE] = (0.0, 0.8)
                np.savez_compressed(source / f"{index}.npz", feature=feature)
            original_meta = {"session": "session-1", "feature_version": FEATURE_VERSION, "mirror": False}
            (source / "meta.json").write_text(json.dumps(original_meta))

            created = augment_sample(source, raw.parent, 2, 1.0, np.random.default_rng(5))

            self.assertEqual([path.name for path in created], ["1", "2"])
            self.assertEqual([path.name for path in available_sources(processed.parent, ["hello"])], ["0"])
            self.assertEqual(plan_augmentation(processed.parent, ["hello"], 3, 2), ([], {"hello": 3}))
            plan, counts = plan_augmentation(processed.parent, ["hello"], 5, 2)
            self.assertEqual((plan, counts), ([(source, 2)], {"hello": 3}))
            self.assertEqual(plan_augmentation(processed.parent, ["hello"], -1, 2)[0], [(source, 2)])
            augment_sample(plan[0][0], raw.parent, plan[0][1], 1.0, np.random.default_rng(6))
            self.assertEqual(plan_augmentation(processed.parent, ["hello"], 5, 2), ([], {"hello": 5}))
            self.assertEqual((raw / "0.mp4").read_bytes(), b"original")
            self.assertEqual(list(raw.glob("*.mp4")), [raw / "0.mp4"])
            self.assertEqual(json.loads((source / "meta.json").read_text()), original_meta)
            with patch("src.data.data_reader.get_project_root_dir", return_value=root):
                handler = DataHandler()
                self.assertEqual(handler.validate_dataset(), 5)
                self.assertEqual(handler.get_origin_counts()["hello"], 1)
                self.assertEqual(handler.sample_groups("sample"), ["hello/0"] * 5)
                self.assertEqual(handler.sample_groups("session"), ["session-1"] * 5)
                self.assertEqual(sum(handler.describe(i)["augmented"] for i in range(5)), 4)

    def test_plan_skips_classes_already_at_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, total in (("full", 3), ("short", 1)):
                for sample_id in range(total):
                    sample_dir = root / name / str(sample_id)
                    sample_dir.mkdir(parents=True)
                    for frame_id in range(8):
                        (sample_dir / f"{frame_id}.npz").touch()
                    if sample_id:
                        (sample_dir / "meta.json").write_text(
                            json.dumps({"augmentation": {"type": "feature_noise"}, "origin_sample_id": "0"})
                        )

            plan, counts = plan_augmentation(root, None, 3, 2)
            self.assertEqual(counts, {"full": 3, "short": 1})
            self.assertEqual([(source.parent.name, source.name, copies) for source, copies in plan],
                             [("short", "0", 2)])
            unlimited, _ = plan_augmentation(root, None, -1, 2)
            self.assertEqual(len(unlimited), 2)

    def test_grouped_folds_keep_origins_together(self):
        labels = np.repeat([0, 1], 12)
        groups = np.array([f"{label}/{origin}" for label in (0, 1)
                           for origin in range(4) for _ in range(3)])
        folds = make_folds(labels, 2, groups, seed=7)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups == group])), 1)
        self.assertEqual(set(folds), {0, 1})


if __name__ == "__main__":
    unittest.main()
