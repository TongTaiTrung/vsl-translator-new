"""Tests for feature re-extraction."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.data_reader import DataHandler
from src.data.reextract import find_raw_samples, reextract_sample
from src.features.extract import FEATURE_DIM, FEATURE_VERSION, SEQ_FEATURE_DIM, process_sequence
from src.features.frame import FRAME_SIZE
from src.features.landmarks import LANDMARKS_FILE, load_landmarks


class ReextractTests(unittest.TestCase):
    def test_reextract_preserves_metadata_and_backs_up_old_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            processed = root / "processed"
            backup = processed / ".reextract_backup_test"
            video = raw / "xin chao" / "3.mp4"
            target = processed / "xin chao" / "3"
            video.parent.mkdir(parents=True)
            target.mkdir(parents=True)
            video.touch()
            (target / "meta.json").write_text(
                json.dumps({"session": "original-session", "mirror": True, "created": "earlier"})
            )
            (target / "old.txt").write_text("old features")
            features = np.zeros((8, FEATURE_DIM), dtype=np.float32)
            features[:, 0] = 1.0
            features[:, -1] = 1.0

            marks = (np.full((8, 33, 4), 0.5, np.float32), np.full((8, 2, 21, 3), np.nan, np.float32),
                     np.full((8, 2, 21, 3), np.nan, np.float32))
            with patch("src.data.reextract.extract_video_features", return_value=(features, 25.0, marks)):
                result = reextract_sample(video, raw, processed, backup)

            self.assertEqual(result["frames"], 8)
            self.assertEqual((backup / "xin chao" / "3" / "old.txt").read_text(), "old features")
            meta = json.loads((target / "meta.json").read_text())
            self.assertEqual(meta["session"], "original-session")
            self.assertTrue(meta["mirror"])
            self.assertEqual(meta["created"], "earlier")
            self.assertEqual(meta["frame_size"], FRAME_SIZE)
            self.assertEqual(meta["feature_version"], FEATURE_VERSION)
            self.assertEqual(meta["source_video"], "xin chao/3.mp4")
            raw_features = np.stack([DataHandler._load_feature(target / f"{index}.npz") for index in range(8)])
            self.assertEqual(process_sequence(raw_features).shape, (30, SEQ_FEATURE_DIM))
            self.assertEqual(find_raw_samples(raw), [video])
            pose, hands, _ = load_landmarks(target / LANDMARKS_FILE)
            self.assertEqual(pose.shape, (8, 33, 4))
            self.assertTrue(np.isnan(hands).all())

    def test_failed_extraction_leaves_old_sample_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            processed = root / "processed"
            video = raw / "hello" / "0.mp4"
            target = processed / "hello" / "0"
            video.parent.mkdir(parents=True)
            target.mkdir(parents=True)
            video.touch()
            (target / "old.txt").write_text("keep")

            with patch("src.data.reextract.extract_video_features", side_effect=ValueError("bad video")):
                with self.assertRaisesRegex(ValueError, "bad video"):
                    reextract_sample(video, raw, processed, processed / ".backup")

            self.assertEqual((target / "old.txt").read_text(), "keep")
            self.assertFalse((processed / ".backup" / "hello" / "0").exists())


if __name__ == "__main__":
    unittest.main()
