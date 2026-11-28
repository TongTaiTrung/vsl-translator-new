"""Tests for MediaPipe wrapper and data collection."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.data.data_collect import get_next_id, save_sample
from src.data.data_reader import DataHandler
from src.features.extract import SEQ_FEATURE_DIM
from src.features.frame import FRAME_SIZE, frame_padding


class BuiltinCollectionTests(unittest.TestCase):
    def test_frame_padding_always_returns_square_canvas(self):
        for height, width in ((720, 1280), (640, 480), (240, 320), (720, 720), (1080, 1920)):
            with self.subTest(shape=(height, width)):
                frame = np.full((height, width, 3), 255, dtype=np.uint8)
                padded = frame_padding(frame)
                self.assertEqual(padded.shape, (FRAME_SIZE, FRAME_SIZE, 3))
                self.assertEqual(padded[FRAME_SIZE // 2, FRAME_SIZE // 2, 0], 255)

    def test_frame_padding_keeps_normalized_geometry_across_resolutions(self):
        for height, width in ((720, 720), (720, 1280), (405, 720)):
            with self.subTest(shape=(height, width)):
                frame = np.full((height, width, 3), 255, dtype=np.uint8)
                padded = frame_padding(frame)
                cols = np.flatnonzero(padded[:, :, 0].max(axis=0))
                rows = np.flatnonzero(padded[:, :, 0].max(axis=1))
                self.assertEqual((cols[0], cols[-1] + 1), (0, FRAME_SIZE))
                expected = round(FRAME_SIZE * height / max(height, width))
                self.assertAlmostEqual((rows[-1] + 1 - rows[0]) / FRAME_SIZE, expected / FRAME_SIZE, delta=2 / FRAME_SIZE)

    def test_builtin_loader_processes_image_and_video_frames(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for mode in (VisionRunningMode.IMAGE, VisionRunningMode.VIDEO):
            with self.subTest(mode=mode):
                loader = LandmarkerLoader(running_mode=mode)
                try:
                    pose, hand, timestamp = loader.detect(
                        frame, timestamp_ms=0 if mode == VisionRunningMode.VIDEO else None
                    )
                    self.assertIsNotNone(pose)
                    self.assertIsNotNone(hand)
                    if mode == VisionRunningMode.VIDEO:
                        self.assertEqual(timestamp, 0)
                        self.assertGreater(loader.detect(frame)[2], timestamp)
                    else:
                        self.assertIsNone(timestamp)
                finally:
                    loader.close()

    def test_save_sample_uses_builtin_results_and_keeps_next_id_unique(self):
        class EmptyDetector:
            def detect(self, frame):
                assert frame.shape == (FRAME_SIZE, FRAME_SIZE, 3)
                return None, None, None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "hello"
            extracted = root / "extracted" / "hello" / "0"
            raw.mkdir(parents=True)
            extracted.mkdir(parents=True)
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]

            shape = save_sample(frames, 0, raw, extracted, 30, EmptyDetector())

            self.assertEqual(shape, (30, SEQ_FEATURE_DIM))
            self.assertTrue((raw / "0.mp4").is_file())
            video = cv2.VideoCapture(str(raw / "0.mp4"))
            try:
                self.assertEqual(
                    (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    (FRAME_SIZE, FRAME_SIZE),
                )
            finally:
                video.release()
            with np.load(extracted / "0.npz") as saved:
                self.assertEqual(saved["X"].shape, shape)
            self.assertEqual(get_next_id(root / "raw", "hello", extracted), 1)

    def test_data_handler_video_output_is_square_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            DataHandler._write_video(path, [np.zeros((360, 640, 3), dtype=np.uint8)], 30)
            video = cv2.VideoCapture(str(path))
            try:
                self.assertEqual(
                    (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    (FRAME_SIZE, FRAME_SIZE),
                )
            finally:
                video.release()


if __name__ == "__main__":
    unittest.main()
