import unittest
from unittest import mock

import cv2
import numpy as np

from src.features import camera


class FakeCapture:
    """Camera giả: chỉ hỗ trợ ``modes``; yêu cầu kích thước khác thì giữ nguyên (như DirectShow)
    hoặc chuyển sang chế độ gốc gần nhất (``snap=True``, như AVFoundation)."""

    def __init__(self, modes, default, snap=False, backend="AVFOUNDATION", opened=True):
        self.modes, self.size, self.snap, self.backend, self.opened = modes, default, snap, backend, opened
        self.pending = {}
        self.released = False
        self.fourcc = None

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True

    def getBackendName(self):
        return self.backend

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOURCC:
            self.fourcc = value
            return True
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            self.pending[prop] = int(value)
            if len(self.pending) == 2:
                want = (self.pending[cv2.CAP_PROP_FRAME_WIDTH], self.pending[cv2.CAP_PROP_FRAME_HEIGHT])
                self.pending = {}
                if want in self.modes:
                    self.size = want
                elif self.snap:
                    self.size = min(self.modes, key=lambda m: abs(m[0] - want[0]) + abs(m[1] - want[1]))
            return True
        return False

    def get(self, prop):
        return 30.0 if prop == cv2.CAP_PROP_FPS else -1.0

    def read(self):
        return True, np.zeros((self.size[1], self.size[0], 3), np.uint8)


class OpenCameraTests(unittest.TestCase):
    def open(self, fake, *args, **kwargs):
        with mock.patch.object(camera.cv2, "VideoCapture", return_value=fake):
            return camera.open_camera(*args, verbose=False, **kwargs)

    def test_picks_largest_16_9_mode_on_avfoundation(self):
        modes = [(1920, 1080), (1280, 720), (1080, 1920), (1760, 1328), (640, 480), (1328, 1760), (1552, 1552)]
        fake = FakeCapture(modes, default=(480, 360), snap=True)
        cap = self.open(fake, 0)
        self.assertEqual(cap.read()[1].shape[:2], (1080, 1920))
        self.assertIsNone(fake.fourcc)

    def test_ignores_unsupported_requests_on_other_backends(self):
        fake = FakeCapture([(640, 480), (1280, 720)], default=(640, 480), backend="DSHOW")
        cap = self.open(fake, 0)
        self.assertEqual(cap.read()[1].shape[:2], (720, 1280))
        self.assertEqual(fake.fourcc, cv2.VideoWriter_fourcc(*"MJPG"))

    def test_falls_back_to_largest_seen_size(self):
        fake = FakeCapture([(640, 480), (1280, 960)], default=(640, 480), snap=True)
        cap = self.open(fake, 0)
        self.assertEqual(cap.read()[1].shape[:2], (960, 1280))

    def test_explicit_resolution(self):
        modes = [(1920, 1080), (1280, 720)]
        fake = FakeCapture(modes, default=(640, 360), snap=True)
        cap = self.open(fake, 0, (1280, 720))
        self.assertEqual(cap.read()[1].shape[:2], (720, 1280))

    def test_unopened_camera_raises_and_releases(self):
        fake = FakeCapture([], default=(0, 0), opened=False)
        with self.assertRaises(RuntimeError):
            self.open(fake, 3)
        self.assertTrue(fake.released)

    def test_parse_resolution(self):
        self.assertEqual(camera.parse_resolution("1920x1080"), (1920, 1080))
        self.assertIsNone(camera.parse_resolution("max"))
        self.assertIsNone(camera.parse_resolution(None))
        with self.assertRaises(ValueError):
            camera.parse_resolution("1920")


if __name__ == "__main__":
    unittest.main()
