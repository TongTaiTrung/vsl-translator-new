import base64
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.app.server import create_app
from src.features.extract import _PALM_Y, FEATURE_DIM, REST_Y, SEQ_FEATURE_DIM
from src.model.predict import VSLPredictor
from src.model.training import Config, Normalizer, VSLNet, default_groups, save_ensemble


def encode_frame(frame: np.ndarray, data_url: bool = True) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    text = base64.b64encode(buffer.tobytes()).decode()
    return f"data:image/jpeg;base64,{text}" if data_url else text


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name) / "ensemble.pt"
        config = Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1)
        model = VSLNet(SEQ_FEATURE_DIM, 2, default_groups(SEQ_FEATURE_DIM), config).eval()
        norm = Normalizer(np.zeros((2, 30, SEQ_FEATURE_DIM), dtype=np.float32))
        save_ensemble(path, [model], [norm], config, default_groups(SEQ_FEATURE_DIM),
                      ["xin chào", "cảm ơn"], 1.0, 0.0, (30, SEQ_FEATURE_DIM))
        cls.predictor = VSLPredictor(path, device="cpu", min_confidence=0.0, min_probability_gap=0.0)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.calls = []

        def fake_extractor(frames, *, fps, mirror_data):
            self.calls.append({"n": len(frames), "shape": frames[0].shape, "fps": fps, "mirror": mirror_data})
            return np.zeros((len(frames), FEATURE_DIM), dtype=np.float32)

        self.client = create_app(self.predictor, feature_extractor=fake_extractor).test_client()
        self.frame = np.full((48, 64, 3), 128, dtype=np.uint8)

    def test_health_lists_classes(self):
        body = self.client.get("/api/health").get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["classes"], ["xin chào", "cảm ơn"])
        self.assertEqual(body["expected_frames"], 30)

    def test_predict_30_frames_returns_model_result(self):
        frames = [encode_frame(self.frame, data_url=i % 2 == 0) for i in range(30)]
        response = self.client.post("/api/predict", json={"frames": frames, "fps": 30, "top_k": 2})
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertIn(body["label"], ("xin chào", "cảm ơn"))
        self.assertTrue(body["accepted"])
        self.assertEqual(len(body["top"]), 2)
        self.assertEqual(body["frames"], 30)
        self.assertIn("cảm ơn".encode(), response.data)          # tiếng Việt không bị escape
        self.assertEqual(self.calls, [{"n": 30, "shape": (48, 64, 3), "fps": 30.0, "mirror": False}])
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_predict_counts_frames_with_active_hands(self):
        def extractor(frames, *, fps, mirror_data):
            features = np.zeros((len(frames), FEATURE_DIM), dtype=np.float32)
            features[:12, list(_PALM_Y)] = np.nan            # 12 frame không thấy tay
            features[12:15, list(_PALM_Y)] = REST_Y + 0.5    # 3 frame tay buông -> không tính
            return features

        client = create_app(self.predictor, feature_extractor=extractor).test_client()
        frames = [encode_frame(self.frame) for _ in range(30)]
        body = client.post("/api/predict", json={"frames": frames}).get_json()
        self.assertEqual(body["hand_frames"], 15)

    def test_predict_accepts_raw_features(self):
        response = self.client.post("/api/predict", json={"features": np.zeros((30, FEATURE_DIM)).tolist()})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(self.calls, [])

    def test_rejects_bad_requests(self):
        good = encode_frame(self.frame)
        cases = [
            {"frames": [good] * 29},
            {"frames": []},
            {"frames": [good] * 29 + ["not base64!"]},
            {"frames": [good] * 29 + [base64.b64encode(b"not an image").decode()]},
            {"frames": [good] * 30, "fps": 0},
            {"frames": [good] * 30, "top_k": 0},
            {"frames": [good] * 30, "mirror_data": "yes"},
            {"features": np.zeros((30, 5)).tolist()},
            {"features": [[float("inf")] * FEATURE_DIM] * 30},
            {},
        ]
        for body in cases:
            with self.subTest(body={k: (v if k != "frames" else len(v)) for k, v in body.items()}):
                response = self.client.post("/api/predict", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        self.assertEqual(self.client.post("/api/predict", data="x", content_type="text/plain").status_code, 400)
        self.assertEqual(self.calls, [])

    def test_root_serves_test_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/api/predict", response.data)
        response.close()

    def test_cors_preflight(self):
        response = self.client.options("/api/predict")
        self.assertEqual(response.status_code, 204)
        self.assertIn("POST", response.headers["Access-Control-Allow-Methods"])


class FakeStreamExtractor:
    """Extractor giả cho /api/stream: ghi lại timestamp; frame có kênh B > 200 thì có tay (palm y = 0)."""

    instances: list["FakeStreamExtractor"] = []

    def __init__(self):
        self.timestamps, self.mirrors, self.closed = [], [], False
        FakeStreamExtractor.instances.append(self)

    def extract(self, frame, *, timestamp_ms, mirror_data):
        self.timestamps.append(timestamp_ms)
        self.mirrors.append(mirror_data)
        feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        if frame[..., 0].mean() <= 200:
            feature[list(_PALM_Y)] = np.nan
        return feature

    def close(self):
        self.closed = True


class StreamTests(unittest.TestCase):
    predictor = None

    @classmethod
    def setUpClass(cls):
        ServerTests.setUpClass()
        cls.predictor, cls.directory = ServerTests.predictor, ServerTests.directory

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        FakeStreamExtractor.instances = []
        self.app = create_app(self.predictor, stream_extractor_factory=FakeStreamExtractor, max_streams=2,
                              stream_idle_s=60.0)
        self.client = self.app.test_client()
        self.hand = encode_frame(np.full((48, 64, 3), 255, dtype=np.uint8))
        self.empty = encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    def open(self) -> str:
        response = self.client.post("/api/stream")
        self.assertEqual(response.status_code, 201, response.get_json())
        self.assertEqual(response.get_json()["window"], 30)
        return response.get_json()["session_id"]

    def send(self, sid, frames, **extra):
        return self.client.post(f"/api/stream/{sid}", json={"frames": frames, **extra})

    def test_window_fills_then_predicts_only_new_frames(self):
        sid = self.open()
        body = self.send(sid, [self.empty] * 25).get_json()
        self.assertEqual((body["ready"], body["buffered"], body["hand_frames"]), (False, 25, 0))
        self.assertNotIn("label", body)

        body = self.send(sid, [self.hand] * 5).get_json()
        self.assertTrue(body["ready"])
        self.assertIn(body["label"], ("xin chào", "cảm ơn"))
        self.assertEqual((body["buffered"], body["frames"], body["hand_frames"]), (30, 30, 5))

        body = self.send(sid, [self.hand] * 5).get_json()          # cửa sổ trượt: giữ 30 frame mới nhất
        self.assertEqual((body["buffered"], body["hand_frames"]), (30, 10))
        extractor, = FakeStreamExtractor.instances
        self.assertEqual(len(extractor.timestamps), 35)          # mỗi frame chỉ trích MỘT lần

    def test_short_window_predicts_early_on_recent_frames(self):
        sid = self.open()
        fast = {"window": 15, "min_frames": 8}
        body = self.send(sid, [self.empty] * 7, **fast).get_json()
        self.assertEqual((body["ready"], body["buffered"], body["window"]), (False, 7, 15))

        body = self.send(sid, [self.hand], **fast).get_json()          # đủ min_frames: dự đoán 8 frame
        self.assertTrue(body["ready"])
        self.assertIn(body["label"], ("xin chào", "cảm ơn"))
        self.assertEqual((body["frames"], body["hand_frames"]), (8, 1))

        body = self.send(sid, [self.hand] * 20, **fast).get_json()     # chỉ 15 frame gần nhất
        self.assertEqual((body["buffered"], body["frames"], body["hand_frames"]), (28, 15, 15))

        body = self.send(sid, [self.empty] * 2).get_json()              # mặc định: cả cửa sổ 30 frame
        self.assertEqual((body["ready"], body["frames"], body["hand_frames"], body["window"]), (True, 30, 21, 30))

    def test_timestamps_strictly_increase(self):
        sid = self.open()
        self.send(sid, [self.empty] * 3, fps=10)
        self.send(sid, [self.empty] * 3, timestamps=[150.4, 150.4, 120.0])
        self.send(sid, [self.empty] * 2, timestamps=[5000, 5066.7])
        extractor, = FakeStreamExtractor.instances
        self.assertEqual(extractor.timestamps, [0, 100, 200, 201, 202, 203, 5000, 5067])

    def test_reset_clears_window_but_keeps_tracking(self):
        sid = self.open()
        self.send(sid, [self.hand] * 30)
        body = self.send(sid, [self.empty] * 4, reset=True, mirror_data=True).get_json()
        self.assertEqual((body["ready"], body["buffered"], body["hand_frames"]), (False, 4, 0))
        extractor, = FakeStreamExtractor.instances
        self.assertEqual(extractor.mirrors[-1], True)
        self.assertFalse(extractor.closed)

    def test_close_limit_and_unknown_session(self):
        first, second = self.open(), self.open()
        self.assertEqual(self.client.post("/api/stream").status_code, 503)
        self.assertEqual(self.client.delete(f"/api/stream/{first}").status_code, 204)
        self.assertEqual(self.send(first, [self.empty]).status_code, 404)
        self.open()                                               # đã có chỗ trống
        self.assertEqual(self.client.get("/api/health").get_json()["stream"], {"window": 30, "active": 2, "max": 2})
        del second

    def test_idle_sessions_expire(self):
        app = create_app(self.predictor, stream_extractor_factory=FakeStreamExtractor, stream_idle_s=0.0)
        client = app.test_client()
        sid = client.post("/api/stream").get_json()["session_id"]
        self.assertEqual(client.post(f"/api/stream/{sid}", json={"frames": [self.empty]}).status_code, 404)

    def test_rejects_bad_stream_requests(self):
        sid = self.open()
        cases = [
            {"frames": []},
            {"frames": [self.empty] * 61},
            {"frames": ["not base64!"]},
            {"frames": [self.empty], "timestamps": [1, 2]},
            {"frames": [self.empty], "timestamps": ["1"]},
            {"frames": [self.empty], "reset": "yes"},
            {"frames": [self.empty], "fps": 0},
            {"frames": [self.empty], "window": 31},
            {"frames": [self.empty], "window": 0},
            {"frames": [self.empty], "window": 10.5},
            {"frames": [self.empty], "window": 10, "min_frames": 11},
            {"frames": [self.empty], "min_frames": True},
        ]
        for body in cases:
            with self.subTest(body={k: (v if k != "frames" else len(v)) for k, v in body.items()}):
                response = self.client.post(f"/api/stream/{sid}", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        self.assertEqual(FakeStreamExtractor.instances[0].timestamps, [])


if __name__ == "__main__":
    unittest.main()
