import base64
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.app.server import create_app
from src.features.extract import FEATURE_DIM, SEQ_FEATURE_DIM
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


if __name__ == "__main__":
    unittest.main()
