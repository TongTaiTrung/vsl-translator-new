import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.app.server import create_app, repeat_frames
from src.features.extract import _PALM_Y, FEATURE_DIM, REST_Y, SEQ_FEATURE_DIM, active_presence, extract
from src.features.landmarks import dict_to_results, results_to_dict
from src.model.predict import VSLPredictor
from src.model.training import Config, Normalizer, VSLNet, default_groups, save_ensemble


def encode_frame(frame: np.ndarray, data_url: bool = True) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    text = base64.b64encode(buffer.tobytes()).decode()
    return f"data:image/jpeg;base64,{text}" if data_url else text


def fake_landmarks(hand: bool = True, seed: int = 0) -> dict:
    """Landmark một frame: người đứng giữa khung (vai rộng 0.2), có/không một tay giơ trước ngực."""
    rng = np.random.default_rng(seed)
    pose = np.column_stack([rng.uniform(0.4, 0.6, 33), rng.uniform(0.2, 0.4, 33), rng.uniform(-0.1, 0.1, 33),
                            np.ones(33)])
    pose[11, :2], pose[12, :2] = (0.6, 0.5), (0.4, 0.5)          # vai trái / phải
    pose[23, :2], pose[24, :2] = (0.57, 0.9), (0.43, 0.9)        # hông
    pose[13, :2], pose[15, :2] = (0.65, 0.7), (0.58, 0.55)       # khuỷu, cổ tay trái
    pose[14, :2], pose[16, :2] = (0.35, 0.7), (0.3, 0.85)
    out = {"pose": pose.tolist(), "hands": [], "hands_world": []}
    if hand:
        img = np.column_stack([0.58 + rng.uniform(-0.04, 0.04, 21), 0.5 + rng.uniform(-0.06, 0.0, 21),
                               rng.uniform(-0.02, 0.02, 21)])
        img[0] = (0.58, 0.55, 0.0)                               # cổ tay trùng cổ tay pose
        out["hands"] = [img.tolist()]
        out["hands_world"] = [((img - img[0]) * 0.4).tolist()]
    return out


class LandmarkTests(unittest.TestCase):
    def test_round_trip_and_extract(self):
        frame = fake_landmarks()
        pose_res, hand_res = dict_to_results(frame)
        self.assertEqual(results_to_dict(pose_res, hand_res), frame)
        feature = extract(pose_res, hand_res)
        self.assertEqual(feature.shape, (FEATURE_DIM,))
        self.assertEqual(active_presence(feature).tolist(), [True, False])
        self.assertTrue(np.isnan(extract(*dict_to_results({"pose": None, "hands": []}))).all())

    def test_rejects_malformed_frames(self):
        good = fake_landmarks()
        bad = [[], {"pose": good["pose"][:32]}, {"pose": [[0, 0, float("nan"), 1]] * 33},
               {"pose": None, "hands": good["hands"] * 3}, {"pose": None, "hands": [[[0, 0]] * 21]},
               {**good, "hands_world": []}, {"pose": "x"}]
        for frame in bad:
            with self.subTest(frame=str(frame)[:40]), self.assertRaises(ValueError):
                dict_to_results(frame)


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

    def test_predict_accepts_landmarks(self):
        landmarks = [fake_landmarks(hand=i % 3 != 0, seed=i) for i in range(30)]
        response = self.client.post("/api/predict", json={"landmarks": landmarks, "mirror_data": True})
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual((body["frames"], body["hand_frames"]), (30, 20))
        self.assertEqual(self.calls, [])                          # không chạy MediaPipe trên server
        for bad in ([fake_landmarks()] * 29, [{"pose": [[0, 0, 0]] * 33}] * 30, "x"):
            self.assertEqual(self.client.post("/api/predict", json={"landmarks": bad}).status_code, 400)

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

    def test_fill_repeat_copies_short_window_to_full_length(self):
        sid = self.open()
        seen = []
        original = self.predictor.predict_raw

        def spy(features, **kwargs):
            seen.append(np.array(features))
            return original(features, **kwargs)

        with mock.patch.object(self.predictor, "predict_raw", side_effect=spy):
            body = self.send(sid, [self.hand], window=1, min_frames=1, fill="repeat").get_json()
            self.assertEqual((body["ready"], body["frames"], body["hand_frames"]), (True, 1, 1))
            self.send(sid, [self.empty], window=2, min_frames=1, fill="repeat")
            self.send(sid, [self.hand], window=2, min_frames=1)                  # mặc định: nội suy
        one, two, interp = seen
        same = lambda rows, ref: all(np.array_equal(r, ref, equal_nan=True) for r in rows)   # frame trống có NaN
        self.assertEqual(one.shape[0], 30)
        self.assertTrue(same(one, one[0]))
        # 2 frame thật → 15 bản của frame trước rồi 15 bản của frame sau (không có frame nội suy ở giữa).
        self.assertEqual(two.shape[0], 30)
        self.assertTrue(same(two[:15], two[0]) and same(two[15:], two[-1]))
        self.assertFalse(np.array_equal(two[0], two[-1], equal_nan=True))
        self.assertEqual(interp.shape[0], 2)

    def test_repeat_frames_keeps_long_windows(self):
        features = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(repeat_frames(features, 30)[:, 0].tolist(), [0.0] * 10 + [4.0] * 10 + [8.0] * 10)
        long = np.zeros((40, 4), dtype=np.float32)
        self.assertIs(repeat_frames(long, 30), long)

    def test_predict_false_only_extracts(self):
        sid = self.open()
        body = self.send(sid, [self.hand] * 30, predict=False).get_json()
        self.assertEqual((body["ready"], body["predicted"], body["buffered"], body["hand_frames"]), (True, False, 30, 30))
        self.assertNotIn("label", body)
        body = self.send(sid, [self.hand]).get_json()                   # mặc định predict = true
        self.assertTrue(body["predicted"])
        self.assertIn(body["label"], ("xin chào", "cảm ơn"))
        self.assertEqual(self.send(sid, [self.hand], predict="yes").status_code, 400)

    def test_multipart_binary_frames(self):
        sid = self.open()
        jpeg = base64.b64decode(self.hand.partition(",")[2])
        data = {"meta": json.dumps({"timestamps": [10, 77], "window": 2, "top_k": 2}),
                "frames": [(io.BytesIO(jpeg), "0.jpg", "image/jpeg"), (io.BytesIO(jpeg), "1.jpg", "image/jpeg")]}
        response = self.client.post(f"/api/stream/{sid}", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual((body["predicted"], body["frames"], body["hand_frames"], len(body["top"])), (True, 2, 2, 2))
        extractor, = FakeStreamExtractor.instances
        self.assertEqual(extractor.timestamps, [10, 77])

        bad = [{"meta": "{not json", "frames": [(io.BytesIO(jpeg), "0.jpg")]},
               {"meta": "{}"},
               {"meta": "{}", "frames": [(io.BytesIO(b"xx"), "0.jpg")]}]
        for data in bad:
            with self.subTest(meta=data["meta"], files=len(data.get("frames", []))):
                response = self.client.post(f"/api/stream/{sid}", data=data, content_type="multipart/form-data")
                self.assertEqual(response.status_code, 400)

    def test_landmark_frames_skip_server_mediapipe(self):
        response = self.client.post("/api/stream")
        self.assertTrue(response.get_json()["landmarks"])
        sid = response.get_json()["session_id"]
        send = lambda frames, **extra: self.client.post(f"/api/stream/{sid}", json={"landmarks": frames, **extra})
        body = send([fake_landmarks(seed=i) for i in range(29)], predict=False).get_json()
        self.assertEqual((body["ready"], body["buffered"], body["hand_frames"]), (False, 29, 29))
        body = send([fake_landmarks(hand=False)], timestamps=[1933.0]).get_json()     # frame thứ 30 ở 15 fps
        self.assertEqual((body["predicted"], body["frames"], body["hand_frames"]), (True, 30, 29))
        self.assertIn(body["label"], ("xin chào", "cảm ơn"))
        body = send([fake_landmarks(hand=False)] * 2, reset=True, window=10).get_json()
        self.assertEqual((body["ready"], body["buffered"], body["hand_frames"]), (False, 2, 0))
        self.assertEqual(FakeStreamExtractor.instances, [])       # phiên landmark không tạo landmarker
        for bad in ([], [fake_landmarks()] * 61, [{"pose": None, "hands": [[[0, 0, 0]]]}]):
            self.assertEqual(send(bad).status_code, 400)
        self.assertEqual(send([fake_landmarks()], timestamps=[1, 2]).status_code, 400)

    def test_window_follows_capture_time_not_frame_count(self):
        """Điện thoại chậm (7.5 fps) hay máy nhanh (30 fps): cửa sổ luôn là 2 giây như lúc train (30 frame ở 15 fps)."""
        sid = self.client.post("/api/stream").get_json()["session_id"]
        send = lambda frames, **extra: self.client.post(f"/api/stream/{sid}", json={"landmarks": frames, **extra})
        slow = [1000 + i * 1000 / 7.5 for i in range(16)]               # 16 frame trải 2 giây
        body = send([fake_landmarks(seed=i) for i in range(16)], timestamps=slow).get_json()
        self.assertTrue(body["ready"])
        self.assertEqual(body["buffered"], 30)
        self.assertEqual(body["frames"], 15)                            # frame đầu đã ra ngoài 2 giây
        self.assertEqual(body["hand_frames"], 30)                       # quy về đơn vị frame 15 fps
        self.assertIn(body["label"], ("xin chào", "cảm ơn"))

        fast = [9000 + i * 1000 / 30 for i in range(40)]                # 40 frame chỉ trải 1.33 giây
        body = send([fake_landmarks(seed=i) for i in range(40)], timestamps=fast, reset=True).get_json()
        self.assertEqual((body["ready"], body["buffered"], body["hand_frames"]), (False, 20, 20))

        more = [fast[-1] + (i + 1) * 1000 / 30 for i in range(30)]      # thêm 1 giây: cửa sổ chỉ giữ 2 giây gần nhất
        body = send([fake_landmarks(seed=i) for i in range(30)], timestamps=more).get_json()
        self.assertEqual((body["ready"], body["buffered"], body["hand_frames"]), (True, 30, 30))
        self.assertEqual(body["frames"], 60)

    def test_cors_preflight_is_cacheable(self):
        response = self.client.options("/api/stream")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.headers["Access-Control-Max-Age"], "86400")

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
            {"frames": [self.empty], "fill": "nearest"},
        ]
        for body in cases:
            with self.subTest(body={k: (v if k != "frames" else len(v)) for k, v in body.items()}):
                response = self.client.post(f"/api/stream/{sid}", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        self.assertEqual([e.timestamps for e in FakeStreamExtractor.instances], [])


if __name__ == "__main__":
    unittest.main()
