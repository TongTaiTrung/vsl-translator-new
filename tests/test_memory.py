import base64
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.app.server import create_app, decode_features, encode_features
from src.features.extract import FEATURE_DIM, SEQ_FEATURE_DIM, extract, process_sequence
from src.features.landmarks import dict_to_results
from src.model.memory import MAX_BONUS, Exemplar
from src.model.predict import VSLPredictor
from src.model.training import (Config, HandPairNet, Normalizer, SignFormer, VSLNet, default_groups,
                                logits_and_embedding, save_ensemble)
from tests.test_server import fake_landmarks

CLASSES = ["xin chào", "cảm ơn", "toi"]


def clip(seed: int, hand: bool = True, frames: int = 30) -> np.ndarray:
    """Feature thô của một clip giả: mỗi seed là một 'cách ký' khác nhau."""
    return np.stack([extract(*dict_to_results(fake_landmarks(hand=hand, seed=seed * 100 + i)))
                     for i in range(frames)]).astype(np.float32)


class EmbeddingTests(unittest.TestCase):
    def test_logits_match_forward_for_every_architecture(self):
        torch.manual_seed(0)
        x = torch.randn(3, 30, SEQ_FEATURE_DIM)
        x[..., 0:2] = 1.0
        models = [
            VSLNet(SEQ_FEATURE_DIM, 4, default_groups(SEQ_FEATURE_DIM), Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1)),
            VSLNet(SEQ_FEATURE_DIM, 4, default_groups(SEQ_FEATURE_DIM),
                   Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1, head="cosine")),
            HandPairNet(SEQ_FEATURE_DIM, 4, Config(kernels=(3,), presence_norm=True)),
            SignFormer(SEQ_FEATURE_DIM, 4, Config(kernels=(3,), presence_norm=True)),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__), torch.no_grad():
                model.eval()
                logits, emb = logits_and_embedding(model, x)
                torch.testing.assert_close(logits, model(x))
                torch.testing.assert_close(emb.norm(dim=-1), torch.ones(3))


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name) / "ensemble.pt"
        torch.manual_seed(0)
        config = Config(d_model=16, gru_hidden=8, kernels=(3,), n_blocks=1)
        models = [VSLNet(SEQ_FEATURE_DIM, len(CLASSES), default_groups(SEQ_FEATURE_DIM), config).eval()
                  for _ in range(2)]
        data = np.stack([process_sequence(clip(s), target_len=30) for s in range(4)])
        norms = [Normalizer(data), Normalizer(data)]
        save_ensemble(path, models, norms, config, default_groups(SEQ_FEATURE_DIM), CLASSES, 1.0, 0.0,
                      (30, SEQ_FEATURE_DIM))
        cls.predictor = VSLPredictor(path, device="cpu", min_confidence=0.0, min_probability_gap=0.0)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def probs(self, raw, memory=None):
        x = torch.as_tensor(process_sequence(raw, target_len=30))[None]
        with torch.no_grad():
            return (memory if memory is not None else self.predictor.ensemble)(x)[0].numpy()

    def test_empty_or_zero_strength_memory_matches_base_model(self):
        query = clip(1)
        base = self.probs(query)
        for memory in (self.predictor.build_memory([]),
                       self.predictor.build_memory([Exemplar("toi", query)], strength=0.0)):
            np.testing.assert_allclose(self.probs(query, memory), base, atol=1e-6)
        self.assertNotIn("memory", self.predictor.predict_raw(query, ensemble=self.predictor.build_memory([])))

    def test_matching_exemplar_pulls_prediction_to_its_class(self):
        query = clip(1)
        base = self.probs(query)
        target = int(np.argmin(base))                       # lớp mô hình gốc ít tin nhất
        memory = self.predictor.build_memory([Exemplar(CLASSES[target], query)], strength=1.0)
        boosted = self.probs(query, memory)
        self.assertEqual(int(np.argmax(boosted)), target)
        self.assertEqual(memory.last_match["label"], CLASSES[target])
        self.assertAlmostEqual(memory.last_match["similarity"], 1.0, places=4)
        result = self.predictor.predict_raw(query, ensemble=memory)
        self.assertEqual(result["top"][0]["label"], CLASSES[target])
        self.assertEqual(result["memory"]["label"], CLASSES[target])

    def test_bonus_is_bounded_and_class_count_does_not_inflate_it(self):
        query = clip(1)
        one = self.predictor.build_memory([Exemplar("toi", query)], strength=1.0)
        many = self.predictor.build_memory([Exemplar("toi", query)] * 5, strength=1.0)
        np.testing.assert_allclose(self.probs(query, one), self.probs(query, many), atol=1e-6)
        x = torch.as_tensor(process_sequence(query, target_len=30))[None]
        logits, _ = self.predictor.ensemble.logits_and_embeddings(x)
        expected = torch.softmax(logits + MAX_BONUS * torch.tensor([0.0, 0.0, 1.0]), -1).mean(0)[0]
        np.testing.assert_allclose(self.probs(query, one), expected.detach().numpy(), atol=1e-4)

    def test_keeps_newest_per_class_and_ignores_unknown_labels(self):
        exemplars = [Exemplar("toi", clip(s)) for s in range(7)] + [Exemplar("không có", clip(9))]
        memory = self.predictor.build_memory(exemplars, max_per_class=3)
        self.assertEqual(len(memory), 3)
        self.assertEqual(memory.summary()["classes"], {"toi": 3})
        self.assertEqual(memory.summary()["ignored"], ["không có"])
        newest = self.predictor.build_memory([Exemplar("toi", clip(6))])
        torch.testing.assert_close(memory.keys[:, -1], newest.keys[:, 0])

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            self.predictor.build_memory([], strength=1.5)
        with self.assertRaises(ValueError):
            self.predictor.build_memory([Exemplar("toi", np.zeros((3, FEATURE_DIM), np.float32))])


class MemoryServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MemoryTests.setUpClass()
        cls.predictor, cls.directory = MemoryTests.predictor, MemoryTests.directory

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.client = create_app(self.predictor).test_client()

    def sample(self, label="toi", hand=True):
        frames = [fake_landmarks(hand=hand, seed=500 + i) for i in range(30)]
        return self.client.post("/api/memory/sample", json={"label": label, "landmarks": frames})

    def test_features_round_trip(self):
        raw = clip(2)
        raw[0, :5] = np.nan
        decoded = decode_features(encode_features(raw), len(raw))
        np.testing.assert_array_equal(np.isnan(decoded), np.isnan(raw))
        np.testing.assert_array_equal(np.nan_to_num(decoded), np.nan_to_num(raw))

    def test_sample_returns_storable_features(self):
        response = self.sample()
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual((body["frames"], body["hand_frames"], body["label"]), (30, 30, "toi"))
        self.assertEqual(len(body["prediction"]["top"]), 3)
        self.assertEqual(decode_features(body["features"], body["frames"]).shape, (30, FEATURE_DIM))

    def test_sample_without_hands_is_refused(self):
        body = self.sample(hand=False).get_json()
        self.assertEqual((body["ok"], body["reason"], body["prediction"]), (False, "few_hands", None))

    def test_sample_rejects_unknown_label(self):
        self.assertEqual(self.sample(label="không có").status_code, 400)

    def test_stream_with_memory_reports_match(self):
        sample = self.sample().get_json()
        memory = {"strength": 1.0, "exemplars": [{"label": "toi", "features": sample["features"],
                                                  "frames": sample["frames"]}]}
        opened = self.client.post("/api/stream", json={"memory": memory})
        self.assertEqual(opened.status_code, 201, opened.get_json())
        self.assertEqual(opened.get_json()["memory"]["classes"], {"toi": 1})
        sid = opened.get_json()["session_id"]
        frames = [fake_landmarks(seed=500 + i) for i in range(30)]
        body = self.client.post(f"/api/stream/{sid}", json={"landmarks": frames}).get_json()
        self.assertEqual(body["top"][0]["label"], "toi")
        self.assertEqual(body["memory"]["label"], "toi")
        self.assertGreater(body["memory"]["similarity"], 0.99)

        plain = self.client.post("/api/stream", json={})
        self.assertIsNone(plain.get_json()["memory"])
        body = self.client.post(f"/api/stream/{plain.get_json()['session_id']}", json={"landmarks": frames}).get_json()
        self.assertNotIn("memory", body)

    def test_memory_update_keeps_window(self):
        sid = self.client.post("/api/stream", json={}).get_json()["session_id"]
        frames = [fake_landmarks(seed=500 + i) for i in range(30)]
        body = self.client.post(f"/api/stream/{sid}", json={"landmarks": frames[:20], "predict": False}).get_json()
        self.assertEqual(body["buffered"], 20)
        sample = self.sample().get_json()
        memory = {"strength": 1.0, "exemplars": [{"label": "toi", "features": sample["features"],
                                                  "frames": sample["frames"]}]}
        updated = self.client.post(f"/api/stream/{sid}/memory", json={"memory": memory})
        self.assertEqual(updated.status_code, 200, updated.get_json())
        self.assertEqual(updated.get_json()["memory"]["classes"], {"toi": 1})
        body = self.client.post(f"/api/stream/{sid}", json={"landmarks": frames[20:]}).get_json()
        self.assertEqual(body["buffered"], 30)                      # cửa sổ không bị xoá
        self.assertEqual(body["memory"]["label"], "toi")

        off = self.client.post(f"/api/stream/{sid}/memory", json={"memory": None})
        self.assertIsNone(off.get_json()["memory"])
        body = self.client.post(f"/api/stream/{sid}", json={"landmarks": frames[:1]}).get_json()
        self.assertNotIn("memory", body)
        self.assertEqual(self.client.post("/api/stream/khong-co/memory", json={"memory": None}).status_code, 404)
        self.assertEqual(self.client.post(f"/api/stream/{sid}/memory", json={"memory": {"strength": 3}}).status_code, 400)

    def test_stream_rejects_bad_memory(self):
        good = base64.b64encode(np.zeros((30, FEATURE_DIM), "<f4").tobytes()).decode()
        bad = [
            {"memory": []},
            {"memory": {"strength": 2}},
            {"memory": {"exemplars": [{"label": "toi", "features": "@@", "frames": 30}]}},
            {"memory": {"exemplars": [{"label": "toi", "features": good, "frames": 29}]}},
            {"memory": {"exemplars": [{"features": good, "frames": 30}]}},
        ]
        for body in bad:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual(self.client.post("/api/stream", json=body).status_code, 400)


if __name__ == "__main__":
    unittest.main()
