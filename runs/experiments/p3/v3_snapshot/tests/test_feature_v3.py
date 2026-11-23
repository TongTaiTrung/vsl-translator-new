import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.features import extract as fx
from src.model import training as tr


def _hand(thumb_x: float, index_tip_x: float = 0.0, middle_tip_x: float = 0.33) -> np.ndarray:
    """Bàn tay phải tổng hợp (21,3): lòng bàn tay trong mặt phẳng xy, ngón duỗi theo -y."""
    pts = np.zeros((21, 3))
    for base, x in zip((5, 9, 13, 17), (0.0, 0.33, 0.66, 1.0)):
        for k in range(4):
            pts[base + k] = (x, -1.0 - 0.3 * k, 0.0)
    pts[8, 0], pts[12, 0] = index_tip_x, middle_tip_x
    pts[4] = (thumb_x, -1.1, -0.2)
    pts[1:4] = np.linspace(pts[0], pts[4], 5)[1:4]
    return pts


class HandshapeTests(unittest.TestCase):
    def test_thumb_across_orders_t_n_m_and_crossing_is_negative(self):
        across = [fx._hand_shape_and_orientation(_hand(x), False, None)[0][32] for x in (0.15, 0.5, 0.85)]
        self.assertTrue(across[0] < across[1] < across[2])
        crossed = fx._hand_shape_and_orientation(_hand(0.15, index_tip_x=0.4, middle_tip_x=0.2), False, None)[0]
        straight = fx._hand_shape_and_orientation(_hand(0.15), False, None)[0]
        self.assertLess(crossed[35], 0)
        self.assertGreater(straight[35], 0)

    def test_mirrored_left_hand_matches_right_hand_after_left_mirror_signs(self):
        right = _hand(0.5)
        forearm = np.array([0.3, 0.9, 0.1]) / np.linalg.norm([0.3, 0.9, 0.1])
        flip = np.array([-1.0, 1.0, 1.0])
        hs_r, ori_r = fx._hand_shape_and_orientation(right, False, forearm)
        hs_l, ori_l = fx._hand_shape_and_orientation(right * flip, True, forearm * flip)
        np.testing.assert_allclose(hs_l, hs_r, atol=1e-5)
        expected = ori_r.copy()
        expected[[0, 3, 8]] *= -1                     # palm_dir.x, palm_normal.x, góc cổ tay ngang
        np.testing.assert_allclose(ori_l, expected, atol=1e-5)


class LayoutTests(unittest.TestCase):
    def test_side_columns_partition_sequence(self):
        cols = np.concatenate([fx.SIDE_COLUMNS[k] for k in ("left", "right", "shared")])
        self.assertEqual(sorted(cols.tolist()), list(range(fx.SEQ_FEATURE_DIM)))
        self.assertEqual(fx.FEATURE_SLICES["presence"], fx.PRESENCE_SLICE)
        self.assertEqual(fx.PRESENCE_SLICE.stop, fx.FEATURE_DIM)


def _clip(T: int, left_y: float | None, right_y: float, seed: int = 0) -> np.ndarray:
    """Clip thô (T, FEATURE_DIM) với quỹ đạo mượt; left_y=None -> không có tay trái."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, T)[:, None]
    raw = np.full((T, fx.FEATURE_DIM), np.nan, dtype=np.float32)
    for side, base_y in ((0, left_y), (1, right_y)):
        if base_y is None:
            continue
        cols = fx._side_raw_cols(side)
        raw[:, cols] = 0.3 * np.sin(t + rng.uniform(0, 3, len(cols)))
        hp = fx.FEATURE_SLICES["hand_pos"].start + 9 * side
        raw[:, hp:hp + 9] = 0.2 * np.sin(t + rng.uniform(0, 3, 9))
        raw[:, [hp + 1, hp + 4, hp + 7]] += base_y
        raw[:, fx.PRESENCE_SLICE.start + side] = 0.9
    if left_y is not None:
        lp = raw[:, 0:9].reshape(T, 3, 3)
        rp = raw[:, 9:18].reshape(T, 3, 3)
        d = np.concatenate([np.linalg.norm(lp - rp, axis=-1),
                            np.linalg.norm(lp[:, :, None] - rp[:, None], axis=-1).reshape(T, -1).min(1, keepdims=True)], 1)
        raw[:, fx.FEATURE_SLICES["inter_hand_vec"]] = lp[:, 0] - rp[:, 0]
        raw[:, fx.FEATURE_SLICES["hand_contact"]] = np.exp(-8.0 * d ** 2)
    return raw


class RestGateTests(unittest.TestCase):
    def test_low_hand_is_dropped_and_raised_hand_is_kept(self):
        seq = fx.process_sequence(_clip(30, left_y=1.2, right_y=0.4))
        left, right = fx.SIDE_COLUMNS["left"], fx.SIDE_COLUMNS["right"]
        self.assertTrue(np.all(seq[:, left] == 0))
        self.assertTrue(np.all(seq[:, fx.SIDE_COLUMNS["shared"]] == 0))
        self.assertTrue(np.abs(seq[:, right]).sum() > 0)
        kept = fx.process_sequence(_clip(30, left_y=0.4, right_y=0.4))
        self.assertTrue(np.abs(kept[:, left]).sum() > 0)

    def test_active_presence_ignores_resting_and_missing_hands(self):
        feature = np.full(fx.FEATURE_DIM, np.nan, dtype=np.float32)
        feature[fx._PALM_Y[0]], feature[fx._PALM_Y[1]] = 1.2, 0.4
        self.assertEqual(fx.active_presence(feature).tolist(), [False, True])
        self.assertFalse(fx.active_presence(np.full(fx.FEATURE_DIM, np.nan)).any())


class NormalizationTests(unittest.TestCase):
    def _data(self):
        X = np.stack([fx.process_sequence(_clip(30, None if i % 2 else 0.4, 0.3, seed=i)) for i in range(6)])
        return X.astype(np.float32)

    def test_presence_aware_zeroes_absent_hand_and_matches_ensemble(self):
        X = self._data()
        norm = tr.Normalizer(X, presence_aware=True)
        Z = norm(X)
        self.assertTrue(np.all(Z[1][:, fx.SIDE_COLUMNS["left"]] == 0))
        self.assertTrue(np.all(Z[1][:, fx.SIDE_COLUMNS["shared"]] == 0))
        np.testing.assert_array_equal(Z[0][:, fx.PRESENCE_SLICE], 1.0)
        cfg = tr.Config(arch="handpair", presence_norm=True)
        model = tr.build_model(fx.SEQ_FEATURE_DIM, 3, tr.default_groups(fx.SEQ_FEATURE_DIM), cfg).eval()
        ensemble = tr.VSLEnsemble([model], norm.mean[None], norm.std[None], presence_aware=True).eval()
        with torch.no_grad():
            expected = torch.softmax(model(torch.from_numpy(Z)), -1)
            np.testing.assert_allclose(ensemble(torch.from_numpy(X)).numpy(), expected.numpy(), atol=1e-5)

    def test_recomputed_shared_columns_match_process_sequence(self):
        two = fx.process_sequence(_clip(30, 0.4, 0.3, seed=3)).astype(np.float32)
        one = two.copy()
        one[:, fx.SIDE_COLUMNS["left"]] = 0.0
        one[:, fx.SIDE_COLUMNS["shared"]] = 0.0
        stats = (torch.zeros(fx.SEQ_FEATURE_DIM), torch.ones(fx.SEQ_FEATURE_DIM), False)
        out = tr.add_distractor_hand_features(torch.from_numpy(one)[None], torch.from_numpy(two)[None], 0, stats)
        shared = fx.SIDE_COLUMNS["shared"]
        np.testing.assert_allclose(out[0][:, shared].numpy(), two[:, shared], atol=1e-3)


class HandPairModelTests(unittest.TestCase):
    def test_forward_shape_head_and_old_checkpoint_without_arch(self):
        D = fx.SEQ_FEATURE_DIM
        cfg = tr.Config(arch="handpair", presence_norm=True)
        model = tr.build_model(D, 4, tr.default_groups(D), cfg).eval()
        self.assertEqual(tuple(model(torch.zeros(2, 30, D)).shape), (2, 4))
        self.assertTrue(hasattr(model, "head"))
        with self.assertRaises(ValueError):
            tr.Config(arch="handpair")                   # handpair cần presence_norm

        old_cfg = tr.Config()
        old = tr.build_model(D, 2, tr.default_groups(D), old_cfg).eval()
        norm = tr.Normalizer(np.zeros((2, 30, D), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            tr.save_ensemble(path, [old], [norm], old_cfg, tr.default_groups(D), ["A", "B"], 1.0, 0.0, (30, D))
            ck = torch.load(path, weights_only=False)
            for key in ("arch", "presence_norm", "distractor_geometry"):
                del ck["cfg"][key]
            torch.save(ck, path)
            ensemble, _ = tr.load_ensemble(path, device="cpu")
        self.assertIsInstance(ensemble.models[0], tr.VSLNet)
        self.assertFalse(ensemble.presence_aware)


if __name__ == "__main__":
    unittest.main()
