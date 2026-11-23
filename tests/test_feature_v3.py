import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        hs_r, ori_r, loc_r = fx._hand_shape_and_orientation(right, False, forearm)
        hs_l, ori_l, loc_l = fx._hand_shape_and_orientation(right * flip, True, forearm * flip)
        np.testing.assert_allclose(hs_l, hs_r, atol=1e-5)
        np.testing.assert_allclose(loc_l, loc_r, atol=1e-5)       # hand_local tự đối xứng gương
        expected = ori_r.copy()
        expected[[0, 3, 8]] *= -1                     # palm_dir.x, palm_normal.x, góc cổ tay ngang
        np.testing.assert_allclose(ori_l, expected, atol=1e-5)


class HandLocalTests(unittest.TestCase):
    def test_hand_local_invariant_to_rotation_scale_and_translation(self):
        hand = _hand(0.5, index_tip_x=0.1)
        a = np.radians(37.0)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rot = rot @ np.array([[1, 0, 0], [0, np.cos(1.1), -np.sin(1.1)], [0, np.sin(1.1), np.cos(1.1)]])
        base = fx._hand_shape_and_orientation(hand, False, None)[2]
        moved = fx._hand_shape_and_orientation(2.5 * hand @ rot.T + 3.0, False, None)[2]
        self.assertEqual(base.shape, (60,))
        np.testing.assert_allclose(moved, base, atol=1e-5)

    def test_hand_local_separates_thumb_positions(self):
        t, m = (fx._hand_shape_and_orientation(_hand(x), False, None)[2] for x in (0.15, 0.85))
        self.assertGreater(np.abs(t - m).max(), 0.3)

    def test_left_mirror_flips_face_vec_x(self):
        perm, sign = fx.LEFT_MIRROR
        left = fx.SIDE_COLUMNS["left"][perm]
        fv = fx.FEATURE_SLICES["face_vec"].start
        for c in (fv, fv + 3):
            self.assertEqual(sign[int(np.flatnonzero(left == c)[0])], -1.0)
        self.assertEqual(sign[int(np.flatnonzero(left == fv + 1)[0])], 1.0)


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


def _pt(row, vis=1.0):
    return SimpleNamespace(x=float(row[0]), y=float(row[1]), z=float(row[2]), visibility=vis)


def _scene():
    """Pose (33,3) toạ độ ảnh + tay phải (ảnh, world) đưa ra trước ngực; hông trong khung."""
    pose = np.tile([0.5, 0.5, 0.0], (33, 1))
    rows = {0: (0.5, 0.30, -0.30), 2: (0.52, 0.28, -0.28), 5: (0.48, 0.28, -0.28), 7: (0.55, 0.29, -0.1),
            8: (0.45, 0.29, -0.1), 9: (0.51, 0.33, -0.28), 10: (0.49, 0.33, -0.28), 11: (0.6, 0.45, 0.0),
            12: (0.4, 0.45, 0.0), 13: (0.65, 0.6, -0.1), 14: (0.35, 0.6, -0.2), 15: (0.62, 0.75, -0.1),
            16: (0.42, 0.52, -0.4), 23: (0.57, 0.85, 0.0), 24: (0.43, 0.85, 0.0)}
    for k, v in rows.items():
        pose[k] = v
    a = 0.6
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    world = (_hand(0.5) - _hand(0.5)[0]) @ tilt.T * 0.02
    image = world * 2.0
    image[:, :2] += pose[16, :2]
    image[:, 2] -= image[0, 2]
    return pose, image, world


def _results(pose, image, world, hip_vis=1.0):
    vis = [hip_vis if i in (23, 24) else 1.0 for i in range(33)]
    pose_res = SimpleNamespace(pose_landmarks=[[_pt(r, v) for r, v in zip(pose, vis)]])
    hand_res = SimpleNamespace(hand_landmarks=[[_pt(r) for r in image]],
                               hand_world_landmarks=[[_pt(r) for r in world]])
    return pose_res, hand_res


def _rotate_scene(pose, image, world, degrees):
    """Camera thấp hơn `degrees`: xoay cảnh quanh trục x qua giữa vai, trong thang độ sâu thật (z / POSE_Z_GAIN)."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)

    def rot(pts, gain):
        q = np.array(pts, dtype=np.float64)
        q[:, 2] /= gain
        y, z = q[:, 1].copy(), q[:, 2].copy()
        q[:, 1], q[:, 2] = y * c + z * s, z * c - y * s
        q[:, 2] *= gain
        return q

    origin = 0.5 * (pose[11] + pose[12])
    pose_r = rot(pose - origin, fx.POSE_Z_GAIN) + origin
    hand_abs = image.copy()
    hand_abs[:, 2] += pose[16, 2]
    hand_abs = rot(hand_abs - origin, fx.POSE_Z_GAIN) + origin
    hand_abs[:, 2] -= pose_r[16, 2]
    return pose_r, hand_abs, rot(world, 1.0)


class FeatureV6Tests(unittest.TestCase):
    """v6: hệ trục thân không phụ thuộc hông; rotate_view_pitch mô phỏng camera cao/thấp để augment."""

    def test_version_bumped(self):
        self.assertEqual(fx.FEATURE_VERSION, 6)

    def test_frame_does_not_depend_on_hip_visibility(self):
        pose, image, world = _scene()
        pose[23, 2] = pose[24, 2] = -0.5            # hông nghiêng sâu: v5 sẽ đổi pitch của hệ trục khi hông "thấy được"
        seen = fx.extract(*_results(pose, image, world, hip_vis=0.9))
        hidden = fx.extract(*_results(pose, image, world, hip_vis=0.1))
        np.testing.assert_array_equal(seen, hidden)

    def test_rotate_view_pitch_matches_extracting_a_rotated_scene(self):
        pose, image, world = _scene()
        base = fx.extract(*_results(pose, image, world))
        S = fx.FEATURE_SLICES
        o = S["orientation"].start
        for degrees in (12.0, -8.0):
            moved = fx.extract(*_results(*_rotate_scene(pose, image, world, degrees)))
            predicted = fx.rotate_view_pitch(base, degrees)
            cols = np.r_[S["hand_pos"], S["face_vec"], o:o + 6, o + 9:o + 15]
            live = cols[np.isfinite(base[cols])]
            self.assertGreater(np.abs(moved[live] - base[live]).max(), 0.05)       # góc nhìn thật sự làm lệch
            np.testing.assert_allclose(predicted[live], moved[live], atol=2e-3)

    def test_rotate_view_pitch_geometry_and_invariants(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5, fx.FEATURE_DIM)).astype(np.float32)
        x[1, :9] = np.nan
        np.testing.assert_allclose(fx.rotate_view_pitch(x, 0.0), x, atol=1e-6)
        np.testing.assert_allclose(fx.rotate_view_pitch(fx.rotate_view_pitch(x, 15.0), -15.0), x, atol=1e-5)
        y = fx.rotate_view_pitch(x, 15.0)
        self.assertTrue(np.isnan(y[1, :9]).all())
        S = fx.FEATURE_SLICES
        for name in ("body_contact", "hand_contact", "hand_scale", "handshape", "hand_local", "presence"):
            np.testing.assert_array_equal(y[:, S[name]], x[:, S[name]])
        o = S["orientation"].start
        np.testing.assert_array_equal(y[:, o + 6:o + 9], x[:, o + 6:o + 9])     # góc cổ tay
        for k in (0, 3, 9, 12):                                                  # palm_dir/normal giữ độ dài
            np.testing.assert_allclose(np.linalg.norm(y[:, o + k:o + k + 3], axis=1),
                                       np.linalg.norm(x[:, o + k:o + k + 3], axis=1), rtol=1e-5)
        # Camera thấp hơn: tay ở trước thân (z < 0) hiện cao hơn (y nhỏ hơn); x không đổi.
        point = np.zeros(fx.FEATURE_DIM, dtype=np.float32)
        point[S["hand_pos"].start + 9: S["hand_pos"].start + 12] = (0.1, 0.4, -1.0)
        lifted = fx.rotate_view_pitch(point, 10.0)[S["hand_pos"].start + 9: S["hand_pos"].start + 12]
        self.assertAlmostEqual(float(lifted[0]), 0.1, places=6)
        self.assertLess(lifted[1], 0.4)
        with self.assertRaises(ValueError):
            fx.rotate_view_pitch(np.zeros((2, 5)), 5.0)


if __name__ == "__main__":
    unittest.main()


class SignFormerTests(unittest.TestCase):
    def _model(self):
        cfg = tr.Config(arch="former", presence_norm=True, base_dropout=0.0)
        return tr.build_model(fx.SEQ_FEATURE_DIM, 5, tr.default_groups(fx.SEQ_FEATURE_DIM), cfg).eval()

    def test_forward_any_length_and_has_head(self):
        torch.manual_seed(0)
        model = self._model()
        self.assertTrue(hasattr(model, "head"))
        for T in (30, 24):
            x = torch.randn(3, T, fx.SEQ_FEATURE_DIM)
            x[..., fx.PRESENCE_SLICE] = 1.0
            self.assertEqual(tuple(model(x).shape), (3, 5))

    def test_absent_hand_ignores_its_columns(self):
        torch.manual_seed(0)
        model = self._model()
        x = torch.randn(2, 30, fx.SEQ_FEATURE_DIM)
        x[..., fx.PRESENCE_SLICE.start] = 0.0             # tay trái vắng
        x[..., fx.PRESENCE_SLICE.start + 1] = 1.0
        y = x.clone()
        y[..., torch.as_tensor(fx.SIDE_COLUMNS["left"][fx.SIDE_COLUMNS["left"] != fx.PRESENCE_SLICE.start])] += 5.0
        y[..., torch.as_tensor(fx.SIDE_COLUMNS["shared"])] += 5.0
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(y))

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        model = self._model()
        cfg = tr.Config(arch="former", presence_norm=True)
        D = fx.SEQ_FEATURE_DIM
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ensemble.pt"
            tr.save_ensemble(path, [model], [SimpleNamespace(mean=np.zeros(D, np.float32), std=np.ones(D, np.float32))], cfg,
                             tr.default_groups(D), list("abcde"), 1.0, 0.2, (30, D))
            ens, ck = tr.load_ensemble(path)
            self.assertEqual(ck["cfg"]["arch"], "former")
            self.assertEqual(tuple(ens(torch.zeros(1, 30, D)).shape), (1, 5))


class FeatureV5Tests(unittest.TestCase):
    """v5: tiếp xúc cơ thể không bị trục z nhiễu lấn át; hand_scale ước lượng độ sâu bàn tay."""

    def _anchors(self):
        return fx._body_anchors(np.zeros((33, 3), dtype=np.float32))

    def test_palm_on_chest_registers_contact_despite_depth_offset(self):
        # "tôi": nắm tay đặt trên ngực, z pose lệch ~-0.95 vai dù tay chạm thân.
        palm = np.array([-0.09, 0.34, -0.95])
        pts = palm + np.array([[0, 0, 0], [0.05, -0.05, 0], [0.0, -0.1, 0], [0, -0.08, 0],
                               [-0.03, -0.06, 0], [-0.06, -0.04, 0]])
        chest = fx._body_contacts(pts, self._anchors())[7]
        self.assertGreater(chest, 0.5)

    def test_hand_near_shoulder_away_from_chest_has_no_chest_contact(self):
        # "m": tay ở ngang vai phải, xa ngực.
        palm = np.array([-0.62, -0.12, -2.0])
        pts = palm + np.array([[0, 0, 0], [0.05, 0.1, 0], [0.1, 0.4, 0], [0.05, 0.42, 0],
                               [0, 0.4, 0], [-0.05, 0.3, 0]])
        self.assertLess(fx._body_contacts(pts, self._anchors())[7], 0.1)

    def test_hand_scale_grows_with_size_and_ignores_rotation_and_mirror(self):
        hand = _hand(0.5) * 0.05 + np.array([0.5, 0.5, 0.0])
        base = fx._hand_scale(hand, 0.25)
        self.assertGreater(fx._hand_scale((hand - 0.5) * 1.5 + 0.5, 0.25), base * 1.4)
        self.assertAlmostEqual(fx._hand_scale(hand, 0.5), base / 2, places=5)
        a = 0.9
        rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        self.assertAlmostEqual(fx._hand_scale(hand @ rot.T, 0.25), base, places=5)
        self.assertAlmostEqual(fx._hand_scale(hand * np.array([-1, 1, 1]), 0.25), base, places=5)

    def test_hand_scale_columns_are_per_side_and_not_flipped_by_mirror(self):
        sl = fx.FEATURE_SLICES["hand_scale"]
        self.assertEqual(sl.stop - sl.start, 2)
        self.assertIn(sl.start, fx.SIDE_COLUMNS["left"])
        self.assertIn(sl.start + 1, fx.SIDE_COLUMNS["right"])
        perm, sign = fx.LEFT_MIRROR
        left = fx.SIDE_COLUMNS["left"][perm]
        self.assertEqual(sign[int(np.flatnonzero(left == sl.start)[0])], 1.0)
        # Nằm trong nhóm vị trí của GroupStem (trước thủ hình).
        self.assertLess(sl.stop, fx.FEATURE_SLICES["handshape"].start + 1)


class StrokeTests(unittest.TestCase):
    """Nét: cú di chuyển mạnh nhất trong cửa sổ, lặp ở mọi frame (dấu sắc/huyền/nặng chỉ vẩy ~4 frame)."""

    @staticmethod
    def _clip(start: int, dy: float, side: int = 1, T: int = 30, absent_until: int = 0) -> np.ndarray:
        raw = np.full((T, fx.FEATURE_DIM), np.nan, dtype=np.float32)
        raw[:, fx.FEATURE_SLICES["presence"]] = 0.0
        hp = fx.FEATURE_SLICES["hand_pos"].start + 9 * side
        x = 0.3 if side else -0.3
        y = 0.2 + dy * np.clip((np.arange(T) - start) / 4.0, 0.0, 1.0)      # vẩy trong 4 frame
        for k, off in enumerate((0.0, 0.1, -0.2)):                         # lòng bàn tay, ngón cái, đầu trỏ
            raw[:, hp + 3 * k], raw[:, hp + 3 * k + 1], raw[:, hp + 3 * k + 2] = x, y + off, 0.0
        raw[:, fx.PRESENCE_SLICE.start + side] = 1.0
        raw[:absent_until, fx._side_raw_cols(side)] = np.nan
        return raw

    def test_stroke_is_constant_and_independent_of_when_the_flick_happens(self):
        right = fx.STROKE_SLICE.start + 6
        early, late = (fx.process_sequence(self._clip(s, -0.6)) for s in (3, 22))
        for seq in (early, late):
            self.assertTrue(np.allclose(seq[:, fx.STROKE_SLICE], seq[:1, fx.STROKE_SLICE]))
        np.testing.assert_allclose(early[0, fx.STROKE_SLICE], late[0, fx.STROKE_SLICE], atol=1e-5)
        self.assertLess(early[0, right + 1], -0.5)                  # đầu trỏ đi lên (y giảm)
        self.assertLess(early[0, right + 4], -0.5)                  # lòng bàn tay cũng vậy
        self.assertTrue(np.all(early[0, fx.STROKE_SLICE.start:right] == 0))   # tay trái vắng -> 0

    def test_up_and_down_strokes_have_opposite_signs_and_static_hand_is_zero(self):
        y = fx.STROKE_SLICE.start + 7
        up, down, still = (fx.process_sequence(self._clip(10, dy))[0, y] for dy in (-0.6, 0.6, 0.0))
        self.assertLess(up, -0.5)
        self.assertGreater(down, 0.5)
        self.assertEqual(still, 0.0)

    def test_hand_entering_the_frame_is_not_a_stroke(self):
        seq = fx.process_sequence(self._clip(0, 0.0, absent_until=12))
        self.assertTrue(np.all(seq[:, fx.STROKE_SLICE] == 0))

    def test_left_stroke_mirrors_to_right_and_has_its_own_group(self):
        left = fx.process_sequence(self._clip(10, -0.6, side=0))
        right = fx.process_sequence(self._clip(10, -0.6, side=1))
        perm, sign = fx.LEFT_MIRROR
        cols = fx.SIDE_COLUMNS["left"]
        st = np.isin(cols, np.arange(fx.STROKE_SLICE.start, fx.STROKE_SLICE.start + 6))
        mirrored = (left[0, cols[perm]] * sign)[st[perm]]
        np.testing.assert_allclose(mirrored, right[0, fx.STROKE_SLICE.start + 6:], atol=1e-5)
        self.assertIn((fx.STROKE_SLICE.start, fx.SEQ_FEATURE_DIM), tr.default_groups(fx.SEQ_FEATURE_DIM))

