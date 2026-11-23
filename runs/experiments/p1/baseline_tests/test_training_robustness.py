import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.features.extract import extract
from src.model.training import (Augmenter, add_distractor_hand_features, confusion_pairs,
                                closest_rivals, hard_negative_margin_loss, load_ensemble, Config)


class TrainingRobustnessTests(unittest.TestCase):
    def test_training_defaults_preserve_distinct_signs(self):
        config = Config()
        self.assertEqual(config.head, "linear")
        self.assertEqual(config.mixup_alpha, 0.0)
        self.assertGreater(config.hard_negative_weight, 0)

    def test_hard_negative_margin_targets_closest_wrong_class(self):
        logits = torch.tensor([[2.0, 1.7, -3.0], [4.0, 0.0, -2.0]], requires_grad=True)
        loss = hard_negative_margin_loss(logits, torch.tensor([0, 0]), margin=0.8)
        self.assertAlmostEqual(loss.item(), 0.25, places=5)
        loss.backward()
        self.assertLess(logits.grad[0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 1].item(), 0)
        self.assertEqual(logits.grad[0, 2].item(), 0)

    def test_confusion_report_uses_only_validation_rows(self):
        logits = np.array([[1.0, 2.0], [3.0, 0.0], [3.0, 0.0]])
        report = confusion_pairs(logits, np.array([0, 0, 1]), ["A", "B"],
                                 np.array([True, True, False]))
        self.assertEqual(report, [{"actual": "A", "predicted": "B", "count": 1}])

    def test_closest_rival_report_ranks_mean_competitor_probability(self):
        probabilities = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.4, 0.5, 0.1]])
        report = closest_rivals(probabilities, np.array([0, 0, 1]), ["A", "B", "C"])
        self.assertEqual(report[0]["actual"], "B")
        self.assertEqual(report[0]["rival"], "A")
        self.assertEqual(report[0]["samples"], 1)

    def test_adding_one_hand_keeps_original_hand_and_neutralizes_shared_features(self):
        x = torch.arange(2 * 4 * 170, dtype=torch.float32).reshape(2, 4, 170)
        donor = torch.full_like(x, 9.0)
        masked = add_distractor_hand_features(x, donor, torch.tensor([0, 1]))

        self.assertTrue(torch.all(masked[0, :, 0:9] == 9))
        self.assertTrue(torch.all(masked[1, :, 9:18] == 9))
        self.assertTrue(torch.all(masked[0, :, 9:18] == x[0, :, 9:18]))
        self.assertTrue(torch.all(masked[1, :, 0:9] == x[1, :, 0:9]))

        for start, end in ((18, 21), (43, 47), (149, 152)):
            self.assertTrue(torch.all(masked[0, :, start:end] == 0))
        for start, end in ((18, 21), (43, 47), (149, 152)):
            self.assertTrue(torch.all(masked[1, :, start:end] == 0))
        self.assertTrue(torch.all(masked[0, :, 129:130] == 9))
        self.assertTrue(torch.all(masked[1, :, 130:131] == 9))
        self.assertTrue(torch.all(masked[0, :, 47:79] == 9))
        self.assertTrue(torch.all(masked[1, :, 79:111] == 9))
        self.assertTrue(torch.all(masked[0, :, 152:161] == 9))
        self.assertTrue(torch.all(masked[1, :, 161:170] == 9))

    def test_unknown_feature_layout_is_left_unchanged(self):
        x = torch.ones(2, 3, 20)
        self.assertTrue(torch.equal(add_distractor_hand_features(x, x, 0), x))

    def test_extra_hand_is_added_only_to_one_hand_clips(self):
        x = torch.zeros(2, 30, 170)
        presence = torch.zeros(2, 30, 2)
        presence[0, :, 1] = 1.0
        presence[1] = 1.0
        donor = torch.full((1, 30, 170), 7.0)

        original, augmented = Augmenter()(x, 0.5, 1.0, presence, (donor, donor))

        self.assertIsNotNone(augmented)
        self.assertTrue(torch.all(augmented[0, :, 0:9] == 7.0))
        self.assertTrue(torch.equal(augmented[1], original[1]))

    def test_missing_hand_has_no_pose_wrist_substitute(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
        landmarks[11].x, landmarks[12].x = 0.6, 0.4
        landmarks[23].y = landmarks[24].y = 0.8
        pose = SimpleNamespace(pose_landmarks=[landmarks])
        hand = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[])
        feature = extract(pose, hand)
        self.assertTrue(np.isnan(feature[:47]).all())
        np.testing.assert_array_equal(feature[-2:], [0.0, 0.0])

    def test_old_checkpoint_keeps_original_presence_scale(self):
        checkpoints = sorted(Path("runs").glob("*/ensemble.pt"))
        if not checkpoints:
            self.skipTest("No existing checkpoint")
        model, checkpoint = load_ensemble(checkpoints[0])
        if "presence_scale" not in checkpoint["cfg"]:
            self.assertEqual(model.models[0].stem.presence_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
