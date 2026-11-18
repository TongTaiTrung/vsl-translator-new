import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.features.extract import SEQ_FEATURE_DIM

from src.model import training
from src.model.training import (Augmenter, Config, calibrate_temperature, default_groups, fit_temperature,
                                full_model_epochs, load_validation_mask, train_one)


def tiny_config(**overrides):
    values = dict(d_model=16, gru_hidden=8, n_blocks=1, warmup_epochs=5, min_epochs=1, patience=100,
                  batch_size=8, extra_hand_probability=0.0, seed=0)
    values.update(overrides)
    return Config(**values)


def tiny_data(n=24, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % classes
    X = rng.normal(size=(n, 30, SEQ_FEATURE_DIM)).astype(np.float32) + y[:, None, None].astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))


class TemperatureTests(unittest.TestCase):
    def test_zero_oof_error_skips_fit_and_keeps_unit_temperature(self):
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
        temperature, status = calibrate_temperature(logits, torch.tensor([0, 1, 2]))
        self.assertEqual(temperature, 1.0)
        self.assertEqual(status, "skipped_no_errors")

    def test_temperature_is_never_sharpened_and_bound_is_reported(self):
        # Gần như đúng hết, một lỗi sát nút: NLL muốn T < 1 -> bị chặn ở T=1 và phải cảnh báo.
        logits = torch.tensor([[3.0, 0.0]] * 20 + [[0.1, 0.0]])
        y = torch.tensor([0] * 20 + [1])
        self.assertGreaterEqual(fit_temperature(logits, y), 1.0 - 1e-6)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            temperature, status = calibrate_temperature(logits, y)
        self.assertEqual(status, "at_bound")
        self.assertAlmostEqual(temperature, 1.0, places=2)
        self.assertIn("cảnh báo", out.getvalue())

    def test_overconfident_errors_fit_inside_bounds(self):
        logits = torch.tensor([[4.0, 0.0]] * 6 + [[0.0, 4.0]] * 6 + [[4.0, 0.0]] * 3 + [[0.0, 4.0]] * 3)
        y = torch.tensor([0] * 6 + [1] * 6 + [1] * 3 + [0] * 3)
        temperature, status = calibrate_temperature(logits, y)
        self.assertEqual(status, "fitted")
        self.assertGreater(temperature, 1.0)


class EpochBudgetTests(unittest.TestCase):
    def test_full_model_budget_extends_past_warmup(self):
        cfg = Config()
        self.assertEqual(full_model_epochs([3, 3, 4, 11, 34], cfg), cfg.warmup_epochs + 20)
        self.assertEqual(full_model_epochs([100, 100], cfg), int(100 * 1.15) + 1)   # CV dài: giữ công thức cũ
        self.assertGreater(full_model_epochs([1], cfg), cfg.warmup_epochs)

    def test_fixed_epoch_model_reaches_end_of_cosine(self):
        X, y = tiny_data()
        cfg = tiny_config(warmup_epochs=2)
        _, info = train_one(X, y, None, None, 3, default_groups(SEQ_FEATURE_DIM), cfg, torch.device("cpu"),
                            fixed_epochs=6, log_every=0)
        self.assertLess(info["final_lr"], 0.05 * cfg.lr)


class SoftPresenceTests(unittest.TestCase):
    def test_weak_soft_presence_does_not_count_as_second_hand(self):
        x = torch.zeros(1, 30, SEQ_FEATURE_DIM)
        presence = torch.zeros(1, 30, 2)
        presence[0, :, 1] = 1.0
        presence[0, :, 0] = 0.2                   # tay trái chỉ khớp yếu -> coi như vắng
        donor = torch.full((1, 30, SEQ_FEATURE_DIM), 7.0)
        _, augmented = Augmenter()(x, 0.5, 1.0, presence, (donor, donor))
        self.assertIsNotNone(augmented)
        self.assertTrue(torch.all(augmented[0, :, 0:9] == 7.0))


class ValidationMaskTests(unittest.TestCase):
    def _npz(self, directory, **arrays):
        path = Path(directory) / "data.npz"
        np.savez(path, X=np.zeros((4, 30, SEQ_FEATURE_DIM), np.float32), y=np.array([0, 0, 1, 1]), **arrays)
        return str(path)

    def test_augmented_flag_builds_validation_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._npz(directory, groups=np.array([0, 0, 1, 1]),
                             augmented=np.array([False, True, False, True]))
            mask = load_validation_mask(path, 4, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_without_augmented_flag_mask_is_not_built(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_validation_mask(self._npz(directory), 4, None))

    def test_cli_refuses_augmented_data_without_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._npz(directory, augmented=np.array([False, True, False, True]))
            argv = ["training", "--data", path, "--out", str(Path(directory) / "out"), "--no-onnx"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(training, "run_training") as run_training:
                with self.assertRaises(ValueError):
                    training.main()
            run_training.assert_not_called()

    def test_cli_passes_validation_mask_to_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._npz(directory, groups=np.array([0, 0, 1, 1]),
                             augmented=np.array([False, True, False, True]))
            argv = ["training", "--data", path, "--out", str(Path(directory) / "out"), "--no-onnx"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(training, "run_training") as run_training:
                training.main()
        np.testing.assert_array_equal(run_training.call_args.kwargs["validation_mask"],
                                      [True, False, True, False])


class EvaluationHookTests(unittest.TestCase):
    def _run(self, directory):
        rng = np.random.default_rng(0)
        y = np.arange(18) % 3
        X = rng.normal(size=(18, 30, SEQ_FEATURE_DIM)).astype(np.float32) + y[:, None, None]
        with contextlib.redirect_stdout(io.StringIO()):
            return training.run_training(X, y, ["a", "b", "c"], np.arange(18), out_dir=directory,
                                         cfg=tiny_config(max_epochs=10), folds=3, add_full=False,
                                         export_onnx_model=False, device=torch.device("cpu"))

    def test_evaluation_report_uses_original_oof_logits(self):
        calls = []

        def oof_report(logits, y, classes, groups=None):
            calls.append((logits.shape, y.shape, list(classes), None if groups is None else groups.shape))
            return {"ok": True}

        fake = types.ModuleType("src.model.evaluation")
        fake.oof_report = oof_report
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"src.model.evaluation": fake}):
            result = self._run(directory)
        self.assertEqual(result["evaluation"], {"ok": True})
        self.assertEqual(calls, [((18, 3), (18,), ["a", "b", "c"], (18,))])
        self.assertIn(result["temperature_status"], {"fitted", "at_bound", "skipped_no_errors"})

    def test_training_runs_without_evaluation_module(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"src.model.evaluation": None}):
            result = self._run(directory)
        self.assertNotIn("evaluation", result)


if __name__ == "__main__":
    unittest.main()
