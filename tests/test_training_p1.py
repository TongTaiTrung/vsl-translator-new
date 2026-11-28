"""Tests for training components and calibration."""

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
                                full_model_epochs, load_validation_mask, ReplaySampler, train_one)


def tiny_config(**overrides):
    values = dict(d_model=16, gru_hidden=8, n_blocks=1, warmup_epochs=5, min_epochs=1, patience=100,
                  batch_size=8, extra_hand_probability=0.0, seed=0)
    values.update(overrides)
    return Config(**values)


def tiny_data(n=24, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % classes
    sequences = rng.normal(size=(n, 30, SEQ_FEATURE_DIM)).astype(np.float32) + labels[:, None, None].astype(np.float32)
    return torch.from_numpy(sequences), torch.from_numpy(labels.astype(np.int64))


class TemperatureTests(unittest.TestCase):
    def test_zero_oof_error_skips_fit_and_keeps_unit_temperature(self):
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
        temperature, status = calibrate_temperature(logits, torch.tensor([0, 1, 2]))
        self.assertEqual(temperature, 1.0)
        self.assertEqual(status, "skipped_no_errors")

    def test_temperature_is_never_sharpened_and_bound_is_reported(self):
        logits = torch.tensor([[3.0, 0.0]] * 20 + [[0.1, 0.0]])
        labels = torch.tensor([0] * 20 + [1])
        self.assertGreaterEqual(fit_temperature(logits, labels), 1.0 - 1e-6)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            temperature, status = calibrate_temperature(logits, labels)
        self.assertEqual(status, "at_bound")
        self.assertAlmostEqual(temperature, 1.0, places=2)
        self.assertIn("cảnh báo", out.getvalue())

    def test_overconfident_errors_fit_inside_bounds(self):
        logits = torch.tensor([[4.0, 0.0]] * 6 + [[0.0, 4.0]] * 6 + [[4.0, 0.0]] * 3 + [[0.0, 4.0]] * 3)
        labels = torch.tensor([0] * 6 + [1] * 6 + [1] * 3 + [0] * 3)
        temperature, status = calibrate_temperature(logits, labels)
        self.assertEqual(status, "fitted")
        self.assertGreater(temperature, 1.0)


class EpochBudgetTests(unittest.TestCase):
    def test_full_model_budget_extends_past_warmup(self):
        cfg = Config()
        self.assertEqual(full_model_epochs([3, 3, 4, 11, 34], cfg), cfg.warmup_epochs + 20)
        self.assertEqual(full_model_epochs([100, 100], cfg), int(100 * 1.15) + 1)
        self.assertGreater(full_model_epochs([1], cfg), cfg.warmup_epochs)

    def test_fixed_epoch_model_reaches_end_of_cosine(self):
        sequences, labels = tiny_data()
        cfg = tiny_config(warmup_epochs=2)
        _, info = train_one(sequences, labels, None, None, 3, default_groups(SEQ_FEATURE_DIM), cfg, torch.device("cpu"),
                            fixed_epochs=6, log_every=0)
        self.assertLess(info["final_lr"], 0.05 * cfg.lr)


class WarmStartTests(unittest.TestCase):
    def test_init_state_loads_weights_and_keeps_start_when_nothing_improves(self):
        sequences, labels = tiny_data()
        cfg = tiny_config(warmup_epochs=0, lr=1e-9, min_lr=1e-9, ema_decay=0.0)
        groups = default_groups(SEQ_FEATURE_DIM)
        base, _ = train_one(sequences, labels, None, None, 3, groups, cfg, torch.device("cpu"), fixed_epochs=2, log_every=0)
        start = {key: value.clone() for key, value in base.state_dict().items()}
        tuned, info = train_one(sequences, labels, sequences, labels, 3, groups, cfg, torch.device("cpu"), log_every=0,
                                fixed_epochs=2, init_state=start)
        self.assertGreaterEqual(info["best_epoch"], 0)
        if info["best_epoch"] == 0:
            for key, value in tuned.state_dict().items():
                torch.testing.assert_close(value, start[key])
        with torch.no_grad():
            base_acc = (base.eval()(sequences).argmax(1) == labels).float().mean().item()
        self.assertGreaterEqual(info["val_acc"], base_acc)


class ReplaySamplerTests(unittest.TestCase):
    def test_always_samples_every_epoch_and_quota_per_class(self):
        labels = np.repeat(np.arange(3), 10)
        always = np.zeros(30, dtype=bool)
        always[[0, 1, 25]] = True
        sampler = ReplaySampler(labels, quota=[4, 2, 20], always=always, seed=0)
        self.assertEqual(len(sampler), 3 + 4 + 2 + 9)
        seen = set()
        for epoch in range(5):
            idx = sampler(epoch).numpy()
            self.assertEqual(len(idx), len(sampler))
            self.assertEqual(len(set(idx)), len(idx))
            self.assertTrue({0, 1, 25} <= set(idx))
            self.assertEqual(np.bincount(labels[idx], minlength=3).tolist(), [6, 2, 10])
            seen |= set(idx)
        self.assertGreater(len(seen), len(sampler))

    def test_train_one_with_sampler_trains_and_keeps_start_candidate(self):
        sequences, labels = tiny_data(n=30)
        cfg = tiny_config(warmup_epochs=0, batch_size=0)
        groups = default_groups(SEQ_FEATURE_DIM)
        base, _ = train_one(sequences, labels, None, None, 3, groups, cfg, torch.device("cpu"), fixed_epochs=1, log_every=0)
        start = {key: value.clone() for key, value in base.state_dict().items()}
        calls = []
        inner = ReplaySampler(labels.numpy(), quota=[2, 6, 2], always=np.arange(30) == 0, seed=0)

        def sampler(epoch):
            calls.append(epoch)
            return inner(epoch)

        _, info = train_one(sequences, labels, sequences, labels, 3, groups, cfg, torch.device("cpu"), log_every=0, fixed_epochs=3,
                            init_state=start, sampler=sampler)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(len(info["history"]), 3)
        self.assertGreaterEqual(info["best_epoch"], 0)


class SoftPresenceTests(unittest.TestCase):
    def test_weak_soft_presence_does_not_count_as_second_hand(self):
        x = torch.zeros(1, 30, SEQ_FEATURE_DIM)
        presence = torch.zeros(1, 30, 2)
        presence[0, :, 1] = 1.0
        presence[0, :, 0] = 0.2
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


class DistractorTrainingTests(unittest.TestCase):
    def test_distractor_batches_train_in_one_fixed_size_forward(self):
        sequences, labels = tiny_data()
        presence = torch.zeros(len(sequences), 30, 2)
        presence[::2, :, 1] = 1.0
        presence[1::2] = 1.0
        shapes = []
        cfg = tiny_config(max_epochs=6, extra_hand_probability=1.0)
        torch.manual_seed(0)
        model = training.build_model(sequences.shape[-1], 3, default_groups(sequences.shape[-1]), cfg)
        hook = model.register_forward_pre_hook(lambda m, inp: shapes.append(inp[0].shape[0]) if m.training else None)
        with mock.patch.object(training, "build_model", return_value=model), \
                contextlib.redirect_stdout(io.StringIO()):
            _, info = train_one(sequences, labels, sequences[:6], labels[:6], 3, default_groups(sequences.shape[-1]), cfg, torch.device("cpu"),
                                hand_presence=presence, log_every=0)
        hook.remove()
        self.assertTrue(np.isfinite(info["val_loss"]))
        self.assertEqual(set(shapes), {8 + 8})


class EvaluationHookTests(unittest.TestCase):
    def _run(self, directory):
        rng = np.random.default_rng(0)
        labels = np.arange(18) % 3
        sequences = rng.normal(size=(18, 30, SEQ_FEATURE_DIM)).astype(np.float32) + labels[:, None, None]
        with contextlib.redirect_stdout(io.StringIO()):
            return training.run_training(sequences, labels, ["a", "b", "c"], np.arange(18), out_dir=directory,
                                         cfg=tiny_config(max_epochs=10), folds=3, add_full=False,
                                         export_onnx_model=False, device=torch.device("cpu"))

    def test_evaluation_report_uses_original_oof_logits(self):
        calls = []

        def oof_report(logits, labels, classes, groups=None):
            calls.append((logits.shape, labels.shape, list(classes), None if groups is None else groups.shape))
            return {"ok": True}

        fake = types.ModuleType("src.model.evaluation")
        fake.oof_report = oof_report
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"src.model.evaluation": fake}):
            result = self._run(directory)
        self.assertEqual(result["evaluation"], {"ok": True})
        self.assertEqual(calls, [((18, 3), (18,), ["a", "b", "c"], (18,))])
        self.assertIn(result["temperature_status"], {"fitted", "at_bound", "skipped_no_errors"})

    def test_fold_assignment_is_saved_for_finetuning(self):
        import json
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"src.model.evaluation": None}):
            self._run(directory)
            folds = json.loads((Path(directory) / "folds.json").read_text(encoding="utf-8"))
        self.assertEqual(folds["k"], 3)
        self.assertEqual(sorted(folds["fold_by_group"]), sorted(str(index) for index in range(18)))
        self.assertEqual(set(folds["fold_by_group"].values()), {0, 1, 2})

    def test_augmented_clips_without_hands_are_not_trained_on(self):
        from src.features.extract import FEATURE_SLICES
        rng = np.random.default_rng(0)
        labels = np.r_[np.arange(18) % 3, np.arange(6) % 3]
        sequences = rng.normal(size=(24, 30, SEQ_FEATURE_DIM)).astype(np.float32) + labels[:, None, None]
        sequences[[18, 19, 20], :, FEATURE_SLICES["hand_pos"]] = 0.0
        groups = np.r_[np.arange(18), np.arange(6)]
        mask = np.r_[np.ones(18, bool), np.zeros(6, bool)]
        sizes, real = [], training.train_one

        def spy(Xtr, ytr, *args, **kwargs):
            sizes.append(len(Xtr))
            return real(Xtr, ytr, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.dict(sys.modules, {"src.model.evaluation": None}), \
                mock.patch.object(training, "train_one", side_effect=spy):
            training.run_training(sequences, labels, ["a", "b", "c"], groups, out_dir=directory,
                                  cfg=tiny_config(max_epochs=6), folds=3, add_full=True,
                                  export_onnx_model=False, device=torch.device("cpu"), validation_mask=mask)
        self.assertIn("Bỏ 3 bản tăng cường", out.getvalue())
        self.assertEqual(sizes[-1], 21)
        self.assertEqual(sum(sizes[:-1]), 2 * 21)

    def test_training_runs_without_evaluation_module(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"src.model.evaluation": None}):
            result = self._run(directory)
        self.assertNotIn("evaluation", result)


if __name__ == "__main__":
    unittest.main()
