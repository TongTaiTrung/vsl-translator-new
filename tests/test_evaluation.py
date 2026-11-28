"""Tests for out-of-fold evaluation metrics."""

import unittest

import numpy as np

from src.model.evaluation import oof_report


class OofReportTests(unittest.TestCase):
    def test_known_values_on_tiny_synthetic_case(self):
        classes = ["a", "b", "c"]
        y = np.array([0, 0, 1, 1, 2, 2])
        logits = np.array([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 1.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ])

        report = oof_report(logits, y, classes)

        self.assertAlmostEqual(report["accuracy"], 5 / 6)
        self.assertEqual(report["labels"], classes)
        self.assertEqual(len(report["confusion_matrix"]), 3)
        self.assertEqual(report["confusion_matrix"][1], [1, 1, 0])
        self.assertEqual(report["confusion_matrix"][0], [2, 0, 0])
        self.assertEqual(report["confusion_matrix"][2], [0, 0, 2])

        a = next(p for p in report["per_class"] if p["class"] == "a")
        self.assertAlmostEqual(a["precision"], 2 / 3)
        self.assertAlmostEqual(a["recall"], 1.0)
        self.assertEqual(a["support"], 2)

        b = next(p for p in report["per_class"] if p["class"] == "b")
        self.assertAlmostEqual(b["recall"], 0.5)
        self.assertEqual(b["support"], 2)

        self.assertEqual(report["top_k"]["1"], report["accuracy"])
        self.assertAlmostEqual(report["top_k"]["3"], 1.0)
        self.assertNotIn("5", report["top_k"])

        self.assertEqual(len(report["hard_pairs"]), 3)
        pair_names = {frozenset((hp["class_a"], hp["class_b"])) for hp in report["hard_pairs"]}
        self.assertIn(frozenset(("a", "b")), pair_names)

        ci = report["bootstrap_ci_95"]
        self.assertTrue(0.0 <= ci["accuracy"][0] <= ci["accuracy"][1] <= 1.0)
        self.assertTrue(0.0 <= ci["macro_f1"][0] <= ci["macro_f1"][1] <= 1.0)

    def test_groups_argument_is_accepted_and_shape_checks_raise(self):
        classes = ["x", "y"]
        y = np.array([0, 1, 0, 1])
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        groups = np.array([0, 0, 1, 1])

        report = oof_report(logits, y, classes, groups=groups)
        self.assertEqual(report["accuracy"], 1.0)

        with self.assertRaises(ValueError):
            oof_report(logits, y, ["only_one_class"], groups=groups)
        with self.assertRaises(ValueError):
            oof_report(logits, y[:-1], classes, groups=groups)
        with self.assertRaises(ValueError):
            oof_report(logits, y, classes, groups=groups[:-1])


if __name__ == "__main__":
    unittest.main()
