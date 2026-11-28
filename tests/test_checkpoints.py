import tempfile
import unittest
from pathlib import Path

from src.model.checkpoints import BEST_NAME, latest_checkpoint, list_runs, publish_run


def make_run(root: Path, name: str) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "ensemble.pt").write_bytes(b"x")
    return run


class CheckpointRotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_makes_best_a_real_dir_and_keeps_five_newest(self):
        for day in range(10, 17):
            make_run(self.root, f"202611{day}_120000")
        experiment = make_run(self.root / "experiments", "bench")
        new = make_run(self.root, "20261123_090000")

        best = publish_run(new, runs_dir=self.root, keep=5)

        best_dir = self.root / BEST_NAME
        self.assertEqual(best, best_dir)
        self.assertTrue(best_dir.is_dir() and not best_dir.is_symlink())
        self.assertFalse(new.exists())
        self.assertEqual([p.name for p in list_runs(self.root)],
                         ["20261116_120000", "20261115_120000", "20261114_120000", "20261113_120000"])
        self.assertEqual(latest_checkpoint(self.root), best_dir / "ensemble.pt")
        self.assertTrue((experiment / "ensemble.pt").is_file())

    def test_republishing_archives_old_best_under_its_own_name(self):
        publish_run(make_run(self.root, "20261123_090000"), runs_dir=self.root, keep=5)
        publish_run(make_run(self.root, "20261124_090000"), runs_dir=self.root, keep=5)
        self.assertEqual([p.name for p in list_runs(self.root)], ["20261123_090000"])
        self.assertEqual((self.root / BEST_NAME / ".run_name").read_text(), "20261124_090000")
        publish_run(make_run(self.root, "20261125_090000"), runs_dir=self.root, keep=1)
        self.assertEqual(list_runs(self.root), [])

    def test_legacy_symlink_best_is_replaced(self):
        old = make_run(self.root, "vsl_20261123_090000")
        (self.root / BEST_NAME).symlink_to(old.name, target_is_directory=True)
        publish_run(make_run(self.root, "20261124_090000"), runs_dir=self.root, keep=5)
        self.assertFalse((self.root / BEST_NAME).is_symlink())
        self.assertTrue(old.is_dir())

    def test_run_outside_runs_dir_is_ignored(self):
        old = make_run(self.root, "vsl_20261101_000000")
        outside = make_run(self.root / "experiments", "vsl_20261123_090000")
        self.assertIsNone(publish_run(outside, runs_dir=self.root, keep=1))
        self.assertFalse((self.root / BEST_NAME).exists())
        self.assertTrue(old.is_dir())

    def test_latest_checkpoint_without_best_uses_name_timestamp(self):
        make_run(self.root, "vsl_20261123_090000")
        newer_mtime = make_run(self.root, "vsl_20261101_000000")
        self.assertTrue(newer_mtime.is_dir())
        self.assertEqual(latest_checkpoint(self.root).parent.name, "vsl_20261123_090000")
        with self.assertRaises(FileNotFoundError):
            latest_checkpoint(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
