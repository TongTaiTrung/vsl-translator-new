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

    def test_publish_points_best_at_new_run_and_keeps_five_newest(self):
        for day in range(10, 17):                                   # 7 run cũ, xen kẽ train / fine-tune
            make_run(self.root, f"vsl_{'ft_' if day % 2 else ''}202611{day}_120000")
        experiment = make_run(self.root / "experiments", "bench")
        new = make_run(self.root, "vsl_20261123_090000")

        removed = publish_run(new, runs_dir=self.root, keep=5)

        self.assertEqual(len(removed), 3)
        self.assertEqual([p.name for p in list_runs(self.root)],
                         ["vsl_20261123_090000", "vsl_20261116_120000", "vsl_ft_20261115_120000",
                          "vsl_20261114_120000", "vsl_ft_20261113_120000"])
        best = self.root / BEST_NAME
        self.assertTrue(best.is_symlink())
        self.assertEqual(best.resolve(), new)
        self.assertEqual(latest_checkpoint(self.root).resolve(), new / "ensemble.pt")
        self.assertTrue((experiment / "ensemble.pt").is_file())

    def test_republishing_moves_best_and_never_deletes_the_new_run(self):
        first = make_run(self.root, "vsl_20261123_090000")
        publish_run(first, runs_dir=self.root, keep=1)
        older_name = make_run(self.root, "vsl_20261101_000000")    # tên cũ hơn nhưng là run vừa lưu
        removed = publish_run(older_name, runs_dir=self.root, keep=1)
        self.assertEqual(removed, [first])
        self.assertEqual((self.root / BEST_NAME).resolve(), older_name)

    def test_run_outside_runs_dir_is_ignored(self):
        old = make_run(self.root, "vsl_20261101_000000")
        outside = make_run(self.root / "experiments", "vsl_20261123_090000")
        self.assertEqual(publish_run(outside, runs_dir=self.root, keep=1), [])
        self.assertFalse((self.root / BEST_NAME).exists())
        self.assertTrue(old.is_dir())

    def test_latest_checkpoint_without_best_uses_name_timestamp(self):
        make_run(self.root, "vsl_20261123_090000")
        newer_mtime = make_run(self.root, "vsl_20261101_000000")   # tạo sau => mtime mới hơn
        self.assertTrue(newer_mtime.is_dir())
        self.assertEqual(latest_checkpoint(self.root).parent.name, "vsl_20261123_090000")
        with self.assertRaises(FileNotFoundError):
            latest_checkpoint(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
