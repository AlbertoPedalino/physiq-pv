"""Deletion tests use synthetic files in temporary directories only."""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cleanup_intermediate_checkpoints import apply_group, main, scan


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write(self, relative, content=b"synthetic checkpoint", recent=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        when = time.time() if recent else time.time() - 48 * 3600
        os.utime(path, (when, when))
        return path

    def cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["--root", str(self.root), *args])
        return code, output.getvalue()

    def test_preview_does_not_modify_any_files(self):
        self.write("nested/run/checkpoint.pt")
        self.write("nested/run/checkpoint_epoch_1.pt")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        code, output = self.cli()
        self.assertEqual(code, 0)
        self.assertIn("Candidati: 1 file", output)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_apply_preserves_models_images_results_and_unknowns(self):
        retained = [
            "run/checkpoint.pt", "run/best_model.pt", "run/last.pt",
            "run/figures/image.png", "run/figure.pdf", "run/predictions.csv",
            "run/scores.npy", "run/config.json", "run/weights_123.pt",
            "run/best_model_epoch_1.pt", "run/checkpoint_epoch_1.pt.backup",
        ]
        for name in retained:
            self.write(name)
        intermediates = [self.write(f"run/checkpoint_epoch_{i}.pt") for i in range(3)]
        self.assertEqual(self.cli("--apply")[0], 0)
        self.assertTrue(all((self.root / name).exists() for name in retained))
        self.assertTrue(all(not path.exists() for path in intermediates))
        self.assertIn("Candidati: 0 file", self.cli("--apply")[1])

    def test_missing_empty_or_different_extension_final_preserves_intermediate(self):
        missing = self.write("missing/checkpoint_epoch_1.pt")
        empty = self.write("empty/checkpoint_epoch_1.pt")
        self.write("empty/checkpoint.pt", b"")
        different = self.write("different/epoch_1.ckpt")
        self.write("different/best_model.pt")
        # A model in a different directory cannot protect this run.
        self.write("best_model.pt")
        self.cli("--apply")
        self.assertTrue(all(p.exists() for p in (missing, empty, different)))

    def test_recent_checkpoint_skips_whole_directory(self):
        self.write("run/checkpoint.pt")
        old = self.write("run/checkpoint_epoch_1.pt")
        new = self.write("run/checkpoint_epoch_2.pt", recent=True)
        self.cli("--apply")
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_common_epoch_names_and_best_from_earlier_epoch(self):
        self.write("predictor/best_model.pth")
        p = self.write("predictor/model_epoch_20.pth")
        self.write("detector/best.ckpt")
        q = self.write("detector/epoch=3-step=250.ckpt")
        self.cli("--apply")
        self.assertFalse(p.exists())
        self.assertFalse(q.exists())

    def test_changed_or_removed_final_aborts_group(self):
        for change in ("remove", "replace"):
            with self.subTest(change=change):
                final = self.write(f"{change}/checkpoint.pt")
                intermediate = self.write(f"{change}/checkpoint_epoch_1.pt")
                root, groups, *_ = scan(self.root, 24)
                group = next(g for g in groups if g["directory"] == final.parent)
                if change == "remove":
                    final.unlink()
                else:
                    final.write_bytes(b"changed")
                count, _, errors = apply_group(root, group)
                self.assertEqual(count, 0)
                self.assertTrue(errors)
                self.assertTrue(intermediate.exists())

    def test_new_checkpoint_after_scan_aborts_group(self):
        self.write("run/checkpoint.pt")
        old = self.write("run/checkpoint_epoch_1.pt")
        root, groups, *_ = scan(self.root, 24)
        self.write("run/checkpoint_epoch_2.pt", recent=True)
        count, _, errors = apply_group(root, groups[0])
        self.assertEqual(count, 0)
        self.assertTrue(errors)
        self.assertTrue(old.exists())

    def test_symlink_final_cannot_authorize_deletion(self):
        final = self.write("outside/checkpoint.pt")
        intermediate = self.write("run/checkpoint_epoch_1.pt")
        try:
            (intermediate.parent / "checkpoint.pt").symlink_to(final)
        except OSError:
            self.skipTest("Symlinks not permitted on this platform")
        self.cli("--apply")
        self.assertTrue(intermediate.exists())
        self.assertTrue(final.exists())

    def test_outside_directory_rejected(self):
        root, groups, *_ = scan(self.root, 24)
        group = {"directory": self.root.parent, "snapshot": {}, "candidates": []}
        count, _, errors = apply_group(root, group)
        self.assertEqual(count, 0)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
