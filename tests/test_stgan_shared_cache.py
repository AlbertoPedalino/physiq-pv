"""A shared cache: the normalised data are built once and read by every run with the same inputs.

The same file serves the ConvGRU and the GAT branch; graph cases run where the GAT exists.
"""
from contextlib import redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan import pipeline as pipeline_module
from physiq_pv.anomaly_detection.stgan.data import ContextArray
from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST, shared_cache, source_identity
from physiq_pv.experiments.stgan_wandb import default_config, execute_training
from scripts import run_era5_stgan
from scripts.run_stgan_wandb import parse_args
from test_stgan_mmd import HAS_GRAPH, NAMES, TEST, TIMES, TRAIN, VALIDATION, fields

USED = AssertionError("the data were normalised again")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "cache"

    def test_a_cache_is_built_once_for_each_set_of_inputs(self):
        calls = []

        def build(directory):
            calls.append(directory)
            (directory / "data.txt").write_text("values", encoding="utf-8")
            return {"minimum": [1.5]}

        first = shared_cache(self.root, {"data": "a"}, build)
        second = shared_cache(self.root, {"data": "a"}, build)
        other = shared_cache(self.root, {"data": "b"}, build)
        self.assertEqual((first[1:], second[1:], other[1:]),
                         (({"minimum": [1.5]}, False), ({"minimum": [1.5]}, True), ({"minimum": [1.5]}, False)))
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[0], other[0])
        self.assertEqual(len(calls), 2)
        # Built under another name and renamed: nothing but the two complete caches is left.
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), sorted((first[0].name, other[0].name)))
        self.assertNotEqual(calls[0], first[0])
        self.assertEqual(sorted(path.name for path in first[0].iterdir()), [MANIFEST, "data.txt"])
        self.assertEqual(json.loads((first[0] / MANIFEST).read_text(encoding="utf-8")),
                         {"inputs": {"data": "a"}, "built": {"minimum": [1.5]}})

    def test_an_interrupted_build_is_never_read(self):
        def stopped(directory):
            (directory / "train.npy").write_text("half", encoding="utf-8")
            raise RuntimeError("stopped")

        with self.assertRaisesRegex(RuntimeError, "stopped"):
            shared_cache(self.root, {"data": "a"}, stopped)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(shared_cache(self.root, {"data": "a"}, lambda directory: {"complete": True})[1:],
                         ({"complete": True}, False))

    def test_of_two_builds_at_the_same_time_the_second_keeps_the_first(self):
        def second(directory):
            # Another run completes the same cache while this one is still writing.
            shared_cache(self.root, {"data": "a"}, lambda other: {"by": "first"})
            return {"by": "second"}

        directory, built, reused = shared_cache(self.root, {"data": "a"}, second)
        self.assertEqual((built, reused), ({"by": "first"}, True))
        self.assertEqual([path.name for path in self.root.iterdir()], [directory.name])

    def test_the_identity_of_the_data_is_that_of_their_files(self):
        folder = Path(self.temp.name)
        np.save(folder / "a.npy", np.arange(48, dtype=np.float32).reshape(6, 4, 2))
        values = np.load(folder / "a.npy", mmap_mode="r")
        try:
            whole = source_identity(values)
            self.assertEqual(len(whole), 1)
            self.assertEqual((whole[0]["file"], whole[0]["shape"], whole[0]["dtype"], whole[0]["bytes"]),
                             (str((folder / "a.npy").resolve()), [6, 4, 2], "float32", (folder / "a.npy").stat().st_size))
            parts = source_identity(ContextArray(values[:2], values[2:]))
            self.assertEqual([part["shape"] for part in parts], [[2, 4, 2], [4, 4, 2]])
            self.assertEqual(parts[1]["offset"] - parts[0]["offset"], 2 * 4 * 2 * 4)
            self.assertEqual(source_identity(values), whole)
            # The same file written again is other data.
            status = (folder / "a.npy").stat()
            os.utime(folder / "a.npy", ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
            self.assertNotEqual(source_identity(values), whole)
            with self.assertRaisesRegex(ValueError, "read from files"):
                source_identity(np.zeros((6, 4, 2), dtype=np.float32))
        finally:
            values._mmap.close()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        values = fields()
        self.data = {}
        for name, part in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST)):
            np.save(self.root / f"source_{name}.npy", values[part])
            self.data[name] = np.load(self.root / f"source_{name}.npy", mmap_mode="r")
            self.addCleanup(self.data[name]._mmap.close)
        self.cache = self.root / "shared"

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4)

    def fit(self, name, *, train=None, test=None, **overrides):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.data["validation"], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=1, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            reconstruction_weights="daily_change", checkpoint_path=self.root / name / "model.pt",
            score_dir=self.root / name / "scores")
        options.update(overrides)
        stream = io.StringIO()
        with redirect_stdout(stream), fit_and_score_stgan(
                self.data["train"] if train is None else train, self.data["test"] if test is None else test,
                **options) as result:
            return SimpleNamespace(scores=np.array(result.test_scores), metadata=result.metadata, printed=stream.getvalue(),
                                   files={path.relative_to(self.root / name).parts[0] for path in (self.root / name).iterdir()})

    def caches(self):
        return sorted(path for path in self.cache.iterdir()) if self.cache.exists() else []

    def test_runs_with_a_shared_cache_are_the_runs_without_and_build_it_once(self):
        self.assertIsNone(inspect.signature(fit_and_score_stgan).parameters["shared_cache_dir"].default)
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                before = len(self.caches())
                own = self.fit(f"{encoder}_own", **extra)
                built = self.fit(f"{encoder}_built", shared_cache_dir=self.cache, **extra)
                if not before:
                    self.assertIn(f"[stgan] shared cache built in {self.cache}", built.printed)
                directory = self.caches()[0]
                self.assertEqual(len(self.caches()), 1)  # The encoder is not an input of the cache.
                self.assertEqual(sorted(path.name for path in directory.iterdir()),
                                 [MANIFEST, "test.npy", "train.npy", "validation.npy"])
                written = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
                # A later run reads the files and the min-max: no pass over the data, nothing rewritten.
                with patch.object(pipeline_module, "_feature_minmax", side_effect=USED), \
                        patch.object(pipeline_module, "normalized_memmap", side_effect=USED):
                    read = self.fit(f"{encoder}_read", shared_cache_dir=self.cache, **extra)
                self.assertIn(f"[stgan] shared cache read from {directory}", read.printed)
                self.assertEqual({path.name: path.stat().st_mtime_ns for path in directory.iterdir()}, written)
                np.testing.assert_array_equal(built.scores, own.scores)
                np.testing.assert_array_equal(read.scores, own.scores)
                self.assertIn("normalized_cache", own.files)
                self.assertNotIn("normalized_cache", built.files | read.files)
                for name in ("train", "validation", "test"):
                    np.testing.assert_array_equal(np.load(directory / f"{name}.npy"),
                                                  np.load(self.root / f"{encoder}_own" / "normalized_cache" / f"{name}.npy"))
                manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
                self.assertEqual((manifest["inputs"]["normalization"], manifest["inputs"]["context_steps"],
                                  manifest["inputs"]["holdout_name"], manifest["inputs"]["train_period"]),
                                 ("minmax", 2, "validation", [str(TIMES[TRAIN][0]), str(TIMES[TRAIN][-1]), 20]))
                self.assertEqual(manifest["inputs"]["train"][0]["file"], str((self.root / "source_train.npy").resolve()))

    def test_the_seasonal_normalization_has_its_own_cache_beside_the_other(self):
        seasonal = dict(normalization="seasonal", seasonal_window_days=183)  # Two days and a half of training.
        self.fit("minmax", shared_cache_dir=self.cache)
        own = self.fit("own", **seasonal)
        built = self.fit("built", shared_cache_dir=self.cache, **seasonal)
        self.assertEqual(len(self.caches()), 2)
        with patch.object(pipeline_module, "_feature_minmax", side_effect=USED), \
                patch.object(pipeline_module, "fit_seasonal_climatology", side_effect=USED), \
                patch.object(pipeline_module, "seasonal_memmap", side_effect=USED):
            read = self.fit("read", shared_cache_dir=self.cache, **seasonal)
        np.testing.assert_array_equal(built.scores, own.scores)
        np.testing.assert_array_equal(read.scores, own.scores)
        self.assertEqual(len(self.caches()), 2)
        details, expected = read.metadata["seasonal_normalization"], own.metadata["seasonal_normalization"]
        directory = Path(details["mean_file"]).parent
        self.assertIn(directory, self.caches())
        self.assertTrue(Path(details["mean_file"]).is_file() and Path(details["std_file"]).is_file())
        self.assertEqual({key: value for key, value in details.items() if not key.endswith("_file")},
                         {key: value for key, value in expected.items() if not key.endswith("_file")})
        self.assertEqual(built.metadata["seasonal_normalization"], details)
        # A window of other days is other data; here too short for the fixture, and the failed build leaves nothing.
        with self.assertRaisesRegex(ValueError, "Too few training samples"):
            self.fit("window", shared_cache_dir=self.cache, normalization="seasonal", seasonal_window_days=182)
        self.assertEqual(len(self.caches()), 2)

    def test_a_shared_cache_needs_files_and_normalised_data(self):
        with self.assertRaisesRegex(ValueError, "read from files"):
            self.fit("memory", train=np.array(self.data["train"]), shared_cache_dir=self.cache)
        with self.assertRaisesRegex(ValueError, "cache_normalized or normalization=seasonal"):
            self.fit("plain", shared_cache_dir=self.cache, cache_normalized=False)
        self.assertEqual(self.caches(), [])


class CommandTests(unittest.TestCase):
    def test_option_is_off_by_default_and_reaches_the_training(self):
        era5 = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main(era5)
            self.assertIsNone(train.call_args.kwargs["shared_cache_dir"])
            run_era5_stgan.main(era5 + ["--shared-cache-dir", "cache"])
        self.assertEqual(train.call_args.kwargs["shared_cache_dir"], Path("cache"))
        self.assertIsNone(inspect.signature(run_era5_stgan.run_training).parameters["shared_cache_dir"].default)
        # The W&B runner: every run of a sweep names the same directory.
        base = ["--backend", "era5", "--prepared-dir", "unused", "--output-root", "runs", "--device", "cpu",
                "--wandb-mode", "offline"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STGAN_SHARED_CACHE_DIR", None)
            self.assertIsNone(parse_args(base).shared_cache_dir)
            os.environ["STGAN_SHARED_CACHE_DIR"] = "from_environment"
            self.assertEqual(parse_args(base).shared_cache_dir, Path("from_environment"))
            self.assertEqual(parse_args(base + ["--shared-cache-dir", "cache"]).shared_cache_dir, Path("cache"))
            os.environ.pop("STGAN_SHARED_CACHE_DIR")
        for arguments, expected in ((base, None), (base + ["--shared-cache-dir", "cache"], Path("cache"))):
            with patch("scripts.run_era5_stgan.run_training", return_value={"backend": {}}) as train:
                execute_training(parse_args(arguments), default_config("era5"), 20, Path("out"), None)
            self.assertEqual(train.call_args.kwargs["shared_cache_dir"], expected)


if __name__ == "__main__":
    unittest.main()
