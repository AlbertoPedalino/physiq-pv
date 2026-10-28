"""Releasing the file pages read by the loading processes must leave every batch exactly as it was."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import STGANFullGridDataset, build_spatial_grid
from physiq_pv.anomaly_detection.stgan import loading
from physiq_pv.anomaly_detection.stgan.data import ContextArray

HEIGHT, WIDTH, FEATURES, STEPS = 6, 8, 3, 120
ROWS, COLS = np.indices((HEIGHT, WIDTH)).reshape(2, -1)
LAT, LON = 45 - ROWS * .5, 7 + COLS * .5
STAMPS = pd.date_range("2001-01-01", periods=STEPS, freq="3h")


def file_pages_of_this_process():
    """Kilobytes of file pages mapped by this process (Linux)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssFile:"):
            return int(line.split()[1])
    raise RuntimeError("no RssFile in /proc/self/status")


class ReleasePagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        values = np.random.default_rng(0).normal(size=(STEPS, HEIGHT * WIDTH, FEATURES)).astype(np.float32)
        np.save(Path(self.temp.name) / "data.npy", values)
        np.save(Path(self.temp.name) / "inputs.npy", values[::-1].copy())

    def dataset(self):
        """A full-grid dataset over two read-only maps of files, as the runs open the shared cache."""
        data = np.load(Path(self.temp.name) / "data.npy", mmap_mode="r")
        inputs = np.load(Path(self.temp.name) / "inputs.npy", mmap_mode="r")
        self.addCleanup(lambda: [array._mmap.close() for array in (data, inputs)])
        grid = build_spatial_grid(LAT, LON, grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
        return STGANFullGridDataset(data, STAMPS, grid, input_data=inputs, feature_minimum=np.full(FEATURES, -4.0),
                                    feature_scale=np.full(FEATURES, 8.0), recent_steps=4, trend_steps=8, stride=1,
                                    time_encoding="cyclic", longitudes=LON)

    def batches(self, *, release_every, num_workers=0):
        torch.manual_seed(0)
        with patch.object(loading, "RELEASE_EVERY", release_every):
            loader = loading.make_loader(self.dataset(), batch_size=5, shuffle=True, num_workers=num_workers,
                                         generator=torch.Generator().manual_seed(0))
            try:
                return [tuple(value.clone() for value in batch) for batch in loader]
            finally:
                loading.close_loader(loader)

    def assert_same(self, first, second):
        self.assertEqual(len(first), len(second))
        for one, other in zip(first, second):
            self.assertEqual(len(one), len(other))
            for a, b in zip(one, other):
                self.assertTrue(torch.equal(a, b))

    def test_batches_are_identical_bit_for_bit(self):
        # Released after every batch, or never: the same batches in the same order.
        self.assert_same(self.batches(release_every=1), self.batches(release_every=10 ** 9))

    def test_batches_are_identical_with_loading_processes(self):
        self.assert_same(self.batches(release_every=1, num_workers=2), self.batches(release_every=10 ** 9))

    def test_releases_the_read_only_maps_every_few_batches(self):
        dataset = self.dataset()
        calls = []
        with patch.object(loading, "release_pages", side_effect=lambda *arrays: calls.append(arrays)), \
                patch.object(loading, "RELEASE_EVERY", 3):
            windows = loading.BatchedWindows(dataset)
            for index in range(7):
                windows[[index]]
        self.assertEqual(len(calls), 2)  # after the 3rd and the 6th batch
        self.assertIs(calls[0][0], dataset.data)
        self.assertIs(calls[0][1], dataset.input_data)

    def test_release_pages_advises_only_read_only_maps(self):
        fake = SimpleNamespace(MADV_DONTNEED=4)
        read_only = np.load(Path(self.temp.name) / "data.npy", mmap_mode="r")
        writable = np.load(Path(self.temp.name) / "inputs.npy", mmap_mode="c")
        self.addCleanup(lambda: [array._mmap.close() for array in (read_only, writable)])
        mappings = {}
        for name, array in (("read_only", read_only), ("writable", writable)):
            mappings[name] = MagicMock(closed=False)
            array._mmap = mappings[name]
        with patch.object(loading, "mmap", fake):
            # A slice of a map, a virtual concatenation of maps, an array in memory and None.
            loading.release_pages(read_only[2:5], ContextArray(read_only[:3], read_only[3:]), writable, np.ones(3), None)
        self.assertEqual(mappings["read_only"].madvise.call_count, 3)
        mappings["read_only"].madvise.assert_called_with(4)
        mappings["writable"].madvise.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith("linux"), "RssFile and madvise are read on Linux")
    def test_the_file_pages_of_the_process_go_down(self):
        big = Path(self.temp.name) / "big.npy"
        np.save(big, np.ones((64, 1024, 1024), dtype=np.float32))  # 256 MB
        array = np.load(big, mmap_mode="r")
        self.addCleanup(array._mmap.close)
        before = file_pages_of_this_process()
        total = float(array.sum(dtype=np.float64))  # every page is read
        read = file_pages_of_this_process()
        loading.release_pages(array)
        after = file_pages_of_this_process()
        self.assertEqual(total, 64 * 1024 * 1024)
        self.assertGreater(read - before, 200 * 1024)
        self.assertLess(after - before, 20 * 1024)
        # Read again from the page cache: the same values.
        self.assertEqual(float(array.sum(dtype=np.float64)), total)


if __name__ == "__main__":
    unittest.main()
