"""BF16 numerical paths on CPU, plus native CUDA integration when available.

CPU autocast is a test surrogate only: production BF16 requires native CUDA.
"""
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection import stgan
from physiq_pv.anomaly_detection.stgan.precision import autocast_context, validate_precision
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from physiq_pv.anomaly_detection.stgan.scoring import score_components

HAS_GRAPH = hasattr(stgan, "STGANGAT")
NATIVE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)


@contextmanager
def cpu_bf16_surrogate():
    with ExitStack() as stack:
        for module in ("precision", "pipeline", "scoring"):
            stack.enter_context(patch(f"physiq_pv.anomaly_detection.stgan.{module}.validate_precision"))
        yield


def fixture(graph=False):
    from pyproj import Transformer
    x, y = np.meshgrid(400000. + np.arange(2)*5000, 5000000. - np.arange(2)*5000)
    lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
    grid = stgan.build_spatial_grid(lat, lon)
    values = np.random.default_rng(21).uniform(-1, 1, (12, 4, 2)).astype(np.float32)
    times = pd.date_range("2018-01-01", periods=len(values), freq="h")
    dataset_type = stgan.STGANGraphDataset if graph else stgan.STGANWindowDataset
    dataset = dataset_type(values, times, grid, feature_minimum=np.full(2, -1, np.float32),
        feature_scale=np.full(2, 2, np.float32), recent_steps=1, trend_steps=2, stride=1)
    options = dict(n_features=2, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1)
    if HAS_GRAPH:
        options.update(dropout_enabled=False)
    if graph:
        options.update(edge_index=stgan.grid_edge_index(grid.row_indices, grid.column_indices),
            node_indices=grid.node_indices, gat_hidden_dim=2, gat_heads=2,
            discriminator_chunk_size=3, trend_chunk_size=2)
    model = (stgan.STGANGAT if graph else stgan.STGAN)(**options)
    return dataset, model, values, times, lat, lon


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_supported_choices_cli_and_hardware_guard(self):
        from scripts.run_pvgis_stgan import parse_args
        self.assertEqual(stgan.STGANCNNConfig().precision, "fp32")
        self.assertEqual(parse_args(["--manifest", "unused", "--out-dir", "unused",
                                    "--precision", "bf16"]).precision, "bf16")
        if HAS_GRAPH:
            from scripts.run_era5_stgan import parse_args as era5_args
            self.assertEqual(era5_args(["train", "--prepared-dir", "unused", "--output-dir",
                                       "unused", "--precision", "bf16"]).precision, "bf16")
        for invalid in ("fp16", "auto", "invalid"):
            with self.assertRaises(ValueError):
                stgan.STGANCNNConfig(precision=invalid)
            with self.assertRaises(ValueError):
                validate_precision(invalid, "cpu")
        with self.assertRaisesRegex(ValueError, "native BF16"):
            validate_precision("bf16", "cpu")
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device", return_value=nullcontext()) as selected, \
             patch.object(torch.cuda, "is_bf16_supported", return_value=False) as supported:
            with self.assertRaisesRegex(ValueError, "native BF16"):
                validate_precision("bf16", "cuda:1")
            selected.assert_called_once_with(torch.device("cuda:1"))
            supported.assert_called_once_with(including_emulation=False)

    def test_fp32_overrides_enclosing_autocast(self):
        layer = torch.nn.Linear(2, 2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with autocast_context("fp32", "cpu"):
                self.assertEqual(layer(torch.ones(1, 2)).dtype, torch.float32)

    def test_logits_preserve_probability_api_and_bce_objective(self):
        dataset, model, *_ = fixture()
        recent, _, mask, _, observed = dataset.fetch_batch([0, 1])[:5]
        discriminator = model.discriminator
        history = discriminator.encode_history(recent, mask)
        logits = discriminator.score_current(history, observed, mask, return_logits=True)
        probability = discriminator.score_current(history, observed, mask)
        torch.testing.assert_close(probability, logits.sigmoid(), rtol=0, atol=0)
        for value in (0., 1.):
            target = torch.full_like(logits, value)
            a = torch.nn.functional.binary_cross_entropy(probability, target)
            b = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            torch.testing.assert_close(a, b)
            ga = torch.autograd.grad(a, tuple(discriminator.parameters()), retain_graph=True)
            gb = torch.autograd.grad(b, tuple(discriminator.parameters()), retain_graph=True)
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)

    def exercise_training_and_scores(self, device, graph):
        dataset, model, *_ = fixture(graph)
        model.to(device)
        batch = tuple(x.to(device) for x in dataset.fetch_batch([0, 1])[:5])
        go = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
        do = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
        dtypes = []
        hook = model.generator.output_projection[0].register_forward_hook(
            lambda module, args, output: dtypes.append(output.dtype))
        try:
            for share_history in (True, False):
                before_g = [p.detach().clone() for p in model.generator.parameters()]
                before_d = [p.detach().clone() for p in model.discriminator.parameters()]
                gl, dl = gan_train_step(model, batch, go, do, precision="bf16",
                    share_history=share_history, reuse_generator=share_history)
                self.assertEqual((gl.dtype, dl.dtype), (torch.float32, torch.float32))
                self.assertTrue(torch.isfinite(gl) and torch.isfinite(dl))
                for module, before in ((model.generator, before_g), (model.discriminator, before_d)):
                    self.assertTrue(any(not torch.equal(x, p) for x, p in zip(before, module.parameters())))
                    for parameter in module.parameters():
                        self.assertEqual(parameter.dtype, torch.float32)
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue(dtypes and all(dtype == torch.bfloat16 for dtype in dtypes))
        finally:
            hook.remove()
        # Score probabilities/residuals must stay finite FP32, including NumPy export.
        options = dict(batch_size=2, device=device, n_features=2, storage="memory")
        if HAS_GRAPH:
            options.update(mc_dropout_enabled=True, mc_samples=2)
        fp = score_components(model, dataset, precision="fp32", **options)
        bf = score_components(model, dataset, precision="bf16", **options)
        try:
            for reference, actual in zip(fp[:3], bf[:3]):
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(np.isfinite(actual).all())
                # Fixed tiny fixture: catches low precision score subtraction/export errors.
                np.testing.assert_allclose(actual, reference, rtol=.04, atol=.01)
        finally:
            fp[-1].close()
            bf[-1].close()

    def test_cpu_bf16_training_and_scoring(self):
        with cpu_bf16_surrogate():
            for graph in ((False, True) if HAS_GRAPH else (False,)):
                with self.subTest(graph=graph):
                    self.exercise_training_and_scores("cpu", graph)

    def exercise_pipeline(self, device, graph):
        _, _, values, times, lat, lon = fixture(graph)
        options = dict(train_timestamps=times[:8], test_timestamps=times[8:],
            location_names=tuple(map(str, range(4))), feature_names=("a", "b"),
            latitudes=lat, longitudes=lon, epochs=1, batch_size=2, score_batch_size=2,
            hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
            recent_steps=1, trend_steps=2, train_samples_per_epoch=2,
            device=device, precision="bf16", cache_normalized=False, score_storage="memory")
        if HAS_GRAPH:
            options.update(score_mode="paper", mc_samples=2, dropout_p=.1)
        if graph:
            options.update(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2,
                discriminator_chunk_size=3, trend_chunk_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            with stgan.fit_and_score_stgan(values[:8], values[8:], checkpoint_path=checkpoint,
                                          **options) as result:
                self.assertEqual(result.metadata["precision"], "bf16")
                self.assertEqual(result.metadata["performance"]["precision"], "bf16")
                self.assertTrue(np.isfinite(result.test_scores).all())
                if HAS_GRAPH:
                    self.assertTrue(np.isfinite(result.anomaly_std).all())
            for path in (checkpoint, checkpoint.with_name("model_epoch_1.pt")):
                restored, payload = stgan.load_stgan_checkpoint(path)
                self.assertEqual(payload["precision"], "bf16")
                self.assertTrue(all(p.dtype == torch.float32 for p in restored.parameters()))
            # An old checkpoint without the added metadata still loads as FP32.
            payload.pop("precision")
            torch.save(payload, checkpoint)
            _, legacy = stgan.load_stgan_checkpoint(checkpoint)
            self.assertEqual(legacy["precision"], "fp32")

    def test_cpu_bf16_pipeline_and_checkpoints(self):
        with cpu_bf16_surrogate():
            for graph in ((False, True) if HAS_GRAPH else (False,)):
                with self.subTest(graph=graph):
                    self.exercise_pipeline("cpu", graph)

    @unittest.skipUnless(NATIVE_BF16, "Requires native CUDA BF16 (run on the Blackwell server)")
    def test_native_cuda_bf16_training_scoring_and_pipeline(self):
        for graph in ((False, True) if HAS_GRAPH else (False,)):
            with self.subTest(graph=graph):
                self.exercise_training_and_scores("cuda", graph)
                self.exercise_pipeline("cuda", graph)


if __name__ == "__main__":
    unittest.main()
