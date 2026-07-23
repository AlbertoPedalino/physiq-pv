import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.data.dataset import (
    FEATURE_NAMES,
    N_FEATURES,
    POA_INPUT_INDICES,
    PVDataset,
)
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.physics_loss import physics_loss_full
from train import _chronological_split, _day_weight


def _dataset() -> xr.Dataset:
    times = pd.date_range("2019-06-01", periods=96, freq="h").to_numpy()
    hour = np.arange(len(times), dtype=float) % 24
    daylight = np.maximum(0.0, np.sin((hour - 6.0) * np.pi / 12.0))
    direct = 650.0 * daylight
    diffuse = 120.0 * daylight
    poa = direct + diffuse
    energy = 0.8 * daylight
    return xr.Dataset(
        data_vars={
            "ENERGIA": (("plant", "time"), np.tile(energy, (2, 1))),
            "temperature_2m": (
                ("plant", "time"),
                np.tile(20.0 + 4.0 * daylight, (2, 1)),
            ),
            "wind_speed_10m": (
                ("plant", "time"),
                np.full((2, len(times)), 3.0),
            ),
            "solar_irradiance_poa": (
                ("plant", "time"),
                np.tile(poa, (2, 1)),
            ),
            "direct_irradiance_tilted": (
                ("plant", "time"),
                np.tile(direct, (2, 1)),
            ),
            "diffuse_irradiance_tilted": (
                ("plant", "time"),
                np.tile(diffuse, (2, 1)),
            ),
            "eta_base": ("plant", [0.8, 0.8]),
        },
        coords={
            "plant": ["plant-0", "plant-1"],
            "time": times,
            "lat": ("plant", [45.0, 45.1]),
            "lon": ("plant", [7.0, 7.1]),
        },
        attrs={
            "pvgis_tilt_angle": 30.0,
            "pvgis_azimuth_angle": 180.0,
        },
    )


def _quality(n_plants: int, n_steps: int) -> dict[str, np.ndarray]:
    return {
        key: np.full((n_plants, n_steps), 0.8)
        for key in ("m1", "m2", "m3", "m4", "m5")
    }


class PoaDatasetTest(unittest.TestCase):
    def test_ablation_keeps_schema_and_masks_only_poa_channels(self) -> None:
        ds = _dataset()
        fit_mask = np.arange(ds.sizes["time"]) < 72
        quality = _quality(ds.sizes["plant"], ds.sizes["time"])
        poa_on = PVDataset(
            ds,
            quality,
            fit_time_mask=fit_mask,
            include_poa_inputs=True,
        )
        poa_off = PVDataset(
            ds,
            quality,
            fit_time_mask=fit_mask,
            include_poa_inputs=False,
        )

        self.assertEqual(N_FEATURES, 17)
        self.assertEqual(poa_on.feats.shape[-1], len(FEATURE_NAMES))
        np.testing.assert_allclose(
            poa_off.feats[..., POA_INPUT_INDICES],
            0.0,
        )
        m3_index = FEATURE_NAMES.index("m3")
        np.testing.assert_allclose(
            poa_on.feats[..., m3_index],
            poa_off.feats[..., m3_index],
        )
        pv_observed_index = FEATURE_NAMES.index("pv_observed")
        np.testing.assert_allclose(
            poa_on.feats[..., pv_observed_index],
            poa_off.feats[..., pv_observed_index],
        )

    def test_validation_changes_do_not_change_fitted_statistics(self) -> None:
        original = _dataset()
        changed = original.copy(deep=True)
        changed["ENERGIA"].values[:, 72:] *= 100.0
        changed["temperature_2m"].values[:, 72:] += 100.0
        changed["solar_irradiance_poa"].values[:, 72:] *= 10.0
        fit_mask = np.arange(original.sizes["time"]) < 72
        quality = _quality(original.sizes["plant"], original.sizes["time"])

        first = PVDataset(original, quality, fit_time_mask=fit_mask)
        second = PVDataset(changed, quality, fit_time_mask=fit_mask)

        self.assertEqual(
            first.preprocessing_state["zscore"],
            second.preprocessing_state["zscore"],
        )
        np.testing.assert_allclose(first.pv_scale, second.pv_scale)
        np.testing.assert_allclose(first.poa_scale, second.poa_scale)
        np.testing.assert_allclose(first.pr_proxy, second.pr_proxy)

    def test_clear_sky_poa_is_zero_at_night(self) -> None:
        ds = _dataset()
        dataset = PVDataset(
            ds,
            _quality(ds.sizes["plant"], ds.sizes["time"]),
            fit_time_mask=np.arange(ds.sizes["time"]) < 72,
        )
        midnight = pd.DatetimeIndex(ds.time.values).hour == 0
        self.assertLess(float(dataset.poa_cs[midnight].max()), 1e-3)

    def test_missing_pv_target_is_returned_with_zero_validity(self) -> None:
        ds = _dataset()
        ds["ENERGIA"].values[0, 30] = np.nan
        dataset = PVDataset(
            ds,
            _quality(ds.sizes["plant"], ds.sizes["time"]),
            fit_time_mask=np.arange(ds.sizes["time"]) < 72,
        )

        sample = dataset[6]  # valid_starts[6] == 30
        self.assertEqual(len(sample), 8)
        pv_target_valid = sample[6]
        pv_lag_valid = sample[7]
        self.assertEqual(float(pv_target_valid[0]), 0.0)
        self.assertEqual(float(pv_lag_valid[0]), 1.0)

    def test_irregular_time_grid_is_rejected(self) -> None:
        ds = _dataset().isel(time=[i for i in range(96) if i != 10])
        with self.assertRaisesRegex(ValueError, "complete hourly grid"):
            PVDataset(
                ds,
                _quality(ds.sizes["plant"], ds.sizes["time"]),
                fit_time_mask=np.arange(ds.sizes["time"]) < 72,
            )


class PreprocessingTest(unittest.TestCase):
    def test_quality_capacity_fit_does_not_use_validation(self) -> None:
        original = _dataset()
        changed = original.copy(deep=True)
        changed["ENERGIA"].values[:, 72:] *= 100.0
        fit_mask = np.arange(original.sizes["time"]) < 72

        _, first = compute_qs(
            original,
            window=12,
            debug=True,
            fit_time_mask=fit_mask,
        )
        _, second = compute_qs(
            changed,
            window=12,
            debug=True,
            fit_time_mask=fit_mask,
        )

        np.testing.assert_allclose(
            first["capacity_scale"],
            second["capacity_scale"],
        )

    def test_naive_scada_time_is_converted_from_rome_to_utc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "2019_UPN_0110065_01.csv"
            pd.DataFrame(
                {
                    "date": ["15/01/19 12:00", "15/07/19 12:00"],
                    "ENERGIA": [1.0, 2.0],
                }
            ).to_csv(csv_path, index=False)

            dataset = load_sentinel_hourly(
                sentinel_dir=directory,
                year=2019,
                source_timezone="Europe/Rome",
            )

        times = pd.DatetimeIndex(dataset.time.values)
        self.assertEqual(times[0], pd.Timestamp("2019-01-15 11:00"))
        self.assertIn(pd.Timestamp("2019-07-15 10:00"), times)
        self.assertTrue(
            np.isnan(
                dataset["ENERGIA"].sel(
                    time=pd.Timestamp("2019-01-15 12:00")
                ).item()
            )
        )
        self.assertEqual(dataset.attrs["time_standard"], "UTC")


class PhysicsLossTest(unittest.TestCase):
    def test_normalized_physics_relation_has_zero_loss(self) -> None:
        pred_poa = torch.tensor([[0.5]])
        poa_scale = torch.tensor([[0.5]])
        pr_proxy = torch.tensor([[0.8]])
        pred_pv = torch.tensor([[0.8]])

        loss, parts = physics_loss_full(
            pred_poa,
            pred_pv,
            pred_poa,
            pred_pv,
            pr_proxy,
            poa_scale,
        )

        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertAlmostEqual(parts["l_physics"], 0.0, places=7)

    def test_missing_pv_target_does_not_contribute_to_pv_loss(self) -> None:
        pred_poa = torch.tensor([[0.5]])
        pred_pv = torch.tensor([[10.0]])
        loss, parts = physics_loss_full(
            pred_poa,
            pred_pv,
            pred_poa,
            torch.tensor([[0.0]]),
            torch.tensor([[0.8]]),
            torch.tensor([[0.5]]),
            pv_valid=torch.tensor([[0.0]]),
        )

        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertAlmostEqual(parts["l_pv"], 0.0, places=7)
        self.assertAlmostEqual(parts["l_physics"], 0.0, places=7)


class SplitAndGraphTest(unittest.TestCase):
    def test_day_weight_uses_clear_sky_not_observed_poa(self) -> None:
        weights = _day_weight(
            torch.tensor([[0.0, 0.2]]),
            night_loss_weight=0.2,
        )
        torch.testing.assert_close(weights, torch.tensor([[0.2, 1.0]]))

    def test_split_is_strictly_chronological(self) -> None:
        valid, train_indices, val_indices, fit_mask = _chronological_split(
            n_steps=100,
            seq_len=24,
            validation_fraction=0.2,
        )
        self.assertLess(valid[train_indices[-1]], valid[val_indices[0]])
        self.assertEqual(np.flatnonzero(fit_mask)[-1], valid[train_indices[-1]])
        self.assertEqual(valid[-1], 99)

    def test_graph_has_no_isolated_nodes_and_bounded_priors(self) -> None:
        edge_index, edge_weight = build_graph(
            np.array([45.0, 45.01, 46.0]),
            np.array([7.0, 7.01, 8.0]),
            max_dist_km=5.0,
        )
        degree = torch.bincount(edge_index[1], minlength=3)
        self.assertTrue(bool((degree > 0).all()))
        self.assertTrue(bool((edge_weight > 0).all()))
        self.assertTrue(bool((edge_weight <= 1).all()))
        self.assertEqual(
            int((edge_index[0] == edge_index[1]).sum()),
            3,
        )

    def test_single_node_graph_is_a_self_loop(self) -> None:
        edge_index, edge_weight = build_graph(
            np.array([45.0]),
            np.array([7.0]),
        )
        torch.testing.assert_close(edge_index, torch.tensor([[0], [0]]))
        torch.testing.assert_close(edge_weight, torch.tensor([1.0]))


class EntrypointTest(unittest.TestCase):
    def test_state_document_matches_feature_contract(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "BILSTM_GAT_STATE.md"
        )
        document = path.read_text(encoding="utf-8")
        match = re.search(
            r"L’ordine è un contratto persistente:\s*```text\s*(.*?)```",
            document,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        documented_features = tuple(
            line.strip().split(maxsplit=1)[1]
            for line in match.group(1).splitlines()
            if line.strip()
        )
        self.assertEqual(documented_features, FEATURE_NAMES)
        self.assertIn("batch passa da 5 a 8 elementi", document)
        self.assertIn("pv_target_valid, pv_lag_valid", document)
        self.assertIn("albedo=0", document)
        self.assertIn("poa_clear_sky > 0.05 kW/m²", document)

    def test_training_notebook_matches_current_api(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "run_training.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sources = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{path}:cell{index}", "exec")
            sources.append(source)
        combined = "\n".join(sources)
        self.assertNotIn("eta_max", combined)
        self.assertNotIn("kwp=", combined)
        self.assertIn("include_poa_inputs=INCLUDE_POA_INPUTS", combined)
        self.assertIn('selection_metric=CONFIG["selection_metric"]', combined)
        self.assertIn("import wandb", combined)
        self.assertIn('"wandb_project": "physiq_pv"', combined)
        self.assertIn('"method": "grid"', combined)
        self.assertIn('"seed": {"values": CONFIG["seeds"]}', combined)
        self.assertIn("wandb.sweep(", combined)
        self.assertIn("wandb.agent(", combined)
        self.assertIn('count=len(CONFIG["seeds"])', combined)
        self.assertNotIn('for seed in CONFIG["seeds"]', combined)


if __name__ == "__main__":
    unittest.main()
