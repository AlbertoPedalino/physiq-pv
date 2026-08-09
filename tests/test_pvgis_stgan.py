from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import (
    STGAN,
    REFERENCE_CONFIG,
    build_geographical_subgraphs,
    fit_and_score_stgan,
    haversine_km,
    load_aligned_manifest_cubes,
    load_stgan_checkpoint,
    regular_target_indices,
)
from scripts.run_pvgis_stgan import CANONICAL_SCORE_COLUMNS, main as run_stgan


def test_reference_configuration_and_notebook_are_paper_aligned() -> None:
    assert REFERENCE_CONFIG.epochs == 6
    assert REFERENCE_CONFIG.batch_size == 256
    assert REFERENCE_CONFIG.hidden_size == 64
    assert REFERENCE_CONFIG.n_layers == 2
    assert REFERENCE_CONFIG.subgraph_size == 9
    assert REFERENCE_CONFIG.trend_steps == 7 * 24
    assert REFERENCE_CONFIG.generator_reconstruction_weight == 500.0
    assert REFERENCE_CONFIG.train_samples_per_epoch == 0

    notebook_path = Path("notebooks/stgan_pvgis_workflow.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{notebook_path}:cell-{index}", "exec")


def test_haversine_known_one_degree_distance() -> None:
    assert float(haversine_km(0.0, 0.0, 0.0, 0.0)) == 0.0
    assert np.isclose(float(haversine_km(0.0, 0.0, 1.0, 0.0)), 111.195, atol=0.01)


def test_geographical_subgraphs_are_local_and_target_first() -> None:
    lats = np.array([45.0, 45.1, 45.2, 46.0])
    lons = np.array([7.0, 7.0, 7.0, 7.0])
    graph = build_geographical_subgraphs(lats, lons, subgraph_size=3)
    assert graph.node_indices.shape == (4, 3)
    assert graph.normalized_adjacency.shape == (4, 3, 3)
    assert np.array_equal(graph.node_indices[:, 0], np.arange(4))
    pairwise = haversine_km(
        lats[:, None], lons[:, None], lats[None, :], lons[None, :]
    )
    expected_edge_distances = np.sort(pairwise, axis=1)[:, 1:3]
    assert np.isclose(graph.sigma_km, np.std(expected_edge_distances, ddof=1))
    assert np.isfinite(graph.normalized_adjacency).all()
    assert np.allclose(
        graph.normalized_adjacency,
        graph.normalized_adjacency.transpose(0, 2, 1),
    )
    # The distant fourth node is not one of location zero's two neighbours.
    assert 3 not in graph.node_indices[0]


def test_stgan_tensor_shapes_and_gradients() -> None:
    torch.manual_seed(7)
    model = STGAN(
        n_features=3,
        hidden_size=8,
        n_layers=1,
        subgraph_size=3,
    )
    recent = torch.randn(4, 2, 3, 3)
    trend = torch.randn(4, 5, 3)
    adjacency = torch.eye(3).repeat(4, 1, 1)
    time_features = torch.randn(4, 31)
    observed = torch.randn(4, 3, 3)
    predicted, real_score, fake_score, feature_error = model.components(
        recent, trend, adjacency, time_features, observed
    )
    assert predicted.shape == observed.shape
    assert real_score.shape == (4, 1)
    assert fake_score.shape == (4, 1)
    assert feature_error.shape == observed.shape
    loss = feature_error.mean() + real_score.mean() + fake_score.mean()
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_regular_targets_do_not_cross_timestamp_gap() -> None:
    first = pd.date_range("2018-01-01", periods=6, freq="h")
    second = pd.date_range("2018-01-02", periods=6, freq="h")
    times = first.append(second)
    targets = regular_target_indices(times, context_steps=3, stride=1)
    assert 6 not in targets
    assert 7 not in targets
    assert 8 not in targets
    assert 9 in targets


def _frame(start: str, periods: int, offset: float) -> pd.DataFrame:
    index = np.arange(periods, dtype=np.float32)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq="h"),
            "solar_irradiance_poa": 400.0 + offset + 20.0 * np.sin(index / 3.0),
            "temperature_2m": 15.0 + offset * 0.1 + np.cos(index / 4.0),
            "wind_speed_10m": 2.0 + 0.1 * index,
            "is_daytime": True,
        }
    )


def test_manifest_cube_and_tiny_stgan_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        for location_index, (lat, lon) in enumerate(((45.0, 7.0), (45.1, 7.1), (45.2, 7.2))):
            site = root / f"site_{location_index}"
            site.mkdir()
            train_path = site / "train.csv"
            test_path = site / "test.csv"
            _frame("2016-01-01", 16, float(location_index)).to_csv(train_path, index=False)
            _frame("2019-01-01", 12, float(location_index)).to_csv(test_path, index=False)
            rows.append(
                {
                    "location": f"loc-{location_index}",
                    "site_key": f"site_{location_index}",
                    "train_csv": str(train_path),
                    "test_csv": str(test_path),
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        manifest = pd.DataFrame(rows)
        manifest_path = root / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        cubes = load_aligned_manifest_cubes(manifest, cache_dir=root / "cache")
        assert cubes.train.shape == (16, 3, 3)
        assert cubes.test.shape == (12, 3, 3)
        checkpoint = root / "checkpoint.pt"
        result = fit_and_score_stgan(
            cubes.train,
            cubes.test,
            train_timestamps=cubes.train_timestamps,
            test_timestamps=cubes.test_timestamps,
            location_names=cubes.location_names,
            feature_names=cubes.feature_names,
            latitudes=cubes.latitudes,
            longitudes=cubes.longitudes,
            epochs=1,
            batch_size=4,
            hidden_size=8,
            n_layers=1,
            subgraph_size=3,
            recent_steps=1,
            trend_steps=4,
            train_score_stride=2,
            score_stride=1,
            train_samples_per_epoch=8,
            device="cpu",
            checkpoint_path=checkpoint,
        )
        assert result.train_scores.shape == (6, 3)
        assert result.test_scores.shape == (8, 3)
        assert result.test_feature_scores.shape == (8, 3, 3)
        assert np.isfinite(result.test_scores).all()
        restored, payload = load_stgan_checkpoint(checkpoint)
        assert isinstance(restored, STGAN)
        assert payload["training"]["test_labels_used"] is False
        assert payload["training"]["discriminator_targets"] == {
            "real_normal": 0,
            "generated_fake": 1,
        }
        assert payload["graph"]["node_indices"].shape == (3, 3)
        del restored, payload, result, cubes
        gc.collect()

        output = root / "runner_output"
        run_stgan(
            [
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(output),
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--hidden-size",
                "8",
                "--n-layers",
                "1",
                "--subgraph-size",
                "3",
                "--recent-steps",
                "1",
                "--trend-steps",
                "4",
                "--train-score-stride",
                "2",
                "--score-stride",
                "1",
                "--train-samples-per-epoch",
                "8",
                "--device",
                "cpu",
            ]
        )
        canonical = pd.read_csv(output / "seed_20" / "anomaly_scores.csv")
        assert list(canonical.columns) == CANONICAL_SCORE_COLUMNS
        assert canonical["method"].eq("stgan").all()
        assert len(canonical) == 8 * 3
        locations = pd.read_csv(output / "locations.csv")
        assert list(locations.columns) == [
            "location",
            "site_key",
            "latitude",
            "longitude",
        ]
        assert len(locations) == 3
        assert (output / "seed_20" / "entity_anomaly_scores.csv").is_file()
        assert (output / "seed_20" / "checkpoint.pt").is_file()
        gc.collect()


if __name__ == "__main__":
    test_reference_configuration_and_notebook_are_paper_aligned()
    test_haversine_known_one_degree_distance()
    test_geographical_subgraphs_are_local_and_target_first()
    test_stgan_tensor_shapes_and_gradients()
    test_regular_targets_do_not_cross_timestamp_gap()
    test_manifest_cube_and_tiny_stgan_smoke()
