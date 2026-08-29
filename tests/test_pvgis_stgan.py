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
    STGANReferenceConfig,
    build_geographical_subgraphs,
    fit_and_score_stgan,
    haversine_km,
    load_aligned_manifest_cubes,
    load_stgan_checkpoint,
    prepend_training_context_to_test,
    regular_target_indices,
)
from scripts.run_pvgis_stgan import (
    CANONICAL_SCORE_COLUMNS,
    paper_top_k_ranking,
    run_stgan,
)


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
    notebook_source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert "expected_cadence = pd.Timedelta(hours=1)" in notebook_source
    assert "FORECAST_HORIZONS = tuple(range(1, 7))" in notebook_source
    assert "subprocess" not in notebook_source


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
    assert graph.topology == "directed_geographical_knn"
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


def test_test_split_uses_contiguous_training_tail_as_context() -> None:
    train = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    test = np.arange(6, 9, dtype=np.float32).reshape(3, 1, 1)
    train_times = pd.date_range("2018-12-31 18:00", periods=6, freq="h")
    test_times = pd.date_range("2019-01-01", periods=3, freq="h")
    combined, times = prepend_training_context_to_test(
        train,
        test,
        train_timestamps=train_times,
        test_timestamps=test_times,
        context_steps=2,
    )
    assert combined[:, 0, 0].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert times[2] == test_times[0]

    with np.testing.assert_raises_regex(ValueError, "contiguous regular history"):
        prepend_training_context_to_test(
            train,
            test,
            train_timestamps=train_times,
            test_timestamps=pd.date_range("2019-01-02", periods=3, freq="h"),
            context_steps=2,
        )


def test_paper_top_k_is_global_exact_and_deterministic() -> None:
    scores = np.array([[1.0, 5.0, 3.0], [2.0, 6.0, 4.0]])
    flags, ranks, percentiles = paper_top_k_ranking(scores, 50.0)
    assert flags.tolist() == [[False, True, False], [False, True, True]]
    assert ranks.tolist() == [[6.0, 2.0, 4.0], [5.0, 1.0, 3.0]]
    assert percentiles[1, 1] == 100.0
    assert np.isclose(percentiles[0, 0], 100.0 / 6.0)


def test_notebook_multihorizon_sweep_uses_target_time_and_global_rank() -> None:
    notebook_path = Path("notebooks/stgan_pvgis_workflow.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def multihorizon_top_k_metrics" in "".join(cell["source"])
    )
    namespace = {
        "ROOT": Path("."),
        "scores": None,
        "np": np,
        "pd": pd,
        "K_PERCENTAGES": np.array([50.0]),
    }
    exec(compile(source, f"{notebook_path}:multihorizon", "exec"), namespace)

    timestamps = pd.date_range("2019-01-01", periods=4, freq="h")
    scores = pd.DataFrame(
        {
            "location": ["0"] * 4,
            "timestamp": timestamps,
            "global_rank": [1.0, 2.0, 3.0, 4.0],
        }
    )
    horizons = tuple(range(1, 7))
    predictions = pd.DataFrame(
        {
            "location": [0] * (4 * len(horizons)),
            "timestamp": np.tile(timestamps, len(horizons)),
            "horizon_hours": np.repeat(horizons, 4),
            "y_true": 1.0,
            "y_pred_mean": np.tile([3.0, 3.0, 1.0, 1.0], len(horizons)),
            "solar_irradiance_poa_target": 20.0,
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.csv"
        predictions.to_csv(path, index=False)
        metrics, coverage = namespace["multihorizon_top_k_metrics"](
            path, scores, np.array([50.0]), horizons
        )

    assert coverage == 1.0
    assert set(metrics["horizon_hours"]) == set(horizons)
    anomalous = metrics.loc[metrics["group"].eq("anomaly")]
    normal = metrics.loc[metrics["group"].eq("normal")]
    assert anomalous["n"].eq(2).all()
    assert anomalous["mae"].eq(2.0).all()
    assert normal["n"].eq(2).all()
    assert normal["mae"].eq(0.0).all()


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
            _frame("2018-12-31 08:00", 16, float(location_index)).to_csv(
                train_path, index=False
            )
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
            score_stride=1,
            train_samples_per_epoch=8,
            device="cpu",
            checkpoint_path=checkpoint,
        )
        assert result.test_scores.shape == (12, 3)
        assert result.test_feature_scores.shape == (12, 3, 3)
        assert result.test_timestamps[0] == pd.Timestamp("2019-01-01 00:00")
        assert np.isfinite(result.test_scores).all()
        assert checkpoint.with_name("checkpoint_epoch_1.pt").is_file()
        generator = result.test_generator_scores
        discriminator = result.test_discriminator_scores
        expected_scores = (generator - generator.min()) / max(
            float(generator.max() - generator.min()), 1e-8
        ) + (discriminator - discriminator.min()) / max(
            float(discriminator.max() - discriminator.min()), 1e-8
        )
        assert np.allclose(result.test_scores, expected_scores)
        restored, payload = load_stgan_checkpoint(checkpoint)
        assert isinstance(restored, STGAN)
        assert payload["training"]["test_labels_used"] is False
        assert payload["training"]["test_context"] == {
            "source": "training_tail_only",
            "steps": 4,
            "targets": "test_timestamps_only",
        }
        assert payload["training"]["discriminator_targets"] == {
            "real_normal": 0,
            "generated_fake": 1,
        }
        assert payload["graph"]["node_indices"].shape == (3, 3)
        assert payload["graph"]["kind"] == "directed_geographical_knn"
        assert payload["score_normalization"]["fit_period"] == (
            "complete_test_time_location_product"
        )
        del restored, payload, result, cubes
        gc.collect()

        output = root / "runner_output"
        run_stgan(
            manifest_path=manifest_path,
            out_dir=output,
            paper_top_k_percent=25.0,
            config=STGANReferenceConfig(
                epochs=1,
                batch_size=4,
                hidden_size=8,
                n_layers=1,
                subgraph_size=3,
                recent_steps=1,
                trend_steps=4,
                score_stride=1,
                train_samples_per_epoch=8,
            ),
            device="cpu",
        )
        canonical = pd.read_csv(output / "seed_20" / "anomaly_scores.csv")
        assert list(canonical.columns) == CANONICAL_SCORE_COLUMNS
        assert canonical["method"].eq("stgan").all()
        assert len(canonical) == 12 * 3
        assert int(canonical["is_anomaly"].sum()) == 9
        assert canonical["threshold"].isna().all()
        assert sorted(canonical["global_rank"].astype(int)) == list(range(1, 37))
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
        assert not (output / "seed_20" / "train_anomaly_scores.csv").exists()
        gc.collect()


if __name__ == "__main__":
    test_reference_configuration_and_notebook_are_paper_aligned()
    test_haversine_known_one_degree_distance()
    test_geographical_subgraphs_are_local_and_target_first()
    test_stgan_tensor_shapes_and_gradients()
    test_regular_targets_do_not_cross_timestamp_gap()
    test_test_split_uses_contiguous_training_tail_as_context()
    test_paper_top_k_is_global_exact_and_deterministic()
    test_notebook_multihorizon_sweep_uses_target_time_and_global_rank()
    test_manifest_cube_and_tiny_stgan_smoke()
