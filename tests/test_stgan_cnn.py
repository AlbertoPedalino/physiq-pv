"""Geometry, missing-cell semantics, checkpoint and CLI integration tests."""
from __future__ import annotations

import gc
import json
import tempfile
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer

from physiq_pv.anomaly_detection.stgan import (
    ConvGRU, ConvGRUCell, STGAN, STGANCNNConfig, STGANWindowDataset, build_spatial_grid,
    load_stgan_checkpoint, masked_cell_mean,
)
from scripts.run_pvgis_stgan import run_stgan, paper_top_k_ranking


def coordinates(size=3):
    x, y = np.meshgrid(400000.0 + np.arange(size)*5000, 5000000.0 - np.arange(size)*5000)
    lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
    return lat, lon


def test_grid_orientation_boundaries_and_permutation():
    lat, lon = coordinates()
    grid = build_spatial_grid(lat, lon)
    assert grid.metadata["crs"] == "EPSG:32632"
    assert grid.metadata["n_complete_patches"] == 1
    assert grid.metadata["complete_patches_matching_knn"] == 1
    assert grid.node_indices[4].tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert grid.node_indices[0].tolist() == [[-1, -1, -1], [-1, 0, 1], [-1, 3, 4]]
    assert grid.valid_mask.sum((1, 2)).tolist() == [4, 6, 4, 6, 9, 6, 4, 6, 4]
    perm = np.array([7, 2, 4, 1, 6, 3, 0, 8, 5])
    shuffled = build_spatial_grid(lat[perm], lon[perm])
    center = np.flatnonzero(perm == 4)[0]
    assert np.array_equal(perm[shuffled.node_indices[center]], grid.node_indices[4])
    for size in (1, 5):
        alternate = build_spatial_grid(lat, lon, patch_size=size)
        assert alternate.node_indices.shape == (9, size, size)
        assert np.array_equal(alternate.node_indices[:, size//2, size//2], np.arange(9))
    with np.testing.assert_raises(ValueError):
        build_spatial_grid(np.append(lat, lat[0]), np.append(lon, lon[0]))
    lon_bad = lon.copy()
    lon_bad[4] += .005
    with np.testing.assert_raises_regex(ValueError, "No verified regular grid"):
        build_spatial_grid(lat, lon_bad)
    angular_lat, angular_lon = np.meshgrid([45., 45.05, 45.10], [7., 7.05, 7.10])
    angular = build_spatial_grid(angular_lat.ravel(), angular_lon.ravel())
    assert angular.metadata["spacing_unit"] == "degrees"


def test_dataset_context_and_zero_after_normalization():
    lat, lon = coordinates()
    grid = build_spatial_grid(lat, lon)
    values = np.arange(8*9*3, dtype=np.float32).reshape(8, 9, 3)
    times = pd.date_range("2018-01-01 00:10", periods=8, freq="h")
    ds = STGANWindowDataset(values, times, grid, feature_minimum=np.ones(3)*10,
                           feature_scale=np.ones(3)*20, recent_steps=1, trend_steps=4, stride=1)
    recent, trend, mask, calendar, observed, time_pos, location = ds[0]
    assert recent.shape == (1, 3, 3, 3) and observed.shape == (3, 3, 3)
    assert torch.equal(recent[0, :, 0, 0], torch.zeros(3))
    assert mask.sum() == 4
    assert np.allclose(trend, (values[:4, 0]-10)/20*2-1)
    assert np.allclose(observed[:, 1, 1], (values[4, 0]-10)/20*2-1)
    assert np.allclose(recent[0, :, 1, 1], (values[3, 0]-10)/20*2-1)
    assert calendar.sum() == 2 and time_pos == 0 and location == 0
    multi = STGANWindowDataset(values, times, grid, feature_minimum=np.ones(3)*10,
                              feature_scale=np.ones(3)*20, recent_steps=3, trend_steps=4, stride=1)
    recent_multi, trend_multi, _, _, observed_multi, _, _ = multi[0]
    assert recent_multi.shape == (3, 3, 3, 3)
    assert np.allclose(recent_multi[:, :, 1, 1], (values[1:4, 0]-10)/20*2-1)
    assert torch.equal(trend_multi, trend) and torch.equal(observed_multi, observed)


def test_convgru_gate_equations_and_missing_state():
    cell = ConvGRUCell(1, 1)
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()
        # Input channels: value, validity mask, previous hidden state.
        cell.candidate.weight[0, 0, 1, 1] = .4
        cell.candidate.weight[0, 2, 1, 1] = .7
    values = torch.full((1, 1, 1, 1), .2)
    hidden = torch.full_like(values, .6)
    mask = torch.ones_like(values)
    expected = .5 * hidden + .5 * torch.tanh(.4 * values + .7 * .5 * hidden)
    assert torch.allclose(cell(values, hidden, mask), expected)
    absent = cell(values * float('nan'), hidden * float('nan'), torch.zeros_like(mask))
    assert torch.equal(absent, torch.zeros_like(absent))


def test_convgru_uses_earlier_grids_and_resets_between_windows():
    torch.manual_seed(7)
    encoder = ConvGRU(3, 4, 2)
    mask = torch.ones(2, 1, 3, 3)
    mask[:, :, 0, 0] = 0
    sequence = torch.randn(2, 3, 3, 3, 3, requires_grad=True)
    output = encoder(sequence, mask)
    changed = sequence.detach().clone()
    changed[:, 0, :, 1, 1] += 2
    assert not torch.allclose(output, encoder(changed, mask))
    assert not torch.allclose(output, encoder(sequence[:, [1, 0, 2]], mask))
    assert torch.allclose(output, encoder(sequence, mask))  # No state across calls.
    assert torch.allclose(output[:1], encoder(sequence[:1], mask[:1]))
    assert torch.equal(output[:, :, 0, 0], torch.zeros_like(output[:, :, 0, 0]))
    output.square().sum().backward()
    assert sequence.grad[:, 0, :, 1, 1].abs().sum() > 0
    assert torch.equal(sequence.grad[:, :, :, 0, 0], torch.zeros_like(sequence.grad[:, :, :, 0, 0]))
    # Both STGAN paths must use the full history, with fixed trend/calendar/current.
    model = STGAN(n_features=3, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=2)
    trend, calendar = torch.randn(2, 6, 3), torch.randn(2, 31)
    observed = torch.randn(2, 3, 3, 3)
    prediction, real, _, _ = model.components(sequence.detach(), trend, mask, calendar, observed)
    changed_prediction, changed_real, _, _ = model.components(changed, trend, mask, calendar, observed)
    assert not torch.allclose(prediction, changed_prediction)
    assert not torch.allclose(real, changed_real)
    with np.testing.assert_raises_regex(ValueError, 'nonempty'):
        encoder(sequence[:, :0], mask)
    with np.testing.assert_raises_regex(ValueError, 'history'):
        model.discriminator(sequence[:, :1], mask)


def test_masked_losses_and_discriminator_cannot_use_fake_padding():
    torch.manual_seed(12)
    for size, kernel in product((1, 3, 5), (1, 3, 5)):
        model = STGAN(n_features=3, hidden_size=8, n_layers=1,
                      cnn_channels=4, cnn_layers=2, patch_size=size, kernel_size=kernel)
        mask = torch.zeros(2, 1, size, size)
        mask[:, :, size//2:, size//2:] = 1
        recent, trend = torch.randn(2, 3, 3, size, size), torch.randn(2, 6, 3)
        calendar, observed = torch.randn(2, 31), torch.randn(2, 3, size, size)
        outputs = model.components(recent, trend, mask, calendar, observed)
        pred, real, fake, errors = outputs
        assert pred.shape == observed.shape and real.shape == fake.shape == (2, 1)
        assert torch.equal(pred.masked_select(~mask.bool()), torch.zeros_like(pred.masked_select(~mask.bool())))
        # Arbitrarily corrupted/unavailable values must not change either D
        # path, the predictions, or reconstruction errors on observed cells.
        recent_bad = recent.masked_fill(~mask[:, None].bool(), float("nan"))
        observed_bad = observed.masked_fill(~mask.bool(), 1e9)
        other = model.components(recent_bad, trend, mask, calendar, observed_bad)
        assert all(torch.allclose(a, b) for a, b in zip(outputs, other))
        loss = masked_cell_mean(errors, mask).mean() + real.mean() + fake.mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.allclose(masked_cell_mean(torch.ones_like(observed), mask), torch.ones(2))
    assert STGAN(n_features=3).parameter_counts() == {"generator": 140931, "discriminator": 114593}
    assert STGANCNNConfig().recent_steps == 1
    assert STGANCNNConfig(recent_steps=3).recent_steps == 3
    for invalid in (0, -1, 1.5):
        with np.testing.assert_raises_regex(ValueError, "recent_steps"):
            STGANCNNConfig(recent_steps=invalid)
    with np.testing.assert_raises_regex(ValueError, "trend_steps >= recent_steps"):
        STGANCNNConfig(recent_steps=5, trend_steps=4)


def test_kernel_spatial_effect_and_default_compatibility():
    for invalid in (0, -1, 2, 4, 7, 1.5, 3.0, True):
        with np.testing.assert_raises_regex(ValueError, "kernel_size"):
            STGANCNNConfig(kernel_size=invalid)
        with np.testing.assert_raises_regex(ValueError, "kernel_size"):
            STGAN(n_features=3, kernel_size=invalid)
    torch.manual_seed(12)
    original = STGAN(n_features=3)
    torch.manual_seed(12)
    explicit = STGAN(n_features=3, kernel_size=3)
    assert all(torch.equal(value, explicit.state_dict()[key])
               for key, value in original.state_dict().items())
    # A neighboring perturbation reaches the center only through a spatial gate.
    for kernel in (1, 3, 5):
        cell = ConvGRUCell(1, 1, kernel_size=kernel)
        with torch.no_grad():
            for parameter in cell.parameters():
                parameter.zero_()
            cell.candidate.weight[0, 0].fill_(.2)
        values = torch.zeros(1, 1, 3, 3)
        hidden, mask = torch.zeros_like(values), torch.ones_like(values)
        before = cell(values, hidden, mask)
        values[:, :, 1, 0] = 1
        after = cell(values, hidden, mask)
        assert bool(after[0, 0, 1, 1] != before[0, 0, 1, 1]) == (kernel != 1)


def test_notebook_kernel_output_selection_and_cli():
    import os
    import re
    from unittest.mock import patch
    from scripts.run_pvgis_stgan import parse_args
    notebook = json.loads((Path(__file__).resolve().parents[1] /
        'notebooks/stgan_cnn_pvgis_workflow.ipynb').read_text(encoding='utf-8'))
    config_source = next(''.join(c['source']) for c in notebook['cells']
                         if c.get('id') == 'configuration')
    paths = []
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('STGAN_CNN_OUT_DIR', 'STGAN_MANIFEST', 'STGAN_GRID_CRS')}
    with patch.dict('os.environ', environment, clear=True):
        for kernel in (1, 3, 5):
            namespace = {}
            exec(re.sub(r'KERNEL_SIZE = [135]', f'KERNEL_SIZE = {kernel}', config_source), namespace)
            assert namespace['CONFIG'].patch_size == 3
            assert namespace['CONFIG'].kernel_size == kernel
            name = f'convgru_patch3_kernel{kernel}_optimized'
            assert namespace['OUT_ROOT'].name == name
            assert namespace['SEED_DIR'] == namespace['OUT_ROOT'] / 'seed_20'
            paths.append(namespace['OUT_ROOT'])
    assert len(set(paths)) == 3
    with patch.dict('os.environ', {**environment, 'STGAN_CNN_OUT_DIR': str(paths[0])}, clear=True):
        namespace = {}
        exec(config_source, namespace)
        assert namespace['OUT_ROOT'] == paths[0]  # Explicit path retains precedence.
    args = parse_args(['--manifest', 'manifest.csv', '--out-dir', 'run', '--kernel-size', '1'])
    assert args.kernel_size == 1 and args.patch_size == 3
    assert parse_args(['--manifest', 'manifest.csv', '--out-dir', 'run']).kernel_size == 3


def test_runner_checkpoint_and_existing_reporting():
    from physiq_pv.reporting.input_target_cases import load_clean_stgan_labels
    lat, lon = coordinates()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        for i in range(9):
            site = root / f"site_{i}"
            site.mkdir()
            for split, start, n in (("train", "2018-12-31 12:10", 12), ("test", "2019-01-01 00:10", 4)):
                pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="h"),
                    "solar_irradiance_poa": 200+np.arange(n)*2+i,
                    "temperature_2m": 10+np.sin(np.arange(n))+i/10,
                    "wind_speed_10m": 2+np.arange(n)/10, "is_daytime": True}).to_csv(site/f"{split}.csv", index=False)
            rows.append({"location": str(i), "site_key": f"site_{i}", "latitude": lat[i],
                "longitude": lon[i], "train_csv": str(site/"train.csv"), "test_csv": str(site/"test.csv")})
        manifest = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        config = STGANCNNConfig(epochs=1, batch_size=8, hidden_size=8, n_layers=1,
            cnn_channels=4, cnn_layers=2, recent_steps=3, trend_steps=4, train_samples_per_epoch=16)
        audit = run_stgan(manifest_path=manifest, out_dir=root/"audit", config=config,
                         paper_top_k_percent=1, audit_only=True)
        assert not (audit/"cube_cache").exists()
        out = run_stgan(manifest_path=manifest, out_dir=root/"run", config=config,
                        paper_top_k_percent=25, device="cpu")
        checkpoint = out / "seed_20/checkpoint.pt"
        restored, payload = load_stgan_checkpoint(checkpoint)
        assert payload["model_class"] == "STGAN_CONVGRU" and "graph" not in payload
        assert payload["format_version"] == 2
        assert payload["model_config"]["kernel_size"] == 3
        assert payload["window_config"] == {"recent_steps": 3, "trend_steps": 4}
        epoch_model, _ = load_stgan_checkpoint(checkpoint.with_name("checkpoint_epoch_1.pt"))
        inputs = (torch.randn(2, 3, 3, 3, 3), torch.randn(2, 4, 3),
                  torch.ones(2, 1, 3, 3), torch.randn(2, 31), torch.randn(2, 3, 3, 3))
        assert all(torch.equal(a, b) for a, b in zip(restored.components(*inputs), epoch_model.components(*inputs)))
        # ConvGRU checkpoints written before the ablation had no kernel field.
        old_payload = {**payload, 'model_config': dict(payload['model_config'])}
        old_payload['model_config'].pop('kernel_size')
        old_path = root / 'old_convgru.pt'
        torch.save(old_payload, old_path)
        old_model, _ = load_stgan_checkpoint(old_path)
        assert all(torch.equal(a, b) for a, b in zip(restored.components(*inputs), old_model.components(*inputs)))
        legacy_path = root / 'legacy.pt'
        torch.save({"format_version": 1, "model_class": "STGAN_CNN"}, legacy_path)
        with np.testing.assert_raises_regex(ValueError, "Legacy feed-forward CNN"):
            load_stgan_checkpoint(legacy_path)
        assert payload["grid"]["node_indices"].shape == (9, 3, 3)
        assert np.allclose(payload["normalization"]["minimum"], [200, 10+np.sin(np.arange(12)).min(), 2])
        assert checkpoint.with_name("checkpoint_epoch_1.pt").exists()
        scores = pd.read_csv(out/"seed_20/anomaly_scores.csv")
        assert len(scores) == 36 and scores.is_anomaly.sum() == 9
        assert pd.to_datetime(scores.timestamp).min() == pd.Timestamp("2019-01-01 00:10")
        labels = load_clean_stgan_labels(out/"seed_20/anomaly_scores.csv", top_percent=25)
        assert len(labels) == 36 and labels.is_anomaly.sum() == 9
        details = pd.concat([pd.read_csv(p) for p in (out/"seed_20/locations").glob("*/test_scores.csv")])
        a, b = details.generator_score_raw, details.discriminator_score_raw
        expected = (a-a.min())/(a.max()-a.min()) + (b-b.min())/(b.max()-b.min())
        assert np.allclose(details.anomaly_score, expected, atol=1e-5)
        boundary = pd.read_csv(out/"seed_20/boundary_summary.csv")
        assert boundary.n_scored.sum() == 36
        metadata = json.loads((out/"seed_20/metadata.json").read_text())
        assert not metadata["backend"]["paper_alignment"]["generator_discriminator_architecture"]
        assert metadata["backend"]["performance"]["scoring_samples_per_second"] > 0
        assert metadata['backend']['kernel_size'] == 3
        ablation_config = replace(config, kernel_size=1)
        ablation_out = run_stgan(manifest_path=manifest, out_dir=root/'kernel1',
            config=ablation_config, paper_top_k_percent=25, device='cpu')
        ablation_model, ablation_payload = load_stgan_checkpoint(ablation_out/'seed_20/checkpoint.pt')
        assert ablation_payload['model_config']['kernel_size'] == 1
        for encoder in (ablation_model.generator.recent_encoder, ablation_model.discriminator.sequence_encoder):
            assert all(cell.candidate.kernel_size == (1, 1) for cell in encoder.layers)
        ablation_metadata = json.loads((ablation_out/'seed_20/metadata.json').read_text())
        assert ablation_metadata['backend']['kernel_size'] == 1
        ablation_run = json.loads((ablation_out/'run_metadata.json').read_text())
        assert ablation_run['configuration']['model']['kernel_size'] == 1
        assert ablation_metadata['backend']['parameter_counts']['generator'] < metadata['backend']['parameter_counts']['generator']
        assert pd.read_csv(ablation_out/'seed_20/anomaly_scores.csv').is_anomaly.sum() == 9
        # Notebook audit, saved-run figures and summary run on actual outputs
        # of this tiny integration run, without starting another training.
        notebook = json.loads((Path(__file__).resolve().parents[1]/"notebooks/stgan_cnn_pvgis_workflow.ipynb").read_text(encoding="utf-8"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        namespace = {"pd": pd, "np": np, "json": json, "plt": plt,
            "display": lambda *args: None, "MANIFEST": manifest, "CONFIG": config,
            "SEED_DIR": out/"seed_20", "STGAN": STGAN, "build_spatial_grid": build_spatial_grid,
            "replace": __import__("dataclasses").replace,
            "OUT_ROOT": out, "SEEDS": (20,), "TOP_K_PERCENT": 25, "RUN_TRAINING": False}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compiled = compile("".join(cell["source"]), cell["id"], "exec")
                if cell["id"] in ("grid-audit", "parameter-counts", "training", "run-summary", "search-plan"):
                    if cell["id"] == "run-summary":
                        # Reading a summary alone must report the SAVED run,
                        # even if the user edited the configuration cell.
                        namespace["CONFIG"] = namespace["replace"](config, hidden_size=32)
                    exec(compiled, namespace)
                    if cell['id'] == 'training':
                        run_metadata_path = out / 'run_metadata.json'
                        original_metadata = run_metadata_path.read_text()
                        legacy_metadata = json.loads(original_metadata)
                        legacy_metadata['alignment_policy'] = 'cnn_spatial_ablation_with_original_lstm_losses_and_score'
                        try:
                            older_metadata = json.loads(original_metadata)
                            older_metadata['configuration']['model'].pop('kernel_size')
                            run_metadata_path.write_text(json.dumps(older_metadata))
                            exec(compiled, namespace)  # Old kernel=3 run is reusable.
                            namespace['CONFIG'] = replace(config, kernel_size=1)
                            with np.testing.assert_raises_regex(ValueError, 'configurazione'):
                                exec(compiled, namespace)
                            namespace['CONFIG'] = config
                            run_metadata_path.write_text(json.dumps(legacy_metadata))
                            with np.testing.assert_raises_regex(ValueError, 'architettura'):
                                exec(compiled, namespace)
                        finally:
                            run_metadata_path.write_text(original_metadata)
                    if cell["id"] == "run-summary":
                        saved_summary = json.loads(namespace["summary_text"].split("\n", 1)[1].rsplit("\nEND_", 1)[0])
                        assert saved_summary["configuration"]["hidden_size"] == 8
                        namespace["CONFIG"] = config
        plt.close("all")
        assert (out/"seed_20/figures/training_losses.png").is_file()
        assert (out/"seed_20/model_results_summary.txt").is_file()
        candidates = pd.read_csv(out/"seed_20/candidate_configurations.csv")
        assert set(candidates.hidden_size) == {config.hidden_size}
        assert set(candidates.kernel_size) == {1, 3, 5}
        assert set(candidates.patch_size) == {3}
        from physiq_pv.reporting.stgan_cnn_comparison import compare_stgan_exports
        agreement, _ = compare_stgan_exports(out/"seed_20", out/"seed_20", out_dir=out/"comparison")
        total = agreement.set_index("group").loc["all"]
        assert total.n_common == 36 and total.anomaly_jaccard == 1
        # Deliberate differences and missing timestamps are audited, rather
        # than counted as normal decisions or compared on different samples.
        import shutil
        baseline = root/"baseline"
        baseline.mkdir()
        shutil.copyfile(out/"locations.csv", baseline/"locations.csv")
        shutil.copytree(out/"seed_20/locations", baseline/"seed_20/locations")
        altered = baseline/"seed_20/locations/site_4/test_scores.csv"
        altered_frame = pd.read_csv(altered).iloc[1:].copy()
        altered_frame["is_anomaly"] = ~altered_frame.is_anomaly
        altered_frame.to_csv(altered, index=False)
        agreement, _ = compare_stgan_exports(out/"seed_20", baseline/"seed_20", out_dir=out/"comparison2")
        interior = agreement.set_index("group").loc["interior"]
        assert interior.n_common == 3 and interior.n_cnn_only_timestamps == 1
        assert interior.cnn_only_anomalous + interior.baseline_only_anomalous == 3
        layout_path = out/"grid_locations.csv"
        original_layout = layout_path.read_bytes()
        pd.read_csv(layout_path).iloc[:-1].to_csv(layout_path, index=False)
        with np.testing.assert_raises_regex(ValueError, "exactly the CNN location IDs"):
            compare_stgan_exports(out/"seed_20", baseline/"seed_20", out_dir=out/"bad_comparison")
        layout_path.write_bytes(original_layout)
        baseline_locations = baseline/"locations.csv"
        baseline_frame = pd.read_csv(baseline_locations)
        baseline_frame.loc[0, "latitude"] += .1
        baseline_frame.to_csv(baseline_locations, index=False)
        with np.testing.assert_raises_regex(ValueError, "Mismatched latitude"):
            compare_stgan_exports(out/"seed_20", baseline/"seed_20", out_dir=out/"bad_comparison")
        baseline_frame.loc[0, "latitude"] -= .1
        extra = baseline_frame.iloc[[0]].copy()
        extra["location"], extra["site_key"] = 99, "site_99"
        pd.concat([baseline_frame, extra], ignore_index=True).to_csv(baseline_locations, index=False)
        shutil.copytree(baseline/"seed_20/locations/site_0", baseline/"seed_20/locations/site_99")
        agreement, _ = compare_stgan_exports(out/"seed_20", baseline/"seed_20", out_dir=out/"comparison3")
        assert agreement.set_index("group").loc["baseline_only_location", "n_baseline_only_timestamps"] == 4
        with np.testing.assert_raises(FileExistsError):
            run_stgan(manifest_path=manifest, out_dir=out, config=config, paper_top_k_percent=1, device="cpu")
        del restored, payload
        gc.collect()
    flags, ranks, _ = paper_top_k_ranking(np.ones((2, 4)), 25)
    assert flags.sum() == 2 and ranks.ravel().tolist() == list(range(1, 9))


def test_failed_cube_loading_closes_memmaps():
    from physiq_pv.anomaly_detection.stgan import load_aligned_manifest_cubes
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=4, freq="h"),
                             "temperature_2m": [1., 2., 3., 4.]})
        data.to_csv(root/"train.csv", index=False)
        data.to_csv(root/"test0.csv", index=False)
        data.loc[3, "timestamp"] = data.loc[2, "timestamp"]
        data.to_csv(root/"test1.csv", index=False)
        manifest = pd.DataFrame([{"location": str(i), "site_key": f"s{i}",
            "latitude": 45+i*.05, "longitude": 7, "train_csv": str(root/"train.csv"),
            "test_csv": str(root/f"test{i}.csv")} for i in range(2)])
        try:
            load_aligned_manifest_cubes(manifest, cache_dir=root/"cache")
        except ValueError as exc:
            assert "duplicate" in str(exc)
            # Keep exception/traceback alive while checking handles were closed.
            for path in (root/"cache").glob("*.npy"):
                path.unlink()
        else:
            raise AssertionError("Expected duplicate timestamp rejection")


def test_preparation_with_current_pvgis_loader():
    import xarray as xr
    from scripts.prepare_pvgis_stgan import main
    lat, lon = coordinates()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for year in (2018, 2019):
            times = pd.date_range(f"{year}-06-01", periods=4, freq="h")
            ds = xr.Dataset({name: (("location", "time"), np.full((9, 4), value))
                for name, value in {"temperature_2m": 15, "wind_speed_10m": 2,
                    "direct_irradiance_tilted": 200, "diffuse_irradiance_tilted": 50,
                    "pv_power_output": 100}.items()},
                coords={"location": np.arange(9), "time": times,
                        "lat": ("location", lat), "lon": ("location", lon)})
            ds.to_netcdf(root/f"piedmont_pvgis_{year}.nc")
            ds.close()
        main(["--pvgis-dir", str(root), "--out-dir", str(root/"prepared"),
              "--train-start", "2018", "--train-end", "2018", "--test-year", "2019"])
        manifest = pd.read_csv(root/"prepared/manifest.csv")
        assert len(manifest) == 9
        data = pd.read_csv(manifest.iloc[0].train_csv)
        assert "pv_power_output" not in data and "is_anomaly" not in data
        assert data.solar_irradiance_poa.eq(250).all()


def test_cuda_reference_dimensions_smoke():
    if not torch.cuda.is_available():
        print("SKIP CUDA unavailable")
        return
    from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
    lat, lon = coordinates()
    data = np.random.default_rng(5).normal(size=(174, 9, 3)).astype(np.float32)
    times = pd.date_range("2018-12-24 22:10", periods=174, freq="h")
    result = fit_and_score_stgan(data[:170], data[170:], train_timestamps=times[:170],
        test_timestamps=times[170:], location_names=tuple(map(str, range(9))),
        feature_names=("solar", "temperature", "wind"), latitudes=lat, longitudes=lon,
        epochs=1, batch_size=8, train_samples_per_epoch=8, device="cuda")
    assert np.isfinite(result.test_scores).all() and result.test_scores.shape == (4, 9)
    assert result.metadata["parameter_counts"] == {"generator": 140931, "discriminator": 114593}
    assert result.metadata["performance"]["peak_cuda_memory_bytes"] > 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
