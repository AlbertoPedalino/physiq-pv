"""Synthetic CPU tests for the self-contained CATCH implementation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.anomaly_detection.catch import (
    CATCH,
    CATCHPreprocessor,
    CATCHWindowDataset,
    temporal_train_validation_split,
)
from physiq_pv.anomaly_detection.catch_model import (
    CATCHModel,
    ResidualFlattenHead,
    frequency_point_error,
)
from physiq_pv.experiments.pvgis_catch_pipeline import PVGISCATCHConfig
from physiq_pv.experiments.pvgis_catch_runner import build_arg_parser, run_from_args


def _segment(length: int, phase: float = 0.0) -> np.ndarray:
    time = np.arange(length, dtype=np.float32)
    first = np.sin(2 * np.pi * time / 12 + phase)
    second = np.cos(2 * np.pi * time / 18 + phase)
    return np.column_stack((first, second, 0.7 * first + 0.3 * second))


def _detector() -> CATCH:
    return CATCH(
        ["pv", "irradiance", "temperature"],
        seq_len=12,
        patch_size=4,
        patch_stride=2,
        inference_patch_size=6,
        inference_patch_stride=1,
        cf_dim=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        head_dim=4,
        d_ff=16,
        dropout=0.0,
        head_dropout=0.0,
        head_layers=1,
        contamination=0.05,
        epochs=1,
        batch_size=16,
        validation_split=0.2,
        training_window_stride=4,
        scoring_window_stride=4,
        model_steps_per_mask=2,
        device="cpu",
        seed=5,
        verbose=False,
    )


def test_model_reconstructs_expected_shapes_and_keeps_mask_diagonal() -> None:
    torch.manual_seed(2)
    model = CATCHModel(
        3,
        seq_len=16,
        patch_size=4,
        patch_stride=2,
        cf_dim=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        head_dim=4,
        d_ff=16,
        dropout=0.0,
        head_dropout=0.0,
        head_layers=3,
    )
    output = model(torch.randn(2, 16, 3))
    assert output.reconstruction.shape == (2, 16, 3)
    assert output.frequency_reconstruction.shape == (2, 16, 3)
    assert output.masks.shape == (2, 7, 3, 3)
    assert isinstance(model.real_head, ResidualFlattenHead)
    assert len(model.real_head.residual_layers) == 3
    assert model.ircom.in_features == 32 and model.ircom.out_features == 16
    diagonal = output.masks.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(diagonal, torch.ones_like(diagonal))
    assert set(output.masks.detach().cpu().unique().tolist()) <= {0.0, 1.0}
    assert torch.isfinite(output.clustering_loss)
    assert torch.isfinite(output.regularization_loss)
    (output.reconstruction.square().mean() + output.clustering_loss).backward()
    mask_gradient = model.mask_generator.projection.weight.grad
    assert mask_gradient is not None
    assert torch.isfinite(mask_gradient).all()
    assert torch.count_nonzero(mask_gradient) > 0


def test_frequency_patch_errors_are_aligned_back_to_points() -> None:
    observed = torch.zeros(1, 24, 2)
    reconstructed = observed.clone()
    reconstructed[:, 9:15, 0] = 4.0
    errors = frequency_point_error(
        reconstructed, observed, patch_size=6, patch_stride=1
    )
    assert errors.shape == observed.shape
    assert torch.isfinite(errors).all()
    assert errors[0, 10:14, 0].mean() > errors[0, :4, 0].mean()
    torch.testing.assert_close(errors[..., 1], torch.zeros_like(errors[..., 1]))


def test_reference_forward_matches_golden_values() -> None:
    """Lock the projected-mask, residual-head, and ircom forward contract."""

    model = CATCHModel(
        2,
        seq_len=4,
        patch_size=2,
        patch_stride=2,
        cf_dim=4,
        d_model=2,
        n_layers=1,
        n_heads=1,
        head_dim=2,
        d_ff=4,
        dropout=0.0,
        head_dropout=0.0,
        head_layers=1,
        temperature=0.07,
        mask_source="projected",
    )
    for parameter in model.parameters():
        parameter.data.fill_(0.01)
    model.eval()
    values = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    output = model(values)
    expected_reconstruction = torch.tensor(
        [[[3.0228839, 4.0228839]] * 4], dtype=torch.float32
    )
    expected_spectrum = torch.full(
        (1, 4, 2), complex(0.01169906, 0.01169906), dtype=torch.complex64
    )
    torch.testing.assert_close(
        output.reconstruction, expected_reconstruction, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        output.frequency_reconstruction,
        expected_spectrum,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output.clustering_loss,
        torch.tensor(0.34657359),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output.regularization_loss,
        torch.tensor(0.5),
        rtol=0,
        atol=1e-7,
    )


def test_windows_never_cross_segments_and_preprocessing_is_train_only() -> None:
    first = _segment(30)
    second = _segment(30, 0.3)
    preprocessor = CATCHPreprocessor(3)
    scaled = preprocessor.fit_transform([first, second])
    mean = preprocessor.mean_.copy()
    scale = preprocessor.scale_.copy()
    dataset = CATCHWindowDataset(scaled, seq_len=8, stride=3)
    assert {entry[0] for entry in dataset.entries} == {0, 1}
    assert all(start <= 22 for _, start in dataset.entries)
    extreme = _segment(20) + 1000
    preprocessor.transform([extreme])
    np.testing.assert_array_equal(preprocessor.mean_, mean)
    np.testing.assert_array_equal(preprocessor.scale_, scale)
    train_parts, validation_parts = temporal_train_validation_split(
        [first, second], validation_split=0.2, min_length=6
    )
    assert [len(part) for part in train_parts] == [24, 24]
    assert [len(part) for part in validation_parts] == [6, 6]
    np.testing.assert_array_equal(train_parts[0][-1], first[23])
    np.testing.assert_array_equal(validation_parts[0][0], first[24])


def test_fit_and_score_are_finite_label_unaware_and_interpretable() -> None:
    train = [_segment(64), _segment(64, 0.2)]
    detector = _detector().fit(train)
    test = _segment(48, 0.1)
    test[24:30, 0] += 8.0
    scores = detector.score_segments([test])
    assert len(scores.global_scores) == len(test)
    assert np.isfinite(scores.global_scores).all()
    assert np.isfinite(scores.frequency_scores).all()
    assert scores.global_scores[24:30].mean() > np.median(scores.global_scores)
    frame = detector.scores_frame(
        scores,
        [pd.date_range("2019-01-01", periods=len(test), freq="h")],
        entity="a",
    )
    required = {
        "location",
        "timestamp",
        "global_score",
        "time_score",
        "frequency_score",
        "threshold",
        "is_anomaly",
        "top_sensor",
        "reconstructed__pv",
        "contribution__pv",
    }
    assert required <= set(frame.columns)
    contribution_columns = [
        f"contribution__{sensor}" for sensor in detector.sensor_names
    ]
    np.testing.assert_allclose(frame[contribution_columns].sum(axis=1), 1.0)


def _tiny_pvgis(year: int) -> xr.Dataset:
    time = pd.date_range(f"{year}-01-01", periods=32, freq="h")
    shape = (2, len(time))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return xr.Dataset(
        {
            "pv_power_output": (("location", "time"), base),
            "direct_irradiance_tilted": (("location", "time"), base * 0.7),
            "diffuse_irradiance_tilted": (("location", "time"), base * 0.3),
            "temperature_2m": (("location", "time"), base * 0.1),
            "wind_speed_10m": (("location", "time"), base * 0.01),
        },
        coords={"location": ["a", "b"], "time": time},
    )


def test_typed_config_and_pvgis_runner_are_self_contained(tmp_path: Path) -> None:
    config = PVGISCATCHConfig(
        pvgis_dir="unused",
        train_years=(2005,),
        test_year=2019,
        sensors=("pv", "irradiance", "temperature"),
        seq_len=8,
        patch_size=4,
        patch_stride=2,
        cf_dim=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        head_dim=4,
        d_ff=16,
        head_layers=1,
        epochs=1,
        batch_size=8,
        training_window_stride=4,
        scoring_window_stride=4,
        verbose=False,
    )
    assert isinstance(config.detector(), CATCH)
    official_defaults = PVGISCATCHConfig(pvgis_dir="unused")
    assert (
        official_defaults.cf_dim,
        official_defaults.d_model,
        official_defaults.head_dim,
        official_defaults.n_layers,
        official_defaults.head_layers,
    ) == (64, 128, 64, 3, 3)

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _tiny_pvgis(2005).to_netcdf(data_dir / "piedmont_pvgis_2005.nc")
    _tiny_pvgis(2019).to_netcdf(data_dir / "piedmont_pvgis_2019.nc")
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--pvgis-dir",
            str(data_dir),
            "--train-years",
            "2005",
            "--test-year",
            "2019",
            "--out-dir",
            str(out_dir),
            "--max-locations",
            "1",
            "--seq-len",
            "8",
            "--patch-size",
            "4",
            "--patch-stride",
            "2",
            "--inference-patch-size",
            "4",
            "--d-model",
            "8",
            "--cf-dim",
            "8",
            "--n-layers",
            "1",
            "--n-heads",
            "2",
            "--head-dim",
            "4",
            "--d-ff",
            "16",
            "--head-layers",
            "1",
            "--dropout",
            "0",
            "--head-dropout",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--training-window-stride",
            "4",
            "--scoring-window-stride",
            "4",
            "--model-steps-per-mask",
            "2",
            "--validation-split",
            "0.25",
            "--device",
            "cpu",
            "--quiet",
        ]
    )
    paths = run_from_args(args, parser)
    assert all(path.exists() for path in paths.values())
    metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert metadata["method"] == "CATCH"
    assert metadata["label_unaware"] is True
    scores = pd.read_csv(paths["scores"])
    assert len(scores) == 32
    assert {"time_score", "frequency_score", "global_score"} <= set(scores.columns)


if __name__ == "__main__":
    test_model_reconstructs_expected_shapes_and_keeps_mask_diagonal()
    test_frequency_patch_errors_are_aligned_back_to_points()
    test_reference_forward_matches_golden_values()
    test_windows_never_cross_segments_and_preprocessing_is_train_only()
    test_fit_and_score_are_finite_label_unaware_and_interpretable()
    with tempfile.TemporaryDirectory() as directory:
        test_typed_config_and_pvgis_runner_are_self_contained(Path(directory))
    print("PASS: CATCH tests")
