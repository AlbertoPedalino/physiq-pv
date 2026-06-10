"""Smoke tests for the configurable PV target upper clip ablation."""

import inspect
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

from physiq_pv.data.pvgis_stgnn_dataset import (
    assemble_feats,
    train_model,
)
from physiq_pv.experiments import pvgis_stgnn_runner
from physiq_pv.experiments.pvgis_stgnn_runner import build_arg_parser
from physiq_pv.model.st_gnn import STGNN


def _raw_and_norm():
    pv = np.array([[-2.0], [5.0], [20.0]], dtype=np.float64)
    zeros = np.zeros_like(pv)
    raw = {
        "pv": pv,
        "temp": zeros,
        "solar_wm2": zeros,
        "wind": zeros,
        "sin": zeros,
        "cos": zeros,
        "kt": zeros,
        "kt_std": zeros,
        "dghi": zeros,
        "dni": zeros,
        "dhi": zeros,
    }
    norm = {
        "pv_scale": np.array([10.0]),
        "z": {
            key: (0.0, 1.0)
            for key in ("temp", "solar_wm2", "wind", "dghi")
        },
    }
    return raw, norm


def test_default_clip_is_backward_compatible() -> None:
    raw, norm = _raw_and_norm()
    feats, pv_norm, _ = assemble_feats(raw, norm)
    np.testing.assert_array_equal(
        pv_norm[:, 0], np.array([0.0, 0.5, 1.5], dtype=np.float32)
    )
    np.testing.assert_array_equal(feats[:, 0, -1], pv_norm[:, 0])


def test_none_removes_only_upper_clip() -> None:
    raw, norm = _raw_and_norm()
    feats, pv_norm, _ = assemble_feats(
        raw, norm, pv_target_clip_max=None
    )
    np.testing.assert_array_equal(
        pv_norm[:, 0], np.array([0.0, 0.5, 2.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(feats[:, 0, -1], pv_norm[:, 0])


def test_cli_default_and_null_parser() -> None:
    parser = build_arg_parser()
    assert parser.parse_args([]).pv_target_clip_max == 1.5
    assert parser.parse_args(
        ["--pv-target-clip-max", "null"]
    ).pv_target_clip_max is None
    assert parser.parse_args(
        ["--pv_target_clip_max", "none"]
    ).pv_target_clip_max is None


def test_training_invariants_remain_explicit() -> None:
    assert "F.softplus" in inspect.getsource(STGNN.forward)
    assert "torch.nn.MSELoss()" in inspect.getsource(train_model)
    runner_source = inspect.getsource(pvgis_stgnn_runner.run_from_args)
    assert runner_source.index("predictions = predict") < runner_source.index(
        "predictions = attach_anomaly_labels"
    )


def test_no_peak_clip_sweep_yaml() -> None:
    config_path = (
        _REPO_ROOT
        / "configs"
        / "sweeps"
        / "pvgis_stgnn_enhanced_mc_dropout_no_peak_clip_paper_style.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parameters = config["parameters"]
    assert parameters["model_type"]["value"] == "stgnn_enhanced_dropout"
    assert parameters["feature_set"]["value"] == "full"
    assert parameters["target_variable"]["value"] == "pv_power_output"
    assert parameters["train_years"]["value"] == "2016,2017,2018"
    assert parameters["test_year"]["value"] == 2019
    assert parameters["pv_target_clip_max"]["value"] is None
    assert parameters["enable_posthoc_calibration"]["value"] is False
    command = config["command"]
    clip_arg_index = command.index("--pv-target-clip-max")
    assert command[clip_arg_index + 1] == "none"


if __name__ == "__main__":
    test_default_clip_is_backward_compatible()
    test_none_removes_only_upper_clip()
    test_cli_default_and_null_parser()
    test_training_invariants_remain_explicit()
    test_no_peak_clip_sweep_yaml()
    print("PASS: PVGIS target clip ablation")
