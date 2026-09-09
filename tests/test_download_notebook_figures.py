"""Packaging and SSH command tests; no live server or SSH credentials required."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_notebook_figures import DEFAULT_PATHS, FIGURE_DIRS, build_bundle, download


def write_file(path, content=b"synthetic-image"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_sources(root):
    for key, directories in FIGURE_DIRS.items():
        for directory in directories:
            write_file(root / DEFAULT_PATHS[key] / directory / "figure.png")
    threshold = root / DEFAULT_PATHS["threshold"]
    for detector in ("mtgflow", "stgan"):
        write_file(threshold / f"figures/{detector}_mae_rmse_sensitivity_t1_t6.png")
    write_file(threshold / "analysis_metadata.json", b'{"mae_bands": "Q1-Q3 and Tukey"}')
    stgan = root / DEFAULT_PATHS["stgan"]
    write_file(stgan / "score_timeline/stgan_anomaly_score_timeline.png")
    write_file(stgan / "predictions.csv", b"do-not-download")
    write_file(stgan / "stgan_may08_may17_t1_t6_pipeline_style/figures/old-event.png")
    cases = root / DEFAULT_PATHS["cases"]
    for metric in ("mae", "rmse"):
        for horizon in (1, 6):
            for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100)):
                write_file(cases / f"figures/{metric}_daytime_{lo}_{hi}_pct_t_plus_{horizon}.png")
    write_file(cases / "input_target_results_summary.txt", b"BEGIN_STGAN_INPUT_TARGET_SUMMARY")
    write_file(cases / "input_target_classification.csv", b"do-not-download")
    return cases


def test_pack_and_download(tmp_path):
    root = tmp_path / "server"
    cases = write_sources(root)
    bundle = build_bundle(root, tmp_path / "bundle.zip")
    with ZipFile(bundle) as archive:
        names = archive.namelist()
        assert len([n for n in names if n.endswith(".png")]) == 29
        assert len({n.split('/')[0] for n in names if n.endswith('.png')}) == 3
        assert not any("predictions.csv" in n or "classification.csv" in n or "old-event" in n for n in names)
        assert any(n.endswith("input_target_results_summary.txt") for n in names)
        manifest = json.loads(archive.read("manifest.json"))
        for row in manifest["files"]:
            assert hashlib.sha256(archive.read(row["archive_path"])).hexdigest() == row["sha256"]

    args = argparse.Namespace(
        host="apedalino@server", remote_root="/home/apedalino/path with spaces",
        remote_python="python3", threshold_dir=None, stgan_dir=None, cases_dir=None,
        port=2222, identity="key with spaces", output=str(tmp_path / "download.zip"),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "scp":
            shutil.copyfile(bundle, command[-1])

    with patch("scripts.download_notebook_figures.shutil.which", return_value="available"), \
            patch("scripts.download_notebook_figures.subprocess.run", side_effect=fake_run):
        download(args)
    assert Path(args.output).read_bytes() == bundle.read_bytes()
    assert calls[0][0][:3] == ["ssh", "-p", "2222"]
    assert "'/home/apedalino/path with spaces'" in calls[0][0][-1]
    assert calls[0][1]["input"].startswith(b'"""Package/download')
    assert calls[1][0][:3] == ["scp", "-P", "2222"]

    # Incomplete output must fail before replacing a previous successful ZIP.
    original = bundle.read_bytes()
    (cases / "figures/rmse_daytime_80_100_pct_t_plus_6.png").unlink()
    try:
        build_bundle(root, bundle)
    except FileNotFoundError as exc:
        assert "rmse_daytime_80_100_pct_t_plus_6.png" in str(exc)
    else:
        raise AssertionError("Missing new figure was accepted")
    assert bundle.read_bytes() == original


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_pack_and_download(Path(directory))
    print("PASS: three-notebook ZIP, report selection, checksums and mocked SSH download")
