"""Package/download figures from the three updated anomaly notebooks.

Server: python scripts/download_notebook_figures.py pack
Local:  python scripts/download_notebook_figures.py download --host USER@HOST

Uses only the standard library; download additionally uses OpenSSH ssh/scp.
The local script is sent to Python's stdin on the server, so it need not have
been installed there. No notebook or model is executed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
DEFAULT_PATHS = {
    "threshold": "outputs/anomaly_threshold_sensitivity_t1_t6",
    "stgan": "outputs/sde_stgan_direct_multihorizon_seed20_quality_filtered",
    "cases": "outputs/stgan_input_target_cases_t1_t6",
}
ENV_PATHS = {
    "threshold": "ANOMALY_SENSITIVITY_OUT_DIR",
    "stgan": "STGAN_POSTHOC_ROOT",
    "cases": "STGAN_INPUT_TARGET_OUT_DIR",
}
LABELS = {
    "threshold": "01_anomaly_threshold_sensitivity_mtgflow_stgan",
    "stgan": "02_stgan_pointwise_posthoc_sdenet",
    "cases": "03_stgan_input_target_cases_sdenet",
}
FIGURE_DIRS = {
    "threshold": ("figures",),
    "stgan": (
        "score_timeline", "figures/direct_multihorizon",
        "posthoc_by_horizon/t_plus_1/figures", "posthoc_by_horizon/t_plus_6/figures",
    ),
    "cases": ("figures",),
}
# Include small, explicit context files, never the large raw prediction/score CSVs.
REPORTS = {
    "threshold": (
        "analysis_metadata.json", "detector_threshold_sensitivity_metrics.csv",
        "detector_threshold_sensitivity_by_bin_metrics.csv",
    ),
    "stgan": ("evaluation_source.json", "score_timeline/stgan_anomaly_score_timeline.csv"),
    "cases": (
        "analysis_metadata.json", "input_target_results_summary.txt",
        "input_target_bin_metrics.csv", "input_target_overall_metrics.csv",
        "input_target_coverage_audit.csv",
    ),
}


def build_bundle(root: Path, destination: Path, overrides=None) -> Path:
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    overrides = overrides or {}
    selected = {}
    sources = {}
    counts = {}
    missing = []
    for key, default in DEFAULT_PATHS.items():
        source = Path(overrides.get(key) or os.environ.get(ENV_PATHS[key]) or default)
        source = (source if source.is_absolute() else root / source).resolve()
        sources[key] = source
        counts[key] = 0
        for relative in FIGURE_DIRS[key]:
            directory = source / relative
            figures = sorted(
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ) if directory.is_dir() else []
            if not figures:
                missing.append(str(directory))
            for path in figures:
                if not path.resolve().is_relative_to(source):
                    raise ValueError(f"Figura fuori dalla cartella del notebook: {path}")
                name = f"{LABELS[key]}/{path.relative_to(source).as_posix()}"
                selected[name] = path
                counts[key] += 1
        for relative in REPORTS[key]:
            path = source / relative
            if path.is_file():
                if not path.resolve().is_relative_to(source):
                    raise ValueError(f"Report fuori dalla cartella del notebook: {path}")
                selected[f"{LABELS[key]}/{relative}"] = path
    required = [
        sources["threshold"] / "figures/mtgflow_mae_rmse_sensitivity_t1_t6.png",
        sources["threshold"] / "figures/stgan_mae_rmse_sensitivity_t1_t6.png",
        sources["stgan"] / "score_timeline/stgan_anomaly_score_timeline.png",
    ]
    required.extend(
        sources["cases"] / f"figures/{metric}_daytime_{lo}_{hi}_pct_t_plus_{horizon}.png"
        for metric in ("mae", "rmse") for horizon in (1, 6)
        for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
    )
    missing.extend(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Figure mancanti: eseguire i tre notebook o correggere i percorsi.\n - "
            + "\n - ".join(missing)
        )
    metadata_path = sources["threshold"] / "analysis_metadata.json"
    if not metadata_path.is_file() or "mae_bands" not in json.loads(metadata_path.read_text(encoding="utf-8")):
        raise ValueError("Rieseguire il notebook threshold aggiornato: mancano i metadata delle fasce MAE.")
    if destination in (path.resolve() for path in selected.values()):
        raise ValueError("La destinazione ZIP non può sovrascrivere un file sorgente.")

    manifest = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(selected.items()):
                data = path.read_bytes()
                archive.writestr(name, data)
                manifest.append({
                    "archive_path": name, "bytes": len(data),
                    "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
            archive.writestr("manifest.json", json.dumps({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "figures_by_notebook": {LABELS[key]: value for key, value in counts.items()},
                "sources": {key: str(value) for key, value in sources.items()},
                "files": manifest,
            }, ensure_ascii=False, indent=2))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"ZIP creato: {destination}")
    for key, count in counts.items():
        print(f"  {LABELS[key]}: {count} immagini")
    print(f"Dimensione: {destination.stat().st_size / 1024**2:.2f} MiB")
    return destination


def download(args):
    for executable in ("ssh", "scp"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Comando {executable} non disponibile: installare il client OpenSSH.")
    if args.host.startswith("-") or any(c.isspace() for c in args.host):
        raise ValueError("Host SSH non valido: usare un alias SSH o utente@hostname.")
    remote_root = PurePosixPath(args.remote_root)
    if not remote_root.is_absolute():
        raise ValueError("--remote-root deve essere un percorso assoluto del server.")
    remote_zip = remote_root / "outputs/updated_notebook_figures.zip"
    remote_command = [args.remote_python, "-", "pack", "--root", str(remote_root),
                      "--output", str(remote_zip)]
    for key in DEFAULT_PATHS:
        value = getattr(args, f"{key}_dir")
        if value:
            remote_command.extend([f"--{key}-dir", value])
    ssh_args = ["ssh", "-p", str(args.port)]
    scp_args = ["scp", "-P", str(args.port)]
    if args.identity:
        ssh_args.extend(["-i", args.identity])
        scp_args.extend(["-i", args.identity])
    print(f"Creo lo ZIP su {args.host}...", flush=True)
    subprocess.run(
        [*ssh_args, args.host, shlex.join(remote_command)],
        input=Path(__file__).read_bytes(), check=True,
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        subprocess.run([*scp_args, f"{args.host}:{remote_zip}", str(temporary)], check=True)
        with ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP scaricato danneggiato: {bad}")
            manifest = json.loads(archive.read("manifest.json"))
            for row in manifest["files"]:
                if hashlib.sha256(archive.read(row["archive_path"])).hexdigest() != row["sha256"]:
                    raise ValueError(f"Checksum non valido: {row['archive_path']}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Scaricato e verificato: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Crea lo ZIP sul server, senza connessioni di rete.")
    pack.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    pack.add_argument("--output", type=Path, help="Default: ROOT/outputs/updated_notebook_figures.zip")
    fetch = commands.add_parser("download", help="Crea lo ZIP via SSH e scaricalo su questo computer.")
    fetch.add_argument("--host", required=True, help="Alias SSH oppure apedalino@HOST")
    fetch.add_argument("--remote-root", default="/home/apedalino/physiq_pv")
    fetch.add_argument("--remote-python", default="python3")
    fetch.add_argument("--port", type=int, default=22)
    fetch.add_argument("--identity", help="Percorso locale della chiave SSH, se necessario.")
    fetch.add_argument("--output", default="outputs/updated_notebook_figures.zip")
    for subparser in (pack, fetch):
        for key in DEFAULT_PATHS:
            subparser.add_argument(f"--{key}-dir", help=f"Cartella output alternativa ({key}).")
    args = parser.parse_args()
    try:
        if args.command == "pack":
            build_bundle(args.root, args.output or args.root / "outputs/updated_notebook_figures.zip",
                         {key: getattr(args, f"{key}_dir") for key in DEFAULT_PATHS})
        else:
            download(args)
    except (OSError, ValueError, RuntimeError, BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Errore: {exc}\n")


if __name__ == "__main__":
    main()
