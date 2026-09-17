"""Preview/delete known post-hoc outputs while preserving original model runs.

Standard library only. Default root: /home/apedalino/physiq_pv/outputs.
Run with notebooks and training stopped. --apply deletes files, including plots.
Small provenance files and reference statistics are retained for regeneration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat


DEFAULT_ROOT = "/home/apedalino/physiq_pv/outputs"
# Exact names verified against notebook configuration and evaluation writers.
POSTHOC_ROOTS = {
    "sde_stgan_direct_multihorizon_seed20": "valutazione SDE con label STGAN",
    "sde_stgan_direct_multihorizon_seed20_quality_filtered": "posthoc STGAN quality filtered",
    "sde_stgan_cnn_quality_filtered": "posthoc STGAN CNN",
    "sde_stgan_classic_quality_filtered": "posthoc STGAN classico",
    "stgan_input_target_cases_t1_t6": "casi input/target STGAN",
    "stgan_cnn_input_target_cases_t1_t6": "casi input/target STGAN CNN",
    "anomaly_spatial_comparison": "confronto spaziale",
    "anomaly_spatial_comparison_quality_filtered": "confronto spaziale quality filtered",
    "anomaly_threshold_sensitivity_t1_t6": "sensibilita soglie",
    "anomaly_threshold_sensitivity_cnn": "sensibilita soglie CNN",
    "anomaly_threshold_sensitivity_cnn_t1_t6": "sensibilita soglie CNN",
    "mtgflow_threshold_sensitivity": "sensibilita soglie MTGFlow",
    "mtgflow_spatiotemporal_2019": "mappe e analisi temporali MTGFlow",
    "anomaly_analysis_summary": "riepilogo analisi",
}
# Only these verified subdirectories inside prediction runs are post-processing.
NESTED_POSTHOC = (
    "posthoc_by_horizon", "score_timeline", "figures/direct_multihorizon",
    "figures/april_dust_event", "figures/april_dust_hourly", "figures/anomaly_driver",
    "figures/events",
)
KEEP_NAMES = {
    "evaluation_source.json", "analysis_metadata.json", "threshold_analysis_metadata.json",
    "summary_metadata.json", "reference_production_peaks.csv", "training_iqr_by_location.csv",
}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".joblib", ".h5"}
MODEL_DATA_NAMES = {
    "anomaly_scores.csv", "train_anomaly_scores.csv", "entity_anomaly_scores.csv",
    "train_scores.csv", "test_scores.csv", "manifest.csv",
}
EVALUATION_MODES = {"detector_evaluation_only", "pointwise_detector_evaluation_only"}
SOURCE_KEYS = ("source_predictions", "detector_scores", "training_detector_scores", "pvgis_quality_source")


def role(name: str) -> str:
    if name in POSTHOC_ROOTS:
        return "POSTHOC: " + POSTHOC_ROOTS[name]
    if name.startswith("pvgis_stgnn_"):
        return "PREDIZIONE: conserva modello, predizioni e metriche originali"
    if name in {"pvgis_mtgflow", "pvgis_stgan", "pvgis_stgan_cnn"}:
        return "ANOMALY DETECTION: conserva run, score, locations, prepared e cache"
    if name.startswith("pvgis_anomaly_"):
        return "CLIMATOLOGIA: conserva score usati dalle analisi"
    return "NON CLASSIFICATO: conserva"


def checked_path(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"Link simbolico/junction: {current}")
    if path.resolve() != path:
        raise ValueError(f"Percorso non canonico: {path}")


def stamp(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"File non regolare: {path}")
    return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino


def tree_snapshot(root: Path, directory: Path) -> dict[Path, tuple]:
    checked_path(root, directory)
    result = {}
    def fail(error):
        raise error
    for folder, dirs, files in os.walk(directory, followlinks=False, onerror=fail):
        for name in dirs + files:
            checked_path(root, Path(folder) / name)
        for name in files:
            path = Path(folder) / name
            result[path] = stamp(path)
    return result


def validate_evaluation(root: Path, predictions: Path) -> set[Path]:
    audit = predictions.parent / "evaluation_source.json"
    metadata = json.loads(audit.read_text(encoding="utf-8"))
    if metadata.get("mode") not in EVALUATION_MODES:
        raise ValueError(f"Nessuna certificazione di valutazione derivata: {audit}")
    if not metadata.get("source_predictions") or not metadata.get("detector_scores"):
        raise ValueError(f"Sorgenti non documentate: {audit}")
    sources = set()
    for key in SOURCE_KEYS:
        raw = metadata.get(key)
        if raw is None:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = root.parent / path
        path = path.resolve(strict=True)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Sorgente mancante/vuota: {path}")
        if path == predictions:
            raise ValueError(f"Il file e la propria sorgente: {path}")
        sources.add(path)
    return sources


def scan(root: Path):
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Non e una directory: {root}")
    classifications = []
    candidates = []
    blocked = []
    for top in sorted(root.iterdir()):
        classifications.append((top.name, role(top.name)))
        if top.name in POSTHOC_ROOTS:
            candidates.append(top)
        elif top.name.startswith("pvgis_stgnn_") and top.is_dir() and not top.is_symlink():
            candidates.extend(top / name for name in NESTED_POSTHOC if (top / name).exists())
    plans = []
    sources = set()
    for directory in candidates:
        try:
            snapshot = tree_snapshot(root, directory)
            if not directory.is_dir():
                raise ValueError(f"Non e una directory: {directory}")
            model_files = [p for p in snapshot if p.suffix.lower() in MODEL_SUFFIXES or p.name in MODEL_DATA_NAMES]
            if model_files:
                raise ValueError(f"Possibili output di modello, conservo la cartella: {model_files[0]}")
            for path in snapshot:
                if path.name == "predictions.csv":
                    sources.update(validate_evaluation(root, path))
            plans.append({"directory": directory, "snapshot": snapshot})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            blocked.append((directory, str(exc)))
    # Never remove a documented input, even when it is in another posthoc tree.
    for plan in plans:
        plan["delete"] = {
            path: record for path, record in plan["snapshot"].items()
            if path.name not in KEEP_NAMES and path not in sources
        }
    return root, classifications, plans, blocked


def apply_plans(root: Path, plans) -> tuple[int, int]:
    # Validate the entire selection before deleting any file.
    for plan in plans:
        directory = plan["directory"]
        relative = directory.relative_to(root)
        allowed = (
            len(relative.parts) == 1 and relative.parts[0] in POSTHOC_ROOTS
        ) or (
            relative.parts[0].startswith("pvgis_stgnn_")
            and Path(*relative.parts[1:]).as_posix() in NESTED_POSTHOC
        )
        if not allowed or not relative.parts:
            raise ValueError(f"Cartella non autorizzata: {directory}")
        if tree_snapshot(root, directory) != plan["snapshot"]:
            raise ValueError(f"Cartella cambiata dopo la scansione: {directory}")
        for path in plan["snapshot"]:
            if path.name == "predictions.csv":
                validate_evaluation(root, path)
        for path, record in plan["delete"].items():
            if path not in plan["snapshot"] or path.name in KEEP_NAMES or record != plan["snapshot"][path]:
                raise ValueError(f"Selezione non valida: {path}")
    removed = total = 0
    for plan in plans:
        for path, record in plan["delete"].items():
            checked_path(root, path)
            if stamp(path) != record:
                raise ValueError(f"File cambiato durante la pulizia: {path}")
            path.unlink()
            removed += 1
            total += record[0]
    # Keep directories and small provenance files; avoid recursive directory removal.
    return removed, total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--apply", action="store_true", help="Elimina gli output posthoc selezionati, incluse le figure")
    parser.add_argument("--verbose", action="store_true", help="Mostra ogni file selezionato")
    args = parser.parse_args(argv)
    try:
        root, classifications, plans, blocked = scan(args.root)
        print(f"{'PULIZIA' if args.apply else 'ANTEPRIMA (nessuna cancellazione)'}: {root}")
        for name, category in classifications:
            print(f"{category}\n  {name}")
        count = total = 0
        for plan in plans:
            size = sum(record[0] for record in plan["delete"].values())
            count += len(plan["delete"])
            total += size
            print(f"POSTHOC ELIMINABILE: {size / 1024**3:.3f} GiB | {len(plan['delete'])} file | {plan['directory'].relative_to(root)}")
            if args.verbose:
                for path in plan["delete"]:
                    print(f"  {path.relative_to(root)}")
        for directory, reason in blocked:
            print(f"CONSERVATO PER VERIFICA: {directory.relative_to(root)} — {reason}")
        print(f"Selezionati: {count} file, {total / 1024**3:.3f} GiB (dimensioni logiche).")
        print("Conservati: modelli, predizioni originali, anomaly score, prepared/cache, metadata e statistiche di riferimento.")
        if args.apply:
            count, total = apply_plans(root, plans)
            print(f"Eliminati: {count} file, {total / 1024**3:.3f} GiB (dimensioni logiche).")
        else:
            print("Con notebook/run fermi, aggiungi --apply per eliminare gli output posthoc, incluse le immagini.")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERRORE (eventuali cancellazioni gia completate non vengono annullate): {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
