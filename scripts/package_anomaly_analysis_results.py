"""Create the compact ZIP used to review the anomaly-analysis notebooks.

The bundle contains every generated figure plus small CSV/JSON/Markdown audit
files. Raw detector scores and the multi-million-row prediction tables are
deliberately excluded. Before writing the ZIP, the script verifies that the
STGAN post-hoc, spatial comparison and threshold sweep use the PVGIS quality
filter and the clean global top-1% protocol.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


QUALITY_POLICY = "isolated_regional_solar_dropout_plus_immediate_recovery"
SDE_RUN_NAME = (
    "pvgis_stgnn_paper_faithful_gaussian_detector_mtgflow_"
    "ep60_h1-2-3-4-5-6_direct_seed1"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
REPORT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
MAX_REPORT_BYTES = 12 * 1024 * 1024
EXCLUDED_REPORT_NAMES = {
    "anomaly_scores.csv",
    "train_anomaly_scores.csv",
    "predictions.csv",
}


@dataclass(frozen=True)
class AnalysisSource:
    label: str
    path: Path
    required_figure_directories: tuple[Path, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadati mancanti: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_equal(metadata: dict[str, object], key: str, expected: object, path: Path) -> None:
    actual = metadata.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Output non quality-filtered: {path} contiene {key}={actual!r}, "
            f"atteso {expected!r}. Rieseguire il notebook corrispondente."
        )


def _validate_quality_filtered_outputs(root: Path, spatial: Path, threshold: Path) -> dict[str, object]:
    audit_path = root / "evaluation_source.json"
    audit = _read_json(audit_path)
    _require_equal(audit, "detector", "stgan", audit_path)
    _require_equal(audit, "quality_filter_policy", QUALITY_POLICY, audit_path)
    if float(audit.get("clean_top_k_percent", -1.0)) != 1.0:
        raise RuntimeError(
            f"{audit_path} non certifica il clean global top-1% STGAN."
        )
    if int(audit.get("data_quality_timestamps", 0)) <= 0:
        raise RuntimeError(
            f"{audit_path} non riporta timestamp PVGIS esclusi per qualità."
        )
    if int(audit.get("excluded_data_quality_rows", 0)) <= 0:
        raise RuntimeError(
            f"{audit_path} non riporta righe escluse per qualità."
        )
    if not (root / "pvgis_data_quality_issues.csv").is_file():
        raise FileNotFoundError(root / "pvgis_data_quality_issues.csv")

    spatial_metadata_path = spatial / "analysis_metadata.json"
    spatial_metadata = _read_json(spatial_metadata_path)
    _require_equal(
        spatial_metadata, "stgan_quality_filter_policy", QUALITY_POLICY,
        spatial_metadata_path,
    )
    if float(spatial_metadata.get("stgan_clean_top_k_percent", -1.0)) != 1.0:
        raise RuntimeError(
            f"{spatial_metadata_path} non usa il clean top-1% STGAN."
        )

    threshold_metadata_path = threshold / "analysis_metadata.json"
    threshold_metadata = _read_json(threshold_metadata_path)
    _require_equal(
        threshold_metadata, "quality_filter_policy", QUALITY_POLICY,
        threshold_metadata_path,
    )
    if int(threshold_metadata.get("excluded_quality_timestamps", 0)) <= 0:
        raise RuntimeError(
            f"{threshold_metadata_path} non riporta timestamp PVGIS esclusi."
        )
    _require_equal(
        threshold_metadata,
        "stgan_ranking",
        "global ranking recomputed after quality exclusion",
        threshold_metadata_path,
    )
    return audit


def _sources(root: Path) -> tuple[AnalysisSource, ...]:
    outputs = root / "outputs"
    mtgflow = outputs / SDE_RUN_NAME
    stgan = outputs / "sde_stgan_direct_multihorizon_seed20_quality_filtered"
    spatial = outputs / "anomaly_spatial_comparison_quality_filtered"
    threshold = outputs / "anomaly_threshold_sensitivity_t1_t6"
    return (
        AnalysisSource(
            "01_sde_mtgflow",
            mtgflow,
            (
                Path("figures"),
                Path("figures/events/t_plus_1/extreme_events"),
                Path("figures/events/t_plus_6/extreme_events"),
            ),
        ),
        AnalysisSource(
            "02_sde_stgan_quality_filtered",
            stgan,
            (
                Path("posthoc_by_horizon/t_plus_1/figures"),
                Path("posthoc_by_horizon/t_plus_6/figures"),
                Path("stgan_may08_may17_t1_t6_pipeline_style/figures"),
            ),
        ),
        AnalysisSource(
            "03_spatial_quality_filtered",
            spatial,
            (Path("figures"),),
        ),
        AnalysisSource(
            "04_threshold_quality_filtered",
            threshold,
            (Path("figures"),),
        ),
    )


def _validate_figure_directories(sources: tuple[AnalysisSource, ...]) -> None:
    errors: list[str] = []
    for source in sources:
        if not source.path.is_dir():
            errors.append(f"cartella mancante: {source.path}")
            continue
        for relative in source.required_figure_directories:
            directory = source.path / relative
            images = [
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ] if directory.is_dir() else []
            if not images:
                errors.append(f"nessuna figura in: {directory}")
    if errors:
        detail = "\n - ".join(errors)
        raise RuntimeError(
            "Output incompleti. Eseguire tutti i notebook richiesti prima dello ZIP:\n"
            f" - {detail}"
        )


def _selected_files(source: AnalysisSource) -> tuple[list[Path], list[Path]]:
    selected: list[Path] = []
    skipped_large: list[Path] = []
    for path in sorted(source.path.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            selected.append(path)
            continue
        if suffix not in REPORT_SUFFIXES or path.name in EXCLUDED_REPORT_NAMES:
            continue
        if path.stat().st_size > MAX_REPORT_BYTES:
            skipped_large.append(path)
            continue
        selected.append(path)
    return selected, skipped_large


def _manifest_csv(rows: list[dict[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("archive_path", "source_path", "kind", "bytes", "sha256"),
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_bundle(root: Path | None = None, destination: Path | None = None) -> Path:
    root = (root or _repo_root()).resolve()
    destination = (
        destination or root / "physiq_pv_analysis_quality_filtered.zip"
    ).resolve()
    sources = _sources(root)
    source_by_label = {source.label: source for source in sources}
    audit = _validate_quality_filtered_outputs(
        source_by_label["02_sde_stgan_quality_filtered"].path,
        source_by_label["03_spatial_quality_filtered"].path,
        source_by_label["04_threshold_quality_filtered"].path,
    )
    _validate_figure_directories(sources)

    manifest: list[dict[str, object]] = []
    skipped_large: list[str] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for source in sources:
                files, skipped = _selected_files(source)
                skipped_large.extend(str(path) for path in skipped)
                for path in files:
                    relative = path.relative_to(source.path).as_posix()
                    archive_path = f"results/{source.label}/{relative}"
                    data = path.read_bytes()
                    archive.writestr(archive_path, data)
                    manifest.append(
                        {
                            "archive_path": archive_path,
                            "source_path": str(path),
                            "kind": "figure" if path.suffix.lower() in IMAGE_SUFFIXES else "report",
                            "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )

            figure_count = sum(row["kind"] == "figure" for row in manifest)
            report_count = sum(row["kind"] == "report" for row in manifest)
            bundle_metadata = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "quality_filter_policy": QUALITY_POLICY,
                "stgan_clean_top_k_percent": 1.0,
                "data_quality_timestamps": audit["data_quality_timestamps"],
                "excluded_data_quality_rows": audit["excluded_data_quality_rows"],
                "figure_count": figure_count,
                "report_count": report_count,
                "excluded_raw_tables": sorted(EXCLUDED_REPORT_NAMES),
                "excluded_large_reports": skipped_large,
            }
            archive.writestr(
                "bundle_metadata.json",
                json.dumps(bundle_metadata, indent=2, ensure_ascii=False),
            )
            archive.writestr("bundle_manifest.csv", _manifest_csv(manifest))
            archive.writestr(
                "README.txt",
                "Bundle delle analisi MTGFlow/STGAN.\n"
                "STGAN e i confronti usano il filtro PVGIS per dropout solari "
                "isolati e recuperi immediati, con ranking globale ricalcolato "
                "e soglia clean top-1%.\n"
                "Le tabelle grezze predictions/anomaly_scores non sono incluse.\n",
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"ZIP creato: {destination}")
    print(f"Figure: {figure_count} | report compatti: {report_count}")
    print(
        "Filtro verificato: "
        f"{audit['data_quality_timestamps']} timestamp, "
        f"{audit['excluded_data_quality_rows']} righe escluse."
    )
    return destination


if __name__ == "__main__":
    build_bundle()
