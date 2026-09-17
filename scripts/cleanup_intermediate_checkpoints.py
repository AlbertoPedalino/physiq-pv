"""Remove recognized epoch checkpoints, keeping final/best models and all results.

Standard library only. Preview by default; --apply enables deletion.
Run with training stopped. No model files are deserialized.
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import re
import stat
import time


DEFAULT_ROOT = "/home/apedalino/physiq_pv/outputs"
EPOCH_NAME = re.compile(
    r"(?:(?:checkpoint|model)[_-])?epoch[_=-]?\d+"
    r"(?:[_-]step[=_-]?\d+)?\.(pt|pth|ckpt)", re.IGNORECASE
)
FINAL_STEMS = {
    "best", "last", "final", "checkpoint", "model", "best_model", "model_best",
    "final_model", "model_final", "best_checkpoint", "checkpoint_best",
    "final_checkpoint", "checkpoint_final", "last_model", "last_checkpoint",
}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def fingerprint(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Non è un file regolare: {path}")
    return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino


def checked_directory(root: Path, directory: Path) -> None:
    """Reject symlinks, including ancestors replaced since the scan."""
    relative = directory.relative_to(root)
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ValueError(f"Directory non regolare o link simbolico: {current}")
    if directory.resolve() != directory:
        raise ValueError(f"Percorso modificato: {directory}")


def checkpoint_snapshot(directory: Path) -> dict[str, tuple[int, int, int, int]]:
    return {
        path.name: fingerprint(path)
        for path in directory.iterdir()
        if path.suffix.lower() in CHECKPOINT_SUFFIXES and not path.is_symlink()
        and path.is_file()
    }


def eligible(snapshot, cutoff_ns):
    intermediates = [name for name in snapshot if EPOCH_NAME.fullmatch(name)]
    if not intermediates:
        return [], ""
    # A recent checkpoint in the same directory may indicate an active run.
    if any(record[1] > cutoff_ns for record in snapshot.values()):
        return [], "checkpoint recenti (possibile run attiva)"
    finals = {
        Path(name).suffix.lower() for name, record in snapshot.items()
        if Path(name).stem.lower() in FINAL_STEMS and record[0] > 0
    }
    candidates = [name for name in intermediates if Path(name).suffix.lower() in finals]
    return sorted(candidates), (
        "checkpoint finale/best mancante o vuoto" if len(candidates) < len(intermediates) else ""
    )


def scan(root: Path, min_age_hours: float):
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Non è una directory: {root}")
    cutoff_ns = int((time.time() - min_age_hours * 3600) * 1_000_000_000)
    groups = []
    skipped = Counter()
    unknown_count = 0
    unknown_bytes = 0
    errors = []
    for folder, dirs, _ in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
        directory = Path(folder)
        dirs[:] = [name for name in dirs if not (directory / name).is_symlink()]
        try:
            checked_directory(root, directory)
            snapshot = checkpoint_snapshot(directory)
            candidates, reason = eligible(snapshot, cutoff_ns)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if reason:
            skipped[reason] += 1
        for name, record in snapshot.items():
            if not EPOCH_NAME.fullmatch(name) and Path(name).stem.lower() not in FINAL_STEMS:
                unknown_count += 1
                unknown_bytes += record[0]
        if candidates:
            groups.append({"directory": directory, "snapshot": snapshot, "candidates": candidates})
    return root, groups, skipped, (unknown_count, unknown_bytes), errors


def apply_group(root: Path, group) -> tuple[int, int, list[str]]:
    directory = group["directory"]
    deleted_count = deleted_bytes = 0
    errors = []
    try:
        checked_directory(root, directory)
        if checkpoint_snapshot(directory) != group["snapshot"]:
            raise ValueError(f"Checkpoint cambiati dopo la scansione, cartella saltata: {directory}")
        for name in group["candidates"]:
            checked_directory(root, directory)
            path = directory / name
            if fingerprint(path) != group["snapshot"][name]:
                raise ValueError(f"File cambiato dopo la scansione: {path}")
            # Recheck a preserved model before each deletion.
            final_exists = any(
                Path(final).stem.lower() in FINAL_STEMS
                and Path(final).suffix.lower() == path.suffix.lower()
                and record[0] > 0
                and fingerprint(directory / final) == record
                for final, record in group["snapshot"].items()
            )
            if not final_exists:
                raise ValueError(f"Checkpoint finale non più disponibile: {directory}")
            path.unlink()
            deleted_count += 1
            deleted_bytes += group["snapshot"][name][0]
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return deleted_count, deleted_bytes, errors


def size_text(size: int) -> str:
    return f"{size / 1024 ** 3:.3f} GiB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT), help="Directory outputs")
    parser.add_argument("--apply", action="store_true", help="Elimina gli intermedi; default: sola anteprima")
    parser.add_argument("--min-age-hours", type=float, default=24,
                        help="Salta cartelle con checkpoint modificati nelle ultime N ore (default: 24)")
    parser.add_argument("--verbose", action="store_true", help="Mostra ogni file candidato")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_age_hours) or args.min_age_hours < 0:
        parser.error("--min-age-hours deve essere un numero finito >= 0")
    try:
        root, groups, skipped, unknown, errors = scan(args.root, args.min_age_hours)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"{'ELIMINAZIONE' if args.apply else 'ANTEPRIMA (nessun file eliminato)'}: {root}")
    count = total = 0
    for group in sorted(groups, key=lambda g: sum(g["snapshot"][n][0] for n in g["candidates"]), reverse=True):
        names = group["candidates"]
        size = sum(group["snapshot"][name][0] for name in names)
        count += len(names)
        total += size
        print(f"  {size_text(size):>12} | {len(names):5d} intermedi | {group['directory'].relative_to(root)}")
        if args.verbose:
            for name in names:
                print(f"    {name}")
    print(f"Candidati: {count} file, {size_text(total)} (dimensioni logiche).")
    print("Conservati: modelli finali/best/last, immagini, risultati e tutti i nomi non riconosciuti.")
    for reason, folders in skipped.items():
        print(f"Cartelle con intermedi conservati: {folders} — {reason}.")
    if unknown[0]:
        print(f"Checkpoint con nomi non riconosciuti conservati: {unknown[0]} ({size_text(unknown[1])}).")

    if args.apply:
        removed = removed_bytes = 0
        for group in groups:
            n, size, group_errors = apply_group(root, group)
            removed += n
            removed_bytes += size
            errors.extend(group_errors)
        print(f"Eliminati: {removed} file, {size_text(removed_bytes)} (dimensioni logiche).")
    else:
        print("Per eliminare i candidati, ripeti il comando aggiungendo --apply. Ferma prima le run.")
    for error in errors:
        print(f"ERRORE: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
