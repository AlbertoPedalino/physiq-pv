"""Do the other scores flag the striking days of the model too?

Reads the striking days of a score (`--cases-dir`, the output of list_era5_striking_days.py: casi.csv and, in
casi.npz, the largest patch of every day on the lattice) and, for every other score of `--by` (NAME=DIR, each at its
highest `--top-percent` with the threshold of the validation years when its run names one), at the timestamp of the
day:
- the share of the cells of the patch it flags at that timestamp, and at least once within one step before or after;
- the cells of its own largest patch at that timestamp (touching cells, diagonals included), as a share of the domain,
  and the share of the patch of the model inside that patch of its own.
"Any" is the union of all the scores of `--by`: a cell is flagged when one of them flags it.

    python scripts/check_era5_striking_days_others.py --cases-dir outputs/optuna27/giorni_eclatanti \
        --by normal=outputs/confronti_good2/climatology/test/normal mcd=outputs/confronti_good2/climatology/test/mcd \
        --csv outputs/optuna27/giorni_eclatanti_altri.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_dust_lead import score_file, score_limit

STRUCTURE = np.ones((3, 3), dtype=bool)


def flags_of(run, top_percent, timestamps):
    """Flags [timestamp, cell] of the score of `run` at `timestamps` (each with the one before and after: [t, 3, cell]),
    and the grid of the run."""
    manifest = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    own = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    scores = np.load(run / score_file(manifest, "combined"), mmap_mode="r")
    try:
        limit = score_limit(scores, top_percent)
        rows = own.get_indexer(timestamps)
        if (rows < 0).any():
            raise ValueError(f"{run} has not scored every striking day")
        around = np.stack([np.clip(rows + shift, 0, len(own) - 1) for shift in (-1, 0, 1)], axis=1)
        with np.errstate(invalid="ignore"):
            flagged = np.asarray(scores[around.ravel()], dtype=np.float32) > limit
    finally:
        scores._mmap.close()
    return flagged.reshape(len(timestamps), 3, -1), manifest["grid"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases-dir", type=Path, required=True, help="Output of list_era5_striking_days.py")
    parser.add_argument("--by", nargs="+", required=True, help="NAME=DIR of the other scores")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    cases = pd.read_csv(args.cases_dir / "casi.csv")
    patches = np.load(args.cases_dir / "casi.npz")
    timestamps = pd.DatetimeIndex(pd.to_datetime(cases["istante"]))
    others = [item.split("=", 1) for item in args.by]
    rows_out, union = [], None
    for name, run in others:
        flagged, grid = flags_of(Path(run), args.top_percent, timestamps)
        union = flagged if union is None else union | flagged
        rows_out += read(name, flagged, grid, cases, patches)
    rows_out += read("any", union, grid, cases, patches)
    table = pd.DataFrame(rows_out)
    pd.set_option("display.width", 200)
    print(f"[striking-days-others] {len(cases)} striking days of {args.cases_dir} | each other score at its highest "
          f"{args.top_percent:g}% | patch: the largest patch of the model on that day")
    for column, title in (("patch flagged %", "% of the patch of the model the other score flags at that timestamp"),
                          ("patch flagged within 3 h %", "the same, at least once within 3 h before or after"),
                          ("own largest patch % of domain", "its own largest patch at that timestamp, % of the domain"),
                          ("patch inside its own largest %", "% of the patch of the model inside that largest patch")):
        print(f"\n{title}")
        print(table.pivot(index="day", columns="score", values=column)
              [[name for name, _ in others] + ["any"]].round(0).to_string())
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


def read(name, flagged, grid, cases, patches):
    """One row per striking day for the score `name`: flags [day, 3, cell] on the grid of the runs."""
    rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
    out = []
    for index, case in enumerate(cases.itertuples()):
        patch = patches[f"macchia_{case.rango:02d}"]
        lattice = np.zeros(patch.shape, dtype=bool)
        lattice[rows, cols] = flagged[index, 1]
        near = np.zeros(patch.shape, dtype=bool)
        near[rows, cols] = flagged[index].any(axis=0)
        labels, count = ndimage.label(lattice, structure=STRUCTURE)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        largest = labels == sizes.argmax() if count else np.zeros_like(lattice)
        cells = max(int(patch.sum()), 1)
        out.append({"day": f"{case.rango}. {case.giorno}", "score": name,
                    "patch flagged %": 100 * (lattice & patch).sum() / cells,
                    "patch flagged within 3 h %": 100 * (near & patch).sum() / cells,
                    "own largest patch % of domain": 100 * largest.sum() / len(rows),
                    "patch inside its own largest %": 100 * (largest & patch).sum() / cells})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
