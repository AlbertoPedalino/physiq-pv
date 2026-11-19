"""The same concordant cube with another minimum size of the studied clusters, or another split between short and
long clusters, for a sensitivity check.

build_era5_concordant_cube.py marks a cluster small when it holds fewer than `--min-values` values, analysed when it
lasts at most `--max-hours` and long otherwise; the vote, the flags and the clusters themselves do not depend on that
minimum or on that duration. This script writes OUTPUT (new or empty) with the votes and the flags of CUBE linked, not
copied, and with clusters.csv and summary.json written again for `--min-values` and `--max-hours` (each by default that
of CUBE): only the kind of the clusters, the counts per kind and the two values in the selection change. The checks of the cube (match, boxes, lead, excess) then read OUTPUT as any cube.

    python scripts/set_era5_cube_min_values.py --cube outputs/concordate_5pc_regola_tesi \\
        --output outputs/concordate_5pc_min64 --min-values 64
    python scripts/set_era5_cube_min_values.py --cube outputs/concordate_5pc_regola_tesi \
        --output outputs/concordate_5pc_96h --max-hours 96
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def relabel(table, min_values, max_hours):
    """The kinds of the clusters for another minimum size, as describe_clusters of the cube sets them."""
    table = table.copy()
    table["kind"] = np.where(table["values"] < min_values, "small",
                             np.where(table["hours"] <= max_hours, "analysed", "long"))
    return table


def kind_counts(table):
    total = float(table["values"].sum())
    return {kind: {"clusters": int((table["kind"] == kind).sum()),
                   "values_percent": 100 * float(table.loc[table["kind"] == kind, "values"].sum()) / max(total, 1.0)}
            for kind in ("analysed", "small", "long")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--min-values", type=int, help="Minimum values of a studied cluster (default: that of CUBE)")
    parser.add_argument("--max-hours", type=float,
                        help="Longest duration of a short (analysed) cluster, hours (default: that of CUBE)")
    args = parser.parse_args(argv)
    if args.min_values is None and args.max_hours is None:
        parser.error("give --min-values, --max-hours or both")
    if args.min_values is not None and args.min_values < 1:
        parser.error("--min-values must be at least 1")
    if args.max_hours is not None and args.max_hours < 0:
        parser.error("--max-hours must not be negative")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Use a new/empty output directory: {args.output}")
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(args.cube / "clusters.csv")
    min_values = summary["selection"]["min_values"] if args.min_values is None else args.min_values
    max_hours = summary["selection"]["max_hours"] if args.max_hours is None else args.max_hours
    table = relabel(table, min_values, max_hours)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("votes", "flags"):
        os.symlink((args.cube / name).resolve(), args.output / name, target_is_directory=True)
    table.to_csv(args.output / "clusters.csv", index=False)
    summary["selection"] = {**summary["selection"], "min_values": min_values, "max_hours": max_hours}
    summary["clusters"] = kind_counts(table)
    summary["source_cube"] = str(args.cube)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    counts = pd.DataFrame(summary["clusters"]).T
    print(f"[min-values] {args.cube} -> {args.output}, at least {min_values} values, at most {max_hours:g} h "
          f"for the analysed\n{counts.to_string()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
