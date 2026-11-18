"""The same concordant cube with another minimum size of the studied clusters, for a sensitivity check.

build_era5_concordant_cube.py marks a cluster small when it holds fewer than `--min-values` values, analysed when it
lasts at most `--max-hours` and long otherwise; the vote, the flags and the clusters themselves do not depend on that
minimum. This script writes OUTPUT (new or empty) with the votes and the flags of CUBE linked, not copied, and with
clusters.csv and summary.json written again for `--min-values`: only the kind of the clusters, the counts per kind and
the minimum in the selection change. The checks of the cube (match, boxes, lead, excess) then read OUTPUT as any cube.

    python scripts/set_era5_cube_min_values.py --cube outputs/concordate_5pc_regola_tesi \\
        --output outputs/concordate_5pc_min64 --min-values 64
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
    parser.add_argument("--min-values", type=int, required=True)
    args = parser.parse_args(argv)
    if args.min_values < 1:
        parser.error("--min-values must be at least 1")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Use a new/empty output directory: {args.output}")
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(args.cube / "clusters.csv")
    max_hours = summary["selection"]["max_hours"]
    table = relabel(table, args.min_values, max_hours)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("votes", "flags"):
        os.symlink((args.cube / name).resolve(), args.output / name, target_is_directory=True)
    table.to_csv(args.output / "clusters.csv", index=False)
    summary["selection"] = {**summary["selection"], "min_values": args.min_values}
    summary["clusters"] = kind_counts(table)
    summary["source_cube"] = str(args.cube)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    counts = pd.DataFrame(summary["clusters"]).T
    print(f"[min-values] {args.cube} -> {args.output}, at least {args.min_values} values, at most {max_hours:g} h "
          f"for the analysed\n{counts.to_string()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
