"""The best of several scores on every known event, from the outputs of check_era5_known_events.py.

For every event: the highest share of its values flagged among the scores (and which score), the median share, and
the most days with more than `--limit` % of the box flagged (from `eventi_per_giorno.csv`, run with --days-before 0).
The best is chosen event by event, after the fact: it is the upper bound of what the simpler methods reach.

    python scripts/summarize_era5_known_events_best.py --runs normal=outputs/x/normal mcd=outputs/x/mcd \
        --csv outputs/optuna27/eventi_noti_39_migliore.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(runs, limit):
    shares, days = {}, {}
    for name, folder in runs.items():
        shares[name] = pd.read_csv(folder / "eventi_noti.csv").set_index("id")["segnalati %"]
        daily = pd.read_csv(folder / "eventi_per_giorno.csv").set_index("id")
        days[name] = (daily > limit).sum(axis=1)
    shares, days = pd.DataFrame(shares), pd.DataFrame(days)
    return pd.DataFrame({"migliore %": shares.max(axis=1), "metodo": shares.idxmax(axis=1),
                         "mediana %": shares.median(axis=1), "giorni visti migliore": days.max(axis=1)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, help="NAME=DIR: output folders of check_era5_known_events.py")
    parser.add_argument("--limit", type=float, default=10, help="A day is seen with more than this % of the box flagged")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    runs = dict((item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.runs)
    table = summarize(runs, args.limit)
    print(table.round(2).to_csv())
    if args.csv is not None:
        table.to_csv(args.csv, float_format="%.4g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
