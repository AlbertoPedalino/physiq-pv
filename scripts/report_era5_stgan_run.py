"""Print what is needed to judge a finished ERA5 STGAN run, read from its own files.

- the training history per epoch: the terms of the two losses, D's mean outputs and the validation
  monitoring;
- from the training log, the range of the progress-line losses in every epoch and the largest jump
  between two consecutive lines: a sudden change shows here even when the epoch mean hides it;
- the Monte Carlo spread of the score by score band, in absolute value and as a share of the score;
- for every `--events-dir`, the largest and the most intense events with their peak.

Nothing is computed with the model: for how each checkpoint treats the highest scores see
`analyze_era5_stgan_checkpoints.py`.

    python scripts/report_era5_stgan_run.py --run-dir outputs/run --log train.log \
        --events-dir outputs/run_events_top5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

HISTORY_COLUMNS = ("epoch", "seconds", "generator_reconstruction_loss", "generator_adversarial_loss", "discriminator_loss",
                   "discriminator_real_mean", "discriminator_fake_mean", "validation_score_spearman",
                   "validation_pca_mmd_mean", "validation_pca_mmd_rolling_mean")
PROGRESS = re.compile(r"epoch=(\d+)/(\d+) batch=(\d+)/(\d+) D_mean=([\d.eE+-]+) G_mean=([\d.eE+-]+)")
BANDS = (0, 50, 90, 95, 99, 99.9, 100)  # Percentiles of the score.
SAMPLE_STRIDE = 50  # One timestamp in fifty is enough for the spread by band.


def history_table(run_dir):
    history = pd.read_csv(Path(run_dir) / "training_history.csv")
    return history[[column for column in HISTORY_COLUMNS if column in history.columns]]


def log_summary(log_path):
    """Per epoch: range of D_mean and G_mean over the progress lines and the largest jump of G_mean.

    The last line of an epoch can cover a handful of batches only: it is left out when it does.
    """
    epochs = {}
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROGRESS.search(line)
        if match:
            epoch, _, batch, total, d_mean, g_mean = match.groups()
            epochs.setdefault(int(epoch), []).append((int(batch), int(total), float(d_mean), float(g_mean)))
    rows = []
    for epoch, lines in sorted(epochs.items()):
        interval = lines[1][0] - lines[0][0] if len(lines) > 1 else lines[0][0]
        kept = [line for index, line in enumerate(lines)
                if index == 0 or line[0] - lines[index - 1][0] >= interval / 2]
        d_values, g_values = [line[2] for line in kept], [line[3] for line in kept]
        jumps = np.abs(np.diff(g_values)) if len(g_values) > 1 else np.zeros(1)
        rows.append({"epoch": epoch, "lines": len(kept), "D_min": min(d_values), "D_max": max(d_values),
                     "G_min": min(g_values), "G_max": max(g_values), "largest_G_jump": float(jumps.max()),
                     "at_batch": int(kept[int(jumps.argmax()) + 1][0]) if len(kept) > 1 else int(kept[0][0])})
    return pd.DataFrame(rows)


def spread_by_band(run_dir, *, stride=SAMPLE_STRIDE):
    """Mean Monte Carlo spread of the score in bands of the score, on one timestamp every `stride`."""
    scores = Path(run_dir) / "scores"
    mean = np.asarray(np.load(scores / "anomaly_mean.npy", mmap_mode="r")[::stride], dtype=np.float64).ravel()
    spread = np.asarray(np.load(scores / "anomaly_std.npy", mmap_mode="r")[::stride], dtype=np.float64).ravel()
    finite = np.isfinite(mean) & np.isfinite(spread)
    mean, spread = mean[finite], spread[finite]
    edges = np.percentile(mean, BANDS)
    rows = []
    for index in range(len(BANDS) - 1):
        inside = (mean >= edges[index]) & (mean <= edges[index + 1])
        rows.append({"score_percentiles": f"{BANDS[index]:g}-{BANDS[index + 1]:g}", "mean_score": mean[inside].mean(),
                     "mean_spread": spread[inside].mean(), "spread_over_score": spread[inside].mean() / mean[inside].mean()})
    order = lambda values: np.argsort(np.argsort(values))
    return pd.DataFrame(rows), float(np.corrcoef(order(mean), order(spread))[0, 1])


def component_spread(run_dir, *, stride=SAMPLE_STRIDE):
    """Mean Monte Carlo spread of the two components, when the run saved them."""
    result = {}
    for name in ("generator", "discriminator"):
        path = Path(run_dir) / "scores" / f"{name}_std.npy"
        if path.exists():
            values = np.asarray(np.load(path, mmap_mode="r")[::stride], dtype=np.float64)
            result[name] = (float(np.nanmean(values)), float(np.nanmax(values)))
    return result


def print_events(events_dir, *, limit=15):
    events_dir = Path(events_dir)
    metadata = json.loads((events_dir / "metadata.json").read_text(encoding="utf-8"))
    database = sqlite3.connect(events_dir / "events.sqlite")
    try:
        print(f"status={metadata.get('status')} threshold={metadata.get('global_threshold')} "
              f"top_percent={metadata['config']['top_percent']}")
        print("period: %s -> %s" % database.execute("SELECT MIN(timestamp), MAX(timestamp) FROM frames").fetchone())
        print("events by duration: one timestep %d | up to 24 h %d | 24-72 h %d | over 72 h %d" % database.execute(
            "SELECT COALESCE(SUM(duration_hours<=3), 0), COALESCE(SUM(duration_hours>3 AND duration_hours<=24), 0), "
            "COALESCE(SUM(duration_hours>24 AND duration_hours<=72), 0), COALESCE(SUM(duration_hours>72), 0) "
            "FROM events").fetchone())
        total = database.execute("SELECT SUM(cells) FROM clusters").fetchone()[0]
        if not total:
            print("no cell above the threshold survived the event filters")
            return
        for label, expression in (("month", "substr(timestamp,6,2)"), ("year", "substr(timestamp,1,4)")):
            shares = database.execute(f"SELECT {expression}, SUM(cells) FROM clusters GROUP BY 1 ORDER BY 1").fetchall()
            print(f"share of flagged cells by {label}: " + " ".join(f"{key}={100 * cells / total:.1f}%" for key, cells in shares))
        for title, order, where in (("LARGEST (cells x timesteps)", "cell_observations", ""),
                                    ("MOST INTENSE (highest score, at least 24 h)", "max_score", "WHERE duration_hours>=24")):
            print(f"\n{title}")
            for event in database.execute(
                    "SELECT event_id, start, end, duration_hours, cell_observations, mean_score, max_score, mean_uncertainty "
                    f"FROM events {where} ORDER BY {order} DESC LIMIT {int(limit)}").fetchall():
                peak = database.execute("SELECT timestamp, cells, area_km2, centroid_lat, centroid_lon FROM event_steps "
                                        "WHERE event_id=? ORDER BY cells DESC LIMIT 1", (event[0],)).fetchone()
                box = database.execute("SELECT MIN(south), MAX(north), MIN(west), MAX(east) FROM clusters WHERE event_id=?",
                                       (event[0],)).fetchone()
                spread = "" if event[7] is None else f" | spread/score {event[7] / event[5]:.2f}"
                print(f"{event[1][:13]} -> {event[2][:13]} | {event[3]:4.0f} h | {event[4]:6d} cell-steps | "
                      f"score mean {event[5]:.3f} max {event[6]:.3f}{spread} | peak {peak[0][:13]}: {peak[1]} cells, "
                      f"{peak[2] / 1000:.0f} thousand km2, centre {peak[3]:.1f}N {peak[4]:.1f}E | "
                      f"box {box[0]:.1f}-{box[1]:.1f}N {box[2]:.1f}-{box[3]:.1f}E")
    finally:
        database.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, help="Training log with the progress lines")
    parser.add_argument("--events-dir", type=Path, action="append", default=[], help="Output of `events`; repeatable")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 250)
    print("=== training history ===")
    print(history_table(args.run_dir).to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    if args.log is not None:
        print("\n=== progress lines per epoch ===")
        print(log_summary(args.log).to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    if (args.run_dir / "scores" / "anomaly_std.npy").exists():
        table, correlation = spread_by_band(args.run_dir)
        print("\n=== Monte Carlo spread of the score by score band ===")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
        print(f"rank correlation between score and spread: {correlation:.3f}")
        for name, (mean, maximum) in component_spread(args.run_dir).items():
            print(f"{name} component: mean spread {mean:.5g}, maximum {maximum:.5g}")
    for events_dir in args.events_dir:
        print(f"\n=== events: {events_dir} ===")
        print_events(events_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
