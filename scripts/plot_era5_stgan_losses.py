"""Plot the training losses recorded by ERA5 STGAN runs.

The current ERA5 protocol does not compute held-out validation losses. This
script never labels training or test scores as validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROGRESS = re.compile(
    r"\[stgan\] epoch=(\d+)/(\d+) batch=(\d+)/(\d+) "
    r"D_mean=([\d.eE+-]+) G_mean=([\d.eE+-]+)"
)
LOSSES = ("generator_loss", "discriminator_loss")


def read_progress(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = PROGRESS.search(line)
            if match is None:
                continue
            epoch, _, batch, batches, discriminator, generator = match.groups()
            rows.append({
                "epoch_position": int(epoch) - 1 + int(batch) / int(batches),
                "generator_loss": float(generator),
                "discriminator_loss": float(discriminator),
            })
    return pd.DataFrame(rows)


def read_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "training_history.csv"
    if not path.is_file():
        return pd.DataFrame()
    history = pd.read_csv(path)
    required = {"epoch", *LOSSES}
    if not required.issubset(history.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return history


def plot_runs(runs: list[tuple[str, Path]], logs: dict[str, Path], output: Path) -> None:
    if len({label for label, _ in runs}) != len(runs):
        raise ValueError("Run labels must be unique")
    if set(logs) - {label for label, _ in runs}:
        raise ValueError("Every --log label must match a --run label")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for label, run_dir in runs:
        history = read_history(run_dir)
        log_path = logs.get(label)
        progress = read_progress(log_path) if log_path is not None else pd.DataFrame()
        if history.empty and progress.empty:
            raise ValueError(f"No training losses found for {label!r} in {run_dir} or {log_path}")
        for axis, column, title in zip(axes, LOSSES, ("Generator: 500 × MSE + BCE", "Discriminator: BCE")):
            if not progress.empty:
                line, = axis.plot(progress.epoch_position, progress[column], label=label, linewidth=1.5)
                if not history.empty:
                    axis.scatter(history.epoch, history[column], color=line.get_color(),
                                 s=22, marker="o", zorder=3)
            else:
                axis.plot(history.epoch, history[column], marker="o", label=label)
            axis.set_title(title)
            axis.set_ylabel("Loss media di training")
            axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Epoca (posizione frazionaria per i log intermedi)")
    axes[0].legend()
    figure.suptitle("ERA5 STGAN — curve di training")
    figure.supxlabel("Nessuna loss di validation è stata registrata da queste run.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    print(f"Saved {output} and {output.with_suffix('.pdf')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "RUN_DIR"), required=True,
                        help="Repeat for each run; reads RUN_DIR/training_history.csv when available")
    parser.add_argument("--log", action="append", nargs=2, metavar=("LABEL", "LOG_FILE"), default=[],
                        help="Optional progress log; label must match a --run label")
    parser.add_argument("--output", type=Path, required=True, help="PNG path; also saves a PDF")
    args = parser.parse_args(argv)
    plot_runs([(label, Path(path)) for label, path in args.run],
              {label: Path(path) for label, path in args.log}, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
