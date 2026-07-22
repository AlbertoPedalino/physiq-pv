"""Join forecast predictions to one detector CSV and report stratified metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from physiq_pv.anomaly_detection.evaluation import (
    attach_detector_scores,
    forecast_metrics_by_detection,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--detector-scores", required=True)
    parser.add_argument("--method")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    predictions = pd.read_csv(args.predictions)
    scores = pd.read_csv(args.detector_scores)
    joined = attach_detector_scores(predictions, scores, method=args.method)
    metrics = forecast_metrics_by_detection(joined)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "predictions_with_detector_scores.csv", index=False)
    metrics.to_csv(out / "forecast_metrics_by_detection.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
