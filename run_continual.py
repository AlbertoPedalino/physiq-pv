"""
Replay-based continual adaptation — single entry point.

Runs sanity checks, then the full pipeline on real Piedmont 2019 data.
Prints metrics at each step.

Usage:
    python run_continual.py                # full run on real data
    python run_continual.py --debug        # smoke test (5 plants, 2 windows)
    python run_continual.py --skip-tests   # skip sanity checks
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run_tests() -> bool:
    print("=" * 60)
    print("  Sanity checks")
    print("=" * 60)
    from tests.test_replay_continual import (
        test_buffer_add_and_size,
        test_buffer_sample_shapes,
        test_buffer_capacity_enforced,
        test_feature_shape_and_m_channels,
        test_temporal_stream_skips_gaps,
    )

    checks = [
        test_buffer_add_and_size,
        test_buffer_sample_shapes,
        test_buffer_capacity_enforced,
        test_feature_shape_and_m_channels,
        test_temporal_stream_skips_gaps,
    ]
    results = [fn() for fn in checks]
    ok = all(results)
    print(f"\n  {sum(results)}/{len(results)} passed.\n")
    return ok


def _run_pipeline(
    debug: bool = False,
    seed: int = 42,
    run_name: str | None = None,
) -> None:
    from physiq_pv.continual.train_replay_continual import main as cl_main

    label = "debug smoke test" if debug else "full run (Piedmont 2019)"
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)

    argv = [
        "--data-mode", "real",
        "--initial-train-start", "2019-03-01",
        "--initial-train-end", "2019-05-31",
        "--window-months", "1",
        "--replay-buffer-size", "5000",
        "--replay-batch-size", "8",
        "--replay-loss-weight", "1.0",
        "--initial-epochs", "5",
        "--update-epochs", "1",
        "--seed", str(seed),
    ]

    if debug:
        argv += ["--debug"]
        run_name = run_name or "smoke_test"
    else:
        run_name = run_name or "full_2019"

    argv += ["--run-name", run_name]

    sys.argv = ["run_continual"] + argv
    cl_main()

    out_dir = Path("outputs/continual_replay") / run_name
    print(f"\n--- {run_name} metrics ---")
    csv_path = out_dir / "metrics_per_window.csv"
    if csv_path.exists():
        print(csv_path.read_text())

    summary_path = out_dir / "final_summary.json"
    if summary_path.exists():
        print(summary_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continual replay pipeline")
    parser.add_argument("--debug", action="store_true", help="Smoke test only (5 plants, 2 windows)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip sanity checks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if not args.skip_tests:
        ok = _run_tests()
        if not ok:
            print("[ABORT] sanity checks failed")
            sys.exit(1)

    if args.debug:
        _run_pipeline(debug=True, seed=args.seed, run_name=args.run_name)
    else:
        _run_pipeline(debug=True, seed=args.seed, run_name="smoke_test")
        print()
        _run_pipeline(debug=False, seed=args.seed, run_name=args.run_name)


if __name__ == "__main__":
    main()
