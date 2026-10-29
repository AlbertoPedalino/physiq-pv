"""Are the flags of G that the discriminator judges most real more often confirmed by the references with no model?

No label is used. The values G flags (`--g`, its highest `--top-percent`, threshold of `score_limit`) are split
into `--groups` groups of the same size by the judgement of the discriminator on them (`--judge`, NAME=DIR or
NAME=DIR@real for D's output on the observation, 1 = judged generated): the first group is the one D judges most
real. For every group the check gives the median of G and of the judge, and the share of its values that each
reference (`--refs`, flagged at its own highest `--top-percent`) flags too, then either of them and both. If the
flags D judges most real were the more trustworthy ones, they would be confirmed more often, in every run; the
median of G says whether a difference comes only from how strong the flags are.

    python scripts/check_era5_discriminator_confidence.py --g outputs/comportamento_good2/generator_clim \
        --judge D_reale=outputs/stgan_good_sweep_2_validation@real \
        --refs normale=outputs/segno_d_good2/normal 3ore=outputs/segno_d_good2/change_3h
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_dust_lead import score_limit
from scripts.check_era5_score_correlation import open_component, parse_score

CHUNK = 256  # timestamps read at a time


def opened(text):
    """NAME=DIR[@component] of a run, or NAME=FILE.npy for another map over the same timestamps and cells, such as
    the spread of the Monte Carlo draws of a scored run (scores/generator_std.npy)."""
    name, _, location = text.partition("=")
    if location.endswith(".npy"):
        values = np.load(location, mmap_mode="r")
        return name, values, [values], Path(location).parent.parent
    name, directory, component = parse_score(text)
    manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    values, maps = open_component(directory, manifest, component)
    return name, values, maps, directory


def split(values, parts):
    """The part, 0 .. parts - 1, of every value among `parts` groups of the same size."""
    return np.searchsorted(np.quantile(values, np.linspace(0, 1, parts + 1)[1:-1]), values, side="right")


def groups_table(g, judge, refs, groups, strata=1):
    """Rows of the table from the values G flags: g, judge [values]; refs {name: flags [values]}. With `strata`,
    the values are first split into that many bands of G, and the groups of the judge are taken within each band,
    so that the groups of a band flag about as strongly."""
    stratum = split(g, strata) if strata > 1 else np.zeros(len(g), dtype=int)
    group = np.empty(len(g), dtype=int)
    for band in range(strata):
        inside = stratum == band
        group[inside] = split(judge[inside], groups)
    names = list(refs)
    rows = []
    for band, index in ((band, index) for band in range(strata) for index in range(groups)):
        chosen = (stratum == band) & (group == index)
        flags = np.stack([refs[name][chosen] for name in names])
        rows.append({"band of G": band + 1, "group": index + 1, "values": int(chosen.sum()),
                     "median G": float(np.median(g[chosen])),
                     "median judge": float(np.median(judge[chosen])),
                     **{f"{name} %": 100 * float(flags[position].mean()) for position, name in enumerate(names)},
                     "either %": 100 * float(flags.any(axis=0).mean()), "both %": 100 * float(flags.all(axis=0).mean())})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--g", required=True, help="DIR of the score of G whose flags are split")
    parser.add_argument("--judge", required=True, help="NAME=DIR[@component]: the judgement of the discriminator")
    parser.add_argument("--refs", nargs="+", required=True, help="NAME=DIR[@component]: the references")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--compare", help="NAME=DIR[@component] or NAME=FILE.npy: its Spearman correlation with the "
                                          "judge over the values G flags")
    parser.add_argument("--strata", type=int, default=1,
                        help="Bands of G of the same size; the groups of the judge are taken within each band")
    args = parser.parse_args(argv)
    _, g_scores, g_maps, g_dir = opened(f"G={args.g}")
    judge_name, judge_scores, judge_maps, _ = opened(args.judge)
    references = [opened(text) for text in args.refs]
    refs = {name: scores for name, scores, _, _ in references}
    maps = [*g_maps, *judge_maps, *(part for _, _, ref_maps, _ in references for part in ref_maps)]
    compare_name, compare_scores = None, None
    if args.compare:
        compare_name, compare_scores, compare_maps, _ = opened(args.compare)
        maps += compare_maps
    timestamps = np.load(g_dir / "test_timestamps.npy")
    limits = {"G": score_limit(g_scores, args.top_percent),
              **{name: score_limit(scores, args.top_percent) for name, scores in refs.items()}}
    kept = {"G": [], "judge": [], "compare": [], **{name: [] for name in refs}}
    for start in range(0, len(timestamps), CHUNK):
        block = slice(start, start + CHUNK)
        g = np.asarray(g_scores[block], dtype=np.float32)
        judge = np.asarray(judge_scores[block], dtype=np.float32)
        ref_blocks = {name: np.asarray(scores[block], dtype=np.float32) for name, scores in refs.items()}
        chosen = (g > limits["G"]) & np.isfinite(judge) & np.all([np.isfinite(v) for v in ref_blocks.values()], axis=0)
        kept["G"].append(g[chosen])
        kept["judge"].append(judge[chosen])
        if compare_scores is not None:
            kept["compare"].append(np.asarray(compare_scores[block], dtype=np.float32)[chosen])
        for name, values in ref_blocks.items():
            kept[name].append(values[chosen] > limits[name])
        release_pages(*maps)
    kept = {name: np.concatenate(values) if values else np.empty(0) for name, values in kept.items()}
    rows = groups_table(kept["G"], kept["judge"], {name: kept[name] for name in refs}, args.groups, args.strata)
    print(f"[confidence] {len(kept['G'])} values flagged by G (highest {args.top_percent:g}%), in {args.groups} groups "
          f"by {judge_name} (group 1: its lowest values; for D(real), judged most real) within {args.strata} band(s) of G (band 1: weakest) | share also flagged by each reference at its highest "
          f"{args.top_percent:g}%")
    if compare_scores is not None:
        from scipy.stats import spearmanr
        finite = np.isfinite(kept["compare"])
        sample = np.flatnonzero(finite)[::max(1, finite.sum() // 1_000_000)]
        print(f"[confidence] Spearman correlation of {judge_name} with {compare_name} over the values G flags: "
              f"{spearmanr(kept['judge'][sample], kept['compare'][sample])[0]:.3f} ({len(sample)} values)")
    columns = list(rows[0])
    print("".join(f"{column:>14}" for column in columns))
    for row in rows:
        print("".join(f"{row[column]:>14.4g}" if isinstance(row[column], float) else f"{row[column]:>14}"
                      for column in columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
