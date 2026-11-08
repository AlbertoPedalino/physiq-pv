"""Do the detectors flag a known event before it starts, in time or in space?

The rules are those of docs/era5_casi_noti/anticipo.md, fixed on 8 November 2026 before any number was counted.
The data are read as a stream every 3 hours: every score at a timestamp uses only the data up to it.

1. Onset of every event of `--events`, measured on the raw ERA5 values of the prepared cache (no model), with
   percentiles per cell over the training years (up to `--train-end-year`):
   - heat: daily maximum of 2 m temperature over the timestamps of the UTC day, above its 90th percentile
     (calendar days within +-15 days); a day counts when a square of 7 x 7 cells centred in a cell of the box
     has at least 60% of its area above; onset = 00 UTC of the first of at least 3 such days in a row
     (Russo et al. 2015; Lavaysse et al. 2018; Stefanon et al. 2012);
   - cold: the same with the daily minimum below its 10th percentile (Lavaysse et al. 2018; ETCCDI);
   - rain: daily total estimated from the one-hour accumulations of the day (24 / their number times their
     sum, x 3 with 8 timestamps) above the 95th percentile of the wet days (>= 1 mm) of the training years,
     in the same square; one day (ETCCDI R95p);
   - wind: speed at 10 m at every timestamp above the 98th percentile of its daily maximum; clusters of cells
     sharing a side, of at least `--wind-area` km2, touching the box, at 8 timestamps in a row (24 h); onset =
     the first of them (Klawa & Ulbrich 2003; Leckebusch et al. 2008, with presence in the box in place of
     their tracking); repeated with half the area (variant "meta area");
   - dust, and every event with no such day or timestamp: 00 UTC of the first day of the source.
   The onset is searched from 3 days before the first day of the source to its last day.
2. For every score (`--model`, `--simple`, `--statistical`, as NAME=DIR[@component]): the share of the cells of
   the box it flags (above its highest `--top-percent`, threshold from the validation years) at every timestamp
   from 72 h before the onset to the last day of the source, and at least 48 h after the onset; the same for
   `--random` dates in the same box, in the same calendar month and at the same hour as the onset, over the
   scored years, whose windows overlap no event of the list. Every window is read against the maximum of the
   random dates over the same lags (Groppe, Urbach & Kutas 2011, sec. 2.2: one threshold for all the lags, so that
   a random date passes it anywhere in the window only 5% of the times): the threshold of a window is the 95th
   (99th) percentile of the maxima of the random dates over its lags. Detected: the share passes the threshold of
   the window from the onset to the end. Lead: hours from the first lag, within 24, 48 or 72 h before the onset,
   where the share passes the threshold of that window, to the onset; counted only for a detected event, otherwise
   a false alarm. For the random dates, the share that passes each threshold (about 5% by construction: the check
   that the rule holds).
3. The same in rings around the box (cells outside it within `--rings` km of its border): a spatial lead when
   the ring passes its threshold before the onset and before the box, for a detected event.

The bands of curve.csv and composito.csv are per lag (2.5-97.5% of the random dates), for the figures only.

Outputs in `--output-dir`: inizi.csv, curve.csv, per_evento.csv, riepilogo.csv, composito.csv, metadata.json.

    python scripts/check_era5_known_events_lead.py --events docs/era5_casi_noti/lista_esatta.csv \
        --prepared-dir data/era5_prepared --train-end-year 1999 \
        --model G_n=outputs/optuna27/test_generator_clim_day \
        --simple normal=outputs/confronti_good2/climatology/test/normal \
                 change_3h=outputs/confronti_good2/climatology/test/change_3h \
        --statistical pca=outputs/confronti_good2/climatology/test/pca ... \
        --output-dir outputs/optuna27/anticipo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd
from scipy import ndimage, sparse

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from physiq_pv.era5.data import load_prepared
from scripts.check_era5_dust_lead import score_limit
from scripts.check_era5_event_zones import open_score, release
from scripts.check_era5_known_events import in_box, read_events
from scripts.check_era5_score_correlation import parse_score

EARTH_RADIUS = 6371.0  # km
STEP = pd.Timedelta(hours=3)
BEFORE, AFTER = 24, 16  # steps of 3 h: 72 h before the onset, at least 48 h after it
WINDOWS = (24, 48, 72)  # hours before the onset where a lead is looked for
SIGNALS = (95, 99)  # percentiles of the random dates
SEARCH_DAYS = 3  # the onset is searched from this many days before the first day of the source
CALENDAR = 15  # +- days of the calendar window of the temperature percentiles
SQUARE, SQUARE_SHARE = 7, 0.6  # Stefanon et al. 2012: 3.75 deg, 60% of the area
HOT_COLD_DAYS = 3
WET_DAY_MM = 1.0
WIND_STEPS = 8  # 24 h
CHUNK = 2048  # timestamps of a score read at a time
TYPES = ("vento", "pioggia", "caldo", "freddo", "polvere")
VARIABLES = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation")


class Layout:
    """The cells of the model on the rectangular grid of the cache: coordinates, areas and 2-D maps."""

    def __init__(self, grid):
        latitudes, longitudes = np.asarray(grid["latitudes"], float), np.asarray(grid["longitudes"], float)
        self.rows, self.cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
        self.shape = (len(latitudes), len(longitudes))
        self.latitudes, self.longitudes = latitudes[self.rows], longitudes[self.cols]
        spacing = lambda values: float(np.abs(np.diff(values)).min()) if len(values) > 1 else 0.5
        half = np.radians(spacing(latitudes) / 2)
        phi = np.radians(self.latitudes)
        self.area = (EARTH_RADIUS ** 2 * np.radians(spacing(longitudes))
                     * (np.sin(np.minimum(phi + half, np.pi / 2)) - np.sin(np.maximum(phi - half, -np.pi / 2))))
        self.area_map = self.to_map(self.area)

    def to_map(self, values, fill=0):
        values = np.asarray(values)
        out = np.full(self.shape, fill, dtype=values.dtype)
        out[self.rows, self.cols] = values
        return out


def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(value, float)) for value in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ring_cells(event, layout, km):
    """Cells outside the box within `km` of its border (distance to the nearest point of the box)."""
    inside = in_box(event, layout.latitudes, layout.longitudes)
    nearest_lat = np.clip(layout.latitudes, event["lat_min"], event["lat_max"])
    nearest_lon = np.clip(layout.longitudes, event["lon_min"], event["lon_max"])
    return ~inside & (distance_km(layout.latitudes, layout.longitudes, nearest_lat, nearest_lon) <= km)


def calendar_day(days):
    """Day of a 365-day year (0..364); 29 February falls on 1 March."""
    days = pd.DatetimeIndex(days)
    return days.dayofyear.to_numpy() - 1 - np.asarray(days.is_leap_year & (days.month > 2), dtype=int)


def first_run(flags, length):
    """Index of the first value of the first run of at least `length` True values, or None."""
    count = 0
    for index, flag in enumerate(flags):
        count = count + 1 if flag else 0
        if count == length:
            return index - length + 1
    return None


def daily_fields(array, timestamps, columns, days_per_chunk=31):
    """Daily fields [days, cells] of raw values: maximum and minimum of 2 m temperature, maximum of the 10 m wind
    speed, and the daily precipitation in mm estimated from the one-hour accumulations of the day."""
    timestamps = pd.DatetimeIndex(timestamps)
    days = timestamps.normalize()
    starts = np.flatnonzero(np.r_[True, np.asarray(days[1:] != days[:-1])])
    ends = np.r_[starts[1:], len(days)]
    fields = {name: np.empty((len(starts), array.shape[1]), np.float32) for name in ("tmax", "tmin", "wind", "rain")}
    for first in range(0, len(starts), days_per_chunk):
        last = min(first + days_per_chunk, len(starts))
        a, b = int(starts[first]), int(ends[last - 1])
        block = np.asarray(array[a:b], dtype=np.float32)
        release_pages(array)
        t2m, u10, v10, tp = (block[:, :, column] for column in columns)
        local = starts[first:last] - a
        fields["tmax"][first:last] = np.maximum.reduceat(t2m, local, axis=0)
        fields["tmin"][first:last] = np.minimum.reduceat(t2m, local, axis=0)
        fields["wind"][first:last] = np.maximum.reduceat(np.hypot(u10, v10), local, axis=0)
        counts = (ends[first:last] - starts[first:last])[:, None]
        fields["rain"][first:last] = np.add.reduceat(tp, local, axis=0) * 1000 * 24 / counts
    return fields, days[starts]


class Base:
    """Percentiles per cell over the training years."""

    def __init__(self, fields, days):
        self.fields, self.calendar = fields, calendar_day(days)
        self.cache = {}
        self.wind = np.percentile(fields["wind"], 98, axis=0)
        wet = np.where(fields["rain"] >= WET_DAY_MM, fields["rain"], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a cell never wet has no percentile
            self.rain = np.nanpercentile(wet, 95, axis=0)
        self.rain = np.where(np.isfinite(self.rain), self.rain, np.inf)  # a cell never wet is never above

    def temperature(self, kind, day):
        """90th percentile of the daily maximum (heat) or 10th of the daily minimum (cold), calendar day +-15."""
        key = (kind, int(calendar_day([day])[0]))
        if key not in self.cache:
            distance = np.abs(self.calendar - key[1])
            near = np.minimum(distance, 365 - distance) <= CALENDAR
            field, q = (self.fields["tmax"], 90) if kind == "caldo" else (self.fields["tmin"], 10)
            self.cache[key] = np.percentile(field[near], q, axis=0)
        return self.cache[key]


def square_share(above, layout):
    """Share of the area of the square of SQUARE x SQUARE cells around every cell that is above, cells of the
    domain only."""
    hit = ndimage.uniform_filter(layout.to_map(np.where(above, layout.area, 0.0)), SQUARE, mode="constant")
    total = ndimage.uniform_filter(layout.area_map, SQUARE, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (hit / total)[layout.rows, layout.cols]


def wind_cluster_in_box(above, box, layout, min_area):
    """Whether a cluster of cells above (sharing a side) of at least `min_area` km2 touches the box."""
    labels, count = ndimage.label(layout.to_map(np.asarray(above, bool), False))
    if count == 0:
        return False
    areas = ndimage.sum(layout.area_map, labels, index=np.arange(1, count + 1))
    large = np.flatnonzero(areas >= min_area) + 1
    return bool(np.isin(large, labels[layout.rows[box], layout.cols[box]]).any())


def measure_onset(event, base, cube, cube_times, columns, layout, min_area):
    """(onset, origin, at the border of the search, qualifying days or timestamps) of one event."""
    source = event["inizio"].normalize()
    start, last = source - pd.Timedelta(days=SEARCH_DAYS), event["fine"].normalize()
    fallback = (source, "fonte", False, 0)
    kind = event["tipo"]
    if kind not in ("caldo", "freddo", "pioggia", "vento"):
        return fallback
    box = in_box(event, layout.latitudes, layout.longitudes)
    extra = pd.Timedelta(days=HOT_COLD_DAYS - 1) if kind in ("caldo", "freddo") else pd.Timedelta(hours=21)
    rows = np.flatnonzero((cube_times >= start) & (cube_times < last + pd.Timedelta(days=1) + extra))
    if not len(rows) or not box.any():
        return fallback
    block_times = cube_times[rows]
    block = np.asarray(cube[rows[0]:rows[-1] + 1], dtype=np.float32)[rows - rows[0]]
    if kind == "vento":
        speed = np.hypot(block[:, :, columns[1]], block[:, :, columns[2]])
        flags = [wind_cluster_in_box(values > base.wind, box, layout, min_area) for values in speed]
        index = first_run(flags, WIND_STEPS)
        if index is None or block_times[index] > last + pd.Timedelta(hours=21):
            return fallback
        return block_times[index], "misurato", index == 0, int(np.sum(flags))
    fields, days = daily_fields(block, block_times, columns, days_per_chunk=len(block))
    flags = []
    for position, day in enumerate(days):
        if kind == "pioggia":
            above = fields["rain"][position] > base.rain
        elif kind == "caldo":
            above = fields["tmax"][position] > base.temperature(kind, day)
        else:
            above = fields["tmin"][position] < base.temperature(kind, day)
        flags.append(bool(np.nanmax(np.where(box, square_share(above, layout), 0)) >= SQUARE_SHARE))
    index = first_run(flags, HOT_COLD_DAYS if kind in ("caldo", "freddo") else 1)
    if index is None or days[index] > last:
        return fallback
    return days[index], "misurato", index == 0, int(np.sum(flags))


def onsets(events, prepared_dir, train_end_year, layout, wind_area):
    cubes, grid, _ = load_prepared(prepared_dir, train_end_year)
    try:
        if not (np.array_equal(np.asarray(grid.rows), layout.rows) and np.array_equal(np.asarray(grid.cols), layout.cols)):
            raise ValueError("The scores and the prepared cache are not on the same cells")
        columns = [list(cubes.feature_names).index(name) for name in VARIABLES]
        print(f"[anticipo] percentiles over {cubes.train_timestamps[0]:%Y}-{cubes.train_timestamps[-1]:%Y}", flush=True)
        base = Base(*daily_fields(cubes.train, cubes.train_timestamps, columns))
        rows = []
        for _, event in events.iterrows():
            variants = [("principale", wind_area)] + ([("meta area", wind_area / 2)] if event["tipo"] == "vento" else [])
            for variant, area in variants:
                onset, origin, border, count = measure_onset(event, base, cubes.test, cubes.test_timestamps, columns,
                                                             layout, area)
                release_pages(cubes.test)
                rows.append({"id": event["id"], "tipo": event["tipo"], "variante": variant,
                             "inizio fonte": event["inizio"].normalize(), "fine fonte": event["fine"].normalize(),
                             "inizio": onset, "origine": origin, "al bordo della ricerca": border,
                             "giorni o istanti oltre la soglia": count})
        return pd.DataFrame(rows)
    finally:
        cubes.close()


def membership(events, layout, rings):
    """Sparse [cells, regions] of the box and the rings of every event, and the name of every region."""
    columns, names = [], []
    for _, event in events.iterrows():
        columns.append(in_box(event, layout.latitudes, layout.longitudes))
        names.append((event["id"], "riquadro"))
        for km in rings:
            columns.append(ring_cells(event, layout, km))
            names.append((event["id"], f"anello {km:g} km"))
    return sparse.csr_matrix(np.column_stack(columns).astype(np.float32)), names


def shares(source, matrix, positions, top_percent):
    """[timestamps of `positions`, regions]: % of the finite cells of every region above the threshold."""
    _, scores, maps = open_score(source)
    try:
        limit = score_limit(scores, top_percent)
        out = np.full((len(positions), matrix.shape[1]), np.nan, np.float32)
        for start in range(0, len(positions), CHUNK):
            part = positions[start:start + CHUNK]
            values = np.asarray(scores[part[0]:part[-1] + 1], dtype=np.float32)[part - part[0]]
            release(scores)
            finite = np.isfinite(values)
            with np.errstate(invalid="ignore"):
                above = (values > limit).astype(np.float32)
            counted = np.asarray((matrix.T @ finite.astype(np.float32).T).T)
            flagged = np.asarray((matrix.T @ above.T).T)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[start:start + len(part)] = np.where(counted > 0, 100 * flagged / counted, np.nan)
        return out, float(limit)
    finally:
        for array in maps:
            array._mmap.close()


def random_onsets(onset, after, times, excluded, count, rng):
    """Positions in `times` of up to `count` random onsets at the hour and in the month of `onset`, whose window
    [onset - BEFORE, onset + after] is inside the scored years and overlaps no excluded interval."""
    positions = np.arange(BEFORE, len(times) - after)
    candidates = times[positions]
    keep = (candidates.hour == onset.hour) & (candidates.month == onset.month)
    starts, ends = candidates - BEFORE * STEP, candidates + after * STEP
    for first, last in excluded:
        keep &= ~((starts <= last) & (ends >= first))
    positions = positions[np.asarray(keep)]
    if len(positions) > count:
        positions = np.sort(rng.choice(positions, count, replace=False))
    return positions


def first_signal(signal, lags, low, high):
    """First lag in [low, high] with a signal, or None."""
    at = np.flatnonzero(signal & (lags >= low) & (lags <= high))
    return int(lags[at[0]]) if len(at) else None


def window_max(curves, mask):
    """Maximum of every curve over the lags of `mask` (NaN when it has no finite value there)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.where(mask, curves, np.nan), axis=-1)


def max_threshold(random_curves, mask, q):
    """The q-th percentile, over the random dates, of their maximum over the lags of `mask`, and the share (%) of
    the random dates above it."""
    maxima = window_max(random_curves, mask)
    if not np.isfinite(maxima).any():
        return np.nan, np.nan
    threshold = float(np.nanpercentile(maxima, q))
    return threshold, 100 * float(np.mean(maxima > threshold))


def analyse(events, starts, series, families, region_names, times, args):
    rng = np.random.default_rng(args.seed)
    region = {name: index for index, name in enumerate(region_names)}
    position = pd.Series(np.arange(len(times)), index=times)
    excluded = [(event["inizio"].normalize() - pd.Timedelta(days=SEARCH_DAYS) - BEFORE * STEP,
                 event["fine"].normalize() + pd.Timedelta(hours=21)) for _, event in events.iterrows()]
    kinds = ["riquadro"] + [f"anello {km:g} km" for km in args.rings]
    curves, per_event, composite_parts = [], [], []
    for _, start in starts.iterrows():
        onset = pd.Timestamp(start["inizio"])
        if onset not in position.index:
            print(f"[anticipo] {start['id']}: onset {onset} outside the scored timestamps, skipped", flush=True)
            continue
        at = int(position[onset])
        end = start["fine fonte"] + pd.Timedelta(hours=21)
        after = max(AFTER, int(np.ceil((end - onset) / STEP)))
        if at - BEFORE < 0 or at + after >= len(times):
            print(f"[anticipo] {start['id']}: window outside the scored timestamps, skipped", flush=True)
            continue
        lags = np.arange(-BEFORE, after + 1)
        randoms = random_onsets(onset, after, times, excluded, args.random, rng)
        for name, values in series.items():
            curve_of = {}
            for kind in kinds:
                column = region[(start["id"], kind)]
                event_curve = values[at + lags, column]
                random_curves = values[randoms[:, None] + lags[None, :], column]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bands = {q: np.nanpercentile(random_curves, q, axis=0) for q in (2.5, 50, 97.5)}
                curve_of[kind] = (event_curve, random_curves, bands)
                for lag_index, lag in enumerate(lags):
                    curves.append({"id": start["id"], "tipo": start["tipo"], "variante": start["variante"],
                                   "metodo": name, "famiglia": families[name], "regione": kind, "ritardo h": 3 * lag,
                                   "evento %": event_curve[lag_index],
                                   **{f"casuali p{q:g}": bands[q][lag_index] for q in bands}})
                composite_parts.append((start["tipo"], start["variante"], name, kind, lags, event_curve, random_curves))
            post = lags >= 0
            for window in WINDOWS:
                low = -(window // 3)
                pre = (lags >= low) & (lags <= -1)
                for q in SIGNALS:
                    event_curve, random_curves, _ = curve_of["riquadro"]
                    before, random_before = max_threshold(random_curves, pre, q)
                    after_limit, random_after = max_threshold(random_curves, post, q)
                    detected = bool(window_max(event_curve, post) > after_limit)
                    early = first_signal(event_curve > before, lags, low, -1)
                    box_first = early if early is not None else first_signal(event_curve > after_limit, lags, 0,
                                                                             lags[-1])
                    record = {"id": start["id"], "tipo": start["tipo"], "variante": start["variante"],
                              "origine inizio": start["origine"], "metodo": name, "famiglia": families[name],
                              "finestra h": window, "segnale percentile": q, "casuali": len(randoms),
                              "soglia prima %": before, "soglia dopo %": after_limit, "rilevato": detected,
                              "anticipo h": -3 * early if detected and early is not None else np.nan,
                              "falso allarme": early is not None and not detected,
                              "casuali con segnale prima %": random_before, "casuali rilevati %": random_after}
                    for kind in kinds[1:]:
                        ring_curve, ring_randoms, _ = curve_of[kind]
                        ring_limit, _ = max_threshold(ring_randoms, pre, q)
                        ring_first = first_signal(ring_curve > ring_limit, lags, low, -1)
                        ahead = (detected and ring_first is not None
                                 and (box_first is None or ring_first < box_first))
                        record[f"{kind}: prima del riquadro"] = ahead
                        record[f"{kind}: ore prima del riquadro"] = (3 * (box_first - ring_first)
                                                                     if ahead and box_first is not None else np.nan)
                    per_event.append(record)
    return pd.DataFrame(curves), pd.DataFrame(per_event), composite(composite_parts)


def composite(parts):
    """By type, variant, method and region: mean of the events at every lag from -72 to +48 h, and the band of
    composites of the same number of random dates (one per event, draw j of every event), 2.5-97.5%."""
    rows = []
    groups = {}
    for kind, variant, name, region, lags, event_curve, random_curves in parts:
        keep = (lags >= -BEFORE) & (lags <= AFTER)
        groups.setdefault((kind, variant, name, region), []).append((event_curve[keep], random_curves[:, keep]))
    for (kind, variant, name, region), items in groups.items():
        draws = min(len(random) for _, random in items)
        events_mean = np.nanmean(np.stack([curve for curve, _ in items]), axis=0)
        if draws:
            composites = np.nanmean(np.stack([random[:draws] for _, random in items]), axis=0)
            low, middle, high = np.nanpercentile(composites, (2.5, 50, 97.5), axis=0)
        else:
            low = middle = high = np.full(len(events_mean), np.nan)
        for index, lag in enumerate(range(-BEFORE, AFTER + 1)):
            rows.append({"tipo": kind, "variante": variant, "metodo": name, "regione": region, "ritardo h": 3 * lag,
                         "eventi": len(items), "evento media %": events_mean[index],
                         "casuali p2.5": low[index], "casuali p50": middle[index], "casuali p97.5": high[index]})
    return pd.DataFrame(rows)


def summary(per_event, rings):
    rows = []
    keys = ["tipo", "variante", "metodo", "famiglia", "finestra h", "segnale percentile"]
    for key, group in per_event.groupby(keys, sort=False):
        lead = group["anticipo h"].dropna()
        row = dict(zip(keys, key))
        row.update({"eventi": len(group), "rilevati %": 100 * group["rilevato"].mean(),
                    "con anticipo %": 100 * group["anticipo h"].notna().mean(),
                    "anticipo mediano h": lead.median() if len(lead) else np.nan,
                    "anticipo q25 h": lead.quantile(0.25) if len(lead) else np.nan,
                    "anticipo q75 h": lead.quantile(0.75) if len(lead) else np.nan,
                    "falsi allarmi eventi %": 100 * group["falso allarme"].mean(),
                    "casuali con segnale prima %": group["casuali con segnale prima %"].mean(),
                    "casuali rilevati %": group["casuali rilevati %"].mean()})
        for km in rings:
            row[f"anello {km:g} km prima %"] = 100 * group[f"anello {km:g} km: prima del riquadro"].mean()
        rows.append(row)
    order = {kind: index for index, kind in enumerate(TYPES)}
    table = pd.DataFrame(rows)
    return table.sort_values(["tipo", "variante", "finestra h", "segnale percentile"],
                             key=lambda column: column.map(order) if column.name == "tipo" else column, kind="stable")


def parse_family(texts, family):
    return [(parse_score(text), family) for text in texts or []]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True, help="CSV of the known events (lista_esatta.csv)")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--train-end-year", type=int, default=1999)
    parser.add_argument("--model", nargs="+", required=True, help="NAME=DIR[@component] of the model")
    parser.add_argument("--simple", nargs="*", help="NAME=DIR[@component] of the simple detectors")
    parser.add_argument("--statistical", nargs="*", help="NAME=DIR[@component] of the statistical detectors")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--random", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rings", type=float, nargs="+", default=[100, 250, 500])
    parser.add_argument("--wind-area", type=float, default=155_000.0, help="km2, minimum area of a wind cluster")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = (parse_family(args.model, "modello") + parse_family(args.simple, "semplici")
               + parse_family(args.statistical, "statistici"))
    names = [source[0] for source, _ in sources]
    if len(set(names)) != len(names):
        parser.error("the names of the scores must be distinct")
    events = read_events(args.events)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [json.loads((source[1] / "metadata.json").read_text(encoding="utf-8")) for source, _ in sources]
    grid = manifests[0]["grid"]
    for (source, _), manifest in zip(sources, manifests):
        if manifest["grid"]["rows"] != grid["rows"] or manifest["grid"]["cols"] != grid["cols"]:
            raise ValueError(f"{source[0]} is not on the cells of {names[0]}")
    layout = Layout(grid)
    stamps = [pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy")) for source, _ in sources]
    times = stamps[0]
    for other in stamps[1:]:
        times = times.intersection(other)
    if len(times) < 2 or (np.diff(times.asi8) != STEP.value).any():
        raise ValueError("The common timestamps of the scores are not every 3 hours without gaps")

    starts = onsets(events, args.prepared_dir, args.train_end_year, layout, args.wind_area)
    starts.to_csv(args.output_dir / "inizi.csv", index=False)
    print(starts[["id", "tipo", "variante", "inizio fonte", "inizio", "origine", "al bordo della ricerca"]]
          .to_string(index=False), flush=True)

    matrix, region_names = membership(events, layout, args.rings)
    series, limits = {}, {}
    for ((name, directory, component), family), own in zip(sources, stamps):
        positions = own.get_indexer(times)
        series[name], limits[name] = shares((name, directory, component), matrix, positions, args.top_percent)
        print(f"[anticipo] {name} ({family}): threshold {limits[name]:.4g}", flush=True)
    families = {source[0]: family for source, family in sources}
    curves, per_event, composite_table = analyse(events, starts, series, families, region_names, times, args)
    table = summary(per_event, args.rings)
    curves.to_csv(args.output_dir / "curve.csv", index=False, float_format="%.4g")
    per_event.to_csv(args.output_dir / "per_evento.csv", index=False, float_format="%.4g")
    table.to_csv(args.output_dir / "riepilogo.csv", index=False, float_format="%.4g")
    composite_table.to_csv(args.output_dir / "composito.csv", index=False, float_format="%.4g")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "rules": "docs/era5_casi_noti/anticipo.md (8 November 2026)", "events": str(args.events),
        "scores": {name: str(directory) + (f"@{component}" if component != "combined" else "")
                   for (name, directory, component), _ in sources},
        "families": families, "thresholds": limits, "top_percent": args.top_percent, "random": args.random,
        "seed": args.seed, "rings_km": args.rings, "wind_area_km2": args.wind_area,
        "train_end_year": args.train_end_year, "scored": [str(times[0]), str(times[-1])]}, indent=2),
        encoding="utf-8")
    main_rows = table[(table["variante"] == "principale") & (table["finestra h"] == 72)
                      & (table["segnale percentile"] == 95)]
    columns = ["tipo", "metodo", "eventi", "rilevati %", "con anticipo %", "anticipo mediano h",
               "falsi allarmi eventi %", "casuali con segnale prima %", "casuali rilevati %",
               f"anello {args.rings[len(args.rings) // 2]:g} km prima %"]
    print("[anticipo] 72 h, threshold = 95th percentile of the maxima of the random dates, main onset:")
    print(main_rows[columns].round(1).to_string(index=False))
    print(f"[anticipo] written {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
