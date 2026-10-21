"""Read back the numbers of the ERA5 checks from their logs, for the summary notebook.

Every reader takes the text the check printed (as `tee` saved it) and returns its numbers; none of them
computes anything new. Logs that hold several runs are cut at their `########` lines first (`blocks`).

- `eobs`: the `[eobs]` header of check_era5_score_against_eobs.py (first score, indicator) and its ratios
  "score alone / all cell-days" and "score and another / all cell-days" (same day, next day, either);
- `precursor`: the rows "radius, lead, case" of check_era5_precursor_precision.py;
- `direction`: the tables of check_era5_edge_direction.py (own events, same events, agreement);
- `event_types`: the first table of check_era5_edge_event_types.py (all seasons and latitudes);
- `confidence`: the classes of confidence of check_era5_tail_surprise.py.
"""
from __future__ import annotations

import re

NUMBER = r"(-?[\d.]+|nan)"
RATIO = rf"{NUMBER} \({NUMBER}-{NUMBER}\)"
MEASURES = ("same day", "next day", "either")
# The indicators of the stations, by the variable asked and the tail of the [eobs] header.
INDICATORS = {("rain", "highest"): "rain", ("pressure", "lowest"): "pressure low",
              ("change of the pressure over 1 day(s)", "lowest"): "pressure drop",
              ("radiation", "lowest"): "radiation low", ("temperature", "highest"): "heat",
              ("temperature", "lowest"): "cold"}


def value(text):
    return float(text)


def blocks(text):
    """{title: text} of a log cut at its `######## title` lines (the text before the first one under "")."""
    result, title, lines = {}, "", []
    for line in text.splitlines():
        if line.startswith("########"):
            result[title] = "\n".join(lines)
            title, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    result[title] = "\n".join(lines)
    return result


def eobs(text):
    """One dict per E-OBS run of the text: first score, indicator and the ratios {"alone", "shared"} -> {measure:
    (ratio, low, high)}. A run starts at the `[eobs]` header of the check, or, in the logs that kept only the
    tables, at a `=== <variable> <high|low> change_days=<n>` line under a `######## score: <run>` line, or at a
    `######## <title> <variable> <high|low> change_days=<n>` line (first score None: the log does not name it)."""
    runs, first = [], None
    for line in text.splitlines():
        header = re.match(r"\[eobs\] .*\| first: (\S+) \| others: (.*?) \| unusual (.*?): (highest|lowest) ", line)
        if header:
            first, others, asked, tail = header.groups()
            runs.append({"first": first, "others": others, "indicator": INDICATORS.get((asked, tail), f"{asked} {tail}"),
                         "ratios": {}})
            continue
        score = re.match(r"#+\s*score:\s*(\S+)", line)
        if score:
            first = score.group(1)
            continue
        short = re.match(r"(?:=== |#+ .*?)(\w+) (high|low) change_days=(\d+)", line)
        if short:
            variable, tail, days = short.group(1), short.group(2), int(short.group(3))
            asked = f"change of the {variable} over {days} day(s)" if days else variable
            tail = "highest" if tail == "high" else "lowest"
            runs.append({"first": first, "others": None,
                         "indicator": INDICATORS.get((asked, tail), f"{asked} {tail}"), "ratios": {}})
            continue
        ratio = re.match(rf"score (alone|and another) / all cell-days\s+{RATIO}\s+{RATIO}\s+{RATIO}\s*$", line)
        if ratio and runs:
            numbers = [value(item) for item in ratio.groups()[1:]]
            runs[-1]["ratios"]["alone" if ratio.group(1) == "alone" else "shared"] = {
                measure: tuple(numbers[3 * k:3 * k + 3]) for k, measure in enumerate(MEASURES)}
    return runs


def precursor(text):
    """{(radius, hours, case): (cells, % of cells, % followed by an event, lift)} of a precursor table."""
    rows = {}
    for line in text.splitlines():
        found = re.match(rf"(\d+) cell\(s\), (\d+)h, (both|first only|second early only|neither)\s+(\d+)\s+{NUMBER}"
                         rf"\s+{NUMBER}\s+{NUMBER}\s*$", line)
        if found:
            radius, hours, case, cells, share, followed, lift = found.groups()
            rows[(int(radius), int(hours), case)] = (int(cells), value(share), value(followed), value(lift))
    return rows


def direction(text):
    """The tables of the direction check: {"own events", "same events"} -> {(direction, target, hours): (events, same
    side %, within 45 %, mean cosine)}; "cases" -> {(reference, target, hours): (both right, edge only, reference
    only, both wrong, edge right when reference wrong)}; "agreement" -> {(reference, target, hours): (agree, right if
    agree, reference right if not, edge right if not, combined right, reference right)}."""
    tables = {"own events": {}, "same events": {}, "cases": {}, "agreement": {}}
    current = None
    for line in text.splitlines():
        if line.startswith("cosine between"):
            current = "same events" if "on the same events" in line else "own events"
        elif line.startswith("on the same events: is the edge"):
            current = "cases"
        elif line.startswith("on the same events: when the edge"):
            current = "agreement"
        found = re.match(rf"(.+?), (shift|growth), (\d+)h\s+((?:{NUMBER}\s*)+)$", line)
        if found and current is not None:
            name, target, hours, numbers = found.group(1), found.group(2), int(found.group(3)), found.group(4).split()
            values = tuple(value(item) for item in numbers)
            if current in ("own events", "same events"):
                values = (int(values[0]), *values[1:])
            tables[current][(name, target, hours)] = values
    return tables


def event_types(text):
    """{population: {"cells", type..., "s:signal"...}} of the first table of the event types check."""
    rows, header = {}, None
    for line in text.splitlines():
        if line.startswith("% of the cells by type"):
            header = []
            continue
        if header == [] and line.strip().startswith("cells"):
            header = line.split()
            continue
        if header:
            if not line.strip():
                break
            parts = line.rsplit(None, len(header))
            if len(parts) == len(header) + 1:
                rows[parts[0].strip()] = {name: value(item) for name, item in zip(header, parts[1:])}
    return rows


def confidence(text):
    """{(flags, class): (points, % still unusual, % unusual in the forecast only)} of the classes of confidence."""
    rows = {}
    for line in text.splitlines():
        found = re.match(rf"(.+?), (\d)\s+(\d+)\s+{NUMBER}\s+{NUMBER}\s*$", line)
        if found:
            label, level, points, unusual, only = found.groups()
            rows[(label.strip(), int(level))] = (int(points), value(unusual), value(only))
    return rows
