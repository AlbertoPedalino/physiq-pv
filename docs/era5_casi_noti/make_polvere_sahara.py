"""The regions of the Saharan dust events, for check_era5_known_events.py (rule fixed on 9 November 2026, before any
share of these regions was read).

For every Saharan dust event of lista_cieca.csv (polvere_03, 04, 05, 06, 07, 09, 10) three regions:
- arrivo: the box and the days of the event, as in the list;
- origine: the northern Sahara, 20-32 N, with the longitudes of the box widened by 10 degrees on each side (within the
  domain, 15 W - 50 E), from 3 days before the first day of the event to that day;
- percorso: the band between 32 N and the southern edge of the box, same longitudes, from 2 days before the first day
  to that day; left out when the box starts south of 32 N (the Canary Islands).
Writes polvere_sahara.csv next to this file.
    python docs/era5_casi_noti/make_polvere_sahara.py ../casi_noti/lista_cieca.csv
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SAHARAN = ["polvere_03", "polvere_04", "polvere_05", "polvere_06", "polvere_07", "polvere_09", "polvere_10"]
SOUTH, NORTH_SAHARA, WIDEN, WEST, EAST = 20.0, 32.0, 10.0, -15.0, 50.0


def main():
    import sys
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "lista_cieca.csv"  # the blind list of 60 events
    events = pd.read_csv(source, keep_default_na=False).set_index("id").loc[SAHARAN]
    rows = []
    for name, event in events.iterrows():
        start = pd.Timestamp(event["inizio"])
        west, east = max(event["lon_min"] - WIDEN, WEST), min(event["lon_max"] + WIDEN, EAST)
        rows.append({"id": f"{name}_arrivo", "inizio": event["inizio"], "fine": event["fine"],
                     "lat_min": event["lat_min"], "lat_max": event["lat_max"],
                     "lon_min": event["lon_min"], "lon_max": event["lon_max"]})
        rows.append({"id": f"{name}_origine", "inizio": str((start - pd.Timedelta(days=3)).date()),
                     "fine": event["inizio"], "lat_min": SOUTH, "lat_max": NORTH_SAHARA,
                     "lon_min": west, "lon_max": east})
        if event["lat_min"] > NORTH_SAHARA:
            rows.append({"id": f"{name}_percorso", "inizio": str((start - pd.Timedelta(days=2)).date()),
                         "fine": event["inizio"], "lat_min": NORTH_SAHARA, "lat_max": event["lat_min"],
                         "lon_min": west, "lon_max": east})
    pd.DataFrame(rows).to_csv(HERE / "polvere_sahara.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
