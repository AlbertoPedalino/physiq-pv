"""Seasonal recurrence of anomaly activity and climatology-normalized scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MONTH_LABELS = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
WEEKDAY_LABELS = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]


def climatology_zscores(scores, timestamps, output, *, uncertainty=None, uncertainty_output=None,
                        min_samples=30, chunk_size=64):
    """Write leave-one-year-out z-scores per location x calendar month x UTC hour.

    ``scores`` is (T, ...) with any trailing location shape, typically the (T, N) STGAN
    scores; ``output`` receives a float32 ``.npy`` of the same shape. Each value becomes
    ``(score - mean) / std`` where mean and std come from the same location, month and hour
    in the *other* years, so a year never enters its own climatology. Groups with fewer than
    ``min_samples`` reference values or zero variance become NaN. When ``uncertainty``
    (MC std of the score) is given, ``uncertainty / std`` is written to
    ``uncertainty_output``: the same spread expressed in z units.
    Memory: three (years, 12 * hours, ...) float32/int32 accumulators.
    """
    timestamps = pd.DatetimeIndex(timestamps)
    if len(timestamps) != len(scores):
        raise ValueError("Scores and timestamps must have the same length.")
    if (uncertainty is None) != (uncertainty_output is None):
        raise ValueError("Pass uncertainty and uncertainty_output together.")
    if uncertainty is not None and uncertainty.shape != scores.shape:
        raise ValueError("Uncertainty must match scores.")
    years, year_of = np.unique(timestamps.year, return_inverse=True)
    hours, hour_of = np.unique(timestamps.hour, return_inverse=True)
    if len(years) < 2:
        raise ValueError("Leave-one-year-out climatology needs at least two years.")
    slot_of = (timestamps.month.to_numpy() - 1) * len(hours) + hour_of
    shape = (len(years), 12 * len(hours), *scores.shape[1:])
    count = np.zeros(shape, np.int32)
    total = np.zeros(shape, np.float32)
    square = np.zeros(shape, np.float32)
    for start in range(0, len(scores), chunk_size):
        block = np.asarray(scores[start:start + chunk_size], dtype=np.float64)
        for offset, frame in enumerate(block):
            valid = np.isfinite(frame)
            values = np.where(valid, frame, 0.0)
            key = year_of[start + offset], slot_of[start + offset]
            count[key] += valid
            total[key] += values
            square[key] += values * values
    all_count = count.sum(axis=0, dtype=np.int64)
    all_total = total.sum(axis=0, dtype=np.float64)
    all_square = square.sum(axis=0, dtype=np.float64)

    z_store = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=scores.shape)
    std_store = (None if uncertainty is None else
                 np.lib.format.open_memmap(uncertainty_output, mode="w+", dtype=np.float32, shape=scores.shape))
    n_finite = 0
    try:
        for start in range(0, len(scores), chunk_size):
            stop = min(start + chunk_size, len(scores))
            year, slot = year_of[start:stop], slot_of[start:stop]
            n = all_count[slot] - count[year, slot]
            sums = all_total[slot] - total[year, slot]
            squares = all_square[slot] - square[year, slot]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = sums / n
                deviation = np.sqrt((squares - sums * mean) / (n - 1))
                z = (np.asarray(scores[start:stop], dtype=np.float64) - mean) / deviation
            unusable = (n < min_samples) | ~(deviation > 0)
            z[unusable] = np.nan
            z_store[start:stop] = z
            n_finite += int(np.isfinite(z).sum())
            if std_store is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    spread = np.asarray(uncertainty[start:stop], dtype=np.float64) / deviation
                spread[unusable] = np.nan
                std_store[start:stop] = spread
        z_store.flush()
        if std_store is not None:
            std_store.flush()
    finally:
        z_store._mmap.close()
        if std_store is not None:
            std_store._mmap.close()
    return {"method": "leave-one-year-out z-score per location x calendar month x UTC hour",
            "years": [int(value) for value in years], "hours_utc": [int(value) for value in hours],
            "min_samples": min_samples, "finite_values": n_finite,
            "nan_values": int(np.prod(scores.shape)) - n_finite}


def _loo_correlation(matrix):
    others = (matrix.sum(axis=0)[None] - matrix) / (len(matrix) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array([np.corrcoef(row, other)[0, 1] for row, other in zip(matrix, others)])


def annual_report(activity, report_dir, *, event_starts=None, header=(), start="2005-01-01",
                  end="2026-01-01", permutations=1000, seed=0):
    """Seasonal distribution of above-threshold cells, saved as summary.txt + CSV + PNG.

    ``activity`` needs ``timestamp``, ``n_valid`` and ``n_above_threshold`` per timestamp.
    Recurrence: each year's weekly rate profile is correlated with the mean of the other
    years; the null shifts every year circularly by a random number of weeks, which keeps
    episode length and autocorrelation but breaks the calendar alignment.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    seasonal = activity.loc[activity.timestamp.ge(start) & activity.timestamp.lt(end)
                            & activity.n_valid.gt(0)].copy()
    if seasonal.empty:
        raise ValueError("No timestamp with valid scores in the selected period.")
    seasonal["year"] = seasonal.timestamp.dt.year
    seasonal["month"] = seasonal.timestamp.dt.month
    seasonal["week"] = np.minimum((seasonal.timestamp.dt.dayofyear - 1) // 7 + 1, 52)  # 29/2, 31/12 -> 52
    seasonal["weekday"] = seasonal.timestamp.dt.dayofweek
    seasonal["hour_utc"] = seasonal.timestamp.dt.hour
    total_above = int(seasonal.n_above_threshold.sum())
    total_valid = int(seasonal.n_valid.sum())
    base_rate = 100 * total_above / total_valid

    def rate_table(keys):
        sums = seasonal.groupby(keys)[["n_above_threshold", "n_valid"]].sum().astype(np.int64)
        table = pd.DataFrame({
            "celle_sopra_soglia": sums.n_above_threshold,
            "quota_anomalie_%": 100 * sums.n_above_threshold / max(1, total_above),
            "quota_osservazioni_%": 100 * sums.n_valid / total_valid,
            "tasso_sopra_soglia_%": 100 * sums.n_above_threshold / sums.n_valid})
        table["lift"] = table["tasso_sopra_soglia_%"] / base_rate if base_rate > 0 else np.nan
        return table

    # Anno x settimana: strisce verticali = stesso periodo anomalo in piu' anni.
    weekly = rate_table(["year", "week"])["tasso_sopra_soglia_%"].unstack().reindex(columns=range(1, 53))
    years = weekly.index.to_numpy()
    week_axis = np.arange(1, 53)
    quartiles = weekly.quantile([.25, .5, .75])
    month_ticks = (pd.date_range("2001-01-01", periods=12, freq="MS").dayofyear - 1) / 7 + 1
    cmap = plt.colormaps["YlOrRd"].with_extremes(bad="#dddddd")
    color_max = max(np.nanpercentile(weekly.to_numpy(), 99), base_rate, 1e-9)
    season_figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, height_ratios=(3, 2),
                                       constrained_layout=True)
    image = axes[0].imshow(np.ma.masked_invalid(weekly.to_numpy()), aspect="auto", origin="lower",
                           cmap=cmap, vmin=0, vmax=color_max,
                           extent=(.5, 52.5, years[0] - .5, years[-1] + .5))
    axes[0].set(ylabel="Anno", title="Localita sopra soglia per anno e settimana (%)")
    axes[0].set_yticks(years[::max(1, len(years) // 12)])
    season_figure.colorbar(image, ax=axes[0], label="Localita sopra soglia (%)", extend="max")
    axes[1].fill_between(week_axis, quartiles.loc[.25], quartiles.loc[.75], color="tab:red",
                         alpha=.2, label="Intervallo interquartile fra anni")
    axes[1].plot(week_axis, quartiles.loc[.5], color="tab:red", label="Mediana fra anni")
    axes[1].plot(week_axis, weekly.mean(), color="black", linewidth=.8, linestyle=":",
                 label="Media fra anni")
    axes[1].axhline(base_rate, color="tab:blue", linestyle="--", label="Quota media complessiva")
    axes[1].set(xlabel="Settimana dell anno", ylabel="Localita sopra soglia (%)",
                title="Profilo stagionale: mediana sopra la linea blu = periodo ricorrente")
    axes[1].set_xticks(month_ticks, MONTH_LABELS)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[1].legend(ncol=4)
    axes[1].grid(alpha=.2)
    season_figure.savefig(report_dir / "anno_settimana.png", dpi=110)

    complete = weekly.dropna()
    recurrence = {"correlazione_mediana": np.nan, "nullo_95": np.nan, "p_value": np.nan,
                  "anni_completi": len(complete)}
    per_year = pd.Series(dtype=float, name="corr_con_altri_anni")
    if len(complete) >= 3:
        profiles = complete.to_numpy()
        per_year = pd.Series(_loo_correlation(profiles), index=complete.index, name="corr_con_altri_anni")
        observed = float(np.nanmedian(per_year))
        rng = np.random.default_rng(seed)
        null = np.array([np.nanmedian(_loo_correlation(np.stack(
            [np.roll(profile, rng.integers(52)) for profile in profiles])))
            for _ in range(permutations)])
        recurrence.update(correlazione_mediana=observed, nullo_95=float(np.nanpercentile(null, 95)),
                          p_value=float((1 + np.sum(null >= observed)) / (1 + permutations)))
        recurrence_line = (f"Correlazione mediana anno vs altri anni: {observed:.3f} | "
                           f"nullo (shift casuali) 95esimo percentile: {recurrence['nullo_95']:.3f} | "
                           f"p = {recurrence['p_value']:.4f} ({len(complete)} anni completi)")
    else:
        recurrence_line = "Meno di tre anni completi: test di ricorrenza saltato."

    # Lift = tasso del gruppo / tasso medio; 1 = nessuna preferenza.
    by_month = rate_table("month").reindex(range(1, 13)).set_axis(MONTH_LABELS)
    by_weekday = rate_table("weekday").reindex(range(7)).set_axis(WEEKDAY_LABELS)
    by_hour = rate_table("hour_utc")
    lift_figure, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    panels = ((axes[0, 0], by_month.lift, MONTH_LABELS, "Mese (non e un input)"),
              (axes[0, 1], by_weekday.lift, WEEKDAY_LABELS, "Giorno della settimana (input one-hot)"),
              (axes[1, 0], by_hour.lift, [str(hour) for hour in by_hour.index], "Ora UTC (input one-hot)"))
    for axis, values, labels, title in panels:
        axis.bar(range(len(values)), values, color="tab:red", alpha=.7)
        axis.axhline(1, color="black", linewidth=.8)
        axis.set_xticks(range(len(values)), labels)
        axis.set(ylabel="Lift (tasso / tasso medio)", title=title)
    tables = {"Tasso per mese": by_month, "Tasso per giorno della settimana": by_weekday,
              "Tasso per ora UTC": by_hour}
    if event_starts is not None:
        starts = pd.DatetimeIndex(pd.to_datetime(event_starts))
        starts = starts[(starts >= pd.Timestamp(start)) & (starts < pd.Timestamp(end))]
        by_event_month = pd.Series(starts.month).value_counts().reindex(range(1, 13), fill_value=0)
        events_table = pd.DataFrame({"eventi_iniziati": by_event_month.to_numpy(),
                                     "quota_%": 100 * by_event_month.to_numpy() / max(1, len(starts))},
                                    index=MONTH_LABELS)
        axes[1, 1].bar(range(12), events_table["quota_%"], color="tab:purple")
        axes[1, 1].axhline(100 / 12, color="black", linewidth=.8, label="Distribuzione uniforme")
        axes[1, 1].set_xticks(range(12), MONTH_LABELS)
        axes[1, 1].set(ylabel="Eventi iniziati (%)", title=f"Mese di inizio degli eventi (n={len(starts):,})")
        axes[1, 1].legend()
        tables["Mese di inizio degli eventi"] = events_table
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(.5, .5, "Eventi non ricalcolati per questo score", ha="center", va="center")
    lift_figure.savefig(report_dir / "lift.png", dpi=110)

    week_starts = pd.Timestamp("2001-01-01") + pd.to_timedelta((week_axis - 1) * 7, unit="D")
    profile = pd.DataFrame({"inizio_settimana": week_starts.strftime("%d/%m"),
                            "mediana_%": quartiles.loc[.5].to_numpy(),
                            "q25_%": quartiles.loc[.25].to_numpy(), "q75_%": quartiles.loc[.75].to_numpy(),
                            "anni_sopra_media": (weekly > base_rate).sum().to_numpy(),
                            "anni_validi": weekly.notna().sum().to_numpy()}, index=week_axis)
    profile.index.name = "settimana"
    tables["Settimane con mediana piu alta (top 10)"] = profile.nlargest(10, "mediana_%")
    tables["Correlazione di ogni anno con gli altri anni"] = per_year.to_frame()
    sections = ["Report distribuzione annuale anomalie ERA5", *header,
                f"Periodo: {seasonal.timestamp.min()} - {seasonal.timestamp.max()} ({len(seasonal):,} timestamp)",
                f"Celle sopra soglia: {total_above:,} su {total_valid:,} ({base_rate:.4f}%)",
                recurrence_line]
    sections += [f"\n== {title} ==\n{table.round(4).to_string()}" for title, table in tables.items()]
    (report_dir / "summary.txt").write_text("\n".join(sections) + "\n", encoding="utf-8")
    weekly.round(5).to_csv(report_dir / "tasso_anno_settimana.csv")
    profile.round(5).to_csv(report_dir / "profilo_settimanale.csv")
    for name, table in (("mese", by_month), ("giorno_settimana", by_weekday), ("ora_utc", by_hour)):
        table.round(5).to_csv(report_dir / f"tasso_{name}.csv")
    per_year.round(5).to_csv(report_dir / "correlazione_per_anno.csv")
    return {"figures": [season_figure, lift_figure], "tables": tables, "weekly": weekly,
            "recurrence": recurrence, "recurrence_line": recurrence_line,
            "base_rate": base_rate, "report_dir": report_dir}


def seasonality_metrics(report):
    """Compact numbers to compare two annual reports side by side."""
    month = report["tables"]["Tasso per mese"].lift
    hour = report["tables"]["Tasso per ora UTC"].lift
    weekday = report["tables"]["Tasso per giorno della settimana"].lift
    return pd.Series({
        "correlazione mediana fra anni": report["recurrence"]["correlazione_mediana"],
        "nullo 95esimo percentile": report["recurrence"]["nullo_95"],
        "p ricorrenza": report["recurrence"]["p_value"],
        "lift mese massimo": month.max(), "lift mese minimo": month.min(),
        "mese massimo / minimo": month.max() / month.min(),
        "lift ora massimo": hour.max(), "lift ora minimo": hour.min(),
        "scarto massimo giorno settimana": (weekday - 1).abs().max()})
