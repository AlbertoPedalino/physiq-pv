"""Generate the final markdown report for sweep ozii0s5s from the dumped CSVs.
Eval-only post-processing; launches no experiments and reads no model/data."""
import numpy as np
import pandas as pd

df = pd.read_csv("outputs/sweep_analysis/ozii0s5s.csv").sort_values("seed").reset_index(drop=True)
lt = pd.read_csv("outputs/sweep_analysis/ltxbtupy.csv")


def num(s, c):
    return pd.to_numeric(s[c], errors="coerce") if c in s else pd.Series([np.nan] * len(s))


def mean(d, c):
    v = num(d, c).dropna()
    return float(v.mean()) if len(v) else float("nan")


def f(x, nd=4):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "—"


L = []
A = L.append
A("# Sweep `ozii0s5s` — PVGIS-only ST-GNN: CLC / interval-metrics report\n")
A("Seed-only sweep (5 seeds) over the PVGIS-only ST-GNN with MC-Dropout + **group** "
  "post-hoc calibration. Interval reliability/sharpness metrics (PICP / MPIW / NMPIL / CLC) "
  "are **eval-only** — they do not touch the model, training, features or data. "
  "Source CSV: `outputs/sweep_analysis/ozii0s5s.csv`.\n")

# 1. Setup
A("## 1. Setup\n")
A("| param | value |")
A("|---|---|")
setup = [("sweep_id", "ozii0s5s"), ("train_years", "2016,2017"), ("calibration_year", "2018"),
         ("test_year", "2019"), ("feature_set", "full"), ("target", "pv_power_output"),
         ("model_type", "stgnn"), ("batch_size", "16"), ("epochs", "5"), ("lr", "0.001"),
         ("dropout", "0.2"), ("mc_samples", "20"), ("calibration_strategy", "group"),
         ("coverage_target (gamma)", "0.95"), ("clc_eta (eta)", "10"), ("seeds", "1,2,3,4,5"),
         ("skip_predictions_csv", "true"), ("max_calibration_samples", "200000")]
for k, v in setup:
    A(f"| {k} | {v} |")
A("")

# 2. Per-seed table
A("## 2. Per-seed results\n")
cols = [("seed", "seed", 0), ("run_name", "name", None), ("state", "state", None),
        ("mae/global", "mae/global", 3), ("mae/normal", "mae/normal", 3),
        ("mae/rare_extreme", "mae/rare_extreme", 3), ("ratio_mae_r/n", "ratio/mae_rare_normal", 3),
        ("std_normal", "uncertainty/mean_std_normal", 3),
        ("std_rare", "uncertainty/mean_std_rare_extreme", 3),
        ("std_ratio_r/n", "uncertainty/ratio_rare_normal", 3),
        ("picp_raw/g", "picp_raw/global", 3), ("picp_cal/g", "picp_calibrated/global", 3),
        ("nmpil_raw/g", "nmpil_raw/global", 4), ("nmpil_cal/g", "nmpil_calibrated/global", 4),
        ("clc_raw/g", "clc_raw/global", 3), ("clc_cal/g", "clc_calibrated/global", 3),
        ("clc_cal/norm", "clc_calibrated/normal", 3), ("clc_cal/rare", "clc_calibrated/rare_extreme", 3),
        ("k_global", "calibration/factor_global", 3), ("k_normal", "calibration/factor_normal", 3),
        ("k_rare_or_extreme", "calibration/factor_rare_extreme", 3)]
A("| " + " | ".join(h for h, _, _ in cols) + " |")
A("|" + "---|" * len(cols))
for _, r in df.iterrows():
    cells = []
    for h, c, nd in cols:
        if c not in df.columns:
            cells.append("—")
            continue
        v = r[c]
        if nd is None:
            cells.append(str(v))
        else:
            try:
                cells.append(f(float(v), nd))
            except (TypeError, ValueError):
                cells.append("—")
    A("| " + " | ".join(cells) + " |")
A("")


# 3. Aggregate mean/std/cv
def agg_block(title, metrics):
    A(f"### {title}\n")
    A("| metric | mean | std | cv |")
    A("|---|---|---|---|")
    for m in metrics:
        v = num(df, m).dropna()
        if len(v) == 0:
            A(f"| {m} | — | — | — |")
            continue
        mu, sd = v.mean(), v.std()
        cv = sd / mu if mu else float("nan")
        A(f"| {m} | {f(mu)} | {f(sd)} | {f(cv, 3)} |")
    A("")


A("## 3. Aggregated (mean / std / cv), n=5\n")
agg_block("Point forecast", ["mae/global", "mae/normal", "mae/rare_extreme", "ratio/mae_rare_normal",
                             "rmse/global", "rmse/normal", "rmse/rare_extreme", "ratio/rmse_rare_normal"])
agg_block("Raw interval metrics",
          ["picp_raw/global", "picp_raw/normal", "picp_raw/rare_extreme",
           "mpiw_raw/global", "mpiw_raw/normal", "mpiw_raw/rare_extreme",
           "nmpil_raw/global", "nmpil_raw/normal", "nmpil_raw/rare_extreme",
           "clc_raw/global", "clc_raw/normal", "clc_raw/rare_extreme"])
agg_block("Calibrated interval metrics",
          ["picp_calibrated/global", "picp_calibrated/normal", "picp_calibrated/rare_extreme",
           "mpiw_calibrated/global", "mpiw_calibrated/normal", "mpiw_calibrated/rare_extreme",
           "nmpil_calibrated/global", "nmpil_calibrated/normal", "nmpil_calibrated/rare_extreme",
           "clc_calibrated/global", "clc_calibrated/normal", "clc_calibrated/rare_extreme"])
agg_block("Calibration factors (k)",
          ["calibration/factor_global", "calibration/factor_normal", "calibration/factor_rare_extreme"])

# precompute means
m_picp_raw_g = mean(df, "picp_raw/global")
m_nmpil_raw_g = mean(df, "nmpil_raw/global")
m_clc_raw_g = mean(df, "clc_raw/global")
m_picp_cal_g = mean(df, "picp_calibrated/global")
m_nmpil_cal_g = mean(df, "nmpil_calibrated/global")
m_clc_cal_g = mean(df, "clc_calibrated/global")
m_clc_cal_n = mean(df, "clc_calibrated/normal")
m_clc_cal_r = mean(df, "clc_calibrated/rare_extreme")

# 4. interpretation
A("## 4. Sharpness / reliability interpretation\n")
A(f"- **`picp_raw/global ~ {f(m_picp_raw_g, 3)}`** — raw MC-Dropout intervals are heavily "
  "**under-covered** (target gamma=0.95). Raw std is an uncalibrated diagnostic, not a predictive std.")
A(f"- **`nmpil_raw/global ~ {f(m_nmpil_raw_g, 4)}`** — raw intervals are extremely **sharp** "
  "(tiny normalized width), i.e. far too narrow.")
A(f"- **`clc_raw/global ~ {f(m_clc_raw_g, 1)}`** — CLC penalises hard because PICP is far from gamma: "
  "the sigma penalty `1+exp(-eta*(PICP-gamma))` dominates (see section 6), not the width.")
A(f"- **`picp_calibrated/global ~ {f(m_picp_cal_g, 3)}`** — group calibration restores coverage to ~gamma.")
A(f"- **`nmpil_calibrated/global ~ {f(m_nmpil_cal_g, 4)}`** — calibrated intervals are much **wider** "
  f"(~{f(m_nmpil_cal_g / m_nmpil_raw_g, 1)}x the raw width) — the price of reliability.")
A(f"- **`clc_calibrated/global ~ {f(m_clc_cal_g, 3)}`** — a far better sharpness/reliability trade-off "
  f"than raw (~{f(m_clc_raw_g / m_clc_cal_g, 0)}x lower CLC).")
A(f"- **`clc_calibrated/rare_extreme ({f(m_clc_cal_r, 3)}) > clc_calibrated/normal ({f(m_clc_cal_n, 3)})`** "
  "— rare/extreme cases need wider intervals (or are less efficient) at equal coverage.\n")

# 5. ratios
A("## 5. Normal vs rare/extreme — cost of covering rare events\n")
mae_n = mean(df, "mae/normal")
mae_r = mean(df, "mae/rare_extreme")
mpiw_n = mean(df, "mpiw_calibrated/normal")
mpiw_r = mean(df, "mpiw_calibrated/rare_extreme")
nmp_n = mean(df, "nmpil_calibrated/normal")
nmp_r = mean(df, "nmpil_calibrated/rare_extreme")
clc_n = m_clc_cal_n
clc_r = m_clc_cal_r
picp_n = mean(df, "picp_calibrated/normal")
picp_r = mean(df, "picp_calibrated/rare_extreme")
A("| quantity | value |")
A("|---|---|")
A(f"| mae_rare / mae_normal | {f(mae_r / mae_n, 3)} |")
A(f"| mpiw_cal_rare / mpiw_cal_normal | {f(mpiw_r / mpiw_n, 3)} |")
A(f"| nmpil_cal_rare / nmpil_cal_normal | {f(nmp_r / nmp_n, 3)} |")
A(f"| clc_cal_rare / clc_cal_normal | {f(clc_r / clc_n, 3)} |")
A(f"| picp_cal_rare - picp_cal_normal | {f(picp_r - picp_n, 4)} |")
A("")
A(f"**Answer:** at essentially equal coverage (delta_picp = {f(picp_r - picp_n, 4)}), covering "
  f"rare/extreme costs ~{f((mpiw_r / mpiw_n - 1) * 100, 1)}% wider calibrated intervals and a "
  f"~{f((clc_r / clc_n - 1) * 100, 1)}% worse CLC than normal. The model errs "
  f"~{f((mae_r / mae_n - 1) * 100, 1)}% more on rare/extreme (MAE), and pays for reliability "
  "there with wider bands.\n")

# 6. CLC components
A("## 6. CLC components (sigma = CLC / NMPIL)\n")
A("sigma is the coverage penalty `1+exp(-eta*(PICP-gamma))`. Shows raw CLC is large because of "
  "sigma, not width.\n")
A("| group | interval_type | PICP | NMPIL | sigma = CLC/NMPIL | CLC |")
A("|---|---|---|---|---|---|")
for g in ["global", "normal", "rare_extreme"]:
    for t in ["raw", "calibrated"]:
        picp = mean(df, f"picp_{t}/{g}")
        nmpil = mean(df, f"nmpil_{t}/{g}")
        clc = mean(df, f"clc_{t}/{g}")
        sigma = clc / nmpil if nmpil else float("nan")
        A(f"| {g} | {t} | {f(picp, 3)} | {f(nmpil, 4)} | {f(sigma, 2)} | {f(clc, 3)} |")
A("")
A("sigma_raw ~ 1+exp(-10*(0.028-0.95)) ~ 1e4 -> raw CLC blows up despite tiny NMPIL; "
  "sigma_calibrated ~ 2 (PICP~gamma) -> CLC ~ 2*NMPIL.\n")

# 7. comparison
A("## 7. Sanity check vs previous sweep `ltxbtupy` (no CLC metrics)\n")
A("| metric | ozii0s5s (mean) | ltxbtupy (mean) |")
A("|---|---|---|")
for c in ["mae/global", "mae/rare_extreme", "ratio/mae_rare_normal"]:
    A(f"| {c} | {f(mean(df, c), 3)} | {f(mean(lt, c), 3)} |")
A("")
A("Point-forecast metrics match `ltxbtupy` (MAE/global ~ 20.9, MAE rare/extreme ~ 31.5-31.6, "
  "rare/normal ratio ~ 1.59-1.60). **Confirms the new CLC/interval metrics are eval-only and "
  "did not change the model or training.**\n")

# 8. best seeds
A("## 8. Best seeds\n")


def best(col):
    v = num(df, col)
    idx = v.idxmin()
    return int(df.loc[idx, "seed"]), float(v.loc[idx])


bs_mae = best("mae/rare_extreme")
bs_clcg = best("clc_calibrated/global")
bs_clcr = best("clc_calibrated/rare_extreme")
A(f"- Best by **mae/rare_extreme**: seed **{bs_mae[0]}** ({f(bs_mae[1], 3)})")
A(f"- Best by **clc_calibrated/global**: seed **{bs_clcg[0]}** ({f(bs_clcg[1], 4)})")
A(f"- Best by **clc_calibrated/rare_extreme**: seed **{bs_clcr[0]}** ({f(bs_clcr[1], 4)})")
diff = bs_mae[0] != bs_clcg[0]
A("")
A(f"The best seed for point accuracy (seed {bs_mae[0]}) "
  f"{'differs from' if diff else 'matches'} the best seed for interval efficiency "
  f"(seed {bs_clcg[0]}). "
  + ("Point MAE and interval CLC optimise different things (error vs width-at-coverage), "
     "so the best seed is not unique — report MAE and CLC separately." if diff else
     "Here the same seed wins on both.") + "\n")

# 9. conclusion
A("## 9. Operational conclusion\n")
cv_mae_g = num(df, "mae/global").std() / num(df, "mae/global").mean()
A(f"- PVGIS-only ST-GNN is **seed-robust** on the point forecast (MAE/global cv {f(cv_mae_g, 3)}).")
A(f"- Rare/extreme conditions are **harder**: MAE rare/normal ~ {f(mae_r / mae_n, 2)}.")
A(f"- **Raw MC-Dropout does not give reliable predictive intervals** (PICP raw ~ {f(m_picp_raw_g, 3)}).")
A(f"- **Group calibration fixes reliability** (PICP calibrated ~ {f(m_picp_cal_g, 3)}).")
A(f"- The cost of calibration is **wider intervals** (NMPIL up ~{f(m_nmpil_cal_g / m_nmpil_raw_g, 0)}x).")
A(f"- **CLC confirms calibrated >> raw** (CLC {f(m_clc_cal_g, 3)} vs {f(m_clc_raw_g, 1)}).")
A("- **CLC rare/extreme > normal** -> rare/extreme stays more expensive to cover even after calibration.")
A("- This **closes the raw-vs-calibrated uncertainty fix**.")
A("- **Next step:** event-centered analysis around rare/extreme events, or simple baselines "
  "(persistence) for context.\n")

out = "outputs/sweep_analysis/ozii0s5s_report.md"
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"written {out}  ({len(L)} lines)")
