"""Generate the paper-style report for sweep 7c8llckm + a direct comparison with the
calibrated sweeps ozii0s5s / ltxbtupy. Pure post-processing of the dumped CSVs;
launches no experiments and touches no model/data."""
import numpy as np
import pandas as pd

df = pd.read_csv("outputs/sweep_analysis/7c8llckm.csv").sort_values("seed").reset_index(drop=True)
oz = pd.read_csv("outputs/sweep_analysis/ozii0s5s.csv")
lt = pd.read_csv("outputs/sweep_analysis/ltxbtupy.csv")


def num(s, c):
    return pd.to_numeric(s[c], errors="coerce") if c in s else pd.Series(dtype=float)


def mean(d, c):
    v = num(d, c).dropna()
    return float(v.mean()) if len(v) else float("nan")


def f(x, nd=4):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "—"


L = []
A = L.append
A("# Sweep `7c8llckm` — PVGIS-only ST-GNN: paper-style uncertainty report\n")
A("Paper-style protocol (uncertainty-aware rainfall prediction): MC-Dropout in test, "
  "predictive intervals built **directly from the MC sample quantiles** (q0.025/q0.975), "
  "**no post-hoc calibration**. PICP is evaluated, not forced. Source CSV: "
  "`outputs/sweep_analysis/7c8llckm.csv`.\n")

# 1. Setup
A("## 1. Setup\n")
A("| param | value |")
A("|---|---|")
for k, v in [("sweep_id", "7c8llckm"), ("train_years", "2016,2017,2018"), ("calibration", "none"),
             ("test_year", "2019"), ("MC Dropout", "true (mc_samples=20)"),
             ("primary PI", "MC sample quantiles q0.025 / q0.975"),
             ("Gaussian band", "diagnostic only (mean ± 1.96·std_raw)"),
             ("post-hoc calibration", "OFF (enable_posthoc_calibration=False)"),
             ("feature_set", "full"), ("target", "pv_power_output"), ("model_type", "stgnn"),
             ("batch_size", "16"), ("epochs", "5"), ("lr", "0.001"), ("dropout", "0.2"),
             ("coverage_target (gamma)", "0.95"), ("clc_eta (eta)", "10"), ("seeds", "1,2,3,4,5")]:
    A(f"| {k} | {v} |")
A("")

# 2. Per-seed
A("## 2. Per-seed results\n")
cols = [("seed", "seed", 0), ("run_name", "name", None),
        ("mae/global", "mae/global", 3), ("mae/normal", "mae/normal", 3),
        ("mae/rare", "mae/rare_extreme", 3), ("ratio_mae", "ratio/mae_rare_normal", 3),
        ("rmse/global", "rmse/global", 3), ("rmse/rare", "rmse/rare_extreme", 3),
        ("std_norm", "uncertainty/mean_std_normal", 3), ("std_rare", "uncertainty/mean_std_rare_extreme", 3),
        ("std_ratio", "uncertainty/ratio_rare_normal", 3),
        ("picp_pi/g", "picp_pi/global", 3), ("picp_pi/n", "picp_pi/normal", 3),
        ("picp_pi/r", "picp_pi/rare_extreme", 3), ("nmpil_pi/g", "nmpil_pi/global", 4),
        ("clc_pi/g", "clc_pi/global", 3), ("clc_pi/n", "clc_pi/normal", 3), ("clc_pi/r", "clc_pi/rare_extreme", 3),
        ("picp_gauss/g", "picp_gaussian/global", 3), ("clc_gauss/g", "clc_gaussian/global", 3)]
A("| " + " | ".join(h for h, _, _ in cols) + " |")
A("|" + "---|" * len(cols))
for _, r in df.iterrows():
    cells = []
    for h, c, nd in cols:
        if c not in df.columns:
            cells.append("—"); continue
        v = r[c]
        cells.append(str(v) if nd is None else (f(float(v), nd) if pd.notna(v) else "—"))
    A("| " + " | ".join(cells) + " |")
A("")


# 3. Aggregates
def agg_block(title, metrics):
    A(f"### {title}\n")
    A("| metric | mean | std | cv |")
    A("|---|---|---|---|")
    for m in metrics:
        v = num(df, m).dropna()
        if len(v) == 0:
            A(f"| {m} | — | — | — |"); continue
        mu, sd = v.mean(), v.std()
        A(f"| {m} | {f(mu)} | {f(sd)} | {f(sd/mu, 3) if mu else '—'} |")
    A("")


A("## 3. Aggregated (mean / std / cv), n=5\n")
agg_block("Point forecast", ["mae/global", "mae/normal", "mae/rare_extreme", "ratio/mae_rare_normal",
                             "rmse/global", "rmse/normal", "rmse/rare_extreme", "ratio/rmse_rare_normal"])
agg_block("Uncertainty (MC std)", ["uncertainty/mean_std_normal", "uncertainty/mean_std_rare_extreme",
                                   "uncertainty/ratio_rare_normal"])
agg_block("PI quantile metrics (primary)",
          ["picp_pi/global", "picp_pi/normal", "picp_pi/rare_extreme",
           "mpiw_pi/global", "mpiw_pi/normal", "mpiw_pi/rare_extreme",
           "nmpil_pi/global", "nmpil_pi/normal", "nmpil_pi/rare_extreme",
           "clc_pi/global", "clc_pi/normal", "clc_pi/rare_extreme"])
agg_block("Gaussian diagnostic metrics",
          ["picp_gaussian/global", "nmpil_gaussian/global", "clc_gaussian/global"])

# precompute
m_picp = mean(df, "picp_pi/global")
m_picp_g = mean(df, "picp_gaussian/global")
mae_n, mae_r = mean(df, "mae/normal"), mean(df, "mae/rare_extreme")
std_n, std_r = mean(df, "uncertainty/mean_std_normal"), mean(df, "uncertainty/mean_std_rare_extreme")
clc_pn, clc_pr = mean(df, "clc_pi/normal"), mean(df, "clc_pi/rare_extreme")
picp_pn, picp_pr = mean(df, "picp_pi/normal"), mean(df, "picp_pi/rare_extreme")

# 4. Interpretation
A("## 4. Paper-style interpretation\n")
A(f"- **`picp_pi/global ≈ {f(m_picp,3)}`** — MC Dropout is strongly **under-dispersed**: the empirical "
  "MC quantile interval covers only ~3% of the truth vs the 0.95 target.")
A("- Intervals are extremely **sharp** (tiny NMPIL) but **not reliable**.")
A(f"- **Gaussian diagnostic ≈ PI quantile** (`picp_gaussian/global ≈ {f(m_picp_g,3)}` vs "
  f"`picp_pi/global ≈ {f(m_picp,3)}`): the problem is NOT the Gaussian assumption — the **MC spread "
  "itself is too narrow**.")
A(f"- Rare/extreme MAE ≈ **{f(mae_r/mae_n,2)}×** normal.")
A(f"- But MC std rises only ~**{f(std_r/std_n,2)}×** (rare/normal) → the spread **does not track** the "
  "error increase on rare/extreme.")
A(f"- **PICP rare ≈ normal** ({f(picp_pr,3)} vs {f(picp_pn,3)}) → MC Dropout does **not discriminate** "
  "coverage across strata.")
A(f"- **CLC rare > normal** ({f(clc_pr,3)} vs {f(clc_pn,3)}) → rare/extreme stays more expensive in the "
  "sharpness/reliability trade-off.\n")

# 5. Direct comparison
A("## 5. Direct comparison: paper-style vs calibrated sweeps\n")
A("| sweep | protocol | train | calibration | test | primary PI | mae/global | mae/rare | ratio r/n | "
  "primary PICP | NMPIL/CLC (primary) | interpretation |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
# 7c8llckm
A(f"| `7c8llckm` | paper-style main | 2016,2017,2018 | none | 2019 | MC quantile PI | "
  f"{f(mean(df,'mae/global'),2)} | {f(mae_r,2)} | {f(mean(df,'ratio/mae_rare_normal'),2)} | "
  f"picp_pi **{f(m_picp,3)}** | nmpil {f(mean(df,'nmpil_pi/global'),4)} / clc {f(mean(df,'clc_pi/global'),2)} | "
  "honest uncalibrated coverage |")
# ozii0s5s
A(f"| `ozii0s5s` | post-hoc calibrated variant | 2016,2017 | 2018 (group) | 2019 | calibrated band mean±k·std | "
  f"{f(mean(oz,'mae/global'),2)} | {f(mean(oz,'mae/rare_extreme'),2)} | {f(mean(oz,'ratio/mae_rare_normal'),2)} | "
  f"coverage_95_calibrated **{f(mean(oz,'coverage_95_calibrated/global'),3)}** | "
  f"clc_calibrated {f(mean(oz,'clc_calibrated/global'),3)} (k≈{f(mean(oz,'calibration/factor_global'),0)}) | "
  "coverage forced to target |")
# ltxbtupy
A(f"| `ltxbtupy` | legacy calibrated | 2016,2017 | 2018 (group) | 2019 | gaussian raw + calibrated band | "
  f"{f(mean(lt,'mae/global'),2)} | {f(mean(lt,'mae/rare_extreme'),2)} | {f(mean(lt,'ratio/mae_rare_normal'),2)} | "
  f"coverage_95_calibrated **{f(mean(lt,'coverage_95_calibrated/global'),3)}** | "
  f"(raw {f(mean(lt,'coverage_95_raw/global'),3)}, k≈{f(mean(lt,'calibration/factor_global'),0)}) | "
  "coverage forced to target |")
A("")
A(f"Key contrast: the **~3% coverage is identical** whether read as `picp_pi` here "
  f"({f(m_picp,3)}) or as `coverage_95_raw` in `ltxbtupy` ({f(mean(lt,'coverage_95_raw/global'),3)}). "
  "The **~0.95 coverage only ever came from post-hoc calibration on 2018** "
  f"(k≈{f(mean(oz,'calibration/factor_global'),0)}), never from the model's own intervals.\n")

# 6. Conclusion
A("## 6. Operational conclusion\n")
A("- The paper-style protocol is **clean**: no post-hoc calibration, no calibration year, no `k`/`*_calibrated` keys logged.")
A("- MC Dropout produces **sharp but under-reliable** intervals (PICP ≈ 0.028, stable across 5 seeds).")
A(f"- Adding 2018 to training **slightly improves** the point forecast (mae/global "
  f"{f(mean(df,'mae/global'),2)} vs {f(mean(oz,'mae/global'),2)} for the train-2016,2017 calibrated sweep), "
  "but does **not** fix under-dispersion.")
A("- The 95% coverage exists **only** in the post-hoc calibrated variant, **not** in the paper-style protocol.")
A("- **Next step:** Deep Ensemble — check whether independently trained seed models give larger "
  "per-sample dispersion (and thus higher PICP) than a single MC-Dropout model.\n")

out = "outputs/sweep_analysis/7c8llckm_report.md"
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")

# compact comparison CSV for the thesis
cmp = pd.DataFrame([
    {"sweep": "7c8llckm", "protocol": "paper_style_main", "train": "2016,2017,2018", "calibration": "none",
     "test": 2019, "primary_PI": "MC_quantile", "mae_global": round(mean(df, "mae/global"), 3),
     "mae_rare": round(mae_r, 3), "ratio_rare_normal": round(mean(df, "ratio/mae_rare_normal"), 3),
     "primary_PICP": round(m_picp, 4), "primary_CLC": round(mean(df, "clc_pi/global"), 3),
     "k_global": None},
    {"sweep": "ozii0s5s", "protocol": "posthoc_calibrated", "train": "2016,2017", "calibration": "2018_group",
     "test": 2019, "primary_PI": "calibrated_band", "mae_global": round(mean(oz, "mae/global"), 3),
     "mae_rare": round(mean(oz, "mae/rare_extreme"), 3), "ratio_rare_normal": round(mean(oz, "ratio/mae_rare_normal"), 3),
     "primary_PICP": round(mean(oz, "coverage_95_calibrated/global"), 4),
     "primary_CLC": round(mean(oz, "clc_calibrated/global"), 3), "k_global": round(mean(oz, "calibration/factor_global"), 2)},
    {"sweep": "ltxbtupy", "protocol": "legacy_calibrated", "train": "2016,2017", "calibration": "2018_group",
     "test": 2019, "primary_PI": "calibrated_band", "mae_global": round(mean(lt, "mae/global"), 3),
     "mae_rare": round(mean(lt, "mae/rare_extreme"), 3), "ratio_rare_normal": round(mean(lt, "ratio/mae_rare_normal"), 3),
     "primary_PICP": round(mean(lt, "coverage_95_calibrated/global"), 4),
     "primary_CLC": None, "k_global": round(mean(lt, "calibration/factor_global"), 2)},
])
cmp.to_csv("outputs/sweep_analysis/paper_style_vs_calibrated_comparison.csv", index=False)

print(f"written {out} ({len(L)} lines)")
print("written outputs/sweep_analysis/paper_style_vs_calibrated_comparison.csv")
