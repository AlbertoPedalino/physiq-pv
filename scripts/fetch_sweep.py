"""Fetch a single W&B sweep."""
import sys
import wandb

api = wandb.Api()
entity = "albertopedalino-politecnico-di-torino"
project = "PhysiQ-PV"
sweep_id = sys.argv[1] if len(sys.argv) > 1 else "z3sckwmj"

print(f"\n===== SWEEP {sweep_id} =====")
sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
print(f"State: {sweep.state}")
print(f"Runs: {len(sweep.runs)}")

for r in sweep.runs:
    cfg = dict(r.config)
    s = r.summary
    print(f"\n  Run: {r.name} | state={r.state}")
    print(f"    window_days={cfg.get('window_days','N/A')} | seed={cfg.get('seed','N/A')} | pv_norm_mode={cfg.get('pv_norm_mode','N/A')}")
    print(f"    final_mae={s.get('final_mae','N/A')} | final_rmse={s.get('final_rmse','N/A')} | initial_mae={s.get('initial_mae','N/A')} | n_windows={s.get('n_windows','N/A')}")

    for b in ["0_20","20_40","40_60","60_80","80_100","over_100"]:
        mae_k = f"bin_{b}_mae"
        if mae_k in s and s[mae_k] is not None:
            rmse_k = f"bin_{b}_rmse"
            print(f"    bin_{b}: mae={s.get(mae_k):.5f} rmse={s.get(rmse_k,'N/A'):.5f}" if isinstance(s.get(rmse_k),(int,float)) else f"    bin_{b}: mae={s.get(mae_k)} rmse={s.get(rmse_k)}")

    for b in ["0_20","20_40","40_60","60_80","80_100","over_100"]:
        wm = f"bin_{b}_weighted_mean_mae"
        if wm in s and s[wm] is not None:
            v = s.get(wm)
            w = s.get(f"bin_{b}_worst_mae", "N/A")
            fm = s.get(f"bin_{b}_final_mae", "N/A")
            try:
                print(f"    bin_{b}_across: wmae={v:.5f} worst={w:.5f} final={fm:.5f}")
            except (TypeError, ValueError):
                print(f"    bin_{b}_across: wmae={v} worst={w} final={fm}")

    hist = r.history(keys=["window_id","window_mae","window_rmse"], pandas=True)
    if hist is not None and len(hist) > 0:
        print(f"    trajectory: ", end="")
        for _, row in hist.iterrows():
            wid = row.get("window_id","?")
            wmae = row.get("window_mae","?")
            if isinstance(wmae, float):
                print(f"w{int(wid) if isinstance(wid,float) else wid}={wmae:.4f} ", end="")
            else:
                print(f"w{wid}={wmae} ", end="")
        print()
