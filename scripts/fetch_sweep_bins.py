"""Fetch bin counts from sweep runs."""
import sys
import wandb

api = wandb.Api()
entity = "albertopedalino-politecnico-di-torino"
project = "PhysiQ-PV"

for sweep_id in sys.argv[1:]:
    print(f"\n===== SWEEP {sweep_id} =====")
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    for r in sweep.runs:
        cfg = dict(r.config)
        s = r.summary
        wd = cfg.get("window_days", "N/A")
        seed = cfg.get("seed", "N/A")
        print(f"\n  {r.name} | wd={wd} seed={seed}")
        print(f"  {'bin':<12} {'mae':>8} {'rmse':>8} {'count':>8} {'wmae_across':>12} {'total_count':>12}")
        for b in ["0_20", "20_40", "40_60", "60_80", "80_100", "over_100"]:
            mae = s.get(f"bin_{b}_mae", "—")
            rmse = s.get(f"bin_{b}_rmse", "—")
            count = s.get(f"bin_{b}_final_count", s.get(f"bin_{b}_mean_count", "—"))
            wmae = s.get(f"bin_{b}_weighted_mean_mae", "—")
            total = s.get(f"bin_{b}_total_count", "—")
            mae_s = f"{mae:.5f}" if isinstance(mae, (int, float)) else str(mae)
            rmse_s = f"{rmse:.5f}" if isinstance(rmse, (int, float)) else str(rmse)
            wmae_s = f"{wmae:.5f}" if isinstance(wmae, (int, float)) else str(wmae)
            count_s = str(int(count)) if isinstance(count, (int, float)) else str(count)
            total_s = str(int(total)) if isinstance(total, (int, float)) else str(total)
            print(f"  {b:<12} {mae_s:>8} {rmse_s:>8} {count_s:>8} {wmae_s:>12} {total_s:>12}")
