"""
PhysiQ-PV end-to-end pipeline entry point.

Runs:
  1. Real dataset loading (Piedmont 2019, PVGIS-aligned)
  2. QS computation per (plant, time) + scenario detection
  3. ST-GNN training on real data
  4. Online agentic loop (ATSF: perception->planning->action->reflection)
  5. Summary report
"""
import xarray as xr
from physiq_pv.data.quality_score import compute_qs, diagnose_scenarios
from physiq_pv.agent.cycle import PhysiQAgent
from train import train
from online_loop import run_online


def main() -> None:
    sep = "=" * 62

    # ------------------------------------------------------------------ #
    # 1. Real data (Piedmont 2019 with PVGIS)
    # ------------------------------------------------------------------ #
    print(sep)
    print("PhysiQ-PV -- End-to-End Pipeline (real Piedmont 2019 data)")
    print(sep)
    print("\n[1] Loading real dataset (PVGIS-aligned 2019)...")
    ds = xr.open_dataset('data/real_data_dataset.nc')
    print(f"    {ds.sizes['plant']} plants x {ds.sizes['time']} timesteps (2019-01-03 to 2019-12-31)")
    print(f"    Variables: {list(ds.data_vars.keys())} [ENERGIA, pvgis_ref, temperature_2m]")

    # ------------------------------------------------------------------ #
    # 2. Quality Score (per-plant per-timestamp) + scenario detection
    # ------------------------------------------------------------------ #
    print("\n[2] Quality Score computation (per-plant per-time):")
    qs = compute_qs(ds)
    qs_valid = qs.values[~qs.values.isnan()]
    fleet_qs = float(qs.mean(skipna=True))
    print(f"    QS shape={qs.shape} (plant={ds.sizes['plant']}, time={ds.sizes['time']})")
    print(f"    Fleet QS mean={fleet_qs:.3f}, median={float(qs.median(skipna=True)):.3f}")
    print(f"    Valid data: {len(qs_valid):,} ({len(qs_valid)/qs.size*100:.1f}%)")

    print("\n    Injected scenario detection:")
    detections = diagnose_scenarios(ds)
    all_detected = True
    for scenario, result in detections.items():
        status = "[OK]" if result["detected"] else "[MISS]"
        print(f"    {status}  {scenario}: {result['description']}")
        if not result["detected"]:
            all_detected = False
    print(f"\n    All scenarios detected: {'YES' if all_detected else 'NO'}")

    # ------------------------------------------------------------------ #
    # 3. ST-GNN training (on real data)
    # ------------------------------------------------------------------ #
    print("\n[3] Training ST-GNN (5 epochs, 30 steps/epoch on real 2019 data)...")
    model, loss_history, updater, edge_index, edge_weight = train(
        ds=ds, n_epochs=5, max_steps_per_epoch=30
    )
    curve = " -> ".join(f"{l:.4f}" for l in loss_history)
    print(f"    Loss curve: {curve}")

    # ------------------------------------------------------------------ #
    # 4. Online agentic loop (ATSF)
    # ------------------------------------------------------------------ #
    print("\n[4] Online agentic loop (window=720h, stride=168h) with QS monitoring...")
    agent = PhysiQAgent(n_clusters=4, drift_window=720)

    # Train causal classifier once on full dataset before streaming starts
    print("    Training causal classifier (MultiROCKET on real QS per-plant-per-time)...")
    clf_summary = agent.train_classifier(ds)
    if clf_summary["trained"]:
        print(f"    Samples: {clf_summary['n_samples']}  classes: {clf_summary['class_counts']}")
    else:
        print(f"    Fallback rule-based: {clf_summary['reason']}")

    print("    Each window: QS computed per (plant, time) → agent decides retraining")
    history = run_online(
        ds=ds,
        model=model,
        updater=updater,
        edge_index=edge_index,
        edge_weight=edge_weight,
        agent=agent,
        window_size=720,
        stride=168,  # Weekly stride for real data
        verbose=True,
    )
    n_retrained = sum(1 for r in history if r.get("action") == "retrain_triggered")
    print(f"    Steps: {len(history)}  retraining triggered: {n_retrained}")

    # ------------------------------------------------------------------ #
    # 5. Summary
    # ------------------------------------------------------------------ #
    print(f"\n{sep}")
    print("Summary - Real Data Pipeline (Piedmont 2019)")
    print(sep)
    print(f"\n  Dataset: Piedmont energy 2019 + PVGIS 2019 (PVGIS-aligned)")
    print(f"  Plants: {ds.sizes['plant']}, Timesteps: {ds.sizes['time']}")
    print(f"  QS: {len(qs_valid):,} valid per-plant-per-time measurements")
    print(f"  Fleet QS: mean={fleet_qs:.3f}")
    print(f"\n  Scenario detection:")
    for scenario, result in detections.items():
        status = "DETECTED" if result["detected"] else "MISSED  "
        print(f"    {status}  {scenario}")
    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Loss: {loss_history[-1]:.4f} (final)")
    print(f"  Online steps: {len(history)}, retraining: {n_retrained}x")
    print(f"\n✅ QS applied to EVERY (plant, time) during online loop!\n")


if __name__ == "__main__":
    main()
