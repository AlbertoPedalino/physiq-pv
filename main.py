"""
PhysiQ-PV end-to-end pipeline entry point.

Runs:
  1. Synthetic dataset generation
  2. QS computation + scenario detection verification
  3. ST-GNN training
  4. Online agentic loop (ATSF: perception->planning->action->reflection)
  5. Summary report
"""
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
from physiq_pv.data.quality_score import compute_qs, diagnose_scenarios
from physiq_pv.agent.cycle import PhysiQAgent
from train import train
from online_loop import run_online


def main() -> None:
    sep = "=" * 62

    # ------------------------------------------------------------------ #
    # 1. Synthetic data
    # ------------------------------------------------------------------ #
    print(sep)
    print("PhysiQ-PV -- End-to-End Pipeline (synthetic data)")
    print(sep)
    print("\n[1] Generating synthetic dataset...")
    ds = generate_synthetic_dataset(seed=42)
    print(f"    {ds.sizes['plant']} plants x {ds.sizes['time']} timesteps")

    # ------------------------------------------------------------------ #
    # 2. Quality Score + scenario verification
    # ------------------------------------------------------------------ #
    print("\n[2] Quality Score diagnostics:")
    qs = compute_qs(ds)
    fleet_qs = float(qs.mean(skipna=True))
    print(f"    QS shape={qs.shape}  fleet_mean={fleet_qs:.3f}")

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
    # 3. ST-GNN training
    # ------------------------------------------------------------------ #
    print("\n[3] Training ST-GNN (3 epochs, 50 steps/epoch on synthetic data)...")
    model, loss_history, updater, edge_index, edge_weight = train(
        ds=ds, n_epochs=3, max_steps_per_epoch=50
    )
    curve = " -> ".join(f"{l:.4f}" for l in loss_history)
    print(f"    Loss curve: {curve}")

    # ------------------------------------------------------------------ #
    # 4. Online agentic loop (ATSF)
    # ------------------------------------------------------------------ #
    print("\n[4] Online agentic loop (window=720h, stride=1000h)...")
    agent = PhysiQAgent(n_clusters=4, drift_window=720)

    # Train causal classifier once on full dataset before streaming starts
    print("    Training causal classifier (MultiROCKET on synthetic QS)...")
    clf_summary = agent.train_classifier(ds)
    if clf_summary["trained"]:
        print(f"    Samples: {clf_summary['n_samples']}  classes: {clf_summary['class_counts']}")
    else:
        print(f"    Fallback rule-based: {clf_summary['reason']}")

    history = run_online(
        ds=ds,
        model=model,
        updater=updater,
        edge_index=edge_index,
        edge_weight=edge_weight,
        agent=agent,
        window_size=720,
        stride=1000,
        verbose=True,
    )
    n_retrained = sum(1 for r in history if r.get("action") == "retrain_triggered")
    print(f"    Steps: {len(history)}  retraining triggered: {n_retrained}")

    # ------------------------------------------------------------------ #
    # 5. Summary
    # ------------------------------------------------------------------ #
    print(f"\n{sep}")
    print("Summary")
    print(sep)
    for scenario, result in detections.items():
        status = "DETECTED" if result["detected"] else "MISSED  "
        print(f"  {status}  {scenario}")
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Final train loss : {loss_history[-1]:.4f}")
    print(f"  Online loop steps: {len(history)}")
    print(f"\n[Done]\n")


if __name__ == "__main__":
    main()
