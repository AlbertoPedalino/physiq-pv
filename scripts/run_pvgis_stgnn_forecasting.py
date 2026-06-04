"""
Run the PVGIS-only ST-GNN forecasting pipeline (thin wrapper).

Delegates to `physiq_pv.experiments.pvgis_stgnn_runner` so this script and
`python main.py --mode pvgis_stgnn` share one implementation (feature-set
ablation, optional W&B, MC-Dropout predisposition, stratified eval).

PVGIS-only input (11 features, subset-able via --feature-set): NO ENERGIA,
NO observed plant production, NO QS, NO kWp/UPN. Anomaly labels are used ONLY
for stratified evaluation.

Example (server):

    PYTHONPATH=$PWD python scripts/run_pvgis_stgnn_forecasting.py \\
      --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \\
      --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \\
      --test-year 2019 \\
      --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q099/pvgis_climatology_scores.csv \\
      --out-dir outputs/pvgis_stgnn_forecasting_2019 \\
      --seq-len 24 --horizon 1 --target-variable pv_power_output \\
      --model-type stgnn --feature-set full --epochs 10 --batch-size 8
"""

from __future__ import annotations

from physiq_pv.experiments.pvgis_stgnn_runner import main

if __name__ == "__main__":
    main()
