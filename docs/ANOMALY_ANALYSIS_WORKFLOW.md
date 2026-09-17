# Anomaly-analysis workflow

All notebooks below are post-processing only. They reuse the frozen MTGFlow,
STGAN and direct SDE-Net outputs and do not start training.

## Required inputs

- `outputs/pvgis_mtgflow/downstream_dense/seed_15/anomaly_scores.csv`
- exact MTGFlow per-site training quantiles or `train_scores.csv` files
- `outputs/pvgis_stgan/paper_reference/seed_20/anomaly_scores.csv`
- the direct SDE-Net `predictions.csv` containing horizons t+1, ..., t+6
- the PVGIS 2019 NetCDF for spatial maps;
- the same NetCDF, or the prepared STGAN manifest, for the data-quality mask

Paths can be overridden through the environment variables documented in the
configuration cell of each notebook.

## Execution order

1. `spatial_anomaly_comparison_mtgflow_stgan.ipynb`
   - regional and reference-KNN maps;
   - MTGFlow and STGAN anomaly intensity/frequency;
   - SDE-Net RMSE maps at t+1 and t+6;
   - neighbourhood anomaly-share and RMSE timelines;
   - April, June and July event windows.

2. `anomaly_threshold_sensitivity_mtgflow_stgan.ipynb`
   - MTGFlow IQR-k sensitivity at t+1 and t+6;
   - STGAN global top-K sensitivity recalculated after removing regional solar
     dropouts and their immediate recovery;
   - common daytime coordinate domain for the two detectors;
   - normal/rare MAE and RMSE;
   - normal/rare group size and fraction;
   - Jaccard, reciprocal capture and daily correlation at the frozen references;
   - frozen references `k=1.5` and top 1%.

3. `mtgflow_pointwise_posthoc_sdenet.ipynb` and
   `stgan_pointwise_posthoc_sdenet.ipynb`
   - pointwise detector join on the direct SDE-Net `predictions.csv` produced by
     `pvgis_sde_pipeline.ipynb` (training only);
   - normal/rare error and uncertainty suite at t+1,...,t+6, written to a
     detector-specific evaluation directory.

## Interpretation rules

- Detector maps refer to the target timestamp and are independent of forecast
  horizon; forecast-error maps are separate at t+1 and t+6.
- Spatial maps retain one pixel/marker per PVGIS location. Neighbourhood
  aggregation is reported separately and never replaces spatial detail.
- MTGFlow and STGAN continuous intensities use different scales and must not be
  compared numerically as if they were calibrated probabilities.
- Threshold curves on 2019 are sensitivity analysis only. A different final
  cutoff must be selected on validation and frozen before test evaluation.
