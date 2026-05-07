# SOTA References — Multi-Site PV Forecasting

Reference collection for thesis literature review and benchmarking.
All entries surveyed via web search (May 2026) for the multi-site PV
forecasting literature relevant to PhysiQ-PV (1116 distributed plants,
Piedmont 2019, hourly horizon, PVGIS-only inputs).

## Tier A — Most directly comparable

### Hasnat, Asadi, Alemazkoor (2025) — GAT generalized-horizon
- **Title**: A graph attention network framework for generalized-horizon multi-plant solar power generation forecasting using heterogeneous data
- **Journal**: Renewable Energy 243 (2025) 122520
- **DOI**: 10.1016/j.renene.2025.122520
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S096014812500182X
- **Affiliation**: University of Virginia
- **Dataset**: NREL Solar Power Data 2024 (synthetic from 2006 weather), 1146 PV plants across six eastern US states
- **Resolution**: 5 min, downsampled per horizon
- **Input**: 8 previous power output samples + Time-of-Day + selected weather
- **Architecture**: GAT, geographic distance-based static graph
- **Hardware**: NVIDIA RTX 4090
- **Capacity-normalised MAE**:
  - 30-min ahead: 0.0200 (UPV 0.0232, DPV 0.0169)
  - 3-hour ahead: 0.0366 (UPV 0.0393, DPV 0.0340)
  - Day-ahead:    0.0440 (UPV 0.0474, DPV 0.0407)
- **Notes**: Synthetic dataset (no real sensor noise / no degradation). Strongly autoregressive (lagged power as primary input). Closest setup to PhysiQ-PV in plant count and graph topology.

### Pinto, Marcjasz, Ferraz de Arruda et al. — GCLSTM / GCTrafo
- **Preprint**: arXiv:2107.13875
- **URL**: https://arxiv.org/abs/2107.13875
- **Dataset**: 304 real PV systems + 1000 simulated PV systems, Switzerland, 1 year, hourly
- **Architecture**: Graph-Convolutional LSTM and Graph-Convolutional Transformer
- **Horizons**: up to 6 hours ahead
- **Notes**: Production data only (no exogenous weather); models PV grid as a "dense network of virtual weather stations". Numerical results behind paywall in the published abstract.

### Cini et al. (2024) — large-scale PV with PV-only input
- Cited within Hasnat 2025 as forecasting 5016 plants, 30-min horizon, no exogenous data.
- **Notes**: Demonstrates scalability with autoregressive PV input alone.

## Tier B — Related GNN approaches

### ST-GNN with Fourier features (2025)
- **Title**: Spatio-temporal Graph Neural Network with Fourier features for multi-site photovoltaic power forecasting
- **Journal**: Electric Power Systems Research (2025)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0378779625007588
- **Highlights**: Hyper-variable graph + Fourier-based feature extraction. 1-step normalised RMSE 0.0267 / 0.0228. Outperforms LSTNet, Autoformer, MTGNN by up to 35.3%.

### DEST-GNN (2024)
- **Title**: DEST-GNN: A double-explored spatio-temporal graph neural network for multi-site intra-hour PV power forecasting
- **Journal**: Applied Energy (2024)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0306261924021275
- **Focus**: Intra-hour multi-site PV.

### GGNet (2024)
- **Title**: GGNet: A novel graph structure for power forecasting in renewable power plants considering temporal lead-lag correlations
- **Journal**: Applied Energy 364 (2024)
- **URL**: https://ideas.repec.org/a/eee/appene/v364y2024ics0306261924005774.html
- **Focus**: Dynamic graph capturing lead-lag correlations from airflow time differences.

### Verdone et al. — Explainable ST-GNN
- **Title**: Explainable Spatio-Temporal Graph Neural Networks for multi-site photovoltaic energy production
- **Journal**: Applied Energy 353 (2024)
- **URL**: https://www.sciencedirect.com/science/article/pii/S0306261923015155
- **Focus**: GNN + 1-D conv with site-to-site explainability.

### Zhang et al. — Spatiotemporal GNN performance prediction
- **Conference**: AAAI 2021
- **URL**: https://ojs.aaai.org/index.php/AAAI/article/view/17799
- **PDF**: https://cdn.aaai.org/ojs/17799/17799-13-21293-1-2-20210518.pdf
- **Focus**: Performance prediction (not directly forecasting).

### Simeunovic et al. — TS multi-window GAT
- Referenced in Hasnat 2025 [22]. Hundreds of PV sites, 4-6 h horizon.

### Zhu et al. — dynamic adjacency from airflow
- Referenced in Hasnat 2025 [24]. 8-plant system; conceptually relevant but small scale.

## Tier C — Physics-informed / sky imagery

### RTI-Net (2025)
- **Title**: RTI-Net: Physics-informed deep learning for photovoltaic power forecasting
- **Journal**: Renewable Energy (2025)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0960148125018166
- **Approach**: Radiation Transmission Index between multi-scale pooled and original sky images. Physics constraint via cloud attenuation modeling.
- **Notes**: Single-site, ultra-short horizon. Requires per-plant sky cameras — not scalable to distributed fleets.

### SkyGPT (2024)
- **Title**: SkyGPT: Probabilistic ultra-short-term solar forecasting using synthetic sky images from physics-constrained VideoGPT
- **Journal**: Solar Energy Advances (2024)
- **URL**: https://www.sciencedirect.com/science/article/pii/S2666792424000106
- **Repo**: https://github.com/yuhao-nie/SkyGPT
- **Approach**: Physics-constrained video prediction of cloud motion. Single-site, ultra-short horizon.

### Physics-informed day-ahead with clear-sky template (2025)
- **Journal**: Energy and AI / Smart Energy (2025)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S2950487225000224
- **Approach**: Embeds clear-sky template for seasonal trend mitigation. Day-ahead. Direct conceptual analogue of PhysiQ-PV's GHI clear-sky residual constraint.

### MIPV-NWP-PINN (2025 preprint)
- **URL**: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4439/egusphere-2025-4439.pdf
- **Approach**: Multi-scale PINN combining NWP with physical PV model.

## Tier D — Surveys and reviews

### Yang et al. — solar irradiance forecasting systematic review (2025)
- **Title**: A systematic review of solar irradiance forecasting across time horizons using physical, satellite, and AI-based methods
- **URL**: https://www.sciencedirect.com/science/article/pii/S2772940025000499

### KimMeen — Awesome GNN4TS (TPAMI 2024)
- **Repo**: https://github.com/KimMeen/Awesome-GNN4TS
- **Use**: Curated list of GNN-for-time-series resources.

### SLR Spatio-Temporal GNN (FlaGer99)
- **Repo**: https://github.com/FlaGer99/SLR-Spatio-Temporal-GNN
- **Companion paper**: A Systematic Literature Review of Spatio-Temporal Graph Neural Network Models for Time Series Forecasting and Classification.

### Master thesis (TU Wien 2023)
- **PDF**: https://repositum.tuwien.at/bitstream/20.500.12708/177395/1/Woschitz%20Martin%20-%202023%20-%20Spatio-temporal%20PV%20forecasting%20with%20graph%20neural...pdf
- **Notes**: Spatio-temporal PV forecasting with graph neural networks. Comparable thesis-level reference.

## Tier E — Statistical / classical baselines (for context)

- Holt-Winters single-site: nMAE 3.7%
- Holt's method: nMAE 5.96%
- Modified simple exponential smoothing: nMAE 0.84% (single short-horizon site)
- Source: surveyed in Tandfonline 2024 review (https://www.tandfonline.com/doi/full/10.1080/00207543.2023.2269565) and others.

## Industry / data references

### PVGIS (JRC)
- **Portal**: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en
- **User manual**: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/pvgis-user-manual_en
- **Status report 2025**: https://publications.jrc.ec.europa.eu/repository/handle/JRC145327
- **Notes**: Public ERA5-derived global irradiance / temperature reanalysis. Used as primary weather input in PhysiQ-PV.

### NREL — Solar Power Data Set
- Cited in Hasnat 2025 [32]. Synthetic 5-min PV output for all 50 US states using 2006 weather. 1146-plant subset used in Hasnat.
- Original sub-hour solar data: Hummon, Ibanez, Brinkman, Lew (NREL TR 2012).

### NREL Solar Industry updates
- Spring 2024: https://docs.nrel.gov/docs/fy24osti/90042.pdf
- Fall 2024: https://docs.nrel.gov/docs/fy25osti/92257.pdf
- Spring 2025: https://docs.nrel.gov/docs/fy25osti/95135.pdf

### IEA Renewables outlooks
- Renewables 2024: https://iea.blob.core.windows.net/assets/17033b62-07a5-4144-8dd0-651cdb6caa24/Renewables2024.pdf
- Renewables 2025: https://iea.blob.core.windows.net/assets/48eccb83-984c-45d2-bf78-67a61e88d241/Renewables2025.pdf

## Comparison summary (PV nMAE landscape)

| Method | n plants | Region | Horizon | nMAE PV | Data type | Notes |
|--------|----------|--------|---------|---------|-----------|-------|
| Hasnat GAT 2025 | 1146 | USA | 30-min | 2.0% | synthetic + lagged power | best fleet-scale |
| Hasnat GAT 2025 | 1146 | USA | 3 h    | 3.7% | synthetic + lagged power | |
| Hasnat GAT 2025 | 1146 | USA | day    | 4.4% | synthetic + lagged power | |
| ST-GNN Fourier 2025 | multi | varied | 1-step | RMSE 0.027 (norm) | varied | |
| GCLSTM/GCTrafo | 304+1000 | CH | 6 h    | ~6-8% est | real + sim, PV-only | |
| RTI-Net | 1 | varied | 5-15 min | 1-3% | sky images | not scalable |
| Holt-Winters | 1 | varied | short | 3.7% | persistence | classical |
| **PhysiQ-PV (clearsky+peak0.25)** | **1116** | **Piemonte** | **1 h** | **8.3%** | **real degraded fleet, meteo-only, no lagged power** | this work |
| PhysiQ-PV (high-QS subset) | ~683 | Piemonte | 1 h | 7.1% | filtered to high-QS plants | apples-closer to literature |

## Positioning notes

- Direct nMAE comparison vs Hasnat 2025 is not apples-to-apples: their dataset is **synthetic** (no degradation, no sensor noise) and uses **autoregressive lagged PV power** as primary input. Removing either would degrade their reported numbers.
- PhysiQ-PV addresses a different operating point: **real distributed fleet with sensor and degradation noise, meteo-only inputs, hourly horizon**. The contribution is the combination of physics-informed dual-head (η for PV + clear-sky residual for GHI), data-centric input (m1..m5 separate quality components), and continual-learning safety infrastructure (QS gating + ADWIN drift in `online_loop` / `agent/cycle.py`).
- Closing the gap to ~3–5% nMAE is achievable by adopting lagged power input and outlier filtering (work in progress on `feat/improvements-fleet-2025`).

## Search provenance

- Primary search: WebSearch queries on multi-site GNN PV forecasting, fleet nMAE benchmarks, physics-informed PV, PVGIS/ERA5 inputs.
- Hasnat 2025 paper text extracted from local PDF copy and used to verify numerical results.
- All other entries summarised from abstracts; full-text access is paywalled where noted.
