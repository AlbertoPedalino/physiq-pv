# CL / CDA — Riferimenti bibliografici per la tesi

## CDA Metodologici (top-tier, A*)

### CoTTA — Continual Test-Time Domain Adaptation (CVPR 2022)
- **Titolo:** Continual Test-Time Domain Adaptation
- **Autori:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai
- **Venue:** CVPR 2022 (A*)
- **Metodo:** Augmentation-averaged pseudo-labels + stochastic weight restoration per prevenire error accumulation e forgetting durante shift continuo a test-time.
- **Rilevanza:** Primo framework CDA rigoroso. Mostra che DA standard accumula errori se applicata continuamente — serve meccanismo anti-forgetting.
- **Codice:** https://github.com/qinenergy/cotta
- **Link:** https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html

### ViDA — Homeostatic Visual Domain Adapter (ICLR 2024)
- **Titolo:** Homeostatic Visual Domain Adapter for Continual Test Time Adaptation
- **Autori:** Senqiao Yang et al.
- **Venue:** ICLR 2024 (A*)
- **Metodo:** High-rank adapters (domain-specific) + low-rank adapters (domain-shared) iniettati in backbone frozen. Homeostatic Knowledge Allotment bilancia dinamicamente i due.
- **Rilevanza:** Separazione architetturale domain-shared vs domain-specific. Il backbone non puo' dimenticare perche' e' frozen. Approccio parameter-efficient.
- **Codice:** https://github.com/Yangsenqiao/vida
- **Link:** https://openreview.net/forum?id=sJ88Wg5Bp5

### Dual Adapters — Continual DA for Time Series (ICLR 2025)
- **Titolo:** Continual Domain Adaptation in Time Series via Parameter-Efficient Dual Adapters and Prompt Tuning
- **Venue:** ICLR 2025 (A*)
- **Metodo:** Backbone Transformer frozen + domain-specific prompt tokens + residual adapters. Solo piccolo subset di parametri aggiornato per ogni nuovo dominio. No replay necessario.
- **Rilevanza:** Direttamente applicabile a time series forecasting. SOTA per CDA su time series. Forgetting strutturalmente impossibile per il backbone.
- **Link:** https://iclr.cc/virtual/2025/33862

## CL su Spatio-Temporal GNN (top-tier, A*)

### TrafficStream (IJCAI 2021) — SEMINALE
- **Titolo:** TrafficStream: A Streaming Traffic Flow Prediction Framework Based on Graph Neural Networks and Continual Learning
- **Autori:** Xu Chen, Junshan Wang et al.
- **Venue:** IJCAI 2021 (A*)
- **Citazioni:** ~200+
- **Metodo:** Historical data replay + parameter smoothing per espandere rete sensori nel tempo senza dimenticare pattern vecchi.
- **Rilevanza:** Primo paper a combinare CL + ST-GNN. Seminale nel campo. Dimostra che retraining da zero e' subottimale rispetto a CL quando la rete spaziale evolve.
- **Link:** https://www.ijcai.org/proceedings/2021/498

### EAC — Expand and Compress (ICLR 2025) — SOTA
- **Titolo:** Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting
- **Venue:** ICLR 2025 (A*)
- **Metodo:** Continuous prompt pool con strategia expand-then-compress. Parameter-efficient. Gestisce sia nuovi nodi che concept drift temporale.
- **Rilevanza:** STATO DELL'ARTE attuale per CL su ST-GNN. Direttamente applicabile alla tua architettura.
- **Link:** https://arxiv.org/abs/2410.12593

## CL per Energy Forecasting (top journal/conference)

### Li et al. 2023 — Benchmark CL per building energy (Applied Energy)
- **Titolo:** Large-Scale Comparison and Demonstration of Continual Learning for Adaptive Building Energy Prediction
- **Autori:** Li, Zhang, Xiao et al.
- **Venue:** Applied Energy (IF ~11, top journal)
- **Metodo:** Confronto empirico di EWC, SI, MAS, GEM, memory replay su 100 edifici. EWC e GEM migliori (~14% improvement vs retraining).
- **Rilevanza:** Unico benchmark rigoroso CL per energy. Mostra che CL batte retraining e fine-tuning. Trasferibile a PV.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0306261923008450

### Proceed — Proactive Concept Drift Adaptation (KDD 2025)
- **Titolo:** Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting
- **Autori:** Zhao, Shen
- **Venue:** KDD 2025 (A*)
- **Metodo:** Stima proattiva del drift + adaptation generator. Non aspetta che il modello fallisca — anticipa lo shift.
- **Rilevanza:** SOTA per online time series con concept drift. Complementare a CL: CL ricorda, Proceed anticipa.
- **Link:** https://dl.acm.org/doi/10.1145/3690624.3709210

### Survey CL per Energy Management (Applied Energy 2025)
- **Titolo:** Continual Learning for Energy Management Systems: A Review
- **Autori:** Sayed, Himeur, Varlamis, Bensaali
- **Venue:** Applied Energy 384 (IF ~11, top journal)
- **Metodo:** Survey che copre EWC, replay, metodi architetturali per CL nel dominio energy.
- **Rilevanza:** Overview completo. Conferma che CL per energy e' campo attivo ma PV specificamente e' gap.
- **Link:** https://www.sciencedirect.com/science/article/pii/S0306261925001886

### CL per Very Short-Term Load Forecasting (Applied Energy 2025)
- **Titolo:** Continual Learning for Very Short-Term Load Forecasting
- **Venue:** Applied Energy (IF ~11)
- **Metodo:** SI e MAS; 21% improvement over offline training.
- **Rilevanza:** Conferma che CL batte offline in scenari con drift temporale.
- **Link:** https://www.sciencedirect.com/science/article/pii/S0306261925016356

## DA per PV (non continual, ma contesto)

### Unsupervised DA for PV Forecasting (Applied Energy 2025)
- **Titolo:** Unsupervised Domain Adaptation Framework for PV Power Forecasting Using Variational Auto-Encoders
- **Venue:** Applied Energy (IF ~11)
- **Metodo:** VAE-based unsupervised DA per cross-site PV transfer.
- **Rilevanza:** DA pura (non continual). Utile come baseline DA da confrontare con CDA.
- **Link:** https://www.sciencedirect.com/science/article/pii/S0306261925013364

## Preprint rilevanti (citare come "concurrent work")

### FreeGNN (arXiv 2025, NON peer-reviewed)
- **Titolo:** Continual Source-Free Graph Neural Network Adaptation for Renewable Energy Forecasting
- **Autori:** Bahi, Ourici, Gasmi, Derrablia, Deghmane, Ferrag
- **Venue:** arXiv preprint 2603.01657 (nessun peer review confermato)
- **Metodo:** Teacher-student + memory replay + drift-aware weighting su ST-GNN. Source-free.
- **Rilevanza:** Piu' vicino al tuo lavoro (PV + GNN + CL), ma NON revisionato. Citare come "concurrent preprint" senza costruirci sopra.

## Il gap nella letteratura

**CL + ST-GNN + PV forecasting = nessun paper pubblicato in venue autorevole.**

Tre comunita' attive ma disconnesse:
1. CL methods (ICLR, NeurIPS, CVPR)
2. ST-GNN forecasting (IJCAI, ICLR — traffico)
3. PV energy prediction (Applied Energy, Solar Energy — DA ma non CL)

La tesi si posiziona nell'intersezione, riempiendo un gap documentabile.

## Argomento per CDA nel contesto PV

Il notebook `cl_evidence_analysis.ipynb` fornisce evidenze quantitative:
- CV degradation rate = 5.24 (eterogeneo)
- Sottogruppi divergono (p=0.028)
- 24% impianti in recovery (regimi coesistenti)
- Concept drift residui differenziato (p=0.021)

Queste evidenze, combinate con:
- TrafficStream/EAC che dimostrano CL su ST-GNN funziona (traffico)
- Li et al. che dimostrano CL batte retraining per energy (building)
- Assenza di lavori CL per PV

...giustificano CDA come approccio e posizionano il lavoro come contributo originale.
