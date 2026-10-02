# Precisione FP32 / BF16 per STGAN

I branch `experiment/stgan-era5-gat-mc-dropout` e `experiment/stgan-cnn`
espongono la stessa scelta `--precision fp32|bf16`. Il default e' `fp32`.
Il server newzealand (RTX PRO 6000 Blackwell, PyTorch 2.11.0+cu128) ha
confermato supporto BF16 nativo: usare `--precision bf16` per provarlo.
Non sono previste opzioni FP16 o selezione automatica della precisione.

## Avvio

Aggiungere `--precision bf16` al comando di training esistente e scegliere
una nuova directory di output. Esempi (adattare i percorsi dei dati):

```bash
# Branch GAT / ERA5
python scripts/run_era5_stgan.py train \
  --prepared-dir outputs/era5/prepared \
  --output-dir outputs/era5/gat_bf16 \
  --spatial-encoder gat --precision bf16

# Branch CNN / PVGIS
python scripts/run_pvgis_stgan.py \
  --manifest outputs/pvgis_stgan/prepared/manifest.csv \
  --out-dir outputs/pvgis_stgan_cnn/convgru_bf16 \
  --precision bf16
```

Nel notebook `stgan_cnn_pvgis_workflow.ipynb`, impostare `PRECISION = 'bf16'`
oppure la variabile d'ambiente `STGAN_PRECISION=bf16` prima dell'avvio del kernel.
La directory predefinita del notebook acquista il suffisso `_bf16`.
`STGAN_CNN_OUT_DIR`, se impostata, continua ad avere precedenza.
I vecchi run senza campo `precision` vengono interpretati come FP32 quando
il notebook controlla la compatibilita' della configurazione salvata.

Nell'API Python usare `STGANCNNConfig(precision='bf16')` oppure
`fit_and_score_stgan(..., precision='bf16')`. La stessa impostazione si
applica al training e a tutto lo scoring, inclusi calibrazione e Monte Carlo
nel branch GAT. Una richiesta BF16 su CPU o GPU senza supporto nativo genera
un errore prima della preparazione del training, senza fallback silenzioso.

## Calcoli e checkpoint

- AMP esegue le operazioni compatibili in BF16. Pesi, gradienti dei parametri
  e stati Adam restano FP32; non serve `GradScaler` per questa modalita'.
- In BF16 la loss avversaria usa i logits con `binary_cross_entropy_with_logits`.
  Il percorso FP32 conserva la precedente BCE sulle probabilita'. Lo scoring
  continua a usare le probabilita' del discriminatore, con sigmoid in FP32.
- Errori quadratici, somme sparse e softmax dell'attenzione GAT e accumulo
  dei gradienti tra i blocchi GAT restano FP32. Le statistiche gia' accumulate
  in FP64 mantengono tale precisione. Gli array degli score sono FP32.
- Checkpoint intermedi/finali e metadata registrano `precision`. I checkpoint
  conservano pesi FP32 e le chiavi del precedente `state_dict`: possono essere
  caricati anche su CPU. `load_stgan_checkpoint` restituisce il campo
  `payload['precision']`, con default `fp32` per i checkpoint precedenti.
  Per rifare lo scoring usare `score_components(..., precision=payload['precision'])`
  su una GPU compatibile, oppure scegliere esplicitamente `fp32`.

## Verifica sul server

Da ciascun branch eseguire:

```bash
python tests/test_stgan_precision.py
```

I test includono aggiornamenti G/D, scoring, confronto numerico su un piccolo
dataset sintetico, metadati e caricamento dei checkpoint. Nel branch GAT
coprono anche i blocchi del discriminatore, il checkpointing del trend LSTM
e il Monte Carlo. Il test CUDA BF16 viene eseguito soltanto su hardware con
supporto nativo; i test CPU BF16 usano un adattamento riservato ai test e
non misurano le prestazioni CUDA.

Per confrontare i tempi reali, eseguire FP32 e BF16 con gli stessi dati,
seed, batch, numero di epoche e campioni Monte Carlo, in directory distinte
e senza altri processi che contendano la GPU. I metadata `performance`
riportano precisione, tempi di training/scoring, campioni al secondo e picco
di memoria CUDA. Confrontare anche loss, score e anomalie selezionate: BF16
non garantisce risultati identici a FP32 e il guadagno va misurato sul server.
