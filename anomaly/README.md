# anomaly — one-class stress detection (O1/O2)

the ML-method track: train on **normal** wrist-PPG, flag deviations (stress).
built on WESAD, evaluated with sensitivity-first, subject-wise metrics.

## what's here

| File | Status | What it does |
|---|---|---|
| `wesad.py` | ✅ | load a subject → wrist BVP @ 64 Hz + aligned labels → clean windows |
| `metrics.py` | ✅ | PR-AUC, ROC-AUC, recall@90% specificity (pure numpy, pre-committed) |
| `splits.py` | ✅ | leave-one-subject-out / k-fold-by-subject (no subject leakage) |
| `features.py` | ✅ | HR / HRV / spectral features per window, for the baseline |
| `baseline.py` | ✅ | Mahalanobis one-class detector (the number to beat) |
| `autoencoder.py` | ✅ | 1D-conv autoencoder, reconstruction error = anomaly score (O1) |
| `run.py` | ✅ | LOSO harness: fit on others' normal → score held-out → metrics |
| `ssl.py` | ✅ | self-supervised contrastive encoder + embedding-space scorer (O2) |
| `export.py` | ✅ | train one deployable autoencoder on all calm data → `saved/` |
| `infer.py` | ✅ | load the saved model → score / level / flag one live window |
| `serve.py` + `static/` | ✅ | **live dashboard**: model running on a WESAD BVP stream |
| `make_plots.py` | ✅ | regenerate the result figures (`fig1/2/3_*.png`) |

## run it

```bash
python3 -m anomaly.run --model baseline                  # statistical baseline
python3 -m anomaly.run --model ae --epochs 30            # O1 autoencoder
python3 -m anomaly.run --model ssl --epochs 50           # O2 self-supervised encoder
python3 -m anomaly.run --model baseline --max-subjects 3 # quick check
```

each held-out subject is scored by a detector fit only on the OTHER subjects'
baseline windows; results print per-subject and as mean ± std.

## live dashboard

```bash
python3 -m anomaly.export                 # train + save the deployable model (once)
python3 -m anomaly.serve                  # → http://localhost:8001
python3 -m anomaly.serve --subject S17    # try a different subject
```

streams a WESAD subject's wrist BVP, runs the saved autoencoder on a rolling
60 s window (re-scored every second, EMA-smoothed), and shows a live anomaly
level + calm/stress flag. flag threshold = 90% specificity on calm. default
subject S5 sits low during calm and pegs the level (flagged) through the stress
(TSST) segment; S17 and S7 are also clean demos.

## design choices

- **signal:** wrist BVP @ 64 Hz only — the consumer-grade analogue of our cheap
  sensor. chest ECG/RespiBAN is ignored on purpose (it would cheat on quality).
- **task:** one-class — train on `baseline`, evaluate stress (`TSST`) as the
  positive. amusement / meditation are left out of the binary task by default
  (`to_binary`); fold them into "normal" only deliberately.
- **windows:** fixed length, pure (a window must sit inside one condition).
- **metric, not accuracy:** stress is rare (~11 min/subject), so report PR-AUC
  and recall @ fixed specificity on **subject-wise** splits.

## quick look

```bash
python3 anomaly/wesad.py            # window counts per condition, all 15 subjects
python3 anomaly/wesad.py S2 --win 60 --step 5
```

(WESAD itself is ~17 GB and gitignored — see the top-level README for the layout.)
