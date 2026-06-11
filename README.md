# edge-AI remote health monitoring system

**Khalifa University of Science and Technology** — Department of Computer Science
**COSC497: Senior Design Project**, Spring 2026
**Supervisor:** Dr. Emadeldeen Eldele

## team

| Name | ID | Owns |
|---|---|---|
| Mohamed Alremeithi | 100060448 | sensing & edge (rig, firmware, on-device deploy) |
| Saif Alafeefi | 100061144 | ML method (anomaly detector, autoencoder, SSL) |
| Khalfan Alantali | 100059479 | signal processing (filtering, quality, features) |
| Zayed Alnuami | 100061300 | dashboard & integration (live viz, alerting, demo) |
| Khaleifah Alhefeiti | 100059431 | data & evaluation (protocol, eval harness, metrics) |

## what it is

a low-cost, privacy-preserving early-warning system that runs **fully on-device**
and flags physiological deviations for a human to review. it is *not* a
diagnostic tool — sensitivity comes first, and a flag triggers human follow-up.

the contribution is **the method and the deployment, not raw signal accuracy**.
we build a **one-class anomaly detector** — trained on abundant *normal* data, it
flags what it hasn't seen — and test whether a detector built on clean public
data survives on cheap, noisy hardware. the result lives in that gap.

**target:** mental stress (primary) and exertion/recovery (complement), using
HR, SpO₂, and accelerometer. **metric:** PR-AUC and recall @ 90% specificity, on
subject-wise splits (pre-committed — no moving goalposts).

what we are **not** doing: diagnosing disease, competing with smartwatches on
signal quality, claiming clinical validity, or detecting "any illness." one
target, one sensor combo, one edge deployment.

## the pipeline

```
sensor → preprocess → quality check → features → anomaly model → alert
HR·SpO₂·accel   filter·resample   artifact reject   extract/embed   autoencoder (+SSL)   dashboard flag
```

signal-quality assessment is a first-class stage, not an afterthought. the edge
target is a **Raspberry Pi** (the guaranteed "runs on device" deliverable);
ESP32-S3 TinyML is the stretch.

## status

the **one-class anomaly detector is built and evaluated** on WESAD wrist BVP: a
statistical baseline, a 1D-conv autoencoder (O1), and a self-supervised encoder
(O2), scored leave-one-subject-out — numbers in
[`anomaly/RESULTS.md`](anomaly/RESULTS.md). a trained model runs live in a web
dashboard (`anomaly/serve.py`). the earlier supervised cardiac model is kept as
prior work; the streaming/dashboard skeleton in `pipeline/` carries over.

| Document | What it covers |
|---|---|
| [`anomaly/README.md`](anomaly/README.md) | one-class detector + eval harness + live dashboard |
| [`anomaly/RESULTS.md`](anomaly/RESULTS.md) | model results (PR-AUC, recall@90%, subject-wise) |
| [`baselines/RESULTS.md`](baselines/RESULTS.md) | earlier supervised baseline (prior work) |
| [`pipeline/README.md`](pipeline/README.md) | original real-time streaming demo (carries over) |

## progress checklist

tick as we go. `[x]` = done.

**S1 · Mohamed — sensing & edge (O3, O7)**
- [ ] assemble the Pi sensor rig (MAX30102 + accelerometer)
- [ ] validate: resting HR within ±5 bpm of reference, 5-min recording, ≥3 people
- [ ] accelerometer logging working
- [ ] (O7, with S2) run the compressed model on the Pi — full sensor→detect→alert loop

**S2 · Saif — ML method (O1, O2)**
- [x] WESAD loader, windowing, subject-wise splits
- [x] PR-AUC / recall@90% metrics (pre-committed)
- [x] statistical baseline (Mahalanobis)
- [x] autoencoder beats baseline on public data (O1)
- [x] self-supervised encoder (O2)
- [ ] TFLite-compress the autoencoder + report accuracy cost (→ O7)
- [ ] per-user calibration: zero-shot vs device-calibrated delta (→ O6)
- [ ] tunable-sensitivity control logic (→ O4, with S4)
- [ ] (optional) exertion model on PPG-DaLiA

**S3 · Khalfan — signal processing (O1)**
- [ ] band-pass / filtering stage
- [ ] artifact rejection
- [ ] signal-quality index that gates windows before the model
- [ ] wire the quality stage into the harness + dashboard

**S4 · Zayed — dashboard & integration (O4)**
- [x] live dashboard: stream → anomaly score → threshold flag → alert + event log
- [ ] tunable sensitivity/precision slider (with S2)
- [ ] end-to-end demo glue
- [ ] M2 device demo

**S5 · Khalifa — data & evaluation (O5, O6)**
- [ ] IRB / consent paperwork (start first — the bottleneck)
- [ ] collection protocol + exclusion criteria
- [ ] collect 10–15 subjects: baseline→induction→recovery, stress + exertion, timestamped
- [ ] (O6, with S2) evaluate induced proxies on device data; report the transfer delta

**milestones**
- [ ] M1 — method beats baseline ✅ · rig validated ❌ (half done)
- [ ] M2 — full demo running on the device

## quick start

```bash
# install dependencies (one-time)
pip3 install -r baselines/requirements.txt -r pipeline/requirements.txt

# live anomaly dashboard — the trained model ships in anomaly/saved/, so this
# runs without WESAD
python3 -m anomaly.serve          # → http://localhost:8001  (▶ Start in the page)

# evaluate the detectors on WESAD (needs WESAD downloaded; leave-one-subject-out)
python3 -m anomaly.run --model ae        # baseline | ae | ssl

# retrain + save the deployable model
python3 -m anomaly.export
```

## data

- **public (develop & benchmark):** [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection),
  PPG-DaLiA, PhysioNet — clean signal, enough subjects for honest splits. not committed (gitignored).
- **our own (test the transfer claim):** modest induced-proxy sessions
  (baseline → induction → recovery), 10–15 consenting volunteers, timestamped.
  no illness data is collected.
- **earlier baseline:** the supervised cardiac model used the UBC PPG dataset
  (Khalili et al.) — [download from Borealis Data](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/HF0OS9)
  (~3.8 GB), unzip into `Code & Data/`.

## repo layout

```
anomaly/                             one-class anomaly detector (current direction)
  wesad.py                           WESAD wrist-BVP loader + windowing
  metrics.py / splits.py             PR-AUC, recall@90%, leave-one-subject-out
  features.py / baseline.py          statistical baseline (Mahalanobis)
  autoencoder.py                     1D-conv autoencoder (O1)
  ssl.py                             self-supervised contrastive encoder (O2)
  run.py                             evaluation harness (LOSO)
  export.py / infer.py               train+save / load the deployable model
  serve.py + static/                 live dashboard (FastAPI + WebSocket + uPlot)
  saved/                             trained model (ae.keras + scorer.npz)
  RESULTS.md                         model results
baselines/                           earlier supervised cardiac model (prior work)
  train.py / configs.py / models.py / losses.py / augment.py
  features.py                        engineered features (reusable)
  data.py                            loader + subject/stratified splits
  inference_lib.py                   Classifier API
  quantize.py                        TFLite int8 conversion (reusable for edge)
  RESULTS.md                         earlier supervised results (prior work)
  runs/                              trained model artifacts
pipeline/                            real-time streaming + dashboard (carries over)
  server.py                          FastAPI + WebSocket dashboard
  static/                            browser UI (index.html + vendored uPlot)
  replay.py                          50 Hz data-source simulator
  vitals.py                          HR / SpO₂ / signal-quality / motion helpers
  pipeline.py / run_cli.py
  make_demo_data.py / demo_data.csv  92 s of curated PPG (ships with repo)
  SENSORS_SETUP.md                   Pi + sensor swap guide
WESAD/ · Code & Data/                datasets (not in git)
README.md                            this file
```
