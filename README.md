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

signal-quality assessment is a first-class stage, not an afterthought — it is
implemented for the live sensor (`device_source.py` filters, `device_check.py` gates).
the edge target is a **Raspberry Pi** (the guaranteed "runs on device" deliverable);
ESP32-S3 TinyML is the stretch, though the ESP32-S3 is what currently streams live.

## status

the **one-class anomaly detector is built and evaluated** on WESAD wrist BVP: a
statistical baseline, a 1D-conv autoencoder (O1), and a self-supervised encoder
(O2), scored leave-one-subject-out — numbers in
[`anomaly/RESULTS.md`](anomaly/RESULTS.md). a bottleneck redesign lifts the
autoencoder to **PR-AUC 0.706** (from 0.67), and it ships as a **4 MB int8 TFLite**
(1.5 ms/window) that fits the **ESP32-S3-N16R8** — the dashboard and the device run
the *same* model. it runs live in two web front-ends (`anomaly/serve.py`): the team's
**Pulse Watch** product UI at `/` and a developer dashboard at `/dev` that scores the
model's flag against ground truth live, both with a tunable sensitivity control. the
earlier supervised cardiac model is kept as prior work; the streaming/dashboard
skeleton in `pipeline/` carries over.

**it now runs on real hardware.** an ESP32-S3 + MAX30102 streams over USB serial;
`anomaly/device_source.py` band-passes and resamples it to the same 64 Hz the model
expects, so `python3 -m anomaly.serve --source device` swaps the WESAD replay for a
live finger without touching the model. the board's own display mirrors the
dashboard's heart rate, and falls back to its own estimate when untethered.

**the flag is calibrated on our own sensor.** `scorer.npz` holds WESAD *wrist*
thresholds; this rig is a *fingertip* sensor, so those thresholds meant nothing here.
`anomaly/device_calibrate.py` — or the dashboard's Calibrate button, which applies
the result live — records the user's own calm, gates it, and re-derives the
threshold on it. **the transfer delta on our rig: our calm median (0.282) scores
above the WESAD threshold (0.212), so a zero-shot wrist model flags essentially
100% of our calm; device-calibrated that is 10% by construction.** measured on one
subject, calm only.

**not yet shown:** that the flag *rises under stress* on this hardware. everything
above establishes what calm looks like. the induced-proxy test is the next step and
the thing that would validate or sink the transfer claim.

| Document | What it covers |
|---|---|
| [`anomaly/README.md`](anomaly/README.md) | one-class detector + eval harness + live dashboard |
| [`anomaly/RESULTS.md`](anomaly/RESULTS.md) | model results (PR-AUC, recall@90%, subject-wise) |
| [`baselines/RESULTS.md`](baselines/RESULTS.md) | earlier supervised baseline (prior work) |
| [`pipeline/README.md`](pipeline/README.md) | original real-time streaming demo (carries over) |

## progress checklist

tick as we go. `[x]` = done.

**S1 · Mohamed — sensing & edge (O3, O7)**
- [x] ESP32-S3 + MAX30102 rig streaming live to the dashboard (band-passed, resampled to 64 Hz)
- [x] on-device heart rate: replaced the library estimate (assumed 25 Hz, swung 28–150 bpm) with a millis()-based detector; board and dashboard now agree within 1–2 bpm
- [ ] assemble the Pi sensor rig (MAX30102 + accelerometer)
- [ ] validate: resting HR within ±5 bpm of **a reference oximeter**, 5-min recording, ≥3 people
- [ ] accelerometer logging working (no accelerometer on the rig yet)
- [ ] (O7, with S2) run the compressed model on the Pi — full sensor→detect→alert loop

**S2 · Saif — ML method (O1, O2)**
- [x] WESAD loader, windowing, subject-wise splits
- [x] PR-AUC / recall@90% metrics (pre-committed)
- [x] statistical baseline (Mahalanobis)
- [x] autoencoder beats baseline on public data (O1)
- [x] self-supervised encoder (O2)
- [x] bottleneck redesign: LOSO PR-AUC 0.67 → 0.71, tighter spread across subjects
- [x] TFLite-compress to a 4 MB int8 model, 1.5 ms/window, fits the ESP32-S3 (→ O7 model ready)
- [x] per-user calibration on WESAD: +0.12 PR-AUC, +0.23 recall (→ O6 method ready)
- [x] tunable-sensitivity control: slider + watch preset → server retunes the flag threshold (O4, with S4)
- [x] device calibration: record our own calm on the real sensor, re-derive the flag threshold, apply it live (CLI + dashboard button)
- [x] transfer delta measured on our own hardware (see status) — O6 is no longer WESAD-only
- [ ] show the flag responds to induced stress on this rig (calm-only so far)
- [ ] (optional) exertion model on PPG-DaLiA

**S3 · Khalfan — signal processing (O1)**

built on the device path already (`anomaly/device_source.py`, `anomaly/device_check.py`)
— reuse rather than rewrite; what is missing is the WESAD-harness side.

- [x] band-pass / filtering stage (0.7–3 Hz causal `sosfilt`, primed to avoid startup ringing)
- [x] artifact rejection (contact loss, drift, spikes, perfusion)
- [x] signal-quality index that gates windows before the model — adaptive: it tunes to
      the session's own signal, because fixed thresholds from one rig rejected 100% of
      another's windows
- [ ] wire the quality stage into the WESAD harness

**S4 · Zayed — dashboard & integration (O4)**
- [x] live dashboard: stream → anomaly score → threshold flag → alert + event log
- [x] tunable sensitivity/precision slider (with S2)
- [x] Pulse Watch product UI integrated on the live pipeline (live Patients + History)
- [x] Calibrate is a real backend session (progress, gate, commit) instead of a mock ramp
- [x] with no finger on the sensor: HR, SpO₂, quality, level and flag blank instead of
      reporting noise, and the waveform holds its scale so noise draws flat
- [ ] end-to-end demo glue
- [ ] M2 device demo

**S5 · Khalifa — data & evaluation (O5, O6)**
- [ ] IRB / consent paperwork (start first — the bottleneck)
- [ ] collection protocol + exclusion criteria
- [ ] collect 10–15 subjects: baseline→induction→recovery, stress + exertion, timestamped
- [ ] (O6, with S2) evaluate induced proxies on device data; report the transfer delta

**milestones**
- [ ] M1 — method beats baseline ✅ · rig streaming live ✅ · rig validated against a reference ❌
- [ ] M2 — full demo running on the device

## quick start

**use a virtualenv, on Python 3.12.** `baselines/requirements.txt` pins
`tensorflow-cpu<2.20`, which has no wheels for 3.13 or 3.14 — on a newer
interpreter pip simply fails to resolve tensorflow. keep the venv outside the repo
and outside any synced folder (iCloud/OneDrive). full setup in
[`COMMANDS.md`](COMMANDS.md).

```bash
# install dependencies (one-time)
python3.12 -m venv ~/.venvs/sdp07
~/.venvs/sdp07/bin/python -m pip install -r baselines/requirements.txt -r pipeline/requirements.txt

# live dashboards — the int8 model ships in anomaly/saved/, so this runs without WESAD
python3 -m anomaly.serve          # → http://localhost:8001  ( / Pulse Watch · /dev developer )

# the same dashboard on the real sensor (ESP32-S3 + MAX30102 over USB)
python3 -m anomaly.device_check           # grip coach: contact, perfusion, drift, quality
python3 -m anomaly.serve --source device  # then hit Calibrate in the UI to set your baseline

# evaluate the detectors on WESAD (needs WESAD downloaded; leave-one-subject-out)
python3 -m anomaly.run --model ae --bottleneck 256 --ch-cap 32   # baseline | ae | ssl

# retrain + compress the deployable int8 model (full reference: COMMANDS.md)
python3 -m anomaly.export --bottleneck 256 --ch-cap 32 && python3 -m anomaly.compress
```

see [`COMMANDS.md`](COMMANDS.md) for the full command cheat sheet.

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
  calibrate.py                       per-user calibration (O6 method)
  compress.py                        TFLite int8 + compression cost (O7)
  export.py / infer.py               train+save / load the deployable model
  serve.py + static/                 live dashboard (FastAPI + WebSocket + uPlot)
  device_source.py                   ESP32-S3 + MAX30102 serial reader → band-pass → 64 Hz
  device_check.py                    grip coach + the shared signal-quality gate
  device_calibrate.py                record our own calm, re-derive the flag threshold (O6)
  make_plots.py                      result figures (fig1–4)
  saved/                             deployed int8 model (ae_int8.tflite + scorer.npz; keras gitignored)
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
sketch_aug3a/                        ESP32-S3 firmware: streams IR/RED + vitals, TFT display
  sketch_aug3a.ino                   time-based HR detector; mirrors the dashboard's HR when connected
pulse/                               Pulse Watch product UI (Zayed) — served at / by anomaly/serve.py
  Pulse Watch.dc.html / support.js   design-tool export; speaks serve.py's /ws protocol
WESAD/ · Code & Data/                datasets (not in git)
COMMANDS.md                          command cheat sheet
README.md                            this file
```
