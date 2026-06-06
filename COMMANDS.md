# command cheat sheet

every command you need, copy-paste ready. run them from the **repo root** unless
noted otherwise.

---

## 1. one-time setup

```bash
# clone the repo (if you haven't)
git clone <repo-url>
cd Spring-2026-COSC497-SDP-Group-07

# install dependencies
pip3 install -r baselines/requirements.txt
pip3 install -r pipeline/requirements.txt

# (optional, only if you'll retrain) put the UBC PPG dataset at Code & Data/
# it's ~3.8 GB and not in the repo — download link below.
```

dataset download: [Borealis Data](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/HF0OS9) (~3.8 GB, unzip into `Code & Data/`).

---

## 2. anomaly detection (current direction)

the one-class stress detector on WESAD wrist BVP. a trained model ships in
`anomaly/saved/`, so the live dashboard runs **without** needing WESAD.

### live dashboard

```bash
python3 -m anomaly.serve                 # → http://localhost:8001, then ▶ Start
python3 -m anomaly.serve --subject S17   # other clean demo subjects: S17, S7
```

### evaluate the detectors (needs WESAD in `WESAD/`)

```bash
python3 -m anomaly.run --model baseline   # statistical floor (~0.64 PR-AUC)
python3 -m anomaly.run --model ae         # autoencoder, O1 (~0.67)
python3 -m anomaly.run --model ssl        # self-supervised, O2 (~0.68)
python3 -m anomaly.wesad                  # window counts per condition
```

leave-one-subject-out; numbers also in `anomaly/RESULTS.md`. the first run reads
~13 GB of WESAD pickles once, then caches to `WESAD/_harness_cache/`.

### retrain + save the deployable model

```bash
python3 -m anomaly.export                      # → anomaly/saved/ae.keras + scorer.npz
python3 -m anomaly.export --spec 0.90 --epochs 40
```

WESAD is ~17 GB and gitignored — download it and unzip into `WESAD/`. `ae`/`ssl`/
`serve` use TensorFlow (a `baselines/requirements.txt` dep); for GPU install
`tensorflow[and-cuda]`.

---

## 3. real-time dashboard (earlier cardiac demo)

a FastAPI + WebSocket server streams to a browser UI (drawn with uPlot). it
reuses the classifier, replay, vitals, and fall-detector code unchanged.

### start it

```bash
python3 pipeline/server.py
```

wait for `warm-up done in Xs` (the model loads at startup so the stream never
hitches), then open `http://localhost:8000` and click **▶ Start**.

**from a phone or another device on the same WiFi:**

```bash
http://<this-device-ip>:8000
```

(the hub device runs the server; the phone is just a browser client.)

### run it in the background

```bash
nohup python3 pipeline/server.py > /tmp/dashboard.log 2>&1 &
```

### shut it down

foreground: press `Ctrl+C`. background:

```bash
pkill -f "pipeline/server.py"
```

if the port is stuck after a crash:

```bash
lsof -i :8000          # find what's holding port 8000
fuser -k 8000/tcp      # force-kill it
```

### use a different port

```bash
PORT=8001 python3 pipeline/server.py
```

---

## 4. terminal-only pipeline (no browser)

```bash
python3 pipeline/run_cli.py            # loops forever
python3 pipeline/run_cli.py --once     # one 92-second pass, then exit
# Ctrl+C stops the loop
```

---

## 5. inference sanity check

quick "the model loads and predicts" test, no streaming:

```bash
python3 baselines/inference_demo.py
```

should print something like `Overall: 28/30 correct (93%)`.

---

## 6. training and model artifacts

### train from scratch

```bash
python3 baselines/train.py --preset phase_a    # ~12 min on CPU
```

### tweak training

```bash
python3 baselines/train.py --preset phase_a --epochs 200      # longer
python3 baselines/train.py --preset phase_a --dropout 0.3     # different dropout
python3 baselines/train.py --preset phase_a --print-config    # config, no training
python3 baselines/train.py --list-presets                     # list presets
```

### quantize to int8 TFLite

```bash
python3 baselines/quantize.py --preset phase_a
```

drops `model_int8.tflite` in the run folder.

### rebuild the demo data file

```bash
# only if the full dataset is present at Code & Data/
python3 pipeline/make_demo_data.py
```

### regenerate the result PNGs

```bash
python3 baselines/make_plots.py
```

---

## 7. git workflow

```bash
git status --short          # see what changed

git add .                   # local-only files are already excluded
git status                  # review before committing

git commit -m "your message here"
git push

git pull                    # get the latest
```

undo the last commit (before pushing):

```bash
git reset --soft HEAD~1     # keep the changes
git reset --hard HEAD~1     # discard them too (destructive)
```

---

## 8. troubleshooting

### `ModuleNotFoundError: No module named 'fastapi'` (or `uvicorn`)

```bash
pip3 install -r pipeline/requirements.txt
```

### `ModuleNotFoundError: No module named 'tensorflow'`

```bash
pip3 install -r baselines/requirements.txt
```

### `FileNotFoundError: No trained model found in runs/`

train it first:

```bash
python3 baselines/train.py --preset phase_a
```

### `FileNotFoundError: pipeline/demo_data.csv not found`

it should be in the repo. if not, rebuild it from the dataset:

```bash
python3 pipeline/make_demo_data.py
```

### page loads but predictions never update

click **▶ Start** — the stream only runs while it's active. the dot in the
top-left goes green only when the WebSocket is connected.

### "Address already in use"

something else is on port 8000. kill it or pick another port:

```bash
pkill -f "pipeline/server.py"
# or:
PORT=8001 python3 pipeline/server.py
```

### phone can't reach the dashboard

put the phone on the **same WiFi** as the host and use the host's LAN IP (e.g.
`http://172.30.140.43:8000`), not `localhost`. the server binds `0.0.0.0`
already; if it still fails, the host firewall is blocking port 8000.

### first prediction is slow

it isn't anymore — the model is warmed up at startup (you'll see
`warm-up done in Xs`). later calls are ~tens of ms.

---

## 9. where things live

| File / folder | Purpose |
|---|---|
| `anomaly/` | **one-class anomaly detector (current direction)** |
| `anomaly/serve.py` + `anomaly/static/` | the live anomaly dashboard |
| `anomaly/saved/ae.keras` | trained deployable autoencoder (3.2 MB, committed) |
| `anomaly/RESULTS.md` | one-class detector results (PR-AUC, recall@90%) |
| `WESAD/` | WESAD dataset (~17 GB, not in git) |
| `baselines/runs/2026-05-17_163328_phase_a/model.keras` | the earlier supervised model |
| `baselines/runs/2026-05-17_163328_phase_a/model_int8.tflite` | quantized model for ESP32 |
| `baselines/inference_lib.py` | the `Classifier` API |
| `pipeline/server.py` | the web dashboard (FastAPI + WebSocket) |
| `pipeline/static/` | browser UI (`index.html` + vendored uPlot) |
| `pipeline/replay.py` | synthetic data source (demo mode) |
| `pipeline/fall_detector.py` | 4-phase fall-detection state machine |
| `pipeline/SENSORS_SETUP.md` | **guide for swapping in real MAX30102 + MPU6050** |
| `Code & Data/` | the 3.8 GB training dataset (not in git) |
