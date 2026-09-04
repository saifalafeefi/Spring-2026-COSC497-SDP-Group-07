# command cheat sheet

every command you need, copy-paste ready. run them from the **repo root** unless
noted otherwise.

---

## 1. one-time setup

```bash
# clone the repo (if you haven't)
git clone <repo-url>
cd Spring-2026-COSC497-SDP-Group-07
```

**make a virtualenv first.** the deps are ~1 GB and version-pinned; installing
them into the system Python fights with everything else on the machine.

**Python 3.12 specifically.** `baselines/requirements.txt` pins
`tensorflow-cpu<2.20`, and no TF below 2.20 ships wheels for 3.13 or 3.14 —
on a newer interpreter pip simply fails to resolve tensorflow.

macOS / Linux:

```bash
python3.12 -m venv ~/.venvs/sdp07
~/.venvs/sdp07/bin/python -m pip install --upgrade pip
~/.venvs/sdp07/bin/python -m pip install -r baselines/requirements.txt -r pipeline/requirements.txt
```

Windows (PowerShell):

```powershell
winget install --id Python.Python.3.12 -e     # only if `py -0p` doesn't list 3.12
py -3.12 -m venv $HOME\.venvs\sdp07           # then open a NEW terminal
$PY = "$HOME\.venvs\sdp07\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r baselines\requirements.txt -r pipeline\requirements.txt
```

note the `&`: PowerShell parses a line that *starts* with `$HOME\...` as an
expression, not a command, and fails with `Unexpected token`. the call operator
`&` (with the path quoted, or held in `$PY`) is what runs it. `$PY` lives only
in that terminal tab — re-set it in a new one.

every `python3 -m ...` command below then runs as
`~/.venvs/sdp07/bin/python -m ...` (macOS/Linux) or `& $PY -m ...` (Windows).

- **don't `activate`** — spelling out the interpreter avoids shell aliases and
  a stale `python3` pointing somewhere else
- **keep the venv outside the repo and outside any synced folder** (iCloud,
  OneDrive, Dropbox). a 1.7 GB venv under an iCloud-synced `~/Documents` made
  `import numpy` take **174 s**; `.gitignore` does not stop a sync client
- **venvs are not relocatable** — console scripts hardcode the interpreter path.
  recreate it rather than moving it
- only `anomaly.device_check` / `anomaly.device_source` run without TensorFlow
  (they need just numpy + scipy + pyserial); everything else needs the full install

```bash
# (optional, only if you'll retrain) put the UBC PPG dataset at Code & Data/
# it's ~3.8 GB and not in the repo — download link below.
```

dataset download: [Borealis Data](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/HF0OS9) (~3.8 GB, unzip into `Code & Data/`).

---

## 2. anomaly detection (current direction)

the one-class stress detector on WESAD wrist BVP. the deployable model ships as
`anomaly/saved/ae_int8.tflite` (4 MB), so the live dashboard runs **without** WESAD.

### setup (once)

```bash
python3.12 -m venv ~/.venvs/sdp07          # NOT inside this repo - see below
~/.venvs/sdp07/bin/pip install -r baselines/requirements.txt -r pipeline/requirements.txt
```

then prefix commands with `~/.venvs/sdp07/bin/python` (no `activate` needed).

- **do not put the venv under `Documents/`.** if iCloud "Desktop & Documents"
  sync is on, it tries to sync all ~1.7 GB / 30k files of it: `fileproviderd`
  and `cloudkitd` peg the CPU and every import crawls (measured: `import numpy`
  took **174 s** instead of 0.2 s). `.gitignore` does not stop iCloud.
- **macOS Intel only:** `tensorflow-cpu` stops at 2.16.2 (no x86 wheels after
  that), which forces `numpy<2` and needs `setuptools<81` — TF 2.16 imports
  `distutils`, removed from the 3.12 stdlib, and setuptools 81 dropped the shim.
  both constraints are already in `baselines/requirements.txt`. Apple Silicon
  gets newer TF and is unaffected.
- **python 3.12**, not 3.13 — there is no TF wheel for 3.13 on this platform.

### live dashboards

```bash
python3 -m anomaly.serve                 # → http://localhost:8001
python3 -m anomaly.serve --subject S17   # other clean demo subjects: S17, S7
```

two data sources, same dashboard:

```bash
python3 -m anomaly.serve                      # DUMMY — WESAD replay loop (default)
python3 -m anomaly.serve --source device      # REAL — MAX30102 over USB, auto-detect port
python3 -m anomaly.serve --source device --device-port /dev/cu.usbmodem101
```

- **one process owns the serial port.** stop `serve` / `device_check` (Ctrl-C) and
  close the Arduino Serial Monitor BEFORE uploading. uploading while the server
  is attached fails at the post-flash reset with `Serial data stream stopped:
  Possible serial noise or corruption` — the flash itself usually succeeded, so
  check with `device_check` before re-flashing. if the TFT comes back white,
  power-cycle the USB (a RESET press does not re-power the display).
- flash `sketch_aug3a/` first; it streams `D,t_ms,ir,red` at 25 Hz over USB while the
  TFT demo keeps running. `anomaly/device_source.py` band-passes it, strips the DC
  pedestal, and resamples to the 64 Hz the model's 3,840-sample window needs
- check the hardware **before** starting the server:
  `python3 -m anomaly.device_source --list-ports` then `python3 -m anomaly.device_source`
  (live self-test — no TensorFlow, no dashboard in the way)
- real mode has no ground truth, so `/dev`'s TP/FP scorecard goes blank; the flag,
  charts, HR and the board's own SpO2 all work. needs `pyserial`
- ⚠️ the saved threshold in `scorer.npz` was calibrated on WESAD wrist BVP, not on
  fingertip MAX30102 — the waveform is live and correct, but the stress flag is not
  meaningful on real hardware until it is recalibrated on your own calm baseline.
  fix it with `anomaly.device_calibrate` (below); `/dev`'s subtitle says which
  thresholds are in force

- `/` (alias `/watch`) — **Pulse Watch** product UI (default view)
- `/dev` — developer dashboard: model flag vs WESAD ground truth (TP/FP/FN/TN +
  precision/recall), true-stress band, and the **sensitivity slider**
- both run the same model + `/ws`; the dashboard runs `ae_int8.tflite` — the exact
  model the ESP32 runs

### evaluate the detectors (needs WESAD in `WESAD/`)

```bash
python3 -m anomaly.run --model baseline   # statistical floor (~0.64 PR-AUC)
python3 -m anomaly.run --model ae         # autoencoder, O1 (~0.67)
python3 -m anomaly.run --model ssl        # self-supervised, O2 (~0.68)
python3 -m anomaly.wesad                  # window counts per condition
```

leave-one-subject-out; numbers also in `anomaly/RESULTS.md`. the first run reads
~13 GB of WESAD pickles once, then caches to `WESAD/_harness_cache/`.

### model-improvement levers (only affect `--model ae`)

```bash
python3 -m anomaly.run --model ae --bottleneck 256              # real latent (over-complete fix)
python3 -m anomaly.run --model ae --bottleneck 256 --ch-cap 32  # ESP32-sized  ← DEPLOYED config
python3 -m anomaly.run --model ae --bottleneck 256 --ch-cap 32 --denoise 0.15   # + noise-robust
```

deployed config (bottleneck-256 ch-cap32) = LOSO **PR-AUC 0.706 / recall@90spec 0.545**.

### deploy a model (train → compress)

```bash
python3 -m anomaly.export --bottleneck 256 --ch-cap 32   # train + save ae.keras
python3 -m anomaly.compress                              # → ae_int8.tflite + int8-calibrated scorer
```

- always run `compress` after `export` — it rewrites `scorer.npz` on the int8 score
  scale, which the dashboard needs to flag correctly
- commit only `ae_int8.tflite` (4 MB) + `scorer.npz`; `ae.keras` (46 MB) +
  `ae_float32.tflite` (16 MB) are regenerated locally and **gitignored**
- deployed int8: 4.0 MB, 1.49 ms/window, fits the ESP32-S3-N16R8 (16 MB flash / 8 MB PSRAM)

### per-user calibration (O6 method)

```bash
python3 -m anomaly.calibrate              # zero-shot vs device-calibrated: PR-AUC 0.75 -> 0.87
```

### calibrate the flag on the REAL sensor (O6 on our own rig)

`scorer.npz` holds WESAD **wrist** thresholds; the rig is a **fingertip** sensor,
so the live flag means nothing until it is re-derived on your own calm. this
records that calm and rewrites the thresholds:

```bash
python3 -m anomaly.device_check                      # get a steady GOOD TO RECORD first
python3 -m anomaly.device_calibrate                  # 5 min of calm, sit still
python3 -m anomaly.serve --source device             # now flags against your own baseline
```

- writes `anomaly/saved/scorer_device.npz`; `scorer.npz` is left alone, so the
  WESAD replay demo is unaffected
- `serve --source device` uses it automatically once it exists. force either one
  with `--scorer wesad` / `--scorer device` — the before/after is the
  domain-transfer delta, live
- windows are gated on the same grip checks `device_check` prints: any tick that
  fails restarts the 60 s clean run, so a fidget never teaches the model "normal"
- `--simulate --minutes 2 --min-windows 5` runs the whole path on a synthetic pulse
  with no board
  attached and writes nothing — use it to check the tool before spending a
  real recording on it
- `--dry-run` reports the numbers without writing. the raw calm windows land in
  `anomaly/saved/device_calm.npz` (gitignored — personal biometric data)

WESAD is ~17 GB and gitignored — download it and unzip into `WESAD/`. `ae`/`ssl`/
`serve`/`compress` use TensorFlow (a `baselines/requirements.txt` dep); for GPU install
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
| `anomaly/serve.py` + `anomaly/static/` | the live anomaly dashboards (`/` Pulse Watch, `/dev` developer) |
| `anomaly/saved/ae_int8.tflite` | deployed model — int8 TFLite (4 MB, committed; dashboard + ESP32 run this) |
| `anomaly/RESULTS.md` | one-class detector results (PR-AUC, recall@90%) |
| `WESAD/` | WESAD dataset (~17 GB, not in git) |
| `baselines/runs/2026-05-17_163328_phase_a/model.keras` | the earlier supervised model |
| `baselines/runs/2026-05-17_163328_phase_a/model_int8.tflite` | quantized model for ESP32 |
| `baselines/inference_lib.py` | the `Classifier` API |
| `pipeline/server.py` | earlier cardiac dashboard (legacy; current one is `anomaly/serve.py`) |
| `pipeline/static/` | browser UI (`index.html` + vendored uPlot) |
| `pipeline/replay.py` | synthetic data source (demo mode) |
| `pipeline/fall_detector.py` | 4-phase fall-detection state machine |
| `pipeline/SENSORS_SETUP.md` | **guide for swapping in real MAX30102 + MPU6050** |
| `Code & Data/` | the 3.8 GB training dataset (not in git) |
