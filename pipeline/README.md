# pipeline — real-time streaming + dashboard

streams a signal through the model in real time and shows the result in your
browser. this is the **carry-over** part of the project (the dashboard, streaming
skeleton, and signal-quality helpers) — it's model-agnostic infrastructure.

> status: today it streams the earlier supervised classifier. as the project
> pivots to one-class anomaly detection (see the top-level README), this same
> pipeline hosts the anomaly score + threshold flag — the dashboard work for O4.

## what it does

```
data source → stream → buffer → model → web UI
   ↑
   (today: replay demo_data.csv)
   (later: real sensors — PPG + accel over the Pi)
```

only the data source changes when real hardware arrives. everything after it
(buffering, scoring, dashboard) stays the same.

## quick start

```bash
# install dependencies
pip3 install -r baselines/requirements.txt -r pipeline/requirements.txt

# (one-time, only if the dataset is present) rebuild demo_data.csv
python3 pipeline/make_demo_data.py

# option A — terminal-only sanity check
python3 pipeline/run_cli.py            # loops forever
python3 pipeline/run_cli.py --once     # one 92-second pass

# option B — web dashboard (FastAPI + WebSocket)
python3 pipeline/server.py
# open http://localhost:8000 and click ▶ Start
# from a phone on the same WiFi: http://<this-device-ip>:8000
```

## files

| File | What it does |
|---|---|
| `replay.py` | `DataReplay` — yields samples at 50 Hz from a CSV |
| `pipeline.py` | buffer + classifier glue → `PredictionEvent` stream |
| `run_cli.py` | terminal runner (prints each inference) |
| `server.py` | FastAPI + WebSocket dashboard (live waveform + prediction + history) |
| `static/` | browser UI — `index.html` + vendored `uPlot` |
| `make_demo_data.py` | builds `demo_data.csv` from the full dataset |
| `demo_data.csv` | 92-second curated PPG stream (3 segments, one per class) |
| `requirements.txt` | pipeline deps (FastAPI, uvicorn, websockets) |

## swapping in real hardware later

when the MAX30102 + MPU6050 arrive, follow [`SENSORS_SETUP.md`](SENSORS_SETUP.md).
the whole switch is one new file (`sensors.py`) and one changed line in
`server.py` — nothing else moves.

## about the demo

`demo_data.csv` is 3 segments of ~30 seconds each:

1. **cardiac** — normal pulse
2. **non-cardiac** — off-body / no skin contact
3. **occlusion** — blood-flow blockage (the danger flag)

the buffer holds 10.24 s of signal and runs inference every 5 s. watch the
dashboard: predictions settle inside each segment and shift at the transitions.
