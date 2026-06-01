# pipeline — real-time PPG inference demo

streams PPG through the trained model in real time and shows the prediction in
your browser.

## what it does

```
data source → 50 Hz stream → 512-sample buffer → classifier → web UI
   ↑
   (today: replay demo_data.csv)
   (later: MAX30102 over USB/serial)
```

only the data source changes when real hardware arrives. everything after it
(buffering, inference, dashboard) is already proven.

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
