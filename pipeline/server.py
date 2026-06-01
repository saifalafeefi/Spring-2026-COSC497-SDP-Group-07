"""FastAPI + WebSocket dashboard server for the multi-sensor PPG pipeline.

built for the real deployment shape: the hub device runs this server and a phone
(or any browser) on the LAN opens the page. the server *pushes* compact JSON
frames over a WebSocket; the browser draws them with uPlot. no per-frame page
reloads — smooth at 50 Hz, instant controls, tiny network use.

all the heavy lifting is reused unchanged from the rest of the pipeline:
  • Classifier      (inference_lib)  — the hybrid 1D-CNN
  • DataReplay      (replay)         — the 50 Hz sample source
  • vitals          (vitals)         — HR / SpO2 / signal-quality / motion
  • FallDetector    (fall_detector)  — the 4-phase fall state machine

run from the repo root:
    python3 pipeline/server.py
then open http://localhost:8000  (or http://<hub-ip>:8000 from a phone).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

from inference_lib import CLASSES, Classifier        # noqa: E402
from features import extract_features, FEATURE_NAMES  # noqa: E402
from replay import FS, DataReplay                     # noqa: E402
from vitals import (estimate_heart_rate, estimate_spo2,  # noqa: E402
                    signal_quality_score, motion_intensity)
from fall_detector import FallDetector                # noqa: E402

# ----- tunables -----
WINDOW_LEN = 512                       # inference window (10.24 s @ 50 Hz)
INFERENCE_EVERY_SAMPLES = 250          # new prediction every 5 s
DISPLAY_WINDOW_SAMPLES = 500           # rolling chart history (10 s)
TICK_SEC = 0.04                        # producer tick → 25 frames/s
SAMPLES_PER_TICK = max(1, round(FS * TICK_SEC))   # 2 samples/tick → 50 Hz
VITALS_EVERY_SAMPLES = FS              # recompute HR/SpO2 ~once per second
PREDICTIONS_MAX = 200                  # server-side history cap
HISTORY_ON_CONNECT = 30                # full preds sent to a newly-joined client


class Engine:
    """single shared streaming engine. one replay feeds every connected client,
    so a phone and a laptop see the same stream, and the controls act on that
    one stream (handy for a multi-screen demo)."""

    def __init__(self):
        self.clf = Classifier()
        # warm up at startup, not on the first live prediction. the first
        # classify() triggers TF graph tracing and the lazy dataset load in
        # _feature_stats() — a few seconds. doing it here keeps the stream
        # from stalling mid-demo.
        try:
            print("  warming up classifier (TF trace + feature stats)…", flush=True)
            t0 = time.perf_counter()
            self.clf.classify(np.zeros(WINDOW_LEN, dtype=np.float32))
            self.clf._feature_stats()
            print(f"  warm-up done in {time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  warm-up skipped: {e}", flush=True)
        self.stream = DataReplay(loop=True).stream(realtime=False)
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.running = False
        self._reset_state()

    def _reset_state(self):
        # display ring buffers (one rolling window of every channel)
        self.d_idx = deque(maxlen=DISPLAY_WINDOW_SAMPLES)
        self.d_ir = deque(maxlen=DISPLAY_WINDOW_SAMPLES)
        self.d_red = deque(maxlen=DISPLAY_WINDOW_SAMPLES)
        self.d_ax = deque(maxlen=DISPLAY_WINDOW_SAMPLES)
        self.d_ay = deque(maxlen=DISPLAY_WINDOW_SAMPLES)
        self.d_az = deque(maxlen=DISPLAY_WINDOW_SAMPLES)
        # inference window buffers
        self.inf_ir = deque(maxlen=WINDOW_LEN)
        self.inf_red = deque(maxlen=WINDOW_LEN)
        self.inf_lab = deque(maxlen=WINDOW_LEN)
        self.total_samples = 0
        self.since_inference = 0
        self.last_vitals_at = -10 ** 9
        self.vitals = {"hr": None, "spo2": None, "sq": 0.0, "mot": 0.0}
        self.last_latency_ms = None
        self.fall = FallDetector(fs=FS)
        self.predictions: list[dict] = []
        self._pred_id = 0

    # ---------- per-tick work ----------

    def _ingest(self) -> dict:
        """pull one tick of samples; return the new-sample arrays for the frame."""
        nidx, nir, nred, nax, nay, naz = [], [], [], [], [], []
        for _ in range(SAMPLES_PER_TICK):
            s = next(self.stream)
            self.d_idx.append(s.sample_idx); self.d_ir.append(s.ir)
            self.d_red.append(s.red); self.d_ax.append(s.accel_x)
            self.d_ay.append(s.accel_y); self.d_az.append(s.accel_z)
            self.inf_ir.append(s.ir); self.inf_red.append(s.red)
            self.inf_lab.append(s.true_label)
            self.fall.update(s.accel_x, s.accel_y, s.accel_z)
            self.total_samples += 1
            self.since_inference += 1
            nidx.append(s.sample_idx)
            nir.append(round(s.ir)); nred.append(round(s.red))
            nax.append(round(s.accel_x, 4)); nay.append(round(s.accel_y, 4))
            naz.append(round(s.accel_z, 4))
        return {"idx": nidx, "ir": nir, "red": nred,
                "ax": nax, "ay": nay, "az": naz}

    def _maybe_vitals(self):
        if self.total_samples - self.last_vitals_at < VITALS_EVERY_SAMPLES:
            return
        self.last_vitals_at = self.total_samples
        if len(self.d_ir) >= FS * 4:
            ir = np.fromiter(self.d_ir, dtype=np.float32)
            red = np.fromiter(self.d_red, dtype=np.float32)
            accel = np.stack([np.fromiter(self.d_ax, dtype=np.float32),
                              np.fromiter(self.d_ay, dtype=np.float32),
                              np.fromiter(self.d_az, dtype=np.float32)], axis=1)
            self.vitals = {
                "hr": estimate_heart_rate(ir, fs=FS),
                "spo2": estimate_spo2(ir, red),
                "sq": signal_quality_score(ir),
                "mot": motion_intensity(accel, fs=FS),
            }

    def _run_inference(self) -> dict | None:
        """blocking — runs the model. called in a thread executor."""
        win_ir = np.fromiter(self.inf_ir, dtype=np.float32)
        win_red = np.fromiter(self.inf_red, dtype=np.float32)
        labels = list(self.inf_lab)
        majority = max(set(labels), key=labels.count)
        t0 = time.perf_counter()
        res = self.clf.classify(win_ir)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        try:
            mu, sd = self.clf._feature_stats()
            feat_z = ((extract_features(win_ir[None])[0] - mu) / sd).tolist()
        except Exception:
            feat_z = None
        self._pred_id += 1
        pred = {
            "id": self._pred_id,
            "t": round(self.total_samples / FS, 1),
            "label": res.label,
            "conf": round(res.confidence, 4),
            "probs": {c: round(p, 4) for c, p in res.probabilities.items()},
            "true": majority,
            "win_ir": [round(v) for v in win_ir.tolist()],
            "win_red": [round(v) for v in win_red.tolist()],
            "feat": [round(v, 3) for v in feat_z] if feat_z else None,
            "lat": round(self.last_latency_ms),
        }
        self.predictions.append(pred)
        if len(self.predictions) > PREDICTIONS_MAX:
            self.predictions.pop(0)
        return pred

    def frame(self, new: dict) -> dict:
        v = self.vitals
        return {
            "type": "f",
            "running": self.running,
            "elapsed": round(self.total_samples / FS, 1),
            "buf": len(self.inf_ir),
            "vit": {"hr": v["hr"], "spo2": v["spo2"],
                    "sq": v["sq"], "mot": v["mot"]},
            "fall": {"st": self.fall.state, "ev": len(self.fall.events)},
            "lat": self.last_latency_ms,
            **new,
        }

    # ---------- broadcast ----------

    async def broadcast(self, payload: dict):
        if not self.clients:
            return
        msg = json.dumps(payload)
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # ---------- producer loop ----------

    async def producer(self):
        self.loop = asyncio.get_running_loop()
        next_t = self.loop.time()
        while True:
            if self.running and self.clients:
                new = self._ingest()
                self._maybe_vitals()
                await self.broadcast(self.frame(new))
                if (len(self.inf_ir) >= WINDOW_LEN
                        and self.since_inference >= INFERENCE_EVERY_SAMPLES):
                    self.since_inference = 0
                    pred = await self.loop.run_in_executor(None, self._run_inference)
                    if pred:
                        await self.broadcast({"type": "p", "p": pred})
                next_t += TICK_SEC
                delay = next_t - self.loop.time()
                await asyncio.sleep(delay if delay > 0 else 0)
                if delay <= 0:
                    next_t = self.loop.time()
            else:
                # idle: don't burn CPU when paused or nobody's watching
                next_t = self.loop.time() + TICK_SEC
                await asyncio.sleep(0.1)

    # ---------- commands ----------

    def handle_cmd(self, cmd: str):
        if cmd == "start":
            self.running = True
        elif cmd == "pause":
            self.running = False
        elif cmd == "reset":
            self.running = False
            self._reset_state()


engine = Engine()


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(engine.producer())
    try:
        yield
    finally:
        task.cancel()


# FastAPI imported here so `python3 server.py` gives a clean error if missing.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse                   # noqa: E402
from fastapi.staticfiles import StaticFiles                  # noqa: E402

app = FastAPI(lifespan=lifespan)
app.mount("/vendor", StaticFiles(directory=os.path.join(STATIC_DIR, "vendor")),
          name="vendor")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    engine.clients.add(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "hello", "fs": FS, "win": WINDOW_LEN,
            "disp": DISPLAY_WINDOW_SAMPLES, "classes": CLASSES,
            "feat_names": FEATURE_NAMES,
            "inf_s": INFERENCE_EVERY_SAMPLES / FS, "running": engine.running,
        }))
        if engine.predictions:
            await ws.send_text(json.dumps(
                {"type": "hist", "preds": engine.predictions[-HISTORY_ON_CONNECT:]}))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if "cmd" in msg:
                engine.handle_cmd(msg["cmd"])
                await engine.broadcast({"type": "state", "running": engine.running})
    except WebSocketDisconnect:
        pass
    finally:
        engine.clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  PPG dashboard → http://localhost:{port}"
          f"   (LAN: http://<this-device-ip>:{port})\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
