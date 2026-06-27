"""live anomaly dashboard — the trained autoencoder running on a WESAD BVP stream.

streams wrist BVP, runs the saved model on a rolling 60 s window, and pushes the
anomaly level + stress flag to the browser over a WebSocket (uPlot front-end).

    python3 -m anomaly.serve                 # → http://localhost:8001
    python3 -m anomaly.serve --subject S16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import deque
from contextlib import asynccontextmanager

import numpy as np

from .wesad import FS
from .wesad_replay import BVPReplay
from .infer import LiveAnomalyDetector

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
STATIC_DIR = os.path.join(HERE, "static")
VENDOR_DIR = os.path.join(REPO_ROOT, "pipeline", "static", "vendor")  # reuse uPlot
PULSE_DIR = os.path.join(REPO_ROOT, "pulse")                          # Zayed's Pulse Watch UI

# heart-rate + signal-quality helpers reused from the pipeline (work on any PPG @ fs)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from vitals import estimate_heart_rate, signal_quality_score  # noqa: E402

WIN_LEN = 60 * FS              # 60 s inference window
DISPLAY = 15 * FS             # 15 s of BVP on the chart
TICK_SEC = 0.05               # 20 frames/s
SAMPLES_PER_TICK = max(1, round(FS * TICK_SEC))
INFER_EVERY = FS              # re-score ~once per second

LABELS = {0: "—", 1: "calm", 2: "STRESS", 3: "amusement",
          4: "meditation", 5: "—", 6: "—", 7: "—"}

# S5 separates calm/stress most cleanly under the deployed model (S17, S7 also good)
SUBJECT = os.environ.get("SUBJECT", "S5")


class Engine:
    def __init__(self, subject: str):
        self.det = LiveAnomalyDetector()
        self.replay = BVPReplay(subject)
        self.stream = self.replay.stream()
        self.clients: set = set()
        self.running = False
        self._reset()
        # sensitivity (0–1) that corresponds to the saved default threshold
        level0 = self.det.level(self.det.threshold)
        self.sensitivity = float(min(1.0, max(0.0, (0.62 - level0) / 0.40)))

    def _reset(self):
        self.disp_idx = deque(maxlen=DISPLAY)
        self.disp_bvp = deque(maxlen=DISPLAY)
        self.infbuf = deque(maxlen=WIN_LEN)
        self.total = 0
        self.since_infer = 0
        self.level = 0.0
        self.flag = False
        self.score = 0.0
        self.score_ema = None     # smoothed score — flag on sustained stress
        self.label = 0
        self.bpm = None           # heart rate, human-readable context (not the flag)
        self.quality = None       # 0–1 signal-quality estimate (context, not the flag)

    def _ingest(self):
        nidx, nbvp = [], []
        for _ in range(SAMPLES_PER_TICK):
            v, lab = next(self.stream)
            self.disp_idx.append(self.total)
            self.disp_bvp.append(round(v, 2))
            self.infbuf.append(v)
            self.label = lab
            self.total += 1
            self.since_infer += 1
            nidx.append(self.total)
            nbvp.append(round(v, 2))
        return nidx, nbvp

    def _infer(self) -> float:
        return self.det.score(np.fromiter(self.infbuf, dtype=np.float32))

    def _heart_rate(self) -> float | None:
        # HR over a ~12 s tail (responsive; the full 60 s window lags too much).
        tail = list(self.infbuf)[-12 * FS:]
        if len(tail) < 8 * FS:
            return None
        return estimate_heart_rate(np.asarray(tail, dtype=np.float32), fs=FS)

    def _quality(self) -> float | None:
        # signal-quality on a ~8 s tail — context for the UI, never gates the flag.
        tail = list(self.infbuf)[-8 * FS:]
        if len(tail) < 8 * FS:
            return None
        return round(signal_quality_score(np.asarray(tail, dtype=np.float32)), 3)

    def frame(self, nidx, nbvp):
        return {"type": "f", "running": self.running,
                "elapsed": round(self.total / FS, 1),
                "buf": len(self.infbuf), "win": WIN_LEN,
                "idx": nidx, "bvp": nbvp,
                "level": round(self.level, 3), "flag": self.flag,
                "score": round(self.score, 5),
                "bpm": round(self.bpm) if self.bpm else None,
                "quality": self.quality,
                "label": LABELS.get(self.label, "—")}

    async def broadcast(self, payload):
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

    async def producer(self):
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        while True:
            if self.running and self.clients:
                nidx, nbvp = self._ingest()
                if len(self.infbuf) >= WIN_LEN and self.since_infer >= INFER_EVERY:
                    self.since_infer = 0
                    raw = await loop.run_in_executor(None, self._infer)
                    # EMA (~5 s memory) so the flag reflects sustained stress,
                    # not single noisy windows.
                    self.score_ema = (raw if self.score_ema is None
                                      else 0.65 * self.score_ema + 0.35 * raw)
                    self.score = self.score_ema
                    self.level = self.det.level(self.score)
                    self.flag = self.det.flag(self.score)
                    self.bpm = await loop.run_in_executor(None, self._heart_rate)
                    self.quality = await loop.run_in_executor(None, self._quality)
                await self.broadcast(self.frame(nidx, nbvp))
                nxt += TICK_SEC
                d = nxt - loop.time()
                await asyncio.sleep(d if d > 0 else 0)
                if d <= 0:
                    nxt = loop.time()
            else:
                await asyncio.sleep(0.1)
                nxt = loop.time() + TICK_SEC

    def set_sensitivity(self, v: float) -> dict:
        """tune the flag threshold from a 0–1 sensitivity (higher = flags more).

        maps v to a display-level threshold with the SAME curve Pulse Watch uses
        (so the two front-ends agree), converts that to the raw MSE threshold the
        flag compares against, and re-flags the current score immediately. note:
        the threshold is global to this engine (one model, all viewers share it).
        """
        v = float(min(1.0, max(0.0, v)))
        level = min(0.85, max(0.12, 0.62 - 0.40 * v))     # match Pulse Watch setSens()
        self.sensitivity = v
        self.det.threshold = self.det.score_for_level(level)
        if self.score_ema is not None:                    # instant feedback on the live score
            self.level = self.det.level(self.score)
            self.flag = self.det.flag(self.score)
        return {"sensitivity": v, "thr_level": level, "threshold": self.det.threshold}

    def cmd(self, c):
        if c == "start":
            self.running = True
        elif c == "pause":
            self.running = False
        elif c == "reset":
            self.running = False
            self.stream = self.replay.stream()
            self._reset()


engine: Engine | None = None


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(engine.producer())
    try:
        yield
    finally:
        task.cancel()


from fastapi import FastAPI, WebSocket, WebSocketDisconnect   # noqa: E402
from fastapi.responses import FileResponse                    # noqa: E402
from fastapi.staticfiles import StaticFiles                   # noqa: E402

app = FastAPI(lifespan=lifespan)
app.mount("/vendor", StaticFiles(directory=VENDOR_DIR), name="vendor")


@app.get("/")
@app.get("/watch")
async def watch():
    # Zayed's Pulse Watch product UI — the default view, running live on this
    # same pipeline + /ws. (/watch kept as an alias.)
    return FileResponse(os.path.join(PULSE_DIR, "Pulse Watch.dc.html"))


@app.get("/dev")
async def dev():
    # developer dashboard — model output vs ground truth + sensitivity, for tuning
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/support.js")
async def watch_runtime():
    # the .dc runtime that Pulse Watch.dc.html loads via ./support.js
    return FileResponse(os.path.join(PULSE_DIR, "support.js"),
                        media_type="application/javascript")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    engine.clients.add(ws)
    try:
        thr_level = engine.det.level(engine.det.threshold)
        await ws.send_text(json.dumps({
            "type": "hello", "fs": FS, "win_s": WIN_LEN / FS,
            "disp": DISPLAY, "infer_s": INFER_EVERY / FS,
            "thr_level": round(thr_level, 3), "threshold": round(engine.det.threshold, 5),
            "subject": engine.replay.subject, "running": engine.running,
            "source": "replay", "device_connected": False,
            "sensitivity": round(engine.sensitivity, 3)}))
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("cmd") == "set_sensitivity":
                info = engine.set_sensitivity(msg.get("value", 0.5))
                await engine.broadcast({
                    "type": "thr",
                    "threshold": round(info["threshold"], 5),
                    "thr_level": round(info["thr_level"], 3),
                    "sensitivity": round(info["sensitivity"], 3)})
            elif "cmd" in msg:
                engine.cmd(msg["cmd"])
                await engine.broadcast({"type": "state", "running": engine.running})
    except WebSocketDisconnect:
        pass
    finally:
        engine.clients.discard(ws)


def main():
    global engine, SUBJECT
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=SUBJECT)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))
    args = ap.parse_args()
    print(f"  loading model + subject {args.subject}…")
    engine = Engine(args.subject)
    print(f"\n  anomaly dashboard → http://localhost:{args.port}"
          f"   (LAN: http://<this-device-ip>:{args.port})\n")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
