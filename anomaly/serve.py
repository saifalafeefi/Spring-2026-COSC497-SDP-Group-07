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

    def frame(self, nidx, nbvp):
        return {"type": "f", "running": self.running,
                "elapsed": round(self.total / FS, 1),
                "buf": len(self.infbuf), "win": WIN_LEN,
                "idx": nidx, "bvp": nbvp,
                "level": round(self.level, 3), "flag": self.flag,
                "score": round(self.score, 5),
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
                await self.broadcast(self.frame(nidx, nbvp))
                nxt += TICK_SEC
                d = nxt - loop.time()
                await asyncio.sleep(d if d > 0 else 0)
                if d <= 0:
                    nxt = loop.time()
            else:
                await asyncio.sleep(0.1)
                nxt = loop.time() + TICK_SEC

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
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


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
            "subject": engine.replay.subject, "running": engine.running}))
        while True:
            msg = json.loads(await ws.receive_text())
            if "cmd" in msg:
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
