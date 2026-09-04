"""live anomaly dashboard — the trained autoencoder over a 64 Hz BVP stream.

runs the saved model on a rolling 60 s window and pushes the anomaly level +
stress flag to the browser over a WebSocket (uPlot front-end).

two interchangeable sources feed it:

  • **replay** (default) — a curated calm→stress WESAD loop. no hardware needed;
    this is the demo mode, and it still carries ground-truth labels so `/dev`
    can score the model against them.
  • **device** — the real ESP32-S3 + MAX30102 over USB serial, conditioned to
    64 Hz band-passed BVP by `device_source.DeviceSource`. no ground truth, so
    `/dev`'s scorecard goes blank; everything else works the same.

    python3 -m anomaly.serve                       # replay (dummy data)
    python3 -m anomaly.serve --subject S16         # replay, another subject
    python3 -m anomaly.serve --source device       # REAL sensor, auto-detect port
    python3 -m anomaly.serve --source device --device-port /dev/cu.usbmodem101

env equivalents: SOURCE=device, DEVICE_PORT=..., SUBJECT=S16, PORT=8001.
"""
from __future__ import annotations

# Printed BEFORE the heavy imports below. numpy + scipy + tensorflow + a 4 MB
# model take ~15-25 s to load, and without this the terminal sits blank the whole
# time and looks hung — which invites a Ctrl-C mid-import.
if __name__ == "__main__":
    print("\n  starting… loading tensorflow + the 4 MB model.\n"
          "  MEASURED: ~24 s for tensorflow, ~19 s for the model = about 45 s\n"
          "  from a cold cache. This is not a hang — wait for the URL below.\n",
          flush=True)

import argparse
import datetime as _dt
import asyncio
import json
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np

from .wesad import FS
from .wesad_replay import BVPReplay
from .infer import (LiveAnomalyDetector, resolve_scorer,
                    SAVE_DIR, DEVICE_SCORER, WESAD_SCORER)
from .device_calibrate import _Recorder
from .device_check import GATE_PI, GATE_QUALITY, GATE_DRIFT

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


class Calibrator:
    """records the user's own calm live, then makes it the new definition of calm.

    same method as `anomaly.device_calibrate`, driven from the dashboard instead of
    a terminal: learn the gate from this person's first seconds, collect clean 60 s
    windows, score each through the deployed int8 model, and take the 90th
    percentile of that score distribution as the flag threshold — so "normal" is
    this user on this sensor, not WESAD's wrist.
    """

    TICK_SAMPLES = FS // 2           # assess twice a second

    # 60 s windows stepped 5 s share 92% of their samples, so 8 of them cover only
    # ~100 s of real signal -- far too little to estimate a 90th percentile from.
    # 20 windows over ~4 min is still overlapping but spans enough distinct pulse
    # for the calm distribution to have a believable shape.
    def __init__(self, win_len: int, target_sec: float = 240.0, min_windows: int = 20,
                 use_pi: bool = True):
        # perfusion is AC/DC on a real IR signal; the WESAD replay has no DC
        # pedestal, so that gate is meaningless there and would reject everything.
        self.use_pi = use_pi
        self.rec = _Recorder(win_len, step_sec=5.0, adaptive=True, use_pi=use_pi)
        self.win_len = win_len
        self.target_sec = target_sec
        self.min_windows = min_windows
        self.phase = "record"
        self.scores: list[float] = []
        self.pending: list[float] = []
        self.t0 = time.monotonic()
        self.done = False

    def feed(self, vals) -> np.ndarray | None:
        """buffer raw samples; assess and collect on each half-second boundary."""
        self.pending.extend(vals)
        if len(self.pending) < self.TICK_SAMPLES:
            return None
        chunk, self.pending = self.pending, []
        return chunk

    def tick(self, chunk, ir_dc: float):
        return self.rec.feed(chunk, ir_dc)

    def status(self) -> dict:
        el = time.monotonic() - self.t0
        a = self.rec.last or {}
        return {"type": "calib", "phase": self.phase,
                "elapsed": round(el, 1), "target": self.target_sec,
                # progress tracks WINDOWS, not elapsed time: the button unlocks on
                # windows, so a time-based bar sat at 100% with commit still
                # disabled and looked hung.
                "progress": (round(min(1.0, len(self.scores) / self.min_windows), 3)
                             if self.phase == "record" else 0.0),
                "windows": len(self.scores), "min_windows": self.min_windows,
                # RECENT pass rate, not lifetime: the UI turns this into a "signal
                # weak" banner, and a lifetime ratio keeps that banner up long after
                # the signal recovered. null until ticks exist so it cannot flash
                # off a 0/0 ratio at the start.
                "usable": (round(self.rec.recent_frac(), 3)
                           if self.rec.recent_frac() is not None else None),
                "usable_session": round(self.rec.clean_frac(), 3) if self.rec.seen else None,
                "gate": self.rec.gate_desc(),
                "blocking": (a.get("blocking") or [])[:2],
                "can_commit": len(self.scores) >= self.min_windows,
                "done": self.done}

    def result(self) -> dict:
        sc = np.asarray(self.scores, dtype=np.float64)
        lo = float(np.median(sc))
        hi = float(np.quantile(sc, 0.99))
        thr = float(np.quantile(sc, 0.90))
        # A very consistent baseline collapses lo..hi to a hair's width, and the
        # display level (score - lo) / (hi - lo) then saturates to 0 or 1 on noise,
        # which reads as a flag flipping at random. Hold the band open to at least
        # 40% of the median so the gauge stays a gauge.
        span = max(hi - lo, 0.40 * lo)
        return {"threshold": thr, "ref_lo": lo, "ref_hi": lo + span, "n": len(sc)}


class Engine:
    def __init__(self, mode: str = "replay", subject: str = SUBJECT,
                 device_port: str | None = None, baud: int = 115200,
                 invert: bool = False, scorer: str | None = None):
        # the device flags against its OWN calm once anomaly.device_calibrate has
        # been run; the replay always flags against the WESAD scorer it was cut for.
        self.det = LiveAnomalyDetector(scorer=scorer or resolve_scorer(mode))
        self.mode = mode
        if mode == "device":
            from .device_source import DeviceSource
            self.source = DeviceSource(port=device_port, baud=baud,
                                       invert=invert).start()
            self.subject = "device"
        else:
            self.source = BVPReplay(subject)
            self.subject = subject
        self.stream = self.source.stream()
        self.calib: Calibrator | None = None
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
        self._last_hr_at = 0
        self._hr_push_logged = False
        self.level = 0.0
        self.flag = False
        self.score = 0.0
        self.score_ema = None     # smoothed score — flag on sustained stress
        self.label = 0
        self.bpm = None           # heart rate, human-readable context (not the flag)
        self.quality = None       # 0–1 signal-quality estimate (context, not the flag)
        self.spo2 = None          # device-reported only; the replay has no RED channel

    def _ingest(self):
        # the replay is infinite, so we set its pace. a real device paces itself:
        # take only what has actually arrived (bounded, so a backlog drains fast)
        # and never call next() past that — otherwise a slow or unplugged board
        # would block the async producer.
        n = SAMPLES_PER_TICK
        if self.mode == "device":
            n = min(self.source.available(), SAMPLES_PER_TICK * 4)

        nidx, nbvp = [], []
        self._new_raw = []
        for _ in range(n):
            v, lab = next(self.stream, (None, None))
            if v is None:
                break
            self.disp_idx.append(self.total)
            self.disp_bvp.append(round(v, 2))
            self.infbuf.append(v)
            self._new_raw.append(v)
            self.label = lab
            self.total += 1
            self.since_infer += 1
            nidx.append(self.total)
            nbvp.append(round(v, 2))

        if self.mode == "device":
            # SpO2 is measured on the board, so surface it as soon as it arrives
            # rather than gating it behind the model's 60 s warm-up. same for the
            # board's own BPM, until our 64 Hz estimator has enough signal.
            # SpO2 is a red/IR AC-DC ratio, independent of sample rate, so the
            # board's value is sound and worth showing immediately. its HR is not.
            self.spo2 = self.source.device_spo2
        return nidx, nbvp

    def _infer(self) -> float:
        return self.det.score(np.fromiter(self.infbuf, dtype=np.float32))

    def _heart_rate(self) -> float | None:
        # HR over a ~12 s tail (responsive; the full 60 s window lags too much).
        tail = list(self.infbuf)[-12 * FS:]
        if len(tail) < 8 * FS:
            return None
        # NOT falling back to self.source.device_bpm: the sketch's maxim estimate
        # assumes the library's hardcoded FreqS=25 while the board delivers ~40 Hz,
        # so it reads about 0.625x true. better no number than a wrong one.
        return estimate_heart_rate(np.asarray(tail, dtype=np.float32), fs=FS)

    def _quality(self) -> float | None:
        # signal-quality on a ~8 s tail — context for the UI, never gates the flag.
        tail = list(self.infbuf)[-8 * FS:]
        if len(tail) < 8 * FS:
            return None
        return round(signal_quality_score(np.asarray(tail, dtype=np.float32)), 3)

    def frame(self, nidx, nbvp):
        f = {"type": "f", "running": self.running,
             "elapsed": round(self.total / FS, 1),
             "buf": len(self.infbuf), "win": WIN_LEN,
             "idx": nidx, "bvp": nbvp,
             "level": round(self.level, 3), "flag": self.flag,
             "score": round(self.score, 5),
             "bpm": round(self.bpm) if self.bpm else None,
             "spo2": self.spo2,
             "quality": self.quality,
             "label": LABELS.get(self.label, "—")}
        if self.mode == "device":
            st = self.source.status()
            f["device"] = {"connected": st["connected"], "port": st["port"],
                           "contact": st["contact"], "dropped": st["dropped"]}
        return f

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
                self.calib_step()
                # HR and signal quality are cheap and have nothing to do with the
                # model's 60 s window -- computing them here means the board gets
                # our number from ~8 s in, not after a minute of warm-up.
                if self.total - self._last_hr_at >= FS:
                    self._last_hr_at = self.total
                    self.bpm = await loop.run_in_executor(None, self._heart_rate)
                    self.quality = await loop.run_in_executor(None, self._quality)
                    if self.mode == "device":
                        # one source of truth: the TFT shows OUR value, so the two
                        # displays cannot disagree. the board reverts to its own
                        # estimate if we stop sending.
                        self.source.send_bpm(self.bpm)
                        if self.bpm and not self._hr_push_logged:
                            self._hr_push_logged = True
                            print(f"  display sync: pushing HR to the board "
                                  f"(first value H,{round(self.bpm)})", flush=True)
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
                if self.calib is not None and self.total % FS < SAMPLES_PER_TICK:
                    st = self.calib.status()
                    await self.broadcast(st)
                    # mirror it to the terminal: when a session collects nothing the
                    # UI can only say "signal weak", while this names the gate.
                    if st["phase"] == "record":
                        print(f"  calib {st['elapsed']:5.0f}s  windows {st['windows']:>2}/"
                              f"{st['min_windows']}  recent "
                              f"{(st['usable'] or 0) * 100:3.0f}%  session "
                              f"{(st['usable_session'] or 0) * 100:3.0f}%  "
                              f"gate {st['gate']}  "
                              f"{'; '.join(st['blocking'])}", flush=True)
                nxt += TICK_SEC
                d = nxt - loop.time()
                await asyncio.sleep(d if d > 0 else 0)
                if d <= 0:
                    nxt = loop.time()
            else:
                await asyncio.sleep(0.1)
                nxt = loop.time() + TICK_SEC

    def calib_start(self) -> dict:
        self.calib = Calibrator(WIN_LEN, use_pi=(self.mode == "device"))
        return self.calib.status()

    def calib_cancel(self):
        self.calib = None

    def calib_step(self):
        """advance a running calibration with whatever samples just arrived."""
        c = self.calib
        if c is None or c.done:
            return
        chunk = c.feed(getattr(self, "_new_raw", []))
        if chunk is None:
            return
        ir = self.source.status()["ir_dc"] if self.mode == "device" else 1.2e5
        w = c.tick(chunk, ir)
        if w is not None:
            c.scores.append(self.det.score(w))

    def calib_commit(self) -> dict | None:
        """make the recorded calm the model's new normal — live and on disk."""
        c = self.calib
        if c is None or len(c.scores) < c.min_windows:
            return None
        r = c.result()
        np.savez(os.path.join(SAVE_DIR, DEVICE_SCORER),
                 threshold=r["threshold"], win_len=int(WIN_LEN),
                 ref_lo=r["ref_lo"], ref_hi=r["ref_hi"],
                 source="device", n_windows=r["n"], fs=int(FS),
                 created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        # apply to the running detector so the flag changes meaning immediately
        self.det.threshold = r["threshold"]
        self.det.ref_lo = r["ref_lo"]
        self.det.ref_hi = r["ref_hi"]
        self.det.calibrated_on = "device"
        self.det.scorer_name = DEVICE_SCORER
        level0 = self.det.level(self.det.threshold)
        self.sensitivity = float(min(1.0, max(0.0, (0.62 - level0) / 0.40)))
        if self.score_ema is not None:
            self.level = self.det.level(self.score)
            self.flag = self.det.flag(self.score)
        c.done = True
        c.phase = "done"
        self.calib = None      # stop the per-second status, which was reverting the UI
        return r

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
            if self.mode == "replay":
                self.stream = self.source.stream()   # rewind the loop
            self._reset()                            # device: just clear buffers


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
        dev = engine.source.status() if engine.mode == "device" else None
        await ws.send_text(json.dumps({
            "type": "hello", "fs": FS, "win_s": WIN_LEN / FS,
            "disp": DISPLAY, "infer_s": INFER_EVERY / FS,
            "thr_level": round(thr_level, 3), "threshold": round(engine.det.threshold, 5),
            "subject": engine.subject, "running": engine.running,
            "source": engine.mode,
            "calibrated_on": engine.det.calibrated_on,
            "device_connected": bool(dev and dev["connected"]),
            "device_port": dev["port"] if dev else None,
            "sensitivity": round(engine.sensitivity, 3)}))
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("cmd") == "calib_start":
                await engine.broadcast(engine.calib_start())
            elif msg.get("cmd") == "calib_cancel":
                engine.calib_cancel()
                await engine.broadcast({"type": "calib", "phase": "cancelled",
                                        "progress": 0.0, "windows": 0, "done": False})
            elif msg.get("cmd") == "calib_commit":
                r = engine.calib_commit()
                if r is None:
                    await engine.broadcast({"type": "calib", "phase": "record",
                                            "error": "not enough clean windows yet"})
                else:
                    await engine.broadcast({"type": "calib", "phase": "done",
                                            "progress": 1.0, "done": True,
                                            "windows": r["n"],
                                            "threshold": round(r["threshold"], 5)})
                    await engine.broadcast({
                        "type": "thr",
                        "threshold": round(engine.det.threshold, 5),
                        "thr_level": round(engine.det.level(engine.det.threshold), 3),
                        "sensitivity": round(engine.sensitivity, 3),
                        "calibrated_on": "device"})
            elif msg.get("cmd") == "set_sensitivity":
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
    global engine
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("replay", "device"),
                    default=os.environ.get("SOURCE", "replay"),
                    help="replay = WESAD demo loop (default); device = real MAX30102")
    ap.add_argument("--subject", default=SUBJECT, help="replay subject")
    ap.add_argument("--device-port", default=os.environ.get("DEVICE_PORT"),
                    help="serial port of the board (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--invert", action="store_true",
                    help="flip the device waveform if the pulse reads upside-down")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")),
                    help="HTTP port for the dashboard")
    ap.add_argument("--scorer", choices=("auto", "wesad", "device"), default="auto",
                    help="which thresholds to flag against: auto = the device's own "
                         "once calibrated (default), else WESAD's")
    args = ap.parse_args()

    scorer = {"auto": None, "wesad": WESAD_SCORER, "device": DEVICE_SCORER}[args.scorer]
    if scorer == DEVICE_SCORER and not os.path.exists(os.path.join(SAVE_DIR, DEVICE_SCORER)):
        print(f"  --scorer device: {DEVICE_SCORER} does not exist yet — record "
              "a calm baseline first:")
        print("      python3 -m anomaly.device_calibrate")
        return 1

    if args.source == "device":
        print("  source: REAL sensor (MAX30102 over serial)")
    else:
        print(f"  source: replay — WESAD {args.subject} (dummy data)")
    print("  loading model…")
    engine = Engine(mode=args.source, subject=args.subject,
                    device_port=args.device_port, baud=args.baud,
                    invert=args.invert, scorer=scorer)
    if engine.det.calibrated_on == "device":
        print(f"  flag: calibrated on THIS sensor ({engine.det.scorer_name})")
    elif args.source == "device":
        print("  flag: NOT calibrated for this sensor — thresholds come from WESAD wrist")
        print("        BVP, so the level and flag are not meaningful yet. waveform, HR")
        print("        and SpO2 are real. fix with: python3 -m anomaly.device_calibrate")
    if args.source == "device":
        import time as _t
        deadline = _t.monotonic() + 3.0          # give the reader a moment to latch on
        while _t.monotonic() < deadline and not engine.source.status()["connected"]:
            _t.sleep(0.2)
        st = engine.source.status()
        if st["connected"]:
            print(f"  serial: streaming from {st['port']}")
        else:
            why = st["error"] or "no data yet"
            print(f"  serial: not streaming ({why}) — the reader keeps retrying, "
                  "so the dashboard starts either way")
            print("          list ports with: python3 -m anomaly.device_source --list-ports")
    print(f"\n  anomaly dashboard → http://localhost:{args.port}"
          f"   (LAN: http://<this-device-ip>:{args.port})\n")
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    finally:
        if args.source == "device":
            engine.source.close()


if __name__ == "__main__":
    raise SystemExit(main())
