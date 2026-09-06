"""the master: find every ESP32 on the LAN, score them all, serve a roster.

ONE command for the whole system.

    python3 -m anomaly.fleet

it scans the local subnet for boards, opens a websocket to each, runs the
autoencoder on their streams, and pushes each verdict back to that board's
/flag. a roster at http://localhost:8002/ lists what it found; clicking a device
opens THAT device's own dashboard, served from the device itself.

the division of labour:

    ESP32                     master (this)
    -----                     -------------
    sensor + conditioning     the ML model, and only the ML model
    heart rate, SpO2          per-device calibration
    its own dashboard         the roster
    TFT verdict display       discovery

each board is exclusive to its own user: a device's dashboard shows only that
device, and boards never learn about each other. only the master sees the fleet.

when the master is unreachable a board keeps sensing, keeps serving its page and
keeps showing HR/SpO2 -- it just reports no verdict. an on-device model is what
would fill that slot later.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import socket
import sys
import time
import urllib.parse
import urllib.request
from collections import deque

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .infer import SAVE_DIR
from .wesad import FS

WIN = 60 * FS
HTTP_TIMEOUT = 1.0

# Every network call here is blocking urllib, so it MUST run off the event loop:
# a 2 s push that stalls the loop stalls every other device's stream too. The
# default executor caps at ~32 threads, which made a two-subnet scan take 16 s.
NET = ThreadPoolExecutor(max_workers=128, thread_name_prefix="net")


# --------------------------------------------------------------------- helpers

def local_subnets() -> list:
    """every private /24 this machine is on, as strings like '192.168.1'.

    ALL of them, not just the default route: a Windows machine sharing its
    connection has both its own network and the hotspot's (192.168.137.x), and
    the board is usually on the hotspot while the default route is the other one.
    Scanning only the default route silently finds nothing.
    """
    found = []
    try:
        for r in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(r[4][0])
    except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # no packet is sent; picks the route
        found.append(s.getsockname()[0])
    except Exception:
        pass
    finally:
        s.close()

    nets = []
    for ip in found:
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if not a.is_private or a.is_loopback:
            continue
        net = ".".join(ip.split(".")[:3])
        if net not in nets:
            nets.append(net)
    return nets


def probe(ip: str) -> dict | None:
    """is there one of our boards at this address?"""
    try:
        with urllib.request.urlopen(f"http://{ip}/health", timeout=HTTP_TIMEOUT) as r:
            if r.status != 200:
                return None
            body = r.read().decode("ascii", "ignore")
    except Exception:
        return None
    if not body.startswith("ok "):
        return None
    out = {"ip": ip}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out if "id" in out else None


async def scan(subnet: str, concurrency: int = 64) -> list:
    """probe every host on the /24 at once. ~2-5 s for 254 addresses."""
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    async def one(n: int):
        async with sem:
            return await loop.run_in_executor(NET, probe, f"{subnet}.{n}")

    found = await asyncio.gather(*[one(n) for n in range(1, 255)])
    return [f for f in found if f]


def push_flag(ip: str, flag: bool, level: float) -> bool:
    q = urllib.parse.urlencode({"f": 1 if flag else 0,
                                "l": int(round(max(0.0, min(1.0, level)) * 100))})
    try:
        with urllib.request.urlopen(f"http://{ip}/flag?{q}", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


def scorer_path(dev_id: str) -> str:
    return os.path.join(SAVE_DIR, f"scorer_{dev_id}.npz")


# ----------------------------------------------------------------- one device

class Device:
    """one board: its websocket, its scorer, its calibration session."""

    def __init__(self, dev_id: str, ip: str, det):
        self.id = dev_id
        self.ip = ip
        self.det = det                    # LiveAnomalyDetector, thresholds swapped per device
        self.buf: deque = deque(maxlen=WIN)
        self.ema = None
        self.level = None
        self.flag = False
        self.score = 0.0
        self.bpm = None
        self.spo2 = None
        self.contact = False
        self.last_seen = 0.0
        self.connected = False
        self.pushes = 0
        self.fails = 0
        self.calib = None                 # a CalibSession while one is running
        self.thresholds = None            # (threshold, lo, hi) once calibrated
        self.load_scorer()

    # ---- per-device calibration ----

    def load_scorer(self) -> bool:
        p = scorer_path(self.id)
        if not os.path.exists(p):
            self.thresholds = None
            return False
        z = np.load(p)
        self.thresholds = (float(z["threshold"]), float(z["ref_lo"]), float(z["ref_hi"]))
        return True

    def save_scorer(self, threshold: float, lo: float, hi: float, n: int):
        import datetime as dt
        np.savez(scorer_path(self.id), threshold=threshold, ref_lo=lo, ref_hi=hi,
                 win_len=int(WIN), source="device", n_windows=n, fs=int(FS),
                 device_id=self.id,
                 created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        self.thresholds = (threshold, lo, hi)

    @property
    def calibrated(self) -> bool:
        return self.thresholds is not None

    def score_window(self) -> float:
        return self.det.score(np.fromiter(self.buf, dtype=np.float32, count=WIN))

    def apply(self, raw: float):
        """turn a raw reconstruction error into a level and a flag for THIS user."""
        self.ema = raw if self.ema is None else 0.65 * self.ema + 0.35 * raw
        self.score = self.ema
        if not self.calibrated:
            self.level, self.flag = None, False
            return
        thr, lo, hi = self.thresholds
        self.level = float(np.clip((self.ema - lo) / (hi - lo + 1e-9), 0.0, 1.0))
        self.flag = bool(self.ema >= thr)

    def status(self) -> dict:
        stale = time.monotonic() - self.last_seen if self.last_seen else None
        return {
            "id": self.id, "ip": self.ip,
            "connected": self.connected and stale is not None and stale < 5,
            "contact": self.contact,
            "bpm": self.bpm, "spo2": self.spo2,
            "level": None if self.level is None else round(self.level, 3),
            "flag": self.flag,
            "score": round(self.score, 5),
            "calibrated": self.calibrated,
            "buf": len(self.buf), "win": WIN,
            "pushes": self.pushes, "fails": self.fails,
            "last_seen": None if stale is None else round(stale, 1),
            "calib": self.calib.status() if self.calib else None,
        }


class CalibSession:
    """collect this user's calm and derive their thresholds. master-side, because
    the model lives here -- a board can stream but cannot score its own windows."""

    TARGET = 20                    # windows needed before commit is allowed

    def __init__(self):
        self.scores: list = []
        self.t0 = time.monotonic()
        self.next_at = 0
        self.done = False

    def offer(self, dev: Device):
        """one scoring opportunity; take a window every 5 s of clean contact."""
        if self.done or len(dev.buf) < WIN or not dev.contact:
            return
        now = time.monotonic()
        if now < self.next_at:
            return
        self.next_at = now + 5.0
        self.scores.append(dev.score_window())

    def status(self) -> dict:
        return {"windows": len(self.scores), "target": self.TARGET,
                "elapsed": round(time.monotonic() - self.t0, 1),
                "can_commit": len(self.scores) >= self.TARGET,
                "done": self.done}

    def result(self) -> dict:
        sc = np.asarray(self.scores, dtype=np.float64)
        lo = float(np.median(sc))
        hi = float(np.quantile(sc, 0.99))
        span = max(hi - lo, 0.40 * lo)     # a hair-thin band makes the gauge useless
        return {"threshold": float(np.quantile(sc, 0.90)),
                "ref_lo": lo, "ref_hi": lo + span, "n": len(sc)}


# ------------------------------------------------------------------ the fleet

class Fleet:
    def __init__(self, det, subnets: list, extra: list):
        self.det = det
        self.subnets = subnets
        self.extra = extra
        self.devices: dict = {}
        self.scanning = False
        self.last_scan = 0.0

    async def discover(self):
        self.scanning = True
        try:
            found = []
            for net in self.subnets:
                found.extend(await scan(net))
            for ip in self.extra:
                info = await asyncio.get_running_loop().run_in_executor(NET, probe, ip)
                if info:
                    found.append(info)
            for info in found:
                dev_id, ip = info["id"], info["ip"]
                if dev_id in self.devices:
                    self.devices[dev_id].ip = ip      # DHCP may have moved it
                    continue
                dev = Device(dev_id, ip, self.det)
                self.devices[dev_id] = dev
                print(f"  found {dev_id} at {ip}"
                      f"{'' if dev.calibrated else '   (not calibrated)'}", flush=True)
                asyncio.create_task(self.pump(dev))
        finally:
            self.scanning = False
            self.last_scan = time.monotonic()

    async def pump(self, dev: Device):
        """one task per device: read its stream, score it, push the verdict back."""
        import websockets
        while True:
            try:
                async with websockets.connect(f"ws://{dev.ip}/ws",
                                              open_timeout=8, max_queue=64) as ws:
                    dev.connected = True
                    next_score = time.monotonic() + 1.0
                    async for raw in ws:
                        try:
                            m = json.loads(raw)
                        except Exception:
                            continue
                        if m.get("type") != "f":
                            continue
                        dev.last_seen = time.monotonic()
                        d = m.get("device") or {}
                        dev.contact = bool(d.get("contact"))
                        dev.bpm = m.get("bpm")
                        dev.spo2 = m.get("spo2")
                        if not dev.contact:
                            dev.buf.clear()        # noise must never enter a window
                            dev.level, dev.flag, dev.ema = None, False, None
                            continue
                        for v in (m.get("bvp") or []):
                            dev.buf.append(float(v))

                        if dev.calib:
                            dev.calib.offer(dev)

                        now = time.monotonic()
                        if now < next_score or len(dev.buf) < WIN:
                            continue
                        next_score = now + 1.0
                        dev.apply(dev.score_window())
                        if dev.calibrated:
                            # off the loop: a slow board must not stall the others
                            ok = await asyncio.get_running_loop().run_in_executor(
                                NET, push_flag, dev.ip, dev.flag, dev.level or 0.0)
                            dev.pushes += ok
                            dev.fails += (not ok)
            except Exception:
                dev.connected = False
                await asyncio.sleep(3.0)

    async def rescan_loop(self, every: float):
        while True:
            await asyncio.sleep(every)
            await self.discover()


# ----------------------------------------------------------------- the server

def build_app(fleet: Fleet):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI()
    here = os.path.dirname(os.path.abspath(__file__))

    @app.get("/")
    async def index():
        with open(os.path.join(here, "static", "fleet.html"), encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.get("/api/devices")
    async def devices():
        return JSONResponse({
            "scanning": fleet.scanning,
            "subnet": ", ".join(fleet.subnets),
            "devices": [d.status() for d in
                        sorted(fleet.devices.values(), key=lambda x: x.id)],
        })

    @app.post("/api/rescan")
    async def rescan():
        asyncio.create_task(fleet.discover())
        return {"ok": True}

    @app.post("/api/calib/{dev_id}/{action}")
    async def calib(dev_id: str, action: str):
        dev = fleet.devices.get(dev_id)
        if dev is None:
            return JSONResponse({"error": "unknown device"}, status_code=404)
        if action == "start":
            dev.calib = CalibSession()
            return {"ok": True}
        if action == "cancel":
            dev.calib = None
            return {"ok": True}
        if action == "commit":
            if not dev.calib or len(dev.calib.scores) < CalibSession.TARGET:
                return JSONResponse({"error": "not enough clean windows yet"},
                                    status_code=400)
            r = dev.calib.result()
            dev.save_scorer(r["threshold"], r["ref_lo"], r["ref_hi"], r["n"])
            dev.calib.done = True
            dev.calib = None
            print(f"  {dev_id}: calibrated on {r['n']} windows, "
                  f"threshold {r['threshold']:.5f}", flush=True)
            return {"ok": True, **r}
        return JSONResponse({"error": "unknown action"}, status_code=400)

    return app


async def amain(args) -> int:
    from .infer import LiveAnomalyDetector

    subnets = [args.subnet] if args.subnet else local_subnets()
    if not subnets:
        print("  could not work out this machine's subnet; pass --subnet 192.168.1")
        return 1

    print("  loading model… (~45 s: TensorFlow + the 4 MB int8 model)", flush=True)
    det = LiveAnomalyDetector()          # thresholds come per device, not from here
    print("  model ready\n")

    fleet = Fleet(det, subnets, args.device or [])
    print("  scanning " + ", ".join(f"{n}.0/24" for n in subnets) + " for boards…",
          flush=True)
    await fleet.discover()
    if not fleet.devices:
        print("  none found. is a board powered and on this network?")
        print("  (looking for an HTTP /health on "
              + ", ".join(f"{n}.1-254" for n in subnets) + ")")
    asyncio.create_task(fleet.rescan_loop(args.rescan))

    import uvicorn
    print(f"\n  roster -> http://localhost:{args.port}\n")
    cfg = uvicorn.Config(build_app(fleet), host="0.0.0.0", port=args.port,
                         log_level="warning")
    await uvicorn.Server(cfg).serve()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subnet", help="e.g. 192.168.1 (default: this machine's)")
    ap.add_argument("--device", action="append",
                    help="extra IP to probe, for boards off this subnet; repeatable")
    ap.add_argument("--port", type=int, default=8002, help="roster port (default 8002)")
    ap.add_argument("--rescan", type=float, default=30.0,
                    help="seconds between rescans (default 30)")
    args = ap.parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n  stopped.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
