"""score a networked ESP32's stream and push the verdict back to it.

the board hosts its own dashboard and streams conditioned 64 Hz BVP; it has no
model. this connects as a websocket client, runs the deployed autoencoder on the
host, and pushes the result to the board's /flag endpoint. the board shows it on
the TFT and forwards it to every browser watching -- and falls back to "--" on
its own if we stop sending.

    python3 -m anomaly.master --host 192.168.1.198

no USB needed: the waveform arrives over WiFi and the verdict goes back the same
way. note the raw waveform does leave the device in this arrangement, which is
the tradeoff for scoring on the host -- an on-device model is what makes the
privacy claim in the proposal true.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import deque

import numpy as np

from .wesad import FS

WIN = 60 * FS               # the model's input, 3840 samples


def normalise_host(host: str) -> str:
    """accept 10.0.0.5, http://10.0.0.5, ws://10.0.0.5/, and so on.

    pasting the URL straight from the browser is the obvious thing to do, and it
    used to produce "ws://http://10.0.0.5//ws" and a DNS failure.
    """
    h = host.strip()
    for scheme in ("http://", "https://", "ws://", "wss://"):
        if h.lower().startswith(scheme):
            h = h[len(scheme):]
    return h.strip("/").split("/")[0]


def push_flag(host: str, flag: bool, level: float, timeout: float = 2.0) -> bool:
    q = urllib.parse.urlencode({"f": 1 if flag else 0,
                                "l": int(round(max(0.0, min(1.0, level)) * 100))})
    try:
        with urllib.request.urlopen(f"http://{host}/flag?{q}", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


async def run(host: str, every: float) -> int:
    import websockets
    from .infer import LiveAnomalyDetector, resolve_scorer

    print("  loading model… (~45 s: TensorFlow + the 4 MB int8 model)", flush=True)
    det = LiveAnomalyDetector(scorer=resolve_scorer("device"))
    print(f"  model ready, thresholds from {det.scorer_name} "
          f"(calibrated on {det.calibrated_on})")
    if det.calibrated_on != "device":
        print("  WARNING: these thresholds are WESAD wrist, not this sensor.")
        print("           run anomaly.device_calibrate -- the flag is arbitrary until then.")

    buf: deque = deque(maxlen=WIN)
    url = f"ws://{host}/ws"
    ema = None
    last_score = 0.0
    pushes = fails = 0

    while True:
        try:
            print(f"\n  connecting to {url}")
            async with websockets.connect(url, open_timeout=8, max_queue=64) as ws:
                print("  connected — scoring once every "
                      f"{every:g}s, pushing to http://{host}/flag\n")
                next_at = time.monotonic() + every
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except Exception:
                        continue
                    if m.get("type") != "f":
                        continue
                    for v in (m.get("bvp") or []):
                        buf.append(float(v))

                    now = time.monotonic()
                    if now < next_at:
                        continue
                    next_at = now + every

                    if len(buf) < WIN:
                        print(f"\r  filling window {len(buf)}/{WIN}"
                              f"  ({(WIN - len(buf)) / FS:.0f}s left) ", end="", flush=True)
                        continue

                    contact = (m.get("device") or {}).get("contact", True)
                    if not contact:
                        print("\r  no finger — not scoring, not pushing        ",
                              end="", flush=True)
                        buf.clear()          # noise must not enter the window
                        continue

                    raw_score = det.score(np.fromiter(buf, dtype=np.float32, count=WIN))
                    # same EMA the dashboard used: flag on sustained deviation,
                    # not on one noisy window
                    ema = raw_score if ema is None else 0.65 * ema + 0.35 * raw_score
                    last_score = ema
                    level = det.level(ema)
                    flag = det.flag(ema)

                    ok = push_flag(host, flag, level)
                    pushes += ok
                    fails += (not ok)
                    print(f"\r  score {ema:.5f}  level {level*100:5.1f}%  "
                          f"{'FLAG' if flag else 'calm'}   "
                          f"pushed {pushes} failed {fails}   ", end="", flush=True)
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print(f"\n  link lost ({type(e).__name__}: {e}); retrying in 3 s")
            print("  the board falls back to its own display after 5 s")
            buf.clear()
            ema = None
            await asyncio.sleep(3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True,
                    help="board IP or URL, e.g. 10.49.10.173 or http://10.49.10.173/")
    ap.add_argument("--every", type=float, default=1.0,
                    help="seconds between scores (default: 1)")
    args = ap.parse_args()
    host = normalise_host(args.host)
    if host != args.host:
        print(f"  (using host {host!r})")
    try:
        return asyncio.run(run(host, args.every))
    except KeyboardInterrupt:
        print("\n  stopped.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
