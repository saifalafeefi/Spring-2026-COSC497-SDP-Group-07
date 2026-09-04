"""live grip coach for the MAX30102 — tells you if you're holding it right.

PPG quality is dominated by how the finger sits on the sensor, and the failure
modes look identical on a chart ("spiky") while having opposite fixes: too much
pressure occludes the capillaries and kills the pulse, too little lets the finger
slide and lets ambient light in. this separates them and says which one you have.

    python3 -m anomaly.device_check
    python3 -m anomaly.device_check --seconds 60 --port /dev/cu.usbmodem14301

what it measures, per ~4 s window:

  contact   IR DC level. below ~50k the finger is off; above ~240k the ADC is
            railing and the waveform clips flat.
  pulse     perfusion index = AC amplitude / DC, as a percent. this is the actual
            strength of the pulse reaching the sensor. squeezing HARDER makes
            this go DOWN, which is the counterintuitive part.
  still     how far the DC baseline wanders. rises when the finger rolls, slides,
            or changes pressure — the usual source of "crazy spikes".
  spikes    samples past 4 sigma. artifacts, not physiology.
  quality   the same 0-1 index the dashboard gates on (pipeline/vitals.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

from .device_source import DeviceSource
from .wesad import FS

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from vitals import estimate_heart_rate, signal_quality_score  # noqa: E402

# thresholds, tuned to what this rig actually produced on a fingertip
DC_LOW, DC_GOOD_LO, DC_GOOD_HI, DC_SAT = 50_000, 80_000, 200_000, 240_000
PI_WEAK, PI_GOOD = 0.3, 1.0        # perfusion index, percent
DRIFT_OK, DRIFT_BAD = 0.05, 0.12   # DC wander as a fraction of DC

# The RECORDING GATE, deliberately looser than the display verdicts above.
# Those verdicts coach you toward an ideal grip; this decides whether a window is
# usable. DC only has to prove the finger is on and the ADC is not railing —
# perfusion index is the direct measure of how much pulse is reaching the sensor,
# so 61k DC at 4% PI is a better trace than 116k DC at 1.8%, and gating on the DC
# band as well was double-counting with the wrong measure in charge. The quality
# bar is 0.60 because real PPG has sharp systolic peaks and scores lower on an
# amplitude-consistency index than a smooth synthetic wave does.
GATE_PI = 0.5          # percent -- the bottom of the healthy 0.5-5% range
                       # documented above. 1.0 rejected the lower fifth of
                       # normal human perfusion, which is not a defect.
GATE_DRIFT = 0.08      # DC wander as a fraction of DC
GATE_QUALITY = 0.60


def bar(frac: float, width: int = 10) -> str:
    n = int(max(0.0, min(1.0, frac)) * width)
    return "#" * n + "." * (width - n)


def contact_verdict(dc: float) -> tuple[str, str]:
    """IR DC level -> (verdict, what to do about it)."""
    if dc < DC_LOW:
        return "NO FINGER", "rest a fingertip over the whole window"
    if dc < DC_GOOD_LO:
        return "LOOSE", "cover the sensor fully; a little more contact"
    if dc > DC_SAT:
        return "SATURATED", "too much light — ease off / lower ledBrightness"
    if dc > DC_GOOD_HI:
        return "HIGH", "slightly too much pressure"
    return "GOOD", ""


def assess(recent, dc: float, dc_hist, min_pi: float = GATE_PI,
           max_drift: float = GATE_DRIFT, min_quality: float = GATE_QUALITY) -> dict:
    """the numeric half of the grip check, with no printing.

    `recent` is a short tail of conditioned BVP (~4 s), `dc` the current IR DC
    level, `dc_hist` the recent DC history used for drift. shared with
    `device_calibrate`, so "clean enough to record" means exactly one thing:
    `ready` here is the same test that prints GOOD TO RECORD below.

    `blocking` lists the gates that failed, so the UI can say why instead of
    just "not clean enough"; `block_kind` is the first failure, for tallying.
    """
    recent = np.asarray(recent, dtype=np.float64)
    if recent.size == 0:
        return {"dc": dc, "pi": 0.0, "drift": 0.0, "spikes": 0, "quality": 0.0,
                "contact": "NO FINGER", "contact_fix": "", "on_skin": False,
                "ready": False, "blocking": ["no signal yet"], "block_kind": "contact"}
    ac = float(recent.max() - recent.min())
    pi = (ac / dc * 100.0) if dc > 0 else 0.0
    sd = float(recent.std())
    spikes = int(np.sum(np.abs(recent - recent.mean()) > 4 * sd)) if sd > 0 else 0
    drift = (float(np.std(dc_hist)) / dc) if dc > 0 and len(dc_hist) > 4 else 0.0
    q = float(signal_quality_score(recent))
    verdict, fix = contact_verdict(dc)
    on_skin = dc >= DC_LOW

    blocking, kinds = [], []
    if not on_skin:
        blocking.append("no finger on the sensor"); kinds.append("contact")
    elif dc > DC_SAT:
        blocking.append(f"ADC saturated (IR {dc:,.0f})"); kinds.append("contact")
    if pi < min_pi:
        blocking.append(f"PI {pi:.2f}% < {min_pi:.2f}%"); kinds.append("pulse")
    if drift > max_drift:
        blocking.append(f"drift {drift*100:.1f}% > {max_drift*100:.0f}%"); kinds.append("motion")
    if q < min_quality:
        blocking.append(f"quality {q:.2f} < {min_quality:.2f}"); kinds.append("quality")

    return {"dc": dc, "pi": pi, "drift": drift, "spikes": spikes, "quality": q,
            "contact": verdict, "contact_fix": fix, "on_skin": on_skin,
            "ready": not blocking, "blocking": blocking,
            "block_kind": kinds[0] if kinds else ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="serial port (default: auto-detect)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after N seconds (default: run until Ctrl-C)")
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()

    src = DeviceSource(port=args.port, invert=args.invert).start()
    stream = src.stream()
    bvp: deque = deque(maxlen=FS * 12)     # conditioned, for HR
    dc_hist: deque = deque(maxlen=40)      # IR DC samples, for drift
    hr_hist: deque = deque(maxlen=8)
    was_on_skin = False

    print("\n  GRIP CHECK — rest a fingertip on the sensor, Ctrl-C to stop\n")
    painted = 0
    t0 = time.monotonic()

    try:
        while True:
            time.sleep(0.5)
            for _ in range(src.available()):
                v = next(stream)[0]
                if v is None:
                    break
                bvp.append(v)
            st = src.status()
            dc = st["ir_dc"]

            # Placing a finger steps the DC from ~12k to ~120k. Keeping that step
            # in the history makes a motionless finger read as 13% "drift", so
            # start the drift window fresh at the moment contact is made.
            now_on_skin = dc >= DC_LOW
            if now_on_skin and not was_on_skin:
                dc_hist.clear()
                bvp.clear()
            was_on_skin = now_on_skin
            dc_hist.append(dc)

            lines = []
            if not st["connected"]:
                lines.append(f"  waiting for the board... ({st['error'] or 'no data'})")
            elif len(bvp) < FS * 3:
                lines.append(f"  filling buffer... {len(bvp)}/{FS*3}")
            else:
                w = np.asarray(bvp, dtype=np.float64)
                recent = w[-FS * 4:]
                a = assess(recent, dc, dc_hist)
                pi, drift, spikes, q = a["pi"], a["drift"], a["spikes"], a["quality"]

                # Everything below contact is meaningless without a finger: the
                # band-passed noise floor still has a "quality" and a periodicity,
                # and reporting those reads as confident nonsense. Gate them.
                on_skin = a["on_skin"]
                if on_skin:
                    hr = estimate_heart_rate(np.asarray(w[-FS * 12:], dtype=np.float32),
                                             fs=FS)
                    if hr:
                        hr_hist.append(hr)
                else:
                    hr_hist.clear()

                # ---- contact ----
                verdict, fix = a["contact"], a["contact_fix"]
                lines.append(f"  contact   {bar(min(dc / DC_GOOD_HI, 1.0))}  "
                             f"IR {dc:>9,.0f}   {verdict:<10} {fix}")

                if not on_skin:
                    lines.append("  pulse     ..........  PI        —     "
                                 "  --         (no finger on the sensor)")
                    lines.append("  still     ..........  drift     —     "
                                 "  --")
                    lines.append("  spikes    ..........    —  pts    "
                                 "  --")
                    lines.append("  quality   ..........          —     "
                                 "  --")
                    lines.append("")
                    lines.append("  HR —                    place a fingertip to begin")
                    if painted:
                        sys.stdout.write(f"\033[{painted}A")
                    for ln in lines:
                        sys.stdout.write("\033[2K" + ln + "\n")
                    sys.stdout.flush()
                    painted = len(lines)
                    if args.seconds and (time.monotonic() - t0) > args.seconds:
                        break
                    continue

                # ---- pulse strength ----
                if pi < PI_WEAK:
                    pv, pfix = "WEAK", "press LESS — squeezing cuts off blood flow"
                elif pi < PI_GOOD:
                    pv, pfix = "FAIR", "ease pressure slightly; warm hands help"
                else:
                    pv, pfix = "GOOD", ""
                lines.append(f"  pulse     {bar(pi / 3.0)}  "
                             f"PI {pi:>8.2f}%   {pv:<10} {pfix}")

                # ---- stillness ----
                if drift > DRIFT_BAD:
                    dv, dfix = "MOVING", "rest your wrist on the table and hold still"
                elif drift > DRIFT_OK:
                    dv, dfix = "DRIFTING", "settle the finger, don't roll it"
                else:
                    dv, dfix = "STEADY", ""
                lines.append(f"  still     {bar(1.0 - drift / DRIFT_BAD)}  "
                             f"drift {drift*100:>5.1f}%   {dv:<10} {dfix}")

                # ---- artifacts + quality ----
                sv = "CLEAN" if spikes == 0 else ("SOME" if spikes < 5 else "MANY")
                lines.append(f"  spikes    {bar(1.0 - min(spikes, 10) / 10)}  "
                             f"{spikes:>3} pts      {sv:<10} "
                             f"{'' if spikes < 5 else 'artifacts — usually pressure changes'}")
                qv = "GOOD" if q >= 0.7 else ("FAIR" if q >= 0.45 else "POOR")
                lines.append(f"  quality   {bar(q)}  {q:>10.2f}   {qv}")

                hr_txt = f"{np.mean(hr_hist):.0f} bpm" if hr_hist else "—"
                stab = (f"+/-{np.std(hr_hist):.0f}" if len(hr_hist) > 2 else "")
                ready = a["ready"]
                lines.append("")
                lines.append(f"  HR {hr_txt} {stab}        "
                             + ("*** GOOD TO RECORD ***" if ready
                                else "not ready: " + "; ".join(a["blocking"][:2])))

            if painted:
                sys.stdout.write(f"\033[{painted}A")
            for ln in lines:
                sys.stdout.write("\033[2K" + ln + "\n")
            sys.stdout.flush()
            painted = len(lines)

            if args.seconds and (time.monotonic() - t0) > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        print("\n  stopped.\n")


if __name__ == "__main__":
    main()
