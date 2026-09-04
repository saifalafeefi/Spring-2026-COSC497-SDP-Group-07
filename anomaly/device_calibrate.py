"""per-user calibration on the REAL sensor — re-derives the score thresholds on device data.

`anomaly/saved/scorer.npz` was calibrated on WESAD **wrist** BVP. this rig is a
**fingertip** MAX30102 — different sensor, different site, different waveform
morphology — so that threshold says nothing about this hardware, and the live
flag is close to arbitrary until it is re-derived here.

this is `calibrate.py`'s method (proved on WESAD: PR-AUC 0.750 → 0.867, recall
0.460 → 0.689 when the detector sees the user's own calm) run on our own device,
which is what O6 actually asks for: not "the method works on public data" but
"here is the number on our rig".

    python3 -m anomaly.device_calibrate                 # 5 minutes of calm
    python3 -m anomaly.device_calibrate --minutes 8
    python3 -m anomaly.device_calibrate --dry-run       # measure, write nothing
    python3 -m anomaly.device_calibrate --simulate --minutes 2 --min-windows 5
                                                        # no board, no writes

it records CALM only — sit still, breathe normally, don't talk. every 60 s window
is gated on the same grip checks `device_check.py` prints: a window is scored only
if the finger stayed on the sensor and the signal stayed clean for its whole
length, so one fidget can't end up defining "normal".

writes `anomaly/saved/scorer_device.npz`. `scorer.npz` (WESAD) is left untouched,
so the replay demo keeps working; `serve.py --source device` picks the device
scorer up automatically once it exists.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from .device_check import assess, DC_LOW, GATE_PI, GATE_DRIFT, GATE_QUALITY
from .device_source import DeviceSource
from .infer import SAVE_DIR, DEVICE_SCORER, WESAD_SCORER
from .wesad import FS

TICK = 0.5                 # how often the grip is re-assessed, seconds
SETTLE_TICKS = 6           # consecutive clean ticks before the clock starts


def _fmt(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


class _Recorder:
    """collects clean 60 s calm windows from the live sensor.

    quality is judged per ~0.5 s tick, and a window is accepted when the ticks
    covering its full 60 s were good enough. "good enough" is a TOLERANCE, not
    perfection: an earlier version required all 120 ticks to pass, which real PPG
    never does — one swallow, breath or micro-shift resets the run, and a five
    minute session yields nothing. a few degraded half-seconds inside a minute of
    otherwise clean pulse do not corrupt a baseline; losing contact does, so that
    stays a hard reset that clears the buffer outright.
    """

    def __init__(self, win_len: int, step_sec: float, min_pi: float = GATE_PI,
                 max_drift: float = GATE_DRIFT, min_quality: float = GATE_QUALITY,
                 tolerance: float = 0.20, adaptive: bool = True, use_pi: bool = True):
        # ADAPTIVE GATE. Absolute thresholds do not survive contact with real
        # hardware: this rig's steady perfusion is 0.2-0.4%, below the "healthy
        # 0.5-5%" range the fixed defaults assume, and a bar learned once from the
        # first seconds locked onto the post-contact transient (0.47%) that the
        # rest of the session could never meet -- a four minute recording that
        # collected nothing. So judge each tick against THIS session's own running
        # median instead: reject what is much worse than the person's typical
        # signal, not what fails someone else's absolute number. Floors stop it
        # following a signal all the way down to noise, and contact stays a hard
        # requirement, which is what actually protects the baseline.
        self.adaptive = adaptive
        self.use_pi = use_pi
        self.pi_hist: deque = deque(maxlen=240)     # ~2 min of ticks
        self.q_hist: deque = deque(maxlen=240)
        self.gate = ({"min_pi": 0.0, "max_drift": 0.15, "min_quality": 0.0}
                     if adaptive else
                     {"min_pi": min_pi, "max_drift": max_drift,
                      "min_quality": min_quality})
        self.tolerance = tolerance
        self.win_len = win_len
        self.step = step_sec
        self.bvp: deque = deque(maxlen=win_len)
        self.dc_hist: deque = deque(maxlen=40)
        # one verdict per tick, spanning exactly one inference window
        self.ticks: deque = deque(maxlen=max(1, int(round(win_len / FS / TICK))))
        self.total = 0            # samples seen since the last discontinuity
        self.next_at = win_len    # sample count of the next scoring opportunity
        self.run = 0              # current consecutive-clean ticks
        self.best_run = 0         # longest clean stretch, ticks
        self.seen = 0             # ticks assessed
        self.good = 0             # ticks that passed the gate
        self.was_on_skin = False
        self.windows: list[np.ndarray] = []
        self.rejects = {"contact": 0, "pulse": 0, "motion": 0, "quality": 0}
        self.last: dict = {}

    def _adapt(self, a) -> None:
        """retune the gate from this session's own running median."""
        self.pi_hist.append(a["pi"])
        self.q_hist.append(a["quality"])
        if len(self.pi_hist) < 20:
            return                       # not enough history to judge "typical" yet
        mpi = float(np.median(self.pi_hist))
        mq = float(np.median(self.q_hist))
        self.gate = {
            "min_pi": max(0.05, 0.40 * mpi) if self.use_pi else 0.0,
            "min_quality": max(0.20, mq - 0.25),
            "max_drift": 0.15,
        }

    def gate_desc(self) -> str:
        return (f"PI>={self.gate['min_pi']:.2f}%  "
                f"q>={self.gate['min_quality']:.2f}")

    def clean_frac(self) -> float:
        """lifetime pass rate -- for the end-of-session report only."""
        return (self.good / self.seen) if self.seen else 0.0

    def recent_frac(self) -> float | None:
        """pass rate over the last window's worth of ticks.

        this is what a live "signal is weak" indicator must use. the lifetime
        ratio latches: a rough patch early on drags it under the bar and it then
        takes minutes to climb back, so the warning stays on screen long after
        the signal recovered -- which reads as the tool being stuck.
        """
        if not self.ticks:
            return None
        return sum(self.ticks) / len(self.ticks)

    def _hard_reset(self):
        """contact was lost — everything buffered is noise, drop it."""
        self.bvp.clear()
        self.dc_hist.clear()
        self.ticks.clear()
        self.total = 0
        self.next_at = self.win_len
        self.run = 0

    def feed(self, samples, ir_dc: float):
        """add newly arrived samples; return a window when one comes up clean."""
        on_skin = ir_dc >= DC_LOW
        if on_skin != self.was_on_skin:      # finger landed or left — start fresh
            self._hard_reset()
        self.was_on_skin = on_skin
        self.dc_hist.append(ir_dc)

        for v in samples:
            self.bvp.append(v)
            self.total += 1

        if len(self.bvp) < FS * 4:
            return None

        a = assess(list(self.bvp)[-FS * 4:], ir_dc, self.dc_hist, **self.gate)
        self.last = a
        if self.adaptive and a["on_skin"]:
            self._adapt(a)
        self.seen += 1
        if a["ready"]:
            self.good += 1
            self.run += 1
            self.best_run = max(self.best_run, self.run)
        else:
            self.run = 0
            self.rejects[a["block_kind"] or "quality"] += 1
            if not a["on_skin"]:
                self._hard_reset()
                return None
        self.ticks.append(bool(a["ready"]))

        if len(self.bvp) < self.win_len or self.total < self.next_at:
            return None
        if len(self.ticks) < self.ticks.maxlen:
            return None
        if (sum(self.ticks) / len(self.ticks)) < (1.0 - self.tolerance):
            return None

        self.next_at = self.total + int(self.step * FS)
        w = np.fromiter(self.bvp, dtype=np.float32, count=self.win_len)
        self.windows.append(w)
        return w


class _FakeSource:
    """synthetic 64 Hz pulse, no board — exercises the gate + scoring path end to end.

    `--simulate` uses this so the tool can be proved before a real five-minute
    recording is spent discovering a bug in it. it is NOT calibration data, so
    simulate refuses to overwrite the real scorer.
    """

    def __init__(self, bpm: float = 68.0, pi_pct: float = 2.0, dc: float = 120_000.0):
        self.f = bpm / 60.0
        self.ac = dc * pi_pct / 100.0
        self.ir_dc = dc
        self.rng = np.random.default_rng(0)
        self._n = 0
        self._t0 = time.monotonic()

    def start(self):
        self._t0 = time.monotonic()
        return self

    def available(self) -> int:
        return max(0, int((time.monotonic() - self._t0) * FS) - self._n)

    def stream(self):
        while True:
            t = self._n / FS
            self._n += 1
            v = 0.5 * self.ac * (np.sin(2 * np.pi * self.f * t)
                                 + 0.35 * np.sin(4 * np.pi * self.f * t))
            yield float(v + self.rng.normal(0.0, 0.01 * self.ac)), 0

    def status(self) -> dict:
        return {"connected": True, "port": "simulated", "contact": True,
                "ir_dc": self.ir_dc, "bpm": None, "spo2": None, "error": None}

    def close(self):
        pass


def _drain(src, stream) -> list:
    vals = []
    for _ in range(src.available()):
        v = next(stream)[0]
        if v is None:
            break
        vals.append(v)
    return vals


def _wait_for_grip(src, rec, stream, simulated: bool = False):
    """block until the grip has been clean for a few ticks running. Ctrl-C aborts."""
    print()
    if simulated:
        print("  synthetic pulse — no board, no finger. this only checks that the")
        print("  gate and the model run end to end; the numbers below mean nothing.")
    else:
        print("  rest a fingertip on the sensor and hold still.")
        print("  the recording clock starts once the signal is clean.")
    print()
    good = 0
    while good < SETTLE_TICKS:
        time.sleep(TICK)
        rec.feed(_drain(src, stream), src.status()["ir_dc"])
        a = rec.last
        st = src.status()
        if not st["connected"]:
            msg = f"waiting for the board ({st['error'] or 'no data'})"
            good = 0
        elif not a:
            msg = "filling buffer…"
            good = 0
        elif a["ready"]:
            good += 1
            msg = f"clean — starting in {SETTLE_TICKS - good}…"
        else:
            good = 0
            msg = "; ".join(a["blocking"][:2]) or a["contact"]
        sys.stdout.write(f"\r\033[2K  {msg:<64}")
        sys.stdout.flush()
    sys.stdout.write("\r\033[2K  recording.\n\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=5.0,
                    help="minutes of clean calm to record (default: 5)")
    ap.add_argument("--port", default=os.environ.get("DEVICE_PORT"),
                    help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--invert", action="store_true",
                    help="flip the waveform if the pulse reads upside-down")
    ap.add_argument("--step", type=float, default=5.0,
                    help="seconds between scored windows (default: 5)")
    ap.add_argument("--min-windows", type=int, default=20,
                    help="refuse to write a scorer built on fewer than this many windows")
    ap.add_argument("--out", default=None,
                    help=f"output path (default: saved/{DEVICE_SCORER})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the numbers, write nothing")
    ap.add_argument("--no-save-windows", action="store_true",
                    help="don't keep the raw calm windows next to the scorer")
    ap.add_argument("--tolerance", type=float, default=0.20,
                    help="fraction of a window's ticks allowed to fail (default: 0.20)")
    ap.add_argument("--min-pi", type=float, default=GATE_PI,
                    help=f"minimum perfusion index %% to accept (default: {GATE_PI})")
    ap.add_argument("--max-drift", type=float, default=GATE_DRIFT,
                    help=f"maximum DC wander to accept (default: {GATE_DRIFT})")
    ap.add_argument("--min-quality", type=float, default=GATE_QUALITY,
                    help=f"minimum signal-quality index (default: {GATE_QUALITY})")
    ap.add_argument("--fixed-gate", action="store_true",
                    help="use the built-in thresholds instead of learning them from "
                         "your own signal (default: learn them)")
    ap.add_argument("--simulate", action="store_true",
                    help="synthetic pulse instead of the board — checks the tool works; "
                         "never writes the real scorer")
    args = ap.parse_args()

    out = args.out or os.path.join(SAVE_DIR, DEVICE_SCORER)
    if args.simulate and args.out is None:
        args.dry_run = True        # a scorer built on a sine wave would poison the flag

    print("\n  DEVICE CALIBRATION — recording your own calm to re-derive the threshold")
    if args.simulate:
        print("  --simulate: synthetic pulse, no board. nothing will be written.")
    print("  loading model… (~45 s: TensorFlow + the 4 MB int8 model)", flush=True)
    from .infer import LiveAnomalyDetector
    det = LiveAnomalyDetector()          # WESAD scorer loaded, but only .score() is used
    win_len = det.win_len
    print(f"  model ready — {win_len} samples/window ({win_len / FS:.0f} s @ {FS} Hz)")

    src = (_FakeSource() if args.simulate else
           DeviceSource(port=args.port, baud=args.baud, invert=args.invert)).start()
    stream = src.stream()
    rec = _Recorder(win_len, args.step, min_pi=args.min_pi,
                    max_drift=args.max_drift, min_quality=args.min_quality,
                    tolerance=args.tolerance, adaptive=not args.fixed_gate)
    if args.fixed_gate:
        print(f"  gate: PI >= {args.min_pi}%  drift <= {args.max_drift*100:.0f}%  "
              f"quality >= {args.min_quality}")
    else:
        print("  gate: adapts to your own signal as it records "
              "(--fixed-gate for the built-in thresholds)")

    scores: list = []
    interrupted = False
    try:
        _wait_for_grip(src, rec, stream, simulated=args.simulate)
        target = args.minutes * 60.0
        t0 = time.monotonic()
        painted = False
        while True:
            time.sleep(TICK)
            w = rec.feed(_drain(src, stream), src.status()["ir_dc"])
            if w is not None:
                scores.append(det.score(w))
            el = time.monotonic() - t0
            a = rec.last or {}
            state = ("clean" if a.get("ready")
                     else "; ".join(a.get("blocking") or [])[:46] or "…")
            if painted:
                sys.stdout.write("\033[1A")
            sys.stdout.write(
                f"\033[2K  {_fmt(el)} / {_fmt(target)}   windows {len(scores):>3}   "
                f"usable {rec.clean_frac()*100:>3.0f}%   "
                f"PI {a.get('pi', 0.0):>5.2f}%   q {a.get('quality', 0.0):.2f}   "
                f"{state}\n")
            sys.stdout.flush()
            painted = True
            if el >= target:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n  stopped early — using what was recorded.")
    finally:
        src.close()

    n = len(scores)
    print(f"\n  recorded {n} clean windows  "
          f"(rejected ticks: {rec.rejects['contact']} contact / "
          f"{rec.rejects['pulse']} weak-pulse / {rec.rejects['motion']} motion / "
          f"{rec.rejects['quality']} noisy)")
    if n == 0:
        # name the gate that actually blocked it, and the flag that relaxes it,
        # so a failed session says what to change instead of 'try again'.
        worst = max(rec.rejects, key=lambda k: rec.rejects[k]) if rec.seen else ""
        flag = {"pulse": f"--min-pi (now {args.min_pi})",
                "motion": f"--max-drift (now {args.max_drift})",
                "quality": f"--min-quality (now {args.min_quality})",
                "contact": "a steadier finger, not a flag"}.get(worst, "")
        print(f"  usable ticks: {rec.clean_frac()*100:.0f}%   longest clean stretch: {rec.best_run * TICK:.0f} s")
        print()
        if interrupted and rec.best_run * TICK >= 10:
            print(f"  the signal was fine — you stopped before the first window was")
            print(f"  complete. let it run the full {args.minutes:g} min.")
        elif args.simulate:
            print("  the synthetic source never came up clean — that is a bug in the tool.")
        elif worst:
            print(f"  most ticks failed on {worst} -- the fix is {flag}.")
            if worst != "contact":
                print(f"  --tolerance (now {args.tolerance:g}) also allows more bad ticks per window.")
            print("  `python3 -m anomaly.device_check` shows this live, for free.")
        else:
            print("  nothing clean was captured — run `python3 -m anomaly.device_check` first.")
        print()
        return 1

    s = np.asarray(scores, dtype=np.float64)
    threshold = float(np.quantile(s, 0.90))       # 90% specificity on your own calm
    ref_lo = float(np.median(s))
    ref_hi = float(np.quantile(s, 0.99))
    print(f"\n  your calm score distribution:  median {ref_lo:.5f}   "
          f"p90 {threshold:.5f}   p99 {ref_hi:.5f}   "
          f"(min {s.min():.5f}  max {s.max():.5f})")

    # the headline: what the WESAD threshold was actually doing to this hardware.
    old = np.load(os.path.join(SAVE_DIR, WESAD_SCORER))
    old_thr = float(old["threshold"])
    fired = float(np.mean(s >= old_thr))
    if args.simulate:
        print()
        print("  (simulate: skipping the transfer-delta report — a sine wave is not",
              "calm PPG, and that number belongs in no writeup.)")
    else:
        print()
        print(f"  WESAD threshold {old_thr:.5f}   vs   device threshold {threshold:.5f}")
        print(f"  under the WESAD threshold {fired * 100:.0f}% of your CALM windows "
              f"flagged (by construction it should be ~10%).")
        print("  that gap is the zero-shot vs device-calibrated delta, measured on our own rig.")

    if n < args.min_windows:
        print(f"  longest clean stretch: {rec.best_run / FS:.0f} s")
        print(f"\n  only {n} windows (< --min-windows {args.min_windows}) — too few to set "
              "a threshold on. nothing written; record longer, or steadier.\n")
        return 1
    if args.dry_run:
        print("\n  --dry-run: nothing written.\n")
        return 0

    np.savez(out, threshold=threshold, win_len=int(win_len),
             ref_lo=ref_lo, ref_hi=ref_hi,
             source="device", n_windows=n, fs=int(FS),
             created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"\n  wrote {out}")
    if not args.no_save_windows:
        wpath = os.path.join(SAVE_DIR, "device_calm.npz")
        np.savez_compressed(wpath, windows=np.stack(rec.windows), fs=int(FS),
                            created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        print(f"  wrote {wpath}  ({len(rec.windows)} raw windows — lets you re-derive "
              "a threshold without recording again)")
    print("\n  the dashboard picks this up on its own:")
    print("      python3 -m anomaly.serve --source device\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
