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
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from collections import deque

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from . import quality
from .db import DEFAULT_PATH as DB_PATH, Db, import_legacy_scorers
from .infer import SAVE_DIR
from .wesad import FS

WIN = 60 * FS

# The buffer holds more than the model needs, so a disturbed second can be
# dropped and back-filled from clean history instead of halting the detector for
# a full window. 150 s leaves 90 s of slack -- enough to ride out a burst of
# tapping without ever stopping, and 30 KB per board.
HIST = 150 * FS

HTTP_TIMEOUT = 1.0

# The roster page is read from disk on every request, but these routes are fixed
# when the process starts. Edit both and an already-running master serves the NEW
# page against its OLD routes -- the page calls an endpoint that does not exist
# yet and the browser reports a bare "failed". Bump this whenever a route is
# added or changed; the page checks it and says plainly that a restart is due.
API_VERSION = 12

# how long a board can go without a finger on it before its session is over.
# generous on purpose: a session is a stretch of monitoring, and closing one
# every time somebody scratches their nose would shred the history into confetti.
SESSION_IDLE_S = 120.0

# --- self-referencing scale -------------------------------------------------
# A calibration used to store WHERE this person's calm sat. That number moves
# the moment the sensor is re-placed: measured on 27 subjects recorded at two
# placements in the same session, the shift is 1.60 robust sigmas at the median
# and 10.60 at p90 -- wider than the whole band, which is why a baseline from
# yesterday reads today's calm as 100% stressed.
#
# What does NOT move is the shape: how far above their own calm this person's
# stress sits. So the centre and the spread are re-estimated continuously from
# the live signal, and calibration stores only k_sigma. Measured cost of that
# swap: placement error 1.60 -> 0.34 sigmas (p90 10.60 -> 0.78), stress effect
# d 2.81 -> 1.55 (about AUC 0.86, still better than the cross-subject 0.759).
NORM_N = 180          # scores the live centre/spread are taken over (~3 min)
NORM_MIN = 20         # below this there is not enough to normalise against
NORM_Q = 20           # percentile of recent scores taken as "calm right now"
Z_FULL = 6.0          # sigmas that read as 100% on the level bar
K_DEFAULT = 5.0       # sigmas to flag at, before anyone has calibrated.
                      # measured: the live z of calm sits at p50 ~1.1 and
                      # p90 ~5.0, so 3.0 flagged a fifth of a calm session.
# A spread of nearly zero -- a very still stretch, or a stuck signal -- would
# turn any tiny deviation into infinite sigmas. Floor it at a fraction of the
# centre so the scale cannot collapse.
SPREAD_FLOOR = 0.02

# --- the knobs that decide how twitchy the flag is --------------------------
# EMA_ALPHA   how much of each new score is taken. lower = steadier number,
#             slower to react. 0.35 reached 63% of a step in ~2.3 s; 0.20 takes
#             ~6 s. this is the one to turn if the level looks jumpy.
# FLAG_ON_S   seconds the score must STAY above the threshold before it flags.
#             stress is sustained; a two-second excursion is a finger moving.
# FLAG_OFF_S  seconds below before the flag clears, so it does not chatter.
# Z_FULL      sigmas that read as 100% on the bar. bigger = a calmer-looking
#             gauge for the same signal; it changes the display, not the flag.
EMA_ALPHA = 0.20
FLAG_ON_S = 5
FLAG_OFF_S = 10


# --- what the roster is allowed to turn, live ------------------------------
# These are read out of the module globals on every scoring tick, so writing a
# new value takes effect on the next second -- no restart. They are persisted
# into the store's meta table so a restart keeps them.
#
# `mod` says where the name lives: None for this module, "quality" for the
# signal-quality stage.
TUNING = [
    {"key": "EMA_ALPHA", "mod": None, "label": "Smoothing", "type": "float",
     "min": 0.05, "max": 0.60, "step": 0.01,
     "help": "How much of each new score is taken into the level. "
             "LOWER = steadier number, slower to react (0.10 takes ~10 s to "
             "catch up). HIGHER = twitchy but immediate. Turn this first if "
             "the level jumps around."},
    {"key": "FLAG_ON_S", "mod": None, "label": "Hold before flagging", "type": "int",
     "min": 1, "max": 30, "step": 1, "unit": "s",
     "help": "Seconds the score must STAY above the line before it counts. "
             "HIGHER = ignores brief excursions, fewer false flags, slower to "
             "catch a real one. LOWER = reacts sooner, flags more twitches."},
    {"key": "FLAG_OFF_S", "mod": None, "label": "Hold before clearing", "type": "int",
     "min": 1, "max": 60, "step": 1, "unit": "s",
     "help": "Seconds below the line before the flag drops. HIGHER = one long "
             "episode instead of several short ones. LOWER = the flag chatters "
             "on and off around the threshold."},
    {"key": "K_DEFAULT", "mod": None, "label": "Flag threshold", "type": "float",
     "min": 1.0, "max": 8.0, "step": 0.1, "unit": "sigma",
     "help": "Where the sensitivity slider is PARKED after a calibration, in "
             "sigmas, for anyone not yet calibrated. It no longer sets the "
             "threshold -- the slider does that directly now -- so this only "
             "decides the starting position. Move the slider to change the line."},
    {"key": "Z_FULL", "mod": None, "label": "Bar full scale", "type": "float",
     "min": 2.0, "max": 15.0, "step": 0.5, "unit": "sigma",
     "help": "How many sigmas read as 100% on the level bar. DISPLAY ONLY -- "
             "it does not move the flag. HIGHER = a calmer-looking gauge for "
             "the same signal. LOWER = the bar swings more."},
    {"key": "NORM_N", "mod": None, "label": "Calm memory", "type": "int",
     "min": 30, "max": 900, "step": 10, "unit": "s",
     "help": "Seconds of recent signal the live calm centre and spread are "
             "measured over. HIGHER = a steadier reference, slower to follow a "
             "re-placed sensor. LOWER = adapts fast, but a long stress episode "
             "can start looking normal."},
    {"key": "ENVELOPE_RATIO", "mod": "quality", "label": "Movement gate", "type": "float",
     "min": 1.2, "max": 6.0, "step": 0.1, "unit": "x",
     "help": "A second of signal louder than this multiple of your quiet "
             "amplitude is treated as movement and skipped. LOWER = stricter, "
             "throws away more (and may refuse to score). HIGHER = lets more "
             "movement reach the model, which is what makes taps flag."},
    {"key": "SESSION_IDLE_S", "mod": None, "label": "Session timeout", "type": "int",
     "min": 30, "max": 900, "step": 10, "unit": "s",
     "help": "How long without a finger before the session is closed and the "
             "history splits. HIGHER = brief breaks stay in one session. "
             "LOWER = tighter sessions, more of them."},
]


def tuning_state() -> list:
    """current value of every knob, with its metadata."""
    out = []
    for t in TUNING:
        holder = quality if t["mod"] == "quality" else sys.modules[__name__]
        d = dict(t)
        d["value"] = getattr(holder, t["key"])
        out.append(d)
    return out


def tuning_apply(vals: dict, db=None) -> dict:
    """set knobs live, clamped to their declared range, and remember them."""
    changed = {}
    for t in TUNING:
        if t["key"] not in vals:
            continue
        try:
            v = float(vals[t["key"]])
        except (TypeError, ValueError):
            continue
        v = max(t["min"], min(t["max"], v))
        v = int(round(v)) if t["type"] == "int" else round(v, 4)
        holder = quality if t["mod"] == "quality" else sys.modules[__name__]
        setattr(holder, t["key"], v)
        changed[t["key"]] = v
        if db is not None:
            db.meta_set("tune:" + t["key"], repr(v))
    return changed


def tuning_restore(db):
    """re-apply whatever was set last time this master ran."""
    vals = {}
    for t in TUNING:
        raw = db.meta_get("tune:" + t["key"])
        if raw is not None:
            try:
                vals[t["key"]] = float(raw)
            except ValueError:
                pass
    return tuning_apply(vals, db=None) if vals else {}

# where the waveform behind each flag is kept, and whether to keep it at all.
# the master already holds the raw signal in RAM to score it, so this stores
# nothing new -- but it is the difference between a reviewable flag and a
# number nobody can check.
FLAG_DIR = os.path.join(os.path.dirname(DB_PATH), "flags")
RECORD_FLAGS = True

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


# The slider IS the threshold, read straight off the bar: 0 puts the line at
# 10%, 1.0 puts it at 90%. It used to be a curve around each person's calibrated
# operating point, which meant the same slider position meant a different line
# for every subject, the line moved when k_sigma changed under you, and it ran
# backwards -- pushing the slider up made the threshold go down.
THR_MIN, THR_MAX = 0.10, 0.90


def sens_to_level(sens: float) -> float:
    """slider position -> the flag threshold, absolutely, on the 0-1 bar."""
    return THR_MIN + (THR_MAX - THR_MIN) * min(1.0, max(0.0, float(sens)))


def level_to_sens(level: float) -> float:
    """the inverse: where to park the slider to sit at this threshold."""
    return min(1.0, max(0.0, (float(level) - THR_MIN) / (THR_MAX - THR_MIN)))


def push_flag(ip: str, flag: bool, level: float, thr_level: float,
              state: str = "ok", wait: int = 0, subject: str = "") -> bool:
    """the verdict, or the reason there is not one.

    the board used to hear from us only when we had a real score, so for the
    first 60 s -- and any time a window was unusable -- it sat there showing
    CALM, which is a verdict, and the wrong one. `s` says which of those it is
    and `w` how many seconds are left to wait.
    """
    q = urllib.parse.urlencode({"f": 1 if flag else 0,
                                "l": int(round(max(0.0, min(1.0, level)) * 100)),
                                "t": int(round(max(0.0, min(1.0, thr_level)) * 100)),
                                "s": state, "w": int(max(0, wait)),
                                # the board only knows its MAC-derived id, so its
                                # own dashboard called the wearer pulse-000000
                                # whatever they had been renamed to
                                "n": subject[:27]})
    try:
        with urllib.request.urlopen(f"http://{ip}/flag?{q}", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


def push_calib(ip: str, phase: str, windows: int, target: int, can_commit: bool) -> bool:
    """tell the board how the calibration is going, for it to pass on.

    The board cannot calibrate anything -- it cannot score its own windows --
    but its dashboard is where somebody stands when they press the button, so
    the progress has to reach it.
    """
    q = urllib.parse.urlencode({"p": phase, "w": int(windows), "t": max(1, int(target)),
                                "c": 1 if can_commit else 0})
    try:
        with urllib.request.urlopen(f"http://{ip}/calib?{q}", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


# ----------------------------------------------------------------- one device

class Device:
    """one board: its websocket, whoever is wearing it, and their baseline."""

    def __init__(self, dev_id: str, ip: str, det, db):
        self.id = dev_id
        self.ip = ip
        self.det = det                    # LiveAnomalyDetector, thresholds swapped per subject
        self.db = db
        self.buf: deque = deque(maxlen=HIST)   # raw history, longer than a window
        self.ema = None
        self.level = None
        self.flag = False
        self.score = 0.0
        self.bpm = None
        self.spo2 = None
        self.contact = False
        self.last_seen = 0.0
        self.last_contact = 0.0           # monotonic, for the idle-session timer
        self.connected = False
        self.pushes = 0
        self.fails = 0
        self.calib = None                 # a CalibSession while one is running
        self.ws = None                    # live socket, for pushing a new default back
        self.sens = 0.5                   # slider position, mirrored from the board
        self.thr_level = 0.42             # where that puts the threshold, 0-1
        self.lost_since = None            # when contact was last lost
        self.thresholds = None            # (threshold, lo, hi) from the subject's baseline
        self.subject = None               # the assigned subject row, or None
        self.session_id = None            # the open monitor session, or None
        self.event_id = None              # the flag episode in progress, or None
        self.name = dev_id                # human label, falls back to the id
        self.quality = None               # 0-1 for the last window assessed
        self.quality_note = ""            # why it was refused, if it was
        self.state = "none"               # ok | warm | hold | none
        self.recent: deque = deque(maxlen=NORM_N)   # live scores, for the scale
        self.k_sigma = K_DEFAULT          # flag this many sigmas above live calm
        self.centre = None                # this wear's calm, re-estimated live
        self.spread = None
        self.above = 0                    # consecutive seconds over the line
        self.below = 0                    # and under it, for the hysteresis
        self.flag_hist: deque = deque(maxlen=600)   # was it flagging? last 10 min
        self.refresh()
        self.refresh_threshold()

    # ---- who is wearing this, and what is normal for them ----

    def refresh(self):
        """re-read this board's label, its wearer and their baseline. cheap, and
        called whenever any of them could have changed rather than cached and
        hoped -- the roster polls once a second and must not show a stale name."""
        row = self.db.device(self.id)
        self.name = (row["name"] if row else None) or self.id
        self.subject = self.db.assigned_subject(self.id)
        b = self.db.active_baseline(self.subject["id"]) if self.subject else None
        self.thresholds = (b["threshold"], b["ref_lo"], b["ref_hi"]) if b else None
        self.baseline = b
        # k_sigma is the whole calibration now; the old ref_lo/ref_hi are kept
        # for provenance but no longer decide anything.
        # falls back rather than sticking: moving a board from a calibrated
        # subject to an uncalibrated one used to leave the previous person's
        # k_sigma in place, quietly judging the new wearer by someone else's calm.
        self.k_sigma = (float(b["k_sigma"]) if b is not None and b["k_sigma"]
                        else K_DEFAULT)

    @property
    def scoring(self) -> bool:
        """a verdict needs somebody to attribute it to. that is the whole gate.

        It used to need a baseline as well, and that reasoning came from the old
        design where the flag sat at an absolute score this person had recorded:
        without their calm you really were judging them against someone else's
        body. The scale is self-referencing now -- centre and spread are
        re-estimated live from this wearer's own recent scores -- so the only
        thing a calibration still supplies is k_sigma, and K_DEFAULT stands in
        until it exists.

        That makes the uncalibrated case the ZERO-SHOT condition: the model,
        trained on other people, judging a new wearer with nothing personal
        behind it. It is worth being able to watch, since the zero-shot vs
        calibrated delta is exactly what O6 asks us to report. It is marked
        everywhere it is shown so it cannot be mistaken for a calibrated verdict.
        """
        return self.subject is not None

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    async def send_sens(self, sens: float):
        """move the board's slider, and ours with it."""
        self.sens = round(float(np.clip(sens, 0.0, 1.0)), 3)   # 3 dp == the board's echo
        self.thr_level = sens_to_level(self.sens)
        if self.ws is None:
            return False
        try:
            await self.ws.send(json.dumps({"cmd": "set_sensitivity", "value": self.sens}))
            return True
        except Exception:
            return False

    # ---- sessions open and close on their own ----

    def ensure_session(self):
        """contact + a subject = a session, with nobody pressing anything.

        the operator has no start button on purpose: detection is meant to be
        continuous, so a session is a consequence of someone wearing the board,
        not a mode somebody remembered to enter.
        """
        if self.session_id is not None or not self.scoring:
            return
        self.session_id = self.db.open_session(
            self.subject["id"], self.id, "monitor",
            model_id=getattr(self.det, "model_id", None), sens=self.sens)

    def end_session(self):
        if self.event_id is not None:
            self.db.close_event(self.event_id)
            self.event_id = None
        if self.session_id is not None:
            self.db.close_session(self.session_id)
            self.session_id = None

    def commit_calibration(self, r: dict) -> float:
        """turn a finished calm collection into this SUBJECT's baseline, and say
        where the slider has to sit to reproduce it.

        the collection gets its own session row spanning the windows it took, so
        a threshold can always be traced back to the calm it came from. the
        monitor session ends here and the next frame with contact opens a fresh
        one -- readings scored against the old baseline do not belong in the same
        session as readings scored against the new one.
        """
        started = time.time() - (time.monotonic() - self.calib.t0)
        self.end_session()
        sess = self.db.record_session(
            self.subject["id"], self.id, "calibration", started, time.time(),
            model_id=getattr(self.det, "model_id", None),
            notes=f"{r['n']} calm windows")
        self.db.save_baseline(self.subject["id"], r["threshold"], r["ref_lo"], r["ref_hi"],
                              n_windows=r["n"], fs=int(FS), win_len=int(WIN),
                              session_id=sess,
                              model_id=getattr(self.det, "model_id", None),
                              source="device", k_sigma=r["k_sigma"])
        self.recent.clear()
        self.refresh()
        # park the slider where this person's measured operating point sits, so
        # the control opens on the calibrated answer and can be moved from there.
        return level_to_sens(r["k_sigma"] / Z_FULL)

    def threshold_z(self) -> float:
        """where the flag sits, in sigmas above this wearer's live calm.

        Purely the slider now -- the bar position times the bar's full scale.
        No score involved: this used to be computed inside apply(), which only
        runs with a finger on the sensor AND a full window AND enough history,
        so the slider moved and the line stayed at its default until all three
        were true.

        The calibrated k_sigma no longer sets this. It sets where the slider
        STARTS after a calibration (see commit_calibration), which is the honest
        division: the measurement suggests an operating point, the operator
        chooses one, and the control reads the same as the chart.
        """
        return sens_to_level(self.sens) * Z_FULL

    def refresh_threshold(self):
        """recompute the drawn line. cheap, and safe to call every tick."""
        self.thr_level = float(np.clip(sens_to_level(self.sens), 0.0, 1.0))

    def score_window(self):
        """assess the window, then score what survives.

        returns (score, quality, reason). a refused window scores None: holding
        the previous verdict is honest, inventing one from a damaged window is
        not. on clean pulse this stage is a no-op -- verified bit-identical on
        28 of 29 WESAD windows -- so it costs nothing when nothing is wrong.
        """
        hist = np.fromiter(self.buf, dtype=np.float32, count=len(self.buf))
        r = quality.clean_window(hist, WIN, FS)
        if r["window"] is None:
            return None, r["quality"], r["reason"]
        note = ("" if not r["dropped"] else
                "skipped %d s of movement (%.1fx)" % (r["dropped"], r["env"]))
        return self.det.score(r["window"]), r["quality"], note

    def apply(self, raw, q=None):
        """turn a raw reconstruction error into a level and a flag for THIS person,
        and record it. this is the only place a reading or an event is written."""
        self.quality = q
        if raw is None:              # window refused; hold, do not guess
            return
        self.recent.append(float(raw))
        self.ema = raw if self.ema is None else \
            (1.0 - EMA_ALPHA) * self.ema + EMA_ALPHA * raw
        self.score = self.ema
        if not self.scoring:
            self.level, self.flag = None, False
            return
        # the scale comes from the LAST FEW MINUTES, not from the calibration day
        r = np.fromiter(self.recent, dtype=np.float64, count=len(self.recent))
        if len(r) < NORM_MIN:
            self.level, self.flag = None, False
            return
        self.centre = float(np.percentile(r, NORM_Q))
        mad = float(np.median(np.abs(r - np.median(r))) * 1.4826)
        self.spread = float(max(mad, SPREAD_FLOOR * abs(self.centre), 1e-6))
        z = (self.ema - self.centre) / self.spread

        # the slider scales the calibrated k rather than replacing it: mid-travel
        # is exactly this person's calibrated operating point. the line the UI
        # draws has to BE the line it flags at, so both come from threshold_z().
        k = self.threshold_z()
        self.refresh_threshold()
        self.level = float(np.clip(z / Z_FULL, 0.0, 1.0))
        was = self.flag

        # hysteresis: crossing the line is not the same as being past it. the
        # score has to STAY over for FLAG_ON_S before this counts as a flag, and
        # stay under for FLAG_OFF_S before it clears. without it every brief
        # excursion became an event, which is most of what the history filled up
        # with.
        if z >= k:
            self.above += 1
            self.below = 0
        else:
            self.below += 1
            self.above = 0
        if not self.flag and self.above >= FLAG_ON_S:
            self.flag = True
        elif self.flag and self.below >= FLAG_OFF_S:
            self.flag = False

        self.flag_hist.append(1 if self.flag else 0)
        if self.session_id is None:
            return
        self.db.add_reading(self.session_id, score=self.score, level=self.level,
                            flag=self.flag, bpm=self.bpm, spo2=self.spo2,
                            contact=self.contact, quality=self.quality)
        # an event is the EPISODE, not the tick: one row from the moment the flag
        # rises to the moment it falls, so a two-minute stress response is one
        # thing to review rather than 120.
        if self.flag and not was:
            self.event_id = self.db.open_event(self.session_id, level=self.level)
            self.save_flag_window()
        elif self.flag and self.event_id is not None:
            self.db.bump_event(self.event_id, self.level)
        elif was and not self.flag and self.event_id is not None:
            self.db.close_event(self.event_id)
            self.event_id = None

    def save_flag_window(self):
        """dump the 60 s that fired this flag.

        the raw waveform is already on the master -- it has to be, the model
        scores it here -- so this stores no more than is in RAM anyway. it is
        what makes a flag reviewable at all: without it "was that stress or did
        I knock the sensor?" is unanswerable after the fact.
        """
        if not RECORD_FLAGS or self.event_id is None:
            return
        try:
            os.makedirs(FLAG_DIR, exist_ok=True)
            path = os.path.join(FLAG_DIR, "event_%d.npz" % self.event_id)
            np.savez_compressed(
                path, bvp=np.fromiter(self.buf, dtype=np.float32,
                                      count=len(self.buf))[-WIN:],
                fs=FS, device=self.id, subject=self.subject["code"],
                level=self.level, score=self.score, quality=self.quality or 0.0,
                t=time.time())
            self.db.set_event_window(self.event_id, path)
        except Exception as e:
            print("  %s: could not save flag window: %s" % (self.id, e), flush=True)

    def subject_label(self) -> str:
        """what the board should call whoever is wearing it."""
        if self.subject is None:
            return ""
        name = (self.subject["display_name"] or "").strip()
        return name or self.subject["code"]

    def baseline_fit(self):
        """is the operating point sane, judged on what it is actually doing?

        This used to compare today's score against the calibration day's
        ref_lo. Under the self-referencing scale nothing reads ref_lo any more,
        so that warning fired on a system that was working fine.

        What is worth saying is how much of the time it flags. Calibration aims
        for 10%; a great deal more than that means k is too low for this person
        and the honest fix is to recalibrate, not to squint at the banner.
        """
        if not self.scoring or len(self.flag_hist) < 60:
            return None
        rate = float(np.mean(self.flag_hist))
        return {"flag_rate": round(rate, 3), "n": len(self.flag_hist),
                "stale": bool(rate > 0.30)}

    def status(self) -> dict:
        stale = time.monotonic() - self.last_seen if self.last_seen else None
        sub = self.subject
        b = self.baseline
        return {
            "id": self.id, "name": self.name, "ip": self.ip,
            "connected": self.connected and stale is not None and stale < 5,
            "contact": self.contact,
            "bpm": self.bpm, "spo2": self.spo2,
            "level": None if self.level is None else round(self.level, 3),
            "flag": self.flag,
            "score": round(self.score, 5),
            "quality": None if self.quality is None else round(self.quality, 3),
            "quality_note": self.quality_note,
            "k_sigma": round(self.k_sigma, 2),
            "centre": None if self.centre is None else round(self.centre, 5),
            "spread": None if self.spread is None else round(self.spread, 5),
            "norm_n": len(self.recent),
            "state": self.state,
            "baseline_fit": self.baseline_fit(),
            "scoring": self.scoring,
            "calibrated": self.calibrated,
            "k_sigma": round(self.k_sigma, 2),
            "subject": None if sub is None else
                       {"id": sub["id"], "code": sub["code"],
                        "name": sub["display_name"] or sub["code"]},
            "baseline": None if b is None else
                        {"threshold": round(b["threshold"], 5),
                         "n_windows": b["n_windows"], "created": b["created"]},
            "session_id": self.session_id,
            "sens": round(self.sens, 3),
            "thr_level": round(self.thr_level, 3),
            "buf": len(self.buf), "win": WIN,
            "pushes": self.pushes, "fails": self.fails,
            "last_seen": None if stale is None else round(stale, 1),
            "calib": self.calib.status() if self.calib else None,
        }


class CalibSession:
    """collect this user's calm and derive their thresholds. master-side, because
    the model lives here -- a board can stream but cannot score its own windows."""

    # Sampled once a SECOND, like the runtime, and for long enough to see the
    # same distribution. At one window every 5 s the calibration measured a
    # different quantity from the one the flag is decided on, and the k it
    # derived flagged 41% of a calm session instead of 10%.
    TARGET = 180                   # 3 minutes. 2 was not enough to see the tail:
                                   # p90 of z over 120 samples came out at 1.5
                                   # while the same person's full session was 5.0

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
        self.next_at = now + 1.0
        sc, _, _ = dev.score_window()
        if sc is not None:          # a baseline learned from knocks is not calm
            self.scores.append(sc)

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
        # k_sigma: the z this person's own calm actually reaches, measured by
        # replaying the exact runtime pipeline -- EMA, then the rolling centre
        # and spread -- over the calibration scores. Deriving it from the raw
        # window scores instead compared two different distributions: the EMA
        # shrinks the spread, so the live z ran ~4x larger than calibration
        # expected and the flag fired 41% of the time on calm.
        ema, hist, z = None, [], []
        for x in sc:
            ema = x if ema is None else (1.0 - EMA_ALPHA) * ema + EMA_ALPHA * x
            hist.append(ema)
            h = np.asarray(hist[-NORM_N:], dtype=np.float64)
            if len(h) < NORM_MIN:
                continue
            c = float(np.percentile(h, NORM_Q))
            sp = max(float(np.median(np.abs(h - np.median(h))) * 1.4826),
                     SPREAD_FLOOR * abs(c), 1e-6)
            z.append((ema - c) / sp)
        # p90 of that is the 90%-specificity operating point this person asked
        # for; clamped so a degenerate calibration cannot make it absurd.
        # Floored at 3.0 rather than trusting a short sample outright: a
        # calibration that happens to sit in a very flat stretch produces a tiny
        # p90 and would flag a third of the day. Measured over one real calm
        # session: k=1.33 flags 33%, k=3.0 flags 15%, k=4.0 flags 12%, k=5.0
        # flags 6%, and the honest p90 was 4.41 (10%). A three-minute sample
        # under-reads the tail, so the floor does the work.
        if len(z) < 100:
            k = K_DEFAULT
        else:
            k = max(float(np.percentile(z, 90)), 4.0)
        return {"threshold": float(np.quantile(sc, 0.90)),
                "ref_lo": lo, "ref_hi": lo + span, "n": len(sc),
                "k_sigma": float(min(k, 12.0))}


# ------------------------------------------------------------------ the fleet

class Fleet:
    def __init__(self, det, subnets: list, extra: list, db):
        self.det = det
        self.subnets = subnets
        self.extra = extra
        self.db = db
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
                self.db.seen_device(dev_id, ip)
                if dev_id in self.devices:
                    self.devices[dev_id].ip = ip      # DHCP may have moved it
                    continue
                dev = Device(dev_id, ip, self.det, self.db)
                self.devices[dev_id] = dev
                why = ("" if dev.scoring else
                       "   (no subject assigned)" if dev.subject is None else
                       f"   ({dev.subject['code']} has no baseline)")
                print(f"  found {dev_id} at {ip}{why}", flush=True)
                asyncio.create_task(self.pump(dev))
        finally:
            self.scanning = False
            self.last_scan = time.monotonic()

    async def housekeeping_loop(self, every: float = 30.0):
        """close sessions nobody is in any more.

        a session cannot end itself from inside the stream loop: the board going
        quiet -- unplugged, out of range, crashed -- is exactly the case where no
        more frames arrive to notice it with.
        """
        while True:
            await asyncio.sleep(every)
            now = time.monotonic()
            for dev in list(self.devices.values()):
                if dev.session_id is None:
                    continue
                idle = now - (dev.last_contact or 0.0)
                if idle > SESSION_IDLE_S:
                    dev.end_session()
                    print(f"  {dev.id}: session closed after {idle:.0f}s idle", flush=True)

    async def pump(self, dev: Device):
        """one task per device: read its stream, score it, push the verdict back."""
        import websockets
        while True:
            try:
                async with websockets.connect(f"ws://{dev.ip}/ws",
                                              open_timeout=8, max_queue=64) as ws:
                    dev.connected = True
                    dev.ws = ws
                    next_score = time.monotonic() + 1.0
                    async for raw in ws:
                        try:
                            m = json.loads(raw)
                        except Exception:
                            continue
                        # the board re-broadcasts calibration commands from its
                        # own dashboard; we are one of its websocket clients, so
                        # this is how that button reaches the model.
                        cmd = m.get("cmd") or ""
                        if cmd.startswith("calib_"):
                            err = await run_calib(dev, cmd[len("calib_"):])
                            if err:
                                print(f"  {dev.id}: calib {cmd} -> {err}", flush=True)
                            continue
                        if m.get("type") != "f":
                            continue
                        dev.last_seen = time.monotonic()
                        d = m.get("device") or {}
                        dev.contact = bool(d.get("contact"))
                        dev.bpm = m.get("bpm")
                        dev.spo2 = m.get("spo2")
                        if m.get("sens") is not None:
                            dev.sens = float(m["sens"])   # the user moved the slider
                        now_c = time.monotonic()
                        if dev.contact:
                            dev.last_contact = now_c
                            dev.ensure_session()          # nobody presses start
                        if dev.contact:
                            dev.lost_since = None
                            for v in (m.get("bvp") or []):
                                dev.buf.append(float(v))
                            if dev.calib:
                                before = len(dev.calib.scores)
                                dev.calib.offer(dev)
                                if len(dev.calib.scores) != before:
                                    st = dev.calib.status()
                                    await asyncio.get_running_loop().run_in_executor(
                                        NET, push_calib, dev.ip, "record",
                                        st["windows"], st["target"], st["can_commit"])
                        else:
                            # a momentary lift should not cost a whole minute of
                            # refilling. hold the buffer briefly; only a real
                            # removal discards it. no samples are appended either
                            # way, so noise still never enters a window.
                            if dev.lost_since is None:
                                dev.lost_since = now_c
                            if now_c - dev.lost_since > 3.0:
                                dev.buf.clear()
                                dev.level, dev.flag, dev.ema = None, False, None
                            # NOTE: no `continue`. this used to skip straight past
                            # the push below, so a board with no finger on it heard
                            # nothing at all from the master -- which meant it never
                            # learned who was wearing it and its own dashboard kept
                            # calling them by the MAC-derived id.

                        now = time.monotonic()
                        if now < next_score:
                            continue
                        next_score = now + 1.0

                        # The line the slider sets is knowable even with no
                        # finger on the sensor, so recompute it before deciding
                        # what to say -- otherwise it sits at its default until
                        # scoring starts and the slider looks dead.
                        dev.refresh_threshold()

                        # Say something EVERY second, even when the answer is
                        # "not yet". Silence read as calm on the board.
                        wait = 0
                        if not dev.contact:
                            # the board is the authority on whether a finger is
                            # there and says so on its own screen; we just keep
                            # the channel warm so the name and state stay fresh
                            state = "idle"
                            dev.quality_note = "no finger on the sensor"
                        elif not dev.scoring:
                            state = "none"
                            dev.quality_note = "no subject assigned"
                        elif len(dev.buf) < WIN:
                            state = "warm"
                            wait = int(np.ceil((WIN - len(dev.buf)) / float(FS)))
                            dev.quality_note = "warming up — %d s left" % wait
                        else:
                            sc, q, why = dev.score_window()
                            dev.quality_note = why
                            dev.apply(sc, q)
                            state = "ok" if sc is not None else "hold"
                            # "zero" reads as a verdict everywhere "ok" does,
                            # and carries the marker with it.
                            if state == "ok" and not dev.calibrated:
                                state = "zero"

                        dev.state = state
                        # off the loop: a slow board must not stall the others
                        ok = await asyncio.get_running_loop().run_in_executor(
                            NET, push_flag, dev.ip, dev.flag, dev.level or 0.0,
                            dev.thr_level, state, wait, dev.subject_label())
                        dev.pushes += ok
                        dev.fails += (not ok)
            except Exception:
                pass
            finally:
                dev.connected = False
                dev.ws = None
                await asyncio.sleep(3.0)

    async def rescan_loop(self, every: float):
        while True:
            await asyncio.sleep(every)
            await self.discover()


async def run_calib(dev, action: str):
    """start / cancel / commit a calibration. returns an error string or None.

    The roster's button and the board dashboard's button both land here, so the
    two cannot drift apart -- the dashboard's used to reach nothing at all.
    """
    if action == "start":
        # a baseline has to belong to somebody, or it is just a number.
        if dev.subject is None:
            return "assign a subject to this board first"
        dev.calib = CalibSession()
        await asyncio.get_running_loop().run_in_executor(
            NET, push_calib, dev.ip, "record", 0, CalibSession.TARGET, False)
        print(f"  {dev.id}: calibrating {dev.subject['code']}", flush=True)
        return None

    if action == "cancel":
        dev.calib = None
        await asyncio.get_running_loop().run_in_executor(
            NET, push_calib, dev.ip, "cancelled", 0, CalibSession.TARGET, False)
        return None

    if action == "commit":
        if not dev.calib or len(dev.calib.scores) < CalibSession.TARGET:
            return "not enough clean windows yet"
        if dev.subject is None:
            return "no subject on this board"
        r = dev.calib.result()
        n = r["n"]
        sens = dev.commit_calibration(r)
        pushed = await dev.send_sens(sens)
        dev.calib.done = True
        dev.calib = None
        await asyncio.get_running_loop().run_in_executor(
            NET, push_calib, dev.ip, "done", n, CalibSession.TARGET, True)
        print(f"  {dev.id}: {dev.subject['code']} calibrated on {n} windows, "
              f"k_sigma {r['k_sigma']:.2f}, slider -> {dev.sens:.2f}"
              f"{'' if pushed else ' (board did not take it)'}", flush=True)
        return None

    return "unknown action"


# ----------------------------------------------------------------- the server

def build_app(fleet: Fleet):
    from fastapi import Body, FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI()
    here = os.path.dirname(os.path.abspath(__file__))

    @app.get("/")
    async def index():
        with open(os.path.join(here, "static", "fleet.html"), encoding="utf-8") as f:
            # the page is read from disk every request precisely so an edit shows
            # up on reload; a browser cache would undo that
            return HTMLResponse(f.read(), headers={"Cache-Control": "no-store"})

    @app.get("/api/devices")
    async def devices():
        return JSONResponse({
            "api": API_VERSION,
            "scanning": fleet.scanning,
            "subnet": ", ".join(fleet.subnets),
            "devices": [d.status() for d in
                        sorted(fleet.devices.values(), key=lambda x: x.id)],
        })

    @app.get("/api/tuning")
    async def get_tuning():
        return JSONResponse({"tuning": tuning_state()})

    @app.post("/api/tuning")
    async def set_tuning(body: dict = Body(default={})):
        changed = tuning_apply(body, db=fleet.db)
        if changed:
            print("  tuning: " + ", ".join("%s=%s" % kv for kv in changed.items()),
                  flush=True)
        return {"ok": True, "changed": changed, "tuning": tuning_state()}

    @app.post("/api/rescan")
    async def rescan():
        asyncio.create_task(fleet.discover())
        return {"ok": True}

    # ---- subjects: the people, independent of whatever board they are on ----

    @app.get("/api/subjects")
    async def subjects():
        return JSONResponse({"subjects": fleet.db.subjects_status()})

    @app.post("/api/subjects")
    async def new_subject(body: dict = Body(default={})):
        code = (body.get("code") or "").strip() or fleet.db.next_subject_code()
        if fleet.db.subject_by_code(code):
            return JSONResponse({"error": f"{code} already exists"}, status_code=400)
        sid = fleet.db.create_subject(code, (body.get("name") or "").strip(),
                                      (body.get("notes") or "").strip())
        print(f"  subject {code} created", flush=True)
        return {"ok": True, "id": sid, "code": code}

    @app.delete("/api/subjects/{sid}")
    async def delete_subject(sid: int):
        sub = fleet.db.subject(sid)
        if sub is None:
            return JSONResponse({"error": "unknown subject"}, status_code=404)
        stats = fleet.db.subject_stats(sid)
        fleet.db.delete_subject(sid)
        # any board they were on is now unattributed, and its in-flight score is
        # about a person who no longer exists -- drop it rather than carry it on
        # to whoever is assigned next.
        for d in fleet.devices.values():
            if d.subject and d.subject["id"] == sid:
                d.session_id, d.event_id = None, None
                d.ema, d.level, d.flag = None, None, False
                d.refresh()
        print(f"  subject {sub['code']} deleted "
              f"({stats['n_sessions']} sessions, {stats['n_readings']} readings, "
              f"{stats['n_events']} flags)", flush=True)
        return {"ok": True, "deleted": sub["code"], **stats}

    @app.get("/api/subjects/{sid}/stats")
    async def subject_stats(sid: int):
        if fleet.db.subject(sid) is None:
            return JSONResponse({"error": "unknown subject"}, status_code=404)
        return JSONResponse(fleet.db.subject_stats(sid))

    @app.get("/api/subjects/{sid}/events")
    async def subject_events(sid: int, limit: int = 50):
        return JSONResponse({"events": fleet.db.events_for_subject(sid, limit)})

    @app.post("/api/events/{eid}/ack")
    async def ack(eid: int, body: dict = Body(default={})):
        v = (body.get("verdict") or "").strip() or None
        try:
            fleet.db.ack_event(eid, verdict=v, note=(body.get("note") or "").strip())
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True}

    @app.get("/api/verdicts")
    async def verdicts(subject_id: int = None):
        return JSONResponse(fleet.db.verdict_tally(subject_id))

    # ---- who is wearing what ----

    @app.post("/api/devices/{dev_id}/assign")
    async def assign(dev_id: str, body: dict = Body(default={})):
        dev = fleet.devices.get(dev_id)
        if dev is None:
            return JSONResponse({"error": "unknown device"}, status_code=404)
        sid = body.get("subject_id")
        fleet.db.assign_subject(dev_id, sid)
        # the open session ended with the old wearer; drop the in-flight state too,
        # or the next reading would carry the previous person's smoothed score.
        dev.session_id, dev.event_id = None, None
        dev.ema, dev.level, dev.flag = None, None, False
        dev.refresh()
        who = dev.subject["code"] if dev.subject else "nobody"
        print(f"  {dev_id}: now worn by {who}", flush=True)
        return {"ok": True, "subject": dev.status()["subject"]}

    @app.post("/api/devices/{dev_id}/rename")
    async def rename_device(dev_id: str, body: dict = Body(default={})):
        dev = fleet.devices.get(dev_id)
        if dev is None:
            return JSONResponse({"error": "unknown device"}, status_code=404)
        fleet.db.rename_device(dev_id, body.get("name") or "")
        dev.refresh()
        return {"ok": True, "name": dev.name}

    @app.post("/api/subjects/{sid}/rename")
    async def rename_subject(sid: int, body: dict = Body(default={})):
        sub = fleet.db.subject(sid)
        if sub is None:
            return JSONResponse({"error": "unknown subject"}, status_code=404)
        code = (body.get("code") or "").strip()
        if code and code != sub["code"] and fleet.db.subject_by_code(code):
            return JSONResponse({"error": f"{code} is taken"}, status_code=400)
        fleet.db.rename_subject(sid, name=body.get("name"), code=code or None)
        # every board showing this person is holding a stale copy of the row
        for d in fleet.devices.values():
            if d.subject and d.subject["id"] == sid:
                d.refresh()
        return {"ok": True, "subject": fleet.db.subject(sid)}

    @app.get("/api/devices/{dev_id}/events")
    async def device_events(dev_id: str, limit: int = 50):
        return JSONResponse({"events": fleet.db.events_for_device(dev_id, limit)})

    @app.delete("/api/devices/{dev_id}/events")
    async def clear_device_events(dev_id: str):
        n = fleet.db.clear_events(device_id=dev_id)
        dev = fleet.devices.get(dev_id)
        if dev is not None:
            # an episode in progress now points at a row that is gone; bumping
            # it would write into nothing. dropping the pointer means the next
            # rising edge opens a fresh event instead of silently losing it.
            dev.event_id = None
        print(f"  {dev_id}: flag history cleared ({n} events)", flush=True)
        return {"ok": True, "cleared": n}

    @app.post("/api/calib/{dev_id}/{action}")
    async def calib(dev_id: str, action: str):
        dev = fleet.devices.get(dev_id)
        if dev is None:
            return JSONResponse({"error": "unknown device"}, status_code=404)
        err = await run_calib(dev, action)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        return {"ok": True, "sens": dev.sens}

    return app


async def amain(args) -> int:
    from .infer import LiveAnomalyDetector

    subnets = [args.subnet] if args.subnet else local_subnets()
    if not subnets:
        print("  could not work out this machine's subnet; pass --subnet 192.168.1")
        return 1

    # The page carries the API version it was written against. They are edited in
    # different files and I have already shipped them out of step once, which
    # surfaced as a browser telling the user to restart a master that was already
    # newer than their page. Catch it here instead, where it is one line to fix.
    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "fleet.html")
    try:
        m = re.search(r"NEEDS_API\s*=\s*(\d+)", open(page, encoding="utf-8").read())
        if m and int(m.group(1)) != API_VERSION:
            print(f"  WARNING: fleet.html expects API v{m.group(1)} but this master is "
                  f"v{API_VERSION} -- bump one of them", flush=True)
    except Exception:
        pass

    db = Db(args.db)
    print(f"  store {db.path}", flush=True)
    for name, code, dev in import_legacy_scorers(db, verbose=False):
        print(f"  imported {name} -> subject {code} on {dev}", flush=True)
    restored = tuning_restore(db)
    if restored:
        print("  tuning restored: " + ", ".join("%s=%s" % kv for kv in restored.items()),
              flush=True)
    stale = db.close_stale_sessions()
    if stale:
        print(f"  closed {stale} session(s) left open by a previous run", flush=True)

    print("  loading model… (~45 s: TensorFlow + the 4 MB int8 model)", flush=True)
    det = LiveAnomalyDetector()          # thresholds come per subject, not from here
    # every stored score records which model produced it: swap the model and the
    # thresholds mean something different, so a level without one is unreadable.
    det.model_id = db.register_model("ae_int8.tflite",
                                     os.path.join(SAVE_DIR, "ae_int8.tflite"))
    print("  model ready\n")

    fleet = Fleet(det, subnets, args.device or [], db)
    print("  scanning " + ", ".join(f"{n}.0/24" for n in subnets) + " for boards…",
          flush=True)
    await fleet.discover()
    if not fleet.devices:
        print("  none found. is a board powered and on this network?")
        print("  (looking for an HTTP /health on "
              + ", ".join(f"{n}.1-254" for n in subnets) + ")")
    asyncio.create_task(fleet.rescan_loop(args.rescan))
    asyncio.create_task(fleet.housekeeping_loop())

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
    ap.add_argument("--db", default=DB_PATH, help=f"the store (default {DB_PATH})")
    args = ap.parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n  stopped.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
