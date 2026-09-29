"""induced-stress protocol sessions: record them on the fleet, then score them.

nothing the project has measured so far says the flag responds to STRESS. the
calibration numbers are all calm, and a finger shaken on the sensor fires the
flag through motion, which proves the plumbing and nothing about the method.
this is the experiment that answers it:

    settle      finger on, sit still, until a full clean 60 s window exists
    baseline    3 min  sit still, breathe normally, do not talk
    induction   3 min  serial subtraction out loud, someone pushing the pace,
                       sensor hand still
    recovery    3 min  stop, sit quietly

the phase marks ARE the ground truth -- the induction timestamp is the label --
so they are written to the store the moment each phase begins, at its scheduled
time, not whenever the next frame happens to arrive.

recording runs inside the fleet master (`anomaly.fleet`), because the master
already holds the model, the raw buffer and the store; this module is the run
itself, an operator console that drives it over the master's API, and the
analysis:

    python -m anomaly.fleet                                  # terminal 1
    python -m anomaly.protocol run                           # terminal 2
    python -m anomaly.protocol run --device pulse-a4f2c1 --task "1022 - 13"
    python -m anomaly.protocol run --baseline 240 --induction 240 --recovery 180
    python -m anomaly.protocol next / stop                   # by hand, if needed

    python -m anomaly.protocol list
    python -m anomaly.protocol report                        # every session
    python -m anomaly.protocol report --session 12 --plot

the roster has the same controls on each board's card.

what a session saves, in data/protocols/session_<id>.npz (data/ is gitignored --
this is personal biometric data):

    per second   raw window score, smoothed score, live level and flag, HR,
                 SpO2, contact, signal quality, pipeline state
    per sample   the conditioned 64 Hz waveform with the board's sample index,
                 so a better model later can re-score the same session
                 (--no-raw leaves it out)

scale freezing. the live flag measures each wearer against their own last
three minutes, which is right for all-day monitoring and wrong for this
experiment: three minutes into the task the "calm" reference IS the task, and
the flag argues itself back down. so by default the reference is pinned to the
baseline phase once induction starts (`--no-freeze` tests the product exactly
as it runs day to day). the analysis does not depend on this either way -- it
scores the raw window scores against the session's own baseline phase.

pre-committed reading of a session, before any data exists:

    HR control check    median HR, induction minus baseline, >= +10 bpm.
                        if HR did not rise, the induction did not take and the
                        session is not evidence either way.
    model separation    AUROC of raw window scores, baseline vs induction,
                        >= 0.70 counts as "responds".
    recall@90spec       fraction of induction windows above the baseline's p90
                        -- the project's headline metric, calibrated on this
                        session's own calm.

a window counts toward a phase only if all 60 s of it lie inside that phase, so
the first minute of each phase (whose windows still hold the previous one) is
left out of the score-based numbers. HR is taken over the whole phase.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from .db import DEFAULT_PATH as DB_PATH, PHASES, Db
from .wesad import FS

WIN = 60 * FS
WIN_S = 60.0

DEFAULT_PLAN = (("baseline", 180), ("induction", 180), ("recovery", 180))
PROTO_DIR = os.path.join(os.path.dirname(DB_PATH), "protocols")

# the pre-committed reading. changing these after seeing data is moving the
# goalposts; see the docstring.
HR_RISE_BPM = 10.0
AUC_RESPONDS = 0.70
MIN_WINDOWS = 30          # pure windows per phase below which a session is unscoreable

CUE = {
    "settle": "finger on the sensor, sit still -- the clock starts once a clean 60 s window exists",
    "baseline": "BASELINE -- sit still, breathe normally, do not talk",
    "induction": "INDUCTION -- start the task NOW{task}. keep the sensor hand still",
    "recovery": "RECOVERY -- stop the task. sit quietly, breathe normally",
    "done": "done",
}


def _f(v) -> float:
    try:
        return float(v) if v is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


# ------------------------------------------------------------------- the run

class ProtocolRun:
    """one protocol session on one board, driven by the master's scoring tick.

    the master calls feed() with every frame and tick() once a second; the
    run keeps its own clock, writes the phase marks, and on finish() saves the
    session file and hands the board back to ordinary monitoring.
    """

    def __init__(self, dev, plan=DEFAULT_PLAN, task: str = "", notes: str = "",
                 freeze: bool = True, record_raw: bool = True, clock=time.time,
                 out_dir: str = None):
        plan = [(str(p), float(s)) for p, s in plan]
        if [p for p, _ in plan] != list(PHASES):
            raise ValueError("the plan is %s, in that order" % (PHASES,))
        if any(s <= 0 for _, s in plan):
            raise ValueError("every phase needs a positive duration")
        self.dev = dev
        self.db = dev.db
        self.plan = plan
        self.task = (task or "").strip()
        self.notes = (notes or "").strip()
        self.freeze = bool(freeze)
        self.record_raw = bool(record_raw)
        self.clock = clock
        self.out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(self.db.path)),
                                               "protocols")
        self.i = -1                       # -1 = settling, before the baseline mark
        self.phase_t0 = None              # scheduled start of the current phase
        self.t_created = clock()
        self.session_id = None
        self.marks: list = []
        self.rows: list = []
        self.raw_t, self.raw_idx, self.raw_bvp, self.raw_contact = [], [], [], []
        self.finished = False
        self.aborted = False
        self.reason = ""
        self.t_end = None
        self.path = None

    # ---- lifecycle ----

    def start(self) -> int:
        dev = self.dev
        if dev.subject is None:
            raise ValueError("assign a subject to this board first")
        # what the live flag is judged with. the SLIDER sets the line now (see
        # fleet.Device.threshold_z); k_sigma only decides where it was parked
        # after calibration, so both are kept but thr_z is the one that matters.
        self.subject_code = dev.subject["code"]
        self.calibrated = bool(dev.calibrated)
        self.k_sigma = float(dev.k_sigma)
        self.sens = float(dev.sens)
        self.thr_z = float(dev.threshold_z())
        dev.end_session()                 # a protocol is its own session
        note = "protocol%s%s" % (" | task: " + self.task if self.task else "",
                                 " | " + self.notes if self.notes else "")
        self.session_id = self.db.open_session(
            dev.subject["id"], dev.id, "protocol",
            model_id=getattr(dev.det, "model_id", None), sens=dev.sens, notes=note)
        dev.session_id = self.session_id
        dev.event_id = None
        dev.freeze_norm = False
        dev.protocol = self
        print("  %s: protocol session %d for %s (%s)" % (
            dev.id, self.session_id, self.subject_code,
            " / ".join("%s %ds" % (p, s) for p, s in self.plan)), flush=True)
        return self.session_id

    @property
    def phase(self) -> str:
        if self.finished:
            return "done"
        return "settle" if self.i < 0 else self.plan[self.i][0]

    def _mark(self, i: int, t: float, by_hand: bool = False):
        phase = self.plan[i][0]
        label = self.task if phase == "induction" else ""
        if by_hand:
            label = (label + " " if label else "") + "(advanced by hand)"
        self.db.add_mark(self.session_id, phase, label=label, t=t)
        self.marks.append((t, phase))
        self.i, self.phase_t0 = i, t
        # pin the live calm reference to the baseline phase: see the docstring
        if self.freeze and i >= 1:
            self.dev.freeze_norm = True
        print("  %s: protocol %d -> %s" % (self.dev.id, self.session_id, phase), flush=True)

    def advance_clock(self):
        """move through the plan on schedule. safe to call as often as you like."""
        if self.finished:
            return
        now = self.clock()
        if self.i < 0:
            # the clock starts only once a full window of clean contact exists,
            # so the baseline phase is scored from its first second
            if self.dev.contact and len(self.dev.buf) >= WIN:
                self._mark(0, now)
            return
        while not self.finished:
            end = self.phase_t0 + self.plan[self.i][1]
            if now < end:
                break
            # the NEXT phase starts when this one was scheduled to end, not when a
            # late tick noticed: the mark is the label, so it must not drift
            if self.i + 1 < len(self.plan):
                self._mark(self.i + 1, end)
            else:
                self.finish(t_end=end)

    def next_phase(self):
        """the operator says the phase has changed now."""
        if self.finished:
            return
        now = self.clock()
        if self.i + 1 < len(self.plan):
            self._mark(self.i + 1, now, by_hand=True)
        else:
            self.finish(t_end=now)

    # ---- data in ----

    def feed(self, idx, bvp, contact: bool):
        """one websocket frame of conditioned 64 Hz samples."""
        if self.finished or not self.record_raw or not bvp:
            return
        t = self.clock()
        idx = idx or []
        for k, v in enumerate(bvp):
            self.raw_t.append(t)
            self.raw_idx.append(int(idx[k]) if k < len(idx) else -1)
            self.raw_bvp.append(_f(v))
            self.raw_contact.append(1 if contact else 0)

    def tick(self, raw_score=None):
        """the master's once-a-second scoring tick, after it has scored."""
        if self.finished:
            return
        self.advance_clock()
        if self.finished:
            return
        d = self.dev
        ond = getattr(d, "ond", None) or {}
        if ond.get("st") not in ("ok", "hold"):      # no score from the board this second
            ond = {}
        self.rows.append((
            self.clock(), self.phase, _f(raw_score),
            _f(d.score) if d.ema is not None else math.nan,
            _f(d.level), -1 if d.level is None else int(bool(d.flag)),
            _f(d.bpm), _f(d.spo2), int(bool(d.contact)), _f(d.quality), d.state or "",
            # the board's OWN detector, which rides along in every frame: the
            # on-device model scored on the same session as the host's
            _f(ond.get("s")), _f(ond.get("l")), int(ond.get("f") or 0) if ond else -1))

    # ---- the end ----

    def finish(self, aborted: bool = False, reason: str = "", t_end: float = None):
        if self.finished:
            return self.status()
        self.finished = True
        self.aborted = bool(aborted)
        self.reason = reason or ("completed" if not aborted else "stopped")
        self.t_end = t_end or self.clock()
        try:
            self.path = self._save()
        except Exception as e:           # the marks and readings are in the store regardless
            print("  %s: could not save protocol file: %s" % (self.dev.id, e), flush=True)
        dev = self.dev
        dev.freeze_norm = False
        dev.protocol = None
        dev.end_session()                # closes any open flag episode with it
        if self.path:
            rel = os.path.relpath(self.path, os.path.dirname(os.path.abspath(self.db.path)))
            self.db.set_session_raw(self.session_id, rel)
        self.db.append_session_note(
            self.session_id, " | %s%s" % ("ABORTED: " if aborted else "", self.reason))
        dev.last_protocol = self.status()
        print("  %s: protocol %d %s (%s)%s" % (
            dev.id, self.session_id, "aborted" if aborted else "complete", self.reason,
            "  -> python -m anomaly.protocol report --session %d" % self.session_id),
            flush=True)
        return dev.last_protocol

    def _save(self) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, "session_%d.npz" % self.session_id)
        R = list(zip(*self.rows)) if self.rows else [[] for _ in range(14)]
        meta = {"session_id": self.session_id, "subject": self.subject_code,
                "device": self.dev.id, "plan": self.plan, "task": self.task,
                "notes": self.notes, "freeze": self.freeze, "record_raw": self.record_raw,
                "calibrated": self.calibrated, "k_sigma": self.k_sigma,
                "sens": self.sens, "thr_z": self.thr_z,
                "started": self.t_created, "ended": self.t_end,
                "aborted": self.aborted, "reason": self.reason, "fs": FS, "win": WIN}
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            t=np.asarray(R[0], np.float64), phase=np.asarray(R[1], dtype=str),
            raw=np.asarray(R[2], np.float64), ema=np.asarray(R[3], np.float64),
            level=np.asarray(R[4], np.float64), flag=np.asarray(R[5], np.int8),
            bpm=np.asarray(R[6], np.float64), spo2=np.asarray(R[7], np.float64),
            contact=np.asarray(R[8], np.int8), quality=np.asarray(R[9], np.float64),
            state=np.asarray(R[10], dtype=str),
            ond_score=np.asarray(R[11], np.float64), ond_level=np.asarray(R[12], np.float64),
            ond_flag=np.asarray(R[13], np.int8),
            mark_t=np.asarray([m[0] for m in self.marks], np.float64),
            mark_phase=np.asarray([m[1] for m in self.marks], dtype=str),
            raw_t=np.asarray(self.raw_t, np.float64),
            raw_idx=np.asarray(self.raw_idx, np.int64),
            raw_bvp=np.asarray(self.raw_bvp, np.float32),
            raw_contact=np.asarray(self.raw_contact, np.int8))
        return path

    def status(self) -> dict:
        now = self.t_end if self.finished else self.clock()
        remaining = None
        if not self.finished and self.i >= 0:
            remaining = max(0.0, self.phase_t0 + self.plan[self.i][1] - now)
        cur = [r for r in self.rows if r[1] == self.phase]
        return {
            "session_id": self.session_id, "phase": self.phase, "index": self.i,
            "phases": [p for p, _ in self.plan], "durations": [s for _, s in self.plan],
            "remaining_s": None if remaining is None else round(remaining, 1),
            "elapsed_s": round(now - (self.marks[0][0] if self.marks else self.t_created), 1),
            "task": self.task, "freeze": self.freeze, "record_raw": self.record_raw,
            "cue": CUE.get(self.phase, "").format(task=": " + self.task if self.task else ""),
            "phase_rows": len(cur),
            "phase_scored": sum(1 for r in cur if not math.isnan(r[2])),
            "finished": self.finished, "aborted": self.aborted, "reason": self.reason,
        }


# ------------------------------------------------------------------ analysis

def _resolve(db: Db, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(db.path)), p)


def load_session(db: Db, sid: int) -> dict:
    s = db.q1("""SELECT s.*, sub.code AS subject_code FROM session s
                 JOIN subject sub ON sub.id = s.subject_id WHERE s.id=?""", (sid,))
    if s is None or s["kind"] != "protocol":
        raise ValueError("session %s is not a protocol session" % sid)
    marks = [(m["t"], m["phase"]) for m in db.marks(sid)]
    out = {"session": s, "marks": marks, "meta": {}, "source": "readings"}
    path = _resolve(db, s["raw_path"]) if s["raw_path"] else None
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        out["meta"] = json.loads(str(z["meta"]))
        for k in ("t", "raw", "ema", "level", "flag", "bpm", "spo2", "contact", "quality"):
            out[k] = z[k]
        if "ond_score" in z.files:             # sessions from before the board scored
            out["ond_score"] = z["ond_score"]
        out["n_raw"] = int(z["raw_bvp"].size)
        out["source"] = "file"
    else:
        # no session file (an aborted master, or it was lost): the store still
        # has the smoothed score per second, so the session is not wasted
        rs = db.readings(sid, limit=100000)
        col = lambda k: np.asarray([_f(r[k]) for r in rs], np.float64)
        out.update(t=col("t"), raw=np.full(len(rs), np.nan), ema=col("score"),
                   level=col("level"), bpm=col("bpm"), spo2=col("spo2"),
                   quality=col("quality"),
                   flag=np.asarray([-1 if r["flag"] is None else r["flag"] for r in rs], np.int8),
                   contact=np.asarray([r["contact"] or 0 for r in rs], np.int8))
        out["n_raw"] = 0
    out["end"] = out["meta"].get("ended") or s["ended"] or (
        float(out["t"][-1]) if len(out["t"]) else None)
    return out


def auroc(neg, pos) -> float:
    """P(a random induction window scores above a random baseline window)."""
    from scipy.stats import rankdata
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if not len(neg) or not len(pos):
        return math.nan
    r = rankdata(np.concatenate([neg, pos]))
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def analyse(S: dict) -> dict:
    t = S["t"]
    # separate on the raw window scores; the smoothed one only if that is all
    # there is (EMA drags each phase's start into the one before it)
    score = S["raw"] if np.isfinite(S["raw"]).any() else S["ema"]
    marks = sorted(S["marks"])
    end = S["end"]
    phases = {}
    for j, (t0, ph) in enumerate(marks):
        if ph in phases:
            continue
        t1 = marks[j + 1][0] if j + 1 < len(marks) else end
        if t1 is None:
            continue
        inp = (t >= t0) & (t < t1)
        pure = inp & (t - WIN_S >= t0)
        sc = score[pure & np.isfinite(score)]
        ond = S.get("ond_score")
        osc = ond[pure & np.isfinite(ond)] if ond is not None else np.array([])
        hr = S["bpm"][inp]
        hr = hr[np.isfinite(hr) & (hr >= 35) & (hr <= 220)]
        lv = S["level"][pure]
        fl = S["flag"][pure]
        fl = fl[fl >= 0]
        phases[ph] = {
            "t0": t0, "t1": t1, "secs": t1 - t0,
            "contact": float(S["contact"][inp].mean()) if inp.any() else math.nan,
            "n_pure": int(pure.sum()), "scores": sc, "n_scored": int(sc.size),
            "ond_scores": osc,
            "hr": float(np.median(hr)) if hr.size else math.nan,
            "score": float(np.median(sc)) if sc.size else math.nan,
            "level": float(np.nanmean(lv)) if np.isfinite(lv).any() else math.nan,
            "flagged": float(fl.mean()) if fl.size else math.nan,
        }
    A = {"phases": phases, "score_kind": "raw" if score is S["raw"] else "smoothed"}
    b, i, r = (phases.get(p) for p in PHASES)
    A["dhr"] = (i["hr"] - b["hr"]) if (b and i) else math.nan
    A["auc"] = A["recall90"] = A["recovery_above"] = A["ttf"] = math.nan
    A["ond_auc"] = A["ond_recall90"] = math.nan
    ok = bool(b and i and b["n_scored"] >= MIN_WINDOWS and i["n_scored"] >= MIN_WINDOWS)
    if ok:
        A["auc"] = auroc(b["scores"], i["scores"])
        thr = float(np.quantile(b["scores"], 0.90))
        A["thr90"] = thr
        A["recall90"] = float(np.mean(i["scores"] > thr))
        if r and r["n_scored"]:
            A["recovery_above"] = float(np.mean(r["scores"] > thr))
    # the on-device detector, judged by the same rule on the same windows
    if b and i and b["ond_scores"].size >= MIN_WINDOWS and i["ond_scores"].size >= MIN_WINDOWS:
        A["ond_auc"] = auroc(b["ond_scores"], i["ond_scores"])
        othr = float(np.quantile(b["ond_scores"], 0.90))
        A["ond_recall90"] = float(np.mean(i["ond_scores"] > othr))
    if i:
        hit = np.nonzero((t >= i["t0"]) & (S["flag"] == 1))[0]
        A["ttf"] = float(t[hit[0]] - i["t0"]) if hit.size else math.nan

    if not ok:
        A["verdict"] = ("UNSCOREABLE", "fewer than %d clean full windows in baseline or "
                        "induction" % MIN_WINDOWS)
    elif math.isnan(A["dhr"]):
        A["verdict"] = ("NO HR", "no valid heart rate in baseline or induction, so the "
                        "control check cannot run -- the session is not evidence")
    elif A["dhr"] < HR_RISE_BPM:
        A["verdict"] = ("NO INDUCTION", "HR rose %+.1f bpm (needs %+.0f) -- the task did not "
                        "take, so this session is not evidence either way" % (A["dhr"], HR_RISE_BPM))
    elif A["auc"] >= AUC_RESPONDS:
        A["verdict"] = ("RESPONDS", "HR rose and the model's score rose with it")
    else:
        A["verdict"] = ("HR ONLY", "HR rose and the model did not follow -- the "
                        "domain-transfer finding; report it")
    return A


def _fmt(v, f="%.2f", none="-"):
    return none if v is None or (isinstance(v, float) and math.isnan(v)) else f % v


def print_session(S: dict, A: dict):
    s, m = S["session"], S["meta"]
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["started"]))
    status = ("ABORTED (%s)" % m.get("reason", "")) if m.get("aborted") else (
        "completed" if m else "no session file -- from stored readings")
    live = ""
    if m:
        # sessions recorded before thr_z was saved only have k_sigma, which did
        # not set the line -- say so rather than print it as if it had
        line = ("flag line %.2f sigma (slider %.2f)" % (m["thr_z"], m["sens"])
                if "thr_z" in m else "flag line not recorded")
        live = " | live flag %s, %s%s" % (
            "calibrated subject" if m.get("calibrated") else "ZERO-SHOT", line,
            ", scale frozen after baseline" if m.get("freeze") else ", scale free-running")
    print("\n  session %d | %s | %s | %s | %s%s" % (
        s["id"], s["subject_code"], s["device_id"], when, status, live))
    if m.get("task"):
        print("  task: %s" % m["task"])
    print("\n  %-10s %6s %8s %7s %7s %9s %7s %8s" % (
        "phase", "secs", "contact", "scored", "HR", "score", "level", "flagged"))
    for ph in PHASES:
        p = A["phases"].get(ph)
        if not p:
            print("  %-10s  (never reached)" % ph)
            continue
        print("  %-10s %6.0f %7s %7d %7s %9s %7s %8s" % (
            ph, p["secs"], _fmt(p["contact"] * 100, "%.0f%%"), p["n_scored"],
            _fmt(p["hr"], "%.1f"), _fmt(p["score"], "%.4f"), _fmt(p["level"]),
            _fmt(p["flagged"] * 100 if not math.isnan(p["flagged"]) else math.nan, "%.0f%%")))
    print()
    ok = "PASS" if A["dhr"] >= HR_RISE_BPM else "FAIL"
    print("  HR control check   %s bpm (needs >= %+.0f)  %s" % (
        _fmt(A["dhr"], "%+.1f"), HR_RISE_BPM, ok if not math.isnan(A["dhr"]) else ""))
    print("  model separation   AUROC %s, baseline vs induction (%s window scores; "
          "responds at >= %.2f)" % (_fmt(A["auc"]), A["score_kind"], AUC_RESPONDS))
    print("  recall@90spec      %s of induction windows above the baseline's p90" % _fmt(A["recall90"]))
    if not math.isnan(A["ond_auc"]):
        print("  on-device model    AUROC %s, recall@90spec %s (the board's own detector; "
              "the gap to the host is the on-device cost)" % (_fmt(A["ond_auc"]), _fmt(A["ond_recall90"])))
    print("  still elevated     %s of recovery windows above that line" % _fmt(A["recovery_above"]))
    print("  first live flag    %s" % (_fmt(A["ttf"], "%.0f s after induction began")
                                     if not math.isnan(A["ttf"]) else "never, during or after induction"))
    print("  verdict            %s -- %s" % A["verdict"])


def plot_session(S: dict, A: dict, out: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = min(S["marks"])[0] if S["marks"] else (S["t"][0] if len(S["t"]) else 0.0)
    x = (S["t"] - t0) / 60.0
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    shade = {"baseline": "#2E86AB", "induction": "#F2645A", "recovery": "#2BAE73"}
    for ph, p in A["phases"].items():
        for a in ax:
            a.axvspan((p["t0"] - t0) / 60, (p["t1"] - t0) / 60, color=shade[ph], alpha=0.10)
        ax[0].text((p["t0"] - t0) / 60, 1.02, " " + ph, transform=ax[0].get_xaxis_transform(),
                   fontsize=9, color=shade[ph])
    ax[0].plot(x, S["bpm"], lw=1, color="#333")
    ax[0].set_ylabel("HR (bpm)")
    sc = S["raw"] if A["score_kind"] == "raw" else S["ema"]
    ax[1].plot(x, sc, lw=0.8, color="#555", label="window score")
    if "thr90" in A:
        ax[1].axhline(A["thr90"], ls="--", color="#F2645A", lw=1, label="baseline p90")
    ax[1].set_ylabel("recon. error")
    ax[1].legend(loc="upper left", fontsize=8)
    lv = S["level"]
    ax[2].plot(x, lv, lw=1, color="#2E86AB", label="live level")
    fl = S["flag"] == 1
    ax[2].fill_between(x, 0, 1, where=fl, color="#F2645A", alpha=0.25, step="post",
                       label="live flag")
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("level")
    ax[2].set_xlabel("minutes from baseline start")
    ax[2].legend(loc="upper left", fontsize=8)
    s = S["session"]
    fig.suptitle("protocol session %d | %s | %s" % (s["id"], s["subject_code"], A["verdict"][0]),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def print_pooled(rows: list):
    print("\n  %-8s %-8s %8s %7s %9s %7s %8s  %s" % (
        "session", "subject", "dHR", "AUROC", "recall90", "flag@", "on-dev", "verdict"))
    for s, A in rows:
        print("  %-8d %-8s %8s %7s %9s %7s %8s  %s" % (
            s["id"], s["subject_code"], _fmt(A["dhr"], "%+.1f"), _fmt(A["auc"]),
            _fmt(A["recall90"]), _fmt(A["ttf"], "%.0fs"), _fmt(A["ond_auc"]), A["verdict"][0]))
    took = [(s, A) for s, A in rows if A["verdict"][0] in ("RESPONDS", "HR ONLY")]
    subj = sorted(set(s["subject_code"] for s, _ in took))
    print("\n  induction took (HR >= %+.0f) in %d of %d sessions, %d subject(s)" % (
        HR_RISE_BPM, len(took), len(rows), len(subj)))
    if took:
        auc = [A["auc"] for _, A in took]
        rec = [A["recall90"] for _, A in took]
        n_resp = sum(1 for _, A in took if A["verdict"][0] == "RESPONDS")
        print("  of those, the model responded in %d | median AUROC %.2f | median recall@90spec %.2f"
              % (n_resp, float(np.median(auc)), float(np.median(rec))))
    print("  the unit of evidence is the SUBJECT: 1 Hz windows overlap by 59 s, so a"
          " session's\n  AUROC describes that session and is not an independent sample.")


# ----------------------------------------------------------------- console

MASTER = "http://localhost:8002"


def _api(base: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error")
        except Exception:
            msg = None
        raise RuntimeError(msg or "HTTP %d for %s" % (e.code, path))


def _pick_device(base: str, want: str = None) -> dict:
    from .fleet import API_VERSION
    try:
        d = _api(base, "/api/devices")
    except (urllib.error.URLError, OSError):
        raise RuntimeError("no master at %s -- start it first: python -m anomaly.fleet" % base)
    if d.get("api") != API_VERSION:
        raise RuntimeError("the master speaks API v%s, this is v%d -- restart it: "
                           "python -m anomaly.fleet" % (d.get("api"), API_VERSION))
    devs = d["devices"]
    if want:
        for x in devs:
            if x["id"] == want or x.get("name") == want:
                return x
        raise RuntimeError("no board %r; the master sees: %s" % (
            want, ", ".join(x["id"] for x in devs) or "none"))
    live = [x for x in devs if x["connected"]]
    if len(live) != 1:
        raise RuntimeError("%d boards connected (%s) -- say which with --device" % (
            len(live), ", ".join(x["id"] for x in live) or "none"))
    return live[0]


def cmd_run(a) -> int:
    base = a.master.rstrip("/")
    dev = _pick_device(base, a.device)
    if not dev.get("subject"):
        print("  %s has nobody assigned -- pick a subject on the roster first" % dev["id"])
        return 1
    body = {"plan": {"baseline": a.baseline, "induction": a.induction, "recovery": a.recovery},
            "task": a.task, "notes": a.notes, "freeze": not a.no_freeze,
            "record_raw": not a.no_raw}
    r = _api(base, "/api/protocol/%s/start" % dev["id"], body)
    sid = r["session_id"]
    print("\n  protocol session %d | %s on %s | %s" % (
        sid, dev["subject"]["code"], dev["id"],
        " / ".join("%s %ds" % kv for kv in body["plan"].items())))
    print("  ctrl+c stops it (the data so far is kept, marked aborted)\n")
    last = None
    try:
        while True:
            d = _pick_device(base, dev["id"])
            p = d.get("protocol")
            if p is None:
                lp = d.get("last_protocol") or {}
                if lp.get("session_id") == sid:
                    print("\n\n  %s -- %s" % ("aborted" if lp.get("aborted") else "done",
                                             lp.get("reason", "")))
                    break
                print("\n  the master lost the run (restarted?)")
                return 1
            if p["phase"] != last:
                last = p["phase"]
                print("\n\n  ===== %s =====\n" % p["cue"])
            rem = p["remaining_s"]
            left = "--:--" if rem is None else "%d:%02d" % (int(rem) // 60, int(rem) % 60)
            line = "  %-9s %s left | HR %s | level %s%s | finger %s | %s" % (
                p["phase"], left, d.get("bpm") if d.get("bpm") is not None else "-",
                "-" if d.get("level") is None else "%3.0f%%" % (d["level"] * 100),
                " FLAG" if d.get("flag") else "", "on" if d.get("contact") else "OFF",
                d.get("quality_note") or d.get("state"))
            sys.stdout.write("\r" + line[:110].ljust(110))
            sys.stdout.flush()
            time.sleep(1.0)
    except KeyboardInterrupt:
        try:
            _api(base, "/api/protocol/%s/stop" % dev["id"], {"reason": "stopped from the console"})
        except Exception as e:
            print("\n  could not stop it: %s -- use the roster" % e)
            return 1
        print("\n\n  stopped; the data so far is kept, marked aborted")
    print("  report: python -m anomaly.protocol report --session %d --plot\n" % sid)
    return 0


def cmd_simple(a, action: str) -> int:
    base = a.master.rstrip("/")
    dev = _pick_device(base, a.device)
    body = {"reason": "stopped from the command line"} if action == "stop" else {}
    r = _api(base, "/api/protocol/%s/%s" % (dev["id"], action), body)
    p = r.get("protocol") or r.get("last") or {}
    print("  %s: %s" % (dev["id"], p.get("phase") if action == "next" else p.get("reason")))
    return 0


def cmd_list(a) -> int:
    db = Db(a.db)
    rows = db.protocol_sessions()
    if not rows:
        print("  no protocol sessions yet")
        return 0
    print("\n  %-8s %-8s %-16s %-17s %7s %6s  %s" % (
        "session", "subject", "device", "started", "minutes", "marks", "notes"))
    for s in rows:
        dur = ((s["ended"] or time.time()) - s["started"]) / 60.0
        print("  %-8d %-8s %-16s %-17s %7.1f %6d  %s" % (
            s["id"], s["subject_code"], s["device_id"],
            time.strftime("%Y-%m-%d %H:%M", time.localtime(s["started"])), dur,
            s["n_marks"], s["notes"] or ""))
    print()
    return 0


def cmd_report(a) -> int:
    db = Db(a.db)
    ids = a.session or [s["id"] for s in db.protocol_sessions() if s["n_marks"]]
    if not ids:
        print("  no protocol sessions with phase marks yet")
        return 0
    done = []
    for sid in ids:
        try:
            S = load_session(db, sid)
        except ValueError as e:
            print("  %s" % e)
            continue
        A = analyse(S)
        print_session(S, A)
        if a.plot:
            out = os.path.join(os.path.dirname(os.path.abspath(db.path)), "protocols",
                               "session_%d.png" % sid)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            print("  plot -> %s" % plot_session(S, A, out))
        done.append((S["session"], A))
    if len(done) > 1:
        print_pooled(done)
    print()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a protocol session on a board, with a live console")
    r.add_argument("--device", help="board id or name (default: the only connected one)")
    r.add_argument("--baseline", type=float, default=180, help="seconds (default 180)")
    r.add_argument("--induction", type=float, default=180, help="seconds (default 180)")
    r.add_argument("--recovery", type=float, default=180, help="seconds (default 180)")
    r.add_argument("--task", default="serial subtraction", help="what the induction is")
    r.add_argument("--notes", default="", help="free text stored with the session")
    r.add_argument("--no-freeze", action="store_true",
                   help="let the live calm reference keep adapting (the product as-is)")
    r.add_argument("--no-raw", action="store_true", help="do not save the waveform")
    for name in ("next", "stop"):
        p = sub.add_parser(name, help="%s the running protocol" % (
            "advance to the next phase of" if name == "next" else "stop (keeps the data)"))
        p.add_argument("--device")
    for p in [r] + [sub.choices["next"], sub.choices["stop"]]:
        p.add_argument("--master", default=MASTER, help="the fleet master (default %s)" % MASTER)

    ls = sub.add_parser("list", help="list protocol sessions in the store")
    rp = sub.add_parser("report", help="score protocol sessions")
    rp.add_argument("--session", type=int, action="append",
                    help="session id; repeatable (default: all)")
    rp.add_argument("--plot", action="store_true", help="save a figure per session")
    for p in (ls, rp):
        p.add_argument("--db", default=DB_PATH)

    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            return cmd_run(a)
        if a.cmd in ("next", "stop"):
            return cmd_simple(a, a.cmd)
        if a.cmd == "list":
            return cmd_list(a)
        return cmd_report(a)
    except RuntimeError as e:
        print("  %s" % e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
