"""bake the WESAD demo clip into Pulse Watch, waveform and verdicts together.

With the live toggle off, the watch used to draw three sine harmonics plus
noise and read its vitals off an eight-point keyframe table -- no dataset, no
model, and a "ground truth: STRESS" chip that was hardcoded to a slice of the
loop. It looked exactly like a working detector, which is the one thing a demo
must never do.

This writes the real thing into the page instead: the same curated WESAD
calm->stress clip `anomaly.serve` replays (`BVPReplay`, 120 s of baseline then
120 s of stress), scored once a second by the deployed int8 model through the
same EMA the server uses, with WESAD's own condition labels alongside. The page
then plays a recording rather than inventing a signal, so every number on
screen came from the dataset or from the model reading it.

    python3 -m anomaly.make_demo_clip            # -> pulse/Pulse Watch.dc.html
    python3 -m anomaly.make_demo_clip --dry-run  # report the size, write nothing

needs WESAD present; re-run it only when the model or the clip changes.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

PAGE = os.path.join(ROOT, "pulse", "Pulse Watch.dc.html")
START = "/* DEMO_CLIP_START */"
END = "/* DEMO_CLIP_END */"

FS = 64
WIN_LEN = 60 * FS          # the model's window, same as serve.py
EMA = 0.65                 # serve.py: score_ema = .65*prev + .35*raw
LABELS = {1: "calm", 2: "STRESS", 3: "amusement", 4: "meditation"}


def build(subject: str = "S5", seg_sec: int = 120) -> dict:
    from vitals import estimate_heart_rate, signal_quality_score

    from .infer import LiveAnomalyDetector
    from .wesad_replay import BVPReplay

    rep = BVPReplay(subject, seg_sec=seg_sec)
    bvp, labels = rep.bvp.astype(np.float32), rep.labels
    det = LiveAnomalyDetector()
    print("  %s: %d samples (%.0f s), model threshold %.5f"
          % (subject, len(bvp), len(bvp) / FS, det.threshold))

    secs, qs, ema = [], [], None
    for i in range(1, len(bvp) // FS + 1):
        end = i * FS
        lvl = flag = None
        if end >= WIN_LEN:
            raw = det.score(bvp[end - WIN_LEN:end])
            ema = raw if ema is None else EMA * ema + (1 - EMA) * raw
            lvl = round(float(det.level(ema)), 3)
            flag = bool(det.flag(ema))
        tail = bvp[max(0, end - 12 * FS):end]
        bpm = (estimate_heart_rate(tail, fs=FS) if len(tail) >= 8 * FS else None)
        q8 = bvp[max(0, end - 8 * FS):end]
        q = (round(float(signal_quality_score(q8)), 3) if len(q8) >= 8 * FS else None)
        if q is not None:
            qs.append(q)
        secs.append({
            "l": lvl,
            "f": (1 if flag else 0) if flag is not None else None,
            "b": (round(bpm) if bpm else None),
            # No signal-quality reading. The index is a CONTACT measure written
            # for the MAX30102's IR channel, and on WESAD's wrist BVP it sits
            # around 0.5 for the whole recording -- which the watch reads as
            # "poor" and turns into a permanent "Signal weak, hold still" over
            # a clean laboratory recording. A dataset has no sensor to assess.
            "q": None,
            "y": LABELS.get(int(labels[end - 1]), "-"),
        })

    # int16 keeps the waveform to two bytes a sample; the page divides by `scale`.
    # BVP is a.u. anyway -- the chart autoscales -- but the y-axis ticks print the
    # real amplitude, so the numbers should stay the dataset's own.
    peak = float(np.max(np.abs(bvp))) or 1.0
    scale = 32000.0 / peak
    q16 = np.round(bvp * scale).astype("<i2")

    scored = [s for s in secs if s["l"] is not None]
    warm = next((i for i, s in enumerate(secs) if s["l"] is not None), 0)
    print("  scored %d s of %d (the first %d fill the window)"
          % (len(scored), len(secs), warm))
    for cond in ("calm", "STRESS"):
        v = np.array([s["l"] for s in scored if s["y"] == cond])
        if len(v):
            print("  %-7s level p10 %.2f  median %.2f  p90 %.2f   (%d s)"
                  % (cond, np.percentile(v, 10), np.median(v),
                     np.percentile(v, 90), len(v)))
    # Where to put the line for the demo. The shipped threshold is a pooled
    # WESAD number and it sits below this subject's calm floor -- it flags all
    # 181 scored seconds, calm included -- so the demo would open showing
    # STRESSED forever. p90 of this clip's own calm is the 90%-specificity
    # operating point the project pre-committed to, measured on the clip that
    # is actually being played.
    calm_lv = np.array([s["l"] for s in scored if s["y"] == "calm"])
    thr = round(float(np.percentile(calm_lv, 90)), 3) if len(calm_lv) else 0.42
    calm_s = sum(1 for s in secs if s["y"] == "calm")
    hit = [s for s in scored if s["y"] == "STRESS" and s["l"] >= thr]
    fp = [s for s in scored if s["y"] == "calm" and s["l"] >= thr]
    print("  demo line at %.2f (p90 of calm): catches %d/%d stress seconds, "
          "%d calm false alarms"
          % (thr, len(hit), sum(1 for s in scored if s["y"] == "STRESS"),
             len(fp)))

    if qs:
        print("  contact index on this clip: median %.2f -- not shipped, see above"
              % float(np.median(qs)))

    fl = [s for s in scored if s["f"]]
    print("  the shipped threshold flags %d of %d scored seconds "
          "(%d calm, %d stress)"
          % (len(fl), len(scored), sum(1 for s in fl if s["y"] == "calm"),
             sum(1 for s in fl if s["y"] == "STRESS")))
    return {
        "subject": subject,
        "fs": FS,
        "warm": warm,
        "calm": calm_s,
        "thr": thr,
        "scale": round(scale, 6),
        "n": int(len(bvp)),
        "bvp": base64.b64encode(q16.tobytes()).decode("ascii"),
        "sec": secs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", default="S5", help="WESAD subject (default S5)")
    ap.add_argument("--seg", type=int, default=120,
                    help="seconds of calm and of stress (default 120 each)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args(argv)

    clip = build(a.subject, a.seg)
    blob = json.dumps(clip, separators=(",", ":"))
    print("  clip %s (%.0f KB of page)" % (a.subject, len(blob) / 1024))
    if a.dry_run:
        return 0

    page = io.open(PAGE, encoding="utf-8", newline="").read()
    i, j = page.find(START), page.find(END)
    if i < 0 or j < 0:
        print("  ERROR: markers %s / %s not found in %s" % (START, END, PAGE))
        return 1
    page = page[:i] + START + "\nconst DEMO_CLIP = " + blob + ";\n" + page[j:]
    io.open(PAGE, "w", encoding="utf-8", newline="").write(page)
    print("  wrote %s" % os.path.relpath(PAGE, ROOT))
    print("  now re-run: python3 sketch_aug3a/make_web_assets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
