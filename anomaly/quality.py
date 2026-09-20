"""signal quality: decide whether a window is worth showing the model at all.

the detector had no preprocessing stage, and a measurement of what that costs:
a 50 ms tap on the sensor -- 3 samples out of 3840, 0.08% of a 60 s window --
drove the reconstruction error from a calm 0.19 to 0.29, which is 97% of a
calibrated band. two things multiply to do that.

the score is a mean of SQUARED per-sample error, so a tap at 5x the pulse
amplitude contributes 25x the error and three catastrophic samples outweigh
3837 good ones. and a calibrated band is narrow by construction -- it is one
person's own calm spread -- so anything that is not calm saturates it.

worse, the 60 s window does not average a tap away. it holds it: the buffer is
rescored every second, so the tap lands in the very next score and stays there
for a full minute until it slides out the far end.

so this stage sits between the buffer and the model:

    mask     which samples are transients rather than pulse
    repair   short bursts are interpolated across -- standard PPG practice
    gate     a window too damaged to repair is not scored at all

the test is on the FIRST DIFFERENCE, not on amplitude. a pulse has legitimately
large peaks but a bounded slew rate; a knock does not. comparing each sample's
step against the robust spread of all the steps separates them without needing
to know the signal's units, which differ between the board and WESAD.

nothing here is learned, so it costs no training and no re-validation, and it
is small enough to port to C for the board later.
"""
from __future__ import annotations

import numpy as np

# how many robust sigmas a single sample-to-sample step may be before it is
# called a transient. MEASURED on WESAD S5 rather than guessed: across 29 clean
# 60 s windows (calm and stress), the fraction of samples above z=60 is 0 at the
# median and 0.3% in the worst window, while even a modest 3x tap reaches z=106
# and a 20x tap z=717. an earlier guess of 8 refused 17 of 19 clean windows.
#
# this is a wrist-BVP number. the MAX30102 fingertip signal has its own noise
# floor, so re-measure on device data before trusting the margin there.
SPIKE_Z = 60.0

# samples either side of a detected spike that also go, since a knock rings.
SPREAD = 3

# A press that lasts is two transients with a plateau between them, and the
# derivative test only sees the edges -- it repaired those and left two seconds
# of invented signal in the middle, still 102% above the clean score. So the
# span between two nearby transients is masked too, but ONLY when it is actually
# displaced: pulse oscillates about the window median, so a clean stretch has a
# span-median near zero displacement while a press sits far off it. Comparing
# span MEDIANS rather than samples is what keeps legitimate pulse peaks (which
# reach 72 robust sigmas on their own) from being caught.
BRIDGE_GAP = 10 * 64        # look this far ahead for the falling edge (~10 s)
BRIDGE_Z = 4.0              # how displaced the span between them has to be

# above this fraction of the window damaged, repairing would be inventing a
# pulse rather than restoring one, so the window is refused instead.
MAX_REPAIR = 0.02

# a window this flat is not a pulse -- no finger, a saturated LED, a dead read.
MIN_MAD = 1e-6

# A real tap, captured on the rig, is not a spike -- it is SECONDS of raised
# amplitude. event_12.npz is six seconds of tapping at up to 12.5x the window's
# median second, and its per-sample steps peak at only z=78, so the transient
# test above masked 0.9% of it and passed the window straight to the model,
# which duly flagged it.
#
# The reference is the 25th PERCENTILE of the slice amplitudes, not the median.
# With the median, a window that is mostly tapping has a tapping second as its
# reference and everything looks normal beside it: in event_29 ten seconds of
# tapping ran 437-1433 against a median of 251, so a 3.5x bar sat at 880 and let
# four of those ten through -- enough to flag. p25 stays honest while up to
# three quarters of the window is spoiled.
#
# Measured on 27 flag captures from the board (1620 slice-seconds): 90% of all
# seconds sit within 1.75x of p25, the three captures with no movement in them
# produce zero refusals at 2.0x, and every disturbance runs 2-62x.
#
# These numbers are DEVICE-SPECIFIC and deliberately so. WESAD wrist BVP is a
# different, much noisier sensor whose own calm windows reach 51x, so it cannot
# be used to set them -- which is why only the fleet master (which sees device
# data) runs this stage, and the WESAD replay in anomaly.serve does not.
ENVELOPE_RATIO = 2.0
ENVELOPE_REF_Q = 25         # percentile of slice amplitude taken as "normal"
ENVELOPE_SEC = 1.0          # the window is chopped into slices this long


def artifact_mask(w: np.ndarray) -> np.ndarray:
    """per-sample: is this a transient rather than pulse?"""
    w = np.asarray(w, dtype=np.float64)
    d = np.diff(w, prepend=w[0])
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826          # ~sigma, outlier-proof
    if mad < MIN_MAD:
        return np.zeros(len(w), dtype=bool)            # flat: not a spike problem
    bad = np.abs(d - med) > SPIKE_Z * mad
    if not bad.any():
        return bad
    # a knock rings for a few samples on either side of the step that caught it
    idx = np.flatnonzero(bad)
    lo = np.maximum(idx - SPREAD, 0)
    hi = np.minimum(idx + SPREAD + 1, len(w))
    out = np.zeros(len(w), dtype=bool)
    for a, b in zip(lo, hi):
        out[a:b] = True
    return _bridge(w, out)


def _runs(mask: np.ndarray):
    """[(start, stop)] for each True run in the mask."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _bridge(w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask the plateau between two transients when it is displaced from the rest.

    this is what turns a sustained press from "two repaired edges around two
    seconds of nonsense" into one contiguous artifact the gate can refuse.
    """
    runs = _runs(mask)
    if len(runs) < 2:
        return mask
    good = w[~mask]
    if len(good) < 2:
        return mask
    med = np.median(good)
    mad = float(np.median(np.abs(good - med)) * 1.4826)
    if mad < MIN_MAD:
        return mask
    out = mask.copy()
    for (_, end), (start, _) in zip(runs, runs[1:]):
        if start - end <= 0 or start - end > BRIDGE_GAP:
            continue
        span = w[end:start]
        if abs(float(np.median(span)) - med) > BRIDGE_Z * mad:
            out[end:start] = True
    return out


def repair(w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """linear-interpolate across the masked runs.

    honest about what this is: for a short knock it restores a plausible stretch
    of pulse, which is why MAX_REPAIR keeps it to a tiny fraction of the window.
    over a long burst it would be fabricating data, and that window is gated.
    """
    w = np.asarray(w, dtype=np.float32).copy()
    if not mask.any():
        return w
    good = ~mask
    if good.sum() < 2:
        return w
    idx = np.arange(len(w))
    w[mask] = np.interp(idx[mask], idx[good], w[good])
    return w


def envelope(w: np.ndarray, fs: int = 64):
    """per-slice amplitude as a ratio to the window's median slice.

    a whole disturbed second is not repairable the way a 50 ms spike is:
    interpolating across it would delete a heartbeat and hand the model a flat
    stretch, which is its own kind of anomaly. so this only ever gates.
    """
    n = max(1, int(fs * ENVELOPE_SEC))
    k = len(w) // n
    if k < 4:
        return np.array([]), 0.0
    amp = np.array([np.std(w[i * n:(i + 1) * n]) for i in range(k)])
    ref = float(np.percentile(amp, ENVELOPE_REF_Q))
    if ref < MIN_MAD:
        return amp, 0.0
    return amp / ref, float(amp.max() / ref)


def clean_window(hist: np.ndarray, need: int, fs: int = 64) -> dict:
    """assemble `need` samples of clean pulse out of a longer history.

    Gating the whole window because one second of it was a tap meant a single
    knock blinded the detector for a full minute -- the bad second sits in the
    window until it slides out the far end. That is a terrible trade.

    The model wants 3840 samples; it does not want them CONTIGUOUS in wall
    clock. So the disturbed seconds are dropped and the window is back-filled
    from clean history that is a little older. A tap costs the few seconds it
    actually spoiled, not sixty, and monitoring never stops.

    The price is a stitch at each excision: two stretches of pulse joined
    mid-beat. Measured on a real capture that costs about as much as the
    quantisation noise already in the int8 model, and one join is a great deal
    cheaper than a minute of not looking.
    """
    hist = np.asarray(hist, dtype=np.float32)
    n = max(1, int(fs * ENVELOPE_SEC))
    need_secs = int(np.ceil(need / float(n)))
    k = len(hist) // n
    out = {"window": None, "quality": 0.0, "dropped": 0, "env": 0.0,
           "span_s": 0.0, "reason": ""}
    if k < need_secs:
        out["reason"] = "filling — %d of %d s" % (k, need_secs)
        return out

    amp = np.array([np.std(hist[i * n:(i + 1) * n]) for i in range(k)])
    ref = float(np.percentile(amp, ENVELOPE_REF_Q))
    if ref < MIN_MAD:
        out["reason"] = "flat — no pulse in this window"
        return out
    ratio = amp / ref
    good = ratio <= ENVELOPE_RATIO

    # newest first, so the window stays as recent as it can be
    picks = [i for i in range(k - 1, -1, -1) if good[i]][:need_secs]
    if len(picks) < need_secs:
        out["env"] = float(ratio.max())
        out["reason"] = ("movement — only %d of %d s usable in the last %d s"
                         % (len(picks), need_secs, k))
        return out

    picks.sort()
    w = np.concatenate([hist[i * n:(i + 1) * n] for i in picks])[:need]
    dropped = picks[-1] - picks[0] + 1 - need_secs

    # sample-level spikes small enough to interpolate across, inside what is left
    mask = artifact_mask(w)
    bad = float(mask.mean())
    if bad and bad <= MAX_REPAIR:
        w = repair(w, mask)
    out.update(window=w, dropped=dropped, span_s=(picks[-1] - picks[0] + 1),
               env=float(ratio.max()),
               quality=float(need_secs) / max(picks[-1] - picks[0] + 1, 1))
    return out


def assess(w: np.ndarray, fs: int = 64) -> dict:
    """look at one window and say whether the model should see it.

    returns quality (0-1), usable, the reason it is not, and the repaired
    window. quality is the fraction of samples that survived untouched, so it
    reads the same way in the dashboard as it does in a results table.
    """
    w = np.asarray(w, dtype=np.float32)
    n = len(w)
    if n == 0:
        return {"quality": 0.0, "usable": False, "reason": "empty",
                "bad_frac": 1.0, "window": w}

    med = np.median(w)
    mad = float(np.median(np.abs(w - med)) * 1.4826)
    if mad < MIN_MAD:
        return {"quality": 0.0, "usable": False, "reason": "flat — no pulse in this window",
                "bad_frac": 1.0, "window": w}

    mask = artifact_mask(w)
    bad = float(mask.mean())
    quality = 1.0 - bad

    if bad > MAX_REPAIR:
        return {"quality": quality, "usable": False, "env": 0.0,
                "reason": "motion artifact — %.1f%% of the window" % (100 * bad),
                "bad_frac": bad, "window": w}

    ratios, env = envelope(w, fs)
    loud = int((ratios > ENVELOPE_RATIO).sum()) if len(ratios) else 0
    if loud:
        return {"quality": min(quality, 1.0 - loud / max(len(ratios), 1)),
                "usable": False, "env": env, "bad_frac": bad,
                "reason": "movement — %d s disturbed, %.1fx normal amplitude"
                          % (loud, env),
                "window": w}

    return {"quality": quality, "usable": True, "reason": "", "env": env,
            "bad_frac": bad, "window": repair(w, mask) if bad else w}
