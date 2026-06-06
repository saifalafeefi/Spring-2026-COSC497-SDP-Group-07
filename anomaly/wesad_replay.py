"""replay a curated calm→stress BVP loop for the live dashboard.

instead of streaming the raw 87-min recording (which opens with a long noisy
prep segment and makes the demo unreadable), this stitches a chunk of the
subject's real **baseline** onto a chunk of their real **stress (TSST)** and
loops it. so the dashboard opens in clearly-labelled calm, then transitions into
clearly-labelled stress, on a short cycle.

yields one (bvp_value, true_label) per call, non-blocking — the server's async
loop sets the pace.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .wesad import load_subject, FS


class BVPReplay:
    def __init__(self, subject: str = "S5", loop: bool = True, seg_sec: int = 120):
        d = load_subject(subject)
        bvp, lab = d["bvp"], d["labels"]
        n = seg_sec * FS
        base = bvp[lab == 1][:n]          # contiguous baseline block
        stress = bvp[lab == 2][:n]        # contiguous stress block
        self.bvp = np.concatenate([base, stress]).astype(np.float32)
        self.labels = np.concatenate([
            np.ones(len(base), np.int8), np.full(len(stress), 2, np.int8)])
        self.n = len(self.bvp)
        self.fs = FS
        self.loop = loop
        self.subject = subject

    def stream(self) -> Iterator[tuple[float, int]]:
        idx = 0
        while True:
            pos = idx % self.n
            yield float(self.bvp[pos]), int(self.labels[pos])
            idx += 1
            if not self.loop and idx >= self.n:
                return
