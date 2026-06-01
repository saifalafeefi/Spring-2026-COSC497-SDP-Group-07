"""Pipeline orchestrator: data source → buffer → classifier → events.

The pipeline consumes a stream of Samples from any data source (replay or,
eventually, a real sensor) and emits PredictionEvent objects whenever a fresh
inference is ready.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

from inference_lib import Classifier, ClassificationResult  # noqa: E402
from replay import FS, Sample  # noqa: E402

WINDOW_LEN = 512                 # samples per inference (10.24 s @ 50 Hz)
INFERENCE_EVERY_N_SAMPLES = 250  # new prediction every 5 s (50 Hz × 5)


@dataclass
class PredictionEvent:
    sample_idx: int                  # index of the most recent sample
    window: np.ndarray               # the 512 samples scored
    result: ClassificationResult     # classifier output
    true_label: str | None = None    # if simulated, what the data was labelled


def run_pipeline(stream: Iterable[Sample],
                 classifier: Classifier | None = None,
                 window_len: int = WINDOW_LEN,
                 stride: int = INFERENCE_EVERY_N_SAMPLES,
                 ) -> Iterator[PredictionEvent]:
    """Consume the sample stream, emit a PredictionEvent every `stride` samples
    once the buffer has filled to `window_len` samples."""
    clf = classifier or Classifier()
    buffer: deque[float] = deque(maxlen=window_len)
    label_buffer: deque[str] = deque(maxlen=window_len)
    samples_since_last = 0
    for s in stream:
        buffer.append(s.value)
        label_buffer.append(s.true_label)
        samples_since_last += 1
        if len(buffer) < window_len:
            continue
        if samples_since_last < stride:
            continue
        samples_since_last = 0
        window = np.array(buffer, dtype=np.float32)
        result = clf.classify(window)
        # majority-vote on the window's per-sample labels for "ground truth"
        labels = list(label_buffer)
        majority = max(set(labels), key=labels.count) if labels else None
        yield PredictionEvent(
            sample_idx=s.sample_idx,
            window=window,
            result=result,
            true_label=majority,
        )
