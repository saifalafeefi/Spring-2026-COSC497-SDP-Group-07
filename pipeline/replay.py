"""Real-time multi-channel sensor stream for the pipeline.

`DataReplay` yields one `Sample` per tick at the real sensor cadence
(50 Hz by default). Each `Sample` carries all sensor channels:
  - ir, red               PPG IR + RED LED values (raw ADC counts)
  - accel_x, accel_y, accel_z   accelerometer in 'g' units
  - true_label            ground-truth class (only available in simulation)

When the hardware arrives, swap this module for one that reads from the
MAX30102 + MPU6050 — the rest of the pipeline doesn't change.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_CSV = os.path.join(HERE, "demo_data.csv")
FS = 50  # Hz — sampling rate everything downstream assumes


@dataclass
class Sample:
    sample_idx: int
    ir: float
    red: float
    accel_x: float
    accel_y: float
    accel_z: float
    true_label: str

    @property
    def accel_magnitude(self) -> float:
        return (self.accel_x ** 2 + self.accel_y ** 2 + self.accel_z ** 2) ** 0.5


class DataReplay:
    """Loops a CSV indefinitely, yielding samples at real-time speed."""

    def __init__(self, csv_path: str | None = None, fs: int = FS, loop: bool = True):
        self.csv_path = csv_path or DEMO_CSV
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"{self.csv_path} not found. Generate it with "
                f"`python3 pipeline/make_demo_data.py` first.")
        df = pd.read_csv(self.csv_path)
        self._ir = df["ir"].to_numpy(dtype=float)
        self._red = df["red"].to_numpy(dtype=float)
        self._ax = df["accel_x"].to_numpy(dtype=float)
        self._ay = df["accel_y"].to_numpy(dtype=float)
        self._az = df["accel_z"].to_numpy(dtype=float)
        self._labels = df["true_label"].to_numpy(dtype=str)
        self._n = len(self._ir)
        self.fs = fs
        self.loop = loop
        self._sample_period = 1.0 / fs

    def stream(self, realtime: bool = True) -> Iterator[Sample]:
        """yield samples one at a time.

        realtime=True  : block (time.sleep) to pace at the sensor cadence —
                         used by the CLI tools that have no other clock.
        realtime=False : yield as fast as the caller pulls, no sleeping — used
                         by the dashboard server (server.py), where the async
                         producer loop paces ingestion on its own timer.
        """
        idx = 0
        next_time = time.perf_counter()
        while True:
            pos = idx % self._n
            yield Sample(
                sample_idx=idx,
                ir=float(self._ir[pos]),
                red=float(self._red[pos]),
                accel_x=float(self._ax[pos]),
                accel_y=float(self._ay[pos]),
                accel_z=float(self._az[pos]),
                true_label=str(self._labels[pos]),
            )
            idx += 1
            if not self.loop and idx >= self._n:
                return
            if realtime:
                next_time += self._sample_period
                sleep_for = next_time - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_time = time.perf_counter()
