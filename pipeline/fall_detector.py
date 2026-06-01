"""Fall-detection state machine.

Sample-by-sample 4-phase detector that runs on accelerometer data only.
Same algorithm we'll port to C for the ESP32 firmware — the Python class
is the reference implementation.

Phases:

    1. Free-fall    – |a| drops below FF_THRESHOLD_G for FF_MIN_MS
    2. Impact       – |a| spikes above IMPACT_THRESHOLD_G within
                      IMPACT_WINDOW_MS of leaving free-fall
    3. Stillness    – low magnitude variance for STILLNESS_MS after impact
    → Fall confirmed; emits an event

If any phase times out / doesn't match (e.g. running has impact spikes
without prior free-fall, or post-impact the person keeps moving), the
detector returns to the Normal state without firing.

C port notes
------------
The state machine compiles directly to a switch-on-state in C. Replace
`collections.deque` with a fixed-size ring buffer; everything else is
plain float math suitable for the ESP32-S3 FPU. Sample rate, thresholds,
and durations are constructor params — keep them as `#define`s on the MCU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --- State labels (also used directly in the dashboard) ---
STATE_NORMAL          = "Normal"
STATE_FREE_FALL       = "Free-fall"
STATE_AWAIT_IMPACT    = "Awaiting impact"
STATE_AWAIT_STILLNESS = "Checking stillness"
STATE_FALL_CONFIRMED  = "FALL"

STATES = [STATE_NORMAL, STATE_FREE_FALL, STATE_AWAIT_IMPACT,
          STATE_AWAIT_STILLNESS, STATE_FALL_CONFIRMED]


@dataclass
class FallEvent:
    sample_idx: int
    t_s: float


class FallDetector:
    """4-phase fall detector.

    Args:
        fs: sampling rate in Hz (50 Hz for our system).
        ff_threshold_g: |a| below this counts as free-fall (default 0.5 g).
        ff_min_ms: how long |a| must stay below threshold (default 120 ms).
        impact_threshold_g: |a| above this counts as impact (default 2.5 g).
        impact_window_ms: max time after free-fall to look for impact
            (default 500 ms).
        stillness_std_g: max std of |a| to count as still (default 0.06 g).
        stillness_ms: how long stillness must be sustained (default 1200 ms).
        recovery_ms: cooldown after a confirmed fall before detecting again
            (default 3000 ms).
    """

    def __init__(
        self,
        fs: int = 50,
        ff_threshold_g: float = 0.5,
        ff_min_ms: int = 120,
        impact_threshold_g: float = 2.5,
        impact_window_ms: int = 500,
        stillness_std_g: float = 0.06,
        stillness_ms: int = 1200,
        recovery_ms: int = 3000,
    ):
        self.fs = fs
        self.ff_thresh = ff_threshold_g
        self.impact_thresh = impact_threshold_g
        self.stillness_std = stillness_std_g

        self.ff_min_samples = max(1, int(fs * ff_min_ms / 1000))
        self.impact_window_samples = max(1, int(fs * impact_window_ms / 1000))
        self.stillness_samples = max(1, int(fs * stillness_ms / 1000))
        self.recovery_samples = max(1, int(fs * recovery_ms / 1000))

        self.state: str = STATE_NORMAL
        self.events: list[FallEvent] = []
        self.sample_idx: int = 0

        # phase tracking
        self._ff_streak = 0
        self._t_state_entered = 0
        # post-impact stillness collection
        self._still_buf: list[float] = []
        self._stillness_timeout_samples = self.stillness_samples * 3

    # ------------------------------------------------------------------

    def update(self, ax: float, ay: float, az: float) -> str:
        """Feed one accelerometer sample (g-units). Returns current state."""
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        self.sample_idx += 1

        # ----- state-by-state transitions -----

        if self.state == STATE_NORMAL:
            if mag < self.ff_thresh:
                self._ff_streak += 1
                if self._ff_streak >= self.ff_min_samples:
                    self._enter(STATE_FREE_FALL)
            else:
                self._ff_streak = 0

        elif self.state == STATE_FREE_FALL:
            if mag >= self.ff_thresh:
                # left free-fall — look for the impact next
                self._enter(STATE_AWAIT_IMPACT)
                # current sample might already be the impact:
                if mag > self.impact_thresh:
                    self._enter(STATE_AWAIT_STILLNESS)

        elif self.state == STATE_AWAIT_IMPACT:
            elapsed = self.sample_idx - self._t_state_entered
            if mag > self.impact_thresh:
                self._enter(STATE_AWAIT_STILLNESS)
            elif elapsed > self.impact_window_samples:
                # free-fall without impact — false alarm
                self._enter(STATE_NORMAL)

        elif self.state == STATE_AWAIT_STILLNESS:
            # Only start collecting "stillness" samples once the impact spike
            # has subsided (otherwise the spike inflates std).
            if mag < self.impact_thresh * 0.8:
                self._still_buf.append(mag)
                if len(self._still_buf) >= self.stillness_samples:
                    std = float(np.std(self._still_buf))
                    if std < self.stillness_std:
                        # all four phases passed → confirmed
                        self.events.append(FallEvent(
                            sample_idx=self.sample_idx,
                            t_s=self.sample_idx / self.fs,
                        ))
                        self._enter(STATE_FALL_CONFIRMED)
                    else:
                        # impact but the person kept moving — not a fall
                        self._enter(STATE_NORMAL)
            # safety timeout
            elapsed = self.sample_idx - self._t_state_entered
            if elapsed > self._stillness_timeout_samples:
                self._enter(STATE_NORMAL)

        elif self.state == STATE_FALL_CONFIRMED:
            # stay in this state for the recovery window, then auto-reset
            elapsed = self.sample_idx - self._t_state_entered
            if elapsed > self.recovery_samples:
                self._enter(STATE_NORMAL)

        return self.state

    # ------------------------------------------------------------------

    def update_batch(self, accel_xyz: np.ndarray) -> str:
        """Convenience: feed a batch of N samples shaped (N, 3). Returns
        the state after the last sample."""
        for ax, ay, az in accel_xyz:
            self.update(float(ax), float(ay), float(az))
        return self.state

    def _enter(self, new_state: str) -> None:
        self.state = new_state
        self._t_state_entered = self.sample_idx
        if new_state == STATE_NORMAL:
            self._ff_streak = 0
            self._still_buf.clear()
        elif new_state == STATE_AWAIT_STILLNESS:
            self._still_buf.clear()


# ----- self-test --------------------------------------------------------

def _selftest():
    """Quick sanity check: a synthesized fall trace should trigger; a
    running-style trace should not."""
    fs = 50

    # Fall trace: 2s normal → 200ms free-fall → 100ms impact → 2s stillness
    def fall_trace():
        rng = np.random.default_rng(0)
        n_normal = fs * 2
        n_ff = int(fs * 0.2)
        n_impact = int(fs * 0.1)
        n_still = fs * 2
        normal = np.column_stack([np.zeros(n_normal), np.zeros(n_normal),
                                  np.ones(n_normal)]) + rng.normal(0, 0.01, (n_normal, 3))
        ff = rng.normal(0, 0.05, (n_ff, 3))  # near zero — falling
        impact = np.column_stack([np.zeros(n_impact), np.zeros(n_impact),
                                  np.full(n_impact, 4.0)]) + rng.normal(0, 0.05, (n_impact, 3))
        still = np.column_stack([np.zeros(n_still), np.zeros(n_still),
                                 np.ones(n_still)]) + rng.normal(0, 0.005, (n_still, 3))
        return np.vstack([normal, ff, impact, still])

    # Running trace: periodic spikes, never below 0.5g
    def running_trace():
        rng = np.random.default_rng(1)
        t = np.arange(fs * 5) / fs
        z = 1.0 + 1.5 * np.sin(2 * np.pi * 2.5 * t)  # 2.5 Hz cadence, +/- 1.5g
        x = 0.3 * np.sin(2 * np.pi * 2.5 * t + 0.3)
        y = 0.2 * np.sin(2 * np.pi * 2.5 * t + 0.7)
        return np.column_stack([x, y, z]) + rng.normal(0, 0.05, (len(t), 3))

    print("fall trace -> ", end="")
    det = FallDetector(fs=fs)
    det.update_batch(fall_trace())
    print(f"{len(det.events)} fall events  (expected: 1)")

    print("running trace -> ", end="")
    det = FallDetector(fs=fs)
    det.update_batch(running_trace())
    print(f"{len(det.events)} fall events  (expected: 0)")


if __name__ == "__main__":
    _selftest()
