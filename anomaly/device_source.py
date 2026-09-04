"""real-sensor BVP source — the live counterpart to `wesad_replay.BVPReplay`.

reads the ESP32-S3 + MAX30102 stream produced by `sketch_aug3a/sketch_aug3a.ino`
and turns it into the same thing the dashboard already consumes: one
`(bvp_value, label)` per `next()`, at `anomaly.wesad.FS` (64 Hz).

the sketch emits raw IR/RED counts at ~100 Hz on a DC pedestal of 50k–200k. that
is not what the model or the charts want, so this does four things on the way:

  1. **parse + timestamp** — the device stamps each sample with its own capture
     time (back-corrected for FIFO backlog), so bursty arrival doesn't distort
     the timebase.
  2. **resample to 64 Hz** — linear interpolation onto a fixed grid driven by
     those device timestamps. the model input is a fixed 3,840-sample (60 s @
     64 Hz) window, so this is not optional. driving the grid off device time
     rather than a declared rate makes this independent of the sketch's
     configured sample rate.
  3. **band-pass 0.7–3 Hz** — causal `sosfilt` with carried state. strips the DC
     pedestal and baseline wander, leaving an AC pulse waveform in the same
     shape family as WESAD BVP. the filter is primed at the current DC level so
     it starts settled instead of ringing for ten seconds.
  4. **contact detection** — IR DC below `contact_threshold` means the finger is
     off the sensor. samples still flow (the timebase must stay continuous); the
     flag is reported so the UI can withhold judgement.

the reader runs on a background thread and reconnects on its own, so unplugging
the board degrades the dashboard instead of killing it.

    python3 -m anomaly.device_source --list-ports
    python3 -m anomaly.device_source                      # live self-test, no TF
    python3 -m anomaly.device_source --port /dev/cu.usbmodem101
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Iterator

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .wesad import FS

DEFAULT_BAUD = 115200
# Informational only. The resampler is driven entirely by the device's own
# timestamps, so it is rate-agnostic: the sketch moved from 25 Hz to 100 Hz with
# no change here. Verified at 10 / 25 / 100 Hz device rates.
DEVICE_FS_NOMINAL = 100
CONTACT_IR = 50_000         # same skin-contact threshold the sketch displays
GAP_MS = 500.0              # timestamp jump this large = discontinuity, re-prime
BUFFER_SEC = 10             # bound the hand-off queue; drop oldest past this

# USB-serial bridges these boards ship with — used for auto-discovery.
_KNOWN_VIDS = {
    0x303A,   # Espressif native USB-CDC (ESP32-S3)
    0x10C4,   # Silicon Labs CP210x
    0x1A86,   # WCH CH340 / CH9102
    0x0403,   # FTDI
}
_PORT_HINTS = ("usbmodem", "usbserial", "slab", "ch34", "wchusb", "ttyusb", "ttyacm")


def list_ports() -> list:
    """every serial port we can see, likely candidates first."""
    from serial.tools import list_ports as lp
    ports = list(lp.comports())
    ports.sort(key=lambda p: (0 if _looks_like_board(p) else 1, p.device))
    return ports


def _looks_like_board(p) -> bool:
    if getattr(p, "vid", None) in _KNOWN_VIDS:
        return True
    blob = f"{p.device} {p.description or ''}".lower()
    return any(h in blob for h in _PORT_HINTS)


def find_port() -> str | None:
    """best guess at the board's port, or None if nothing plausible is plugged in."""
    for p in list_ports():
        if _looks_like_board(p):
            return p.device
    return None


class DeviceSource:
    """live MAX30102 stream, shaped like `BVPReplay` so `serve.py` can swap them.

    `stream()` never blocks: it yields only what the reader thread has already
    conditioned. call `available()` first and take that many — the server's
    producer does exactly that, which is what keeps a slow or absent device from
    stalling the async loop.
    """

    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD,
                 fs: int = FS, invert: bool = False,
                 band: tuple[float, float] = (0.7, 3.0),
                 contact_threshold: int = CONTACT_IR):
        self.port_hint = port
        self.baud = baud
        self.fs = fs
        self.invert = invert
        self.contact_threshold = contact_threshold
        self.subject = "device"          # what the UI shows instead of "S5"

        self._sos = butter(2, list(band), btype="band", fs=fs, output="sos")
        self._sos_zi = sosfilt_zi(self._sos)
        self._zi = None                  # primed on the first sample / after a gap

        self._out: deque = deque(maxlen=fs * BUFFER_SEC)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ser = None

        # resampler state (device milliseconds)
        self._t_prev = None
        self._ir_prev = 0.0
        self._red_prev = 0.0
        self._t_grid = None
        self._step_ms = 1000.0 / fs

        # telemetry the dashboard surfaces
        self.connected = False
        self.port = None
        self.contact = False
        self.ir_dc = 0.0
        self.red_dc = 0.0
        self.device_bpm = None
        self.device_spo2 = None
        self.raw_count = 0
        self.out_count = 0
        self.dropped = 0
        self.last_sample_at = 0.0
        self.last_error = None
        self._tx_bpm = None      # host HR waiting to be sent to the board

    # ---------- lifecycle ----------

    def start(self) -> "DeviceSource":
        try:
            import serial  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyserial is required for --source device: pip3 install pyserial"
            ) from e
        if self._thread is None:
            self._thread = threading.Thread(target=self._reader, name="ppg-serial",
                                            daemon=True)
            self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_serial()

    # ---------- server-facing API ----------

    def available(self) -> int:
        with self._lock:
            return len(self._out)

    def stream(self) -> Iterator[tuple[float, int]]:
        """yields (bvp, label). label is always 0 ("—"): real capture has no
        ground truth, unlike the WESAD replay.

        when the buffer is dry it yields (None, 0) rather than returning — a
        generator that returns is exhausted for good, and one empty tick (board
        unplugged, finger off, host faster than the sensor) would permanently
        kill the source. the caller stops drawing for this tick and comes back.
        """
        while True:
            with self._lock:
                v = self._out.popleft() if self._out else None
            yield (None, 0) if v is None else (v, 0)

    def status(self) -> dict:
        stale = (time.monotonic() - self.last_sample_at) if self.last_sample_at else None
        return {"connected": self.connected and (stale is not None and stale < 2.0),
                "port": self.port, "contact": self.contact,
                "ir_dc": round(self.ir_dc, 1),
                "bpm": self.device_bpm, "spo2": self.device_spo2,
                "raw": self.raw_count, "out": self.out_count,
                "dropped": self.dropped, "buffered": self.available(),
                "error": self.last_error}

    def send_bpm(self, bpm) -> None:
        """queue the host's heart rate for the board's display.

        the write happens on the reader thread rather than here, so the port is
        only ever touched from one thread. the value is a single slot, not a
        queue: if the reader is busy the newest reading simply replaces the
        older one, which is what a live display wants anyway.
        """
        with self._lock:
            self._tx_bpm = int(round(bpm)) if bpm else None

    def _flush_tx(self) -> None:
        with self._lock:
            bpm, self._tx_bpm = self._tx_bpm, None
        if bpm is None or self._ser is None:
            return
        try:
            self._ser.write(("H,%d\n" % bpm).encode("ascii"))
        except Exception as e:                # a failed display update is not fatal
            self.last_error = str(e)

    # ---------- serial ----------

    def _open_serial(self):
        import serial
        port = self.port_hint or find_port()
        if port is None:
            raise OSError("no serial port found — plug the board in, or pass --device-port "
                          "(see: python3 -m anomaly.device_source --list-ports)")
        self._ser = serial.Serial(port, self.baud, timeout=1.0)
        self.port = port
        self.connected = True
        self.last_error = None

    def _close_serial(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.connected = False

    def _reader(self):
        import serial
        while not self._stop.is_set():
            try:
                if self._ser is None:
                    self._open_serial()
                raw = self._ser.readline()
                if not raw:
                    continue                      # read timeout — board idle
                self._handle(raw.decode("ascii", "ignore").strip())
                self._flush_tx()
            except (serial.SerialException, OSError) as e:
                self.last_error = str(e)
                self._close_serial()
                self._reset_timebase()
                self._stop.wait(1.0)              # board unplugged — retry, don't die
            except Exception as e:                # a malformed line must not kill the thread
                self.last_error = str(e)

    def _handle(self, line: str):
        if not line or line.startswith("#"):
            return
        parts = line.split(",")
        if parts[0] == "D" and len(parts) >= 4:
            try:
                t, ir, red = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                return
            self._on_sample(t, ir, red)
        elif parts[0] == "V" and len(parts) >= 6:
            try:
                hr, hr_ok, spo2, spo2_ok = (int(parts[2]), int(parts[3]),
                                            int(parts[4]), int(parts[5]))
            except ValueError:
                return
            # keep only what the device itself marked valid + physiologically sane
            self.device_bpm = hr if (hr_ok and 25 < hr < 240) else None
            self.device_spo2 = spo2 if (spo2_ok and 70 <= spo2 <= 100) else None

    # ---------- conditioning ----------

    def _reset_timebase(self):
        self._t_prev = None
        self._t_grid = None
        self._zi = None

    def _prime_filter(self, dc: float):
        """start the filter in steady state for a constant input at `dc`.

        without this the 0.7 Hz high-pass sees a step from 0 to ~90,000 counts on
        the first sample and rings for several seconds — long enough to swamp the
        pulse and poison the first inference window.
        """
        self._zi = self._sos_zi * dc

    def _on_sample(self, t_ms: float, ir: float, red: float):
        self.raw_count += 1
        self.last_sample_at = time.monotonic()

        # DC tracking → skin contact. slow EMA (~1 s at 25 Hz) so a single dark
        # sample doesn't toggle the flag.
        a = 0.04          # DC time constant ~25 samples (0.25 s at 100 Hz)
        self.ir_dc = ir if self.raw_count == 1 else (1 - a) * self.ir_dc + a * ir
        self.red_dc = red if self.raw_count == 1 else (1 - a) * self.red_dc + a * red
        self.contact = self.ir_dc > self.contact_threshold

        # first sample, or the stream jumped (reconnect / millis() rollover)
        if (self._t_prev is None or t_ms < self._t_prev
                or t_ms - self._t_prev > GAP_MS):
            self._t_prev, self._ir_prev, self._red_prev = t_ms, ir, red
            self._t_grid = t_ms
            self._prime_filter(ir)
            return

        if t_ms == self._t_prev:                 # duplicate stamp — nothing to span
            self._ir_prev, self._red_prev = ir, red
            return

        # walk the 64 Hz grid across the interval this raw sample just closed
        span = t_ms - self._t_prev
        vals = []
        while self._t_grid <= t_ms:
            frac = (self._t_grid - self._t_prev) / span
            vals.append(self._ir_prev + frac * (ir - self._ir_prev))
            self._t_grid += self._step_ms

        self._t_prev, self._ir_prev, self._red_prev = t_ms, ir, red
        if not vals:
            return

        if self._zi is None:
            self._prime_filter(vals[0])
        y, self._zi = sosfilt(self._sos, np.asarray(vals, dtype=np.float64), zi=self._zi)
        if self.invert:
            y = -y

        with self._lock:
            room = self._out.maxlen - len(self._out)
            if len(y) > room:
                self.dropped += len(y) - room     # consumer fell behind
            self._out.extend(float(v) for v in y)
        self.out_count += len(y)


def _self_test(args):
    """live sanity check with no TensorFlow and no dashboard in the way."""
    src = DeviceSource(port=args.port, baud=args.baud, invert=args.invert).start()
    print(f"  listening on {args.port or 'auto-detected port'} @ {args.baud} baud")
    print("  put a fingertip on the sensor — Ctrl-C to stop\n")
    stream = src.stream()
    recent: deque = deque(maxlen=FS * 4)
    t0 = time.monotonic()
    try:
        while True:
            time.sleep(0.25)
            for _ in range(src.available()):
                v = next(stream)[0]
                if v is None:
                    break
                recent.append(v)
            s = src.status()
            el = time.monotonic() - t0
            amp = (max(recent) - min(recent)) if len(recent) > 1 else 0.0
            eff = s["raw"] / el if el > 0 else 0.0
            state = "connected" if s["connected"] else f"waiting ({s['error'] or '…'})"
            print(f"\r  {state:<34} port={s['port'] or '—'}  "
                  f"raw={s['raw']:>6} ({eff:4.1f} Hz)  out={s['out']:>6}  "
                  f"contact={'yes' if s['contact'] else 'no ':<3}  "
                  f"IR_dc={s['ir_dc']:>9.0f}  amp={amp:>7.1f}  "
                  f"dev_bpm={s['bpm'] or '—':<4} spo2={s['spo2'] or '—':<4}", end="",
                  flush=True)
    except KeyboardInterrupt:
        print("\n\n  stopped.")
    finally:
        src.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MAX30102 serial source self-test")
    ap.add_argument("--port", default=None, help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--invert", action="store_true",
                    help="flip the waveform if the pulse reads upside-down")
    ap.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list_ports:
        ports = list_ports()
        if not ports:
            print("  no serial ports found.")
            return
        print("  serial ports (likely board first):\n")
        for p in ports:
            mark = "→" if _looks_like_board(p) else " "
            vid = f"{p.vid:04X}" if getattr(p, "vid", None) else "----"
            print(f"  {mark} {p.device:<28} vid={vid}  {p.description or ''}")
        return

    _self_test(args)


if __name__ == "__main__":
    main()
