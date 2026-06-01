# swapping the demo for real sensors

how to plug a real MAX30102 (PPG/SpO₂) and MPU6050 (IMU) into the pipeline once
the hardware arrives. the pipeline was built so this is **one new file + one
changed line**.

## architecture

today:
```
demo_data.csv  →  DataReplay.stream()  →  Sample objects  →  pipeline  →  dashboard
```

after the swap:
```
MAX30102 + MPU6050  →  SensorSource.stream()  →  Sample objects  →  pipeline  →  dashboard
                       └────────── only new code ──────────┘
```

the classifier, FallDetector, vitals math, and dashboard don't change.

---

## step 1 — verify the sensors are wired right

wire each breakout to the Pi's GPIO (4 wires each; both share SDA/SCL):

| Pi 4B pin | Goes to |
|---|---|
| Pin 1 (3.3 V) | MAX30102 VIN + MPU6050 VCC |
| Pin 3 (SDA)  | MAX30102 SDA + MPU6050 SDA |
| Pin 5 (SCL)  | MAX30102 SCL + MPU6050 SCL |
| Pin 6 (GND)  | MAX30102 GND + MPU6050 GND |

then on the Pi:

```bash
sudo raspi-config        # → Interface Options → I2C → Enable → reboot
sudo apt install i2c-tools
sudo i2cdetect -y 1
```

expected: `0x57` (MAX30102) and `0x68` (MPU6050) show up in the grid. if not,
fix the wiring before moving on.

---

## step 2 — install the Python sensor libraries

```bash
pip3 install max30102 mpu6050-raspberrypi
```

---

## step 3 — quick sensor test (before touching the pipeline)

save this as `pipeline/sensor_test.py` and run it. cover the PPG with a finger;
IR should jump to 50,000+. wiggle the Pi; accel values should move.

```python
from max30102 import MAX30102
from mpu6050 import mpu6050
import time

ppg = MAX30102()
imu = mpu6050(0x68)

for _ in range(50):
    red, ir = ppg.read_sequential(amount=1)
    accel = imu.get_accel_data()
    print(f"IR={ir[-1]}  RED={red[-1]}  "
          f"ax={accel['x']:.2f} ay={accel['y']:.2f} az={accel['z']:.2f}")
    time.sleep(0.02)
```

if the numbers look sensible, the hardware is good.

---

## step 4 — create `pipeline/sensors.py`

the only new file. ~30 lines, mirroring `replay.py`:

```python
"""Real-sensor data source — drop-in replacement for DataReplay."""
import time
from typing import Iterator

from max30102 import MAX30102
from mpu6050 import mpu6050

from replay import FS, Sample

class SensorSource:
    """Reads MAX30102 + MPU6050 at FS Hz and yields Sample objects."""

    def __init__(self, fs: int = FS):
        self.fs = fs
        self.ppg = MAX30102()
        self.imu = mpu6050(0x68)
        self._period = 1.0 / fs

    def stream(self) -> Iterator[Sample]:
        idx = 0
        next_tick = time.perf_counter()
        while True:
            red, ir = self.ppg.read_sequential(amount=1)
            accel = self.imu.get_accel_data()
            yield Sample(
                sample_idx=idx,
                ir=float(ir[-1]),
                red=float(red[-1]),
                accel_x=float(accel["x"]),
                accel_y=float(accel["y"]),
                accel_z=float(accel["z"]),
                true_label="unknown",
            )
            idx += 1
            next_tick += self._period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
```

---

## step 5 — add a synthetic/real switch to `server.py`

open `pipeline/server.py` and find this line in `Engine.__init__`:

```python
self.stream = DataReplay(loop=True).stream(realtime=False)
```

replace it with:

```python
if os.environ.get("REAL_SENSORS") == "1":
    from sensors import SensorSource
    self.stream = SensorSource().stream()          # real, self-paced at 50 Hz
else:
    self.stream = DataReplay(loop=True).stream(realtime=False)
```

(`os` is already imported at the top of `server.py`.) that's the whole change
to the dashboard.

> on pacing: the replay source is pulled with `realtime=False` because the
> server's async producer sets the cadence. a **real** sensor source paces
> itself — it can only emit samples as fast as the hardware produces them — so
> it doesn't need the flag; the producer just reads whatever is ready.

---

## step 6 — run it

```bash
# synthetic demo (unchanged)
python3 pipeline/server.py

# real sensors on the Pi
REAL_SENSORS=1 python3 pipeline/server.py
```

---

## verification checklist

in real-sensor mode, with the MAX30102 firmly on a finger:

- [ ] within 4 s the **heart rate** tile shows a BPM matching your wrist pulse (count beats in 15 s × 4)
- [ ] within 10 s the **prediction** shows `Cardiac` with high confidence
- [ ] lift your finger off → within 5 s it shifts to `Non-Cardiac`
- [ ] squeeze the sensor area to restrict blood flow → it shifts toward `Occlusion`
- [ ] tilt the Pi → accel X/Y/Z move live in the All Sensors tab
- [ ] don't actually drop the Pi — but a hard tap on the IMU shouldn't fire a false fall (impacts without a free-fall phase don't trigger the detector)

if all six work, the synthetic → real transition is done. same pipeline, same
model, same dashboard.

## what changes in behavior

| Synthetic mode | Real-sensor mode |
|---|---|
| predictions ✓/✗ vs known ground truth | "true label" shows `—` (no ground truth in the real world) |
| demo loops every 92 seconds | runs continuously as long as the sensor produces data |
| fall event fires once per loop (injected) | fall event only fires if you actually simulate one |
| HR / SpO₂ vary predictably per segment | HR / SpO₂ vary with your real physiology |

everything else (charts, inspector, fall state machine, latency) works the same.

## common gotchas

| Symptom | Fix |
|---|---|
| `Permission denied: '/dev/i2c-1'` | `sudo usermod -aG i2c $USER`, then log out and back in |
| `i2cdetect` shows nothing | wiring — re-seat jumpers, check pin numbering (Pin 1 ≠ Pin 2) |
| MAX30102 reads zeros / very low | LED current too low; after init, call `ppg.set_config(led_pa_red=0x1F, led_pa_ir=0x1F)` |
| predictions stuck on `non_cardiac` | finger not firmly on the sensor, or LED current too low. IR should read 50,000+ on skin contact |
| timing drifts at higher rates | naive `time.sleep` accumulates jitter. for production, use the MAX30102 FIFO + interrupt instead of polling |
| `ModuleNotFoundError: max30102` | `pip3 install max30102 mpu6050-raspberrypi` |

## files that DON'T change

to show how isolated the swap is:

| File | Changes? |
|---|---|
| `baselines/runs/.../model.keras` | no |
| `baselines/inference_lib.py` | no |
| `pipeline/pipeline.py` | no |
| `pipeline/vitals.py` | no |
| `pipeline/fall_detector.py` | no |
| `pipeline/replay.py` | no (kept for synthetic mode) |
| `pipeline/static/` | no (browser UI is source-agnostic) |
| `pipeline/server.py` | one line in `Engine.__init__` — see step 5 |
| `pipeline/sensors.py` | **new file** — see step 4 |

that's the entire scope of the synthetic → real swap.
