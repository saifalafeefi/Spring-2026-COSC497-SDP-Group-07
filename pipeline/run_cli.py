"""Run the pipeline in the terminal — no dashboard.

Use this to confirm the data source + classifier glue is working before
firing up the web dashboard (server.py).

    python3 pipeline/run_cli.py             # loops forever
    python3 pipeline/run_cli.py --once      # one pass through demo_data.csv
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

from pipeline import run_pipeline
from replay import DataReplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="play demo_data.csv once instead of looping forever")
    args = ap.parse_args()

    replay = DataReplay(loop=not args.once)
    print(f"Streaming {replay.csv_path} at {replay.fs} Hz "
          f"({'loop' if replay.loop else 'one-shot'})")
    print("Inference every 5 s after the 10.24 s buffer fills.\n")

    for ev in run_pipeline(replay.stream()):
        true = f"true={ev.true_label}" if ev.true_label else ""
        ok = " ✓" if ev.true_label == ev.result.label else " ✗" if ev.true_label else ""
        probs = "  ".join(f"{k}={v:.2f}" for k, v in ev.result.probabilities.items())
        t_s = ev.sample_idx / replay.fs
        print(f"[t={t_s:6.1f}s] predicted={ev.result.label:12s} "
              f"conf={ev.result.confidence:.2f}  {true}{ok}  ({probs})")


if __name__ == "__main__":
    main()
