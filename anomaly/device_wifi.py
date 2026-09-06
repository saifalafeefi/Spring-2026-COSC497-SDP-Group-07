"""set the board's WiFi credentials over USB, and read back its IP address.

the ESP32 keeps the SSID and password in NVS, so they survive a reflash and never
sit in source control. this just types the command down the same serial link the
dashboard uses:

    python3 -m anomaly.device_wifi --ssid MyNetwork --password hunter2
    python3 -m anomaly.device_wifi --status        # what is it connected to?
    python3 -m anomaly.device_wifi --forget

stop the dashboard first -- only one process can hold the port.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

from .device_source import DEFAULT_BAUD, find_port


def _open(port: str | None, baud: int):
    import serial
    port = port or find_port()
    if port is None:
        raise SystemExit("  no board found -- plug it in, or pass --port "
                         "(list them: python3 -m anomaly.device_source --list-ports)")
    return serial.Serial(port, baud, timeout=1.0), port


def _drain(ser, seconds: float = 6.0, want: str = "") -> list:
    """collect the board's '#' log lines for a moment; stop early on `want`."""
    out, t0 = [], time.time()
    while time.time() - t0 < seconds:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("ascii", "ignore").strip()
        if line.startswith("#"):
            out.append(line)
            print("   ", line)
            if want and want in line:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ssid")
    ap.add_argument("--password", help="omit to be prompted without echoing")
    ap.add_argument("--status", action="store_true", help="report SSID and IP")
    ap.add_argument("--forget", action="store_true", help="clear the stored network")
    ap.add_argument("--port", default=os.environ.get("DEVICE_PORT"))
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = ap.parse_args()

    if not (args.status or args.forget or args.ssid):
        ap.error("give --ssid, or --status, or --forget")

    ser, port = _open(args.port, args.baud)
    print(f"\n  board on {port}")
    try:
        if args.status:
            ser.write(b"W?\n")
            _drain(ser, 3.0, want="ssid=")
        elif args.forget:
            ser.write(b"W!\n")
            _drain(ser, 3.0, want="forgotten")
        else:
            pw = args.password
            if pw is None:
                pw = getpass.getpass("  password (not echoed): ")
            if "," in args.ssid:
                raise SystemExit("  the SSID may not contain a comma "
                                 "(the board splits the command on it)")
            # Opening the port RESETS the board. It then reboots and reconnects
            # using whatever was already stored, printing that network's IP. An
            # earlier version wrote immediately and stopped at the first "ip="
            # line, so the command could land in a booting board and the OLD
            # network's address was reported as if it were the new one.
            print("  waiting for the board to finish booting…")
            _drain(ser, 15.0, want="setup done")
            ser.reset_input_buffer()

            ser.write(f"W,{args.ssid},{pw}\n".encode("ascii"))
            print(f"  sent credentials for {args.ssid!r}")
            if not _drain(ser, 5.0, want="wifi saved"):
                print("\n  the board did not acknowledge the command. is the")
                print("  dashboard still running and holding the port?")
                return 1

            print("  connecting…")
            log = _drain(ser, 25.0, want="wifi connected")
            if not any("wifi connected" in l for l in log):
                print("\n  no IP. check the SSID and password, and that the network")
                print("  is 2.4 GHz -- the ESP32 cannot join a 5 GHz-only SSID.")
                return 1
    finally:
        ser.close()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
