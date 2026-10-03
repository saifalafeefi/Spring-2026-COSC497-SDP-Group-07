"""set the board's WiFi credentials over USB, and read back its IP address.

the ESP32 keeps the SSID and password in NVS, so they survive a reflash and never
sit in source control. this just types the command down the same serial link the
dashboard uses:

    python3 -m anomaly.device_wifi --hotspot       # join THIS PC's Windows hotspot
    python3 -m anomaly.device_wifi --ssid MyNetwork --password hunter2
    python3 -m anomaly.device_wifi --status        # what is it connected to?
    python3 -m anomaly.device_wifi --forget

--hotspot needs no password from you: it reads the hotspot's name and password
from Windows (anomaly.hotspot), switches the hotspot on if it is off, and sends
them down the cable. nothing is typed, printed or saved. once done, the board
rejoins that hotspot by itself on every boot.

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
    ap.add_argument("--hotspot", action="store_true",
                    help="join this PC's Windows Mobile Hotspot -- no password needed")
    ap.add_argument("--ssid")
    ap.add_argument("--password", help="omit to be prompted without echoing "
                                       "(or set PULSE_WIFI_PASS, as the control panel does)")
    ap.add_argument("--save", action="store_true",
                    help="with --ssid: remember it for later instead of joining it now")
    ap.add_argument("--status", action="store_true", help="report SSID, IP and saved networks")
    ap.add_argument("--forget", action="store_true",
                    help="forget every saved network, or just --ssid's")
    ap.add_argument("--port", default=os.environ.get("DEVICE_PORT"))
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = ap.parse_args()

    if not (args.status or args.forget or args.ssid or args.hotspot):
        ap.error("give --hotspot, --ssid, --status or --forget")

    # an env var keeps the password off the command line, where any process
    # listing would show it
    pw = args.password if args.password is not None else os.environ.get("PULSE_WIFI_PASS")
    if args.hotspot:
        # BEFORE opening the port: opening it resets the board, and it should
        # come back up with the hotspot already there to join
        from . import hotspot
        try:
            cfg = hotspot.config()
            bad = hotspot.band_problem(cfg)
            if bad:
                raise SystemExit("  " + bad)
            if cfg.get("state") != "On":
                print("  switching the hotspot on…")
                cfg = hotspot.ensure_on()
        except hotspot.HotspotError as e:
            raise SystemExit("  %s" % e)
        print("  " + hotspot.describe(cfg))
        args.ssid, pw = cfg["ssid"], cfg["passphrase"]
        if not pw:
            raise SystemExit("  the hotspot has no password set -- the board needs one (WPA2)")

    ser, port = _open(args.port, args.baud)
    print(f"\n  board on {port}")
    try:
        if args.status:
            ser.write(b"W?\n")
            _drain(ser, 3.0, want="ssid=")
        elif args.forget:
            _drain(ser, 15.0, want="setup done")
            ser.reset_input_buffer()
            if args.ssid:
                ser.write(f"F,{args.ssid}\n".encode("utf-8"))
                _drain(ser, 5.0, want="wifi forgot")
            else:
                ser.write(b"W!\n")
                _drain(ser, 5.0, want="forgotten")
        else:
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

            # W joins now; N only remembers it for when nothing earlier answers
            ser.write(f"{'N' if args.save else 'W'},{args.ssid},{pw}\n".encode("utf-8"))
            print(f"  sent credentials for {args.ssid!r}")
            if not _drain(ser, 5.0, want="wifi saved"):
                print("\n  the board did not acknowledge the command. is the")
                print("  dashboard still running and holding the port?")
                return 1
            if args.save:
                print(f"\n  saved. the board tries {args.ssid!r} whenever its current")
                print("  network is out of reach.")
                return 0

            # a failed join now goes BACK to the old network after 25 s and says
            # "wifi connected" for that one -- so success means connected to the
            # network that was asked for, not just connected
            print("  connecting…")
            want = f"wifi connected ssid={args.ssid} "
            log = _drain(ser, 35.0, want=want)
            if not any(want in l for l in log):
                print("\n  it did not join. check the SSID and password, and that the network")
                print("  is 2.4 GHz -- the ESP32 cannot join a 5 GHz-only SSID.")
                print("  (it goes back to its previous network by itself.)")
                if args.hotspot:
                    print("  also: the board must be in range of THIS PC, and the")
                    print("  hotspot caps clients (python -m anomaly.hotspot).")
                return 1
            if args.hotspot:
                print(f"\n  done. it rejoins {args.ssid!r} by itself from now on, whenever")
                print("  the hotspot is on. run the fleet with:  python -m anomaly.fleet --hotspot")
    finally:
        ser.close()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
