"""the Windows 11 Mobile Hotspot, from Python: read it, switch it on, keep it on.

the boards join the PC's own hotspot -- it sits next to them, so range stops
being a problem, and its network (192.168.137.x) is one the fleet already
scans. this module is how the rest of the project talks to it, through the
same WinRT API the Settings page uses (driven from PowerShell, so there is
nothing to install):

    config()      ssid, passphrase, band, on/off, clients
    ensure_on()   switch it on if it is off

the passphrase comes back to the caller through a pipe and is never printed or
written anywhere -- it goes straight down the USB cable to the board.

    python -m anomaly.hotspot            # status (never shows the password)
    python -m anomaly.hotspot --on       # switch it on

Windows only. two of its habits matter here:
  - it will only start while the PC has a connection to share (Ethernet or
    WiFi with internet). no internet in the demo room = no hotspot.
  - by default it switches itself OFF after a few minutes with nobody
    connected, and a board that rebooted then has nothing to rejoin. turn that
    off: Settings > Network & internet > Mobile hotspot > Power saving.
    `anomaly.fleet --hotspot` also switches it back on whenever it drops.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys

# Windows' Internet Connection Sharing always hands the hotspot this /24 unless
# someone has edited the registry to change it.
SUBNET = "192.168.137"

_PRELUDE = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # else "preparing modules" lands on stderr as CLIXML
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Networking.Connectivity.NetworkInformation, Windows.Networking.Connectivity, ContentType=WindowsRuntime]
$null = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager, Windows.Networking.NetworkOperators, ContentType=WindowsRuntime]
$cp = [Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile()
if ($null -eq $cp) { @{ error = 'no internet connection to share -- Windows will not start a hotspot without one' } | ConvertTo-Json -Compress; exit 0 }
$tm = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::CreateFromConnectionProfile($cp)
function Await($op, [Type]$t) {
  $m = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
  $task = $m.MakeGenericMethod($t).Invoke($null, @($op))
  $null = $task.Wait(20000)
  $task.Result
}
function State {
  $c = $tm.GetCurrentAccessPointConfiguration()
  @{ ssid = $c.Ssid; passphrase = $c.Passphrase; band = [string]$c.Band;
     state = [string]$tm.TetheringOperationalState; clients = $tm.ClientCount;
     max_clients = $tm.MaxClientCount; shared_from = $cp.ProfileName }
}
"""

_STATUS = _PRELUDE + r"""
State | ConvertTo-Json -Compress
"""

_START = _PRELUDE + r"""
$out = State
if ($out.state -ne 'On') {
  $r = Await ($tm.StartTetheringAsync()) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
  $out = State
  $out.start_status = [string]$r.Status
  $out.start_message = [string]$r.AdditionalErrorMessage
}
$out | ConvertTo-Json -Compress
"""


class HotspotError(RuntimeError):
    pass


def _run(script: str, timeout: float = 40.0) -> dict:
    if sys.platform != "win32":
        raise HotspotError("the Mobile Hotspot helpers are Windows-only")
    # -EncodedCommand, NOT the script on stdin: `-Command -` parses stdin like an
    # interactive console, so after a multi-line `if { }` it waits for a
    # possible `else` and silently swallows the line that prints the result --
    # the hotspot started (or not) and the caller heard nothing at all.
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                            "-ExecutionPolicy", "Bypass", "-EncodedCommand", enc],
                           capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise HotspotError("powershell.exe not found")
    except subprocess.TimeoutExpired:
        raise HotspotError("Windows did not answer within %d s" % timeout)
    lines = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        # stderr can carry a stack trace, never the passphrase: it is only
        # ever emitted inside the JSON on stdout
        msg = (r.stderr or r.stdout or "").strip().splitlines()
        raise HotspotError("could not read the hotspot: %s" % (msg[-1] if msg else "no output"))
    out = json.loads(lines[-1])
    if out.get("error"):
        raise HotspotError(out["error"])
    return out


def config() -> dict:
    """the hotspot as Windows has it configured, passphrase included."""
    return _run(_STATUS)


def ensure_on() -> dict:
    """switch it on if it is off; the result says whether that worked."""
    out = _run(_START)
    if out.get("state") != "On":
        why = out.get("start_message") or out.get("start_status") or "unknown"
        raise HotspotError("Windows would not start the hotspot (%s)" % why)
    return out


def band_problem(cfg: dict) -> str | None:
    """the ESP32 is 2.4 GHz only."""
    b = (cfg.get("band") or "").lower()
    if "five" in b or b.startswith("5"):
        return ("the hotspot is set to 5 GHz, which the ESP32 cannot join. "
                "Settings > Network & internet > Mobile hotspot > Properties > "
                "Edit > Network band: 2.4 GHz")
    return None


def describe(cfg: dict) -> str:
    """one line, safe to print: no passphrase."""
    return "hotspot %r: %s, %s, %s/%s clients, sharing %s" % (
        cfg.get("ssid"), cfg.get("state"), cfg.get("band"), cfg.get("clients"),
        cfg.get("max_clients"), cfg.get("shared_from"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--on", action="store_true", help="switch it on if it is off")
    a = ap.parse_args()
    try:
        cfg = ensure_on() if a.on else config()
    except HotspotError as e:
        print("  %s" % e)
        return 1
    print("  " + describe(cfg))
    bad = band_problem(cfg)
    if bad:
        print("  WARNING: " + bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
