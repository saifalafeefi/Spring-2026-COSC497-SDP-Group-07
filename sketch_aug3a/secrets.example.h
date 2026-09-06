// Copy this file to `secrets.h` in the same folder and fill in your network.
// `secrets.h` is gitignored, so credentials never reach the repository.
//
// These are the FALLBACK: anything stored on the board (set with
// `python3 -m anomaly.device_wifi --ssid ...`) wins, because it survives a
// reflash. Clear the stored one with `--forget` to fall back to these.
//
// The ESP32 radio is 2.4 GHz only -- it cannot join a 5 GHz-only SSID.

#define WIFI_SSID ""
#define WIFI_PASS ""
