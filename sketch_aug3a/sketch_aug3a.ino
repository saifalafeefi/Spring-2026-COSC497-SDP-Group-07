// MAX30102 pulse monitor + host data stream.
//
// Two jobs at once:
//   1. the on-device TFT demo (BPM / SpO2 / live waveform) — unchanged
//   2. a machine-readable serial stream the host dashboard ingests
//      (anomaly/device_source.py → anomaly/serve.py --source device)
//
// Stream protocol (newline-terminated ASCII, 115200 baud):
//   # <text>                                 banner / comments — host ignores
//   D,<t_ms>,<ir>,<red>                      one PPG sample
//   V,<t_ms>,<hr>,<hrOk>,<spo2>,<spo2Ok>     device-computed vitals
//
// t_ms is the sample's acquisition time, back-corrected for FIFO backlog, so
// the host can resample onto a clean grid even when the display work makes the
// read loop bursty.
//
// Set STREAM_ENABLED to 0 to get the old chatty human-readable debug prints.

#include <Wire.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>

#include <WiFi.h>
#include <esp_mac.h>      // esp_read_mac: the MAC without the driver
#include <Preferences.h>

// Copy secrets.example.h to secrets.h and fill it in. It is gitignored, so
// credentials stay off the repository while still being hard-coded for you.
#include "secrets.h"

#include <ESPAsyncWebServer.h>

// The dashboard, gzipped into PROGMEM by sketch_aug3a/make_web_assets.py.
// Re-run that script whenever pulse/ or anomaly/static/ changes.
#include "web_assets.h"

#include "MAX30105.h"
#include "spo2_algorithm.h"

// =====================================================
// Host stream
// =====================================================
#define STREAM_ENABLED 1
#define STREAM_BAUD    115200

// =====================================================
// TFT pins for your ESP32-S3 expansion board
// =====================================================
#define TFT_CS    14
#define TFT_RST   21
#define TFT_DC    47
#define TFT_MOSI  45
#define TFT_SCLK   3
#define TFT_MISO  46

// =====================================================
// MAX30102 I2C pins
// =====================================================
#define MAX_SDA   17
#define MAX_SCL   18

// Use a separate SPI bus for the display
SPIClass displaySPI(FSPI);

// TFT object
Adafruit_ILI9341 tft(
  &displaySPI,
  TFT_DC,
  TFT_CS,
  TFT_RST
);

// MAX30102 object
MAX30105 particleSensor;

// The SparkFun library already defines BUFFER_SIZE,
// so we use a different name here.
// HARD CONSTRAINT: maxim_heart_rate_and_oxygen_saturation() declares its work
// arrays as int32_t an_x[BUFFER_SIZE] on the STACK and indexes them 0..n-1 from
// the length we pass. Passing more than BUFFER_SIZE smashes the stack and the
// board reboots mid-setup. Sizing from the library's own constant means the two
// cannot drift apart again.
const int SAMPLE_BUFFER_SIZE = BUFFER_SIZE;
const int SAMPLES_PER_BATCH = SAMPLE_BUFFER_SIZE / 4;


// Sensor timing. The FIFO averages SENSOR_AVERAGE raw conversions into one
// entry, so the rate the host actually sees is SENSOR_SAMPLE_RATE / SENSOR_AVERAGE.
//
// MEASURED: with SENSOR_AVERAGE = 4 the board delivered only ~10 Hz, not the
// 25 Hz this config implies, and instrumentation put 99 ms of every 100 ms
// sample budget inside the blocking FIFO wait — the sensor simply was not
// producing. Averaging 1 takes one FIFO entry per conversion instead of one per
// four, which lifts the delivered rate well clear of the 25 Hz the host needs.
const int SENSOR_SAMPLE_RATE = 100;
const int SENSOR_AVERAGE     = 1;
const int STREAM_FS          = SENSOR_SAMPLE_RATE / SENSOR_AVERAGE;   // 100 Hz nominal

// Measured delivered rate (`# stats hz=`), for reference only — nothing computes
// timestamps from it any more. The host resampler reads the device's millis()
// stamps directly and is rate-agnostic, so this staying stale cannot break it.
const int DELIVERED_FS = 40;

// The host gets every raw sample; the SpO2 buffer keeps one in SPO2_DECIMATE, so
// the maxim algorithm still sees roughly the ~25 Hz it assumes.
//
// TUNE THIS to whatever the board actually delivers: measured throughput was
// ~40 Hz (not the 100 Hz the register config implies), so 2 lands the SpO2
// buffer near 20 Hz. If the `# stats hz=` line reports something different,
// set this to round(hz / 25). It affects ONLY the on-device HR/SpO2 readout —
// the host computes its own heart rate from the 64 Hz resampled stream and is
// unaffected by this value.
const int SPO2_DECIMATE = 2;

uint32_t irBuffer[SAMPLE_BUFFER_SIZE];
uint32_t redBuffer[SAMPLE_BUFFER_SIZE];

int32_t spo2 = 0;
int8_t validSpo2 = 0;

int32_t heartRate = 0;
int8_t validHeartRate = 0;

// Heart rate. The algorithm lives with the conditioning below, because it runs
// on the conditioned 64 Hz signal; these are declared here because
// streamVitals() reports the value long before that code appears in the file.
int32_t bpmLive = 0;
int8_t bpmValid = 0;
uint32_t hrLastCompute = 0;

// Heart rate pushed back from the dashboard. It measures on a band-passed 64 Hz
// stream over a 12 s window -- longer and cleaner than anything the board can
// hold -- so when it is connected its value is the better one and the screen
// shows it, which also means the two displays cannot disagree. If the host goes
// quiet the board reverts to its own bpmLive within HOST_BPM_TTL: this device
// has to keep working untethered, so the link is an improvement, not a crutch.
const uint32_t HOST_BPM_TTL = 5000;   // ms before a host value is considered stale
int32_t hostBpm = 0;
uint32_t hostBpmMs = 0;

// The stress verdict, pushed from the dashboard as "S,<flag>,<level%>". The model
// does not run on this board -- the host scores the 60 s window and sends the
// result, exactly as it already does for heart rate. Same staleness rule: if the
// host goes quiet the panel says so rather than leaving an old verdict on screen.
int32_t hostFlag = -1;                // -1 unknown, 0 calm, 1 stressed
int32_t hostLevel = 0;                // 0-100, the deviation percentage
uint32_t hostFlagMs = 0;
// The dashboard's WESAD demo pushes verdicts here too, so the panel shows the
// recording instead of "no finger" while somebody is watching the demo. It is
// marked DEMO on screen and it locks the master out while it runs -- two
// writers on one panel would flicker between a recording and a real reading,
// which is the worst of both.
int8_t hostDemo = 0;
uint32_t hostDemoMs = 0;
// A demo supplies its own SpO2 or it does not, and the board simply shows what
// it is given. The simulated scenario has one; the WESAD clip cannot -- that
// wrist sensor records BVP from a single photodiode, and SpO2 is a red/IR
// ratio, so there is no number to send and none that could honestly be shown.
int32_t hostSpo2 = 0;
uint32_t hostSpo2Ms = 0;
const uint32_t DEMO_TTL = 4000;       // the page pushes once a second

// WHY there is no verdict, when there is none. The master used to push only
// once it had a real score, so for the first 60 s -- while its window filled --
// the board heard nothing and showed CALM, which is a verdict, and the wrong
// one: nobody watching could tell warm-up from a genuine all-clear.
//   ok    a real verdict, use hostFlag
//   warm  the master is still filling its 60 s window, hostWait seconds to go
//   hold  the window was unusable (movement), no verdict this second
//   none  no subject assigned to this board, or they have no baseline
char hostState[6] = "none";
// Who the master says is wearing this board. The board only knows its own
// MAC-derived id, so its dashboard called the wearer "pulse-000000" no matter
// what they were renamed to -- the name lives in the master's store, so the
// master has to say it. Pushed with every verdict, which means a board that
// reboots picks the name back up within a second rather than staying wrong.
char hostSubject[28] = "";
int32_t hostWait = 0;                 // seconds of warm-up left
int32_t shownWait = -1;               // last warm-up number painted
int32_t shownFlag = -2;               // what is currently painted, to avoid redraws
int8_t shownBadge = -1;               // which corner badge is painted, if any
uint32_t irDcDisplay = 0;             // slow IR average, for the no-finger check
// Big enough for "W,<ssid>,<password>": an SSID is up to 32 chars and a WPA2
// passphrase up to 63, so the old 16-byte buffer could never carry credentials.
char rxLine[160];
uint16_t rxLen = 0;

// ---- WiFi -----------------------------------------------------------------
// Credentials are typed over the USB serial link and kept in NVS, so they
// survive a reflash of the sketch and never sit in source control.
//     W,<ssid>,<password>   set and connect (SSID may not contain a comma)
//     W?                    report status and IP
//     W!                    forget the stored network
Preferences prefs;

// An ASYNC server on purpose: the sample loop blocks on the sensor FIFO for most
// of every 25 ms, so a synchronous server would only be serviced in those gaps
// and the page would crawl. AsyncWebServer runs on its own task instead.
AsyncWebServer server(80);
AsyncWebSocket ws("/ws");
bool webStarted = false;

// Stable per-board identity, derived from the MAC. The master keys everything on
// this -- which device is which in the roster, and which calibration file holds
// that person's baseline -- so it must survive reboots and DHCP changing the IP.
String devId = "pulse-unknown";

// The sensitivity slider lives in the dashboards, but the MODEL is on the master,
// so the board stores what the user picked and reports it in every frame; the
// master reads it back and decides the flag with it.
//
// Where the threshold LINE sits is a different question, and it is pure
// arithmetic on the slider -- no model needed -- so the board answers it itself.
// It used to echo whatever the master had last pushed to /flag, which meant a
// drag was answered with the PREVIOUS threshold and the line snapped back to the
// 0.42 default: the master only pushes as a side effect of a scoring tick, so an
// uncalibrated board (no push at all), a lifted finger or a refilling window left
// it stuck there for good.
float hostSens = 0.5f;          // 0..1 from the slider

// Where the flag threshold sits on the 0..1 level scale. The board no longer
// derives this. It once was a pure function of the slider, but the threshold is
// now k_sigma above the wearer's own live calm and only the master knows
// k_sigma -- so the master pushes the real number with every verdict and the
// board just draws what it is told. Guessing here drew 52% while the master
// flagged at 28%.
float hostThrLevel = 0.42f;     // until the master's first push arrives
String wifiSsid;
String wifiPass;
bool wifiWanted = false;         // credentials exist, so keep trying
wl_status_t wifiLast = WL_NO_SHIELD;
uint32_t wifiRetryMs = 0;


// =====================================================
// Graph area
// =====================================================
const int GRAPH_X = 10;
const int GRAPH_Y = 115;
const int GRAPH_WIDTH = 300;
const int GRAPH_HEIGHT = 110;


// Drawing every sample was starving the sensor read loop: the FIFO overflowed
// and we lost ~60% of the samples. Draw one point in GRAPH_DECIMATE, and scan
// the buffer for its min/max once per batch instead of once per sample.
const int GRAPH_DECIMATE = 4;
uint32_t graphMin = 0;
uint32_t graphMax = 1;

// Loop instrumentation — reported on a `#` line the host ignores. Tells us where
// the per-sample budget actually goes, instead of guessing at it.
uint32_t statSamples = 0;
uint32_t statReadUs = 0;
uint32_t statDrawUs = 0;
uint32_t statPrintUs = 0;
uint32_t statCalcUs = 0;
uint32_t statLastReport = 0;

// =====================================================
// Stream helpers
// =====================================================

// Read one FIFO entry and return its capture time.
//
// This used to back-date the timestamp by (backlog - 1) * SAMPLE_PERIOD_MS to
// compensate for reading a queued burst. That was wrong here and actively
// harmful: the loop blocks on the sensor (measured ~25 ms of a ~25 ms period in
// readSampleTimed), so samples are read as they arrive and the backlog is ~1.
// With a backlog that is occasionally 2, the correction subtracted a full period
// from samples that were NOT late, stretching device time to half of real time —
// 562 samples spanning 14 s of wall clock reported themselves as 20.4 Hz.
//
// Downstream that is corrosive rather than merely inaccurate: the host builds its
// 64 Hz grid from these timestamps, so a 2x stretch halves the apparent pulse
// frequency and drops it onto the 0.7 Hz edge of the band-pass, where it is
// attenuated into the noise floor.
//
// millis() at the moment of the read is simply correct.
// The sensor's configuration, in one place so the recovery path applies exactly
// what setup() did. ledBrightness 60 saturated the ADC on skin contact (IR read
// ~250,000 against a 262,143 ceiling), so it stays at 30.
uint32_t sensorRecoveries = 0;         // diagnostics: how often the sensor wedged

// A healthy sensor delivers every ~25 ms. A second of silence means it is gone,
// not slow.
#define SENSOR_STALL_MS 1000

void condReset();                      // defined with the conditioning, below
void hrReset();                        // defined with the heart rate, below
void hrPush(float v);                  // fed from the conditioner, below

void sensorConfigure() {
  particleSensor.setup(
    /* ledBrightness */ 30,
    /* sampleAverage */ SENSOR_AVERAGE,
    /* ledMode       */ 2,
    /* sampleRate    */ SENSOR_SAMPLE_RATE,
    /* pulseWidth    */ 411,
    /* adcRange      */ 4096
  );
}

// Bring a wedged sensor back. WiFi transmit bursts pull 250-350 mA, and on a
// marginal supply that sag browns out the MAX30102 or leaves its I2C mid-
// transaction; the symptom is IR collapsing to the ADC floor and the FIFO never
// filling again. Re-cycling the bus and re-applying the configuration recovers
// it without a reboot.
bool sensorRecover() {
  sensorRecoveries++;
  Serial.println("# sensor stalled - reinitialising");
  Wire.end();
  delay(20);
  Wire.begin(MAX_SDA, MAX_SCL);
  Wire.setClock(400000);
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("# sensor did not come back");
    return false;
  }
  sensorConfigure();
  condReset();                   // the timebase has a hole in it now
  return true;
}

// Bounded wait. The old version looped forever on available(), so a stalled
// sensor froze the whole sketch -- the web server kept answering while the
// sample loop, the display and the stream all stopped dead, which reads exactly
// like a crash.
uint32_t readSampleTimed(uint32_t *red, uint32_t *ir) {
  uint32_t waitStart = millis();
  while (!particleSensor.available()) {
    particleSensor.check();
    delay(1);
    if (millis() - waitStart > SENSOR_STALL_MS) {
      sensorRecover();
      waitStart = millis();
      if (!particleSensor.available()) {
        *red = 0;                // report nothing rather than block forever
        *ir = 0;
        return millis();
      }
      break;
    }
  }

  uint32_t now = millis();

  *red = particleSensor.getRed();
  *ir = particleSensor.getIR();

  particleSensor.nextSample();

  return now;
}

void streamSample(uint32_t timestamp, uint32_t ir, uint32_t red) {
#if STREAM_ENABLED
  if (!Serial) {
    return;                 // no host on the USB side; skip the formatting too
  }
  char line[48];
  int n = snprintf(line, sizeof(line), "D,%lu,%lu,%lu\n",
                   (unsigned long)timestamp, (unsigned long)ir,
                   (unsigned long)red);
  Serial.write((const uint8_t *)line, n);
#else
  Serial.print("Red: ");
  Serial.print(red);
  Serial.print(" | IR: ");
  Serial.println(ir);
#endif
}

void streamVitals() {
#if STREAM_ENABLED
  char line[64];
  int n = snprintf(line, sizeof(line), "V,%lu,%ld,%d,%ld,%d\n",
                   (unsigned long)millis(), (long)bpmLive, (int)bpmValid,
                   (long)spo2, (int)validSpo2);
  Serial.write((const uint8_t *)line, n);
#else
  Serial.print("Heart rate: ");
  Serial.print(bpmLive);
  Serial.print(" BPM, valid: ");
  Serial.print(bpmValid);

  Serial.print(" | SpO2: ");
  Serial.print(spo2);
  Serial.print("%, valid: ");
  Serial.println(validSpo2);
#endif
}

// Per-sample timing, emitted as a comment the host skips. Without this we are
// guessing at which of read / draw / print / SpO2-math is eating the budget.
void streamStats() {
#if STREAM_ENABLED
  uint32_t now = millis();
  if (statLastReport != 0 && (now - statLastReport) < 2000) {
    return;
  }
  if (statSamples == 0) {
    statLastReport = now;
    return;
  }

  float hz = statSamples * 1000.0f / (now - statLastReport);

  Serial.print("# stats hz=");
  Serial.print(hz, 1);
  Serial.print(" n=");
  Serial.print(statSamples);
  Serial.print(" read_us=");
  Serial.print(statReadUs / statSamples);
  Serial.print(" draw_us=");
  Serial.print(statDrawUs / statSamples);
  Serial.print(" print_us=");
  Serial.print(statPrintUs / statSamples);
  Serial.print(" calc_us=");
  Serial.println(statCalcUs);

  statLastReport = now;
  statSamples = 0;
  statReadUs = 0;
  statDrawUs = 0;
  statPrintUs = 0;
  statCalcUs = 0;
#endif
}

void streamBanner() {
#if STREAM_ENABLED
  Serial.println();
  Serial.println("# pulse-watch sensor stream v1");
  Serial.print("# fs=");
  Serial.print(STREAM_FS);
  Serial.println(" sample=D,t_ms,ir,red vitals=V,t_ms,hr,hr_ok,spo2,spo2_ok");
#endif
}

// =====================================================
// Show a fatal error on the TFT
// =====================================================
// Report every I2C address that answers. "Not found" is ambiguous on its own:
// an empty bus means wiring or power, whereas 0x57 present but begin() failing
// means the sensor is alive and something else is wrong.
void scanI2C() {
#if STREAM_ENABLED
  int found = 0;
  Serial.println("# i2c scan:");
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      found++;
      Serial.print("#   device at 0x");
      Serial.print(addr, HEX);
      if (addr == 0x57) {
        Serial.print("  <- MAX30102");
      }
      Serial.println();
    }
  }
  if (found == 0) {
    Serial.println("#   NOTHING on the bus - check VIN/GND/SDA/SCL and that the");
    Serial.println("#   breakout is on 3.3V. A full USB unplug/replug also clears");
    Serial.println("#   an I2C bus left hung by a reset mid-transaction.");
  }
#endif
}

void showError(const char *message) {
  tft.fillScreen(ILI9341_BLACK);

  tft.setTextColor(ILI9341_RED);
  tft.setTextSize(2);
  tft.setCursor(10, 30);
  tft.println(message);

  // Repeat forever: the host often attaches well after boot, and a message
  // printed once is a message nobody reads.
  while (true) {
    Serial.print("# error: ");
    Serial.println(message);
    scanI2C();
    delay(2000);
  }
}

// =====================================================
// Draw the main screen layout
// =====================================================
void drawInterface() {
  tft.fillScreen(ILI9341_BLACK);

  tft.setTextColor(ILI9341_CYAN);
  tft.setTextSize(2);
  tft.setCursor(10, 8);
  tft.println("MAX30102 Monitor");

  tft.drawFastHLine(0, 32, 320, ILI9341_DARKGREY);

  tft.setTextColor(ILI9341_WHITE);
  tft.setCursor(10, 43);
  tft.print("Heart rate:");

  tft.setCursor(10, 72);
  tft.print("Blood oxygen:");

  tft.drawRect(
    GRAPH_X - 1,
    GRAPH_Y - 1,
    GRAPH_WIDTH + 2,
    GRAPH_HEIGHT + 2,
    ILI9341_DARKGREY
  );

  tft.setTextSize(1);
  tft.setTextColor(ILI9341_LIGHTGREY);
  tft.setCursor(GRAPH_X, GRAPH_Y - 10);
  tft.print("Status");

  shownFlag = -2;              // force the first status paint
  wifiLast = WL_NO_SHIELD;     // force the header (IP) to repaint too
}

// =====================================================
// CALM / STRESSED panel
// =====================================================
// Repaints only when the verdict changes, so the screen is not cleared 40x a
// second (which is what made the old waveform panel flicker).
void drawStatus() {
  bool fresh = (hostFlag >= 0 && (millis() - hostFlagMs) < HOST_BPM_TTL);
  bool finger = (irDcDisplay > 50000);
  // -1 no finger, -2 master silent, -3 warming, -4 holding, -5 nobody assigned
  // -6 is "elevated": above two thirds of the way to the line but not over it.
  // Going straight from CALM to STRESSED made a near-miss look identical to a
  // real crossing, on the panel as well as in the browser.
  bool elevated = (hostFlag == 0 && hostThrLevel > 0.0f &&
                   hostLevel > 66.0f * hostThrLevel);
  // the master holds a flag for a few seconds after the level drops back, so
  // STRESSED here would contradict the number beside it
  bool settling = (hostFlag == 1 && hostThrLevel > 0.0f &&
                   hostLevel < 100.0f * hostThrLevel);
  // A recording has no finger on the sensor and no master behind it, so both
  // gates would answer "--" forever. Skip them, and say DEMO on the panel.
  bool demo = (hostDemo != 0 && (millis() - hostDemoMs) < DEMO_TTL);
  if (!demo) hostDemo = 0;
  // the master is scoring, but with no calibration behind it -- the model
  // against a stranger. a real verdict, and not the same claim as a calibrated
  // one, so it is shown and labelled rather than hidden.
  bool zeroShot = (strcmp(hostState, "zero") == 0);
  const char *badge = demo ? "DEMO" : (zeroShot ? "ZERO-SHOT" : NULL);
  int8_t badgeId = demo ? 1 : (zeroShot ? 2 : 0);
  int32_t state = demo
                ? ((strcmp(hostState, "warm") == 0) ? -3
                   : elevated ? -6
                   : settling ? -7
                   : hostFlag)
                : (!finger) ? -1
                : (!fresh) ? -2
                : (strcmp(hostState, "warm") == 0) ? -3
                : (strcmp(hostState, "hold") == 0) ? -4
                : (strcmp(hostState, "none") == 0) ? -5
                : elevated ? -6
                : settling ? -7
                : hostFlag;

  // the warm-up counts down, so it has to repaint even when the state is the same
  if (state == shownFlag && badgeId == shownBadge &&
      !(state == -3 && hostWait != shownWait)) {
    return;                    // nothing changed, leave the panel alone
  }
  shownFlag = state;
  shownWait = hostWait;
  shownBadge = badgeId;

  uint16_t bg, fg;
  const char *word;
  const char *sub;
  char warmWord[8];
  if (state == 1) {
    bg = ILI9341_RED;    fg = ILI9341_WHITE;
    word = "STRESSED";   sub = "deviation above your baseline";
  } else if (state == 0) {
    bg = ILI9341_DARKGREEN; fg = ILI9341_WHITE;
    word = "CALM";       sub = "within your baseline";
  } else if (state == -1) {
    bg = ILI9341_BLACK;  fg = ILI9341_DARKGREY;
    word = "--";         sub = "no finger on the sensor";
  } else if (state == -6) {
    bg = ILI9341_ORANGE; fg = ILI9341_BLACK;
    word = "ELEVATED";   sub = "above your calm, not flagged yet";
  } else if (state == -7) {
    bg = ILI9341_ORANGE; fg = ILI9341_BLACK;
    word = "SETTLING";   sub = "was flagged, back under the line";
  } else if (state == -3) {
    // the model needs a full 60 s window before it can say anything at all,
    // and showing CALM meanwhile was the wrong answer, not a missing one
    bg = ILI9341_NAVY;   fg = ILI9341_WHITE;
    snprintf(warmWord, sizeof(warmWord), "%lds", (long)hostWait);
    word = warmWord;     sub = "warming up -- filling the 60s window";
  } else if (state == -4) {
    bg = ILI9341_OLIVE;  fg = ILI9341_WHITE;
    word = "--";         sub = "movement -- holding the verdict";
  } else if (state == -5) {
    bg = ILI9341_BLACK;  fg = ILI9341_DARKGREY;
    word = "--";         sub = "nobody assigned to this board";
  } else {
    bg = ILI9341_BLACK;  fg = ILI9341_DARKGREY;
    word = "--";         sub = "waiting for the dashboard";
  }

  tft.fillRect(GRAPH_X, GRAPH_Y, GRAPH_WIDTH, GRAPH_HEIGHT, bg);

  tft.setTextColor(fg);
  tft.setTextSize(4);
  int16_t w = strlen(word) * 24;                 // 6 px glyph * size 4
  tft.setCursor(GRAPH_X + (GRAPH_WIDTH - w) / 2, GRAPH_Y + 26);
  tft.print(word);

  tft.setTextSize(1);
  int16_t sw = strlen(sub) * 6;
  tft.setCursor(GRAPH_X + (GRAPH_WIDTH - sw) / 2, GRAPH_Y + 70);
  tft.print(sub);

  // Nobody should be able to mistake a recording for a measurement, least of
  // all on the device's own screen.
  if (badge != NULL) {
    tft.setTextSize(1);
    tft.setTextColor(fg);
    tft.setCursor(GRAPH_X + GRAPH_WIDTH - 6 - (int16_t)strlen(badge) * 6,
                  GRAPH_Y + 6);
    tft.print(badge);
  }

  if (state == 0 || state == 1) {                // deviation bar
    int bx = GRAPH_X + 30, bw = GRAPH_WIDTH - 60, by = GRAPH_Y + 88;
    tft.drawRect(bx, by, bw, 8, fg);
    int fillw = (bw - 2) * hostLevel / 100;
    if (fillw > 0) tft.fillRect(bx + 1, by + 1, fillw, 6, fg);
  }
}

// =====================================================
// Signal conditioning: resample to 64 Hz, band-pass 0.7-3 Hz
// =====================================================
// A straight port of anomaly/device_source.py so the board produces the same
// waveform the host used to. Coefficients from scipy
// butter(2, [0.7, 3.0], btype='band', fs=64), and the difference equation below
// is transposed direct form II -- the same form scipy's sosfilt uses, verified
// to match it to zero error on a synthetic pulse before being written here.
#define COND_FS      64
#define COND_GAP_MS  500.0f            // bigger jump = discontinuity, re-prime
#define COND_CAP     512               // ring buffer of conditioned samples

static const int SOS_SECTIONS = 2;
static const float SOS[SOS_SECTIONS][6] = {
  {1.0957805345e-02f, 2.1915610689e-02f, 1.0957805345e-02f,
   1.0000000000e+00f, -1.7227369981e+00f, 7.8253493658e-01f},
  {1.0000000000e+00f, -2.0000000000e+00f, 1.0000000000e+00f,
   1.0000000000e+00f, -1.9227142752e+00f, 9.2858394305e-01f},
};
// Steady state for a unit input (scipy sosfilt_zi). Scaled by the current DC so
// the filter starts settled: without this the 0.7 Hz high-pass sees a step from
// 0 to ~120,000 counts and rings for seconds, swamping the pulse.
static const float SOS_ZI[SOS_SECTIONS][2] = {
  { 7.2203103171e-01f, -5.6263156777e-01f},
  {-7.3298883706e-01f,  7.3298883706e-01f},
};

float condZ[SOS_SECTIONS][2];
bool  condPrimed = false;
float condIrDc = 0.0f;                 // slow DC, alpha 0.04 as on the host
float condTPrev = -1.0f;               // device ms of the previous raw sample
float condIrPrev = 0.0f;
float condTGrid = 0.0f;                // next 64 Hz grid point, in device ms

float condBuf[COND_CAP];
volatile uint16_t condHead = 0;        // written by the sample loop
volatile uint16_t condTail = 0;        // read by the websocket task
uint32_t condTotal = 0;                // conditioned samples since boot
uint32_t condDropped = 0;
uint32_t condSentIdx = 0;              // x-axis index of the next sample sent
uint32_t wsFramesSent = 0;             // diagnostics: frames actually pushed
uint32_t wsTicks = 0;                  // diagnostics: wsTick() entries
uint32_t wsSkipped = 0;                // diagnostics: frames dropped for backpressure

// 8 frames a second, each carrying ~8 samples of 64 Hz data. The chart redraws
// far faster than an eye can follow either way, and a WebSocket message costs
// the same in overhead whether it holds one sample or twenty -- so send fewer,
// fuller frames. This is what keeps a -80 dBm link usable.
#define WS_PERIOD_MS    125
#define WS_MAX_SAMPLES  20

void condPrime(float dc) {
  for (int i = 0; i < SOS_SECTIONS; i++) {
    condZ[i][0] = SOS_ZI[i][0] * dc;
    condZ[i][1] = SOS_ZI[i][1] * dc;
  }
  condPrimed = true;
}

float condFilter(float x) {
  float v = x;
  for (int i = 0; i < SOS_SECTIONS; i++) {
    float y = SOS[i][0] * v + condZ[i][0];
    condZ[i][0] = SOS[i][1] * v - SOS[i][4] * y + condZ[i][1];
    condZ[i][1] = SOS[i][2] * v - SOS[i][5] * y;
    v = y;
  }
  return v;
}

void condPush(float v) {
  uint16_t next = (condHead + 1) % COND_CAP;
  if (next == condTail) {                 // consumer fell behind
    condTail = (condTail + 1) % COND_CAP;
    condDropped++;
  }
  condBuf[condHead] = v;
  condHead = next;
  condTotal++;
}

void condReset() {
  condTPrev = -1.0f;
  condPrimed = false;
  hrReset();                          // the signal is discontinuous now
}

// One raw sample in; zero or more conditioned 64 Hz samples out.
void condFeed(uint32_t tMs, uint32_t ir) {
  float t = (float)tMs;
  float x = (float)ir;

  condIrDc = (condIrDc == 0.0f) ? x : (0.96f * condIrDc + 0.04f * x);

  if (condTPrev < 0.0f || t < condTPrev || (t - condTPrev) > COND_GAP_MS) {
    condTPrev = t;                        // first sample, or the stream jumped
    condIrPrev = x;
    condTGrid = t;
    condPrime(x);
    return;
  }
  if (t == condTPrev) {
    condIrPrev = x;                       // duplicate stamp, nothing to span
    return;
  }

  const float step = 1000.0f / (float)COND_FS;
  float span = t - condTPrev;
  while (condTGrid <= t) {
    float frac = (condTGrid - condTPrev) / span;
    float interp = condIrPrev + frac * (x - condIrPrev);
    if (!condPrimed) {
      condPrime(interp);
    }
    float y = condFilter(interp);
    condPush(y);
    hrPush(y);                        // the HR algorithm runs on the same signal
    condTGrid += step;
  }
  condTPrev = t;
  condIrPrev = x;
}

// =====================================================
// Heart rate -- the SAME algorithm the host uses
// =====================================================
// A port of pipeline/vitals.estimate_heart_rate: band-pass the (already
// conditioned) signal again with zero phase, find peaks constrained by distance
// and prominence, and take the median interval. Verified against scipy before
// being written here -- identical to 0.0 bpm across 48-150 bpm and three noise
// levels, in the one regime where they disagreed BOTH were wrong the same way.
//
// The board used to run its own beat-timing detector instead. Two algorithms for
// one number meant the TFT and the dashboard could disagree for no physical
// reason, and the board's version swung 50 bpm on a vibrating finger because it
// had no notion of a physiologically plausible rate. There is now one algorithm.
//
// Known limitation, inherited from the host: below about 55 bpm the dicrotic
// notch clears the prominence bar and the rate reads double.
#define HR_WIN_SEC   12
#define HR_WIN       (COND_FS * HR_WIN_SEC)      // 768 samples
#define HR_MIN_N     (COND_FS * 4)               // needs 4 s before it answers
#define HR_MAX_PEAKS 64

float hrBuf[HR_WIN];                 // last 12 s of conditioned signal
uint16_t hrHead = 0;
uint16_t hrN = 0;
float hrWork[HR_WIN];                // filtfilt scratch

void hrReset() {
  hrHead = 0;
  hrN = 0;
  bpmLive = 0;
  bpmValid = 0;
}

void hrPush(float v) {
  hrBuf[hrHead] = v;
  hrHead = (hrHead + 1) % HR_WIN;
  if (hrN < HR_WIN) {
    hrN++;
  }
}

// zero-phase: filter forward, then backward over the result
void hrFiltFilt(float *x, int n) {
  float z[SOS_SECTIONS][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  for (int i = 0; i < n; i++) {
    float v = x[i];
    for (int sct = 0; sct < SOS_SECTIONS; sct++) {
      float y = SOS[sct][0] * v + z[sct][0];
      z[sct][0] = SOS[sct][1] * v - SOS[sct][4] * y + z[sct][1];
      z[sct][1] = SOS[sct][2] * v - SOS[sct][5] * y;
      v = y;
    }
    x[i] = v;
  }
  for (int sct = 0; sct < SOS_SECTIONS; sct++) {
    z[sct][0] = 0.0f;
    z[sct][1] = 0.0f;
  }
  for (int i = n - 1; i >= 0; i--) {
    float v = x[i];
    for (int sct = 0; sct < SOS_SECTIONS; sct++) {
      float y = SOS[sct][0] * v + z[sct][0];
      z[sct][0] = SOS[sct][1] * v - SOS[sct][4] * y + z[sct][1];
      z[sct][1] = SOS[sct][2] * v - SOS[sct][5] * y;
      v = y;
    }
    x[i] = v;
  }
}

void hrCompute() {
  if (hrN < HR_MIN_N) {
    bpmValid = 0;
    return;
  }
  int n = hrN;
  for (int i = 0; i < n; i++) {              // unroll the ring, oldest first
    hrWork[i] = hrBuf[(hrHead + HR_WIN - n + i) % HR_WIN];
  }
  hrFiltFilt(hrWork, n);

  float mean = 0.0f;
  for (int i = 0; i < n; i++) mean += hrWork[i];
  mean /= n;
  float var = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = hrWork[i] - mean;
    var += d * d;
  }
  float sd = sqrtf(var / n);
  if (sd <= 0.0f) {
    bpmValid = 0;
    return;
  }
  const float minProm = 0.4f * sd;
  const int minDist = (int)(COND_FS * 0.4f);   // 150 bpm ceiling

  int peaks[HR_MAX_PEAKS];
  float heights[HR_MAX_PEAKS];
  int np = 0;
  for (int i = 1; i < n - 1 && np < HR_MAX_PEAKS; i++) {
    if (!(hrWork[i] > hrWork[i - 1] && hrWork[i] >= hrWork[i + 1])) {
      continue;
    }
    int j = i;                                  // walk down to the left trough
    while (j > 0 && hrWork[j - 1] < hrWork[j]) j--;
    float left = hrWork[i];
    for (int k = j; k <= i; k++) if (hrWork[k] < left) left = hrWork[k];
    int m = i;                                  // and the right trough
    while (m < n - 1 && hrWork[m + 1] < hrWork[m]) m++;
    float right = hrWork[i];
    for (int k = i; k <= m; k++) if (hrWork[k] < right) right = hrWork[k];
    float base = left > right ? left : right;
    if (hrWork[i] - base >= minProm) {
      peaks[np] = i;
      heights[np] = hrWork[i];
      np++;
    }
  }
  if (np < 2) {
    bpmValid = 0;
    return;
  }

  // tallest first, then drop anything within minDist of one already kept
  for (int a = 1; a < np; a++) {
    int pi = peaks[a];
    float ph = heights[a];
    int b = a - 1;
    while (b >= 0 && heights[b] < ph) {
      peaks[b + 1] = peaks[b];
      heights[b + 1] = heights[b];
      b--;
    }
    peaks[b + 1] = pi;
    heights[b + 1] = ph;
  }
  int kept[HR_MAX_PEAKS];
  int nk = 0;
  for (int a = 0; a < np; a++) {
    bool ok = true;
    for (int b = 0; b < nk; b++) {
      int d = peaks[a] - kept[b];
      if (d < 0) d = -d;
      if (d < minDist) { ok = false; break; }
    }
    if (ok) kept[nk++] = peaks[a];
  }
  if (nk < 2) {
    bpmValid = 0;
    return;
  }
  for (int a = 1; a < nk; a++) {               // back into time order
    int v = kept[a];
    int b = a - 1;
    while (b >= 0 && kept[b] > v) { kept[b + 1] = kept[b]; b--; }
    kept[b + 1] = v;
  }

  int gaps[HR_MAX_PEAKS];
  int ng = 0;
  for (int a = 1; a < nk; a++) gaps[ng++] = kept[a] - kept[a - 1];
  for (int a = 1; a < ng; a++) {               // median of the intervals
    int v = gaps[a];
    int b = a - 1;
    while (b >= 0 && gaps[b] > v) { gaps[b + 1] = gaps[b]; b--; }
    gaps[b + 1] = v;
  }
  float med = (ng % 2) ? (float)gaps[ng / 2]
                       : 0.5f * (gaps[ng / 2 - 1] + gaps[ng / 2]);
  if (med <= 0.0f) {
    bpmValid = 0;
    return;
  }
  float bpm = 60.0f * COND_FS / med;
  if (bpm < 30.0f || bpm > 200.0f) {
    bpmValid = 0;
    return;
  }
  bpmLive = (int32_t)(bpm + 0.5f);
  bpmValid = 1;
}

// =====================================================
// Web server
// =====================================================
// Everything is served straight out of flash, pre-gzipped. No filesystem, no
// upload plugin, and the assets cannot drift away from the firmware serving them.
void webBegin() {
  if (webStarted) {
    return;
  }

  for (size_t i = 0; i < WEB_ASSET_COUNT; i++) {
    const WebAsset *a = &WEB_ASSETS[i];
    server.on(a->route, HTTP_GET, [a](AsyncWebServerRequest *req) {
      AsyncWebServerResponse *res =
          req->beginResponse_P(200, a->mime, a->data, a->len);
      res->addHeader("Content-Encoding", "gzip");
      res->addHeader("Cache-Control", "no-cache");
      req->send(res);
    });
  }

  // Pulse Watch is a single-patient view here, so /watch is just an alias.
  server.on("/watch", HTTP_GET, [](AsyncWebServerRequest *req) {
    AsyncWebServerResponse *res = req->beginResponse_P(
        200, WEB_ASSETS[0].mime, WEB_ASSETS[0].data, WEB_ASSETS[0].len);
    res->addHeader("Content-Encoding", "gzip");
    req->send(res);
  });

  // The master pushes its verdict here: GET /flag?f=0|1&l=0..100
  // It lands in exactly the state the serial "S,<flag>,<level>" command sets, so
  // the TFT panel, the websocket stream and the staleness rule all work the same
  // whether the verdict arrived over USB or over the network.
  server.on("/flag", HTTP_GET, [](AsyncWebServerRequest *req) {
    if (!req->hasParam("f")) {
      req->send(400, "text/plain", "need ?f=0|1[&l=0..100]");
      return;
    }
    int f = req->getParam("f")->value().toInt();
    int l = req->hasParam("l") ? req->getParam("l")->value().toInt() : 0;
    if (f != 0 && f != 1) {
      req->send(400, "text/plain", "f must be 0 or 1");
      return;
    }
    bool isDemo = req->hasParam("d") && req->getParam("d")->value().toInt() == 1;
    if (!isDemo && hostDemo != 0 && (millis() - hostDemoMs) < DEMO_TTL) {
      req->send(200, "text/plain", "demo");   // the demo owns the panel for now
      return;
    }
    hostDemo = isDemo ? 1 : 0;
    if (isDemo) hostDemoMs = millis();
    hostFlag = f;
    hostLevel = constrain(l, 0, 100);
    if (req->hasParam("s")) {
      // strncpy + explicit terminator rather than strlcpy: same result, and
      // no dependency on which libc the core happens to ship.
      strncpy(hostState, req->getParam("s")->value().c_str(), sizeof(hostState) - 1);
      hostState[sizeof(hostState) - 1] = 0;
    }
    if (req->hasParam("n")) {
      strncpy(hostSubject, req->getParam("n")->value().c_str(), sizeof(hostSubject) - 1);
      hostSubject[sizeof(hostSubject) - 1] = 0;
    }
    hostWait = req->hasParam("w") ? req->getParam("w")->value().toInt() : 0;
    if (req->hasParam("b")) {         // heart rate, the WiFi counterpart of "H,"
      int b = req->getParam("b")->value().toInt();
      if (b > 0) { hostBpm = b; hostBpmMs = millis(); }
    }
    if (req->hasParam("o")) {         // blood oxygen, when the sender has one
      int o = req->getParam("o")->value().toInt();
      if (o >= 70 && o <= 100) { hostSpo2 = o; hostSpo2Ms = millis(); }
    }
    // the master is the only authority on where the line sits
    if (req->hasParam("t")) {         // where the master's threshold sits, 0-100
      int t = req->getParam("t")->value().toInt();
      hostThrLevel = constrain(t, 0, 100) / 100.0f;
    }
    hostFlagMs = millis();
    req->send(200, "text/plain", "ok");
  });

  // Calibration progress, pushed by the master while a session runs. The board
  // does not calibrate anything -- it cannot score its own windows -- it just
  // forwards the master's progress to whichever dashboards are open.
  //   /calib?p=<phase>&w=<windows>&t=<target>&c=<0|1>
  server.on("/calib", HTTP_GET, [](AsyncWebServerRequest *req) {
    const char *phase = req->hasParam("p") ? req->getParam("p")->value().c_str() : "record";
    int w = req->hasParam("w") ? req->getParam("w")->value().toInt() : 0;
    int t = req->hasParam("t") ? req->getParam("t")->value().toInt() : 1;
    int c = req->hasParam("c") ? req->getParam("c")->value().toInt() : 0;
    if (t < 1) t = 1;
    char msg[224];
    snprintf(msg, sizeof(msg),
             "{\"type\":\"calib\",\"phase\":\"%s\",\"windows\":%d,"
             "\"min_windows\":%d,\"progress\":%.3f,\"can_commit\":%s}",
             phase, w, t, (float)w / (float)t, c ? "true" : "false");
    ws.textAll(msg);
    req->send(200, "text/plain", "ok");
  });

  // A plain-text health check, so a failure can be told apart from a hung page.
  server.on("/health", HTTP_GET, [](AsyncWebServerRequest *req) {
    String body = "ok id=" + devId +
                  " ip=" + WiFi.localIP().toString() +
                  " rssi=" + String(WiFi.RSSI()) +
                  " heap=" + String(ESP.getFreeHeap()) +
                  " ir=" + String(irDcDisplay) +
                  " bpm=" + String(bpmValid ? bpmLive : 0) +
                  // conditioning + websocket diagnostics: which link is dead?
                  " cond=" + String(condTotal) +
                  " head=" + String(condHead) +
                  " tail=" + String(condTail) +
                  " drop=" + String(condDropped) +
                  " wsn=" + String(ws.count()) +
                  " sent=" + String(wsFramesSent) +
                  " skip=" + String(wsSkipped) +
                  " rec=" + String(sensorRecoveries) +
                  " ticks=" + String(wsTicks);
    req->send(200, "text/plain", body);
  });

  ws.onEvent([](AsyncWebSocket *srv, AsyncWebSocketClient *client,
                AwsEventType type, void *arg, uint8_t *data, size_t len) {
    if (type == WS_EVT_DATA) {
      AwsFrameInfo *info = (AwsFrameInfo *)arg;
      if (!(info->final && info->index == 0 && info->len == len)) {
        return;                      // only whole single-frame text messages
      }
      char buf[192];
      size_t n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
      memcpy(buf, data, n);
      buf[n] = 0;
      // The master is a websocket CLIENT of this board, so anything broadcast
      // here reaches it. That is the whole transport for calibration: the
      // dashboard's Calibrate button used to send calib_start into a handler
      // that only knew set_sensitivity, and the command was dropped on the
      // floor while the modal sat at 0% waiting for a reply that never came.
      if (strstr(buf, "calib_") != NULL) {
        srv->textAll(buf);
        return;
      }
      if (strstr(buf, "set_sensitivity") != NULL) {
        const char *v = strstr(buf, "\"value\"");
        if (v != NULL && (v = strchr(v, ':')) != NULL) {
          float sv = atof(v + 1);
          if (sv >= 0.0f && sv <= 1.0f) {
            hostSens = sv;
            // The board used to derive the line itself, back when it was a pure
            // function of the slider. It is not any more: the threshold is
            // k_sigma above the wearer's live calm, and only the master knows
            // k_sigma. Guessing here drew 52% while the master flagged at 28%.
            // The master pushes the real number every second via /flag?t=.
            char thr[128];
            snprintf(thr, sizeof(thr),
                     "{\"type\":\"thr\",\"sensitivity\":%.3f,\"thr_level\":%.3f,\"threshold\":0}",
                     hostSens, hostThrLevel);
            srv->textAll(thr);       // keep every open dashboard in step
          }
        }
      }
      return;
    }
    if (type == WS_EVT_CONNECT) {
      Serial.print("# ws client ");
      Serial.println(client->id());
      char hello[384];
      snprintf(hello, sizeof(hello),
               // "role" tells the dashboard which of us is serving it. the same
               // page is baked into this board and served by the host, and the
               // board has no business keeping a long history -- so it says so
               // rather than the page guessing from the URL.
               "{\"type\":\"hello\",\"role\":\"device\",\"fs\":%d,\"win_s\":60,\"disp\":%d,\"infer_s\":1,"
               "\"subject\":\"%s\",\"running\":true,\"source\":\"device\","
               "\"calibrated_on\":\"none\",\"model\":false,\"device_connected\":true,"
               "\"device_port\":\"esp32\",\"sensitivity\":%.3f,\"thr_level\":%.3f,"
               "\"threshold\":0}",
               COND_FS, COND_FS * 15,
               hostSubject[0] ? hostSubject : devId.c_str(),
               hostSens, hostThrLevel);
      client->text(hello);
    }
  });
  server.addHandler(&ws);

  server.onNotFound([](AsyncWebServerRequest *req) {
    req->send(404, "text/plain", "not found");
  });

  server.begin();
  webStarted = true;
  Serial.print("# web server on http://");
  Serial.println(WiFi.localIP());
}

// Drain whatever the sample loop has conditioned and push it to the browser.
// Called from the main loop; sends nothing when there is no client, so an
// unopened dashboard costs almost nothing.
void wsTick() {
  static uint32_t lastSend = 0;
  wsTicks++;
  if (!webStarted || ws.count() == 0) {
    condTail = condHead;                  // nobody listening, do not accumulate
    return;
  }
  if (millis() - lastSend < WS_PERIOD_MS) {
    return;
  }

  // BACKPRESSURE. textAll() only QUEUES a message. Queueing faster than a weak
  // link can drain grows the queue without bound: the page lags, the heap
  // drains, and the board dies in about half a minute -- which is exactly what
  // it did at -80 dBm. Skip the frame instead; the ring buffer already drops
  // the oldest samples, so we shed load rather than accumulate it.
  if (!ws.availableForWriteAll()) {
    wsSkipped++;
    return;
  }
  if (ESP.getFreeHeap() < 60000) {        // last-ditch guard
    wsSkipped++;
    return;
  }
  lastSend = millis();

  char idx[256];
  char bvp[512];
  int ni = 0, nb = 0;
  int n = 0;
  while (condTail != condHead && n < WS_MAX_SAMPLES) {
    float v = condBuf[condTail];
    condTail = (condTail + 1) % COND_CAP;
    ni += snprintf(idx + ni, sizeof(idx) - ni, n ? ",%lu" : "%lu",
                   (unsigned long)condSentIdx++);
    nb += snprintf(bvp + nb, sizeof(bvp) - nb, n ? ",%.2f" : "%.2f", v);
    n++;
    if (ni > 200 || nb > 440) {
      break;
    }
  }
  if (n == 0) {
    return;
  }
  idx[ni] = 0;
  bvp[nb] = 0;

  bool finger = (irDcDisplay > 50000);
  uint32_t buffered = condTotal < (COND_FS * 60) ? condTotal : (COND_FS * 60);

  // The verdict comes from the master. If it stops arriving we report null
  // rather than leaving the last answer on screen pretending to be current.
  bool verdictFresh = finger && hostFlag >= 0 &&
                      (millis() - hostFlagMs) < HOST_BPM_TTL;
  char levelStr[16];
  if (verdictFresh) {
    snprintf(levelStr, sizeof(levelStr), "%.3f", hostLevel / 100.0f);
  } else {
    snprintf(levelStr, sizeof(levelStr), "null");
  }

  // idx caps at ~208 chars and bvp at ~448, plus ~290 of scaffolding oncesl
  // state and warm_s are in it. 1024 left barely 70 bytes of headroom and a
  // truncated frame is invalid JSON the dashboard silently drops.
  char frame[1280];
  snprintf(frame, sizeof(frame),
           "{\"type\":\"f\",\"running\":true,\"elapsed\":%.1f,\"buf\":%lu,\"win\":%d,"
           "\"idx\":[%s],\"bvp\":[%s],\"level\":%s,\"flag\":%s,\"score\":0,"
           "\"bpm\":%s,\"spo2\":%s,\"quality\":null,\"label\":\"-\",\"sens\":%.3f,"
           "\"thr_level\":%.3f,\"state\":\"%s\",\"warm_s\":%ld,"
           "\"subject\":\"%s\","
           "\"device\":{\"connected\":true,\"port\":\"esp32\",\"contact\":%s,\"dropped\":%lu}}",
           condTotal / (float)COND_FS, (unsigned long)buffered, COND_FS * 60,
           idx, bvp, levelStr,
           (verdictFresh && hostFlag == 1) ? "true" : "false",
           (finger && bpmValid) ? String(bpmLive).c_str() : "null",
           (finger && validSpo2 && spo2 >= 70 && spo2 <= 100)
               ? String(spo2).c_str() : "null",
           hostSens, hostThrLevel,
           // `fresh` is drawStatus()'s local; here freshness is judged on its
           // own, without requiring a finger -- the master's reason for having
           // no verdict is worth reporting either way.
           ((hostFlag >= 0 && (millis() - hostFlagMs) < HOST_BPM_TTL)
                ? hostState : "stale"), (long)hostWait,
           hostSubject[0] ? hostSubject : devId.c_str(),
           finger ? "true" : "false",
           (unsigned long)condDropped);
  ws.textAll(frame);
  wsFramesSent++;
  ws.cleanupClients();
}

// =====================================================
// WiFi
// =====================================================
void wifiShowStatus() {
  // The header line doubles as the address bar: once connected it shows the IP
  // to type into a browser, which is the only thing the user actually needs.
  tft.fillRect(0, 0, 320, 30, ILI9341_BLACK);
  tft.setTextColor(ILI9341_CYAN);
  tft.setTextSize(2);
  tft.setCursor(10, 8);
  if (WiFi.status() == WL_CONNECTED) {
    tft.print(WiFi.localIP());
  } else if (!wifiWanted) {
    tft.setTextSize(1);
    tft.setCursor(10, 12);
    tft.print("no wifi set - send W,ssid,pass");
  } else {
    tft.print("connecting...");
  }
}

void wifiConnect() {
  if (!wifiWanted) {
    return;
  }
  WiFi.mode(WIFI_STA);
  {
    // WiFi.macAddress() reads the WiFi DRIVER, which is not up yet -- begin() is
    // three lines below -- so it returned 00:00:00 and every board called itself
    // "pulse-000000". Harmless with one board; with two they collide in the
    // roster and the store, since the id is what everything keys on.
    // esp_read_mac() reads the eFuse, which is valid from power-on.
    uint8_t m[6] = {0};
    esp_read_mac(m, ESP_MAC_WIFI_STA);
    char b[24];
    snprintf(b, sizeof(b), "pulse-%02x%02x%02x", m[3], m[4], m[5]);
    devId = String(b);
    Serial.print("# device id ");
    Serial.println(devId);
  }
  WiFi.setSleep(false);            // sleep adds latency to the websocket
  // Full transmit power is what makes the 3.3 V rail sag hard enough to take the
  // sensor down with it. At -59 dBm there is plenty of link margin to give back.
  WiFi.setTxPower(WIFI_POWER_11dBm);
  WiFi.begin(wifiSsid.c_str(), wifiPass.c_str());
  Serial.print("# wifi connecting to ");
  Serial.println(wifiSsid);
  wifiRetryMs = millis();
}

void wifiLoad() {
  prefs.begin("netcfg", true);
  wifiSsid = prefs.getString("ssid", "");
  wifiPass = prefs.getString("pass", "");
  prefs.end();

  // Anything stored on the board wins: it survives a reflash, so a network set
  // once from the host is not silently undone by whatever secrets.h happens to
  // hold. Clear it with `device_wifi --forget` to fall back to the header.
  if (wifiSsid.length() == 0) {
    wifiSsid = String(WIFI_SSID);
    wifiPass = String(WIFI_PASS);
    if (wifiSsid.length() > 0) {
      Serial.println("# wifi using secrets.h");
    }
  } else {
    Serial.println("# wifi using stored credentials");
  }

  wifiWanted = wifiSsid.length() > 0;
  if (!wifiWanted) {
    Serial.println("# wifi not configured - set secrets.h, or send W,<ssid>,<password>");
  }
}

void wifiSave(const String &ssid, const String &pass) {
  prefs.begin("netcfg", false);
  prefs.putString("ssid", ssid);
  prefs.putString("pass", pass);
  prefs.end();
  wifiSsid = ssid;
  wifiPass = pass;
  wifiWanted = ssid.length() > 0;
  Serial.print("# wifi saved ssid=");
  Serial.println(ssid);
  WiFi.disconnect();
  wifiConnect();
}

void wifiReport() {
  Serial.print("# wifi ssid=");
  Serial.print(wifiSsid.length() ? wifiSsid : String("(none)"));
  Serial.print(" status=");
  Serial.print((int)WiFi.status());
  Serial.print(" ip=");
  Serial.println(WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString()
                                               : String("-"));
}

// Called from the sample loop: repaint on any change, and retry a dropped
// connection every 10 s. Never blocks -- the sensor read loop must keep running
// whether or not the network is up.
void wifiPoll() {
  wl_status_t st = WiFi.status();
  if (st != wifiLast) {
    wifiLast = st;
    if (st == WL_CONNECTED) {
      Serial.print("# wifi connected ssid=");
      Serial.print(WiFi.SSID());
      Serial.print(" ip=");
      Serial.println(WiFi.localIP());
      webBegin();                // safe to call repeatedly; starts once
    }
    wifiShowStatus();
  }
  if (wifiWanted && st != WL_CONNECTED && (millis() - wifiRetryMs) > 10000) {
    wifiRetryMs = millis();
    WiFi.disconnect();
    WiFi.begin(wifiSsid.c_str(), wifiPass.c_str());
  }
}

// =====================================================
// Receive the host's heart rate  ("H,<bpm>\n")
// =====================================================
void pollHostSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      rxLine[rxLen] = '\0';
      if (rxLen > 2 && rxLine[0] == 'H' && rxLine[1] == ',') {
        int v = atoi(&rxLine[2]);
        if (v > 25 && v < 240) {
          hostBpm = v;
          hostBpmMs = millis();
        }
      } else if (rxLen >= 2 && rxLine[0] == 'W' && rxLine[1] == '?') {
        wifiReport();
      } else if (rxLen >= 2 && rxLine[0] == 'W' && rxLine[1] == '!') {
        wifiSave("", "");
        Serial.println("# wifi forgotten");
      } else if (rxLen > 3 && rxLine[0] == 'W' && rxLine[1] == ',') {
        // "W,<ssid>,<password>" -- split on the FIRST comma after the prefix,
        // so a password may contain commas even though an SSID may not.
        char *body = &rxLine[2];
        char *comma = strchr(body, ',');
        if (comma != NULL) {
          *comma = '\0';
          wifiSave(String(body), String(comma + 1));
          wifiShowStatus();
        } else {
          Serial.println("# usage: W,<ssid>,<password>");
        }
      } else if (rxLen > 3 && rxLine[0] == 'S' && rxLine[1] == ',') {
        // "S,<flag>,<level>"  e.g. S,0,18
        int f = atoi(&rxLine[2]);
        char *comma = strchr(&rxLine[2], ',');
        if (comma != NULL && (f == 0 || f == 1)) {
          hostFlag = f;
          hostLevel = atoi(comma + 1);
          if (hostLevel < 0) hostLevel = 0;
          if (hostLevel > 100) hostLevel = 100;
          hostFlagMs = millis();
        }
      }
      rxLen = 0;
    } else if (rxLen < sizeof(rxLine) - 1) {
      rxLine[rxLen++] = c;
    } else {
      rxLen = 0;                    // overlong garbage: resync on next newline
    }
  }
}

// =====================================================
// Update BPM and SpO2 values
// =====================================================
void updateNumbers() {
  // Clear only the changing number area
  tft.fillRect(165, 40, 150, 60, ILI9341_BLACK);

  tft.setTextSize(3);

  // The board knows whether a finger is present and must never print a rate
  // without one, whatever the host claims. Defence in depth: the host now stops
  // sending on contact loss, but a stale or buggy sender cannot put a number
  // back on this screen.
  bool fingerOn = (irDcDisplay > 50000);
  // The one deliberate exception to that rule is the dashboard's WESAD demo:
  // it is a recording, so it has no finger by definition. It gets a number
  // only if the demo actually pushed one -- never the board's own estimate,
  // which would be a reading of the empty sensor -- and the panel beside it
  // says DEMO, so it cannot be taken for a measurement of whoever is holding
  // the board.
  bool demo = (hostDemo != 0 && (millis() - hostDemoMs) < DEMO_TTL);
  bool hostFresh = (fingerOn || demo) && hostBpm > 0 &&
                   (millis() - hostBpmMs) < HOST_BPM_TTL;
  int32_t showBpm = hostFresh ? hostBpm : bpmLive;
  bool showValid = demo ? hostFresh
                        : (fingerOn && (hostFresh ? true : (bpmValid != 0)));

  if (showValid) {
    tft.setTextColor(ILI9341_GREEN);
    tft.setCursor(165, 40);
    tft.print(showBpm);

    tft.setTextSize(1);
    tft.print(" BPM");
  } else {
    tft.setTextColor(ILI9341_YELLOW);
    tft.setCursor(165, 40);
    tft.print("--");
  }

  tft.setTextSize(3);

  bool hostSpo2Fresh = demo && hostSpo2 >= 70 && hostSpo2 <= 100 &&
                       (millis() - hostSpo2Ms) < HOST_BPM_TTL;
  if (hostSpo2Fresh) {
    tft.setTextColor(ILI9341_CYAN);
    tft.setCursor(165, 70);
    tft.print(hostSpo2);
    tft.print("%");
  } else if (!demo && fingerOn && validSpo2 && spo2 >= 70 && spo2 <= 100) {
    tft.setTextColor(ILI9341_CYAN);
    tft.setCursor(165, 70);
    tft.print(spo2);
    tft.print("%");
  } else {
    tft.setTextColor(ILI9341_YELLOW);
    tft.setCursor(165, 70);
    tft.print("--");
  }
}

// =====================================================
// Show whether a finger is detected
// =====================================================
void showFingerMessage(bool fingerPresent) {
  static bool previousState = true;

  if (fingerPresent == previousState) {
    return;
  }

  previousState = fingerPresent;

  tft.fillRect(10, 98, 300, 14, ILI9341_BLACK);

  tft.setTextSize(1);
  tft.setCursor(10, 100);

  if (fingerPresent) {
    tft.setTextColor(ILI9341_GREEN);
    tft.print("Finger detected - keep still");
  } else {
    tft.setTextColor(ILI9341_YELLOW);
    tft.print("Place fingertip gently on sensor");
  }
}

// =====================================================
// Draw one point on the waveform graph
// =====================================================
// =====================================================
// Collect the first 100 samples
// =====================================================
void readInitialSamples() {
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextSize(2);
  tft.setCursor(20, 100);
  tft.println("Collecting samples...");

  for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
    for (int k = 0; k < SPO2_DECIMATE; k++) {
      uint32_t red, ir;
      uint32_t timestamp = readSampleTimed(&red, &ir);

      // Stream the warm-up block too — the host wants an unbroken record from
      // boot, not a 4 s hole before the first calculateReadings().
      streamSample(timestamp, ir, red);
      pollHostSerial();
      irDcDisplay = (irDcDisplay == 0) ? ir : (irDcDisplay * 15 + ir) / 16;
      condFeed(timestamp, ir);
      if ((statSamples & 0xFF) == 0) wifiPoll();
      if (millis() - hrLastCompute > 1000) {   // once a second, like the host
        hrLastCompute = millis();
        hrCompute();
      }
      wsTick();

      if (k == SPO2_DECIMATE - 1) {     // keep the last of each group
        redBuffer[i] = red;
        irBuffer[i] = ir;
      }
    }

#if STREAM_ENABLED
    if ((i % SAMPLES_PER_BATCH) == 0) {
      Serial.print("# warmup ");
      Serial.print(i);
      Serial.print("/");
      Serial.print(SAMPLE_BUFFER_SIZE);
      Serial.print(" t=");
      Serial.println(millis());
    }
#endif
  }
}

// =====================================================
// Calculate BPM and oxygen saturation
// =====================================================
void calculateReadings() {
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer,
    SAMPLE_BUFFER_SIZE,
    redBuffer,
    &spo2,
    &validSpo2,
    &heartRate,
    &validHeartRate
  );

  // The library's heartRate is DELIBERATELY IGNORED. It counts beats in SAMPLES
  // against its hardcoded FreqS, and even after rescaling the output it swung
  // 28-150 bpm on a steady 85 bpm pulse: its peak detection is tuned for a 25 Hz
  // buffer and mis-counts on ours, catching dicrotic notches as beats. bpmLive
  // below replaces it. SpO2 from this call is kept and is sound -- it is a
  // red/IR amplitude ratio with no time term, so the sample rate never enters it.

  streamVitals();

  updateNumbers();
}

// =====================================================
// Setup
// =====================================================
void setup() {
  Serial.begin(STREAM_BAUD);

  // Native USB-CDC blocks on write when no host is draining the port — the
  // default TX timeout is 250 ms PER WRITE. With five prints per sample that
  // stalls the sketch outright whenever nothing is attached, which looks exactly
  // like a frozen screen. 0 = never block; drop bytes instead when unread.
#if ARDUINO_USB_CDC_ON_BOOT
  // 0 = never block, drop bytes instead. 10 ms looks harmless until nothing is
  // draining the port: with USB plugged in for power but no reader attached, the
  // CDC buffer fills and EVERY write stalls for the timeout. At 40 samples a
  // second, with a stream line plus stats per sample, that throttled the whole
  // sample loop to one iteration every few seconds -- which looked exactly like
  // the sensor dying, and is why it only appeared once we went WiFi-only.
  Serial.setTxTimeoutMs(0);
#endif

  // Native USB-CDC on the ESP32-S3 only enumerates after boot, so anything
  // printed immediately is written into the void. Wait for the host to attach
  // before the banner — but bound the wait, so the board still runs standalone
  // (TFT only) when nothing is plugged into USB.
  uint32_t waitStart = millis();
  while (!Serial && (millis() - waitStart) < 2000) {
    delay(10);
  }
  delay(200);

  streamBanner();

  // Start display SPI
  displaySPI.begin(
    TFT_SCLK,
    TFT_MISO,
    TFT_MOSI,
    TFT_CS
  );

  tft.begin();
  tft.setRotation(1);

  drawInterface();

  // Start I2C for MAX30102
  wifiLoad();
  wifiConnect();
  wifiShowStatus();

  Wire.begin(MAX_SDA, MAX_SCL);
  Wire.setClock(400000);

  // Detect sensor. Retry a few times: a bus still settling after reset can fail
  // a single cold begin() on hardware that is perfectly fine.
  bool sensorOk = false;
  for (int attempt = 1; attempt <= 5 && !sensorOk; attempt++) {
    sensorOk = particleSensor.begin(Wire, I2C_SPEED_FAST);
#if STREAM_ENABLED
    Serial.print("# sensor begin attempt ");
    Serial.print(attempt);
    Serial.println(sensorOk ? " OK" : " failed");
#endif
    if (!sensorOk) {
      delay(300);
    }
  }

  if (!sensorOk) {
    scanI2C();
    showError("MAX30102 not found");
  }

  // Sensor configuration.
  //
  // ledBrightness 60 saturated the ADC on skin contact: IR read ~250,000 against
  // an 18-bit ceiling of 262,143, which pins the waveform flat at the top and
  // makes the maxim SpO2 algorithm return -999. 30 lands a fingertip around
  // 100-150k with headroom. If IR still reads >240,000 with a finger on, drop it
  // further; if it reads <50,000, raise it.
  sensorConfigure();

  // Collect first block of samples
  readInitialSamples();

  // Redraw clean interface
  drawInterface();

  // Calculate first reading
  calculateReadings();

#if STREAM_ENABLED
  Serial.print("# setup done t=");
  Serial.println(millis());
#endif
}

// =====================================================
// Main loop
// =====================================================
void loop() {
  // Slide the buffer down by one batch, keeping the newest samples
  for (int i = SAMPLES_PER_BATCH; i < SAMPLE_BUFFER_SIZE; i++) {
    redBuffer[i - SAMPLES_PER_BATCH] = redBuffer[i];
    irBuffer[i - SAMPLES_PER_BATCH] = irBuffer[i];
  }

  // Refill the tail with a fresh batch
  for (int i = SAMPLE_BUFFER_SIZE - SAMPLES_PER_BATCH; i < SAMPLE_BUFFER_SIZE; i++) {
    for (int k = 0; k < SPO2_DECIMATE; k++) {
      uint32_t red, ir;

      uint32_t t0 = micros();
      uint32_t timestamp = readSampleTimed(&red, &ir);
      uint32_t t1 = micros();

      bool fingerPresent = ir > 50000;
      showFingerMessage(fingerPresent);

      if ((statSamples % GRAPH_DECIMATE) == 0) {
        drawStatus();
      }
      uint32_t t2 = micros();

      // Always stream, finger or not — the host decides what counts as contact
      // and needs the gaps to stay on a continuous timebase.
      streamSample(timestamp, ir, red);
      pollHostSerial();
      irDcDisplay = (irDcDisplay == 0) ? ir : (irDcDisplay * 15 + ir) / 16;
      condFeed(timestamp, ir);
      if ((statSamples & 0xFF) == 0) wifiPoll();
      if (millis() - hrLastCompute > 1000) {   // once a second, like the host
        hrLastCompute = millis();
        hrCompute();
      }
      wsTick();
      uint32_t t3 = micros();

      if (k == SPO2_DECIMATE - 1) {     // keep the last of each group
        redBuffer[i] = red;
        irBuffer[i] = ir;
      }

      statSamples++;
      statReadUs += (t1 - t0);
      statDrawUs += (t2 - t1);
      statPrintUs += (t3 - t2);

      // Report from INSIDE the batch. Reporting after it assumed the batch
      // completes, which is exactly what we are trying to find out.
      streamStats();
    }
  }

  // Recalculate after each new group of samples
  uint32_t c0 = micros();
  calculateReadings();
  statCalcUs = micros() - c0;

#if STREAM_ENABLED
  Serial.print("# batch done t=");
  Serial.println(millis());
#endif
}
