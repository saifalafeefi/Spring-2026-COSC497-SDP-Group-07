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

// Heart rate measured in TIME (see updateHeartRate below). Declared here because
// streamVitals() reports it long before that function appears in the file.
const int HR_BEATS = 8;             // intervals kept for the median
const uint32_t HR_MIN_MS = 300;     // refractory: 200 bpm ceiling
const uint32_t HR_MAX_MS = 2000;    // 30 bpm floor

float hrBaseline = 0.0f;            // slow DC follower
float hrEnvelope = 0.0f;            // typical AC magnitude
bool hrArmed = false;               // inside a beat, waiting to re-arm
uint32_t hrLastBeatMs = 0;
uint32_t hrIntervals[HR_BEATS];
int hrCount = 0;
int hrIdx = 0;

int32_t bpmLive = 0;
int8_t bpmValid = 0;

// Heart rate pushed back from the dashboard. It measures on a band-passed 64 Hz
// stream over a 12 s window -- longer and cleaner than anything the board can
// hold -- so when it is connected its value is the better one and the screen
// shows it, which also means the two displays cannot disagree. If the host goes
// quiet the board reverts to its own bpmLive within HOST_BPM_TTL: this device
// has to keep working untethered, so the link is an improvement, not a crutch.
const uint32_t HOST_BPM_TTL = 5000;   // ms before a host value is considered stale
int32_t hostBpm = 0;
uint32_t hostBpmMs = 0;
char rxLine[16];
uint8_t rxLen = 0;


// =====================================================
// Graph area
// =====================================================
const int GRAPH_X = 10;
const int GRAPH_Y = 115;
const int GRAPH_WIDTH = 300;
const int GRAPH_HEIGHT = 110;

int graphX = GRAPH_X;
int previousGraphY = GRAPH_Y + GRAPH_HEIGHT / 2;

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
uint32_t readSampleTimed(uint32_t *red, uint32_t *ir) {
  while (!particleSensor.available()) {
    particleSensor.check();
    delay(1);
  }

  uint32_t now = millis();

  *red = particleSensor.getRed();
  *ir = particleSensor.getIR();

  particleSensor.nextSample();

  return now;
}

void streamSample(uint32_t timestamp, uint32_t ir, uint32_t red) {
#if STREAM_ENABLED
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
  tft.print("Pulse waveform");
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
// Heart rate, measured in TIME rather than in samples
// =====================================================
// Beats are timed with millis(), so nothing here depends on the sample rate and
// it cannot fall out of tune when the delivered rate drifts or SPO2_DECIMATE is
// retuned. Same principle the host uses on its 64 Hz stream, which is why the
// two agree.
void hrReset() {
  hrBaseline = 0.0f; hrEnvelope = 0.0f; hrArmed = false;
  hrLastBeatMs = 0; hrCount = 0; hrIdx = 0;
  bpmLive = 0; bpmValid = 0;
}

void updateHeartRate(uint32_t ir, uint32_t nowMs) {
  if (ir < 50000) {                 // no finger: nothing to time
    hrReset();
    return;
  }
  float v = (float)ir;
  if (hrBaseline == 0.0f) { hrBaseline = v; hrEnvelope = 0.0f; }

  hrBaseline += (v - hrBaseline) * 0.02f;
  float ac = v - hrBaseline;
  hrEnvelope += (fabsf(ac) - hrEnvelope) * 0.02f;

  float thresh = hrEnvelope * 0.6f;
  if (thresh < 20.0f) return;       // envelope not established yet

  if (!hrArmed && ac > thresh) {
    uint32_t dt = nowMs - hrLastBeatMs;
    // A real refractory. Inside HR_MIN_MS this crossing is a noise spike riding
    // on the upstroke, not a new beat. Returning here (rather than falling
    // through) leaves hrLastBeatMs untouched: the earlier version reset the beat
    // clock to the spike, so the NEXT genuine beat was timed from the wrong
    // instant. Measured at 25% sample noise that read 109 bpm for a true 85.
    if (hrLastBeatMs != 0 && dt < HR_MIN_MS) {
      return;
    }
    hrArmed = true;
    if (hrLastBeatMs != 0 && dt <= HR_MAX_MS) {
      hrIntervals[hrIdx] = dt;
      hrIdx = (hrIdx + 1) % HR_BEATS;
      if (hrCount < HR_BEATS) hrCount++;

      if (hrCount >= 4) {           // median of recent intervals rejects outliers
        uint32_t tmp[HR_BEATS];
        for (int i = 0; i < hrCount; i++) tmp[i] = hrIntervals[i];
        for (int i = 1; i < hrCount; i++) {
          uint32_t key = tmp[i];
          int j = i - 1;
          while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
          tmp[j + 1] = key;
        }
        uint32_t med = tmp[hrCount / 2];
        if (med > 0) {
          bpmLive = (int32_t)(60000.0f / (float)med + 0.5f);
          bpmValid = (bpmLive > 25 && bpmLive < 240) ? 1 : 0;
        }
      }
    }
    hrLastBeatMs = nowMs;
  } else if (hrArmed && ac < thresh * 0.4f) {
    hrArmed = false;                // hysteresis: re-arm on the falling edge
  }
}

// =====================================================
// Update BPM and SpO2 values
// =====================================================
void updateNumbers() {
  // Clear only the changing number area
  tft.fillRect(165, 40, 150, 60, ILI9341_BLACK);

  tft.setTextSize(3);

  bool hostFresh = (hostBpm > 0 && (millis() - hostBpmMs) < HOST_BPM_TTL);
  int32_t showBpm = hostFresh ? hostBpm : bpmLive;
  bool showValid = hostFresh ? true : (bpmValid != 0);

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

  if (validSpo2 && spo2 >= 70 && spo2 <= 100) {
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
void updateGraphScale() {
  uint32_t minimumValue = irBuffer[0];
  uint32_t maximumValue = irBuffer[0];

  for (int i = 1; i < SAMPLE_BUFFER_SIZE; i++) {
    if (irBuffer[i] < minimumValue) {
      minimumValue = irBuffer[i];
    }

    if (irBuffer[i] > maximumValue) {
      maximumValue = irBuffer[i];
    }
  }

  if (maximumValue <= minimumValue + 100) {
    maximumValue = minimumValue + 100;
  }

  graphMin = minimumValue;
  graphMax = maximumValue;
}

void drawGraphPoint(uint32_t irValue) {
  int currentY = map(
    irValue,
    graphMin,
    graphMax,
    GRAPH_Y + GRAPH_HEIGHT - 3,
    GRAPH_Y + 3
  );

  currentY = constrain(
    currentY,
    GRAPH_Y + 2,
    GRAPH_Y + GRAPH_HEIGHT - 2
  );

  // Erase the current graph column
  tft.drawFastVLine(
    graphX,
    GRAPH_Y + 1,
    GRAPH_HEIGHT - 2,
    ILI9341_BLACK
  );

  if (graphX > GRAPH_X) {
    tft.drawLine(
      graphX - 1,
      previousGraphY,
      graphX,
      currentY,
      ILI9341_GREEN
    );
  }

  previousGraphY = currentY;
  graphX++;

  if (graphX >= GRAPH_X + GRAPH_WIDTH) {
    graphX = GRAPH_X;
    previousGraphY = currentY;
  }
}

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
      updateHeartRate(ir, timestamp);
      pollHostSerial();

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
  Serial.setTxTimeoutMs(10);
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
  byte ledBrightness = 30;
  byte sampleAverage = SENSOR_AVERAGE;
  byte ledMode = 2;
  int sampleRate = SENSOR_SAMPLE_RATE;
  int pulseWidth = 411;
  int adcRange = 4096;

  particleSensor.setup(
    ledBrightness,
    sampleAverage,
    ledMode,
    sampleRate,
    pulseWidth,
    adcRange
  );

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

  // Rescale the graph once per batch, not once per sample
  updateGraphScale();

  // Refill the tail with a fresh batch
  for (int i = SAMPLE_BUFFER_SIZE - SAMPLES_PER_BATCH; i < SAMPLE_BUFFER_SIZE; i++) {
    for (int k = 0; k < SPO2_DECIMATE; k++) {
      uint32_t red, ir;

      uint32_t t0 = micros();
      uint32_t timestamp = readSampleTimed(&red, &ir);
      uint32_t t1 = micros();

      bool fingerPresent = ir > 50000;
      showFingerMessage(fingerPresent);

      if (fingerPresent && (statSamples % GRAPH_DECIMATE) == 0) {
        drawGraphPoint(ir);
      }
      uint32_t t2 = micros();

      // Always stream, finger or not — the host decides what counts as contact
      // and needs the gaps to stay on a continuous timebase.
      streamSample(timestamp, ir, red);
      updateHeartRate(ir, timestamp);
      pollHostSerial();
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
