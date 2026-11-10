// Battery level monitor for a 2S LiPo using an ADS1115 16-bit I2C ADC (Arduino Uno)
//
// LIBRARY: install "Adafruit ADS1X15" from the Arduino Library Manager
//          (it may also ask to install Adafruit BusIO -- accept).
//
// WIRING
//   ADS1115:  VDD -> 5V,  GND -> GND,  SCL -> A5,  SDA -> A4,  ADDR -> GND (address 0x48)
//   Divider:  Battery + -> 10k -> [junction] -> 10k -> GND ;  junction -> ADS1115 A0
//             (optional 0.1 uF capacitor from the junction to GND)
//   Battery - must share GND with the Arduino (the buck converter's GND does this).
//   The GY-521 (0x68) can share SDA/SCL: different address, no conflict.
//
// WHY GAIN_TWOTHIRDS: the divider puts up to 4.2 V on the ADS1115 input. GAIN_ONE only
// reads up to 4.096 V and would clip, so we use the +/-6.144 V range (LSB = 0.1875 mV).
// The input must never exceed VDD + 0.3 V (about 5.3 V when powered from 5 V).
//
// TESTING WITHOUT THE BATTERY (laptop USB power)
//   SIM_MODE 1 : no battery, no wiring needed. Fakes a pack draining 8.4 V -> 6.6 V.
//   SIM_MODE 0 + POT_TEST 1 : 100k pot, 5 V stands in for 8.4 V (DIVIDER_RATIO = 1.68).
//                Pot top -> 5V, wiper -> ADS1115 A0, pot bottom -> 220k resistor -> GND
//                (the 220k stretches the 6-8.4 V range across most of the pot's travel).
//                Turn until Serial shows pack_v ~ 7.48 (20 %). Set POT_TEST 0 for the real battery.
//   Do NOT connect the buck converter's 5 V output to the Uno while USB is plugged in.
//
// Serial (9600 baud) CSV: timestamp_ms,adc_raw,pack_v,cell_v,batt_pct,low_batt

#include <Wire.h>
#include <Adafruit_ADS1X15.h>

#define SIM_MODE      0       // 1 = software-simulated battery, 0 = read the ADS1115
#define POT_TEST      1       // 1 = potentiometer test: 5 V at the pot stands in for 8.4 V
                              // 0 = real battery through the 10k/10k divider
#define LOW_LED_PIN   13      // built-in LED turns on when battery is low

Adafruit_ADS1115 ads;

// ---- Pack / ADC ----
const int   CELL_COUNT     = 2;         // 2S LiPo
#if POT_TEST
const float DIVIDER_RATIO  = 8.4 / 5.0; // 1.68: pot at full 5 V reads as a full 8.4 V pack
#else
const float DIVIDER_RATIO  = 2.0;       // (R1+R2)/R2 -- change if you use other resistors
#endif
const float ADS_LSB_V      = 0.0001875; // volts per count at GAIN_TWOTHIRDS
const int   NUM_SAMPLES    = 8;         // averaged readings

// ---- Alerts (percent, with hysteresis so the alert doesn't flicker) ----
const float LOW_ON_PCT   = 20.0;
const float LOW_OFF_PCT  = 25.0;
const float CRITICAL_PCT = 5.0;

// ---- Simulation ----
const float SIM_START_V = 8.40;
const float SIM_END_V   = 6.60;
const unsigned long SIM_DURATION_MS = 60000;

// ---- Per-cell voltage -> percent (resting voltage, approximate) ----
struct Point { float v; float pct; };
const Point CURVE[] = {
  {4.20, 100}, {4.06, 90}, {3.98, 80}, {3.92, 70}, {3.87, 60},
  {3.82, 50},  {3.79, 40}, {3.77, 30}, {3.74, 20}, {3.68, 10},
  {3.45, 5},   {3.30, 0}
};
const int CURVE_N = sizeof(CURVE) / sizeof(CURVE[0]);

bool lowBattery = false;
bool criticalShown = false;
bool adsOk = false;
unsigned long lastPrintMs = 0;
const unsigned long PRINT_INTERVAL_MS = 500;

float cellVoltageToPercent(float v) {
  if (v >= CURVE[0].v) return 100.0;
  if (v <= CURVE[CURVE_N - 1].v) return 0.0;
  for (int i = 0; i < CURVE_N - 1; i++) {
    if (v <= CURVE[i].v && v >= CURVE[i + 1].v) {
      float frac = (v - CURVE[i + 1].v) / (CURVE[i].v - CURVE[i + 1].v);
      return CURVE[i + 1].pct + frac * (CURVE[i].pct - CURVE[i + 1].pct);
    }
  }
  return 0.0;
}

// Returns pack voltage; sets rawOut to the ADC count (or -1 on read failure)
float readPackVoltage(long &rawOut) {
#if SIM_MODE
  unsigned long t = millis() % (SIM_DURATION_MS + 3000);
  if (t > SIM_DURATION_MS) t = SIM_DURATION_MS;
  float frac = (float)t / SIM_DURATION_MS;
  float v = SIM_START_V - frac * (SIM_START_V - SIM_END_V);
  rawOut = (long)((v / DIVIDER_RATIO) / ADS_LSB_V);
  return v;
#else
  if (!adsOk) { rawOut = -1; return 0; }
  long sum = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += ads.readADC_SingleEnded(0);
  }
  rawOut = sum / NUM_SAMPLES;
  return (rawOut * ADS_LSB_V) * DIVIDER_RATIO;
#endif
}

void setup() {
  Serial.begin(9600);
  pinMode(LOW_LED_PIN, OUTPUT);
  digitalWrite(LOW_LED_PIN, LOW);

#if !SIM_MODE
  if (ads.begin(0x48)) {
    ads.setGain(GAIN_TWOTHIRDS);       // +/-6.144 V range
    adsOk = true;
  } else {
    Serial.println("# ERROR: ADS1115 not found at 0x48 - check wiring/ADDR pin");
  }
#endif
  Serial.println("timestamp_ms,adc_raw,pack_v,cell_v,batt_pct,low_batt");
}

void loop() {
  unsigned long now = millis();
  if (now - lastPrintMs < PRINT_INTERVAL_MS) return;
  lastPrintMs = now;

  long raw;
  float packV = readPackVoltage(raw);
  if (raw < 0) return;                         // ADS1115 missing: error already printed

  float cellV = packV / CELL_COUNT;            // assumes the cells are balanced
  float pct = cellVoltageToPercent(cellV);

  if (!lowBattery && pct <= LOW_ON_PCT) {
    lowBattery = true;
    Serial.println("# LOW BATTERY: 20% or less remaining");
  } else if (lowBattery && pct > LOW_OFF_PCT) {
    lowBattery = false;
    criticalShown = false;
    Serial.println("# Battery recovered above 25%");
  }
  if (pct <= CRITICAL_PCT && !criticalShown) {
    criticalShown = true;
    Serial.println("# CRITICAL: battery nearly empty, stop and recharge");
  }
  digitalWrite(LOW_LED_PIN, lowBattery ? HIGH : LOW);

  Serial.print(now);        Serial.print(",");
  Serial.print(raw);        Serial.print(",");
  Serial.print(packV, 3);   Serial.print(",");
  Serial.print(cellV, 3);   Serial.print(",");
  Serial.print(pct, 1);     Serial.print(",");
  Serial.println(lowBattery ? 1 : 0);
}
