// Flash this sketch to the harness board (BLE33) that powers/reads the INA228.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA228.h>
#include <cmath>

namespace {

Adafruit_INA228 power_monitor;
bool power_monitor_ready = false;

// Match the production sketch settings.
constexpr float kShuntResistanceOhms = 0.015f;
constexpr float kMaxExpectedCurrentAmps = 0.30f;
constexpr INA228_ConversionTime kCurrentConversionTime = INA228_TIME_280_us;
constexpr INA228_ConversionTime kVoltageConversionTime = INA228_TIME_150_us;
constexpr INA228_AveragingCount kAveragingCount = INA228_COUNT_64;
constexpr float kInvalidTelemetryValue = -1.0f;

#ifndef INA228_ADCRANGE_40_96MV
#define INA228_ADCRANGE_40_96MV 1
#endif

float SanitizeFloat(float value) {
  if (isnan(value) || isinf(value)) {
    return kInvalidTelemetryValue;
  }
  return value;
}

bool InitializePowerMonitor() {
  if (power_monitor_ready) {
    return true;
  }
  if (!power_monitor.begin()) {
    Serial.println("INA228 init failed. Check wiring/I2C address.");
    return false;
  }

  power_monitor.setShunt(kShuntResistanceOhms, kMaxExpectedCurrentAmps);
  power_monitor.setADCRange(INA228_ADCRANGE_40_96MV);
  power_monitor.setCurrentConversionTime(kCurrentConversionTime);
  power_monitor.setVoltageConversionTime(kVoltageConversionTime);
  power_monitor.setTemperatureConversionTime(INA228_TIME_150_us);
  power_monitor.setAveragingCount(kAveragingCount);
  power_monitor.setMode(INA228_MODE_CONTINUOUS);
  power_monitor.resetAccumulators();

  power_monitor_ready = true;
  Serial.println("INA228 ready.");
  return true;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000) {
    delay(10);
  }

  Wire.begin();
  InitializePowerMonitor();
}

void loop() {
  if (!power_monitor_ready) {
    delay(1000);
    return;
  }

  const float bus_voltage_v = power_monitor.getBusVoltage_V();
  const float power_mw = power_monitor.getPower_mW();
  const float current_ma = (bus_voltage_v > 0.0f)
                               ? (power_mw / bus_voltage_v)
                               : kInvalidTelemetryValue;

  Serial.print("Bus voltage (V): ");
  Serial.println(SanitizeFloat(bus_voltage_v), 3);
  Serial.print("Power (mW): ");
  Serial.println(SanitizeFloat(power_mw), 3);
  Serial.print("Current (mA): ");
  Serial.println(SanitizeFloat(current_ma), 3);
  Serial.println("---");

  delay(1000);
}
