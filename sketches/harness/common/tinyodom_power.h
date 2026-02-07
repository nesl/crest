#pragma once

#include <Arduino.h>
#include <cmath>

#ifndef TINYODOM_USE_INA228
#define TINYODOM_USE_INA228 0
#endif

#if TINYODOM_USE_INA228
#include <Wire.h>
#include <Adafruit_INA228.h>
#endif

namespace tinyodom_power {

constexpr float kInvalidTelemetryValue = -1.0f;

// INA228-backed telemetry (enabled on harness only).
#if TINYODOM_USE_INA228

static Adafruit_INA228 power_monitor;
static bool power_monitor_ready = false;
static float idle_baseline_power_mw = kInvalidTelemetryValue;

constexpr float kShuntResistanceOhms = 0.015f;
constexpr float kMaxExpectedCurrentAmps = 0.30f;
constexpr INA228_ConversionTime kCurrentConversionTime = INA228_TIME_280_us;
constexpr INA228_ConversionTime kVoltageConversionTime = INA228_TIME_150_us;
constexpr INA228_AveragingCount kAveragingCount = INA228_COUNT_64;
constexpr uint16_t kTriggeredFlushDelayMs = 20;

#ifndef INA228_ADCRANGE_40_96MV
#define INA228_ADCRANGE_40_96MV 1
#endif

#ifdef INA228_MODE_TRIGGERED_SHUNT_BUS_TEMP
constexpr auto kTriggerStopMode = INA228_MODE_TRIGGERED_SHUNT_BUS_TEMP;
#else
constexpr auto kTriggerStopMode =
    static_cast<decltype(INA228_MODE_CONTINUOUS)>(0x7);
#endif

inline float SanitizeFloat(float value) {
  if (isnan(value) || isinf(value)) {
    return kInvalidTelemetryValue;
  }
  return value;
}

inline bool IsReady() {
  return power_monitor_ready;
}

inline void ResetAccumulators() {
  if (power_monitor_ready) {
    power_monitor.resetAccumulators();
  }
}

inline float GetIdleBaselinePower() {
  return idle_baseline_power_mw;
}

// Initialize the INA228 and capture an idle baseline.
inline bool InitializePowerMonitor() {
  if (power_monitor_ready) {
    return true;
  }
  Wire.begin();
  if (!power_monitor.begin()) {
    Serial.println("energy error: init_failed");
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
  Serial.println("energy monitor: ready");
  delay(30);
  idle_baseline_power_mw = power_monitor.getPower_mW();
  return true;
}

// Stop triggered accumulation, read energy, then re-arm continuous mode.
inline float FlushEnergyWindow() {
  if (!power_monitor_ready) {
    return kInvalidTelemetryValue;
  }
  power_monitor.setMode(kTriggerStopMode);
  delay(kTriggeredFlushDelayMs);
  const float energy_total_j = power_monitor.readEnergy();
  power_monitor.resetAccumulators();
  power_monitor.setMode(INA228_MODE_CONTINUOUS);
  return energy_total_j;
}

// Print a standardized set of energy and power lines for the host parser.
inline void EmitPowerTelemetry(uint32_t sequence_id, float latency_s, int runs_completed,
                               float baseline_power_mw, float energy_total_j) {
  const bool energy_valid = (energy_total_j >= 0.0f);
  const float energy_per_inference_j =
      (runs_completed > 0 && energy_valid)
          ? (energy_total_j / static_cast<float>(runs_completed))
          : kInvalidTelemetryValue;
  const float avg_power_per_inference_mw =
      (latency_s > 0.0f && energy_per_inference_j >= 0.0f)
          ? (energy_per_inference_j / latency_s) * 1000.0f
          : kInvalidTelemetryValue;
  const float bus_voltage_v = power_monitor.getBusVoltage_V();
  const float avg_current_ma =
      (bus_voltage_v > 0.0f && avg_power_per_inference_mw >= 0.0f)
          ? (avg_power_per_inference_mw / bus_voltage_v)
          : kInvalidTelemetryValue;

  Serial.print("inference seq: ");
  Serial.println(sequence_id);
  Serial.print("energy output (mJ): ");
  Serial.println(SanitizeFloat(energy_per_inference_j * 1000.0f), 6);
  Serial.print("avg power output (mW): ");
  Serial.println(SanitizeFloat(avg_power_per_inference_mw), 3);
  Serial.print("avg current output (mA): ");
  Serial.println(SanitizeFloat(avg_current_ma), 3);
  Serial.print("bus voltage output (V): ");
  Serial.println(SanitizeFloat(bus_voltage_v), 3);
  Serial.print("idle power baseline (mW): ");
  Serial.println(SanitizeFloat(baseline_power_mw), 3);
}

// Stub telemetry for DUT builds without INA228.
#else

inline bool IsReady() {
  return false;
}

inline bool InitializePowerMonitor() {
  return false;
}

inline void ResetAccumulators() {}

inline float FlushEnergyWindow() {
  return kInvalidTelemetryValue;
}

inline float GetIdleBaselinePower() {
  return kInvalidTelemetryValue;
}

// Emit the same keys with invalid values when INA228 is disabled.
inline void EmitPowerTelemetry(uint32_t sequence_id, float /*latency_s*/, int /*runs_completed*/,
                               float /*baseline_power_mw*/, float /*energy_total_j*/) {
  Serial.print("inference seq: ");
  Serial.println(sequence_id);
  Serial.print("energy output (mJ): ");
  Serial.println(kInvalidTelemetryValue, 6);
  Serial.print("avg power output (mW): ");
  Serial.println(kInvalidTelemetryValue, 3);
  Serial.print("avg current output (mA): ");
  Serial.println(kInvalidTelemetryValue, 3);
  Serial.print("bus voltage output (V): ");
  Serial.println(kInvalidTelemetryValue, 3);
  Serial.print("idle power baseline (mW): ");
  Serial.println(kInvalidTelemetryValue, 3);
}

#endif

}  // namespace tinyodom_power
