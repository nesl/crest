// Minimal harness sketch: measure duration + INA228 energy during D2 HIGH pulse.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA228.h>
#include <cmath>

namespace {

constexpr int kTriggerPin = 2;
constexpr int kArmPin = 3;
constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStableLowMs = 500;

// INA228 configuration (match production defaults).
Adafruit_INA228 power_monitor;
bool power_monitor_ready = false;

constexpr float kShuntResistanceOhms = 0.015f;
constexpr float kMaxExpectedCurrentAmps = 0.30f;
constexpr INA228_ConversionTime kCurrentConversionTime = INA228_TIME_280_us;
constexpr INA228_ConversionTime kVoltageConversionTime = INA228_TIME_150_us;
constexpr INA228_AveragingCount kAveragingCount = INA228_COUNT_64;
constexpr float kInvalidTelemetryValue = -1.0f;

#ifndef INA228_ADCRANGE_40_96MV
#define INA228_ADCRANGE_40_96MV 1
#endif

#ifdef INA228_MODE_TRIGGERED_SHUNT_BUS_TEMP
constexpr auto kTriggerStopMode = INA228_MODE_TRIGGERED_SHUNT_BUS_TEMP;
#elif defined(INA228_MODE_TRIG_SHUNT)
constexpr auto kTriggerStopMode = INA228_MODE_TRIG_SHUNT;
#else
constexpr auto kTriggerStopMode =
    static_cast<decltype(INA228_MODE_CONTINUOUS)>(0x7);
#endif

// ISR-owned timing data (written in interrupt, read in loop).
volatile bool edge_done = false;
volatile uint32_t isr_start_us = 0;
volatile uint32_t isr_end_us = 0;

// Loop-owned state.
bool seen_high = false;
bool armed = false;
bool require_disarm_high = false;
uint32_t start_us = 0;
uint32_t end_us = 0;
float idle_baseline_power_mw = kInvalidTelemetryValue;
uint32_t inference_seq = 0;
uint32_t pulse_count = 0;
uint32_t arm_low_start_ms = 0;
uint32_t last_arm_log_ms = 0;

}  // namespace

float SanitizeFloat(float value) {
  // Protect against NaN/inf from the sensor.
  if (isnan(value) || isinf(value)) {
    return kInvalidTelemetryValue;
  }
  return value;
}

bool InitializePowerMonitor() {
  // One-time INA228 setup; returns false if wiring/I2C address is wrong.
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
  delay(30);
  idle_baseline_power_mw = power_monitor.getPower_mW();
  Serial.println("INA228 ready.");
  return true;
}

void EmitPowerTelemetry(float latency_s, int runs_completed, float energy_total_j) {
  // Emit the same energy/power fields as the production harness.
  const bool energy_valid = (energy_total_j >= 0.0f);
  const float energy_per_inference_j =
      (runs_completed > 0 && energy_valid)
          ? (energy_total_j / static_cast<float>(runs_completed))
          : kInvalidTelemetryValue;
  const float avg_power_per_inference_mw =
      (latency_s > 0.0f && energy_per_inference_j >= 0.0f)
          ? (energy_per_inference_j / latency_s) * 1000.0f
          : kInvalidTelemetryValue;
  const float bus_voltage_v =
      power_monitor_ready ? power_monitor.getBusVoltage_V() : kInvalidTelemetryValue;
  const float avg_current_ma =
      (bus_voltage_v > 0.0f && avg_power_per_inference_mw >= 0.0f)
          ? (avg_power_per_inference_mw / bus_voltage_v)
          : kInvalidTelemetryValue;

  Serial.print("inference seq: ");
  Serial.println(inference_seq);
  Serial.print("energy output (mJ): ");
  Serial.println(SanitizeFloat(energy_per_inference_j * 1000.0f), 6);
  Serial.print("avg power output (mW): ");
  Serial.println(SanitizeFloat(avg_power_per_inference_mw), 3);
  Serial.print("avg current output (mA): ");
  Serial.println(SanitizeFloat(avg_current_ma), 3);
  Serial.print("bus voltage output (V): ");
  Serial.println(SanitizeFloat(bus_voltage_v), 3);
  Serial.print("idle power baseline (mW): ");
  Serial.println(SanitizeFloat(idle_baseline_power_mw), 3);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
  while (!Serial && millis() < 2000) {
    delay(10);
  }

  // Initialize I2C + INA228 before waiting on the trigger.
  Wire.begin();
  InitializePowerMonitor();

  pinMode(kArmPin, INPUT_PULLUP);
  pinMode(kTriggerPin, INPUT_PULLDOWN);
  // Interrupt captures the exact edge timestamps; keep ISR tiny.
  attachInterrupt(digitalPinToInterrupt(kTriggerPin), []() {
    const int state = digitalRead(kTriggerPin);
    if (state == HIGH) {
      isr_start_us = micros();
    } else {
      isr_end_us = micros();
      edge_done = true;
    }
  }, CHANGE);
  Serial.println("HARNESS: waiting for armed D2 pulse");
}

void loop() {
  // Enforce a disarm period until D3 returns HIGH after each pulse.
  if (require_disarm_high) {
    if (digitalRead(kArmPin) == HIGH) {
      require_disarm_high = false;
      arm_low_start_ms = 0;
      last_arm_log_ms = 0;
    } else {
      const uint32_t now_ms = millis();
      if (now_ms - last_arm_log_ms >= 1000) {
        Serial.println("HARNESS: waiting for D3 HIGH to disarm");
        last_arm_log_ms = now_ms;
      }
      if (edge_done) {
        edge_done = false;
      }
      seen_high = false;
      return;
    }
  }

  // Arm only when D3 is LOW and D2 is LOW for a stable window.
  if (!armed) {
    const int arm_state = digitalRead(kArmPin);
    const int trigger_state = digitalRead(kTriggerPin);
    if (arm_state == LOW && trigger_state == LOW) {
      if (arm_low_start_ms == 0) {
        arm_low_start_ms = millis();
      }
      if (millis() - arm_low_start_ms >= kStableLowMs) {
        armed = true;
        arm_low_start_ms = 0;
        Serial.println("HARNESS: ARMED");
      }
    } else {
      arm_low_start_ms = 0;
    }
    const uint32_t now_ms = millis();
    if (now_ms - last_arm_log_ms >= 1000) {
      Serial.print("HARNESS: arm wait D3=");
      Serial.print(arm_state == LOW ? "LOW" : "HIGH");
      Serial.print(" D2=");
      Serial.print(trigger_state == LOW ? "LOW" : "HIGH");
      if (arm_low_start_ms != 0) {
        Serial.print(" stable_ms=");
        Serial.print(now_ms - arm_low_start_ms);
      }
      Serial.println();
      last_arm_log_ms = now_ms;
    }
  }

  if (!armed) {
    if (edge_done) {
      edge_done = false;
    }
    seen_high = false;
    return;
  }

  // Record the rising edge once per pulse so logs include an explicit timestamp.
  const int state = digitalRead(kTriggerPin);
  if (!seen_high && state == HIGH) {
    seen_high = true;
    if (power_monitor_ready) {
      power_monitor.resetAccumulators();
    }
    noInterrupts();
    start_us = isr_start_us;
    interrupts();
    Serial.print("HARNESS: D2 HIGH @ ");
    Serial.println(start_us);
  }

  // When the ISR reports the falling edge, compute duration + telemetry.
  if (edge_done) {
    noInterrupts();
    end_us = isr_end_us;
    edge_done = false;
    interrupts();
    if (seen_high) {
      const uint32_t duration_us = end_us - start_us;
      float energy_total_j = kInvalidTelemetryValue;
      if (power_monitor_ready) {
        power_monitor.setMode(kTriggerStopMode);
        delay(20);
        energy_total_j = power_monitor.readEnergy();
        power_monitor.resetAccumulators();
        power_monitor.setMode(INA228_MODE_CONTINUOUS);
      }
      Serial.print("HARNESS: D2 LOW @ ");
      Serial.println(end_us);
      Serial.print("HARNESS: duration_us: ");
      Serial.println(duration_us);
      const float latency_s = static_cast<float>(duration_us) / 1000000.0f;
      Serial.print("HARNESS: duration_s: ");
      Serial.println(latency_s, 6);
      inference_seq++;
      pulse_count++;
      Serial.print("HARNESS: pulse: ");
      Serial.println(pulse_count);
      Serial.print("runs: ");
      Serial.println(1);
      EmitPowerTelemetry(latency_s, /*runs_completed=*/1, energy_total_j);
      Serial.print("harness timer output: ");
      Serial.println(latency_s, 6);
      Serial.println("HARNESS: DONE");
    }
    seen_high = false;
    armed = false;
    require_disarm_high = true;
  }
}
