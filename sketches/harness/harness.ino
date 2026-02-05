// Harness sketch for INA228 power measurement triggered by DUT GPIO.
#define TINYODOM_USE_INA228 1

#include "common/tinyodom_hil_config.h"
#include "common/tinyodom_power.h"

namespace {

// Timeouts are shared with the host via tinyodom_hil_config.h.
constexpr uint32_t kArmTimeoutMs = TINYODOM_HARNESS_ARM_TIMEOUT_MS;
constexpr uint32_t kActiveTimeoutMs = TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS;
constexpr uint32_t kSerialTimeoutMs = 50;

enum class HarnessState {
  kIdle,
  kArmed,
  kActive,
  kWaitDisarmHigh,
};

HarnessState state = HarnessState::kIdle;
uint32_t inference_seq = 0;
uint32_t arm_start_ms = 0;
uint32_t active_start_ms = 0;
uint32_t active_start_us = 0;
uint32_t stable_low_start_ms = 0;
int last_pin_state = LOW;

// Non-blocking serial read helper (empty string if nothing available).
String ReadLineIfAvailable() {
  if (!Serial.available()) {
    return String();
  }
  String line = Serial.readStringUntil('\n');
  line.trim();
  return line;
}

// Reset any arm-related timing and edge detection state.
void ResetArmState() {
  arm_start_ms = millis();
  stable_low_start_ms = 0;
}

// Return to idle and await arm-low stabilization.
void EnterIdle() {
  state = HarnessState::kIdle;
  ResetArmState();
}

// Begin watching for a valid rising edge on the trigger pin.
void EnterArmed() {
  state = HarnessState::kArmed;
  ResetArmState();
  Serial.println(TINYODOM_RESP_ARMED);
}

// Start an active measurement window.
void EnterActive() {
  state = HarnessState::kActive;
  active_start_ms = millis();
  active_start_us = micros();
  tinyodom_power::ResetAccumulators();
}

// Emit run count, energy telemetry, harness timing, and DONE marker.
void EmitHarnessTelemetry(uint32_t runs_completed, float latency_s, float energy_total_j) {
  ++inference_seq;
  Serial.print("runs: ");
  Serial.println(runs_completed);
  tinyodom_power::EmitPowerTelemetry(
      inference_seq, latency_s, runs_completed,
      tinyodom_power::GetIdleBaselinePower(), energy_total_j);
  Serial.print("harness timer output: ");
  Serial.println(latency_s, 6);
  Serial.println(TINYODOM_RESP_DONE);
}

}  // namespace

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
  while (!Serial && millis() < 2000) {
    delay(10);
  }

  // Monitor the DUT trigger line and initialize the INA228.
  pinMode(TINYODOM_HARNESS_ARM_PIN, INPUT_PULLUP);
  pinMode(TINYODOM_HARNESS_TRIGGER_PIN, INPUT_PULLDOWN);
  tinyodom_power::InitializePowerMonitor();

  // Advertise readiness to the host.
  Serial.println(TINYODOM_HARNESS_READY);
}

void loop() {
  // Handle PING commands from the host.
  const String line = ReadLineIfAvailable();
  if (line.length() > 0) {
    if (line.equalsIgnoreCase(TINYODOM_CMD_PING)) {
      Serial.println(TINYODOM_HARNESS_READY);
    }
  }

  // Sample the arm and trigger pins once per loop for edge detection.
  const int arm_pin_state = digitalRead(TINYODOM_HARNESS_ARM_PIN);
  const int pin_state = digitalRead(TINYODOM_HARNESS_TRIGGER_PIN);

  // IDLE: wait for D3 LOW and D2 LOW to remain stable before arming.
  if (state == HarnessState::kIdle) {
    if (arm_pin_state == LOW && pin_state == LOW) {
      if (stable_low_start_ms == 0) {
        stable_low_start_ms = millis();
      }
      if (millis() - stable_low_start_ms >= TINYODOM_HARNESS_STABLE_LOW_MS) {
        EnterArmed();
      }
    } else {
      stable_low_start_ms = 0;
    }
  }

  // ARMED: enforce stable-low then wait for rising edge.
  if (state == HarnessState::kArmed) {
    if (arm_pin_state == HIGH) {
      // DUT disarmed before asserting D2 HIGH; return to idle.
      EnterIdle();
    } else
    if (kArmTimeoutMs > 0 && millis() - arm_start_ms > kArmTimeoutMs) {
      Serial.println("harness error: arm_timeout");
      EnterIdle();
    } else if (pin_state == HIGH && last_pin_state == LOW) {
      EnterActive();
    }
  // ACTIVE: wait for falling edge or timeout, then emit telemetry.
  } else if (state == HarnessState::kActive) {
    if (kActiveTimeoutMs > 0 && millis() - active_start_ms > kActiveTimeoutMs) {
      Serial.println("harness error: active_timeout");
      state = HarnessState::kWaitDisarmHigh;
    } else if (pin_state == LOW && last_pin_state == HIGH) {
      const uint32_t total_latency_us = micros() - active_start_us;
      const uint32_t runs_completed = TINYODOM_INFERENCE_RUNS;
      const float latency_s =
          (static_cast<float>(total_latency_us) / runs_completed) / 1000000.0f;
      const float energy_total_j = tinyodom_power::FlushEnergyWindow();
      EmitHarnessTelemetry(runs_completed, latency_s, energy_total_j);
      state = HarnessState::kWaitDisarmHigh;
    }
  } else if (state == HarnessState::kWaitDisarmHigh) {
    // Require D3 HIGH between pulses to avoid accidental re-arming.
    if (arm_pin_state == HIGH) {
      EnterIdle();
    }
  }

  last_pin_state = pin_state;
}
