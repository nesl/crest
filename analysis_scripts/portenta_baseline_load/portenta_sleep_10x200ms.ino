// Portenta baseline sleep-workload sketch:
// 10 iterations, each delay(200 ms), measured as one harness window.
#include <Arduino.h>

#include "common/tinyodom_hil_config.h"
#include "common/tinyodom_power.h"

#ifndef TINYODOM_AUTOSTART
#define TINYODOM_AUTOSTART 1
#endif

#ifndef TINYODOM_AUTOSTART_DELAY_MS
#define TINYODOM_AUTOSTART_DELAY_MS 0
#endif

#ifndef TINYODOM_SKIP_SERIAL_WAIT
#define TINYODOM_SKIP_SERIAL_WAIT 0
#endif

namespace {

constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStartCommandTimeoutMs = 5000;
constexpr uint32_t kIterationDelayMs = 200;

uint32_t inference_seq = 0;

}  // namespace

bool WaitForStartCommand(uint32_t timeout_ms) {
  const uint32_t start_ms = millis();
  while (millis() - start_ms < timeout_ms) {
    if (!Serial.available()) {
      delay(5);
      continue;
    }
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.equalsIgnoreCase(TINYODOM_CMD_START)) {
      return true;
    }
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
#if !TINYODOM_SKIP_SERIAL_WAIT
  while (!Serial && millis() < 2000) {
    delay(10);
  }
#endif

  delay(500);

  pinMode(TINYODOM_HARNESS_TRIGGER_PIN, OUTPUT);
  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, LOW);
  pinMode(TINYODOM_HARNESS_ARM_PIN, OUTPUT);
  digitalWrite(TINYODOM_HARNESS_ARM_PIN, HIGH);

  tinyodom_power::InitializePowerMonitor();

#if TINYODOM_AUTOSTART
  if (TINYODOM_AUTOSTART_DELAY_MS > 0) {
    delay(TINYODOM_AUTOSTART_DELAY_MS);
  }
#else
  Serial.println(TINYODOM_DUT_READY);
  while (!WaitForStartCommand(kStartCommandTimeoutMs)) {
    Serial.println(TINYODOM_DUT_READY);
  }
#endif

  const int kRuns = TINYODOM_INFERENCE_RUNS;
  int runs_completed = 0;

  digitalWrite(TINYODOM_HARNESS_ARM_PIN, LOW);
  delay(TINYODOM_DUT_ARM_HOLD_MS);
  tinyodom_power::ResetAccumulators();
  delay(100);
  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, HIGH);

  const uint32_t start_us = micros();
  for (int i = 0; i < kRuns; ++i) {
    delay(kIterationDelayMs);
    runs_completed++;
  }
  const uint32_t total_latency_us = micros() - start_us;

  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, LOW);
  digitalWrite(TINYODOM_HARNESS_ARM_PIN, HIGH);

  if (runs_completed > 0) {
    const float energy_total_j = tinyodom_power::FlushEnergyWindow();
    const float latency_s =
        (static_cast<float>(total_latency_us) / runs_completed) / 1000000.0f;
    ++inference_seq;
    Serial.print("runs: ");
    Serial.println(runs_completed);
    tinyodom_power::EmitPowerTelemetry(
        inference_seq, latency_s, runs_completed,
        tinyodom_power::GetIdleBaselinePower(), energy_total_j);
    Serial.print("timer output: ");
    Serial.println(latency_s, 6);
    Serial.print("workload: ");
    Serial.println("sleep");
  }
}

void loop() {
  delay(1000);
}
