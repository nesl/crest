// Toy DUT sketch for validating the two-board handshake + trigger path.
// Prints DUT READY, waits for START, toggles D2 for a fixed window, and emits
// runs/timer output lines for host parsing.

#include <Arduino.h>

namespace {

constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStartCommandTimeoutMs = 5000;
constexpr int kTriggerPin = 2;
constexpr int kRuns = 10;
constexpr uint32_t kWindowDelayMs = 1000;  // Total window duration.

bool WaitForStartCommand(uint32_t timeout_ms) {
  const uint32_t start_ms = millis();
  while (millis() - start_ms < timeout_ms) {
    if (!Serial.available()) {
      delay(5);
      continue;
    }
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.equalsIgnoreCase("START")) {
      return true;
    }
  }
  return false;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
  while (!Serial && millis() < 2000) {
    delay(10);
  }

  pinMode(kTriggerPin, OUTPUT);
  digitalWrite(kTriggerPin, LOW);

  Serial.println("DUT READY");
  while (!WaitForStartCommand(kStartCommandTimeoutMs)) {
    Serial.println("DUT READY");
  }

  delay(100);  // Let any pending serial output drain.
  digitalWrite(kTriggerPin, HIGH);
  const uint32_t start_us = micros();
  delay(kWindowDelayMs);
  const uint32_t total_latency_us = micros() - start_us;
  digitalWrite(kTriggerPin, LOW);

  const float latency_s =
      (static_cast<float>(total_latency_us) / static_cast<float>(kRuns)) / 1000000.0f;
  Serial.print("runs: ");
  Serial.println(kRuns);
  Serial.print("timer output: ");
  Serial.println(latency_s, 6);
}

void loop() {
  delay(1000);
}
