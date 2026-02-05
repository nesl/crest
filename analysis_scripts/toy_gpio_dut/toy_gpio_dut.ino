// Minimal DUT sketch: wait 5 seconds, drive D2 HIGH for a random 1-20s window, then LOW.
#include <Arduino.h>

namespace {

constexpr int kTriggerPin = 2;
constexpr int kArmPin = 3;
constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kBootDelayMs = 5000;
constexpr uint32_t kStableLowMs = 600;
constexpr uint32_t kMinWorkWindowMs = 1000;
constexpr uint32_t kMaxWorkWindowMs = 20000;
constexpr int kEntropyPin = A0;

uint32_t start_us = 0;
bool done = false;
uint32_t work_window_ms = kMinWorkWindowMs;

}  // namespace

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
  while (!Serial && millis() < 2000) {
    delay(10);
  }

  pinMode(kTriggerPin, OUTPUT);
  digitalWrite(kTriggerPin, LOW);
  pinMode(kArmPin, OUTPUT);
  digitalWrite(kArmPin, HIGH);  // Active-low arm line; HIGH means disarmed.

  // Seed RNG using entropy from a floating analog pin and micros().
  pinMode(kEntropyPin, INPUT);
  randomSeed(static_cast<unsigned long>(analogRead(kEntropyPin)) ^ micros());
  work_window_ms = static_cast<uint32_t>(
      random(kMinWorkWindowMs, kMaxWorkWindowMs + 1));
  Serial.print("DUT: work_window_ms: ");
  Serial.println(work_window_ms);

  delay(kBootDelayMs);
}

void loop() {
  if (done) {
    delay(1000);
    return;
  }

  const uint32_t arm_low_us = micros();
  Serial.print("DUT: ARM LOW @ ");
  Serial.println(arm_low_us);
  digitalWrite(kArmPin, LOW);
  delay(kStableLowMs);

  start_us = micros();
  Serial.print("DUT: D2 HIGH @ ");
  Serial.println(start_us);
  digitalWrite(kTriggerPin, HIGH);

  const uint32_t deadline_us = start_us + (work_window_ms * 1000UL);
  volatile uint32_t sink = 0;
  while (micros() < deadline_us) {
    // Busy-loop math to keep the core active.
    sink += static_cast<uint32_t>(micros() & 0xFF);
  }

  const uint32_t end_us = micros();
  Serial.print("DUT: D2 LOW @ ");
  Serial.println(end_us);
  digitalWrite(kTriggerPin, LOW);
  const uint32_t arm_high_us = micros();
  Serial.print("DUT: ARM HIGH @ ");
  Serial.println(arm_high_us);
  digitalWrite(kArmPin, HIGH);

  const uint32_t duration_us = end_us - start_us;
  Serial.print("DUT: duration_us: ");
  Serial.println(duration_us);
  Serial.print("DUT: duration_s: ");
  Serial.println(static_cast<float>(duration_us) / 1000000.0f, 6);
  Serial.println("DUT: DONE");

  (void)sink;
  done = true;
}
