// Synthetic MCU workload probe for TinyODOM HIL energy measurements.
//
// The host runner compiles this sketch with TINYODOM_MICRO_WORKLOAD_MODE and
// TINYODOM_MICRO_WINDOW_MS. The measured window is one harness run: the DUT
// raises the trigger at the start of the requested window and lowers it at the
// end so the external harness owns energy/current/voltage telemetry.
#include <Arduino.h>

#include "common/tinyodom_hil_config.h"

#ifndef TINYODOM_AUTOSTART
#define TINYODOM_AUTOSTART 1
#endif

#ifndef TINYODOM_AUTOSTART_DELAY_MS
#define TINYODOM_AUTOSTART_DELAY_MS 0
#endif

#ifndef TINYODOM_SKIP_SERIAL_WAIT
#define TINYODOM_SKIP_SERIAL_WAIT 0
#endif

#ifndef TINYODOM_MICRO_WORKLOAD_MODE
#define TINYODOM_MICRO_WORKLOAD_MODE 0
#endif

#ifndef TINYODOM_MICRO_WINDOW_MS
#define TINYODOM_MICRO_WINDOW_MS 1000
#endif

namespace {

constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStartCommandTimeoutMs = 30000;
constexpr uint32_t kWindowUs = static_cast<uint32_t>(TINYODOM_MICRO_WINDOW_MS) * 1000UL;
constexpr uint32_t kBlockIterations = 512;
constexpr uint32_t kWaitBlockMs = 1;

volatile float g_float_sink = 1.0001f;
volatile uint32_t g_int_sink = 0x12345678UL;
volatile uint32_t g_poll_sink = 0x31415926UL;

struct WorkloadTelemetry {
  uint64_t iterations;
  uint64_t work_units;
  const char* work_unit_label;
  float sleep_ms;
  const char* sleep_mode;
};

bool TimeReached(uint32_t now_us, uint32_t target_us) {
  return static_cast<int32_t>(now_us - target_us) >= 0;
}

void PrintUint64(uint64_t value) {
  char buffer[21];
  size_t pos = sizeof(buffer);
  buffer[--pos] = '\0';
  do {
    buffer[--pos] = static_cast<char>('0' + (value % 10ULL));
    value /= 10ULL;
  } while (value > 0ULL && pos > 0U);
  Serial.print(&buffer[pos]);
}

void EmitReadyUntilStart() {
  const uint32_t start_ms = millis();
  while (millis() - start_ms < kStartCommandTimeoutMs) {
    Serial.println(TINYODOM_DUT_READY);
    const uint32_t poll_start = millis();
    while (millis() - poll_start < 250) {
      if (!Serial.available()) {
        delay(5);
        continue;
      }
      String line = Serial.readStringUntil('\n');
      line.trim();
      if (line.equalsIgnoreCase(TINYODOM_CMD_START)) {
        return;
      }
    }
  }
}

WorkloadTelemetry RunPollWorkload(uint32_t end_us) {
  uint32_t value = g_poll_sink;
  uint64_t blocks = 0;
  while (!TimeReached(micros(), end_us)) {
    for (uint32_t iter = 0; iter < kBlockIterations; ++iter) {
      __asm__ __volatile__("" ::: "memory");
    }
    value += static_cast<uint32_t>(blocks);
    ++blocks;
  }
  g_poll_sink = value;
  return {blocks * kBlockIterations, blocks * kBlockIterations, "poll_iters", 0.0f, "none"};
}

WorkloadTelemetry RunWaitWorkload(uint32_t end_us) {
  uint64_t blocks = 0;
  while (!TimeReached(micros(), end_us)) {
    delay(kWaitBlockMs);
    yield();
    ++blocks;
  }
  return {blocks, blocks, "wait_blocks", 0.0f, "none"};
}

WorkloadTelemetry RunFloatWorkload(uint32_t end_us) {
  float value = g_float_sink;
  uint64_t blocks = 0;
  while (!TimeReached(micros(), end_us)) {
    for (uint32_t iter = 0; iter < kBlockIterations; ++iter) {
      value = (value * 1.000091f) + 0.000173f;
      value = (value * 0.999947f) - 0.000031f;
      if (value > 65536.0f || value < -65536.0f) {
        value *= 0.0001f;
      }
    }
    ++blocks;
  }
  g_float_sink = value;
  return {blocks * kBlockIterations, blocks * kBlockIterations * 4ULL, "fp_ops", 0.0f, "none"};
}

WorkloadTelemetry RunIntWorkload(uint32_t end_us) {
  uint32_t value = g_int_sink;
  uint64_t blocks = 0;
  while (!TimeReached(micros(), end_us)) {
    for (uint32_t iter = 0; iter < kBlockIterations; ++iter) {
      value = (value * 1664525UL) + 1013904223UL;
      value ^= value << 13;
      value ^= value >> 17;
      value ^= value << 5;
    }
    ++blocks;
  }
  g_int_sink = value;
  return {blocks * kBlockIterations, blocks * kBlockIterations * 7ULL, "int_ops", 0.0f, "none"};
}

WorkloadTelemetry RunSleepWorkload(uint32_t end_us) {
  uint64_t loops = 0;
  while (!TimeReached(micros(), end_us)) {
    __WFI();
    ++loops;
  }
  return {loops, 0, "none", static_cast<float>(TINYODOM_MICRO_WINDOW_MS), "wfi_idle"};
}

const char* WorkloadLabel() {
#if TINYODOM_MICRO_WORKLOAD_MODE == 0
  return "sleep";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 1
  return "wait";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 2
  return "poll";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 3
  return "float";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 4
  return "int";
#else
  return "sleep";
#endif
}

WorkloadTelemetry RunSelectedWorkload(uint32_t end_us) {
#if TINYODOM_MICRO_WORKLOAD_MODE == 0
  return RunSleepWorkload(end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 1
  return RunWaitWorkload(end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 2
  return RunPollWorkload(end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 3
  return RunFloatWorkload(end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 4
  return RunIntWorkload(end_us);
#else
  return RunSleepWorkload(end_us);
#endif
}

}  // namespace

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

#if TINYODOM_AUTOSTART
  if (TINYODOM_AUTOSTART_DELAY_MS > 0) {
    delay(TINYODOM_AUTOSTART_DELAY_MS);
  }
#else
  EmitReadyUntilStart();
#endif

  digitalWrite(TINYODOM_HARNESS_ARM_PIN, LOW);
  delay(TINYODOM_DUT_ARM_HOLD_MS);
  delay(100);

  const uint32_t start_us = micros();
  const uint32_t end_us = start_us + kWindowUs;
  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, HIGH);
  WorkloadTelemetry telemetry = RunSelectedWorkload(end_us);
  const uint32_t total_latency_us = micros() - start_us;
  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, LOW);
  digitalWrite(TINYODOM_HARNESS_ARM_PIN, HIGH);

  Serial.println("runs: 1");
  Serial.print("timer output: ");
  Serial.println(static_cast<float>(total_latency_us) / 1000000.0f, 6);
  Serial.print("workload output: ");
  Serial.println(WorkloadLabel());
  Serial.print("requested window ms: ");
  Serial.println(TINYODOM_MICRO_WINDOW_MS);
  Serial.print("dut iterations output: ");
  PrintUint64(telemetry.iterations);
  Serial.println();
  Serial.print("dut work units output: ");
  PrintUint64(telemetry.work_units);
  Serial.println();
  Serial.print("dut work unit label output: ");
  Serial.println(telemetry.work_unit_label);
  Serial.print("dut elapsed us output: ");
  Serial.println(static_cast<unsigned long>(total_latency_us));
  Serial.println("dut cycles output: -1");
  Serial.print("dut sleep ms output: ");
  Serial.println(telemetry.sleep_ms, 3);
  Serial.print("dut sleep mode output: ");
  Serial.println(telemetry.sleep_mode);
  Serial.println("micro workload run: ok");
}

void loop() {
  delay(1000);
}
